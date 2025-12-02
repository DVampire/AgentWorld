"""Deep Analyzer Tool - A workflow agent for multi-step analysis of tasks with files."""

import asyncio
import os
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image

from src.utils import dedent
from src.utils import make_image_url
from src.utils import encode_image_base64
from src.utils import assemble_project_path
from src.utils import get_file_info
from src.logger import logger
from src.tools.protocol import tcp
from src.config import config
from src.models import model_manager
from src.tools.default_tools.mdify import MdifyTool
from src.tools.protocol.tool import BaseTool
from src.tools.protocol.types import ToolResponse

class FileTypeInfo(BaseModel):
    """File type information for a single file."""
    file_path: str = Field(description="The file path")
    file_type: str = Field(description="File type: 'text', 'image', 'audio', or 'video'")

class FileTypeClassification(BaseModel):
    """Classification of multiple files by type."""
    files: List[FileTypeInfo] = Field(description="List of files with their types")

class Summary(BaseModel):
    """Result of analyzing a chunk of text."""
    id: int = Field(description="Unique identifier for this summary")
    summary: str = Field(description="Summary of findings from this chunk (2-3 sentences)")
    found_answer: bool = Field(description="Whether the answer to the task has been found in this chunk")
    answer: Optional[str] = Field(default=None, description="The answer if found_answer is True, otherwise None")

class SummaryResponse(BaseModel):
    """Response from the deep analyzer tool."""
    summary: str = Field(description="Summary of findings from this chunk (2-3 sentences)")
    found_answer: bool = Field(description="Whether the answer to the task has been found in this chunk")
    answer: Optional[str] = Field(default=None, description="The answer if found_answer is True, otherwise None")

_DEEP_ANALYZER_DESCRIPTION = """Deep analysis tool that performs multi-step analysis of complex reasoning tasks with attached files.

🎯 BEST FOR: Complex reasoning tasks that require:
- Multi-step analysis and synthesis
- Integration of information from multiple sources
- Deep understanding of relationships and patterns
- Comprehensive evaluation and conclusion drawing

This tool will:
1. Analyze the provided task and files (text, images, PDFs, Excel, etc.)
2. Extract relevant information from files using appropriate methods
3. Perform multimodal analysis preserving visual information from images
4. Perform step-by-step analysis with intelligent approach selection
5. Generate insights and conclusions
6. Continue analysis until answer is found or max steps reached

Supports comprehensive file formats:
• Text & Markup: TXT, MD, JSON, CSV, XML, YAML
• Programming: PY, JS, HTML, CSS, Java, C/C++
• Documents: PDF, DOCX, XLSX, PPTX
• Compressed: ZIP, RAR, 7Z, TAR, GZ, BZ2, XZ
• Images: JPG, PNG, GIF, BMP, WebP, TIFF, SVG (multimodal analysis)
• Audio: MP3, WAV, OGG, FLAC, AAC, M4A
• Video: MP4, AVI, MOV, WMV, WebM

For images, preserves visual information by analyzing them directly as message inputs.

💡 Use this tool for complex tasks like:
- Research analysis and synthesis
- Technical document review
- Game strategy analysis (chess, go, etc.)
- Data pattern recognition
- Multi-source information integration
- Complex problem solving requiring multiple perspectives
"""

class DeepAnalyzerArgs(BaseModel):
    task: str = Field(description="The analysis task or question to investigate")
    files: Optional[List[str]] = Field(
        default=None,
        description="Optional list of file paths to analyze along with the task"
    )

@tcp.tool()
class DeepAnalyzerTool(BaseTool):
    """A deep analysis tool that performs multi-step analysis of tasks with files."""

    name: str = "deep_analyzer"
    type: str = "Deep Analyzer"
    description: str = _DEEP_ANALYZER_DESCRIPTION
    args_schema: Type[DeepAnalyzerArgs] = DeepAnalyzerArgs
    metadata: Dict[str, Any] = {}
    
    # Configuration parameters
    max_rounds: int = Field(default=3, description="Maximum analysis rounds in __call__ main loop")
    max_file_size: int = Field(default=10 * 1024 * 1024, description="Max file size in bytes (10MB)")
    chunk_size: int = Field(default=400, description="Number of lines per chunk for text analysis")
    max_steps: int = Field(default=3, description="Maximum steps for image analysis without finding answer")
    
    model_name: str = Field(
        default="o3",
        description="The model to use for the deep analyzer."
    )
    model: Any = Field(
        default=None,
        description="The model to use for the deep analyzer."
    )
    mdify_tool: MdifyTool = Field(
        default=None,
        description="The mdify tool to use for the deep analyzer."
    )
    next_summary_id: int = Field(
        default=1,
        description="Next summary ID for auto-increment"
    )

    def __init__(self, model_name: Optional[str] = None, **kwargs):
        """Initialize the deep analyzer tool."""
        
        super().__init__(**kwargs)
        
        if model_name is not None:
            self.model_name = model_name
        else:
            if "deep_analyzer_tool" in config:
                self.model_name = config.deep_analyzer_tool.get("model_name", "o3")
        
        # Initialize model
        self.model = model_manager.get(self.model_name)
        
        # Initialize tools
        self.mdify_tool = MdifyTool()
        
        # Initialize summary ID counter
        if not hasattr(self, 'next_summary_id'):
            self.next_summary_id = 1
    
    def _get_next_summary_id(self) -> int:
        """Get next summary ID and increment counter."""
        current_id = self.next_summary_id
        self.next_summary_id += 1
        return current_id
    
    async def _classify_files(self, files: List[str]) -> List[FileTypeInfo]:
        """Use LLM to classify file types."""
        try:
            # Build file list for LLM
            file_list = "\n".join([f"- {file_path}" for file_path in files])
            
            prompt = dedent(f"""Classify the following files by type. For each file, determine if it is:
            - 'text': Text files, markup files, programming files, documents (PDF, DOCX, XLSX, PPTX), or compressed files (ZIP, RAR, 7Z, TAR, GZ, BZ2, XZ)
            - 'image': Image files (JPG, PNG, GIF, BMP, WebP, TIFF, SVG)
            - 'audio': Audio files (MP3, WAV, OGG, FLAC, AAC, M4A)
            - 'video': Video files (MP4, AVI, MOV, WMV, WebM)
            
            Files to classify:
            {file_list}
            
            Classify each file based on its content type, not just the extension.
            """)
            
            messages = [
                SystemMessage(content="You are an expert at classifying file types based on their content and purpose."),
                HumanMessage(content=prompt)
            ]
            
            structured_llm = self.model.with_structured_output(FileTypeClassification)
            response = await structured_llm.ainvoke(messages)
            
            return response.files
                
        except Exception as e:
            logger.warning(f"Error classifying files with LLM: {e}, using extension fallback")
            return self._classify_by_extension(files)
    
    def _classify_by_extension(self, files: List[str]) -> List[FileTypeInfo]:
        """Fallback: classify files by extension."""
        result = []
        for file_path in files:
            _, ext = os.path.splitext(file_path.lower())
            
            # Text, markup, programming, documents, compressed
            text_exts = [".txt", ".md", ".json", ".csv", ".xml", ".yaml", ".yml",
                        ".py", ".js", ".html", ".css", ".java", ".cpp", ".c", ".h",
                        ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
                        ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"]
            image_exts = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg"]
            audio_exts = [".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".m4b", ".m4p"]
            video_exts = [".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v"]
            
            if ext in text_exts:
                file_type = "text"
            elif ext in image_exts:
                file_type = "image"
            elif ext in audio_exts:
                file_type = "audio"
            elif ext in video_exts:
                file_type = "video"
            else:
                file_type = "text"  # Default to text
            
            result.append(FileTypeInfo(file_path=file_path, file_type=file_type))
        
        return result

    async def _arun(self, task: str, files: Optional[List[str]] = None) -> ToolResponse:
        """Execute deep analysis workflow.

        Args:
            task: The analysis task or question to investigate
            files: Optional list of absolute file paths to analyze along with the task
        """
        try:
            logger.info(f"| 🚀 Starting DeepAnalyzerTool: {task}")
            if files:
                logger.info(f"| 📂 Attached files: {files}")
            
            # Maintain summaries list in __call__
            summaries: List[Summary] = []
            
            # Validate files
            valid_files = []
            if files:
                for file_path in files:
                    if await self._validate_file(file_path):
                        valid_files.append(file_path)
                    else:
                        logger.warning(f"Skipping invalid file: {file_path}")
            
            # If no files or no valid files, analyze task directly
            if not valid_files:
                logger.info(f"| 📝 No files or no valid files, analyzing task directly")
                await self._analyze_task_only(task, summaries)
                
                # Check if answer found
                summary = await self._summarize_summaries(task, summaries)
                if summary.found_answer:
                    return ToolResponse(success=True, message=f"Answer found from task analysis.\n\nTask: {task}\n\nAnswer: {summary.answer}")
                else:
                    summaries.append(summary)
                    result = f"Analysis completed but no definitive answer found.\n\nTask: {task}\n\nSummaries:\n" + "\n".join([f"- {s.summary}" for s in summaries])
                    return ToolResponse(success=False, message=result)
            
            # Step 1: Get overall file information summary before detailed analysis
            logger.info(f"| 📊 Getting overall file information summary...")
            summary = await self._get_overall_file_summary(task, valid_files)
            if summary and summary.found_answer:
                return ToolResponse(success=True, message=f"Answer found from file information summary.\n\nTask: {task}\n\nAnswer: {summary.answer}")
            elif summary:
                summaries.append(summary)
            
            # Use LLM to classify file types
            logger.info(f"| 🔍 Classifying {len(valid_files)} files by type...")
            file_classifications = await self._classify_files(valid_files)
            
            # Log classifications
            for file_info in file_classifications:
                logger.info(f"| 📋 {os.path.basename(file_info.file_path)}: {file_info.file_type}")
            
            # Main analysis loop with max_rounds
            for round_num in range(1, self.max_rounds + 1):
                logger.info(f"| 🔄 Main analysis round {round_num}/{self.max_rounds}")
                
                round_summaries: List[Summary] = []
                
                # Process each file in this round
                for file_info in file_classifications:
                    file_path = file_info.file_path
                    file_type = file_info.file_type
                    
                    logger.info(f"| 📄 Processing {file_type} file: {os.path.basename(file_path)}")
                    
                    # Analyze based on file type
                    if file_type == "text":
                        await self._analyze_text_file(task, file_path, round_summaries)
                    elif file_type == "image":
                        await self._analyze_image_file(task, file_path, round_summaries)
                    elif file_type == "audio":
                        await self._analyze_audio_file(task, file_path, round_summaries)
                    elif file_type == "video":
                        await self._analyze_video_file(task, file_path, round_summaries)
                    
                    # Check if answer found after processing this file
                    round_summary = await self._summarize_summaries(task, round_summaries)
                    if round_summary.found_answer:
                        return ToolResponse(success=True, message=f"Answer found from file analysis.\n\nTask: {task}\n\nAnswer: {round_summary.answer}")
                    else:
                        summaries.append(round_summary)
            
            final_summary = await self._summarize_summaries(task, summaries)
            if final_summary.found_answer:
                return ToolResponse(success=True, message=f"Answer found from all file analysis.\n\nTask: {task}\n\nAnswer: {final_summary.answer}")
            else:
                return ToolResponse(success=False, message=f"Analysis completed after {self.max_rounds} rounds but no definitive answer found.\n\nTask: {task}\n\nSummaries:\n" + "\n".join([f"- {s.summary}" for s in summaries[-10:]]))
            
        except Exception as e:
            logger.error(f"| ❌ Error in deep analysis: {e}")
            return ToolResponse(success=False, message=f"Error during deep analysis: {str(e)}")
    
    async def _get_overall_file_summary(self, task: str, files: List[str]) -> Optional[Summary]:
        """Get overall summary of all files' information before detailed analysis."""
        try:
            # Get file info for all files
            file_infos = []
            for file_path in files:
                try:
                    file_info = get_file_info(file_path)
                    file_infos.append({
                        "path": file_path,
                        "name": os.path.basename(file_path),
                        "info": file_info
                    })
                except Exception as e:
                    logger.warning(f"Failed to get info for {file_path}: {e}")
            
            if not file_infos:
                return None
            
            # Format file information for LLM
            files_info_text = "\n".join([
                dedent(f"""
                File: {info['name']}
                Path: {info['path']}
                Size: {info['info'].get('size', 'unknown')}
                Created: {info['info'].get('created', 'unknown')}
                Modified: {info['info'].get('modified', 'unknown')}
                """).strip()
                for info in file_infos
            ])
            
            prompt = dedent(f"""Analyze the following task and provide a summary based on the file information provided.
            
            Task: {task}
            
            File Information:
            {files_info_text}
            
            Based on the file information (sizes, types, names, timestamps, etc.), provide a summary that:
            1. Describes what information can be found from the file metadata
            2. Answers the task if it can be answered from file information alone (e.g., file sizes, video durations, file counts, etc.)
            3. If the task requires file content analysis, indicate what needs to be analyzed
            
            Provide a concise summary (3-5 sentences).
            """)
            
            messages = [
                SystemMessage(content="You are an expert at analyzing file metadata and determining if questions can be answered from file information alone."),
                HumanMessage(content=prompt)
            ]
            
            structured_llm = self.model.with_structured_output(SummaryResponse)
            response = await structured_llm.ainvoke(messages)
            
            summary = Summary(
                id=self._get_next_summary_id(),
                summary=response.summary,
                found_answer=response.found_answer,
                answer=response.answer
            )
            
            logger.info(f"| ✅ Overall file summary generated")
            return summary
            
        except Exception as e:
            logger.warning(f"Failed to generate overall file summary: {e}")
            return None
    
    async def _summarize_summaries(self, task: str, summaries: List[Summary]) -> Summary:
        """Summarize all summaries to get a new Summary."""
        try:
            if not summaries:
                return Summary(
                    id=self._get_next_summary_id(),
                    summary="No summaries to summarize.",
                    found_answer=False,
                    answer=None
                )
            
            # Combine all summaries
            summaries_text = "\n".join([f"- {s.summary}" for s in summaries])
            
            prompt = dedent(f"""Based on the following analysis summaries, provide a comprehensive summary.
            
            Task: {task}
            
            Analysis summaries:
            {summaries_text}
            
            Synthesize all the information from the summaries and provide:
            1. A comprehensive summary (3-5 sentences) that integrates all findings
            2. Determine if we have found the answer to the task based on all summaries
            3. If the answer is found, provide it in the answer field
            """)
            
            messages = [
                SystemMessage(content="You are an expert at synthesizing information from multiple analysis summaries."),
                HumanMessage(content=prompt)
            ]
            
            structured_llm = self.model.with_structured_output(SummaryResponse)
            response = await structured_llm.ainvoke(messages)
            
            return Summary(
                id=self._get_next_summary_id(),
                summary=response.summary,
                found_answer=response.found_answer,
                answer=response.answer
            )
            
        except Exception as e:
            logger.error(f"| ❌ Error summarizing summaries: {e}")
            return Summary(
                id=self._get_next_summary_id(),
                summary=f"Error summarizing summaries: {e}",
                found_answer=False,
                answer=None
            )
    
    async def _analyze_task_only(self, task: str, summaries: List[Summary]) -> None:
        """Analyze task without files (text games, math problems, logic puzzles, etc.)."""
        try:
            logger.info(f"| 🧠 Analyzing task directly (no files)")
            
            # Multi-round analysis for complex tasks
            for round_num in range(1, self.max_rounds + 1):
                logger.info(f"| 🔄 Analysis round {round_num}/{self.max_rounds}")
                
                prompt = dedent(f"""Analyze the following task step by step. This could be a text game, math problem, logic puzzle, or reasoning task.
                
                Task: {task}
                
                For this round, perform detailed analysis:
                1. Break down the task into components
                2. Identify key information and constraints
                3. Apply logical reasoning or mathematical operations
                4. Generate insights and partial solutions
                5. If you find the complete answer, clearly state it
                
                Provide a concise summary (2-4 sentences) of your analysis for this round.
                """)
                
                messages = [
                    SystemMessage(content="You are an expert at solving complex reasoning tasks, text games, math problems, and logic puzzles."),
                    HumanMessage(content=prompt)
                ]
                
                structured_llm = self.model.with_structured_output(Summary)
                response = await structured_llm.ainvoke(messages)
                
                # Assign ID to parsed summary
                response.id = self._get_next_summary_id()
                summaries.append(response)
                
                # Check if we found the answer
                if response.found_answer:
                    logger.info(f"| ✅ Answer found in round {round_num}, early stopping.")
                    return None
            
            logger.info(f"| ✅ Task analysis completed after {self.max_rounds} rounds")
            
        except Exception as e:
            logger.error(f"| ❌ Error analyzing task: {e}")
            summaries.append(Summary(id=self._get_next_summary_id(), summary=f"Error analyzing task: {e}", found_answer=False, answer=None))
            return None
    
    async def _analyze_text_file(self, task: str, file_path: str, summaries: List[Summary]) -> None:
        """Analyze a single text file: get file info, convert to markdown, analyze in chunks."""
        try:
            # Get file basic info
            file_info = get_file_info(file_path)
            logger.info(f"| 📄 Processing text file: {os.path.basename(file_path)} ({file_info.get('size', 'unknown')} bytes)")
            
            # Convert to markdown using mdify_tool (automatically saves to base_dir)
            mdify_response = await self.mdify_tool._arun(file_path=file_path, output_format="markdown")
            if not mdify_response.success:
                logger.warning(f"Failed to convert file to markdown: {mdify_response.message}")
                summaries.append(Summary(id=self._get_next_summary_id(), summary=f"Failed to convert file to markdown: {file_path}", found_answer=False, answer=None))
                return None
            
            saved_path = mdify_response.extra["saved_path"]
            
            # Read all lines once
            with open(saved_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            total_chunks = (total_lines + self.chunk_size - 1) // self.chunk_size
            
            # Internal loop: analyze chunks one by one
            for chunk_num in range(1, total_chunks + 1):
                logger.info(f"| 🔄 Analyzing text file chunk {chunk_num}/{total_chunks}")
                
                # Extract chunk text
                start_line = (chunk_num - 1) * self.chunk_size
                end_line = min(start_line + self.chunk_size, total_lines)
                chunk_lines = lines[start_line:end_line]
                chunk_text = "".join(chunk_lines)
                
                summary = await self._analyze_markdown_chunk(task, chunk_text, chunk_num, start_line + 1, end_line)
                summaries.append(summary)
                
                if summary.found_answer:
                    logger.info(f"| ✅ Answer found in chunk {chunk_num}, early stopping.")
                    return None
            
            logger.info(f"| ✅ All chunks of text file analyzed")
            
        except Exception as e:
            summaries.append(Summary(id=self._get_next_summary_id(), summary=f"Error analyzing text file {file_path}: {e}", found_answer=False, answer=None))
            return None
    
    async def _analyze_markdown_chunk(self, task: str, chunk_text: str, chunk_num: int, start_line: int, end_line: int) -> Summary:
        """Analyze a chunk of markdown text."""
        try:
            logger.info(f"| 🔍 Analyzing chunk {chunk_num} (lines {start_line}-{end_line})")
            
            context = f"Task: {task}\n\n"
            context += f"Current chunk (lines {start_line}-{end_line}):\n{chunk_text}"
            
            prompt = dedent(f"""Analyze this chunk of the document and extract information relevant to the task.
            
            {context}
            
            Extract key information that helps answer the task. Provide a concise summary (2-3 sentences) of findings from this chunk.
            If this chunk contains the answer to the task, set found_answer to True and provide the answer in the answer field.
            """)
            
            messages = [
                SystemMessage(content="You are an expert at extracting key information from documents."),
                HumanMessage(content=prompt)
            ]
            
            structured_llm = self.model.with_structured_output(Summary)
            response = await structured_llm.ainvoke(messages)
            
            # Assign ID to parsed summary
            response.id = self._get_next_summary_id()
            return response
            
        except Exception as e:
            logger.error(f"| ❌ Error analyzing markdown chunk: {e}")
            return Summary(id=self._get_next_summary_id(), summary=f"Error analyzing markdown chunk: {e}", found_answer=False, answer=None)
    
    async def _analyze_image_file(self, task: str, file_path: str, summaries: List[Summary]) -> None:
        """Analyze a single image file: directly send to LLM without mdify conversion."""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Image file not found: {file_path}")
                return None
            
            # Internal loop: analyze image multiple times
            for step_num in range(1, self.max_steps + 1):
                logger.info(f"| 🔄 Analyzing image step {step_num}/{self.max_steps}")
                
                # Build multimodal message with the image
                message_content = [
                    {"type": "text", "text": dedent(f"""Analyze the following image to answer the task.
                    
                    Task: {task}
                    
                    Extract key information from the image that helps answer the task.
                    Focus on visual elements, text in images, patterns, and any relevant details.
                    """)}
                ]
                
                # Add the image
                try:
                    image = Image.open(assemble_project_path(file_path))
                    image_url = make_image_url(encode_image_base64(image))
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    })
                    logger.info(f"| ✅ Added image: {os.path.basename(file_path)}")
                except Exception as e:
                    logger.warning(f"Failed to process image {file_path}: {e}")
                    return None
                
                messages = [
                    SystemMessage(content="You are an expert at analyzing images and extracting visual information."),
                    HumanMessage(content=message_content)
                ]
                
                structured_llm = self.model.with_structured_output(Summary)
                response = await structured_llm.ainvoke(messages)
                
                # Assign ID to parsed summary
                response.id = self._get_next_summary_id()
                summaries.append(response)
                
                # Check if answer found after each step
                if response.found_answer:
                    logger.info(f"| ✅ Answer found in image step {step_num}, early stopping.")
                    return None
            
            logger.info(f"| ✅ Image analysis completed after {self.max_steps} steps")
            
        except Exception as e:
            logger.error(f"| ❌ Error analyzing image file {file_path}: {e}")
            summaries.append(Summary(id=self._get_next_summary_id(), summary=f"Error analyzing image file {file_path}: {e}", found_answer=False, answer=None))
            return None
    
    async def _analyze_audio_file(self, task: str, file_path: str, summaries: List[Summary]) -> None:
        """Analyze a single audio file: convert to markdown, then analyze like text."""
        try:
            # Get file basic info
            file_info = get_file_info(file_path)
            logger.info(f"| 🎵 Processing audio file: {os.path.basename(file_path)} ({file_info.get('size', 'unknown')} bytes)")
            
            # Convert to markdown using mdify_tool (automatically saves to base_dir)
            mdify_response = await self.mdify_tool._arun(file_path=file_path, output_format="markdown")
            if not mdify_response.success:
                logger.warning(f"Failed to convert audio file to markdown: {mdify_response.message}")
                summaries.append(Summary(id=self._get_next_summary_id(), summary=f"Failed to convert audio file to markdown: {file_path}", found_answer=False, answer=None))
                return None
            
            saved_path = mdify_response.extra["saved_path"]
            
            # Read all lines once
            with open(saved_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            total_chunks = (total_lines + self.chunk_size - 1) // self.chunk_size
            
            # Internal loop: analyze chunks one by one
            for chunk_num in range(1, total_chunks + 1):
                logger.info(f"| 🔄 Analyzing audio file chunk {chunk_num}/{total_chunks}")
                
                # Extract chunk text
                start_line = (chunk_num - 1) * self.chunk_size
                end_line = min(start_line + self.chunk_size, total_lines)
                chunk_lines = lines[start_line:end_line]
                chunk_text = "".join(chunk_lines)
                
                summary = await self._analyze_markdown_chunk(task, chunk_text, chunk_num, start_line + 1, end_line)
                summaries.append(summary)
                
                if summary.found_answer:
                    logger.info(f"| ✅ Answer found in chunk {chunk_num}, early stopping.")
                    return None
            
            logger.info(f"| ✅ All chunks of audio file analyzed")
            
        except Exception as e:
            summaries.append(Summary(id=self._get_next_summary_id(), summary=f"Error analyzing audio file {file_path}: {e}", found_answer=False, answer=None))
            return None
    
    async def _analyze_video_file(self, task: str, file_path: str, summaries: List[Summary]) -> None:
        """Analyze a single video file: convert to markdown, then analyze like text."""
        try:
            # Get file basic info
            file_info = get_file_info(file_path)
            logger.info(f"| 🎬 Processing video file: {os.path.basename(file_path)} ({file_info.get('size', 'unknown')} bytes)")
            
            # Convert to markdown using mdify_tool (automatically saves to base_dir)
            mdify_response = await self.mdify_tool._arun(file_path=file_path, output_format="markdown")
            if not mdify_response.success:
                logger.warning(f"Failed to convert video file to markdown: {mdify_response.message}")
                summaries.append(Summary(id=self._get_next_summary_id(), summary=f"Failed to convert video file to markdown: {file_path}", found_answer=False, answer=None))
                return None
            
            saved_path = mdify_response.extra["saved_path"]
            
            # Read all lines once
            with open(saved_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            total_chunks = (total_lines + self.chunk_size - 1) // self.chunk_size
            
            # Internal loop: analyze chunks one by one
            for chunk_num in range(1, total_chunks + 1):
                logger.info(f"| 🔄 Analyzing video file chunk {chunk_num}/{total_chunks}")
                
                # Extract chunk text
                start_line = (chunk_num - 1) * self.chunk_size
                end_line = min(start_line + self.chunk_size, total_lines)
                chunk_lines = lines[start_line:end_line]
                chunk_text = "".join(chunk_lines)
                
                summary = await self._analyze_markdown_chunk(task, chunk_text, chunk_num, start_line + 1, end_line)
                summaries.append(summary)
                
                if summary.found_answer:
                    logger.info(f"| ✅ Answer found in chunk {chunk_num}, early stopping.")
                    return None
            
            logger.info(f"| ✅ All chunks of video file analyzed")
            
        except Exception as e:
            summaries.append(Summary(id=self._get_next_summary_id(), summary=f"Error analyzing video file {file_path}: {e}", found_answer=False, answer=None))
            return None
    
    async def _validate_file(self, file_path: str) -> bool:
        """Validate if file can be processed."""
        try:
            if not os.path.exists(file_path):
                logger.warning(f"File does not exist: {file_path}")
                return False
            
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                logger.warning(f"File too large: {file_path} ({file_size} bytes)")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {e}")
            return False

    def _run(self, task: str, files: Optional[List[str]] = None) -> ToolResponse:
        """Execute deep analysis synchronously (fallback)."""
        try:
            # Run async version
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(task, files))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error in synchronous execution: {e}")
            return ToolResponse(success=False, message=f"Error in synchronous execution: {e}")

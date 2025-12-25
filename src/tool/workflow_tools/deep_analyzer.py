"""Deep Analyzer Tool - A workflow agent for multi-step analysis of tasks with files."""

import os
import re
import asyncio
import urllib.request
import uuid
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from urllib.parse import urlparse

from src.utils import dedent
from src.utils import make_file_url
from src.utils import assemble_project_path
from src.utils import get_file_info
from src.utils import fetch_url
from src.utils import encode_file_base64
from src.tool.types import Tool, ToolResponse, ToolExtra
from src.tool.workflow_tools.report import Report

from src.logger import logger
from src.model import model_manager
from src.tool.default_tools.mdify import MdifyTool
from src.message import HumanMessage, SystemMessage
from src.registry import TOOL

class FileTypeInfo(BaseModel):
    """File type information for a single file."""
    file: str = Field(description="The file path or URL")
    file_type: str = Field(description="File type: 'text', 'pdf', 'image', 'audio', or 'video'")

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
1. Analyze the provided task and files (text, images, PDFs, Excel, audio, video, etc.)
2. Extract relevant information from files using appropriate methods
3. Perform multimodal analysis preserving visual information from images
4. Perform step-by-step analysis with intelligent approach selection
5. Generate insights and conclusions
6. Continue analysis until answer is found or max steps reached

Supports comprehensive file formats:
• Text & Markup: TXT, MD, JSON, CSV, XML, YAML (supports both local files and URLs like https://example.com/text.txt)
• Programming: PY, JS, HTML, CSS, Java, C/C++ (supports both local files and URLs like https://example.com/code.py)
• Documents: DOCX, XLSX, PPTX (supports both local files and URLs like https://example.com/document.docx)
• Compressed: ZIP, RAR, 7Z, TAR, GZ, BZ2, XZ (supports both local files and URLs like https://example.com/compressed.zip)
• Audio: MP3, WAV, OGG, FLAC, AAC, M4A (supports both local files and URLs like https://example.com/audio.mp3)
• PDF: PDF files (supports both local files and URLs like https://example.com/document.pdf)
• Images: JPG, PNG, GIF, BMP, WebP, TIFF, SVG (multimodal analysis, supports both local files and URLs like https://example.com/image.jpg)
• Video: MP4, AVI, MOV, WMV, WebM or video URL like https://www.youtube.com/watch?v=dQw4w9WgXcQ (supports both local files and URLs, non-YouTube URLs will be downloaded automatically like https://www.youtube.com/watch?v=dQw4w9WgXcQ)

For images, audio, video, preserves visual information by analyzing them directly as message inputs.

💡 Use this tool for complex tasks like:
- Research analysis and synthesis
- Technical document review
- Game strategy analysis (chess, go, etc.)
- Data pattern recognition
- Multi-source information integration
- Complex problem solving requiring multiple perspectives
"""

@TOOL.register_module(force=True)
class DeepAnalyzerTool(Tool):
    """A deep analysis tool that performs multi-step analysis of tasks with files."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "deep_analyzer"
    description: str = _DEEP_ANALYZER_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    
    # Configuration parameters
    max_rounds: int = Field(default=3, description="Maximum analysis rounds in __call__ main loop")
    max_file_size: int = Field(default=10 * 1024 * 1024, description="Max file size in bytes (10MB)")
    chunk_size: int = Field(default=400, description="Number of lines per chunk for text analysis")
    max_steps: int = Field(default=3, description="Maximum steps for image analysis without finding answer")
    
    model_name: str = Field(
        default="o3",
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
    base_dir: str = Field(
        default="workdir/deep_analyzer",
        description="The base directory to use for the deep analyzer."
    )
    file_model_name: str = Field(
        default="gemini-2.5-flash",
        description="The model to use for the file analysis."
    )

    def __init__(self, model_name: Optional[str] = None, base_dir: Optional[str] = None, **kwargs):
        """Initialize the deep analyzer tool."""
        super().__init__(**kwargs)
        
        if model_name is not None:
            self.model_name = model_name
        
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(self.base_dir)
        
        # Create base directory if it doesn't exist
        if self.base_dir:
            os.makedirs(self.base_dir, exist_ok=True)
        
        # Initialize tools
        self.mdify_tool = MdifyTool(base_dir=self.base_dir)
        
        # Initialize summary ID counter
        if not hasattr(self, 'next_summary_id'):
            self.next_summary_id = 1
        
        # File path (will be set in __call__)
        self.file_path = None
    
    def _get_next_summary_id(self) -> int:
        """Get next summary ID and increment counter."""
        current_id = self.next_summary_id
        self.next_summary_id += 1
        return current_id
    
    def _is_url(self, file_path: str) -> bool:
        """Check if the file path is a URL."""
        return file_path.startswith(("http://", "https://"))
    
    def _is_youtube_url(self, url: str) -> bool:
        """Check if the URL is a YouTube URL."""
        youtube_patterns = [
            r"youtube\.com/watch\?v=",
            r"youtu\.be/",
            r"youtube\.com/embed/",
            r"youtube\.com/v/",
        ]
        return any(re.search(pattern, url, re.IGNORECASE) for pattern in youtube_patterns)
    
    def _get_url_type(self, url: str) -> Optional[str]:
        """Determine the type of URL (text, pdf, image, audio, video)."""
        url_lower = url.lower()
        
        # Check for image URLs
        image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg"]
        if any(url_lower.endswith(ext) for ext in image_extensions):
            return "image"
        
        # Check for PDF URLs
        if url_lower.endswith(".pdf"):
            return "pdf"
        
        # Check for audio URLs
        audio_extensions = [".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".m4b", ".m4p"]
        if any(url_lower.endswith(ext) for ext in audio_extensions):
            return "audio"
        
        # Check for YouTube video URLs
        if self._is_youtube_url(url):
            return "video"
        
        # Check for other video URLs (by extension)
        video_extensions = [".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v"]
        if any(url_lower.endswith(ext) for ext in video_extensions):
            return "video"
        
        # Check for text/document/compressed file URLs (default to text)
        # Text, markup, programming, documents, compressed
        text_extensions = [".txt", ".md", ".json", ".csv", ".xml", ".yaml", ".yml",
                          ".py", ".js", ".html", ".css", ".java", ".cpp", ".c", ".h",
                          ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
                          ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"]
        if any(url_lower.endswith(ext) for ext in text_extensions):
            return "text"
        
        # If URL doesn't match any known extension, we can still try to download it as text
        # This allows handling URLs without clear file extensions
        return "text"
    
    async def _download_file(self, url: str, file_type: str = "file") -> Optional[str]:
        """Download file from URL to local base_dir and return the local file path.
        
        Args:
            url: The URL to download
            file_type: Type of file (e.g., "video", "audio", "file") for default filename
        
        Returns:
            Local file path if successful, None otherwise
        """
        try:
            # Create downloads directory in base_dir if it doesn't exist
            downloads_dir = os.path.join(self.base_dir, "downloads")
            os.makedirs(downloads_dir, exist_ok=True)
            
            # Get filename from URL or use a default name
            parsed_url = urlparse(url)
            filename = os.path.basename(parsed_url.path)
            if not filename or '.' not in filename:
                # Use a default filename with extension from URL or content-type
                # Try to get extension from URL path
                path_ext = os.path.splitext(parsed_url.path)[1]
                if path_ext:
                    default_ext = path_ext
                else:
                    # Default extensions based on file type
                    default_exts = {
                        "video": ".mp4",
                        "audio": ".mp3",
                        "file": ".bin"
                    }
                    default_ext = default_exts.get(file_type, ".bin")
                filename = f"{file_type}_{hash(url) % 100000}{default_ext}"
            
            local_path = os.path.join(downloads_dir, filename)
            
            # Download the file asynchronously
            logger.info(f"| 📥 Downloading {file_type} from {url} to {local_path}")
            await asyncio.to_thread(urllib.request.urlretrieve, url, local_path)
            logger.info(f"| ✅ {file_type.capitalize()} downloaded successfully: {local_path}")
            return local_path
        except Exception as e:
            logger.error(f"| ❌ Error downloading {file_type} from {url}: {e}")
            return None
    
    async def _classify_files(self, files: List[str]) -> List[FileTypeInfo]:
        """Use LLM to classify file types (supports both file paths and URLs)."""
        try:
            # First, check for URLs and classify them directly
            url_files = []
            path_files = []
            url_classifications = []
            
            for file_path in files:
                if self._is_url(file_path):
                    url_type = self._get_url_type(file_path)
                    if url_type:
                        url_classifications.append(FileTypeInfo(file=file_path, file_type=url_type))
                    else:
                        # URL not supported, skip it
                        logger.warning(f"Unsupported URL type: {file_path}")
                else:
                    path_files.append(file_path)
            
            # Classify non-URL files with LLM
            if path_files:
                # Build file list for LLM
                file_list = "\n".join([f"- {file_path}" for file_path in path_files])
                
                prompt = dedent(f"""Classify the following files by type. For each file, determine if it is:
                - 'text': Text files, markup files, programming files, documents (DOCX, XLSX, PPTX), or compressed files (ZIP, RAR, 7Z, TAR, GZ, BZ2, XZ)
                - 'pdf': PDF files
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
                
                response = await model_manager(
                    model=self.model_name,
                    messages=messages,
                    response_format=FileTypeClassification
                )
                
                if response.extra and response.extra.parsed_model:
                    classification = response.extra.parsed_model
                    path_classifications = classification.files
                else:
                    # Fallback: use file extension
                    logger.warning("LLM classification failed, using file extension fallback")
                    path_classifications = self._classify_by_extension(path_files)
            else:
                path_classifications = []
            
            # Combine URL and path classifications
            return url_classifications + path_classifications
                
        except Exception as e:
            logger.warning(f"Error classifying files with LLM: {e}, using extension fallback")
            # Fallback: classify all files by extension/URL
            all_classifications = []
            for file_path in files:
                if self._is_url(file_path):
                    url_type = self._get_url_type(file_path)
                    if url_type:
                        all_classifications.append(FileTypeInfo(file=file_path, file_type=url_type))
                else:
                    # Use extension-based classification
                    _, ext = os.path.splitext(file_path.lower())
                    text_exts = [".txt", ".md", ".json", ".csv", ".xml", ".yaml", ".yml",
                                ".py", ".js", ".html", ".css", ".java", ".cpp", ".c", ".h",
                                ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
                                ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"]
                    pdf_exts = [".pdf"]
                    image_exts = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg"]
                    audio_exts = [".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".m4b", ".m4p"]
                    video_exts = [".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v"]
                    
                    if ext in text_exts:
                        file_type = "text"
                    elif ext in pdf_exts:
                        file_type = "pdf"
                    elif ext in image_exts:
                        file_type = "image"
                    elif ext in audio_exts:
                        file_type = "audio"
                    elif ext in video_exts:
                        file_type = "video"
                    else:
                        file_type = "text"
                    
                    all_classifications.append(FileTypeInfo(file=file_path, file_type=file_type))
            
            return all_classifications
    
    def _classify_by_extension(self, files: List[str]) -> List[FileTypeInfo]:
        """Fallback: classify files by extension or URL."""
        result = []
        for file_path in files:
            # Check if it's a URL
            if self._is_url(file_path):
                url_type = self._get_url_type(file_path)
                if url_type:
                    result.append(FileTypeInfo(file=file_path, file_type=url_type))
                else:
                    logger.warning(f"Unsupported URL: {file_path}, defaulting to text")
                    result.append(FileTypeInfo(file=file_path, file_type="text"))
            else:
                # Classify by file extension
                _, ext = os.path.splitext(file_path.lower())
                
                # Text, markup, programming, documents (excluding PDF), compressed
                text_exts = [".txt", ".md", ".json", ".csv", ".xml", ".yaml", ".yml",
                            ".py", ".js", ".html", ".css", ".java", ".cpp", ".c", ".h",
                            ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
                            ".zip", ".rar", ".7z", ".tar", ".gz", ".bz2", ".xz"]
                pdf_exts = [".pdf"]
                image_exts = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg"]
                audio_exts = [".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".m4b", ".m4p"]
                video_exts = [".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v"]
                
                if ext in text_exts:
                    file_type = "text"
                elif ext in pdf_exts:
                    file_type = "pdf"
                elif ext in image_exts:
                    file_type = "image"
                elif ext in audio_exts:
                    file_type = "audio"
                elif ext in video_exts:
                    file_type = "video"
                else:
                    file_type = "text"  # Default to text
                
                result.append(FileTypeInfo(file=file_path, file_type=file_type))
        
        return result

    async def __call__(self, task: str, files: Optional[List[str]] = None, **kwargs) -> ToolResponse:
        """Execute deep analysis workflow.

        Args:
            task (str): The analysis task or question to investigate
            files (Optional[List[str]]): Optional list of absolute file paths or specific URLs (image, video, PDF) to analyze along with the task
        """
        try:
            logger.info(f"| 🚀 Starting DeepAnalyzerTool: {task}")
            if files:
                logger.info(f"| 📂 Attached files: {files}")
            
            # Create file path for markdown report
            md_filename = f"analysis_{uuid.uuid4().hex[:8]}.md"
            self.file_path = os.path.join(self.base_dir, md_filename) if self.base_dir else None
            
            # Initialize Report instance
            report = Report(
                title="Deep Analysis Report",
                model_name=self.model_name
            )
            
            # Add initial task information
            task_content = f"## Analysis Task\n\n{task}\n\n"
            if files:
                task_content += f"## Files\n\n"
                for file_path in files:
                    file_display = file_path if self._is_url(file_path) else os.path.basename(file_path)
                    task_content += f"- {file_display}\n"
                task_content += "\n"
            
            await report.add_item(task_content)
            
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
                await self._analyze_task_only(task, summaries, report)
                
                # Check if answer found
                summary = await self._summarize_summaries(task, summaries)
                if summary.found_answer:
                    answer_content = f"## Final Answer\n\n**Answer Found**: Yes\n\n**Answer**: {summary.answer}\n\n"
                    await report.add_item(answer_content)
                    
                    if self.file_path:
                        final_report_content = await report.complete(self.file_path)
                        logger.info(f"✅ Analysis report saved to: {self.file_path}")
                        
                        message = f"Answer found from task analysis.\n\nTask: {task}\n\nAnswer: {summary.answer}, Report saved to: {self.file_path}"
                        
                        return ToolResponse(
                            success=True,
                            message=message,
                            extra=ToolExtra(
                                file_path=self.file_path,
                                data={
                                    "task": task,
                                    "answer_found": True,
                                    "answer": summary.answer,
                                    "file_path": self.file_path
                                }
                            )
                        )
                    else:
                        message = f"Answer found from task analysis.\n\nTask: {task}\n\nAnswer: {summary.answer}"
                        return ToolResponse(success=True, message=message)
                else:
                    summaries.append(summary)
                    result = f"Analysis completed but no definitive answer found.\n\nTask: {task}\n\nSummaries:\n" + "\n".join([f"- {s.summary}" for s in summaries])
                    
                    if self.file_path:
                        final_report_content = await report.complete(self.file_path)
                        logger.info(f"✅ Analysis report saved to: {self.file_path}")
                        
                        message = f"Analysis completed but no definitive answer found.\n\nTask: {task}\n\nSummaries:\n" + "\n".join([f"- {s.summary}" for s in summaries]) + "\n\nReport saved to: {self.file_path}"
                        
                        return ToolResponse(
                            success=False,
                            message=message,
                            extra=ToolExtra(
                                file_path=self.file_path,
                                data={
                                    "task": task,
                                    "answer_found": False,
                                    "file_path": self.file_path
                                }
                            )
                        )
                    else:
                        message = f"Analysis completed but no definitive answer found.\n\nTask: {task}\n\nSummaries:\n" + "\n".join([f"- {s.summary}" for s in summaries])
                        return ToolResponse(success=False, message=message)
            
            # Step 1: Get overall file information summary before detailed analysis
            logger.info(f"| 📊 Getting overall file information summary...")
            summary = await self._get_overall_file_summary(task, valid_files)
            if summary and summary.found_answer:
                answer_content = f"## Final Answer\n\n**Answer Found**: Yes\n\n**Answer**: {summary.answer}\n\n"
                await report.add_item(answer_content)
                
                if self.file_path:
                    final_report_content = await report.complete(self.file_path)
                    logger.info(f"✅ Analysis report saved to: {self.file_path}")
                    
                    message = f"Answer found from file information summary.\n\nTask: {task}\n\nAnswer: {summary.answer}, Report saved to: {self.file_path}"
                    
                    return ToolResponse(
                        success=True,
                        message=message,
                        extra=ToolExtra(
                            file_path=self.file_path,
                            data={
                                "task": task,
                                "answer_found": True,
                                "answer": summary.answer,
                                "file_path": self.file_path
                            }
                        )
                    )
                else:
                    message = f"Answer found from file information summary.\n\nTask: {task}\n\nAnswer: {summary.answer}"
                    return ToolResponse(success=True, message=message)
            elif summary:
                summaries.append(summary)
                summary_content = f"## File Information Summary\n\n{summary.summary}\n\n"
                await report.add_item(summary_content)
            
            # Use LLM to classify file types
            logger.info(f"| 🔍 Classifying {len(valid_files)} files by type...")
            file_classifications = await self._classify_files(valid_files)
            
            # Log classifications
            classification_content = "## File Classifications\n\n"
            for file_info in file_classifications:
                file_display = file_info.file if self._is_url(file_info.file) else os.path.basename(file_info.file)
                logger.info(f"| 📋 {file_display}: {file_info.file_type}")
                classification_content += f"- **{file_display}**: {file_info.file_type}\n"
            classification_content += "\n"
            await report.add_item(classification_content)
            
            # Main analysis loop with max_rounds
            for round_num in range(1, self.max_rounds + 1):
                logger.info(f"| 🔄 Main analysis round {round_num}/{self.max_rounds}")
                
                round_summaries: List[Summary] = []
                round_content = f"## Round {round_num}\n\n"
                
                # Process each file in this round
                for file_info in file_classifications:
                    file = file_info.file
                    file_type = file_info.file_type
                    
                    file_display = file if self._is_url(file) else os.path.basename(file)
                    logger.info(f"| 📄 Processing {file_type} file: {file_display}")
                    
                    round_content += f"### Processing {file_type} file: {file_display}\n\n"
                    
                    # Analyze based on file type
                    if file_type == "text":
                        await self._analyze_text_file(task, file, round_summaries)
                    elif file_type == "pdf":
                        await self._analyze_pdf_file(task, file, round_summaries)
                    elif file_type == "image":
                        await self._analyze_image_file(task, file, round_summaries)
                    elif file_type == "audio":
                        await self._analyze_audio_file(task, file, round_summaries)
                    elif file_type == "video":
                        await self._analyze_video_file(task, file, round_summaries)
                    
                    # Add summaries from this file to round content
                    for s in round_summaries:
                        round_content += f"- {s.summary}\n"
                        if s.found_answer:
                            round_content += f"  **Answer Found**: {s.answer}\n"
                    round_content += "\n"
                    
                    # Check if answer found after processing this file
                    round_summary = await self._summarize_summaries(task, round_summaries)
                    if round_summary.found_answer:
                        round_content += f"### Round {round_num} Summary\n\n**Answer Found**: Yes\n\n**Answer**: {round_summary.answer}\n\n"
                        await report.add_item(round_content)
                        
                        if self.file_path:
                            final_report_content = await report.complete(self.file_path)
                            logger.info(f"✅ Analysis report saved to: {self.file_path}")
                            
                            message = f"Answer found from file analysis.\n\nTask: {task}\n\nAnswer: {round_summary.answer}, Report saved to: {self.file_path}"
                            
                            return ToolResponse(
                                success=True,
                                message=message,
                                extra=ToolExtra(
                                    file_path=self.file_path,
                                    data={
                                        "task": task,
                                        "round": round_num,
                                        "answer_found": True,
                                        "answer": round_summary.answer,
                                        "file_path": self.file_path
                                    }
                                )
                            )
                        else:
                            message = f"Answer found from file analysis.\n\nTask: {task}\n\nAnswer: {round_summary.answer}"
                            return ToolResponse(success=True, message=message)
                    else:
                        summaries.append(round_summary)
                
                # Add round summary to content
                round_summary = await self._summarize_summaries(task, round_summaries)
                round_content += f"### Round {round_num} Summary\n\n{round_summary.summary}\n\n"
                if round_summary.found_answer:
                    round_content += f"**Answer Found**: Yes\n\n**Answer**: {round_summary.answer}\n\n"
                else:
                    round_content += f"**Answer Found**: No\n\n"
                
                await report.add_item(round_content)
            
            # Final summary
            final_summary = await self._summarize_summaries(task, summaries)
            final_content = f"## Final Summary\n\n{final_summary.summary}\n\n"
            if final_summary.found_answer:
                final_content += f"**Answer Found**: Yes\n\n**Answer**: {final_summary.answer}\n\n"
            else:
                final_content += f"**Answer Found**: No\n\n"
            await report.add_item(final_content)
            
            if self.file_path:
                final_report_content = await report.complete(self.file_path)
                logger.info(f"✅ Analysis report saved to: {self.file_path}")
                
                # Build message parts separately to avoid f-string backslash issue
                status_text = 'Answer found' if final_summary.found_answer else 'No definitive answer found'
                if final_summary.found_answer:
                    answer_text = f'Answer: {final_summary.answer}'
                else:
                    summaries_list = [f'- {s.summary}' for s in summaries[-10:]]
                    answer_text = 'Summaries:\n' + '\n'.join(summaries_list)
                
                message = f"Analysis completed after {self.max_rounds} rounds.\n\nTask: {task}\n\n{status_text}.\n\n{answer_text}"
                message += f"\n\nReport saved to: {self.file_path}"
                
                return ToolResponse(
                    success=final_summary.found_answer,
                    message=message,
                    extra=ToolExtra(
                        file_path=self.file_path,
                        data={
                            "task": task,
                            "rounds": self.max_rounds,
                            "answer_found": final_summary.found_answer,
                            "answer": final_summary.answer if final_summary.found_answer else None,
                            "file_path": self.file_path
                        }
                    )
                )
            else:
                if final_summary.found_answer:
                    message = f"Answer found from all file analysis.\n\nTask: {task}\n\nAnswer: {final_summary.answer}"
                    return ToolResponse(success=True, message=message)
                else:
                    message = f"Analysis completed after {self.max_rounds} rounds but no definitive answer found.\n\nTask: {task}\n\nSummaries:\n" + "\n".join([f"- {s.summary}" for s in summaries[-10:]])
                    return ToolResponse(success=False, message=message)
            
        except Exception as e:
            logger.error(f"| ❌ Error in deep analysis: {e}")
            return ToolResponse(success=False, message=f"Error during deep analysis: {str(e)}")
    
    async def _get_overall_file_summary(self, task: str, files: List[str]) -> Optional[str]:
        """Get overall summary of all files' information before detailed analysis."""
        try:
            # Get file info for all files
            file_infos = []
            for file_path in files:
                try:
                    if self._is_url(file_path):
                        # For URLs, create a simple info dict
                        file_infos.append({
                            "path": file_path,
                            "name": file_path,  # Use full URL as name
                            "info": {
                                "type": "url",
                                "url": file_path,
                                "url_type": self._get_url_type(file_path) or "unknown"
                            }
                        })
                    else:
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
            files_info_text = chr(10).join([
                dedent(f"""
                File: {info['name']}
                Path: {info['path']}
                {('URL Type: ' + info['info'].get('url_type', 'unknown')) 
                if info['info'].get('type') == 'url' else (f"Size: {info['info'].get('size', 'unknown')}" + 
                chr(10) + f"Created: {info['info'].get('created', 'unknown')}" + 
                chr(10) + f"Modified: {info['info'].get('modified', 'unknown')}")}
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
            
            response = await model_manager(model=self.model_name, messages=messages, response_format=SummaryResponse)
            summary_response = response.extra.parsed_model
            summary = Summary(
                id=self._get_next_summary_id(),
                summary=summary_response.summary,
                found_answer=summary_response.found_answer,
                answer=summary_response.answer
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
            
            response = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=SummaryResponse
            )
            
            if response.extra and response.extra.parsed_model:
                summary_response = response.extra.parsed_model
                return Summary(
                    id=self._get_next_summary_id(),
                    summary=summary_response.summary,
                    found_answer=summary_response.found_answer,
                    answer=summary_response.answer
                )
            else:
                # Fallback: parse from message
                summary_text = response.message.strip()
                return Summary(
                    id=self._get_next_summary_id(),
                    summary=summary_text,
                    found_answer=False,
                    answer=None
                )
            
        except Exception as e:
            logger.error(f"| ❌ Error summarizing summaries: {e}")
            return Summary(
                id=self._get_next_summary_id(),
                summary=f"Error summarizing summaries: {e}",
                found_answer=False,
                answer=None
            )
    
    async def _analyze_task_only(self, task: str, summaries: List[Summary], report: Optional[Report] = None) -> None:
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
                
                response = await model_manager(
                    model=self.model_name,
                    messages=messages,
                    response_format=Summary
                )
                
                if response.extra and response.extra.parsed_model:
                    summary = response.extra.parsed_model
                    # Assign ID to parsed summary
                    summary.id = self._get_next_summary_id()
                else:
                    # Fallback: parse from message
                    summary_text = response.message.strip()
                    summary = Summary(
                        id=self._get_next_summary_id(),
                        summary=summary_text,
                        found_answer=False,
                        answer=None
                    )
                
                summaries.append(summary)
                
                # Add round content to report if provided
                if report:
                    round_content = f"## Round {round_num}\n\n{summary.summary}\n\n"
                    if summary.found_answer:
                        round_content += f"**Answer Found**: Yes\n\n**Answer**: {summary.answer}\n\n"
                    else:
                        round_content += f"**Answer Found**: No\n\n"
                    await report.add_item(round_content)
                
                # Check if we found the answer
                if summary.found_answer:
                    logger.info(f"| ✅ Answer found in round {round_num}, early stopping.")
                    return None
            
            logger.info(f"| ✅ Task analysis completed after {self.max_rounds} rounds")
            
        except Exception as e:
            logger.error(f"| ❌ Error analyzing task: {e}")
            error_summary = Summary(id=self._get_next_summary_id(), summary=f"Error analyzing task: {e}", found_answer=False, answer=None)
            summaries.append(error_summary)
            
            # Add error to report if provided
            if report:
                error_content = f"## Error\n\nError analyzing task: {e}\n\n"
                await report.add_item(error_content)
            
            return None
    
    async def _analyze_text_file(self, task: str, file: str, summaries: List[Summary]) -> None:
        """Analyze a single text file or text URL: get file info, convert to markdown, analyze in chunks.
        """
        try:
            # Check if it's a URL, if so download to local first
            local_file_path = file
            if self._is_url(file):
                logger.info(f"| 📄 Text file URL detected, downloading to local...")
                downloaded_path = await self._download_file(file, file_type="text")
                if not downloaded_path:
                    logger.warning(f"| ❌ Failed to download text file from URL: {file}")
                    summaries.append(Summary(id=self._get_next_summary_id(), summary=f"Failed to download text file from URL: {file}", found_answer=False, answer=None))
                    return None
                local_file_path = downloaded_path
                logger.info(f"| 📄 Using downloaded text file: {os.path.basename(local_file_path)}")
            
            # Get file basic info
            file_info = get_file_info(local_file_path)
            logger.info(f"| 📄 Processing text file: {os.path.basename(local_file_path)} ({file_info.get('size', 'unknown')} bytes)")
            
            # Convert to markdown using mdify_tool (automatically saves to base_dir)
            mdify_response = await self.mdify_tool(file_path=local_file_path, output_format="markdown")
            if mdify_response.extra and mdify_response.extra.file_path:
                saved_path = mdify_response.extra.file_path
            
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
            logger.error(f"| ❌ Error analyzing text file {file}: {e}")
            summaries.append(Summary(id=self._get_next_summary_id(), summary=f"Error analyzing text file {file}: {e}", found_answer=False, answer=None))
            return None
    
    async def _analyze_pdf_file(self, task: str, file: str, summaries: List[Summary]) -> None:
        """Analyze a single PDF file or PDF URL: first try LLM direct analysis, then convert to markdown and analyze in chunks if needed.
        """
        try:
            # Step 1: Try LLM direct analysis first
            logger.info(f"| 📄 Step 1: Trying LLM direct analysis of PDF")
            
            # Prepare PDF file_data for LLM
            # For URLs: use the URL string directly (e.g., "https://bitcoin.org/bitcoin.pdf")
            # For local files: convert to data URL using base64 encoding (e.g., "data:application/pdf;base64,...")
            if self._is_url(file):
                # URL: use directly as file_data
                file_data = file
                logger.info(f"| 📄 Using PDF URL: {file}")
            else:
                # Local file: convert to data URL
                file_data = make_file_url(file_path=assemble_project_path(file))
                logger.info(f"| 📄 Using local PDF file: {os.path.basename(file)}")
            
            # Build message with PDF file
            message_content = [
                {
                    "type": "text",
                    "text": dedent(f"""Analyze the following PDF document to answer the task.
                    
                    Task: {task}
                    
                    Extract key information from the PDF that helps answer the task.
                    If the PDF contains the answer to the task, clearly state it.
                    """)
                },
                {
                    "type": "file",
                    "file": {
                        "filename": "document.pdf",
                        "file_data": file_data  # Can be URL string or data URL
                    }
                }
            ]
            
            messages = [
                SystemMessage(content="You are an expert at analyzing PDF documents and extracting key information."),
                HumanMessage(content=message_content)
            ]
            
            # Try LLM direct analysis
            try:
                response = await model_manager(
                    model=self.file_model_name,
                    messages=messages,
                    response_format=SummaryResponse
                )
                
                if response.extra and response.extra.parsed_model:
                    summary_response = response.extra.parsed_model
                    summary = Summary(
                        id=self._get_next_summary_id(),
                        summary=summary_response.summary,
                        found_answer=summary_response.found_answer,
                        answer=summary_response.answer
                    )
                    summaries.append(summary)
                    
                    if summary.found_answer:
                        logger.info(f"| ✅ Answer found via LLM direct analysis, early stopping.")
                        return None
                    else:
                        logger.info(f"| ⚠️ LLM direct analysis did not find answer, proceeding to chunk-based analysis")
                else:
                    logger.warning(f"| ⚠️ LLM direct analysis failed to parse response, proceeding to chunk-based analysis")
            except Exception as e:
                logger.warning(f"| ⚠️ LLM direct analysis failed: {e}, proceeding to chunk-based analysis")
            
            # Step 2: If answer not found, convert to markdown and analyze in chunks
            logger.info(f"| 📄 Step 2: Converting PDF to markdown and analyzing in chunks")
            
            # Check if it's a URL (PDF)
            if self._is_url(file):
                logger.info(f"| 📄 Processing PDF URL: {file}")
                # Fetch PDF content from URL
                doc_result = await fetch_url(file)
                if not doc_result or not doc_result.markdown:
                    logger.warning(f"Failed to fetch PDF from URL: {file}")
                    summaries.append(Summary(id=self._get_next_summary_id(), summary=f"Failed to fetch PDF from URL: {file}", found_answer=False, answer=None))
                    return None
                
                # Save fetched content to a temporary markdown file
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, dir=self.base_dir) as tmp_file:
                    tmp_file.write(doc_result.markdown)
                    saved_path = tmp_file.name
            else:
                # Get file basic info
                file_info = get_file_info(file)
                logger.info(f"| 📄 Processing PDF file: {os.path.basename(file)} ({file_info.get('size', 'unknown')} bytes)")
                
                # Convert to markdown using mdify_tool (automatically saves to base_dir)
                mdify_response = await self.mdify_tool(file_path=file, output_format="markdown")
                if mdify_response.extra and mdify_response.extra.file_path:
                    saved_path = mdify_response.extra.file_path
            
            # Read all lines once
            with open(saved_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            total_lines = len(lines)
            total_chunks = (total_lines + self.chunk_size - 1) // self.chunk_size
            
            # Internal loop: analyze chunks one by one
            for chunk_num in range(1, total_chunks + 1):
                logger.info(f"| 🔄 Analyzing PDF file chunk {chunk_num}/{total_chunks}")
                
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
            
            logger.info(f"| ✅ All chunks of PDF file analyzed")
            
        except Exception as e:
            logger.error(f"| ❌ Error analyzing PDF file {file}: {e}")
            summaries.append(Summary(id=self._get_next_summary_id(), summary=f"Error analyzing PDF file {file}: {e}", found_answer=False, answer=None))
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
            
            response = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=Summary
            )
            
            if response.extra and response.extra.parsed_model:
                parsed_summary = response.extra.parsed_model
                # Assign ID to parsed summary
                parsed_summary.id = self._get_next_summary_id()
                return parsed_summary
            else:
                summary = response.message.strip()
                return Summary(
                    id=self._get_next_summary_id(),
                    summary=summary,
                    found_answer=False,
                    answer=None
                )
            
        except Exception as e:
            logger.error(f"| ❌ Error analyzing markdown chunk: {e}")
            return Summary(id=self._get_next_summary_id(), summary=f"Error analyzing markdown chunk: {e}", found_answer=False, answer=None)
    
    async def _analyze_image_file(self, task: str, file: str, summaries: List[Summary]) -> None:
        """Analyze a single image file or image URL: first try LLM direct analysis, then analyze multiple times if needed.
        """
        try:
            # Check if it's a URL
            is_url = self._is_url(file)
            if not is_url and not os.path.exists(file):
                logger.warning(f"Image file not found: {file}")
                return None
            
            # Step 1: Try LLM direct analysis first
            logger.info(f"| 🖼️ Step 1: Trying LLM direct analysis of image")
            
            # Prepare image URL for LLM
            # For URLs: use the URL string directly (e.g., "https://example.com/image.jpg")
            # For local files: convert to data URL using base64 encoding (e.g., "data:image/jpeg;base64,...")
            if is_url:
                # URL: use directly
                image_url_value = file
                logger.info(f"| 🖼️ Using image URL: {file}")
            else:
                # Local file: convert to data URL
                image_url_value = make_file_url(file_path=assemble_project_path(file))
                logger.info(f"| 🖼️ Using local image file: {os.path.basename(file)}")
            
            # Build message with image
            message_content = [
                {
                    "type": "text",
                    "text": dedent(f"""Analyze the following image to answer the task.
                    
                    Task: {task}
                    
                    Extract key information from the image that helps answer the task.
                    Focus on visual elements, text in images, patterns, and any relevant details.
                    If the image contains the answer to the task, clearly state it.
                    """)
                },
                {
                    "type": "image_url",
                    "image_url": {"url": image_url_value}  # Can be URL string or data URL
                }
            ]
            
            messages = [
                SystemMessage(content="You are an expert at analyzing images and extracting visual information."),
                HumanMessage(content=message_content)
            ]
            
            # Try LLM direct analysis
            try:
                response = await model_manager(
                    model=self.model_name,
                    messages=messages,
                    response_format=SummaryResponse
                )
                
                if response.extra and response.extra.parsed_model:
                    summary_response = response.extra.parsed_model
                    summary = Summary(
                        id=self._get_next_summary_id(),
                        summary=summary_response.summary,
                        found_answer=summary_response.found_answer,
                        answer=summary_response.answer
                    )
                    summaries.append(summary)
                    
                    if summary.found_answer:
                        logger.info(f"| ✅ Answer found via LLM direct analysis, early stopping.")
                        return None
                    else:
                        logger.info(f"| ⚠️ LLM direct analysis did not find answer, proceeding to multi-step analysis")
                else:
                    logger.warning(f"| ⚠️ LLM direct analysis failed to parse response, proceeding to multi-step analysis")
            except Exception as e:
                logger.warning(f"| ⚠️ LLM direct analysis failed: {e}, proceeding to multi-step analysis")
            
            # Step 2: If answer not found, analyze image multiple times
            logger.info(f"| 🖼️ Step 2: Analyzing image with multiple steps")
            
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
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": image_url_value}
                    })
                except Exception as e:
                    logger.warning(f"Failed to process image {file}: {e}")
                    return None
                
                messages = [
                    SystemMessage(content="You are an expert at analyzing images and extracting visual information."),
                    HumanMessage(content=message_content)
                ]
                
                response = await model_manager(
                    model=self.model_name,
                    messages=messages,
                    response_format=Summary
                )
                
                if response.extra and response.extra.parsed_model:
                    summary = response.extra.parsed_model
                    # Assign ID to parsed summary
                    summary.id = self._get_next_summary_id()
                else:
                    # Fallback: parse from message
                    summary_text = response.message.strip()
                    summary = Summary(
                        id=self._get_next_summary_id(),
                        summary=summary_text,
                        found_answer=False,
                        answer=None
                    )
                
                summaries.append(summary)
                
                # Check if answer found after each step
                if summary.found_answer:
                    logger.info(f"| ✅ Answer found in image step {step_num}, early stopping.")
                    return None
            
            logger.info(f"| ✅ Image analysis completed after {self.max_steps} steps")
            
        except Exception as e:
            logger.error(f"| ❌ Error analyzing image file {file}: {e}")
            summaries.append(Summary(id=self._get_next_summary_id(), summary=f"Error analyzing image file {file}: {e}", found_answer=False, answer=None))
            return None
    
    async def _analyze_audio_file(self, task: str, file: str, summaries: List[Summary]) -> None:
        """Analyze a single audio file or audio URL: directly send to LLM for analysis."""
        try:
            # Check if it's a URL, if so download to local first
            local_file_path = file
            if self._is_url(file):
                logger.info(f"| 🎵 Audio URL detected, downloading to local...")
                downloaded_path = await self._download_file(file, file_type="audio")
                if not downloaded_path:
                    logger.warning(f"| ❌ Failed to download audio from URL: {file}")
                    summaries.append(Summary(id=self._get_next_summary_id(), summary=f"Failed to download audio from URL: {file}", found_answer=False, answer=None))
                    return None
                local_file_path = downloaded_path
                logger.info(f"| 🎵 Using downloaded audio file: {os.path.basename(local_file_path)}")
            
            if not os.path.exists(local_file_path):
                logger.warning(f"Audio file not found: {local_file_path}")
                return None
            
            # Get file basic info
            file_info = get_file_info(local_file_path)
            logger.info(f"| 🎵 Processing audio file: {os.path.basename(local_file_path)} ({file_info.get('size', 'unknown')} bytes)")
            
            # Encode audio file to base64
            audio_base64 = encode_file_base64(file_path=assemble_project_path(local_file_path))
            audio_format = os.path.splitext(local_file_path)[1][1:].lower()  # Get extension without dot
            
            # Build message with audio file
            message_content = [
                {
                    "type": "text",
                    "text": dedent(f"""Analyze the following audio file to answer the task.
                    
                    Task: {task}
                    
                    Transcribe the audio and extract key information that helps answer the task.
                    If the audio contains the answer to the task, clearly state it.
                    """)
                },
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_base64,
                        "format": audio_format  # "mp3", "wav", "m4a", "ogg", "flac"
                    }
                }
            ]
            
            messages = [
                SystemMessage(content="You are an expert at analyzing audio files, transcribing speech, and extracting key information."),
                HumanMessage(content=message_content)
            ]
            
            # Try LLM direct analysis
            try:
                response = await model_manager(
                    model=self.file_model_name,
                    messages=messages,
                    response_format=SummaryResponse
                )
                
                if response.extra and response.extra.parsed_model:
                    summary_response = response.extra.parsed_model
                    summary = Summary(
                        id=self._get_next_summary_id(),
                        summary=summary_response.summary,
                        found_answer=summary_response.found_answer,
                        answer=summary_response.answer
                    )
                    summaries.append(summary)
                    
                    if summary.found_answer:
                        logger.info(f"| ✅ Answer found via LLM direct analysis.")
                        return None
                    else:
                        logger.info(f"| ⚠️ LLM direct analysis did not find answer.")
                else:
                    logger.warning(f"| ⚠️ LLM direct analysis failed to parse response.")
            except Exception as e:
                logger.error(f"| ❌ Error in LLM audio analysis: {e}")
                summaries.append(Summary(id=self._get_next_summary_id(), summary=f"Error analyzing audio file {local_file_path}: {e}", found_answer=False, answer=None))
                return None
            
        except Exception as e:
            logger.error(f"| ❌ Error analyzing audio file {file}: {e}")
            summaries.append(Summary(id=self._get_next_summary_id(), summary=f"Error analyzing audio file {file}: {e}", found_answer=False, answer=None))
            return None
    
    async def _analyze_video_file(self, task: str, file: str, summaries: List[Summary]) -> None:
        """Analyze a single video file or video URL: first try LLM direct analysis, then convert to markdown and analyze in chunks if needed.
        """
        try:
            # Check if it's a URL
            is_url = self._is_url(file)
            local_file_path = file
            
            if is_url:
                if self._is_youtube_url(file):
                    # YouTube URL: use directly for LLM analysis
                    video_url_value = file
                    logger.info(f"| 🎬 Using YouTube video URL: {file}")
                else:
                    # Non-YouTube URL: download to local first
                    logger.info(f"| 🎬 Non-YouTube video URL detected, downloading to local...")
                    downloaded_path = await self._download_file(file, file_type="video")
                    if not downloaded_path:
                        logger.warning(f"| ❌ Failed to download video from URL: {file}")
                        summaries.append(Summary(id=self._get_next_summary_id(), summary=f"Failed to download video from URL: {file}", found_answer=False, answer=None))
                        return None
                    local_file_path = downloaded_path
                    # For LLM analysis, convert to data URL
                    video_url_value = make_file_url(file_path=assemble_project_path(local_file_path))
                    logger.info(f"| 🎬 Using downloaded video file: {os.path.basename(local_file_path)}")
            else:
                # Local file: check if exists
                if not os.path.exists(file):
                    logger.warning(f"Video file not found: {file}")
                    return None
                
                # Local file: convert to data URL for LLM analysis
                video_url_value = make_file_url(file_path=assemble_project_path(file))
                logger.info(f"| 🎬 Using local video file: {os.path.basename(file)}")
            
            # Step 1: Try LLM direct analysis first
            logger.info(f"| 🎬 Step 1: Trying LLM direct analysis of video")
            
            # Build message with video
            message_content = [
                {
                    "type": "text",
                    "text": dedent(f"""Analyze the following video to answer the task.
                    
                    Task: {task}
                    
                    Extract key information from the video that helps answer the task.
                    If the video contains the answer to the task, clearly state it.
                    """)
                },
                {
                    "type": "video_url",
                    "video_url": {"url": video_url_value}  # Can be YouTube URL or data URL
                }
            ]
            
            messages = [
                SystemMessage(content="You are an expert at analyzing videos and extracting key information."),
                HumanMessage(content=message_content)
            ]
            
            # Try LLM direct analysis
            try:
                response = await model_manager(
                    model=self.file_model_name,
                    messages=messages,
                    response_format=SummaryResponse
                )
                
                if response.extra and response.extra.parsed_model:
                    summary_response = response.extra.parsed_model
                    summary = Summary(
                        id=self._get_next_summary_id(),
                        summary=summary_response.summary,
                        found_answer=summary_response.found_answer,
                        answer=summary_response.answer
                    )
                    summaries.append(summary)
                    
                    if summary.found_answer:
                        logger.info(f"| ✅ Answer found via LLM direct analysis, early stopping.")
                        return None
                    else:
                        logger.info(f"| ⚠️ LLM direct analysis did not find answer, proceeding to chunk-based analysis")
                else:
                    logger.warning(f"| ⚠️ LLM direct analysis failed to parse response, proceeding to chunk-based analysis")
            except Exception as e:
                logger.warning(f"| ⚠️ LLM direct analysis failed: {e}, proceeding to chunk-based analysis")
            
            # Step 2: If answer not found, convert to markdown and analyze in chunks
            logger.info(f"| 🎬 Step 2: Converting video to markdown and analyzing in chunks")
            
            # Now use local_file_path (which may be the original file or downloaded file)
            if is_url and self._is_youtube_url(file):
                # YouTube URL: fetch content
                logger.info(f"| 🎬 Processing YouTube video URL: {file}")
                doc_result = await fetch_url(file)
                if not doc_result or not doc_result.markdown:
                    logger.warning(f"Failed to fetch video content from URL: {file}")
                    summaries.append(Summary(id=self._get_next_summary_id(), summary=f"Failed to fetch video content from URL: {file}", found_answer=False, answer=None))
                    return None
                
                # Save fetched content to a temporary markdown file
                import tempfile
                with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, dir=self.base_dir) as tmp_file:
                    tmp_file.write(doc_result.markdown)
                    saved_path = tmp_file.name
            else:
                # Local file (original or downloaded): convert to markdown
                file_info = get_file_info(local_file_path)
                logger.info(f"| 🎬 Processing video file: {os.path.basename(local_file_path)} ({file_info.get('size', 'unknown')} bytes)")
                
                # Convert to markdown using mdify_tool (automatically saves to base_dir)
                mdify_response = await self.mdify_tool(file_path=local_file_path, output_format="markdown")
                # Get saved_path from ToolExtra: check file_path first, then data dict
                if mdify_response.extra and mdify_response.extra.file_path:
                    saved_path = mdify_response.extra.file_path
                elif mdify_response.extra and mdify_response.extra.data:
                    saved_path = mdify_response.extra.data.get("saved_path")
                else:
                    raise ValueError("mdify_tool did not return saved_path in extra")
            
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
            logger.error(f"| ❌ Error analyzing video file {file}: {e}")
            summaries.append(Summary(id=self._get_next_summary_id(), summary=f"Error analyzing video file {file}: {e}", found_answer=False, answer=None))
            return None
    
    async def _validate_file(self, file_path: str) -> bool:
        """Validate if file can be processed (supports both file paths and URLs)."""
        try:
            # Check if it's a URL
            if self._is_url(file_path):
                url_type = self._get_url_type(file_path)
                if url_type:
                    # Validate URL type
                    if url_type == "video" and not self._is_youtube_url(file_path):
                        logger.warning(f"Video URL must be YouTube: {file_path}")
                        return False
                    return True
                else:
                    logger.warning(f"Unsupported URL type: {file_path}")
                    return False
            
            # Validate local file
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

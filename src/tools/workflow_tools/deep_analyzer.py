"""Deep Analyzer Tool - A workflow agent for multi-step analysis of tasks with files."""

import asyncio
import os
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field
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
from src.infrastructures.models import model_manager
from src.tools.default_tools.mdify import MdifyTool
from src.tools.protocol.tool import BaseTool
from src.tools.protocol.types import ToolResponse

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

class AnalysisStep(BaseModel):
    """Represents a single analysis step."""
    step_number: int
    analysis_type: str  # e.g., "file_analysis", "task_analysis", "synthesis"
    input_data: str
    insights: List[str]
    conclusion: str
    confidence: float  # 0.0 to 1.0

@tcp.tool()
class DeepAnalyzerTool(BaseTool):
    """A deep analysis tool that performs multi-step analysis of tasks with files."""

    name: str = "deep_analyzer"
    type: str = "Deep Analyzer"
    description: str = _DEEP_ANALYZER_DESCRIPTION
    args_schema: Type[DeepAnalyzerArgs] = DeepAnalyzerArgs
    metadata: Dict[str, Any] = {}
    
    # Configuration parameters as class attributes
    max_steps: int = Field(default=3, description="Maximum analysis steps")
    max_file_size: int = Field(default=10 * 1024 * 1024, description="Max file size in bytes (10MB)")
    supported_file_types: List[str] = Field(
        default=[
            # Text and Markup Files
            ".txt", ".md", ".json", ".csv", ".xml", ".yaml", ".yml",
            # Programming Files
            ".py", ".js", ".html", ".css", ".java", ".cpp", ".c", ".h",
            # Document Files
            ".pdf", ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
            # Image Files (for multimodal analysis)
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".svg",
            # Audio Files
            ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".m4b", ".m4p",
            # Video Files
            ".mp4", ".avi", ".mov", ".wmv", ".flv", ".webm", ".m4v"
        ],
        description="Supported file types organized by category"
    )
    analysis_prompt: str = Field(
        default="Analyze the provided content and extract key insights relevant to the task. Focus on factual information, patterns, and actionable findings.",
        description="Prompt for analyzing content"
    )
    synthesis_prompt: str = Field(
        default="Based on all the analysis steps completed so far, synthesize the findings and determine if we have sufficient information to answer the task. If not, suggest the next analysis approach.",
        description="Prompt for synthesizing analysis results"
    )
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
    analysis_history: List[AnalysisStep] = Field(
        default=[],
        description="The analysis history."
    )
    all_insights: List[str] = Field(
        default=[],
        description="All insights collected during analysis."
    )
    file_contents: Dict[str, str] = Field(
        default={},
        description="Cached file contents."
    )
    image_files: List[str] = Field(
        default=[],
        description="List of image file paths for multimodal analysis."
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
        
        # Store analysis history
        self.analysis_history: List[AnalysisStep] = []
        self.all_insights: List[str] = []
        self.file_contents: Dict[str, str] = {}
        self.image_files: List[str] = []

    async def _arun(self, task: str, files: Optional[List[str]] = None) -> ToolResponse:
        """Execute deep analysis workflow."""
        try:
            logger.info(f"| 🚀 Starting DeepAnalyzerTool: {task}")
            if files:
                logger.info(f"| 📂 Attached files: {files}")
            
            # Reset analysis history
            self.analysis_history = []
            self.all_insights = []
            self.file_contents = {}
            self.image_files = []
            
            # Step 1: Load and organize files into enhanced task
            enhanced_task = await self._prepare_enhanced_task(task, files)
            logger.info(f"| ✅ Enhanced task prepared")
            
            # Execute multiple analysis rounds
            for round_num in range(1, self.max_steps + 1):
                logger.info(f"| 🔄 Starting analysis round {round_num}/{self.max_steps}")
                
                # Step 2: Analyze enhanced task and images
                insights = await self._analyze_enhanced_task(enhanced_task, round_num)
                logger.info(f"| ✅ Analysis completed for round {round_num}: {len(insights)} insights")
                self.all_insights.extend(insights)
                
                # Step 3: Summarize current round
                round_summary = await self._summarize_round(insights, enhanced_task, round_num)
                logger.info(f"| ✅ Summarized round {round_num}: {round_summary}")
                
                # Record round information
                analysis_step = AnalysisStep(
                    step_number=round_num,
                    analysis_type="enhanced_analysis",
                    input_data=enhanced_task[:200] + "..." if len(enhanced_task) > 200 else enhanced_task,
                    insights=insights,
                    conclusion=round_summary,
                    confidence=0.0  # Will be calculated in evaluation
                )
                self.analysis_history.append(analysis_step)
                
                # Step 4: Check if analysis is complete
                final_summary = await self._evaluate_completeness(task)
                if "ANALYSIS_COMPLETE" in final_summary:
                    logger.info(f"| ✅ Analysis completed in round {round_num}")
                    result = await self._format_final_result(final_summary, round_num)
                    return ToolResponse(content=result)
                
                logger.info(f"| ✅ Round {round_num} completed, continuing to next round")
            
            # If all rounds completed without finding answer
            logger.warning("| ❌ Maximum rounds reached without completing analysis")
            result = await self._format_failure_result(task)
            return ToolResponse(content=result)
            
        except Exception as e:
            logger.error(f"| ❌ Error in deep analysis: {e}")
            return ToolResponse(content=f"Error during deep analysis: {str(e)}")
        
    async def _prepare_enhanced_task(self, task: str, files: Optional[List[str]]) -> str:
        """Prepare enhanced task by loading and organizing files."""
        if not files:
            return task
        
        file_infos = []
        
        for file_path in files:
            try:
                # Validate file
                if not await self._validate_file(file_path):
                    logger.warning(f"Skipping invalid file: {file_path}")
                    continue
                
                # Check if it's an image file (for multimodal analysis)
                _, ext = os.path.splitext(file_path.lower())
                image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.tiff', '.svg']
                if ext in image_extensions:
                    self.image_files.append(file_path)
                    logger.info(f"| ✅ Added image file for multimodal analysis: {file_path}")
                    continue
                
                # Extract file content for non-image files
                file_info = await self._extract_file_content(file_path)
                file_infos.append(file_info)
                self.file_contents[file_path] = file_info["content"]
                logger.info(f"| ✅ Loaded file: {file_path}")
                
            except Exception as e:
                logger.error(f"| ❌ Error processing file {file_path}: {e}")
        
        # Generate enhanced task
        enhanced_task = await self._generate_enhanced_task(task, file_infos)
        return enhanced_task

    async def _extract_file_content(self, file: str) -> Dict[str, Any]:
        """Extract file information."""
        info = get_file_info(file)
        
        # Extract file content
        file_content = await self.mdify_tool._arun(file, "markdown")
        
        # Use LLM to summarize the file content
        system_prompt = "You are a helpful assistant that summarizes file content."
        
        user_prompt = dedent(f"""
            Summarize the following file content as 1-3 sentences:
            {file_content}
        """)
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]
        
        summary = await self.model.ainvoke(messages)
        
        info["content"] = file_content
        info["summary"] = summary.content
        
        return info

    async def _generate_enhanced_task(self, task: str, file_infos: List[Dict[str, Any]]) -> str:
        """Generate enhanced task with file information."""
        if not file_infos:
            return task
        
        attach_files_string = "\n".join([
            f"File: {file_info['path']}\nSummary: {file_info['summary']}" 
            for file_info in file_infos
        ])
        
        enhanced_task = dedent(f"""
        - Task:
        {task}
        - Attached files:
        {attach_files_string}
        """)
        
        return enhanced_task

    async def _analyze_enhanced_task(self, enhanced_task: str, round_num: int) -> List[str]:
        """Analyze enhanced task with images using multimodal approach."""
        try:
            # Prepare text context
            text_context = enhanced_task
            
            # Add previous insights for context
            if self.all_insights:
                previous_insights = "\n".join(self.all_insights[-5:])
                text_context += f"\n\nPrevious insights from round {round_num-1}:\n{previous_insights}"
            
            # Build multimodal message
            message_content = [
                {"type": "text", "text": dedent(f"""Analyze the following task and files in detail.
                
                {text_context}
                
                Extract key insights, patterns, and findings that help answer the task.
                Focus on:
                - Key information from the files
                - Patterns and relationships
                - Actionable insights
                - Important details that might be overlooked
                
                Provide specific insights as bullet points.""")}
            ]
            
            # Add all images to the message
            for image_path in self.image_files:
                if os.path.exists(image_path):
                    try:
                        image = Image.open(assemble_project_path(image_path))
                        image_url = make_image_url(encode_image_base64(image))
                        message_content.append({
                            "type": "image_url", 
                            "image_url": {"url": image_url}
                        })
                    except Exception as e:
                        logger.warning(f"Failed to process image {image_path}: {e}")
            
            messages = [
                SystemMessage(content="You are an expert analyst specializing in comprehensive file and visual content analysis."),
                HumanMessage(content=message_content)
            ]
            
            response = await self.model.ainvoke(messages)
            
            if response and response.content.strip():
                content = response.content.strip()
                insights = [line.strip() for line in content.split('\n') if line.strip().startswith(('-', '•', '*'))]
                return insights
            
        except Exception as e:
            logger.warning(f"Failed to analyze enhanced task: {e}")
        
        return []

    async def _summarize_round(self, insights: List[str], enhanced_task: str, round_num: int) -> str:
        """Summarize the current analysis round."""
        if not insights:
            return f"No insights found in round {round_num}"
        
        prompt = dedent(f"""Summarize the analysis results for this round.
        
        Round: {round_num}
        Insights found: {len(insights)}
        
        Key insights:
        {chr(10).join(insights)}
        
        Provide a brief summary (1-2 sentences) of what was discovered in this round.
        Focus on the most important findings and their relevance to the task.""")
        
        try:
            message = HumanMessage(content=prompt)
            response = await self.model.ainvoke([message])
            return response.content.strip()
        
        except Exception as e:
            logger.warning(f"Failed to summarize round: {e}")
            return f"Round {round_num} analysis completed with {len(insights)} insights"

    async def _evaluate_completeness(self, task: str) -> str:
        """Evaluate if we have found a complete answer."""
        if not self.all_insights:
            return "No insights collected yet"
        
        prompt = dedent(f"""Evaluate if we have collected sufficient information to answer the analysis task.
        
        Analysis Task: {task}
        
        Insights collected so far:
        {chr(10).join(self.all_insights)}
        
        Determine if we have enough information to provide a complete answer.
        
        If YES, respond with: "ANALYSIS_COMPLETE: [brief explanation of what we found]"
        If NO, respond with: "INCOMPLETE: [explanation of what information is still missing]"
        
        Consider:
        - Does the information directly address the task?
        - Is there sufficient detail and depth?
        - Are there multiple perspectives or sources?
        - Is the information comprehensive and reliable?""")
        
        try:
            message = HumanMessage(content=prompt)
            response = await self.model.ainvoke([message])
            if response and response.content.strip():   
                return response.content.strip()
            else:
                return self._fallback_completeness_check(task)
        
        except Exception as e:
            logger.warning(f"Failed to evaluate completeness with LLM: {e}")
            return self._fallback_completeness_check(task)
    
    def _fallback_completeness_check(self, task: str) -> str:
        """Fallback method for completeness evaluation using simple heuristics."""
        if len(self.all_insights) >= 5:
            task_lower = task.lower()
            key_terms = task_lower.split()
            
            coverage = 0
            for insight in self.all_insights:
                insight_lower = insight.lower()
                for term in key_terms:
                    if term in insight_lower:
                        coverage += 1
            
            if coverage >= len(key_terms) * 2:
                return "ANALYSIS_COMPLETE: Sufficient information collected to answer the task"
        
        return "INCOMPLETE: Need more information to provide a complete answer"

    async def _validate_file(self, file_path: str) -> bool:
        """Validate if file can be processed."""
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"File does not exist: {file_path}")
                return False
            
            # Check file size
            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size:
                logger.warning(f"File too large: {file_path} ({file_size} bytes)")
                return False
            
            # Check file type
            _, ext = os.path.splitext(file_path.lower())
            if ext not in self.supported_file_types:
                logger.warning(f"Unsupported file type: {file_path} ({ext})")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating file {file_path}: {e}")
            return False

    async def _format_final_result(self, conclusion: str, round_num: int) -> str:
        """Format the final successful result."""
        try:
            prompt = dedent(f"""Format the final analysis results into a comprehensive report.
            
            Analysis completed in {round_num} rounds.
            
            Final Conclusion: {conclusion}
            
            All Insights Collected ({len(self.all_insights)} total):
            {chr(10).join(self.all_insights)}
            
            Analysis Statistics:
            - Total rounds: {len(self.analysis_history)}
            - Total insights: {len(self.all_insights)}
            - Files analyzed: {len(self.file_contents)}
            - Images analyzed: {len(self.image_files)}
            
            Create a professional analysis report that includes:
            1. Executive summary
            2. Key findings (prioritize the most important insights)
            3. Supporting evidence
            4. Analysis methodology
            5. Conclusions and recommendations
            
            Use appropriate formatting and make it engaging and easy to read.""")
            
            message = HumanMessage(content=prompt)
            response = await self.model.ainvoke([message])
            
            if response and response.content.strip():
                return response.content.strip()
            else:
                return self._fallback_format_result(conclusion, round_num)
                
        except Exception as e:
            logger.warning(f"Failed to format final result: {e}")
            return self._fallback_format_result(conclusion, round_num)
    
    def _fallback_format_result(self, conclusion: str, round_num: int) -> str:
        """Basic fallback formatting for successful results."""
        result = f"🎯 Analysis completed in {round_num} rounds!\n\n"
        result += "📋 Summary:\n"
        result += conclusion.replace("ANALYSIS_COMPLETE: ", "") + "\n\n"
        
        result += "🔍 Key Insights:\n"
        for i, insight in enumerate(self.all_insights[-10:], 1):
            result += f"{i}. {insight}\n"
        
        result += f"\n📊 Analysis Statistics:\n"
        result += f"- Total rounds: {len(self.analysis_history)}\n"
        result += f"- Total insights: {len(self.all_insights)}\n"
        result += f"- Files analyzed: {len(self.file_contents)}\n"
        result += f"- Images analyzed: {len(self.image_files)}\n"
        
        return result

    async def _format_failure_result(self, task: str) -> str:
        """Format the failure result when max steps are reached."""
        try:
            prompt = dedent(f"""The analysis task was incomplete after maximum steps. Format a helpful failure report.
            
            Task: {task}
            
            Partial insights collected ({len(self.all_insights)} total):
            {chr(10).join(self.all_insights) if self.all_insights else "No insights collected"}
            
            Analysis Statistics:
            - Total rounds: {len(self.analysis_history)}
            - Total insights: {len(self.all_insights)}
            - Files analyzed: {len(self.file_contents)}
            - Images analyzed: {len(self.image_files)}
            
            Create a professional failure report that includes:
            1. Clear explanation of what happened
            2. Summary of partial findings (if any)
            3. Specific recommendations for improvement
            4. Alternative approaches to try
            
            Be constructive and helpful.""")
            
            message = HumanMessage(content=prompt)
            response = await self.model.ainvoke([message])
            
            if response and response.content.strip():
                return response.content.strip()
            else:
                return self._fallback_format_failure_result(task)
                
        except Exception as e:
            logger.warning(f"Failed to format failure result: {e}")
            return self._fallback_format_failure_result(task)
    
    def _fallback_format_failure_result(self, task: str) -> str:
        """Basic fallback formatting for failure results."""
        result = f"❌ Analysis incomplete after maximum steps reached.\n\n"
        result += f"📋 Task: {task}\n\n"
        
        if self.all_insights:
            result += "🔍 Partial insights collected:\n"
            for i, insight in enumerate(self.all_insights[-5:], 1):
                result += f"{i}. {insight}\n"
            
            result += "\n💡 Recommendations:\n"
            result += "- Try providing more specific files or data\n"
            result += "- Consider breaking down the task into smaller subtasks\n"
            result += "- Check if the files contain the required information"
        else:
            result += "🔍 No insights collected. The task might be too complex or the files might not contain relevant information."
        
        return result

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
            return ToolResponse(content=f"Error in synchronous execution: {e}")

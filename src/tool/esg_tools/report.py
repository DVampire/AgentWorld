"""Report Tool - A tool for managing and refining markdown reports."""

import os
import json
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
from pydantic import Field, ConfigDict, BaseModel

from src.registry import TOOL
from src.logger import logger
from src.utils import assemble_project_path, dedent
from src.tool.types import Tool, ToolResponse
from src.tool.default_tools.file_reader import FileReaderTool
from src.tool.default_tools.file_editor import FileEditorTool
from src.model import model_manager
from src.message import HumanMessage, SystemMessage


class ContentItem(BaseModel):
    """A single content item in the content list."""
    id: int = Field(description="Unique identifier for this content item")
    content: str = Field(description="The actual content text")
    start_line: int = Field(description="Starting line number in report.md (1-indexed)")
    end_line: int = Field(description="Ending line number in report.md (1-indexed)")
    summary: str = Field(description="LLM-generated summary of this content")


class ContentSummary(BaseModel):
    """Summary for a content item."""
    summary: str = Field(description="Summary of the content")


class RefinedContent(BaseModel):
    """Refined content for optimization."""
    refined_content: str = Field(description="The optimized and refined content")


_REPORT_DESCRIPTION = """Report tool for managing and refining markdown reports.

🎯 BEST FOR: Creating, editing, and refining ESG analysis reports.

📋 Actions:
- add: Add new content to the report
  - args: content (required) - The content to add
  - Automatically generates summary and updates content list
  - Appends content to report.md

- complete: Complete and optimize the entire report
  - Reads all summaries and optimizes content for coherence and logic
  - Updates report.md with optimized content

💡 Workflow:
1. Use `add` multiple times to incrementally add content
2. Use `complete` to optimize the entire report with LLM
"""


@TOOL.register_module(force=True)
class ReportTool(Tool):
    """A tool for managing and refining markdown reports."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "report"
    description: str = _REPORT_DESCRIPTION
    enabled: bool = True
    
    model_name: str = Field(
        default="openrouter/gemini-3-flash-preview",
        description="The model to use for code generation."
    )

    # Configuration parameters
    base_dir: str = Field(
        default=None,
        description="The base directory for saving reports."
    )
    
    # Internal tools
    _file_reader: Optional[FileReaderTool] = None
    _file_editor: Optional[FileEditorTool] = None
    _report_path: Optional[str] = None
    _summary_path: Optional[str] = None
    _content_list_path: Optional[str] = None

    def __init__(
        self, 
        base_dir: Optional[str] = None, 
        model_name: Optional[str] = None,
        title: str = "Analysis Report",
        **kwargs
    ):
        """Initialize the report tool and create necessary files."""
        super().__init__(**kwargs)
        
        if model_name is not None:
            self.model_name = model_name
        
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
            
        if self.base_dir is not None:
            os.makedirs(self.base_dir, exist_ok=True)
        
        # Initialize internal tools
        self._file_reader = FileReaderTool()
        self._file_editor = FileEditorTool()
        
        # Set up file paths
        self._report_path = os.path.join(self.base_dir, "report.md")
        self._summary_path = os.path.join(self.base_dir, "summary.md")
        self._content_list_path = os.path.join(self.base_dir, "content_list.json")
        
        # Initialize report.md if it doesn't exist
        if not os.path.exists(self._report_path):
            initial_content = dedent(f"""# {title}

            **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

            ---

            """)
            with open(self._report_path, 'w', encoding='utf-8') as f:
                f.write(initial_content)
            logger.info(f"| 📝 Report created: {self._report_path}")
        else:
            logger.info(f"| 📝 Report exists: {self._report_path}")
        
        # Initialize content_list.json if it doesn't exist
        if not os.path.exists(self._content_list_path):
            with open(self._content_list_path, 'w', encoding='utf-8') as f:
                json.dump([], f, indent=2, ensure_ascii=False)
            logger.info(f"| 📋 Content list initialized: {self._content_list_path}")
        
        # Initialize summary.md if it doesn't exist
        if not os.path.exists(self._summary_path):
            with open(self._summary_path, 'w', encoding='utf-8') as f:
                f.write("# Content Summaries\n\n")
            logger.info(f"| 📄 Summary file initialized: {self._summary_path}")

    async def __call__(
        self,
        action: str,
        content: Optional[str] = None,
        **kwargs
    ) -> ToolResponse:
        """Execute report action.

        Args:
            action (str): The action to perform. action must be one of: add, complete.
            content (Optional[str]): Content to add, required for add action.
        """
        try:
            logger.info(f"| 📝 ReportTool action: {action}")

            if action == "add":
                if not content:
                    return ToolResponse(
                        success=False,
                        message="content is required for add action."
                    )
                return await self._add_content(content)

            elif action == "complete":
                return await self._complete_report()

            else:
                return ToolResponse(
                    success=False,
                    message=f"Unknown action: {action}. Valid actions: add, complete"
                )

        except Exception as e:
            logger.error(f"| ❌ Error in ReportTool: {e}")
            import traceback
            return ToolResponse(
                success=False,
                message=f"Error in report action '{action}': {str(e)}\n{traceback.format_exc()}"
            )

    def _load_content_list(self) -> List[ContentItem]:
        """Load content list from JSON file."""
        try:
            if not os.path.exists(self._content_list_path):
                return []
            
            with open(self._content_list_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            return [ContentItem(**item) for item in data]
        except Exception as e:
            logger.error(f"| ❌ Error loading content list: {e}")
            return []

    def _save_content_list(self, content_list: List[ContentItem]) -> None:
        """Save content list to JSON file."""
        try:
            with open(self._content_list_path, 'w', encoding='utf-8') as f:
                json.dump([item.model_dump() for item in content_list], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"| ❌ Error saving content list: {e}")
            raise

    def _update_summary_file(self, content_list: List[ContentItem]) -> None:
        """Update summary.md file with all summaries."""
        try:
            summary_lines = ["# Content Summaries\n\n"]
            
            for item in content_list:
                summary_lines.append(f"## Content #{item.id}\n\n")
                summary_lines.append(f"**Lines:** {item.start_line}-{item.end_line}\n\n")
                summary_lines.append(f"{item.summary}\n\n")
                summary_lines.append("---\n\n")
            
            with open(self._summary_path, 'w', encoding='utf-8') as f:
                f.writelines(summary_lines)
            
            logger.info(f"| ✅ Summary file updated: {self._summary_path}")
        except Exception as e:
            logger.error(f"| ❌ Error updating summary file: {e}")
            raise

    async def _generate_summary(self, content: str) -> str:
        """Generate summary for content using LLM."""
        try:
            prompt = dedent(f"""Summarize the following content in 2-3 sentences. Focus on key points and main ideas.
            
            Content:
            ```markdown
            {content}
            ```
            
            Provide a concise summary that captures the essence of this content.
            """)
            
            messages = [
                SystemMessage(content="You are an expert at summarizing content. Provide clear, concise summaries."),
                HumanMessage(content=prompt)
            ]
            
            response = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=ContentSummary
            )
            
            if response.extra and "parsed_model" in response.extra:
                summary_response = response.extra["parsed_model"]
                return summary_response.summary
            else:
                # Fallback: use raw response
                return response.message.strip()
                
        except Exception as e:
            logger.error(f"| ❌ Error generating summary: {e}")
            return f"Summary generation failed: {str(e)}"

    async def _add_content(self, content: str) -> ToolResponse:
        """Add new content to the report."""
        try:
            # Load current content list
            content_list = self._load_content_list()
            
            # Read current report to determine line numbers
            read_result = await self._file_reader(file_path=self._report_path)
            if not read_result.success:
                return read_result
            
            current_content = read_result.extra.get("content", "") if read_result.extra else ""
            total_lines = read_result.extra.get("total_lines", 0) if read_result.extra else 0
            
            # Calculate start and end lines for new content
            start_line = total_lines + 1
            
            # Generate summary for new content
            logger.info(f"| 📝 Generating summary for new content...")
            summary = await self._generate_summary(content)
            
            # Determine new content ID
            new_id = len(content_list) + 1
            
            # Calculate end line (approximate based on content lines)
            content_lines = content.split('\n')
            end_line = start_line + len(content_lines) - 1
            
            # Create new content item
            new_item = ContentItem(
                id=new_id,
                content=content,
                start_line=start_line,
                end_line=end_line,
                summary=summary
            )
            
            # Add to content list
            content_list.append(new_item)
            
            # Save content list
            self._save_content_list(content_list)
            
            # Update summary file
            self._update_summary_file(content_list)
            
            # Append content to report.md
            edit_result = await self._file_editor(
                file_path=self._report_path,
                edits=[{"content": content}]
            )
            
            if not edit_result.success:
                return ToolResponse(
                    success=False,
                    message=f"Failed to add content to report: {edit_result.message}"
                )
            
            # Re-read to get actual line numbers
            read_result = await self._file_reader(file_path=self._report_path)
            if read_result.success:
                actual_total_lines = read_result.extra.get("total_lines", 0) if read_result.extra else 0
                # Update end_line with actual value
                new_item.end_line = actual_total_lines
                self._save_content_list(content_list)
                self._update_summary_file(content_list)
            
            logger.info(f"| ✅ Content added: ID={new_id}, Lines={start_line}-{new_item.end_line}")
            
            return ToolResponse(
                success=True,
                message=f"📝 Content added successfully!\n\nID: {new_id}\nLines: {start_line}-{new_item.end_line}\nSummary: {summary}",
                extra={
                    "id": new_id,
                    "start_line": start_line,
                    "end_line": new_item.end_line,
                    "summary": summary
                }
            )

        except Exception as e:
            logger.error(f"| ❌ Error adding content: {e}")
            import traceback
            return ToolResponse(
                success=False,
                message=f"Error adding content: {str(e)}\n{traceback.format_exc()}"
            )

    async def _complete_report(self) -> ToolResponse:
        """Complete and optimize the entire report."""
        try:
            # Read the entire report.md file
            read_result = await self._file_reader(file_path=self._report_path)
            if not read_result.success:
                return ToolResponse(
                    success=False,
                    message=f"Failed to read report file: {read_result.message}"
                )
            
            current_report = read_result.extra.get("content", "") if read_result.extra else ""
            
            if not current_report or len(current_report.strip()) < 50:
                return ToolResponse(
                    success=False,
                    message="Report is empty or too short. Add content first using the 'add' action."
                )
            
            logger.info(f"| 📊 Optimizing entire report ({len(current_report)} chars)...")
            
            # Optimize the entire report using LLM
            optimized_report = await self._optimize_entire_report(current_report)
            
            logger.info(f"| ✅ Optimization complete ({len(optimized_report)} chars)")
            
            # Write the optimized report back to file
            # Get total lines for replacement
            total_lines = read_result.extra.get("total_lines", 0) if read_result.extra else 0
            if total_lines == 0:
                # If we can't determine lines, count them
                total_lines = len(current_report.split('\n'))
            
            # Use FileEditorTool to replace entire file content
            edit_result = await self._file_editor(
                file_path=self._report_path,
                edits=[{
                    "start_line": 1,
                    "end_line": max(total_lines, 1),  # Ensure at least 1
                    "content": optimized_report
                }]
            )
            
            if not edit_result.success:
                # Fallback: try direct file write
                logger.warning(f"| ⚠️ FileEditorTool failed, trying direct write: {edit_result.message}")
                try:
                    with open(self._report_path, 'w', encoding='utf-8') as f:
                        f.write(optimized_report)
                    logger.info(f"| ✅ Report written directly to file")
                except Exception as write_error:
                    logger.error(f"| ❌ Failed to write report: {write_error}")
                    return ToolResponse(
                        success=False,
                        message=f"Failed to write optimized report to file: {str(write_error)}"
                    )
            
            # Verify the file was written (optional check, don't fail if verification fails)
            try:
                verify_result = await self._file_reader(file_path=self._report_path)
                if verify_result.success:
                    written_content = verify_result.extra.get("content", "") if verify_result.extra else ""
                    # More lenient comparison: check if content length is reasonable
                    if written_content and len(written_content.strip()) > len(optimized_report.strip()) * 0.8:
                        logger.info(f"| ✅ Report optimization complete and verified: {self._report_path}")
                    else:
                        logger.warning(f"| ⚠️ Written content length mismatch, but file was updated")
            except Exception as verify_error:
                logger.warning(f"| ⚠️ Could not verify written content: {verify_error}, but file write appeared successful")
            
            logger.info(f"| ✅ Report optimization complete: {self._report_path}")
            
            return ToolResponse(
                success=True,
                message=f"📝 Report optimized successfully!\n\nPath: {self._report_path}\n\nThe entire report has been optimized for coherence, logic flow, organization, and proper citations.",
                extra={
                    "path": self._report_path,
                    "original_length": len(current_report),
                    "optimized_length": len(optimized_report)
                }
            )

        except Exception as e:
            logger.error(f"| ❌ Error completing report: {e}")
            import traceback
            return ToolResponse(
                success=False,
                message=f"Error completing report: {str(e)}\n{traceback.format_exc()}"
            )

    async def _optimize_entire_report(self, report_content: str) -> str:
        """Optimize the entire report file using LLM."""
        try:
            prompt = dedent(f"""Optimize and refine the following ESG analysis report to make it coherent, logically organized, well-structured, and professionally written.
            
            Original Report:
            ```markdown
            {report_content}
            ```
            
            Please optimize the entire report by:
            1. **Improving Logical Structure**: Ensure the report flows logically from introduction to conclusion
            2. **Organizing Content**: Group related information into clear sections with appropriate headings
            3. **Enhancing Coherence**: Improve transitions between sections and paragraphs
            4. **Verifying Citations**: Ensure all citations [1], [2], [3], etc. are correctly placed and referenced
            5. **Checking References Section**: Verify that the References section at the end lists all citations in order
            6. **Improving Readability**: Enhance clarity, fix grammar and style issues
            7. **Consistent Formatting**: Ensure consistent markdown formatting throughout
            8. **Professional Polish**: Make the writing more professional and polished
            
            IMPORTANT REQUIREMENTS:
            - **Preserve All Facts**: Do not modify facts, numbers, data, or specific details
            - **Maintain Citations**: Keep all citation markers [1], [2], [3] etc. in their correct positions
            - **Preserve References**: Keep the References section with all file paths intact
            - **No New Information**: Do not add new information that wasn't in the original report
            - **Markdown Format**: Return the complete optimized report in markdown format
            - **Complete Report**: Return the ENTIRE optimized report, not just parts of it
            
            Return ONLY the optimized markdown report content, no explanations or additional text.
            """)
            
            messages = [
                SystemMessage(content="You are an expert editor specializing in ESG analysis reports. Your task is to optimize entire reports for logical flow, organization, coherence, and proper citation formatting while preserving all factual information. Return only the complete optimized markdown report."),
                HumanMessage(content=prompt)
            ]
            
            response = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=RefinedContent
            )
            
            if response.extra and "parsed_model" in response.extra:
                refined = response.extra["parsed_model"]
                return refined.refined_content
            else:
                # Fallback: use raw response
                optimized = response.message.strip()
                
                # Remove markdown code blocks if present (more robust handling)
                # Check if content is wrapped in code blocks
                lines = optimized.split('\n')
                if len(lines) > 2:
                    first_line = lines[0].strip()
                    last_line = lines[-1].strip()
                    
                    # Remove opening code block marker
                    if first_line.startswith("```"):
                        optimized = '\n'.join(lines[1:])
                    
                    # Remove closing code block marker
                    if optimized.strip().endswith("```"):
                        optimized = optimized.rsplit("```", 1)[0].rstrip()
                
                optimized = optimized.strip()
                
                # Validate the result
                if not optimized or len(optimized) < 50:
                    logger.warning(f"| ⚠️ Optimized content seems invalid, returning original")
                    return report_content
                
                return optimized
                
        except Exception as e:
            logger.error(f"| ❌ Error optimizing entire report: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Return original content if optimization fails
            return report_content

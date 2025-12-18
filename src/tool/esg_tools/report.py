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
        default="openrouter/o3",
        description="The model to use for refinement."
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
        title: str = "Analysis Report",
        **kwargs
    ):
        """Initialize the report tool and create necessary files."""
        super().__init__(**kwargs)
        
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path("workdir/tool/report")
            
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
        action: Literal["add", "complete"],
        content: Optional[str] = None,
        **kwargs
    ) -> ToolResponse:
        """Execute report action.

        Args:
            action: The action to perform:
                - "add": Add new content to the report
                - "complete": Complete and optimize the entire report
            content: Content to add (required for add action)
        
        Returns:
            ToolResponse with action results.
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
            # Load content list
            content_list = self._load_content_list()
            
            if not content_list:
                return ToolResponse(
                    success=False,
                    message="No content to optimize. Add content first using the 'add' action."
                )
            
            # Read summary.md
            read_summary_result = await self._file_reader(file_path=self._summary_path)
            if not read_summary_result.success:
                return ToolResponse(
                    success=False,
                    message=f"Failed to read summary file: {read_summary_result.message}"
                )
            
            summaries_text = read_summary_result.extra.get("content", "") if read_summary_result.extra else ""
            
            logger.info(f"| 📊 Optimizing {len(content_list)} content items...")
            
            # Optimize each content item using LLM
            optimized_edits: List[Dict[str, Any]] = []
            
            for item in content_list:
                logger.info(f"| ✏️ Optimizing content #{item.id} (lines {item.start_line}-{item.end_line})")
                
                optimized_content = await self._optimize_content(
                    content=item.content,
                    item_id=item.id,
                    summary=item.summary,
                    summaries_context=summaries_text
                )
                
                optimized_edits.append({
                    "start_line": item.start_line,
                    "end_line": item.end_line,
                    "content": optimized_content
                })
            
            # Apply all optimizations to report.md
            # Sort edits by start_line in descending order to preserve line numbers
            sorted_edits = sorted(optimized_edits, key=lambda x: -x["start_line"])
            
            edit_result = await self._file_editor(
                file_path=self._report_path,
                edits=sorted_edits
            )
            
            if not edit_result.success:
                return ToolResponse(
                    success=False,
                    message=f"Failed to apply optimizations: {edit_result.message}"
                )
            
            logger.info(f"| ✅ Report optimization complete: {self._report_path}")
            
            return ToolResponse(
                success=True,
                message=f"📝 Report optimized successfully!\n\nPath: {self._report_path}\nTotal items optimized: {len(content_list)}\n\nThe report has been optimized for coherence, logic flow, and professional formatting.",
                extra={
                    "path": self._report_path,
                    "items_optimized": len(content_list)
                }
            )

        except Exception as e:
            logger.error(f"| ❌ Error completing report: {e}")
            import traceback
            return ToolResponse(
                success=False,
                message=f"Error completing report: {str(e)}\n{traceback.format_exc()}"
            )

    async def _optimize_content(
        self,
        content: str,
        item_id: int,
        summary: str,
        summaries_context: str
    ) -> str:
        """Optimize a single content item using LLM."""
        try:
            prompt = dedent(f"""Optimize and refine the following content to make it coherent, logically organized, and well-written.
            
            This is content item #{item_id}.
            
            Summary of this content:
            {summary}
            
            Context: All content summaries (for reference):
            {summaries_context}
            
            Original content:
            ```markdown
            {content}
            ```
            
            Please refine this content by:
            1. Improving logical flow and coherence with other content items
            2. Enhancing clarity and readability
            3. Fixing any grammar or style issues
            4. Ensuring consistent formatting
            5. Making transitions smooth and logical
            6. Making the writing more professional and polished
            
            IMPORTANT:
            - Preserve all factual information, data, and key findings
            - Keep the same markdown structure (headers, lists, etc.)
            - Do not add new information that wasn't in the original
            - Ensure the content flows well with adjacent content items
            - Return ONLY the refined content, no explanations
            """)
            
            messages = [
                SystemMessage(content="You are an expert editor specializing in ESG analysis reports. Your task is to refine and polish content while preserving all factual information and ensuring logical flow with other content sections. Return only the refined markdown content."),
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
                return response.message.strip()
                
        except Exception as e:
            logger.error(f"| ❌ Error optimizing content #{item_id}: {e}")
            # Return original content if optimization fails
            return content

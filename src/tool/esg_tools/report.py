"""Report Tool - A tool for managing and refining markdown reports."""

import os
from typing import Optional, Dict, Any
from pydantic import Field, ConfigDict

from src.registry import TOOL
from src.logger import logger
from src.utils import assemble_project_path
from src.tool.types import Tool, ToolResponse
from src.tool.types import Report

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
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    
    model_name: str = Field(
        default="openrouter/gemini-3-flash-preview",
        description="The model to use for code generation."
    )

    # Configuration parameters
    base_dir: str = Field(
        default=None,
        description="The base directory for saving reports."
    )
    
    # Internal state
    _report_path: Optional[str] = None
    _report: Optional[Report] = None

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
        
        # Set up file path
        self._report_path = os.path.join(self.base_dir, "report.md")
        
        # Initialize Report instance
        self._report = Report(
            title=title,
            model_name=self.model_name
        )

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

    async def _add_content(self, content: str) -> ToolResponse:
        """Add new content to the report using Report.add_item()."""
        try:
            # Add content to Report using add_item (which extracts ReportItem with citations and references)
            report_item = await self._report.add_item(content)
            
            item_id = len(self._report.items)
            logger.info(f"| ✅ Content added: ID={item_id}, Summary={report_item.content.summary[:100]}...")
            
            return ToolResponse(
                success=True,
                message=f"📝 Content added successfully!\n\nID: {item_id}\nSummary: {report_item.content.summary}",
                extra={
                    "id": item_id,
                    "summary": report_item.content.summary,
                    "reference_ids": report_item.content.reference_ids,
                    "references": [{"id": ref.id, "description": ref.description} for ref in report_item.references]
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
        """Complete and optimize the entire report using Report.complete()."""
        try:
            if not self._report.items:
                return ToolResponse(
                    success=False,
                    message="Report is empty. Add content first using the 'add' action."
                )
            
            logger.info(f"| 📊 Completing report with {len(self._report.items)} items...")
            
            # Use Report.complete() to generate final report with renumbered citations and references
            final_report_content = await self._report.complete(self._report_path)
            
            logger.info(f"| ✅ Report completion successful ({len(final_report_content)} chars)")
            
            return ToolResponse(
                success=True,
                message=f"📝 Report completed successfully!\n\nPath: {self._report_path}\n\nThe entire report has been generated with properly numbered citations and references.",
                extra={
                    "path": self._report_path,
                    "items_count": len(self._report.items),
                    "report_length": len(final_report_content)
                }
            )

        except Exception as e:
            logger.error(f"| ❌ Error completing report: {e}")
            import traceback
            return ToolResponse(
                success=False,
                message=f"Error completing report: {str(e)}\n{traceback.format_exc()}"
            )


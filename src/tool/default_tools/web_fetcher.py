"""Web fetcher tool for retrieving content from web pages."""

from pydantic import Field
from typing import Dict, Any

from src.utils import fetch_url
from src.logger import logger
from src.tool.types import Tool, ToolResponse, ToolExtra
from src.registry import TOOL

_WEB_FETCHER_DESCRIPTION = """Visit a webpage at a given URL and return its text content.
Use this tool to fetch and read content from web pages.
The tool will return the page title and markdown-formatted content.

Args:
- url (str): The URL of the webpage to fetch.

Example: {"name": "web_fetcher_tool", "args": {"url": "https://www.google.com"}}.
"""

@TOOL.register_module(force=True)
class WebFetcherTool(Tool):
    """A tool for fetching web content asynchronously."""

    name: str = "web_fetcher_tool"
    description: str = _WEB_FETCHER_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    require_grad: bool = Field(default=False, description="Whether the tool requires gradients")
    
    def __init__(self, require_grad: bool = False, **kwargs):
        """A tool for fetching web content asynchronously."""
        super().__init__(require_grad=require_grad, **kwargs)

    async def __call__(self, url: str, **kwargs) -> ToolResponse:
        """
        Fetch content from a given URL asynchronously.

        Args:
            url (str): The relative or absolute URL of the webpage to visit.
        """
        try:
            res = await fetch_url(url)
            if not res:
                logger.error(f"Failed to fetch content from {url}")
                return ToolResponse(
                    success=False, 
                    message=f"Failed to fetch content from {url}",
                    extra=ToolExtra(
                        data={
                            "url": url,
                            "status": "failed"
                        }
                    )
                )
            title = res.get("title", "")
            markdown = res.get("markdown", "")
            formatted = f"Title: {title}\nContent: {markdown}"
            return ToolResponse(
                success=True,
                message=formatted,
                extra=ToolExtra(
                    data={
                        "url": url,
                        "status": "success",
                        "content_length": len(formatted),
                        "title": title,
                        "markdown_length": len(markdown) if markdown else 0
                    }
                )
            )
        except Exception as e:
            logger.error(f"Error fetching content: {e}")
            return ToolResponse(
                success=False,
                message=f"Failed to fetch content: {e}",
                extra=ToolExtra(
                    data={
                        "url": url,
                        "status": "error",
                        "error_type": type(e).__name__,
                        "error_message": str(e)
                    }
                )
            )

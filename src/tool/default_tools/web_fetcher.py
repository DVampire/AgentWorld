"""Web fetcher tool for retrieving content from web pages."""

from src.utils import fetch_url
from src.logger import logger
from src.tool.types import Tool, ToolResponse

_WEB_FETCHER_DESCRIPTION = """Visit a webpage at a given URL and return its text content.
Use this tool to fetch and read content from web pages.
The tool will return the page title and markdown-formatted content.
"""

class WebFetcherTool(Tool):
    """A tool for fetching web content asynchronously."""

    name: str = "web_fetcher"
    description: str = _WEB_FETCHER_DESCRIPTION
    enabled: bool = True
    
    def __init__(self, **kwargs):
        """A tool for fetching web content asynchronously."""
        super().__init__(**kwargs)

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
                    extra={"url": url, "status": "failed"}
                )
            formatted = f"Title: {res.title}\nContent: {res.markdown}"
            return ToolResponse(
                success=True,
                message=formatted,
                extra={"url": url, "status": "success", "content_length": len(formatted)}
            )
        except Exception as e:
            logger.error(f"Error fetching content: {e}")
            return ToolResponse(
                success=False,
                message=f"Failed to fetch content: {e}",
                extra={"url": url, "status": "error", "error_type": type(e).__name__}
            )

"""Web fetcher tool for retrieving content from web pages."""

import asyncio
from pydantic import BaseModel, Field
from langchain.tools import BaseTool
from typing import Type, Dict, Any

from src.utils import fetch_url
from src.logger import logger
from src.tools.protocol.tool import ToolResponse
from src.tools.protocol import tcp

_WEB_FETCHER_DESCRIPTION = """Visit a webpage at a given URL and return its text content.
Use this tool to fetch and read content from web pages.
The tool will return the page title and markdown-formatted content.
"""

class WebFetcherToolArgs(BaseModel):
    url: str = Field(description="The relative or absolute url of the webpage to visit.")

@tcp.tool()
class WebFetcherTool(BaseTool):
    """A tool for fetching web content asynchronously."""
    
    name: str = "web_fetcher"
    description: str = _WEB_FETCHER_DESCRIPTION
    args_schema: Type[BaseModel] = WebFetcherToolArgs
    metadata: Dict[str, Any] = {"type": "Web Interaction"}
    
    def __init__(self, **kwargs):
        """A tool for fetching web content asynchronously."""
        super().__init__(**kwargs)
    
    async def _arun(self, url: str) -> ToolResponse:
        """Fetch content from a given URL asynchronously."""
        try:
            res = await fetch_url(url)
            res = f"Title: {res.title}\nContent: {res.markdown}"
            if not res:
                logger.error(f"Failed to fetch content from {url}")
                return ToolResponse(
                    content=f"Failed to fetch content from {url}",
                    extra={"url": url, "status": "failed"}
                )
            return ToolResponse(
                content=res,
                extra={"url": url, "status": "success", "content_length": len(res)}
            )
        except Exception as e:
            logger.error(f"Error fetching content: {e}")
            return ToolResponse(
                content=f"Failed to fetch content: {e}",
                extra={"url": url, "status": "error", "error_type": type(e).__name__}
            )
    
    def _run(self, url: str) -> ToolResponse:
        """Fetch content from a given URL synchronously (fallback)."""
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(url))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error in synchronous execution: {e}")
            return ToolResponse(
                content=f"Error in synchronous execution: {e}",
                extra={"status": "error", "error_type": type(e).__name__}
            )


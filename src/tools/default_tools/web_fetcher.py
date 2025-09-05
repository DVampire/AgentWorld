from typing import Optional, Dict, Any, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
import asyncio

from src.utils import fetch_url
from src.logger import logger
from src.tools.base import ToolResponse

_WEB_FETCHER_DESCRIPTION = """Visit a webpage at a given URL and return its text. """

class WebFetcherToolArgs(BaseModel):
    url: str = Field(description="The relative or absolute url of the webpage to visit.")


class WebFetcherTool(BaseTool):
    """A tool for fetching web content asynchronously."""
    
    name: str = "web_fetcher_tool"
    description: str = _WEB_FETCHER_DESCRIPTION
    args_schema: Type[WebFetcherToolArgs] = WebFetcherToolArgs
    
    def __init__(self, **kwargs):
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
    
    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "type": "web_fetcher"
        }


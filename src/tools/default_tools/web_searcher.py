from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tenacity import retry, stop_after_attempt, wait_exponential
from langchain.tools import BaseTool
import time
import asyncio

from src.tools.default_tools.web_fetcher import WebFetcherTool
from src.tools.default_tools.search import (
    GoogleSearchEngine,
    FirecrawlSearchEngine,
    WebSearchEngine,
    SearchItem
)
from src.logger import logger
from src.tools.base import ToolResponse

_WEB_SEARCHER_DESCRIPTION = """Search the web for real-time information about any topic.
This tool returns comprehensive search results with relevant information, URLs, titles, and descriptions.
If the primary search engine fails, it automatically falls back to alternative engines."""

class SearchResult(BaseModel):
    """Represents a single search result returned by a search engine."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    position: int = Field(description="Position in search results")
    url: str = Field(description="URL of the search result")
    title: str = Field(default="", description="Title of the search result")
    description: str = Field(
        default="", description="Description or snippet of the search result"
    )
    source: str = Field(description="The search engine that provided this result")
    raw_content: Optional[str] = Field(
        default=None, description="Raw content from the search result page if available"
    )

    def __str__(self) -> str:
        """String representation of a search result."""
        return f"{self.title} ({self.url})"


class SearchMetadata(BaseModel):
    """Metadata about the search operation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    total_results: int = Field(description="Total number of results found")
    language: str = Field(description="Language code used for the search")
    country: str = Field(description="Country code used for the search")

class SearchResponse(BaseModel):
    """Structured response from the web search tool."""
    query: str = Field(description="The search query that was executed")
    results: List[SearchResult] = Field(default_factory=list, description="List of search results")
    metadata: Optional[SearchMetadata] = Field(default=None, description="Metadata about the search")
    output: str = Field(default="", description="Formatted output string")

    @model_validator(mode="after")
    def populate_output(self) -> "SearchResponse":
        """Populate output field based on search results."""
        result_text = [f"Search results for '{self.query}':"]

        for i, result in enumerate(self.results, 1):
            # Add title with position number
            title = result.title.strip() or "No title"
            result_text.append(f"\n{i}. {title}")

            # Add URL with proper indentation
            result_text.append(f"   URL: {result.url}")

            # Add description if available
            if result.description.strip():
                result_text.append(f"   Description: {result.description}")

            # Add content preview if available
            if result.raw_content:
                content_preview = result.raw_content.replace("\n", " ").strip()
                result_text.append(f"   Content: {content_preview}")

        # Add metadata at the bottom if available
        if self.metadata:
            result_text.extend(
                [
                    f"\nMetadata:",
                    f"- Total results: {self.metadata.total_results}",
                    f"- Language: {self.metadata.language}",
                    f"- Country: {self.metadata.country}",
                ]
            )

        self.output = "\n".join(result_text)
        return self

class WebSearcherToolArgs(BaseModel):
    query: str = Field(description="(required) The search query to submit to the search engine.")
    filter_year: Optional[int] = Field(
        default=None,
        description="(optional) Filter results by year (e.g., 2025)."
    )

class WebSearcherTool(BaseTool):
    """Search the web for information using various search engines."""

    name: str = "web_searcher_tool"
    description: str = _WEB_SEARCHER_DESCRIPTION
    args_schema: Type[WebSearcherToolArgs] = WebSearcherToolArgs
    
    # Configure parameters as class attributes
    engine: str = Field(
        default="firecrawl", 
        description="The search engine to use."
    )
    fallback_engines: List[str] = Field(
        default=["duckduckgo", "baidu", "bing"],
        description="The fallback search engines to use."
    )
    max_length: int = Field(
        default=4096,
        description="The maximum length of the content to fetch."
    )
    retry_delay: int = Field(
        default=10,
        description="The retry delay in seconds."
    )
    max_retries: int = Field(
        default=3,
        description="The maximum number of retries."
    )
    lang: str = Field(
        default="en",
        description="The language to use."
    )
    country: str = Field(
        default="us", 
        description="The country to use."
    )
    num_results: int = Field(
        default=5,
        description="The number of results to return."
    )
    fetch_content: bool = Field(
        default=False,
        description="Whether to fetch content from the search results."
    )
    search_engines: Dict[str, WebSearchEngine] = Field(default_factory=dict, description="The search engines to use.")
    content_fetcher: WebFetcherTool = Field(default_factory=WebFetcherTool, description="The content fetcher to use.")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Initialize search engines and content fetcher
        self.search_engines: Dict[str, WebSearchEngine] = {
            "firecrawl": FirecrawlSearchEngine(),
        }
        self.content_fetcher = WebFetcherTool()

    async def _arun(self, query: str, filter_year: Optional[int] = None) -> ToolResponse:
        """
        Execute a Web search and return detailed search results.

        Args:
            query: The search query to submit to the search engine
            num_results: The number of search results to return (default: 5)
            lang: Language code for search results (default from config)
            country: Country code for search results (default from config)
            fetch_content: Whether to fetch content from result pages (default: False)

        Returns:
            A structured response containing search results and metadata
        """
        search_params = {"lang": self.lang, "country": self.country}

        if filter_year is not None:
            search_params["filter_year"] = filter_year

        # Try searching with retries when all engines fail
        for retry_count in range(self.max_retries + 1):
            results = await self._try_all_engines(query, self.num_results, search_params)
            if results:
                # Fetch content if requested
                if self.fetch_content:
                    results = await self._fetch_content_for_results(results)

                # Return a successful structured response
                response = SearchResponse(
                    query=query,
                    results=results,
                    metadata=SearchMetadata(
                        total_results=len(results),
                        language=self.lang,
                        country=self.country,
                    ),
                )
                return ToolResponse(
                    content=response.output,
                    extra={
                        "query": query,
                        "status": "success",
                        "results": results,
                        "total_results": len(results),
                        "language": self.lang,
                        "country": self.country,
                        "search_engines_used": [r.source for r in results]
                    }
                )

            if retry_count < self.max_retries:
                # All engines failed, wait and retry
                res = f"All search engines failed. Waiting {self.retry_delay} seconds before retry {retry_count + 1}/{self.max_retries}..."
                logger.warning(res)
                time.sleep(self.retry_delay)
            else:
                res = f"All search engines failed after {self.max_retries} retries. Giving up."
                logger.error(res)
                # Return an error response
                return ToolResponse(
                    content=f"Error: All search engines failed to return results after multiple retries.",
                    extra={
                        "query": query,
                        "status": "failed",
                        "results": [],
                        "total_results": 0,
                        "language": self.lang,
                        "country": self.country,
                        "search_engines_used": [],
                    }
                )

    async def _try_all_engines(
        self, query: str, num_results: int, search_params: Dict[str, Any]
    ) -> List[SearchResult]:
        """Try all search engines in the configured order."""
        engine_order = self._get_engine_order()
        failed_engines = []

        for engine_name in engine_order:
            engine = self.search_engines[engine_name]
            logger.info(f"🔎 Attempting search with {engine_name.capitalize()}...")
            search_items = await self._perform_search_with_engine(
                engine, query, num_results, search_params
            )

            if not search_items:
                continue

            if failed_engines:
                logger.info(
                    f"Search successful with {engine_name.capitalize()} after trying: {', '.join(failed_engines)}"
                )

            # Transform search items into structured results
            return [
                SearchResult(
                    position=i + 1,
                    url=item.url,
                    title=item.title
                    or f"Result {i+1}",  # Ensure we always have a title
                    description=item.description or "",
                    source=engine_name,
                )
                for i, item in enumerate(search_items)
            ]

        if failed_engines:
            logger.error(f"All search engines failed: {', '.join(failed_engines)}")
        return []

    async def _fetch_content_for_results(
            self, results: List[SearchResult]
    ) -> List[SearchResult]:
        """Fetch and add web content to search results."""
        if not results:
            return []

        # Create tasks for each result
        # fetched_results = [await self._fetch_single_result_content(result) for result in results]
        fetched_results = await asyncio.gather(
            *[self._fetch_single_result_content(result) for result in results]
        )

        # Explicit validation of return type
        return [
            (
                result
                if isinstance(result, SearchResult)
                else SearchResult(**result.dict())
            )
            for result in fetched_results
        ]

    async def _fetch_single_result_content(self, result: SearchResult) -> SearchResult:
        """Fetch content for a single search result."""
        if result.url:
            res = await self.content_fetcher._arun(result.url)
            if isinstance(res, str):
                content = res
                if content and len(content) > self.max_length:
                    content = content[: self.max_length] + "..."
                result.raw_content = content
        return result

    def _get_engine_order(self) -> List[str]:
        """Determines the order in which to try search engines."""
        preferred = (
            self.engine if self.engine else "firecrawl"
        )
        fallbacks = [engine for engine in self.fallback_engines]

        # Start with preferred engine, then fallbacks, then remaining engines
        engine_order = [preferred] if preferred in self.search_engines else []
        engine_order.extend(
            [
                fb
                for fb in fallbacks
                if fb in self.search_engines and fb not in engine_order
            ]
        )
        engine_order.extend([e for e in self.search_engines if e not in engine_order])

        return engine_order

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def _perform_search_with_engine(
        self,
        engine: WebSearchEngine,
        query: str,
        num_results: int,
        search_params: Dict[str, Any],
    ) -> List[SearchItem]:
        """Execute search with the given engine and parameters."""

        results = [result
            for result in await engine.perform_search(
                query,
                num_results=num_results,
                lang=search_params.get("lang"),
                country=search_params.get("country"),
                filter_year=search_params.get("filter_year"),
            )
        ]
        return results
    
    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "engine": self.engine,
            "max_length": self.max_length,
            "retry_delay": self.retry_delay,
            "max_retries": self.max_retries,
            "lang": self.lang,
            "country": self.country,
            "num_results": self.num_results,
            "fetch_content": self.fetch_content,
            "type": "web_searcher"
        }
    
    def _run(self, query: str, filter_year: Optional[int] = None) -> str:
        """Execute a Web search synchronously (fallback)."""
        try:
            # Run the async version in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(query, filter_year))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error in synchronous execution: {e}")
            return f"Error in synchronous execution: {e}"
    
    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "type": "web_searcher"
        }

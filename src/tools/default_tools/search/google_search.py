from __future__ import annotations
from typing import Any, Optional, Dict, List, Type
import asyncio
import json

import requests
from bs4 import BeautifulSoup
from urllib.parse import unquote
from time import sleep
from pydantic import Field
from googlesearch.user_agents import get_useragent
from langchain_core.tools import BaseTool
from langchain_core.utils import secret_from_env

from src.tools.default_tools.search.base import SearchItem, SearchToolArgs
from src.tools.base import ToolResponse


class GoogleSearch(BaseTool):
    """Tool that queries the Google search engine.

    Example usages:
    .. code-block:: python
        # basic usage
        tool = GoogleSearch()

    .. code-block:: python
        # with custom search kwargs
        tool = GoogleSearch.from_search_kwargs({"num": 5})
    """

    name: str = "google_search"
    description: str = (
        "a search engine. "
        "useful for when you need to answer questions about current events."
        " input should be a search query."
    )
    args_schema: Type[SearchToolArgs] = SearchToolArgs
    search_kwargs: Dict[str, Any] = Field(default_factory=dict)
    base_url: Optional[str] = Field(default_factory=secret_from_env(["SKYWORK_GOOGLE_SEARCH_BASE_URL"]))
    api_key: Optional[str] = Field(default=None, description="The API key to use for the Google search engine.")

    def __init__(self, **data):
        """Initialize the GoogleSearch tool."""
        super().__init__(**data)

    @classmethod
    def from_search_kwargs(cls, search_kwargs: dict, **kwargs: Any) -> GoogleSearch:
        """Create a tool from search kwargs.

        Args:
            search_kwargs: Any additional kwargs to pass to the search function.
            **kwargs: Any additional kwargs to pass to the tool.

        Returns:
            A tool.
        """
        return cls(search_kwargs=search_kwargs, **kwargs)

    def _req(self, term, results, tbs, lang, start, proxies, timeout, safe, ssl_verify, region):
        """Make a request to Google search."""
        params = {
            "q": term,
            "num": results + 2,  # Prevents multiple requests
            "hl": lang,
            "start": start,
            "safe": safe,
            "gl": region,
        }
        if tbs is not None:
            params["tbs"] = tbs
            
        resp = requests.get(
            url=self.base_url.get_secret_value(),
            headers={
                "User-Agent": get_useragent(),
                "Accept": "*/*"
            },
            params=params,
            proxies=proxies,
            timeout=timeout,
            verify=ssl_verify,
            cookies={
                'CONSENT': 'PENDING+987',  # Bypasses the consent page
                'SOCS': 'CAESHAgBEhIaAB',
            }
        )
        resp.raise_for_status()
        return resp

    def _google_search_scraping(self, term, num_results=10, tbs=None, lang="en", proxy=None, 
                               advanced=True, sleep_interval=0, timeout=5, safe="active",
                               ssl_verify=None, region=None, start_num=0, unique=False):
        """Search the Google search engine using web scraping."""
        # Proxy setup
        proxies = {"https": proxy, "http": proxy} if proxy and (proxy.startswith("https") or proxy.startswith("http")) else None

        start = start_num
        fetched_results = 0
        fetched_links = set()
        results = []

        while fetched_results < num_results:
            # Send request
            resp = self._req(term, num_results - start, tbs, lang, start, proxies, 
                           timeout, safe, ssl_verify, region)
            
            # Parse
            soup = BeautifulSoup(resp.text, "html.parser")
            result_block = soup.find_all("div", class_="ezO2md")
            new_results = 0

            for result in result_block:
                # Find the link tag within the result block
                link_tag = result.find("a", href=True)
                # Find the title tag within the link tag
                title_tag = link_tag.find("span", class_="CVA68e") if link_tag else None
                # Find the description tag within the result block
                description_tag = result.find("span", class_="FrIlee")

                # Check if all necessary tags are found
                if link_tag and title_tag and description_tag:
                    # Extract and decode the link URL
                    link = unquote(link_tag["href"].split("&")[0].replace("/url?q=", ""))
                    
                    # Check if the link has already been fetched and if unique results are required
                    if link in fetched_links and unique:
                        continue
                    
                    # Add the link to the set of fetched links
                    fetched_links.add(link)
                    # Extract the title text
                    title = title_tag.text if title_tag else ""
                    # Extract the description text
                    description = description_tag.text if description_tag else ""
                    
                    # Create SearchItem
                    search_item = SearchItem(
                        title=title,
                        url=link,
                        description=description,
                    )
                    results.append(search_item)
                    
                    # Increment counters
                    fetched_results += 1
                    new_results += 1

                    if fetched_results >= num_results:
                        break

            if new_results == 0:
                break

            start += 10
            sleep(sleep_interval)

        return results

    def _google_search_api(self, params):
        """Search using Google Search API."""
        base_url = self.base_url.get_secret_value()
        query = params.get("q", "")
        filter_year = params.get("filter_year", None)
        
        response = requests.get(base_url, params=params)
        
        if response.status_code == 200:
            items = response.json()
        else:
            raise ValueError(response.json())

        if "organic" not in items.keys():
            if filter_year is not None:
                raise Exception(
                    f"No results found for query: '{query}' with filtering on year={filter_year}. Use a less restrictive query or do not filter on year."
                )
            else:
                raise Exception(f"No results found for query: '{query}'. Use a less restrictive query.")

        results = []
        if "organic" in items:
            for idx, page in enumerate(items["organic"]):
                title = page.get("title", f"Google Result {idx + 1}")
                url = page.get("link", "")
                position = page.get("position", idx + 1)
                description = page.get("snippet", None)
                date = page.get("date", None)
                source = page.get("source", None)

                results.append(
                    SearchItem(
                        title=title,
                        url=url,
                        description=description,
                    )
                )
        return results

    async def _search_google(self, query: str, num_results: int = 10, filter_year: Optional[int] = None) -> List[SearchItem]:
        """Perform a Google search using the provided parameters."""
        params = {
            "q": query,
            "num": num_results,
        }
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

        # Use API if available, otherwise use web scraping
        if self.api_key:
            return self._google_search_api(params)
        else:
            return self._google_search_scraping(
                term=query,
                num_results=num_results,
                tbs=params.get("tbs", None),
                lang="en",
                proxy=None,
                advanced=True,
                sleep_interval=0,
                timeout=5,
            )
    
    async def _arun(
        self,
        query: str,
        num_results: Optional[int] = 10,
        country: Optional[str] = "us",
        lang: Optional[str] = "en",
        filter_year: Optional[int] = None,
    ) -> ToolResponse:
        """Use the tool asynchronously."""
        
        try: 
            # Perform search
            search_items = await self._search_google(query, num_results=num_results, filter_year=filter_year)
            
            # Format results as JSON string
            results_json = json.dumps([{
                "title": item.title,
                "url": item.url,
                "description": item.description or ""
            } for item in search_items], ensure_ascii=False, indent=2)
            
            return ToolResponse(content=results_json, extra={"data": search_items})
            
        except Exception as e:
            return ToolResponse(content=f"Error in asynchronous execution: {str(e)}")

    def _run(self, query: str, num_results: Optional[int] = 10, country: Optional[str] = "us", lang: Optional[str] = "en", filter_year: Optional[int] = None) -> ToolResponse:
        """Use the tool synchronously (fallback)."""
        try:
            # Run the async version in a new event loop
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(query, num_results, country, lang, filter_year))
            finally:
                loop.close()
        except Exception as e:
            return ToolResponse(content=f"Error in synchronous execution: {str(e)}")
    
    def get_tool_config(self) -> Dict[str, Any]:
        """Get tool configuration."""
        return {
            "name": self.name,
            "description": self.description,
            "args_schema": self.args_schema,
            "type": "google_search"
        }
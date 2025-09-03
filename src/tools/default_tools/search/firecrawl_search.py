import os
import time

from dotenv import load_dotenv
load_dotenv(verbose=True)

from typing import List
from firecrawl import AsyncFirecrawlApp
from firecrawl.types import ScrapeOptions
import asyncio

from src.tools.default_tools.search.base import WebSearchEngine, SearchItem

async def search(params):
    """
    Perform a Google search using the provided parameters.
    Returns a list of SearchItem objects.
    """
    app = AsyncFirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY", None))
    search_kwargs = {
        "query": params["q"],
        "limit": params.get("num", 10),
    }

    # Only include tbs if it's a valid value
    tbs_value = params.get("tbs")
    if tbs_value and tbs_value.strip():
        search_kwargs["tbs"] = tbs_value
    
    response = await app.search(**search_kwargs)

    results = []
    for item in response.web:
        title = item.title
        url = item.url
        description = item.description
        results.append(SearchItem(title=title,
                                  url=url,
                                  description=description)
                       )

    return results

class FirecrawlSearchEngine(WebSearchEngine):
    async def perform_search(
        self,
        query: str,
        num_results: int = 10,
        filter_year: int = None,
        *args, **kwargs
    ) -> List[SearchItem]:
        """
        Google search engine.

        Returns results formatted according to SearchItem model.
        """
        params = {
            "q": query,
            "num": num_results,
        }
        if filter_year is not None:
            params["tbs"] = f"cdr:1,cd_min:01/01/{filter_year},cd_max:12/31/{filter_year}"

        results = await search(params)

        return results


if __name__ == '__main__':
    # Example usage
    start_time = time.time()
    search_engine = FirecrawlSearchEngine()
    query = "OpenAI GPT-4"
    results = asyncio.run(search_engine.perform_search(query, num_results=5))

    for item in results:
        print(f"Title: {item.title}\nURL: {item.url}\nDescription: {item.description}\n")

    end_time = time.time()

    print(end_time - start_time, "seconds elapsed for search")
import os
import asyncio
from typing import Optional
from dotenv import load_dotenv
load_dotenv(verbose=True)

from markitdown._base_converter import DocumentConverterResult
from crawl4ai import AsyncWebCrawler
from firecrawl import AsyncFirecrawlApp

# Default timeout for web fetching (in seconds)
DEFAULT_FETCH_TIMEOUT = 15  # 15 seconds per fetch attempt

async def firecrawl_fetch_url(url: str, timeout: int = DEFAULT_FETCH_TIMEOUT):
    """Fetch content using Firecrawl with timeout."""
    try:
        app = AsyncFirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY", None))

        # Wrap the scrape call with timeout
        response = await asyncio.wait_for(
            app.scrape(url),
            timeout=timeout
        )

        result = response.markdown
        return result
    except asyncio.TimeoutError:
        return None
    except Exception as e:
        return None

async def fetch_crawl4ai_url(url: str, timeout: int = DEFAULT_FETCH_TIMEOUT):
    """Fetch content from a given URL using the crawl4ai library with timeout."""
    try:
        async with AsyncWebCrawler() as crawler:
            # Wrap the arun call with timeout
            response = await asyncio.wait_for(
                crawler.arun(url=url),
                timeout=timeout
            )

            if response:
                result = response.markdown
                return result
            else:
                return None
    except asyncio.TimeoutError:
        return None
    except Exception as e:
        return None

async def fetch_url(url: str, timeout: int = DEFAULT_FETCH_TIMEOUT) -> Optional[DocumentConverterResult]:
    """Fetch content from a URL using Firecrawl and Crawl4AI with timeout.
    
    Args:
        url: The URL to fetch
        timeout: Timeout in seconds for each fetch attempt (default: 15)
    
    Returns:
        DocumentConverterResult if successful, None otherwise
    """
    try:
        # Try Firecrawl first with timeout
        firecrawl_result = await firecrawl_fetch_url(url, timeout=timeout)

        if firecrawl_result:
            return DocumentConverterResult(
                markdown=firecrawl_result,
                title=f"Fetched content from {url}",
            )

        # Fallback to Crawl4AI with timeout
        crawl4ai_result = await fetch_crawl4ai_url(url, timeout=timeout)
        if crawl4ai_result:
            return DocumentConverterResult(
                markdown=crawl4ai_result,
                title=f"Fetched content from {url}",
            )

    except Exception as e:
        return None
    
    return None

if __name__ == '__main__':
    import asyncio
    url = "https://www.google.com/"
    result = asyncio.run(firecrawl_fetch_url(url))
    print(result)
import os
import asyncio
from typing import Optional, Dict, Any
from urllib.parse import quote
from dotenv import load_dotenv
load_dotenv(verbose=True)

import aiohttp
from crawl4ai import AsyncWebCrawler

# Default timeout for web fetching (in seconds)
DEFAULT_FETCH_TIMEOUT = 15  # 15 seconds per fetch attempt

async def jina_fetch_url(url: str, timeout: int = DEFAULT_FETCH_TIMEOUT):
    """Fetch content using Jina AI Reader (r.jina.ai) with timeout."""
    try:
        safe_chars = ":/?#[]@!$&'()*+,;="
        reader_url = f"https://r.jina.ai/{quote(url, safe=safe_chars)}"
        headers = {"Accept": "text/plain", "X-Return-Format": "markdown"}
        api_key = os.getenv("JINA_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        async with aiohttp.ClientSession() as session:
            response = await asyncio.wait_for(
                session.get(reader_url, headers=headers),
                timeout=timeout,
            )
            async with response as resp:
                resp.raise_for_status()
                return await resp.text()
    except asyncio.TimeoutError:
        return None
    except Exception:
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

async def fetch_url(url: str, timeout: int = DEFAULT_FETCH_TIMEOUT) -> Dict[str, Any]:
    """Fetch content from a URL using Jina Reader and Crawl4AI with timeout.

    Args:
        url: The URL to fetch
        timeout: Timeout in seconds for each fetch attempt (default: 15)

    Returns:
        DocumentConverterResult if successful, None otherwise
    """
    try:
        # Try Jina Reader first with timeout
        jina_result = await jina_fetch_url(url, timeout=timeout)
        if jina_result:
            return {
                "markdown": jina_result,
                "title": f"Fetched content from {url} using Jina Reader",
            }

        # Fallback to Crawl4AI with timeout
        crawl4ai_result = await fetch_crawl4ai_url(url, timeout=timeout)
        if crawl4ai_result:
            return {
                "markdown": crawl4ai_result,
                "title": f"Fetched content from {url} using Crawl4AI",
            }
    except Exception as e:
        return None

    return None

if __name__ == '__main__':
    import asyncio
    url = "https://www.google.com/"
    result = asyncio.run(fetch_url(url))
    print(result)
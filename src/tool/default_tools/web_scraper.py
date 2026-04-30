"""Web scraper tool — Jina Reader fetch + LLM extraction.

Two-stage pipeline mirroring MiroFlow's jina_scrape.py:
1. Fetch page content via Jina Reader (r.jina.ai), fallback to httpx direct.
2. Extract relevant information using a small LLM.

The extraction step is optional and controlled by ``use_extraction``.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, Dict, Optional

import httpx
from pydantic import ConfigDict, Field

from src.logger import logger
from src.model import model_manager
from src.message.types import HumanMessage
from src.registry import TOOL
from src.tool.types import Tool, ToolExtra, ToolResponse

JINA_API_KEY = os.getenv("JINA_API_KEY", "")
JINA_BASE_URL = os.getenv("JINA_BASE_URL", "https://r.jina.ai")

# Max characters to keep from scraped content (aligned with MiroFlow: 102400*4)
DEFAULT_MAX_CHARS = 102400 * 4

# ---------------------------------------------------------------------------
# Extraction prompt — mirrors MiroFlow's EXTRACT_INFO_PROMPT
# ---------------------------------------------------------------------------

EXTRACT_INFO_PROMPT = """You are given a piece of content and the requirement of information to extract. Your task is to extract the information specifically requested. Be precise and focus exclusively on the requested information.

INFORMATION TO EXTRACT:
{}

INSTRUCTIONS:
1. Extract the information relevant to the focus above.
2. If the exact information is not found, extract the most closely related details.
3. Be specific and include exact details when available.
4. Clearly organize the extracted information for easy understanding.
5. Do not include general summaries or unrelated content.

CONTENT TO ANALYZE:
{}

EXTRACTED INFORMATION:"""


def _is_huggingface_dataset_or_space_url(url: str) -> bool:
    if not url:
        return False
    return "huggingface.co/datasets" in url or "huggingface.co/spaces" in url


# ---------------------------------------------------------------------------
# Stage 1 — Jina Reader fetch (with httpx fallback)
# ---------------------------------------------------------------------------

async def _fetch_with_jina(
    url: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    custom_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Fetch page content via Jina Reader API.

    Returns dict with keys: success, content, error, char_count, all_content_displayed.
    """
    if not JINA_API_KEY:
        return {"success": False, "content": "", "error": "JINA_API_KEY not set",
                "char_count": 0, "all_content_displayed": False}

    # Avoid duplicate Jina URL prefix
    target = url
    if target.startswith("https://r.jina.ai/") and target.count("http") >= 2:
        target = target[len("https://r.jina.ai/"):]

    jina_url = f"{JINA_BASE_URL}/{target}"
    headers = {"Authorization": f"Bearer {JINA_API_KEY}"}
    if custom_headers:
        headers.update(custom_headers)

    retry_delays = [1, 2, 4, 8]
    last_error = None

    for attempt, delay in enumerate(retry_delays, 1):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    jina_url,
                    headers=headers,
                    timeout=httpx.Timeout(None, connect=20, read=60),
                    follow_redirects=True,
                )
            response.raise_for_status()

            content = response.text
            if not content:
                return {"success": False, "content": "", "error": "Empty response from Jina",
                        "char_count": 0, "all_content_displayed": False}

            # Check for Jina-specific errors
            try:
                content_dict = json.loads(content)
            except json.JSONDecodeError:
                content_dict = None
            if isinstance(content_dict, dict) and content_dict.get("name") == "InsufficientBalanceError":
                return {"success": False, "content": "", "error": "Jina insufficient balance",
                        "char_count": 0, "all_content_displayed": False}

            total_chars = len(content)
            displayed = content[:max_chars]
            return {
                "success": True,
                "content": displayed,
                "error": "",
                "char_count": total_chars,
                "all_content_displayed": total_chars <= max_chars,
            }

        except (httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadTimeout) as e:
            last_error = e
            if attempt < len(retry_delays):
                logger.info(f"| Jina fetch: {type(e).__name__}, retry in {delay}s (attempt {attempt})")
                await asyncio.sleep(delay)
                continue
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            should_retry = status >= 500 or status in [408, 409, 425, 429]
            last_error = e
            if should_retry and attempt < len(retry_delays):
                logger.info(f"| Jina fetch: HTTP {status}, retry in {delay}s")
                await asyncio.sleep(delay)
                continue
            break
        except Exception as e:
            last_error = e
            break

    return {"success": False, "content": "", "error": f"Jina fetch failed: {last_error}",
            "char_count": 0, "all_content_displayed": False}


async def _fetch_with_httpx(
    url: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    custom_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Fallback: fetch page content directly via httpx."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        )
    }
    if custom_headers:
        headers.update(custom_headers)

    retry_delays = [1, 2, 4]
    last_error = None

    for attempt, delay in enumerate(retry_delays, 1):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url,
                    headers=headers,
                    timeout=httpx.Timeout(None, connect=20, read=60),
                    follow_redirects=True,
                )
            response.raise_for_status()

            content = response.text
            if not content:
                return {"success": False, "content": "", "error": "Empty response",
                        "char_count": 0, "all_content_displayed": False}

            total_chars = len(content)
            displayed = content[:max_chars]
            return {
                "success": True,
                "content": displayed,
                "error": "",
                "char_count": total_chars,
                "all_content_displayed": total_chars <= max_chars,
            }

        except (httpx.ConnectTimeout, httpx.ConnectError, httpx.ReadTimeout) as e:
            last_error = e
            if attempt < len(retry_delays):
                await asyncio.sleep(delay)
                continue
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            should_retry = status >= 500 or status in [408, 409, 425, 429]
            last_error = e
            if should_retry and attempt < len(retry_delays):
                await asyncio.sleep(delay)
                continue
            break
        except Exception as e:
            last_error = e
            break

    return {"success": False, "content": "", "error": f"httpx fetch failed: {last_error}",
            "char_count": 0, "all_content_displayed": False}


# ---------------------------------------------------------------------------
# Stage 2 — LLM extraction
# ---------------------------------------------------------------------------

async def _extract_info_with_llm(
    content: str,
    info_to_extract: str,
    model_name: str,
) -> Dict[str, Any]:
    """Use a small LLM to extract relevant information from content.

    Uses AgentOS model_manager (not raw httpx like MiroFlow) for consistency.
    """
    if not content or not content.strip():
        return {"success": False, "extracted_info": "", "error": "Content is empty"}

    prompt_text = EXTRACT_INFO_PROMPT.format(info_to_extract, content)

    try:
        response = await model_manager(
            model=model_name,
            messages=[HumanMessage(content=prompt_text)],
        )
        if response and response.message and response.message.strip():
            return {
                "success": True,
                "extracted_info": response.message.strip(),
                "error": "",
            }
        return {"success": False, "extracted_info": "", "error": "LLM returned empty response"}

    except Exception as e:
        # If context length exceeded, try with truncated content
        error_str = str(e)
        if "context length" in error_str.lower() or "too long" in error_str.lower():
            logger.warning(f"| Content too long for LLM, truncating and retrying")
            truncated = content[: len(content) // 2] + "\n[...truncated]"
            prompt_text = EXTRACT_INFO_PROMPT.format(info_to_extract, truncated)
            try:
                response = await model_manager(
                    model=model_name,
                    messages=[HumanMessage(content=prompt_text)],
                )
                if response and response.message and response.message.strip():
                    return {
                        "success": True,
                        "extracted_info": response.message.strip(),
                        "error": "",
                    }
            except Exception as e2:
                logger.error(f"| LLM extraction failed after truncation: {e2}")

        logger.error(f"| LLM extraction failed: {e}")
        return {"success": False, "extracted_info": "", "error": f"LLM extraction failed: {e}"}


# ---------------------------------------------------------------------------
# WebScraperTool
# ---------------------------------------------------------------------------

_WEB_SCRAPER_DESCRIPTION = """Scrape content from a URL and optionally extract relevant information using an LLM.

Two-stage pipeline:
1. Fetch page content via Jina Reader (with httpx fallback)
2. (Optional) Extract specific information using a small LLM

Args:
- url (str): The URL to scrape.
- info_to_extract (str): What information to extract from the page (usually the research question/query).

Example: {"name": "web_scraper_tool", "args": {"url": "https://example.com", "info_to_extract": "What is the population of France?"}}.
"""


@TOOL.register_module(force=True)
class WebScraperTool(Tool):
    """Scrape web content and extract information — Jina Reader + LLM."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "web_scraper_tool"
    description: str = _WEB_SCRAPER_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    require_grad: bool = Field(default=False)

    extraction_model_name: str = Field(
        default="openrouter/gemini-3.1-flash-lite-preview",
        description="Model used for content extraction",
    )
    use_extraction: bool = Field(
        default=True,
        description="Whether to use LLM extraction after fetching",
    )
    max_chars: int = Field(
        default=DEFAULT_MAX_CHARS,
        description="Maximum characters to keep from scraped content",
    )

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_extraction: bool = True,
        max_chars: int = DEFAULT_MAX_CHARS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if model_name:
            self.extraction_model_name = model_name
        self.use_extraction = use_extraction
        self.max_chars = max_chars

    async def __call__(
        self,
        url: str,
        info_to_extract: str = "",
        custom_headers: Optional[Dict[str, str]] = None,
        **kwargs,
    ) -> ToolResponse:
        """Scrape a URL and optionally extract relevant information.

        Args:
            url: The URL to scrape.
            info_to_extract: What information to look for (used by LLM extraction).
            custom_headers: Additional HTTP headers.
        """
        # Block HuggingFace dataset URLs
        if _is_huggingface_dataset_or_space_url(url):
            return ToolResponse(
                success=False,
                message="Cannot scrape HuggingFace dataset/space URLs (data leakage prevention).",
            )

        # Stage 1: Fetch content
        scrape_result = await _fetch_with_jina(url, self.max_chars, custom_headers)

        if not scrape_result["success"]:
            logger.warning(f"| Jina fetch failed for {url}: {scrape_result['error']}, trying httpx fallback")
            scrape_result = await _fetch_with_httpx(url, self.max_chars, custom_headers)

        if not scrape_result["success"]:
            logger.error(f"| All fetch methods failed for {url}")
            return ToolResponse(
                success=False,
                message=f"Failed to fetch content from {url}: {scrape_result['error']}",
                extra=ToolExtra(data={"url": url, "char_count": 0}),
            )

        content = scrape_result["content"]
        char_count = scrape_result["char_count"]

        # Stage 2: LLM extraction (if enabled and info_to_extract provided)
        if self.use_extraction and info_to_extract:
            extract_result = await _extract_info_with_llm(
                content=content,
                info_to_extract=info_to_extract,
                model_name=self.extraction_model_name,
            )

            if extract_result["success"]:
                return ToolResponse(
                    success=True,
                    message=extract_result["extracted_info"],
                    extra=ToolExtra(
                        data={
                            "url": url,
                            "char_count": char_count,
                            "all_content_displayed": scrape_result["all_content_displayed"],
                            "extraction_used": True,
                            "extraction_model": self.extraction_model_name,
                        }
                    ),
                )
            else:
                # Extraction failed — fall through to return raw content
                logger.warning(
                    f"| LLM extraction failed for {url}: {extract_result['error']}, "
                    f"returning raw content instead"
                )

        # Return raw content (extraction disabled, not requested, or failed)
        return ToolResponse(
            success=True,
            message=content,
            extra=ToolExtra(
                data={
                    "url": url,
                    "char_count": char_count,
                    "all_content_displayed": scrape_result["all_content_displayed"],
                    "extraction_used": False,
                }
            ),
        )

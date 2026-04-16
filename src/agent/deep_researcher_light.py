"""Deep Researcher Light — single-round research, no file output.

Stripped-down version of DeepResearcherAgent:
- One search round only (web searcher or LLM searcher)
- No query generation LLM call — uses task directly as query
- No report file saved to disk
- Returns merged search result directly
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from src.agent.types import Agent, AgentExtra, AgentResponse
from src.logger import logger
from src.model import model_manager
from src.registry import AGENT
from src.session import SessionContext
from src.tool.default_tools.web_searcher import WebSearcherTool


@AGENT.register_module(force=True)
class DeepResearcherLightAgent(Agent):
    """Single-round research agent — fast, no file output.

    Performs one search using either WebSearcherTool or a search-capable LLM
    model, then returns the result immediately without saving any files.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="deep_researcher_light_agent")
    description: str = Field(
        default=(
            "Lightweight single-round research agent. Searches the web or uses an LLM "
            "search model once and returns the result immediately without saving files."
        )
    )
    metadata: Dict[str, Any] = Field(default_factory=dict)
    require_grad: bool = Field(default=False)

    def __init__(
        self,
        workdir: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        model_name: Optional[str] = None,
        prompt_name: Optional[str] = None,
        memory_name: Optional[str] = None,
        require_grad: bool = False,
        num_results: int = 5,
        use_llm_search: bool = True,
        search_llm_models: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            workdir=workdir,
            name=name,
            description=description,
            metadata=metadata,
            model_name=model_name or "openrouter/gemini-3.1-flash-lite-preview",
            prompt_name=prompt_name,
            memory_name=memory_name,
            require_grad=require_grad,
            **kwargs,
        )
        self.num_results = num_results
        self.search_llm_models: List[str] = search_llm_models or [
            "openrouter/gemini-3.1-flash-lite-preview-plugins",
        ]
        self.use_llm_search = use_llm_search and len(self.search_llm_models) > 0
        self._web_searcher = WebSearcherTool(model_name=self.model_name)

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------

    async def __call__(
        self,
        task: str,
        files: Optional[List[str]] = None,
        **kwargs,
    ) -> AgentResponse:
        filter_year: int = kwargs.get("filter_year", __import__("datetime").datetime.now().year)

        logger.info(f"| 🔍 DeepResearcherLightAgent starting: {task}")

        try:
            search_results = await self._search(task, filter_year)

            if not search_results:
                logger.warning("| ❌ All searches failed")
                return AgentResponse(
                    success=False,
                    message="No search results found.",
                )

            merged = self._merge(search_results)
            logger.info("| ✅ DeepResearcherLightAgent done")

            return AgentResponse(
                success=True,
                message=merged,
                extra=AgentExtra(data={"task": task}),
            )

        except Exception as exc:
            logger.error(f"| ❌ DeepResearcherLightAgent error: {exc}", exc_info=True)
            return AgentResponse(
                success=False,
                message=f"Error during research: {exc}",
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _search(self, task: str, filter_year: Optional[int]) -> List[Dict[str, Any]]:
        """Run all configured searches in parallel and return successful ones."""
        search_tasks = []

        if self.use_llm_search:
            from src.message.types import HumanMessage
            prompt = (
                f"Research the following task and provide a comprehensive summary "
                f"with citations where possible.\n\nTask: {task}\n\n"
                f"Return your findings as a comprehensive summary."
            )

            async def _llm_search(model: str) -> Dict[str, Any]:
                try:
                    logger.info(f"| Using LLM {model} to search.")
                    resp = await model_manager(model=model, messages=[HumanMessage(content=prompt)])
                    if not resp or not resp.success:
                        raise ValueError(f"Model call failed: {resp.message if resp else 'no response'}")
                    text = resp.message.strip() if resp.message else ""
                    if not text:
                        raise ValueError("Empty response content")
                    return {"source": model, "summary": text, "success": True}
                except Exception as exc:
                    logger.warning(f"| LLM search {model} failed: {exc}")
                    return {"source": model, "summary": None, "success": False}

            search_tasks = [_llm_search(m) for m in self.search_llm_models]
        else:
            async def _web_task():
                try:
                    resp = await self._web_searcher(
                        query=task,
                        num_results=self.num_results,
                        filter_year=filter_year,
                    )
                    if resp.success:
                        return {"source": "web_searcher", "summary": resp.message.strip(), "success": True}
                    return {"source": "web_searcher", "summary": None, "success": False}
                except Exception as exc:
                    logger.warning(f"| Web searcher failed: {exc}")
                    return {"source": "web_searcher", "summary": None, "success": False}
            search_tasks.append(_web_task())

        raw = await asyncio.gather(*search_tasks, return_exceptions=True)
        return [
            r for r in raw
            if not isinstance(r, Exception) and r.get("success") and r.get("summary")
        ]

    def _merge(self, results: List[Dict[str, Any]]) -> str:
        if len(results) == 1:
            return results[0]["summary"]
        parts = [f"## {r.get('source', f'Source {i}')}\n\n{r['summary']}" for i, r in enumerate(results, 1)]
        return "\n\n".join(parts)

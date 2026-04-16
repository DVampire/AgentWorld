"""Deep Analyzer Light — single-round analysis, images only, no file output.

Stripped-down version of DeepAnalyzerAgent:
- One analysis round only
- Only handles image files (no text/PDF/audio/video classification)
- No report file saved to disk
- If images provided: multimodal message; otherwise plain-text message
- Multiple analyzer_llm_models called in parallel; results merged by model_name
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Dict, List, Optional

from pydantic import ConfigDict, Field

from src.agent.types import Agent, AgentExtra, AgentResponse
from src.logger import logger
from src.model import model_manager
from src.registry import AGENT
from src.utils import assemble_project_path, make_file_url

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}


@AGENT.register_module(force=True)
class DeepAnalyzerLightAgent(Agent):
    """Single-round analysis agent — images + text, no file output.

    Calls all analyzer_llm_models in parallel, then uses model_name to
    aggregate their results into a final answer.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="deep_analyzer_light_agent")
    description: str = Field(
        default=(
            "Lightweight single-round analysis agent. Analyzes a task with optional "
            "image attachments using multiple LLMs in parallel, then synthesizes results."
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
        analyzer_llm_models: Optional[List[str]] = None,
        prompt_name: Optional[str] = None,
        memory_name: Optional[str] = None,
        require_grad: bool = False,
        **kwargs,
    ):
        super().__init__(
            workdir=workdir,
            name=name,
            description=description,
            metadata=metadata,
            model_name=model_name or "openrouter/gemini-3.1-pro-preview",
            prompt_name=prompt_name,
            memory_name=memory_name,
            require_grad=require_grad,
            **kwargs,
        )
        self.analyzer_llm_models: List[str] = analyzer_llm_models or [
            "openrouter/gemini-3.1-pro-preview",
            "openrouter/claude-opus-4.6",
            "openrouter/gpt-5.4-pro",
        ]

    # ------------------------------------------------------------------
    # Main call
    # ------------------------------------------------------------------

    async def __call__(
        self,
        task: str,
        files: Optional[List[str]] = None,
        **kwargs,
    ) -> AgentResponse:
        logger.info(f"| 🔍 DeepAnalyzerLightAgent starting: {task}")

        try:
            images = [f for f in (files or []) if self._is_image(f)]

            # Run all analyzer models in parallel
            results = await asyncio.gather(
                *[self._analyze(model, task, images) for model in self.analyzer_llm_models],
                return_exceptions=True,
            )

            analyses: List[Dict[str, str]] = []
            for model, r in zip(self.analyzer_llm_models, results):
                if isinstance(r, Exception):
                    logger.warning(f"| ⚠️ {model} failed: {r}")
                elif r:
                    analyses.append({"model": model, "analysis": r})
                    logger.info(f"| ✅ {model} done")

            if not analyses:
                return AgentResponse(success=False, message="All analyzer models failed.")

            # Single model — return directly without a synthesis call
            if len(analyses) == 1:
                final = analyses[0]["analysis"]
            else:
                final = await self._merge(task, analyses)

            logger.info("| ✅ DeepAnalyzerLightAgent done")
            return AgentResponse(
                success=True,
                message=final,
                extra=AgentExtra(data={"task": task}),
            )

        except Exception as exc:
            logger.error(f"| ❌ DeepAnalyzerLightAgent error: {exc}", exc_info=True)
            return AgentResponse(success=False, message=f"Error during analysis: {exc}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _is_image(self, path: str) -> bool:
        _, ext = os.path.splitext(path.lower())
        return ext in _IMAGE_EXTS

    def _build_messages(self, task: str, images: List[str]) -> list:
        from src.message.types import HumanMessage, SystemMessage
        from src.message import ContentPartText, ContentPartImage, ImageURL

        system = SystemMessage(content="You are a helpful analysis assistant.")
        if not images:
            return [system, HumanMessage(content=task)]

        content = [ContentPartText(text=task)]
        for img in images:
            url = img if img.startswith(("http://", "https://")) else make_file_url(
                file_path=assemble_project_path(img)
            )
            content.append(ContentPartImage(image_url=ImageURL(url=url, detail="high")))
        return [system, HumanMessage(content=content)]

    async def _analyze(self, model: str, task: str, images: List[str]) -> Optional[str]:
        messages = self._build_messages(task, images)
        resp = await model_manager(model=model, messages=messages)
        text = resp.message.strip() if resp else ""
        return text or None

    async def _merge(self, task: str, analyses: List[Dict[str, str]]) -> str:
        from src.message.types import HumanMessage, SystemMessage

        parts = "\n\n".join(
            f"### Analysis from {a['model']}\n{a['analysis']}" for a in analyses
        )
        prompt = (
            f"You are synthesizing multiple analyses of the same task.\n\n"
            f"Task: {task}\n\n"
            f"{parts}\n\n"
            f"Provide a single, comprehensive answer that integrates the above analyses."
        )
        messages = [
            SystemMessage(content="You are a helpful analysis assistant."),
            HumanMessage(content=prompt),
        ]
        resp = await model_manager(model=self.model_name, messages=messages)
        return resp.message.strip() if resp else parts

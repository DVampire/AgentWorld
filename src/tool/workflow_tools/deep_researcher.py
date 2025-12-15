"""Deep Researcher Tool - A workflow agent for multi-round web research."""

import asyncio
import os
import uuid
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict

from src.logger import logger
from src.model import model_manager
from src.utils import make_file_url
from src.utils import assemble_project_path
from src.utils import dedent
from src.message.types import HumanMessage, SystemMessage
from src.tool.types import Tool, ToolResponse
from src.tool.default_tools.web_searcher import WebSearcherTool
from src.registry import TOOL


_DEEP_RESEARCHER_DESCRIPTION = """Deep research tool that performs multi-round web search and content analysis.
This tool will:
1. Generate appropriate search queries based on the task
2. Use web_searcher to search, fetch, and summarize web pages
3. Evaluate if the answer is found based on the summaries
4. If not found, generate new queries and repeat (up to max rounds)
"""

class CompletenessEvaluation(BaseModel):
    """Response format for evaluating if research is complete."""
    is_complete: bool = Field(description="Whether the summary provides a complete answer to the research task")
    reasoning: str = Field(description="Brief explanation of why the answer is or isn't complete")


class ResearchSummary(BaseModel):
    """Summary of the research report."""
    summary: str = Field(description="Comprehensive summary of the research findings")
    answer_found: bool = Field(description="Whether a complete answer was found to the research task")
    answer_status: str = Field(description="Clear statement about whether the answer was found or not found")

@TOOL.register_module(force=True)
class DeepResearcherTool(Tool):
    """A deep research tool that performs multi-round web search and content analysis."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "deep_researcher"
    description: str = _DEEP_RESEARCHER_DESCRIPTION
    enabled: bool = True
    
    # Configuration parameters
    max_rounds: int = Field(default=3, description="Maximum search rounds")
    num_results: int = Field(default=5, description="Number of search results per round")
    model_name: str = Field(
        default="o3",
        description="The model to use for query generation and answer evaluation."
    )
    web_searcher: WebSearcherTool = Field(
        default=None,
        description="The web searcher to use for the deep researcher."
    )
    research_history: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="The research history with queries and summaries."
    )
    use_llm_search: bool = Field(
        default=True,
        description="Whether to use LLM to search the web."
    )
    search_llm_models: List[str] = Field(
        default=["o3-deep-research", "sonar-deep-research"],
        description="The LLM models to use for searching the web."
    )
    base_dir: str = Field(
        default="workdir/deep_researcher",
        description="The base directory for the deep researcher."
    )

    def __init__(self, 
                 base_dir: Optional[str] = None,
                 model_name: Optional[str] = None, 
                 use_llm_search: Optional[bool] = None, 
                 search_llm_models: Optional[List[str]] = None,
                 **kwargs):
        """Initialize the deep researcher tool."""
        super().__init__(**kwargs)
        
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(self.base_dir)
        
        # Create base directory if it doesn't exist
        if self.base_dir:
            os.makedirs(self.base_dir, exist_ok=True)
        
        if model_name is not None:
            self.model_name = model_name
        
        # Initialize web searcher with the same model
        if self.web_searcher is None:
            self.web_searcher = WebSearcherTool(model_name=self.model_name)

        if search_llm_models is not None:
            self.search_llm_models = search_llm_models
        
        if use_llm_search is not None:
            self.use_llm_search = use_llm_search
        
        # Only enable LLM search if models are available
        self.use_llm_search = self.use_llm_search and len(self.search_llm_models) > 0
        
        # Store research history
        self.research_history = []
        
        # File path (will be set in __call__)
        self.file_path = None

    async def __call__(self, 
                       task: str, 
                       image: Optional[str] = None, 
                       filter_year: Optional[int] = None,
                       **kwargs) -> ToolResponse:
        """
        Execute deep research workflow.
        
        Args:
            task (str): The research task or question to investigate
            image (Optional[str]): Optional image absolute path to analyze along with the task
            filter_year (Optional[int]): Optional year filter for search results
        """
        try:
            logger.info(f"🔍 Starting deep research for task: {task}")
            
            # Reset research history
            self.research_history = []
            
            # Create file path
            md_filename = f"research_{uuid.uuid4().hex[:8]}.md"
            self.file_path = os.path.join(self.base_dir, md_filename) if self.base_dir else None
            
            # Initialize file with task
            if self.file_path:
                with open(self.file_path, 'w', encoding='utf-8') as f:
                    f.write(f"# Research Report\n\n")
                    f.write(f"## Research Task\n\n{task}\n\n")
                    if image:
                        f.write(f"## Image\n\n{image}\n\n")
                    f.write(f"---\n\n")
            
            final_evaluation = None
            
            # Execute multiple search rounds
            for round_num in range(1, self.max_rounds + 1):
                logger.info(f"📋 Starting round {round_num}/{self.max_rounds}")
                
                # Generate search query
                query = await self._generate_search_query(task, round_num, image)
                logger.info(f"| ✅ Generated query for round {round_num}: {query}")
                
                # Execute parallel searches: web_searcher and LLM models
                search_results = await self._parallel_search(task, query, filter_year)
                
                if not search_results:
                    logger.warning(f"| ❌ All searches failed in round {round_num}")
                    # Still write to file
                    if self.file_path:
                        self._append_round_to_md(round_num, query, "No search results found.", None)
                    continue
                
                # Merge all search results
                merged_summary = self._merge_search_results(search_results)
                logger.info(f"| ✅ Merged {len(search_results)} search results: {merged_summary[:200]}...")
                
                # Record round information
                round_info = {
                    "round_number": round_num,
                    "query": query,
                    "summary": merged_summary,
                }
                self.research_history.append(round_info)
                
                # Check if answer is found
                evaluation = await self._evaluate_completeness(task, merged_summary)
                
                # Append round to file
                if self.file_path:
                    self._append_round_to_md(round_num, query, merged_summary, evaluation)
                
                if evaluation.is_complete:
                    logger.info(f"✅ Answer found in round {round_num}: {evaluation.reasoning[:100]}...")
                    final_evaluation = evaluation
                    break
                
                logger.info(f"| ⏭️ Round {round_num} completed, continuing to next round")
            
            # Finalize file and generate summary
            answer_found = final_evaluation.is_complete if final_evaluation else False
            
            if self.file_path:
                # Read the file
                with open(self.file_path, 'r', encoding='utf-8') as f:
                    md_content = f.read()
                
                # Reorganize and generate summary
                reorganized_md, summary = await self._reorganize_and_summarize(
                    task, md_content, answer_found, final_evaluation
                )
                
                # Write reorganized content back
                with open(self.file_path, 'w', encoding='utf-8') as f:
                    f.write(reorganized_md)
                
                # Build evaluation info for extra
                evaluation_info = None
                if final_evaluation:
                    evaluation_info = {
                        "is_complete": final_evaluation.is_complete,
                        "reasoning": final_evaluation.reasoning,
                    }
                
                # Return with summary
                return ToolResponse(
                    success=answer_found,
                    message=summary,
                    extra={
                        "task": task,
                        "rounds": len(self.research_history),
                        "history": self.research_history,
                        "file_path": self.file_path,
                        "answer_found": answer_found,
                        "evaluation": evaluation_info
                    }
                )
            else:
                # Fallback if MD file path is not available (shouldn't happen normally)
                if answer_found and final_evaluation:
                    final_message = f"{self.research_history[-1]['summary']}\n\n## Evaluation\n\n{final_evaluation.reasoning}"
                elif self.research_history:
                    final_message = f"Research incomplete after {self.max_rounds} rounds.\n\n{self.research_history[-1]['summary']}"
                else:
                    final_message = "No search results found in any round."
                
                return ToolResponse(
                    success=answer_found,
                    message=final_message,
                    extra={
                        "task": task,
                        "rounds": len(self.research_history),
                        "history": self.research_history,
                        "answer_found": answer_found,
                    }
                )
            
        except Exception as e:
            logger.error(f"❌ Error in deep research: {e}")
            return ToolResponse(success=False, message=f"Error during deep research: {e}")

    async def _generate_search_query(self, task: str, round_num: int, image: Optional[str] = None) -> str:
        """Generate search query using LLM based on task, image, and round number."""
        system_prompt = """You are a helpful assistant that can analyze tasks and images to generate optimized search queries."""
        
        # Build context from previous rounds
        previous_summaries = []
        for i, round_info in enumerate(self.research_history[-2:], 1):  # Last 2 rounds
            previous_summaries.append(f"Round {round_info['round_number']} query: {round_info['query']}")
            previous_summaries.append(f"Summary: {round_info['summary'][:200]}...")  # First 200 chars
        
        previous_context = "\n".join(previous_summaries) if previous_summaries else "No previous searches yet."
        image_context = f"And this image: {image}" if image else "No image provided"
        round_context = f"Round: {round_num}" if round_num > 1 else "Round: 1 (initial)"
        
        # Build instruction
        if round_num > 1:
            instruction = "generate a new search query that might help find missing information. Focus on different aspects, use different keywords, or explore related topics."
        else:
            instruction = "generate an optimized search query for web research. Focus on the most important keywords and concepts."
        
        user_prompt = dedent(f"""Given this research task: "{task}"
        
        {image_context}
        {round_context}
        Previous search results:
        {previous_context}
        
        Analyze the image if provided and combine it with the text task to {instruction}
        
        IMPORTANT: The search query must be concise and focused. Use only the most important keywords (typically 3-8 words). Avoid long phrases or complete sentences. Keep it short and search-friendly.
        
        Return only the search query, nothing else.""")
        
        if image and image.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Multimodal query with image
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": make_file_url(file_path=assemble_project_path(image))}}
                ])
            ]
        else:
            # Text-only query
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
        
        response = await model_manager(model=self.model_name, messages=messages)
        return response.message.strip()

    async def _parallel_search(self, task: str, query: str, filter_year: Optional[int]) -> List[Dict[str, Any]]:
        """Execute parallel searches using web_searcher and LLM models."""
        search_tasks = []
        
        # Add web_searcher task
        async def web_search_task():
            try:
                response = await self.web_searcher(
                    query=query,
                    num_results=self.num_results,
                    filter_year=filter_year
                )
                if response.success:
                    return {
                        "source": "web_searcher",
                        "summary": response.message.strip(),
                        "success": True
                    }
                else:
                    return {
                        "source": "web_searcher",
                        "summary": None,
                        "success": False,
                        "error": response.message
                    }
            except Exception as e:
                logger.warning(f"Web searcher failed: {e}")
                return {
                    "source": "web_searcher",
                    "summary": None,
                    "success": False,
                    "error": str(e)
                }
        
        search_tasks.append(web_search_task())
        
        # Add LLM search tasks if enabled
        if self.use_llm_search and self.search_llm_models:
            for model_name in self.search_llm_models:
                # Create a closure to capture model_name correctly
                def create_llm_task(model):
                    async def llm_search_task():
                        try:
                            summary = await self._llm_search(model, task, query)
                            return {
                                "source": model,
                                "summary": summary,
                                "success": True
                            }
                        except Exception as e:
                            logger.warning(f"LLM search with {model} failed: {e}")
                            return {
                                "source": model,
                                "summary": None,
                                "success": False,
                                "error": str(e)
                            }
                    return llm_search_task
                
                search_tasks.append(create_llm_task(model_name)())
        
        # Execute all searches in parallel
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Process results and filter out failures
        search_results = []
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Search task raised exception: {result}")
                continue
            if result.get("success") and result.get("summary"):
                search_results.append(result)
            else:
                logger.warning(f"Search from {result.get('source', 'unknown')} failed: {result.get('error', 'Unknown error')}")
        
        return search_results

    async def _llm_search(self, model_name: str, task: str, query: str) -> str:
        """Use LLM model to search the web and return summary."""
        prompt = dedent(f"""You are an expert web researcher. Based on the research task and search query, perform a comprehensive web search and provide a detailed summary.

        Research Task: {task}
        Search Query: {query}
        
        Please search the web for information related to this task and query, then provide a comprehensive summary that:
        1. Directly addresses the research task
        2. Includes relevant information from multiple sources
        3. Provides detailed insights and findings
        4. Includes citations or references when possible
        5. Is well-structured and easy to read
        
        Return your research findings as a comprehensive summary.""")
        
        logger.info(f"| Using LLM {model_name} to search the web.")
        
        message = HumanMessage(content=prompt)
        response = await model_manager(model=model_name, messages=[message])
        
        logger.info(f"| LLM {model_name} response: {response.message.strip()[:200]}...")
        
        if response and response.message.strip():
            return response.message.strip()
        else:
            raise ValueError(f"LLM {model_name} returned empty response")

    def _merge_search_results(self, search_results: List[Dict[str, Any]]) -> str:
        """Merge multiple search results into a single comprehensive summary."""
        if not search_results:
            return "No search results available."
        
        if len(search_results) == 1:
            return search_results[0]["summary"]
        
        # Combine all summaries with source labels
        combined_parts = []
        for i, result in enumerate(search_results, 1):
            source = result.get("source", f"Source {i}")
            summary = result.get("summary", "")
            combined_parts.append(f"## {source}\n\n{summary}\n")
        
        combined_text = "\n".join(combined_parts)
        
        # If we have multiple sources, we could optionally use LLM to merge them
        # For now, just return the combined text
        return combined_text

    def _append_round_to_md(self, round_num: int, query: str, summary: str, evaluation: Optional[CompletenessEvaluation]):
        """Append a research round to the file."""
        if not self.file_path:
            return
        
        try:
            with open(self.file_path, 'a', encoding='utf-8') as f:
                f.write(f"## Round {round_num}\n\n")
                f.write(f"### Search Query\n\n{query}\n\n")
                f.write(f"### Search Results\n\n{summary}\n\n")
                if evaluation:
                    f.write(f"### Evaluation\n\n")
                    f.write(f"- **Answer Found**: {'Yes' if evaluation.is_complete else 'No'}\n")
                    f.write(f"- **Reasoning**: {evaluation.reasoning}\n\n")
                f.write(f"---\n\n")
        except Exception as e:
            logger.warning(f"Failed to append round {round_num} to MD file: {e}")

    async def _reorganize_and_summarize(
        self, 
        task: str, 
        md_content: str, 
        answer_found: bool,
        evaluation: Optional[CompletenessEvaluation]
    ) -> tuple[str, str]:
        """Reorganize the MD file structure and generate a summary."""
        prompt = dedent(f"""You are an expert at organizing research reports. Reorganize the following research report markdown file to make it well-structured and easy to read.

        Research Task: {task}
        Answer Found: {'Yes' if answer_found else 'No'}
        {'Evaluation: ' + evaluation.reasoning if evaluation else 'No evaluation available'}

        Current Markdown Content:
        {md_content}

        Please reorganize this markdown file with the following structure:
        1. Title and Research Task (clear and prominent)
        2. Executive Summary (brief overview of findings)
        3. Research Methodology (if applicable)
        4. Detailed Findings (organized by rounds or topics)
        5. Conclusion (with clear statement about whether the answer was found)
        6. Evaluation (if available)

        Make sure the conclusion section clearly states:
        - Whether a complete answer to the research task was found
        - A summary of the key findings
        - Any limitations or gaps in the research

        Return the reorganized markdown content. Keep all important information from the original content.
        """)

        messages = [
            SystemMessage(content="You are an expert at organizing and structuring research reports. Create clear, well-organized markdown documents."),
            HumanMessage(content=prompt)
        ]

        response = await model_manager(
            model=self.model_name,
            messages=messages
        )

        reorganized_md = response.message.strip()

        # Generate summary with explicit answer status
        answer_status = f"Answer Found: {'Yes' if answer_found else 'No'}"
        evaluation_text = f"\n\nEvaluation: {evaluation.reasoning}" if evaluation else ""
        
        summary_prompt = dedent(f"""Based on the reorganized research report below, generate a comprehensive summary.

        Research Task: {task}
        Answer Found Status: {answer_status}{evaluation_text}
        
        Reorganized Report:
        {reorganized_md}

        Generate a summary that:
        1. MUST start with a clear statement: "Answer Found: Yes" or "Answer Found: No" (use this exact format)
        2. Provides a concise overview of the key findings
        3. Highlights the most important information discovered
        4. Mentions the number of research rounds conducted
        5. If answer was found, summarize the key answer points
        6. If answer was not found, explain what information was discovered and what gaps remain

        The summary should be clear and informative, suitable as a final response message.
        The first line must explicitly state whether the answer was found.
        """)

        summary_messages = [
            SystemMessage(content="You are an expert at summarizing research reports. Generate clear, informative summaries that MUST explicitly state whether answers were found in the first line using 'Answer Found: Yes' or 'Answer Found: No'."),
            HumanMessage(content=summary_prompt)
        ]

        summary_response = await model_manager(
            model=self.model_name,
            messages=summary_messages,
            response_format=ResearchSummary
        )

        # Extract summary text
        if summary_response.extra and "parsed_model" in summary_response.extra:
            research_summary = summary_response.extra["parsed_model"]
            summary_text = research_summary.summary
            # Verify answer_found matches
            if research_summary.answer_found != answer_found:
                logger.warning(f"Answer found mismatch: model says {research_summary.answer_found}, actual is {answer_found}")
        else:
            summary_text = summary_response.message.strip()
        
        # Ensure answer status is at the beginning
        if not summary_text.startswith("Answer Found:"):
            final_summary = f"{answer_status}\n\n{summary_text}"
        else:
            final_summary = summary_text
        
        return reorganized_md, final_summary

    async def _evaluate_completeness(self, task: str, summary: str) -> CompletenessEvaluation:
        """Evaluate if we have found a complete answer using LLM with structured output."""
        prompt = dedent(f"""Evaluate if the following summary provides a complete answer to the research task.
        
        Research Task: {task}
        
        Summary from web search:
        {summary}
        
        Determine if this summary provides enough information to answer the research task completely.
        
        Consider:
        - Does the information directly address the task?
        - Is there sufficient detail and depth?
        - Are there multiple perspectives or sources mentioned?
        - Is the information comprehensive enough?""")
        
        try:
            message = HumanMessage(content=prompt)
            response = await model_manager(
                model=self.model_name,
                messages=[message],
                response_format=CompletenessEvaluation
            )
            
            if response and response.extra and "parsed_model" in response.extra:
                evaluation = response.extra["parsed_model"]
                logger.info(f"| Evaluation: is_complete={evaluation.is_complete}, reasoning={evaluation.reasoning[:100]}...")
                return evaluation
            
            # Fallback if response_format parsing failed
            logger.warning("Failed to parse response_format, falling back to text parsing")
            is_complete = False
            reasoning = "Failed to parse structured response."
            if response and response.message.strip():   
                answer = response.message.strip().upper()
                is_complete = answer.startswith("YES")
                reasoning = f"Text-based evaluation: {response.message.strip()}"
            
            return CompletenessEvaluation(is_complete=is_complete, reasoning=reasoning)
        
        except Exception as e:
            logger.warning(f"Failed to evaluate completeness with LLM: {e}")
            # Fallback: if summary is long enough and contains task keywords, consider it complete
            task_lower = task.lower()
            summary_lower = summary.lower()
            key_terms = [term for term in task_lower.split() if len(term) > 3]  # Filter out short words
            
            is_complete = len(summary) > 500 and any(term in summary_lower for term in key_terms)
            reasoning = f"Fallback heuristic evaluation: summary length={len(summary)}, keyword match={'yes' if is_complete else 'no'}"
            return CompletenessEvaluation(is_complete=is_complete, reasoning=reasoning)

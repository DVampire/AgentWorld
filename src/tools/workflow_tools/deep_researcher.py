"""Deep Researcher Tool - A workflow agent for multi-round web research."""

import asyncio
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel, Field
from src.tools.protocol.tool import BaseTool
from langchain_core.messages import HumanMessage, SystemMessage
from inspect import cleandoc
from PIL import Image

from src.tools.default_tools.web_searcher import WebSearcherTool
from src.tools.default_tools.web_fetcher import WebFetcherTool
from src.logger import logger
from src.infrastructures.models import model_manager
from src.config import config
from src.utils import make_image_url
from src.utils import encode_image_base64
from src.utils import assemble_project_path
from src.tools.default_tools.web_searcher import SearchResult
from src.tools.protocol.types import ToolResponse
from src.tools.protocol import tcp


_DEEP_RESEARCHER_DESCRIPTION = """Deep research tool that performs multi-round web search and content analysis.
This tool will:
1. Generate appropriate search queries based on the task
2. Search the web for relevant pages
3. Fetch and analyze content from multiple pages
4. Summarize insights and determine if the answer is found
5. If not found, generate new queries and repeat (up to max rounds)
"""

class DeepResearcherArgs(BaseModel):
    task: str = Field(description="The research task or question to investigate")
    image: Optional[str] = Field(
        default=None,
        description="Optional image absolute path to analyze along with the task"
    )
    filter_year: Optional[int] = Field(
        default=None,
        description="Optional year filter for search results"
    )

class SearchRound(BaseModel):
    """Represents a single search round."""
    round_number: int
    query: str
    search_results: List[SearchResult]
    insights: List[str]
    summary: str

@tcp.tool()
class DeepResearcherTool(BaseTool):
    """A deep research tool that performs multi-round web search and content analysis."""

    name: str = "deep_researcher"
    type: str = "Deep Researcher"
    description: str = _DEEP_RESEARCHER_DESCRIPTION
    args_schema: Type[DeepResearcherArgs] = DeepResearcherArgs
    metadata: Dict[str, Any] = {}
    
    # Configuration parameters as class attributes
    max_rounds: int = Field(default=10, description="Maximum search rounds")
    max_pages_per_round: int = Field(default=5, description="Max pages per round")
    max_content_length: int = Field(default=4000, description="Max content length to analyze")
    max_num_reviews_insights: int = Field(default=5, description="Max number of reviews for insights")
    insight_extraction_prompt: str = Field(
        default="Extract 1-3 key insights related to the query from this content. Focus on factual information and direct answers.",
        description="Prompt for extracting insights from content"
    )
    query_generation_prompt: str = Field(
        default="Based on the current task and previous search results, generate a new search query that might help find the answer. Focus on different aspects or use different keywords.",
        description="Prompt for generating new search queries"
    )
    summary_prompt: str = Field(
        default="Based on all the insights collected so far, determine if we have found a complete answer to the task. If yes, provide a comprehensive answer. If no, explain what information is still missing.",
        description="Prompt for summarizing insights and determining if answer is complete"
    )
    model_name: str = Field(
        default="o3",
        description="The model to use for the deep researcher."
    )
    web_searcher: WebSearcherTool = Field(
        default=None,
        description="The web searcher to use for the deep researcher."
    )
    web_fetcher: WebFetcherTool = Field(
        default=None,
        description="The web fetcher to use for the deep researcher."
    )
    model: Any = Field(
        default=None,
        description="The model to use for the deep researcher."
    )
    research_history: List[SearchRound] = Field(
        default=[],
        description="The research history."
    )
    all_insights: List[str] = Field(
        default=[],
        description="The all insights."
    )

    def __init__(self, model_name: Optional[str] = None, **kwargs):
        model_name = model_name or config.deep_researcher_tool.get("model_name", "o3")
        super().__init__(model_name=model_name, **kwargs)
        self.model_name = model_name
        
        # Initialize tools
        self.web_searcher = WebSearcherTool()
        self.web_fetcher = WebFetcherTool()
        
        # Initialize model
        self.model = model_manager.get(self.model_name)
        
        # Store research history
        self.research_history: List[SearchRound] = []
        self.all_insights: List[str] = []

    async def _arun(self, task: str, image: Optional[str] = None, filter_year: Optional[int] = None) -> ToolResponse:
        """Execute deep research workflow."""
        try:
            logger.info(f"Starting deep research for task: {task}")
            
            # Reset research history
            self.research_history = []
            self.all_insights = []
            
            # Execute multiple search rounds
            for round_num in range(1, self.max_rounds + 1):
                logger.info(f"Starting round {round_num}/{self.max_rounds}")
                
                # Generate search query
                query = await self._generate_search_query(task, round_num, image)
                logger.info(f"| ✅ Generated query for round {round_num}: {query}")
                
                # Execute search
                search_results = await self._execute_search(query, self.max_pages_per_round, filter_year)
                logger.info(f"| ✅ Search results: {len(search_results)}")
                if not search_results:
                    logger.warning(f"| ❌ No search results found in round {round_num}")
                    continue
                
                # Fetch page contents
                search_results = await self._fetch_page_contents(search_results)
                logger.info(f"| ✅ Fetched page contents for round {round_num}")
                
                # Extract insights
                insights = await self._extract_insights(search_results, query)
                logger.info(f"| ✅ Extracted insights for round {round_num}: {chr(10).join(insights)}")
                self.all_insights.extend(insights)
                
                # Summarize current round
                round_summary = await self._summarize_round(insights, query)
                logger.info(f"| ✅ Summarized round {round_num}: {round_summary}")
                
                # Record round information
                search_round = SearchRound(
                    round_number=round_num,
                    query=query,
                    search_results=search_results,
                    insights=insights,
                    summary=round_summary
                )
                self.research_history.append(search_round)
                
                # Check if answer is found
                final_summary = await self._evaluate_completeness(task)
                if "ANSWER_FOUND" in final_summary:
                    logger.info(f"Answer found in round {round_num}")
                    return await self._format_final_result(final_summary, round_num)
                
                logger.info(f"Round {round_num} completed, continuing to next round")
            
            # If all rounds completed without finding answer
            logger.warning("Maximum rounds reached without finding complete answer")
            return await self._format_failure_result(task)
            
        except Exception as e:
            logger.error(f"Error in deep research: {e}")
            return ToolResponse(content=f"Error during deep research: {str(e)}")

    async def _generate_search_query(self, task: str, round_num: int, image: Optional[str] = None) -> str:
        """Generate search query using LLM based on task, image, and round number."""
        system_prompt = """You are a helpful assistant that can analyze tasks and images to generate optimized search queries."""
        
        # Build the user prompt based on context
        previous_insights = chr(10).join(self.all_insights[-self.max_num_reviews_insights:]) if self.all_insights else "No previous insights available"
        image_context = f"And this image: {image}" if image else "No image provided"
        round_context = f"Round: {round_num}" if round_num > 1 else "Round: 1 (initial)"
        
        # Build instruction components
        instruction = "Analyze the image if provided and combine it with the text task to "
        if round_num > 1:
            instruction += "generate a new search query that might help find missing information. Focus on different aspects, use different keywords, or explore related topics."
        else:
            instruction += "generate an optimized search query for web research. Focus on the most important keywords and concepts."
        
        user_prompt = cleandoc(f"""Given this research task: "{task}"
        
        {image_context}
        {round_context}
        Previous insights collected:
        {previous_insights}
        
        {instruction}
        Return only the search query, nothing else.""")
        
        if image and image.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Multimodal query with image
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=[
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": make_image_url(encode_image_base64(Image.open(assemble_project_path(image))))}}
                ])
            ]
        else:
            # Text-only query
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
        
        response = await self.model.ainvoke(messages)
        
        return response.content.strip()

    async def _execute_search(self, query: str, max_pages: int, filter_year: Optional[int]) -> List[SearchResult]:
        """Execute web search and return results."""
        try:
            # Use web searcher to execute search
            search_response = await self.web_searcher.ainvoke(input={"query": query, "filter_year": filter_year})
            
            search_results = search_response.extra["results"]
            
            return search_results
        
        except Exception as e:
            logger.error(f"Error executing search: {e}")
            return []

    async def _fetch_page_contents(self, search_results: List[SearchResult]) -> List[SearchResult]:
        """Fetch content from multiple pages concurrently."""
        if not search_results:
            return []
        
        async def fetch_single_page(result: SearchResult) -> str:
            """Fetch content from a single page."""
            try:
                response = await self.web_fetcher._arun(result.url)
                return response.content
            except Exception as e:
                logger.warning(f"Failed to fetch page content for {result.url}: {e}")
                return f"Failed to fetch page content: {e}."
        
        # Create async tasks
        tasks = []
        for result in search_results:
            task = fetch_single_page(result)
            tasks.append(task)
        
        # Execute concurrently
        try:
            page_contents = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for i, content in enumerate(page_contents):
                if isinstance(content, Exception):
                    logger.warning(f"Failed to fetch page {i}: {content}")
                    search_results[i].raw_content = f"Failed to fetch page content for {search_results[i].url}: {content}."
                    continue
                
                search_results[i].raw_content = content[:self.max_content_length] if content else "No content."
            
            return search_results
        
        except Exception as e:
            logger.error(f"Error fetching page contents: {e}")
            return []

    async def _extract_insights(self, search_results: List[SearchResult], query: str) -> List[str]:
        """Extract insights from page contents using LLM with parallel processing."""
        
        async def extract_insights_from_result(search_result: SearchResult) -> str:
            """Extract insights from a single search result."""
            if not search_result.raw_content:
                return "No content."
            
            # Use LLM to extract insights from page content
            prompt = cleandoc(f"""Given this search query: "{query}"
            
            And this webpage content:
            Title: {search_result.title}
            URL: {search_result.url}
            Content: {search_result.raw_content[:self.max_content_length]}
            
            Extract 1-3 key insights that are most relevant to the query.
            Focus on factual information, direct answers, and actionable knowledge.
            Format each insight as a clear, concise statement.
            Return only the insights, one per line, nothing else.""")
            
            try:
                message = HumanMessage(content=prompt)
                response = await self.model.ainvoke([message])
                if response:
                    insights = response.content.strip().split('\n')
                    return insights
                else:
                    return []
            except Exception as e:
                logger.warning(f"Failed to extract insights from {search_result.title}: {e}")
                return []
        
        # Process all search results in parallel
        tasks = []
        for result in search_results:
            task = extract_insights_from_result(result)
            tasks.append(task)
        
        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Flatten the results and filter out exceptions
            insights = []
            for result in results:
                if isinstance(result, Exception):   
                    logger.warning(f"Exception during insight extraction: {result}")
                    continue
                insights.extend(result)
            return insights
        
        except Exception as e:
            logger.error(f"Error extracting insights: {e}")
            return []

    async def _summarize_round(self, insights: List[str], query: str) -> str:
        """Summarize the current round using LLM."""
        if not insights:
            return f"No relevant insights found for query: {query}"
        
        prompt = cleandoc(f"""Summarize the search results for this round.
        
        Query: {query}
        Insights found: {len(insights)}
        
        Key insights:
        {chr(10).join(insights)}
        
        Provide a brief summary (1-2 sentences) of what was discovered in this round.
        Focus on the most important findings and their relevance to the research task.""")
        
        try:
            message = HumanMessage(content=prompt)
            response = await self.model.ainvoke([message])
            return response.content.strip()
        
        except Exception as e:
            logger.warning(f"Failed to summarize round: {e}")
            return f"Failed to summarize round: {e}."

    async def _evaluate_completeness(self, task: str) -> str:
        """Evaluate if we have found a complete answer using LLM."""
        if not self.all_insights:
            return "No insights collected yet"
        
        prompt = cleandoc(f"""Evaluate if we have collected sufficient information to answer the research task.
        
        Research Task: {task}
        
        Insights collected so far:
        {chr(10).join(self.all_insights)}
        
        Determine if we have enough information to provide a complete answer.
        
        If YES, respond with: "ANSWER_FOUND: [brief explanation of what we found]"
        If NO, respond with: "INCOMPLETE: [explanation of what information is still missing]"
        
        Consider:
        - Does the information directly address the task?
        - Is there sufficient detail and depth?
        - Are there multiple perspectives or sources?
        - Is the information recent and reliable?""")
        
        try:
            message = HumanMessage(content=prompt)
            response = await self.model.ainvoke([message])
            if response and response.content.strip():   
                return response.content.strip()
            else:
                return self._fallback_completeness_check(task)
        
        except Exception as e:
            logger.warning(f"Failed to evaluate completeness with LLM: {e}")
            return self._fallback_completeness_check(task)
    
    def _fallback_completeness_check(self, task: str) -> str:
        """Fallback method for completeness evaluation using simple heuristics."""
        if len(self.all_insights) >= 5:
            task_lower = task.lower()
            key_terms = task_lower.split()
            
            coverage = 0
            for insight in self.all_insights:
                insight_lower = insight.lower()
                for term in key_terms:
                    if term in insight_lower:
                        coverage += 1
            
            if coverage >= len(key_terms) * 2:
                return "ANSWER_FOUND: Sufficient information collected to answer the task"
        
        return "INCOMPLETE: Need more information to provide a complete answer"

    async def _format_final_result(self, summary: str, round_num: int) -> ToolResponse:
        """Format the final successful result using LLM."""
        try:
            prompt = cleandoc(f"""You are formatting the final research results. Please create a well-structured, comprehensive summary.

            Research completed in {round_num} rounds.
            
            Final Summary: {summary}
            
            All Insights Collected ({len(self.all_insights)} total):
            {chr(10).join(self.all_insights)}
            
            Research Statistics:
            - Total rounds: {len(self.research_history)}
            - Total insights: {len(self.all_insights)}
            - Pages analyzed: {sum(len(r.search_results) for r in self.research_history)}
            
            Please format this into a clear, professional research report that includes:
            1. A compelling title/header
            2. Executive summary
            3. Key findings (prioritize the most important insights)
            4. Supporting evidence
            5. Research methodology summary
            
            Use appropriate emojis and formatting to make it engaging and easy to read.
            Focus on the most valuable insights and present them in a logical order.""")
            
            message = HumanMessage(content=prompt)
            response = await self.model.ainvoke([message])
            
            if response and response.content.strip():
                return ToolResponse(content=response.content.strip())
            else:
                return self._fallback_format_result(summary, round_num)
                
        except Exception as e:
            logger.warning(f"Failed to format final result with LLM: {e}")
            # Simple fallback
            return self._fallback_format_result(summary, round_num)
    
    def _fallback_format_result(self, summary: str, round_num: int) -> ToolResponse:
        """Basic fallback formatting for successful results."""
        result = f"🎯 Research completed in {round_num} rounds!\n\n"
        result += "📋 Summary:\n"
        result += summary.replace("ANSWER_FOUND: ", "") + "\n\n"
        
        result += "🔍 Key Insights:\n"
        for i, insight in enumerate(self.all_insights[-5:], 1):
            result += f"{i}. {insight}\n"
        
        result += f"\n📊 Research Statistics:\n"
        result += f"- Total rounds: {len(self.research_history)}\n"
        result += f"- Total insights: {len(self.all_insights)}\n"
        
        return ToolResponse(content=result) 

    async def _format_failure_result(self, task: str) -> ToolResponse:
        """Format the failure result when max rounds are reached using LLM."""
        try:
            prompt = cleandoc(f"""The research task was incomplete after maximum rounds. Please format a helpful failure report.

            Task: {task}
            
            Partial insights collected ({len(self.all_insights)} total):
            {chr(10).join(self.all_insights) if self.all_insights else "No insights collected"}
            
            Research Statistics:
            - Total rounds: {len(self.research_history)}
            - Total insights: {len(self.all_insights)}
            - Pages analyzed: {sum(len(r.search_results) for r in self.research_history)}
            
            Create a professional failure report that includes:
            1. Clear explanation of what happened
            2. Summary of partial findings (if any)
            3. Specific recommendations for improvement
            4. Alternative approaches to try
            
            Be constructive and helpful, not just negative.""")
            
            message = HumanMessage(content=prompt)
            response = await self.model.ainvoke([message])
            
            if response and response.content.strip():
                return ToolResponse(content=response.content.strip())
            else:
                return self._fallback_format_failure_result(task)
                
        except Exception as e:
            logger.warning(f"Failed to format failure result with LLM: {e}")
            # Simple fallback
            return self._fallback_format_failure_result(task)
    
    def _fallback_format_failure_result(self, task: str) -> ToolResponse:
        """Basic fallback formatting for failure results."""
        result = f"❌ Research incomplete after maximum rounds reached.\n\n"
        result += f"📋 Task: {task}\n\n"
        
        if self.all_insights:
            result += "🔍 Partial insights collected:\n"
            for i, insight in enumerate(self.all_insights[-3:], 1):
                result += f"{i}. {insight}\n"
            
            result += "\n💡 Recommendations:\n"
            result += "- Try rephrasing the query with different keywords\n"
            result += "- Consider breaking down the task into smaller subtasks\n"
            result += "- Check if the information might be available in different sources"
        else:
            result += "🔍 No insights collected. The task might be too specific or the information might not be available online."
        
        return ToolResponse(content=result)

    def _run(self, task: str, image: Optional[str] = None, filter_year: Optional[int] = None) -> ToolResponse:
        """Execute deep research synchronously (fallback)."""
        try:
            # Run async version
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._arun(task, image, filter_year))
            finally:
                loop.close()
        except Exception as e:
            logger.error(f"Error in synchronous execution: {e}")
            return ToolResponse(content=f"Error in synchronous execution: {e}")

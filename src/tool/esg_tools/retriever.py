"""Retriever Tool - A tool for retrieving ESG information from the LightRAG vector database."""

import os
import re
import json
import uuid
from typing import Optional, Dict, Any
import pandas as pd
from tabulate import tabulate
from pydantic import Field, ConfigDict

from dotenv import load_dotenv
load_dotenv(verbose=True)

from src.utils import assemble_project_path
from src.registry import TOOL
from src.logger import logger
from src.tool.types import Tool, ToolResponse, ToolExtra
from src.tool.workflow_tools.report import Report

from src.tool.esg_tools.lightrag import LightRAG, QueryParam
from src.tool.esg_tools.lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from src.tool.esg_tools.lightrag.kg.shared_storage import initialize_pipeline_status


_RETRIEVER_DESCRIPTION = """Retriever tool that retrieves ESG data from a local RAG database.

🎯 BEST FOR: Retrieving ESG-related information from local knowledge base:
- ESG reports and sustainability documents
- Company environmental, social, and governance data
- Carbon emissions, energy consumption, and other ESG metrics

This tool will:
1. Query the local LightRAG vector database for relevant ESG information
2. Extract structured ESG metadata from retrieved documents
3. Return both raw context and structured ESG facts

💡 Use this tool for:
- Looking up ESG metrics from company reports
- Finding sustainability information
- Retrieving governance and compliance data
- If the information is not found, please ask the `browser_use_agent` to search the web.
"""


@TOOL.register_module(force=True)
class RetrieverTool(Tool):
    """A tool that retrieves ESG information from a LightRAG vector database."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = "retriever"
    description: str = _RETRIEVER_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")

    # Configuration parameters
    model_name: str = Field(
        default="openrouter/gemini-3-flash-preview",
        description="The model to use for metadata extraction."
    )
    base_dir: str = Field(
        default="workdir/retriever",
        description="The base directory for LightRAG storage."
    )
    top_k: int = Field(
        default=20,
        description="Number of top results to retrieve."
    )
    query_mode: str = Field(
        default="naive",
        description="Query mode: 'naive', 'local', 'global', 'hybrid', or 'mix'."
    )
    extract_metadata: bool = Field(
        default=True,
        description="Whether to extract structured ESG metadata from results."
    )
    rag: Optional[LightRAG] = Field(
        default=None,
        description="The LightRAG instance (lazily initialized)."
    )

    def __init__(
        self, 
        model_name: Optional[str] = None, 
        base_dir: Optional[str] = None,
        top_k: int = 20,
        query_mode: str = "naive",
        extract_metadata: bool = True,
        **kwargs
    ):
        """Initialize the retriever tool.
        
        Args:
            model_name: The model to use for metadata extraction.
            base_dir: The base directory for LightRAG storage.
            top_k: Number of top results to retrieve.
            query_mode: Query mode for retrieval.
            extract_metadata: Whether to extract structured ESG metadata.
        """
        super().__init__(**kwargs)

        if model_name is not None:
            self.model_name = model_name
        
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(self.base_dir)
            
        if self.base_dir is not None:
            os.makedirs(self.base_dir, exist_ok=True)
        
        self.top_k = top_k
        self.query_mode = query_mode
        self.extract_metadata = extract_metadata
        self.rag = None  # Will be lazily initialized

    async def initialize_rag(self) -> LightRAG:
        """Initialize the LightRAG instance with OpenAI embeddings.
        
        Returns:
            Initialized LightRAG instance.
        """
        if self.base_dir is None:
            raise ValueError("base_dir must be specified for RetrieverTool")
        
        rag = LightRAG(
            working_dir=self.base_dir,
            embedding_func=openai_embed,
            llm_model_func=gpt_4o_mini_complete,
        )

        await rag.initialize_storages()
        await initialize_pipeline_status()

        return rag

    async def _ensure_rag_initialized(self) -> None:
        """Ensure the LightRAG instance is initialized."""
        if self.rag is None:
            logger.info(f"| Initializing LightRAG with base_dir: {self.base_dir}")
            self.rag = await self.initialize_rag()
            logger.info(f"| LightRAG initialized successfully")

    async def finalize(self) -> None:
        """Finalize and cleanup the LightRAG storage."""
        if self.rag is not None:
            try:
                await self.rag.finalize_storages()
                logger.info(f"| 🧹 LightRAG storage finalized")
            except Exception as e:
                logger.warning(f"Error finalizing LightRAG storage: {e}")

    async def __call__(
        self, 
        query: str, 
        top_k: Optional[int] = None,
        **kwargs
    ) -> ToolResponse:
        """Execute retrieval on the given query.

        Args:
            query (str): The query to search for in the vector database.
            top_k (Optional[int]): Override the number of top results to retrieve.
            
        Returns:
            ToolResponse: The retrieval results including context and optional metadata.
        """
        try:
            logger.info(f"| 🔍 Starting RetrieverTool with query: {query[:100]}...")

            # Create file path for markdown report
            md_filename = f"retrieval_{uuid.uuid4().hex[:8]}.md"
            file_path = os.path.join(self.base_dir, md_filename) if self.base_dir else None
            
            # Initialize Report instance
            report = Report(
                title="Retrieval Report",
                model_name=self.model_name
            )
            
            # Add initial query information
            query_content = f"## Query\n\n{query}\n\n"
            await report.add_item(query_content)

            # Ensure RAG is initialized
            await self._ensure_rag_initialized()

            # Use provided parameters or defaults
            query_mode = "naive"
            result_top_k = top_k if top_k is not None else self.top_k

            # Create query parameters
            query_param = QueryParam(
                mode=query_mode,
                only_need_context=True,
                top_k=result_top_k
            )

            logger.info(f"| 📊 Querying with mode={query_mode}, top_k={result_top_k}")

            # Execute the query
            content = await self.rag.aquery(query, param=query_param)

            if not content:
                no_result_content = "## Retrieval Results\n\nNo relevant documents found for the query.\n\n"
                await report.add_item(no_result_content)
                
                if file_path:
                    final_report_content = await report.complete(file_path)
                    logger.info(f"✅ Retrieval report saved to: {file_path}")
                
                return ToolResponse(
                    success=True,
                    message="No relevant documents found for the query.",
                )

            logger.info(f"| ✅ Retrieved content ({len(str(content))} chars)")

            # Parse the JSON context
            try:
                # Extract JSON from the context string (it may be wrapped in markdown code blocks)
                content = str(content).strip()
                
                # Add retrieved content to report
                content_text = f"## Retrieved Content\n\n{content}\n\n"
                await report.add_item(content_text)
                logger.info(f"| ✅ Content added to report")
                
                # Generate final report
                if file_path:
                    final_report_content = await report.complete(file_path)
                    logger.info(f"✅ Retrieval report saved to: {file_path}")
                    
                    # Generate summary message
                    summary_message = f"Retrieved {result_top_k} documents for query: {query}\n\nReport saved to: {file_path}"
                    
                    return ToolResponse(
                        success=True,
                        message=summary_message,
                        extra=ToolExtra(
                            file_path=file_path,
                            data={
                                "query": query,
                                "content": content,
                                "top_k": result_top_k,
                                "query_mode": query_mode,
                                "report_length": len(final_report_content),
                                "file_path": file_path,
                                "items_count": len(report.items)
                            }
                        )
                    )
                else:
                    # Fallback if file_path is not available
                    summary_message = f"Retrieved {result_top_k} documents for query: {query}"
                    return ToolResponse(
                        success=True,
                        message=summary_message,
                        extra=ToolExtra(
                            data={
                                "query": query,
                                "content": content,
                                "top_k": result_top_k,
                                "query_mode": query_mode
                            }
                        )
                    )
            except json.JSONDecodeError as e:
                logger.warning(f"| ⚠️ Failed to parse context as JSON: {e}")
                
                # Add error content to report
                error_content = f"## Retrieval Results\n\n{str(content)}\n\n**Note**: Failed to parse context as JSON: {str(e)}\n\n"
                await report.add_item(error_content)
                
                if file_path:
                    final_report_content = await report.complete(file_path)
                    logger.info(f"✅ Retrieval report saved to: {file_path}")
                
                message = f"Retrieved content (parse warning): {str(content)}"
                
                return ToolResponse(
                    success=True,
                    message=message,
                    extra=ToolExtra(
                        file_path=file_path,
                        data={
                            "query": query,
                            "content": content,
                            "top_k": result_top_k,
                            "query_mode": query_mode,
                            "parse_error": str(e),
                            "file_path": file_path
                        }
                    )
                )

        except Exception as e:
            logger.error(f"| ❌ Error in retriever: {e}")
            import traceback
            return ToolResponse(
                success=False,
                message=f"Error during retrieval: {str(e)}\n{traceback.format_exc()}"
            )

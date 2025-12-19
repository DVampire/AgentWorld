"""Retriever Tool - A tool for retrieving ESG information from the LightRAG vector database."""

import os
import re
import json
from typing import Optional, List
import pandas as pd
from tabulate import tabulate
from pydantic import Field, ConfigDict

from dotenv import load_dotenv
load_dotenv(verbose=True)

from src.utils import assemble_project_path
from src.registry import TOOL
from src.logger import logger
from src.model import model_manager
from src.tool.types import Tool, ToolResponse
from src.message import HumanMessage, SystemMessage

from src.tool.esg_tools.lightrag import LightRAG, QueryParam
from src.tool.esg_tools.lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from src.tool.esg_tools.lightrag.kg.shared_storage import initialize_pipeline_status


_METADATA_PROMPT = """You are an ESG metadata extractor. You will be given text content retrieved from ESG reports. Please extract structured ESG facts in JSON format according to the following format.
Given the text, extract structured ESG facts in JSON format:

Fields:
- Aspect (Environmental/Social/Governance)
- KPI (e.g., CO2 Emissions, Energy Use, Waste Recycling)
- Topic (What the statement is about)
- Quantity: {"value": "...", "unit": "...", "year": "..."}
- Source: Use a short identifier

Example:
Input: 
[{"id": 1, "content": "In 2023, Company A reduced its Scope 1 emissions by 12% compared to 2022.", "file_path": A_2022_42}, {"id":2, "content":"The total Scope 1 emissions in 2022 were 5000 tons CO2.","file_path":A_2022_96}]
Output:
[{"Aspect": "Environmental", "KPI": "CO2 Emissions", "Topic": "Scope 1 Reduction", "Quantity": {"value": "12", "unit": "% reduction", "year": "2023"}}, {"Aspect": "Environmental", "KPI": "CO2 Emissions", "Topic": "Scope 1 Emissions", "Quantity": {"value": "value in 2022", "unit": "tons CO2", "year": "2022"}}]
"""


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
    enabled: bool = True

    # Configuration parameters
    model_name: str = Field(
        default="openrouter/gemini-3-flash-preview",
        description="The model to use for metadata extraction."
    )
    base_dir: str = Field(
        default=None,
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

    async def insert(
        self, 
        target_dir: str, 
        delete_old_data: bool = False, 
        file_type: str = "json"
    ) -> ToolResponse:
        """Insert documents from a directory into the LightRAG database.
        
        Args:
            target_dir: Directory containing documents to insert.
            delete_old_data: Whether to delete old data files before inserting.
            file_type: Type of files to process ('json' or 'csv').
            
        Returns:
            ToolResponse indicating success or failure.
        """
        # Check if OPENAI_API_KEY environment variable exists
        if not os.getenv("OPENAI_API_KEY"):
            logger.error(
                "OPENAI_API_KEY environment variable is not set. Please set this variable before running the program."
            )
            logger.info("You can set the environment variable by running:")
            logger.info("  export OPENAI_API_KEY='your-openai-api-key'")
            return ToolResponse(
                success=False,
                message="OPENAI_API_KEY environment variable is not set."
            )

        try:
            # Ensure RAG is initialized
            await self._ensure_rag_initialized()
            
            # Clear old data files
            files_to_delete = [
                "graph_chunk_entity_relation.graphml",
                "kv_store_doc_status.json",
                "kv_store_full_docs.json",
                "kv_store_text_chunks.json",
                "vdb_chunks.json",
                "vdb_entities.json",
                "vdb_relationships.json",
            ]

            if delete_old_data:
                for file in files_to_delete:
                    file_path = os.path.join(self.base_dir, file)
                    if os.path.exists(file_path):
                        os.remove(file_path)
                        logger.info(f"| Deleted old file: {file_path}")

            # Test embedding function
            test_text = ["This is a test string for embedding."]
            embedding = await self.rag.embedding_func(test_text)
            embedding_dim = embedding.shape[1]
            logger.info("| =======================")
            logger.info("| Test embedding function")
            logger.info("| =======================")
            logger.info(f"| Test text: {test_text}")
            logger.info(f"| Detected embedding dimension: {embedding_dim}")

            inserted_count = 0

            if file_type == "json":
                for dirpath, _, filenames in os.walk(target_dir):
                    for filename in filenames:
                        if filename.endswith(".json"):
                            file_path = os.path.join(dirpath, filename)
                            logger.info(f"| 📄 Processing JSON file: {file_path}")

                            with open(file_path, "r", encoding="utf-8") as f:
                                pages = json.load(f)
                                filename_ = filename.split(".json")[0]
                                company_name = filename_.split("_")[0]
                                year = filename_.split("_")[-1]
                                content_list, id_list = [], []

                                for page in pages:
                                    page_number = page.get("number", -1)
                                    elements = page.get("elements", [])
                                    content = ""
                                    doc_id = company_name + "_" + year + "_" + str(page_number)

                                    for element in elements:
                                        elem_type = element.get("type")
                                        content_i = element.get("content")

                                        if elem_type == "text":
                                            if content == "":
                                                content = content_i if content_i else ""
                                            else:
                                                content += "\n\n" + (content_i if content_i else "")
                                        elif elem_type == "table":
                                            table_content = (content_i.strip() if content_i else "") + "\n" + (element.get("caption", "").strip())
                                            if content == "":
                                                content = table_content
                                            else:
                                                content += table_content
                                        elif elem_type == "image":
                                            continue

                                    if content:
                                        id_list.append(doc_id)
                                        content_list.append(content)

                            if content_list:
                                await self.rag.ainsert(input=content_list, ids=id_list, file_paths=id_list)
                                inserted_count += len(content_list)
                                logger.info(f"| ✅ Inserted {len(content_list)} documents from {filename}")

            elif file_type == "csv":
                for root, dirs, files in os.walk(target_dir):
                    content_list, id_list = [], []
                    for file in files:
                        if file.endswith(".csv"):
                            file_path = os.path.join(root, file)
                            logger.info(f"| 📄 Processing CSV file: {file_path}")
                            
                            stock_name = os.path.splitext(file)[0].replace(".csv", "").split("_")[0]
                            caption_name = f"ESG Ratings of {stock_name} from 2015 to 2024"

                            df = pd.read_csv(file_path)
                            markdown_table = tabulate(df, headers="keys", tablefmt="github", showindex=False)
                            # Delete multiple spaces
                            lines = markdown_table.split("\n")
                            compact_lines = [re.sub(r'\s{2,}', ' ', line) for line in lines]
                            compact_markdown = "\n".join(compact_lines)

                            content = compact_markdown.strip() + " \n\n " + caption_name.strip()
                            doc_id = os.path.splitext(file)[0].replace(".csv", "")

                            content_list.append(content)
                            id_list.append(doc_id)

                    if content_list:
                        await self.rag.ainsert(input=content_list, ids=id_list, file_paths=id_list)
                        inserted_count += len(content_list)
                        logger.info(f"| ✅ Inserted {len(content_list)} CSV documents")

            logger.info(f"| 🎉 Total documents inserted: {inserted_count}")
            
            return ToolResponse(
                success=True,
                message=f"Successfully inserted {inserted_count} documents from {target_dir}",
                extra={"inserted_count": inserted_count, "file_type": file_type}
            )

        except Exception as e:
            logger.error(f"| ❌ An error occurred during insert: {e}")
            import traceback
            return ToolResponse(
                success=False,
                message=f"Error during insert: {str(e)}\n{traceback.format_exc()}"
            )

    async def finalize(self) -> None:
        """Finalize and cleanup the LightRAG storage."""
        if self.rag is not None:
            try:
                await self.rag.finalize_storages()
                logger.info(f"| 🧹 LightRAG storage finalized")
            except Exception as e:
                logger.warning(f"Error finalizing LightRAG storage: {e}")

    async def _extract_metadata(self, context: str) -> Optional[str]:
        """Extract structured ESG metadata from the context.
        
        Args:
            context: The raw context retrieved from the database.
            
        Returns:
            JSON string of extracted ESG metadata, or None if extraction fails.
        """
        try:
            messages = [
                SystemMessage(content=_METADATA_PROMPT),
                HumanMessage(content=context)
            ]

            response = await model_manager(
                model=self.model_name,
                messages=messages,
            )
            
            # Try to parse the response as JSON to validate it
            if response.message:
                try:
                    # Try to parse JSON from the response
                    metadata = json.loads(response.message)
                    return json.dumps(metadata, indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    # If not valid JSON, return the raw message
                    return response.message
            
            return None
            
        except Exception as e:
            logger.warning(f"Failed to extract ESG metadata: {e}")
            return None

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
            context = await self.rag.aquery(query, param=query_param)

            if not context:
                return ToolResponse(
                    success=True,
                    message="No relevant documents found for the query.",
                    extra={"query": query, "context": None, "metadata": None}
                )

            logger.info(f"| ✅ Retrieved context ({len(str(context))} chars)")

            # Build the result
            result_parts = []
            result_parts.append("---Document Chunks---")
            result_parts.append(str(context))
            
            final_result = "\n".join(result_parts)

            return ToolResponse(
                success=True,
                message=final_result,
                extra={
                    "query": query,
                    "context": str(context),
                    "top_k": result_top_k
                }
            )

        except Exception as e:
            logger.error(f"| ❌ Error in retriever: {e}")
            import traceback
            return ToolResponse(
                success=False,
                message=f"Error during retrieval: {str(e)}\n{traceback.format_exc()}"
            )

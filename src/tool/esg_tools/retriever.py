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
from src.utils import dedent

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

    async def _generate_report_with_citations(
        self, 
        query: str, 
        content: str
    ) -> str:
        """Generate a report with citations from retrieved content.
        
        Args:
            query (str): The original query
            content (str): The retrieved content
            
        Returns:
            Formatted report with citations
        """
        try:
            system_prompt = dedent("""You are an expert researcher and report writer. Your task is to create a comprehensive, well-structured report based on retrieved content.

                IMPORTANT REQUIREMENTS:
                1. **Preserve Original Content**: Keep the retrieved content as much as possible. Do not modify facts, numbers, or specific details.
                2. **Organize Logically**: Structure the content in a clear, logical manner with appropriate sections and headings.
                3. **Add Citations**: Throughout the report, mark where information comes from using citation numbers [1], [2], [3], etc.
                4. **Citation Format**: 
                - In the main content, use [1], [2], [3] to reference sources
                - At the end, provide a "References" section listing all citations in format:
                    [1] file_path
                    [2] file_path
                    etc.
                5. **Report Structure**:
                - Start with a brief introduction addressing the query
                - Organize content into logical sections
                - Use markdown formatting (headings, lists, etc.)
                - End with a References section

                Your goal is to create a clear, well-organized report that preserves the original information while making it easy to read and understand.""")
            
            user_prompt = dedent(f"""Based on the following query and retrieved document chunks, generate a comprehensive report.

                Query: {query}

                Retrieved Documents:
                {content}

                Please generate a well-structured report that:
                - Directly addresses the query
                - Preserves all important information from the content (keep facts, numbers, and details unchanged)
                - Organizes content logically with clear sections and headings
                - Includes citation markers [1], [2], [3], etc. throughout the text where information comes from the content
                - Ends with a References section listing all citations in the format:
                [1] file_path
                [2] file_path
                etc.

                IMPORTANT: Use citation numbers [1], [2], [3] etc. corresponding to the content in the order they appear above.

                Report:""")
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            logger.info(f"| 🤖 Generating report with citations using {self.model_name}")
            response = await model_manager(
                model=self.model_name,
                messages=messages
            )
            
            report = response.message.strip()
            logger.info(f"| ✅ Generated report ({len(report)} chars)")
            
            return report
            
        except Exception as e:
            logger.error(f"| ❌ Error generating report: {e}")
            return content

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
                return ToolResponse(
                    success=True,
                    message="No relevant documents found for the query.",
                    extra={"query": query, "context": None, "metadata": None}
                )

            logger.info(f"| ✅ Retrieved content ({len(str(content))} chars)")

            # Parse the JSON context
            try:
                # Extract JSON from the context string (it may be wrapped in markdown code blocks)
                content = str(content).strip()
                
                # Generate report with citations using LLM
                report = await self._generate_report_with_citations(query, content)
                
                return ToolResponse(
                    success=True,
                    message=report,
                    extra={
                        "query": query,
                        "content": str(content) ,
                        "top_k": result_top_k
                    }
                )
            except json.JSONDecodeError as e:
                logger.warning(f"| ⚠️ Failed to parse context as JSON: {e}")
                
                return ToolResponse(
                    success=True,
                    message=str(content),
                    extra={
                        "query": query,
                        "content": content,
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

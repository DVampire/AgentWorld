from typing import Any, Dict, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool as LangchainBaseTool

class BaseTool(LangchainBaseTool):
    """Base tool for the Tool Context Protocol."""
    name: str = Field(description="The name of the tool.")
    type: str = Field(description="The type of the tool.")
    description: str = Field(description="The description of the tool.")
    args_schema: Type[BaseModel] = Field(description="The args schema of the tool.")
    metadata: Dict[str, Any] = Field(description="The metadata of the tool.")
        
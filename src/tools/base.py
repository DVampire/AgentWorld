from typing import Any, Optional, List, Dict
from pydantic import BaseModel, Field

class ToolResponse(BaseModel):
    content: str = Field(description="The content of the tool response.")
    extra: Optional[Dict[str, Any]] = Field(description="The extra data of the tool response.")
    
    def __str__(self) -> str:
        return f"ToolResponse(content={self.content}, extra={self.extra})"
    
    def __repr__(self) -> str:
        return self.__str__()
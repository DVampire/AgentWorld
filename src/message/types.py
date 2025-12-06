from __future__ import annotations
from typing import Any, Dict, List, Literal, Union
from pydantic import BaseModel, Field, ConfigDict

class Message(BaseModel):
    """Base message model implemented with Pydantic."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: Union[str, List[Dict[str, Any]]] = Field(
        description="Message payload, either plain text or structured content blocks."
    )
    role: Literal["user", "assistant", "system"] = Field(
        description="Role associated with this message."
    )


class HumanMessage(Message):
    """User-authored message."""
    def __init__(self, content: Union[str, List[Dict]], **data):
        super().__init__(content=content, role="user", **data)


class AIMessage(Message):
    """Assistant-authored message."""
    def __init__(self, content: Union[str, List[Dict]], **data):
        super().__init__(content=content, role="assistant", **data)


class SystemMessage(Message):
    """System directive message."""
    def __init__(self, content: Union[str, List[Dict]], **data):
        super().__init__(content=content, role="system", **data)
from pydantic import BaseModel, Field
from src.utils import generate_unique_id

class SessionContext(BaseModel):
    id: str = Field(default_factory=lambda: generate_unique_id("session"), description="The unique identifier for the session.")
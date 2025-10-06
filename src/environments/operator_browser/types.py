from pydantic import BaseModel, Field
from typing import List

from src.environments.protocol.types import ActionResult

class ClickRequest(BaseModel):
    type: str = Field(description="The type of the action", default="click")
    x: int = Field(description="The x coordinate of the click")
    y: int = Field(description="The y coordinate of the click")
    button: str = Field(description="The button to click", default="left")
    

class ClickResult(ActionResult):
    pass
    
class DoubleClickRequest(BaseModel):
    type: str = Field(description="The type of the action", default="double_click")
    x: int = Field(description="The x coordinate of the double click")
    y: int = Field(description="The y coordinate of the double click")

class DoubleClickResult(ActionResult):
    pass
    
class ScrollRequest(BaseModel):
    type: str = Field(description="The type of the action", default="scroll")
    x: int = Field(description="The x coordinate of the scroll")
    y: int = Field(description="The y coordinate of the scroll")
    scroll_x: int = Field(description="The x coordinate of the scroll")
    scroll_y: int = Field(description="The y coordinate of the scroll")    

class ScrollResult(ActionResult):
    pass
    
class TypeRequest(BaseModel):
    type: str = Field(description="The type of the action", default="type")
    text: str = Field(description="The text to type")
    
class TypeResult(ActionResult):
    pass
    
class WaitRequest(BaseModel):
    type: str = Field(description="The type of the action", default="wait")
    ms: int = Field(description="The number of milliseconds to wait")
    
class WaitResult(ActionResult):
    pass
    
class MoveRequest(BaseModel):
    type: str = Field(description="The type of the action", default="move")
    x: int = Field(description="The x coordinate of the move")
    y: int = Field(description="The y coordinate of the move")
    
class MoveResult(ActionResult):
    pass
    
class KeypressRequest(BaseModel):
    type: str = Field(description="The type of the action", default="keypress")
    keys: List[str] = Field(description="The keys to press, e.g., ['CTRL', 'C'] or ['ENTER']")
    
class KeypressResult(ActionResult):
    pass
    
class DragRequest(BaseModel):
    type: str = Field(description="The type of the action", default="drag")
    path: List[List[int]] = Field(description="The path to drag, e.g., [[x1, y1], [x2, y2]]")
    
class DragResult(ActionResult):
    pass
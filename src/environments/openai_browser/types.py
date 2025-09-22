from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class ClickRequest(BaseModel):
    type: str = Field(description="The type of the action", default="click")
    x: int = Field(description="The x coordinate of the click")
    y: int = Field(description="The y coordinate of the click")
    button: str = Field(description="The button to click", default="left")
    
class ClickResult(BaseModel):
    success: bool = Field(description="Whether the click was successful")
    message: str = Field(description="The message of the click")
    
class DoubleClickRequest(BaseModel):
    type: str = Field(description="The type of the action", default="double_click")
    x: int = Field(description="The x coordinate of the double click")
    y: int = Field(description="The y coordinate of the double click")

class DoubleClickResult(BaseModel):
    success: bool = Field(description="Whether the double click was successful")
    message: str = Field(description="The message of the double click")
    
class ScrollRequest(BaseModel):
    type: str = Field(description="The type of the action", default="scroll")
    x: int = Field(description="The x coordinate of the scroll")
    y: int = Field(description="The y coordinate of the scroll")
    scroll_x: int = Field(description="The x coordinate of the scroll")
    scroll_y: int = Field(description="The y coordinate of the scroll")    

class ScrollResult(BaseModel):
    success: bool = Field(description="Whether the scroll was successful")
    message: str = Field(description="The message of the scroll")
    
class TypeRequest(BaseModel):
    type: str = Field(description="The type of the action", default="type")
    text: str = Field(description="The text to type")
    
class TypeResult(BaseModel):
    success: bool = Field(description="Whether the type was successful")
    message: str = Field(description="The message of the type")
    
class WaitRequest(BaseModel):
    type: str = Field(description="The type of the action", default="wait")
    ms: int = Field(description="The number of milliseconds to wait")
    
class WaitResult(BaseModel):
    success: bool = Field(description="Whether the wait was successful")
    message: str = Field(description="The message of the wait")
    
class MoveRequest(BaseModel):
    type: str = Field(description="The type of the action", default="move")
    x: int = Field(description="The x coordinate of the move")
    y: int = Field(description="The y coordinate of the move")
    
class MoveResult(BaseModel):
    success: bool = Field(description="Whether the move was successful")
    message: str = Field(description="The message of the move")
    
class KeypressRequest(BaseModel):
    type: str = Field(description="The type of the action", default="keypress")
    keys: List[str] = Field(description="The keys to press, e.g., ['CTRL', 'C'] or ['ENTER']")
    
class KeypressResult(BaseModel):
    success: bool = Field(description="Whether the keypress was successful")
    message: str = Field(description="The message of the keypress")
    
class DragRequest(BaseModel):
    type: str = Field(description="The type of the action", default="drag")
    path: List[List[int]] = Field(description="The path to drag, e.g., [[x1, y1], [x2, y2]]")
    
class DragResult(BaseModel):
    success: bool = Field(description="Whether the drag was successful")
    message: str = Field(description="The message of the drag")

class NavigateRequest(BaseModel):
    url: str = Field(description="URL to navigate to")

class NavigateResult(BaseModel):
    success: bool = Field(description="Whether the navigation was successful")
    url: str = Field(description="Current URL after navigation")
    title: str = Field(description="Page title after navigation")
    message: str = Field(description="The message of the navigation")

class ScreenshotRequest(BaseModel):
    full_page: bool = Field(description="Whether to capture the full page", default=False)
    save_path: Optional[str] = Field(description="Optional path to save screenshot file", default=None)

class ScreenshotResult(BaseModel):
    success: bool = Field(description="Whether the screenshot was successful")
    screenshot: Optional[str] = Field(description="Base64 encoded screenshot", default=None)
    save_path: Optional[str] = Field(description="Path where screenshot was saved", default=None)
    message: str = Field(description="The message of the screenshot")

class PageContentResult(BaseModel):
    success: bool = Field(description="Whether getting page content was successful")
    title: str = Field(description="Page title")
    url: str = Field(description="Current URL")
    content: Optional[Dict[str, Any]] = Field(description="Page content information", default=None)
    error: Optional[str] = Field(description="Error message if failed", default=None)
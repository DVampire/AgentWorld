"""Playwright data types."""

from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field

# Request types for service layer

class SearchGoogleRequest(BaseModel):
    """Request for searching Google."""
    query: str = Field(..., description="Search query like humans search in Google")


class GoToUrlRequest(BaseModel):
    """Request for navigating to URL."""
    url: str = Field(..., description="URL to navigate to")
    new_tab: bool = Field(False, description="Whether to open in new tab")


class GoBackRequest(BaseModel):
    """Request for going back in browser history."""
    pass


class WaitRequest(BaseModel):
    """Request for waiting."""
    seconds: int = Field(3, ge=1, le=30, description="Seconds to wait (max 30)")


class ClickElementRequest(BaseModel):
    """Request for clicking element by index."""
    index: int = Field(..., ge=0, description="Element index from browser state")
    while_holding_ctrl: bool = Field(False, description="Whether to hold Ctrl while clicking")


class InputTextRequest(BaseModel):
    """Request for inputting text."""
    index: int = Field(..., ge=0, description="Element index to input text into")
    text: str = Field(..., description="Text to input")
    clear: bool = Field(True, description="Whether to clear existing text first")


class ScrollRequest(BaseModel):
    """Request for scrolling."""
    down: bool = Field(True, description="Whether to scroll down")
    num_pages: float = Field(1.0, ge=0.1, le=100.0, description="Number of pages to scroll")
    frame_element_index: Optional[int] = Field(None, description="Frame element index if scrolling in frame")


class SendKeysRequest(BaseModel):
    """Request for sending special keys."""
    keys: str = Field(..., description="Special keys to send (e.g., Escape, Control+o)")


class ScrollToTextRequest(BaseModel):
    """Request for scrolling to text."""
    text: str = Field(..., description="Text to scroll to")


class GetDropdownOptionsRequest(BaseModel):
    """Request for getting dropdown options."""
    index: int = Field(..., ge=0, description="Dropdown element index")


class SelectDropdownOptionRequest(BaseModel):
    """Request for selecting dropdown option."""
    index: int = Field(..., ge=0, description="Dropdown element index")
    text: str = Field(..., description="Option text to select")


class UploadFileRequest(BaseModel):
    """Request for uploading file."""
    index: int = Field(..., ge=0, description="Element index to upload file to")
    path: str = Field(..., description="File path to upload")
    available_file_paths: Optional[List[str]] = Field(None, description="Available file paths")


class SwitchTabRequest(BaseModel):
    """Request for switching tab."""
    tab_id: str = Field(..., description="Tab ID to switch to")


class CloseTabRequest(BaseModel):
    """Request for closing tab."""
    tab_id: str = Field(..., description="Tab ID to close")


class ExtractStructuredDataRequest(BaseModel):
    """Request for extracting structured data."""
    schema: Dict[str, Any] = Field(..., description="Schema for data extraction")
    timeout: int = Field(30, ge=1, le=300, description="Timeout in seconds")


class ExecuteJsRequest(BaseModel):
    """Request for executing JavaScript."""
    code: str = Field(..., description="JavaScript code to execute")

class ScreenshotRequest(BaseModel):
    """Request for taking screenshot."""
    highlight_elements: bool = Field(False, description="Whether to highlight elements")
    filter_highlight_ids: Optional[List[int]] = Field(None, description="Element IDs to highlight")


class StoreScreenshotRequest(BaseModel):
    """Request for storing screenshot."""
    step_number: int = Field(..., ge=1, description="Step number for screenshot")
    highlight_elements: bool = Field(False, description="Whether to highlight elements")


class GetScreenshotRequest(BaseModel):
    """Request for getting screenshot from disk."""
    screenshot_path: str = Field(..., description="Path to screenshot file")


class BrowserStateRequest(BaseModel):
    """Request for getting browser state."""
    include_screenshot: bool = Field(False, description="Whether to include screenshot")

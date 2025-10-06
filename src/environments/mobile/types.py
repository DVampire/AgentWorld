"""Mobile device types and request/response models."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field

from src.environments.protocol.types import ActionResult

class MobileDeviceInfo(BaseModel):
    """Mobile device information."""
    device_id: str = Field(description="Device ID or serial number")
    screen_width: int = Field(description="Screen width in pixels")
    screen_height: int = Field(description="Screen height in pixels")
    screen_density: float = Field(description="Screen density (DPI)")
    is_connected: bool = Field(description="Whether device is connected")


class TapRequest(BaseModel):
    """Tap action request."""
    x: int = Field(description="X coordinate")
    y: int = Field(description="Y coordinate")
    duration: Optional[int] = Field(default=None, description="Tap duration in milliseconds")


class SwipeRequest(BaseModel):
    """Swipe action request."""
    start_x: int = Field(description="Start X coordinate")
    start_y: int = Field(description="Start Y coordinate")
    end_x: int = Field(description="End X coordinate")
    end_y: int = Field(description="End Y coordinate")
    duration: int = Field(default=300, description="Swipe duration in milliseconds")


class PressRequest(BaseModel):
    """Long press action request."""
    x: int = Field(description="X coordinate")
    y: int = Field(description="Y coordinate")
    duration: int = Field(default=1000, description="Press duration in milliseconds")


class TypeTextRequest(BaseModel):
    """Type text request."""
    text: str = Field(description="Text to input")


class KeyEventRequest(BaseModel):
    """Key event request."""
    keycode: int = Field(description="Android keycode to press")


class ScreenshotRequest(BaseModel):
    """Screenshot request."""
    save_path: Optional[str] = Field(default=None, description="Path to save screenshot")


class SwipePathRequest(BaseModel):
    """Swipe along a path request."""
    path: List[List[int]] = Field(description="Path coordinates as [[x1, y1], [x2, y2], ...]")
    duration: int = Field(default=300, description="Total swipe duration in milliseconds")

class ScrollRequest(BaseModel):
    """Scroll request."""
    direction: str = Field(description="Scroll direction")
    distance: int = Field(default=500, description="Scroll distance in pixels")


class TapResult(ActionResult):
    """Tap action result."""
    pass


class SwipeResult(ActionResult):
    """Swipe action result."""
    pass


class PressResult(ActionResult):
    """Press action result."""
    pass


class TypeTextResult(ActionResult):
    """Type text result."""
    pass


class KeyEventResult(ActionResult):
    """Key event result."""
    pass


class ScreenshotResult(ActionResult):
    """Screenshot result."""
    screenshot_path: str = Field(description="Path to saved screenshot")


class SwipePathResult(ActionResult):
    """Swipe path result."""
    pass


class ScrollResult(ActionResult):
    """Scroll result."""
    pass

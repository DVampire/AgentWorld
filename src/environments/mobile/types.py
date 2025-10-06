"""Mobile device types and request/response models."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


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


# Response types
class MobileActionResult(BaseModel):
    """Base result for mobile actions."""
    success: bool = Field(description="Whether the action succeeded")
    message: str = Field(description="Result message")
    screenshot_path: Optional[str] = Field(default=None, description="Path to screenshot if taken")


class TapResult(MobileActionResult):
    """Tap action result."""
    pass


class SwipeResult(MobileActionResult):
    """Swipe action result."""
    pass


class PressResult(MobileActionResult):
    """Press action result."""
    pass


class TypeTextResult(MobileActionResult):
    """Type text result."""
    pass


class KeyEventResult(MobileActionResult):
    """Key event result."""
    pass


class ScreenshotResult(MobileActionResult):
    """Screenshot result."""
    screenshot_path: str = Field(description="Path to saved screenshot")


class SwipePathResult(MobileActionResult):
    """Swipe path result."""
    pass


class ScrollResult(MobileActionResult):
    """Scroll result."""
    pass


class MobileDeviceState(BaseModel):
    """Mobile device state."""
    device_info: MobileDeviceInfo = Field(description="Device information")
    screenshot_path: Optional[str] = Field(default=None, description="Current screenshot path")
    is_recording: bool = Field(default=False, description="Whether video is being recorded")
    recording_path: Optional[str] = Field(default=None, description="Video recording path")

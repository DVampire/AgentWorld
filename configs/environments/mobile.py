"""Mobile environment configuration."""

from pydantic import BaseModel, Field
from typing import Optional


class MobileConfig(BaseModel):
    """Mobile environment configuration."""
    
    # Device settings
    device_id: Optional[str] = Field(default=None, description="Device ID or serial number")
    video_save_path: Optional[str] = Field(default="./workdir/mobile_agent/videos", description="Video save directory")
    video_save_name: str = Field(default="mobile_record", description="Video file base name")
    
    # Recording settings
    fps: int = Field(default=30, description="Frame rate for screen capture")
    bitrate: int = Field(default=50000000, description="Video bitrate (50Mbps)")
    chunk_duration: int = Field(default=60, description="Video chunk duration in seconds")
    
    # Work directory
    base_dir: str = Field(default="./workdir/mobile_agent", description="Base directory for mobile agent")
    
    # Screenshot settings
    screenshot_format: str = Field(default="png", description="Screenshot format")
    screenshot_quality: int = Field(default=95, description="Screenshot quality (1-100)")
    
    # Touch settings
    default_tap_duration: int = Field(default=100, description="Default tap duration in ms")
    default_swipe_duration: int = Field(default=300, description="Default swipe duration in ms")
    default_long_press_duration: int = Field(default=1000, description="Default long press duration in ms")
    
    # Advanced settings
    enable_recording: bool = Field(default=True, description="Enable video recording")
    auto_screenshot: bool = Field(default=True, description="Take screenshot after each action")
    screenshot_delay: float = Field(default=0.5, description="Delay before taking screenshot (seconds)")


# Default configuration
DEFAULT_MOBILE_CONFIG = MobileConfig()

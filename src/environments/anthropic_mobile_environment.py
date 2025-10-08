"""Mobile Environment for AgentWorld - provides mobile device automation operations as an environment."""

from typing import Any, Dict, List, Union, Optional, Type, Literal
from langgraph.store.base import Op
from pydantic import BaseModel, Field, ConfigDict
from src.environments.mobile.service import MobileService
from src.logger import logger
from src.environments.protocol.server import ecp
from src.utils import dedent, ScreenshotService, encode_image_base64, decode_image_base64
from src.environments.protocol.types import ScreenshotInfo
from src.environments.mobile_environment import MobileEnvironment

ScrollDirection = Literal["up", "down", "left", "right"]

_ACTION_DESCRIPTION="""The operation to perform. 

Available operations:
1. type: Type a string of text on the keyboard. Supports any languages.
    - text: The text to type.
    - Examples: "你好!", "Hello World!", "こんにちは！"
2. left_click: Click the left mouse button at the specified (x, y) pixel coordinates on the screen. 
    - coordinate: The (x, y) pixel coordinates to click.
    - Examples: [100, 200]
3. scroll: Scroll the screen at the specified (x, y) pixel coordinates by a given number of wheel ticks in the specified direction. Do not use PageUp/PageDown to scroll.
    - scroll_direction: The direction to scroll.
    - scroll_amount: The amount to scroll.
4. wait: Wait for a specified amount of time (in seconds).
    - duration: The duration to wait (in seconds).

Note: Screenshots are automatically captured after each action - do not use screenshot action.
"""

@ecp.environment()
class AnthropicMobileEnvironment(MobileEnvironment):
    """Mobile Environment that provides mobile device automation operations as an environment interface."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="anthropic_mobile", description="The name of the Mobile environment.")
    type: str = Field(default="Anthropic Mobile", description="The type of the Mobile environment.")
    description: str = Field(default="Mobile device automation environment for Android device control", description="The description of the Mobile environment.")
    args_schema: Type[BaseModel] = Field(default=None, description="The args schema of the Mobile environment.")
    metadata: Dict[str, Any] = Field(default={
        "has_vision": True,
        "additional_rules": {
            "state": "The state of the mobile device including device info, screen size, and current screenshot.",
        }
    }, description="The metadata of the Mobile environment.")
    
    def __init__(
        self,
        base_dir: str = "./workdir/mobile_agent",
        device_id: Optional[str] = None,
        fps: int = 2,
        bitrate: int = 50000000,
        chunk_duration: int = 60,
        **kwargs
    ):
        """
        Initialize the Mobile Environment.
        
        Args:
            base_dir: Base directory for mobile agent work
            device_id: Target device ID (defaults to first connected device)
            fps: Frame rate for screen capture
            bitrate: Video bitrate for recording
            chunk_duration: Video chunk duration in seconds
        """
        super().__init__()
        
        self.base_dir = base_dir
        self.device_id = device_id
        self.fps = fps
        self.bitrate = bitrate
        self.chunk_duration = chunk_duration
        
        self.mobile_service = MobileService(
            base_dir=base_dir,
            device_id=device_id,
            fps=fps,
            bitrate=bitrate,
            chunk_duration=chunk_duration
        )
        
        # Initialize screenshot service
        self.step_number = 0
        
        self.screenshot: ScreenshotInfo = None
        self.previous_screenshot: ScreenshotInfo = None
        
        # Target window size
        self.target_window_width = 1024
        self.target_window_height = 768
        self.pad_color = (0, 0, 0)
        
        self.screenshot_service = ScreenshotService(
            base_dir=self.base_dir,
            adapt_window_size=True,
            target_window_width=self.target_window_width,
            target_window_height=self.target_window_height,
            pad_color=self.pad_color
        )
        
        
    async def initialize(self) -> None:
        """Initialize the mobile environment."""
        await self.mobile_service.start()
        logger.info(f"| 📱 Mobile Environment initialized at: {self.base_dir}")
        
    async def cleanup(self) -> None:
        """Cleanup the mobile environment."""
        await self.mobile_service.stop()
        logger.info("| 📱 Mobile Environment cleanup completed")
        
    @ecp.action(
        name = "computer",
        description = "Run operations on the mobile device.",
        type = "Anthropic Mobile Environment",
    )
    async def computer(self, 
                       action: str,
                       text: Optional[str] = None,
                       coordinate: Optional[List[int]] = None,
                       scroll_direction: Optional[ScrollDirection] = None,
                       scroll_amount: Optional[int] = None,
                       duration: Optional[int] = None,
                       key: Optional[str] = None,
                       ) -> str:
        """
        Perform a step of mobile device operation.
        
        Args:
            action (str): Action to perform
            text (Optional[str]): Text to type
            coordinate (Optional[List[int]]): Coordinate to click (x, y)
            scroll_direction (ScrollDirection): Direction to scroll
            scroll_amount (Optional[int]): Amount to scroll
            duration (Optional[int]): Duration to wait in seconds
            key (Optional[str]): Key to press (e.g. "a", "Return", "alt+Tab", "ctrl+s", "Up", "KP_0" (for the numeric keypad 0 key))
        
        Returns:
            str: Result message
        """
        
        try:
            if action == "left_click":
                x, y = coordinate
                return await self.tap(x, y)
            elif action == "scroll":
                return await self.scroll(scroll_direction, scroll_amount)
            elif action == "wait":
                return await self.wait(duration)
            elif action == "type":
                return await self.type_text(text)
            elif action == "screenshot":
                return f"Screenshots are automatically captured after each action. You DO NOT need to use this action. And the current screenshot path is: {self.screenshot.screenshot_path}."  
            else:
                return f"Invalid action: {action}"
        except Exception as e:
            logger.error(f"Error in step operation: {e}")
            return f"Step failed: {e}"
    
    async def get_state(self) -> Dict[str, Any]:
        """Get the current state of the mobile device."""
        try:
            mobile_device_state = await self.mobile_service.get_state()
            device_info = mobile_device_state.get("device_info", {})
            
            state = dedent(f"""
                <info>
                Screen Width: {self.target_window_width}
                Screen Height: {self.target_window_height}
                Screen Density: {device_info["screen_density"]}
                Is Connected: {device_info["is_connected"]}
                </info>
            """)
            
            # Transform screenshot
            screenshot = decode_image_base64(mobile_device_state["screenshot"])
            source_width, source_height = screenshot.size
            transformed_screenshot = self.transform_screenshot(screenshot)
            screenshot_filename = f'step_{self.step_number:04d}_state.png'
            screenshot_path = await self.screenshot_service.store_screenshot(transformed_screenshot, self.step_number, screenshot_filename)
            screenshot_description = "A screenshot of the device at current step."
            
            self.screenshot = ScreenshotInfo(
                transformed=True,
                screenshot=encode_image_base64(transformed_screenshot),
                screenshot_path=screenshot_path,
                screenshot_description=screenshot_description,
                transform_info={
                    "source_width": source_width,
                    "source_height": source_height,
                    "target_width": self.target_window_width,
                    "target_height": self.target_window_height,
                    "pad_color": self.pad_color,
                }
            )
            
            if not self.previous_screenshot:
                self.previous_screenshot = self.screenshot
            
            screenshots = [
                self.previous_screenshot,
                self.screenshot,
            ]

            extra = {
                "screenshots": screenshots,
            }
            
            return {
                "state": state,
                "extra": extra,
            }
        except Exception as e:
            logger.error(f"Error getting mobile device state: {e}")
            return {
                "state": "Failed to get mobile device state",
                "extra": {
                    "error": str(e),
                },
            }
    

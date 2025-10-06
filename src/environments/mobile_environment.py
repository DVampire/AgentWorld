"""Mobile Environment for AgentWorld - provides mobile device automation operations as an environment."""

from typing import Any, Dict, List, Union, Optional, Type
from pydantic import BaseModel, Field, ConfigDict
import shutil
import os

from src.environments.mobile.service import MobileService
from src.environments.mobile.types import (
    TapRequest,
    SwipeRequest,
    PressRequest,
    TypeTextRequest,
    KeyEventRequest,
    SwipePathRequest,
    ScrollRequest,
)
from src.logger import logger
from src.environments.protocol.server import ecp
from src.environments.protocol.environment import BaseEnvironment
from src.utils import dedent, ScreenshotService, encode_image_base64
from src.environments.protocol.types import ScreenshotInfo, ActionResult

@ecp.environment()
class MobileEnvironment(BaseEnvironment):
    """Mobile Environment that provides mobile device automation operations as an environment interface."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="mobile", description="The name of the Mobile environment.")
    type: str = Field(default="Mobile", description="The type of the Mobile environment.")
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
        
        self.screenshot_service = ScreenshotService(base_dir=base_dir)
        
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
        
        self.screenshot_service = ScreenshotService(base_dir=self.base_dir)
        
        
    async def initialize(self) -> None:
        """Initialize the mobile environment."""
        await self.mobile_service.start()
        logger.info(f"| 📱 Mobile Environment initialized at: {self.base_dir}")
        
    async def cleanup(self) -> None:
        """Cleanup the mobile environment."""
        await self.mobile_service.stop()
        logger.info("| 📱 Mobile Environment cleanup completed")
    
    # ==================== BASIC OPERATIONS ====================
    @ecp.action(
        name="tap",
        description="Tap at specified coordinates on the mobile device",
        type="Mobile Environment",
    )
    async def tap(self, x: int, y: int) -> str:
        """
        Tap at specified coordinates on the mobile device.
        
        Args:
            x (int): X coordinate for tap
            y (int): Y coordinate for tap
            
        Returns:
            TapResult: Result of the tap operation
        """
        try:
            request = TapRequest(x=x, y=y)
            
            # Draw a cursor on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_tap.png'
            screenshot_path = await self.screenshot_service.store_screenshot(self.screenshot.screenshot,
                                                                             self.step_number,
                                                                             screenshot_filename)
            screenshot_path = await self.screenshot_service.draw_cursor(screenshot_path, x, y) # draw a cursor on the screenshot
            screenshot = encode_image_base64(screenshot_path)
            screenshot_description = f"Action: Tap at ({x}, {y})"
            self.previous_screenshot = ScreenshotInfo(screenshot=screenshot,
                                                      screenshot_path=screenshot_path,
                                                      screenshot_description=screenshot_description)
            
            # Perform tap
            result = await self.mobile_service.tap(request)
            
            self.step_number += 1
            
            return result.message
            
        except Exception as e:
            logger.error(f"Error in tap operation: {e}")
            return f"Tap failed: {e}"
    
    @ecp.action(
        name="swipe",
        description="Swipe at specified coordinates on the mobile device",
        type="Mobile Environment",
    )
    async def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: int = 300) -> str:
        """
        Swipe gesture from start to end coordinates.
        
        Args:
            start_x (int): Start X coordinate
            start_y (int): Start Y coordinate
            end_x (int): End X coordinate
            end_y (int): End Y coordinate
            duration (int): Swipe duration in milliseconds
            
        Returns:
            SwipeResult: Result of the swipe operation
        """
        try:
            request = SwipeRequest(
                start_x=start_x,
                start_y=start_y,
                end_x=end_x,
                end_y=end_y,
                duration=duration
            )
            
            # Draw a path on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_swipe.png'
            screenshot_path = await self.screenshot_service.store_screenshot(self.screenshot.screenshot, self.step_number, screenshot_filename)
            screenshot_path = await self.screenshot_service.draw_path(screenshot_path, [[start_x, start_y], [end_x, end_y]]) # draw a path on the screenshot
            screenshot = encode_image_base64(screenshot_path)
            screenshot_description = f"Action: Swipe from ({start_x}, {start_y}) to ({end_x}, {end_y})"
            self.previous_screenshot = ScreenshotInfo(screenshot=screenshot,
                                                      screenshot_path=screenshot_path,
                                                      screenshot_description=screenshot_description)
            
            # Perform swipe
            result = await self.mobile_service.swipe(request)
            
            self.step_number += 1
            
            return result.message
            
        except Exception as e:
            logger.error(f"Error in swipe operation: {e}")
            return f"Swipe failed: {e}"
    
    @ecp.action(
        name="press",
        description="Long press at specified coordinates on the mobile device",
        type="Mobile Environment",
    )
    async def press(self, x: int, y: int, duration: int = 1000) -> str:
        """
        Long press at specified coordinates.
        
        Args:
            x (int): X coordinate for press
            y (int): Y coordinate for press
            duration (int): Press duration in milliseconds
            
        Returns:
            PressResult: Result of the press operation
        """
        try:
            request = PressRequest(x=x, y=y, duration=duration)

            # Draw a cursor on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_press.png'
            screenshot_path = await self.screenshot_service.store_screenshot(self.screenshot.screenshot, self.step_number, screenshot_filename)
            screenshot_path = await self.screenshot_service.draw_cursor(screenshot_path, x, y) # draw a cursor on the screenshot
            screenshot = encode_image_base64(screenshot_path)
            screenshot_description = f"Action: Press at ({x}, {y}) for {duration}ms"
            self.previous_screenshot = ScreenshotInfo(screenshot=screenshot,
                                                      screenshot_path=screenshot_path,
                                                      screenshot_description=screenshot_description)
            
            # Perform press
            result = await self.mobile_service.press(request)
            
            self.step_number += 1
            
            return result.message
        
        except Exception as e:
            logger.error(f"Error in press operation: {e}")
            return f"Press failed: {e}"
    
    @ecp.action(
        name="type_text",
        description="Type text at the current cursor position on the mobile device",
        type="Mobile Environment",
    )
    async def type_text(self, text: str) -> str:
        """
        Type text on the mobile device.
        
        Args:
            text (str): Text to input
            
        Returns:
            TypeTextResult: Result of the type operation
        """
        try:
            request = TypeTextRequest(text=text)

            # DO NOT draw anything on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_type_text.png'
            screenshot_path = await self.screenshot_service.store_screenshot(self.screenshot.screenshot, self.step_number, screenshot_filename)
            screenshot = encode_image_base64(screenshot_path)
            screenshot_description = f"Action: Type text: {text}"
            self.previous_screenshot = ScreenshotInfo(screenshot=screenshot,
                                                      screenshot_path=screenshot_path,
                                                      screenshot_description=screenshot_description)
            
            # Perform type text
            result = await self.mobile_service.type_text(request)
            
            self.step_number += 1
            
            return result.message
        
        except Exception as e:
            logger.error(f"Error in type operation: {e}")
            return f"Type failed: {e}"
    
    @ecp.action(
        name="key_event",
        description="Press a key on the mobile device",
        type="Mobile Environment",
    )
    async def key_event(self, keycode: int) -> str:
        """
        Press a key on the mobile device.
        
        Args:
            keycode (int): Android keycode to press
            
        Returns:
            KeyEventResult: Result of the key event operation
        """
        try:
            request = KeyEventRequest(keycode=keycode)

            # DO NOT draw anything on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_key_event.png'
            screenshot_path = await self.screenshot_service.store_screenshot(self.screenshot.screenshot, self.step_number, screenshot_filename)
            screenshot = encode_image_base64(screenshot_path)
            screenshot_description = f"Action: Key event: {keycode}"
            self.previous_screenshot = ScreenshotInfo(screenshot=screenshot,
                                                      screenshot_path=screenshot_path,
                                                      screenshot_description=screenshot_description)
            
            # Perform key event
            result = await self.mobile_service.key_event(request)
            
            self.step_number += 1
            
            return result.message
        
        except Exception as e:
            logger.error(f"Error in key event operation: {e}")
            return f"Key event failed: {e}"
    
    # ==================== ADVANCED OPERATIONS ====================
    
    @ecp.action(
        name="swipe_path",
        description="Swipe along a path of coordinates on the mobile device",
        type="Mobile Environment",
    )
    async def swipe_path(self, path: List[List[int]], duration: int = 300) -> str:
        """
        Swipe along a path of coordinates.
        
        Args:
            path (List[List[int]]): List of [x, y] coordinates
            duration (int): Total swipe duration in milliseconds
            
        Returns:
            str: Result of the swipe path operation
        """
        try:
            request = SwipePathRequest(path=path, duration=duration)

            # Draw a path on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_swipe_path.png'
            screenshot_path = await self.screenshot_service.store_screenshot(self.screenshot.screenshot, self.step_number, screenshot_filename)
            screenshot_path = await self.screenshot_service.draw_path(screenshot_path, path) # draw a path on the screenshot
            screenshot = encode_image_base64(screenshot_path)
            screenshot_description = f"Action: Swipe path with {len(path)} points"
            self.previous_screenshot = ScreenshotInfo(screenshot=screenshot,
                                                      screenshot_path=screenshot_path,
                                                      screenshot_description=screenshot_description)
            
            # Perform swipe path
            result = await self.mobile_service.swipe_path(request)
            
            self.step_number += 1
            
            return result.message
        
        except Exception as e:
            logger.error(f"Error in swipe path operation: {e}")
            return f"Swipe path failed: {e}"
    
    # ==================== SCROLL OPERATIONS ====================
    
    @ecp.action(
        name="scroll",
        description="Scroll on the mobile device in specified direction",
        type="Mobile Environment",
    )
    async def scroll(self, direction: str, distance: int = 500) -> str:
        """
        Scroll on the mobile device in specified direction.
        
        Args:
            direction (str) : Scroll direction ("up", "down", "left", "right")
            distance: Scroll distance in pixels
            
        Returns:
            str: Result message
        """
        try:
            request = ScrollRequest(direction=direction, distance=distance)
            
            # DO NOT draw anything on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_scroll.png'
            screenshot_path = await self.screenshot_service.store_screenshot(self.screenshot.screenshot, self.step_number, screenshot_filename)
            screenshot = encode_image_base64(screenshot_path)
            screenshot_description = f"Action: Scroll {direction} by {distance} pixels"
            self.previous_screenshot = ScreenshotInfo(screenshot=screenshot,
                                                      screenshot_path=screenshot_path,
                                                      screenshot_description=screenshot_description)
            
            # Perform scroll
            result = await self.mobile_service.scroll(request)
            
            self.step_number += 1
            
            return result.message
        
        except Exception as e:
            logger.error(f"Error in scroll operation: {e}")
            return f"Scroll failed: {e}"
    
    async def get_state(self) -> Dict[str, Any]:
        """Get the current state of the mobile device."""
        try:
            mobile_device_state = await self.mobile_service.get_state()
            device_info = mobile_device_state.get("device_info", {})
            
            state = dedent(f"""
                <info>
                Screen Width: {device_info["screen_width"]}
                Screen Height: {device_info["screen_height"]}
                Screen Density: {device_info["screen_density"]}
                Is Connected: {device_info["is_connected"]}
                </info>
            """)

            screenshot_filename = f'step_{self.step_number:04d}_state.png'
            screenshot_path = await self.screenshot_service.store_screenshot(mobile_device_state["screenshot"], self.step_number, screenshot_filename)
            screenshot_description = "A screenshot of the device at current step."
            
            self.screenshot = ScreenshotInfo(
                screenshot=mobile_device_state["screenshot"],
                screenshot_path=screenshot_path,
                screenshot_description=screenshot_description
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
    

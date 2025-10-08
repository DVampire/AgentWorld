"""Mobile Environment for AgentWorld - provides mobile device automation operations as an environment."""

from typing import Any, Dict, List, Union, Optional, Type
from langgraph.store.base import Op
from pydantic import BaseModel, Field, ConfigDict
import shutil
import asyncio
import os
from PIL import Image

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
from src.utils import dedent, ScreenshotService, encode_image_base64, decode_image_base64
from src.environments.protocol.types import ScreenshotInfo

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
            
            # Draw a cursor on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_tap.png'
            screenshot = decode_image_base64(self.screenshot.screenshot)
            screenshot = await self.screenshot_service.draw_cursor(screenshot, x, y)
            screenshot_description = f"Action: Tap at ({x}, {y})"
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot,
                                                                             self.step_number,
                                                                             screenshot_filename)
            self.previous_screenshot = ScreenshotInfo(
                transformed=self.screenshot.transformed,
                screenshot=encode_image_base64(screenshot),
                screenshot_path=screenshot_path,
                screenshot_description=screenshot_description,
                transform_info=self.screenshot.transform_info
            )
            
            # inverse transform the x and y
            source_width, source_height = self.screenshot.transform_info["source_width"], self.screenshot.transform_info["source_height"]
            inverse_x, inverse_y = self.screenshot_service.inverse_transform_point(x, 
                                                                   y,
                                                                   source_width,
                                                                   source_height,
                                                                   self.target_window_width,
                                                                   self.target_window_height
                                                                   )
            
            # Perform tap
            request = TapRequest(x=inverse_x, y=inverse_y)
            await self.mobile_service.tap(request)
            
            self.step_number += 1
            
            return f"Tapped at ({x}, {y})"
            
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
            # Draw a path on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_swipe.png'
            screenshot = decode_image_base64(self.screenshot.screenshot)
            screenshot = await self.screenshot_service.draw_path(screenshot, [[start_x, start_y], [end_x, end_y]]) # draw a path on the screenshot
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot,
                                                                             self.step_number,
                                                                             screenshot_filename)
            screenshot_description = f"Action: Swipe from ({start_x}, {start_y}) to ({end_x}, {end_y})"
            self.previous_screenshot = ScreenshotInfo(
                transformed=self.screenshot.transformed,
                screenshot=encode_image_base64(screenshot),
                screenshot_path=screenshot_path,
                screenshot_description=screenshot_description,
                transform_info=self.screenshot.transform_info
            )
            
            # inverse transform the x and y
            source_width, source_height = self.screenshot.transform_info["source_width"], self.screenshot.transform_info["source_height"]
            inverse_start_x, inverse_start_y = self.screenshot_service.inverse_transform_point(start_x, 
                                                                               start_y,
                                                                               source_width,
                                                                               source_height,
                                                                               self.target_window_width,
                                                                               self.target_window_height)
            inverse_end_x, inverse_end_y = self.screenshot_service.inverse_transform_point(end_x, 
                                                                           end_y,
                                                                           source_width,
                                                                           source_height,
                                                                           self.target_window_width,
                                                                           self.target_window_height)
            
            
            request = SwipeRequest(
                start_x=inverse_start_x,
                start_y=inverse_start_y,
                end_x=inverse_end_x,
                end_y=inverse_end_y,
                duration=duration
            )
            
            # Perform swipe
            await self.mobile_service.swipe(request)
            
            self.step_number += 1
            
            return f"Swiped from ({start_x}, {start_y}) to ({end_x}, {end_y})"
            
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
            # Draw a cursor on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_press.png'
            screenshot = decode_image_base64(self.screenshot.screenshot)
            screenshot = await self.screenshot_service.draw_cursor(screenshot, x, y)
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot, self.step_number, screenshot_filename)
            screenshot_b64 = encode_image_base64(screenshot)
            screenshot_description = f"Action: Press at ({x}, {y}) for {duration}ms"
            self.previous_screenshot = ScreenshotInfo(screenshot=screenshot_b64,
                                                      screenshot_path=screenshot_path,
                                                      screenshot_description=screenshot_description)
            
            # inverse transform the x and y
            source_width, source_height = self.screenshot.transform_info["source_width"], self.screenshot.transform_info["source_height"]
            inverse_x, inverse_y = self.screenshot_service.inverse_transform_point(x, 
                                                                   y,
                                                                   source_width,
                                                                   source_height,
                                                                   self.target_window_width,
                                                                   self.target_window_height)
            
            request = PressRequest(x=inverse_x, y=inverse_y, duration=duration)
            
            # Perform press
            await self.mobile_service.press(request)
            
            self.step_number += 1
            
            return f"Pressed at ({x}, {y}) for {duration}ms"
        
        except Exception as e:
            logger.error(f"Error in press operation: {e}")
            return f"Press failed: {e}"
    
    @ecp.action(
        name="type",
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
            # DO NOT draw anything on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_type.png'
            screenshot = decode_image_base64(self.screenshot.screenshot)
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot,
                                                                             self.step_number,
                                                                             screenshot_filename)
            screenshot_description = f"Action: Type text: {text}"
            self.previous_screenshot = ScreenshotInfo(
                transformed=self.screenshot.transformed,
                screenshot=encode_image_base64(screenshot),
                screenshot_path=screenshot_path,
                screenshot_description=screenshot_description,
                transform_info=self.screenshot.transform_info
            )
            
            request = TypeTextRequest(text=text)
            
            # Perform type text
            await self.mobile_service.type_text(request)
            
            self.step_number += 1
            
            return f"Typed text: {text}"
        
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
            # DO NOT draw anything on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_key_event.png'
            screenshot = decode_image_base64(self.screenshot.screenshot)
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot,
                                                                             self.step_number,
                                                                             screenshot_filename)
            screenshot_description = f"Action: Key event: {keycode}"
            self.previous_screenshot = ScreenshotInfo(
                transformed=self.screenshot.transformed,
                screenshot=encode_image_base64(screenshot),
                screenshot_path=screenshot_path,
                screenshot_description=screenshot_description,
                transform_info=self.screenshot.transform_info
            )
            
            request = KeyEventRequest(keycode=keycode)
            
            # Perform key event
            await self.mobile_service.key_event(request)
            
            self.step_number += 1
            
            return f"Key event: {keycode}"
        
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
            # Draw a path on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_swipe_path.png'
            screenshot = decode_image_base64(self.screenshot.screenshot)
            screenshot = await self.screenshot_service.draw_path(screenshot, path)
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot,
                                                                             self.step_number,
                                                                             screenshot_filename)
            screenshot_description = f"Action: Swipe path with {len(path)} points"
            self.previous_screenshot = ScreenshotInfo(
                transformed=self.screenshot.transformed,
                screenshot=encode_image_base64(screenshot),
                screenshot_path=screenshot_path,
                screenshot_description=screenshot_description,
                transform_info=self.screenshot.transform_info
            )
            
            # inverse transform the path
            source_width, source_height = self.screenshot.transform_info["source_width"], self.screenshot.transform_info["source_height"]
            new_path = []
            for point_x, point_y in path:
                inverse_point_x, inverse_point_y = self.screenshot_service.inverse_transform_point(point_x, 
                                                                                   point_y,
                                                                                   source_width,
                                                                                   source_height,
                                                                                   self.target_window_width,
                                                                                   self.target_window_height)
                new_path.append([inverse_point_x, inverse_point_y])
                
            request = SwipePathRequest(path=new_path, duration=duration)
            
            # Perform swipe path
            await self.mobile_service.swipe_path(request)
            
            self.step_number += 1
            
            return f"Swiped path with {len(path)} points"
        
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
            # DO NOT draw anything on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_scroll.png'
            screenshot = decode_image_base64(self.screenshot.screenshot)
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot,
                                                                             self.step_number,
                                                                             screenshot_filename)
            screenshot_description = f"Action: Scroll {direction} by {distance} pixels"
            self.previous_screenshot = ScreenshotInfo(
                transformed=self.screenshot.transformed,
                screenshot=encode_image_base64(screenshot),
                screenshot_path=screenshot_path,
                screenshot_description=screenshot_description,
                transform_info=self.screenshot.transform_info
            )
            
            request = ScrollRequest(direction=direction, distance=distance)
            
            # Perform scroll
            await self.mobile_service.scroll(request)
            
            self.step_number += 1
            
            return f"Scrolled {direction} by {distance} pixels"
        
        except Exception as e:
            logger.error(f"Error in scroll operation: {e}")
            return f"Scroll failed: {e}"
        
    @ecp.action(
        name="screenshot",
        description="Take a screenshot of the mobile device",
        type="Mobile Environment",
    )
    async def taske_screenshot(self) -> str:
        # DO NOT capture the screenshot here, just return the screenshot path
        return f"Screenshot taken successfully: {self.screenshot.screenshot_path}."
    
    @ecp.action(
        name="wait",
        description="Wait for a specified duration",
        type="Mobile Environment",
    )
    async def wait(self, duration: int) -> str:
        """
        Wait for a specified duration in seconds.
        
        Args:
            duration (int): Wait duration in seconds
        """
        await asyncio.sleep(int(duration))
        return f"Waited for {duration} seconds"
        
    def transform_screenshot(self, screenshot: Image.Image) -> Image.Image:
        """Transform the screenshot to the target window size."""
        transformed_screenshot = self.screenshot_service.transform_screenshot(screenshot,
                                                                              target_width=self.target_window_width,
                                                                              target_height=self.target_window_height,
                                                                              pad_color=self.pad_color)
        return transformed_screenshot
    
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
    

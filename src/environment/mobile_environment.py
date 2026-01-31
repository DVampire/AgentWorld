"""Mobile Environment for AgentWorld - provides mobile device automation operations as an environment."""
import io
import asyncio
from PIL import Image
from typing import Any, Dict, List, Optional
from pydantic import Field, ConfigDict

from src.environment.mobile.service import MobileService
from src.environment.mobile.types import (
    TapRequest,
    SwipeRequest,
    PressRequest,
    TypeTextRequest,
    KeyEventRequest,
    SwipePathRequest,
    ScrollRequest,
)
from src.logger import logger
from src.environment.types import Environment, ScreenshotInfo
from src.utils import dedent, ScreenshotService, encode_file_base64, decode_file_base64, make_file_url
from src.environment.server import ecp
from src.registry import ENVIRONMENT

@ENVIRONMENT.register_module(force=True)
class MobileEnvironment(Environment):
    """Mobile Environment that provides mobile device automation operations as an environment interface."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="mobile", description="The name of the Mobile environment.")
    description: str = Field(default="Mobile device automation environment for Android device control", description="The description of the Mobile environment.")
    metadata: Dict[str, Any] = Field(default={
        "has_vision": True,
        "additional_rules": {
            "state": "The state of the mobile device including device info, screen size, and current screenshot.",
        }
    }, description="The metadata of the Mobile environment.")
    require_grad: bool = Field(default=False, description="Whether the environment requires gradients")
    
    def __init__(
        self,
        base_dir: str = "./workdir/mobile_agent",
        device_id: Optional[str] = None,
        fps: int = 2,
        bitrate: int = 50000000,
        chunk_duration: int = 60,
        require_grad: bool = False,
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
    )
    async def tap(self, x: int, y: int, **kwargs) -> Dict[str, Any]:
        """
        Tap at specified coordinates on the mobile device.
        
        Args:
            x (int): X coordinate for tap
            y (int): Y coordinate for tap
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            
            # Draw a cursor on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_tap.png'
            screenshot = self.screenshot.screenshot
            screenshot_image = Image.open(io.BytesIO(decode_file_base64(screenshot)))
            screenshot_image = await self.screenshot_service.draw_cursor(screenshot_image, x, y)
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot_image, self.step_number, screenshot_filename)
            screenshot_description = f"Action: Tap at ({x}, {y})"
            self.previous_screenshot = ScreenshotInfo(
                transformed=self.screenshot.transformed,
                screenshot=encode_file_base64(file_path=screenshot_path),
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
            result = await self.mobile_service.tap(request)
            
            self.step_number += 1
            
            extra = result.extra.copy() if result.extra else {}
            extra["x"] = x
            extra["y"] = y
            extra["inverse_x"] = inverse_x
            extra["inverse_y"] = inverse_y
            extra["screenshot_path"] = screenshot_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
            
        except Exception as e:
            logger.error(f"Error in tap operation: {e}")
            return {
                "success": False,
                "message": f"Tap failed: {str(e)}",
                "extra": {"error": str(e), "x": x, "y": y}
            }
    
    @ecp.action(
        name="swipe",
        description="Swipe at specified coordinates on the mobile device",
    )
    async def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: int = 300, **kwargs) -> Dict[str, Any]:
        """
        Swipe gesture from start to end coordinates.
        
        Args:
            start_x (int): Start X coordinate
            start_y (int): Start Y coordinate
            end_x (int): End X coordinate
            end_y (int): End Y coordinate
            duration (int): Swipe duration in milliseconds
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            # Draw a path on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_swipe.png'
            screenshot = self.screenshot.screenshot
            screenshot_image = Image.open(io.BytesIO(decode_file_base64(screenshot)))
            screenshot_image = await self.screenshot_service.draw_path(screenshot_image, [[start_x, start_y], [end_x, end_y]])
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot_image, self.step_number, screenshot_filename)
            screenshot_description = f"Action: Swipe from ({start_x}, {start_y}) to ({end_x}, {end_y})"
            self.previous_screenshot = ScreenshotInfo(
                transformed=self.screenshot.transformed,
                screenshot=encode_file_base64(file_path=screenshot_path),
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
            result = await self.mobile_service.swipe(request)
            
            self.step_number += 1
            
            extra = result.extra.copy() if result.extra else {}
            extra["start_x"] = start_x
            extra["start_y"] = start_y
            extra["end_x"] = end_x
            extra["end_y"] = end_y
            extra["duration"] = duration
            extra["screenshot_path"] = screenshot_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
            
        except Exception as e:
            logger.error(f"Error in swipe operation: {e}")
            return {
                "success": False,
                "message": f"Swipe failed: {str(e)}",
                "extra": {"error": str(e), "start_x": start_x, "start_y": start_y, "end_x": end_x, "end_y": end_y}
            }
    
    @ecp.action(
        name="press",
        description="Long press at specified coordinates on the mobile device",
    )
    async def press(self, x: int, y: int, duration: int = 1000, **kwargs) -> Dict[str, Any]:
        """
        Long press at specified coordinates.
        
        Args:
            x (int): X coordinate for press
            y (int): Y coordinate for press
            duration (int): Press duration in milliseconds
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            # Draw a cursor on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_press.png'
            screenshot = self.screenshot.screenshot
            screenshot_image = Image.open(io.BytesIO(decode_file_base64(screenshot)))
            screenshot_image = await self.screenshot_service.draw_cursor(screenshot_image, x, y)
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot_image, self.step_number, screenshot_filename)
            screenshot_description = f"Action: Press at ({x}, {y}) for {duration}ms"
            self.previous_screenshot = ScreenshotInfo(
                transformed=self.screenshot.transformed,
                screenshot=encode_file_base64(file_path=screenshot_path),
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
                                                                   self.target_window_height)
            
            request = PressRequest(x=inverse_x, y=inverse_y, duration=duration)
            
            # Perform press
            result = await self.mobile_service.press(request)
            
            self.step_number += 1
            
            extra = result.extra.copy() if result.extra else {}
            extra["x"] = x
            extra["y"] = y
            extra["duration"] = duration
            extra["screenshot_path"] = screenshot_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        
        except Exception as e:
            logger.error(f"Error in press operation: {e}")
            return {
                "success": False,
                "message": f"Press failed: {str(e)}",
                "extra": {"error": str(e), "x": x, "y": y, "duration": duration}
            }
    
    @ecp.action(
        name="type",
        description="Type text at the current cursor position on the mobile device",
    )
    async def type_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """
        Type text on the mobile device.
        
        Args:
            text (str): Text to input
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            # DO NOT draw anything on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_type.png'
            screenshot = self.screenshot.screenshot
            screenshot_image = Image.open(io.BytesIO(decode_file_base64(screenshot)))
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot_image, self.step_number, screenshot_filename)
            screenshot_description = f"Action: Type text: {text}"
            self.previous_screenshot = ScreenshotInfo(
                transformed=self.screenshot.transformed,
                screenshot=encode_file_base64(file_path=screenshot_path),
                screenshot_path=screenshot_path,
                screenshot_description=screenshot_description,
                transform_info=self.screenshot.transform_info
            )
            
            request = TypeTextRequest(text=text)
            
            # Perform type text
            result = await self.mobile_service.type_text(request)
            
            self.step_number += 1
            
            extra = result.extra.copy() if result.extra else {}
            extra["text"] = text
            extra["screenshot_path"] = screenshot_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        
        except Exception as e:
            logger.error(f"Error in type operation: {e}")
            return {
                "success": False,
                "message": f"Type failed: {str(e)}",
                "extra": {"error": str(e), "text": text}
            }
    
    @ecp.action(
        name="key_event",
        description="Press a key on the mobile device",
    )
    async def key_event(self, keycode: int, **kwargs) -> Dict[str, Any]:
        """
        Press a key on the mobile device.
        
        Args:
            keycode (int): Android keycode to press
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            # DO NOT draw anything on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_key_event.png'
            screenshot = self.screenshot.screenshot
            screenshot_image = Image.open(io.BytesIO(decode_file_base64(screenshot)))
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot_image, self.step_number, screenshot_filename)
            screenshot_description = f"Action: Key event: {keycode}"
            self.previous_screenshot = ScreenshotInfo(
                transformed=self.screenshot.transformed,
                screenshot=encode_file_base64(file_path=screenshot_path),
                screenshot_path=screenshot_path,
                screenshot_description=screenshot_description,
                transform_info=self.screenshot.transform_info
            )
            
            request = KeyEventRequest(keycode=keycode)
            
            # Perform key event
            result = await self.mobile_service.key_event(request)
            
            self.step_number += 1
            
            extra = result.extra.copy() if result.extra else {}
            extra["keycode"] = keycode
            extra["screenshot_path"] = screenshot_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        
        except Exception as e:
            logger.error(f"Error in key event operation: {e}")
            return {
                "success": False,
                "message": f"Key event failed: {str(e)}",
                "extra": {"error": str(e), "keycode": keycode}
            }
    
    # ==================== ADVANCED OPERATIONS ====================
    
    @ecp.action(
        name="swipe_path",
        description="Swipe along a path of coordinates on the mobile device",
    )
    async def swipe_path(self, path: List[List[int]], duration: int = 300, **kwargs) -> Dict[str, Any]:
        """
        Swipe along a path of coordinates.
        
        Args:
            path (List[List[int]]): List of [x, y] coordinates
            duration (int): Total swipe duration in milliseconds
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            # Draw a path on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_swipe_path.png'
            screenshot = self.screenshot.screenshot
            screenshot_image = Image.open(io.BytesIO(decode_file_base64(screenshot)))
            screenshot_image = await self.screenshot_service.draw_path(screenshot_image, path)
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot_image, self.step_number, screenshot_filename)
            screenshot_description = f"Action: Swipe path with {len(path)} points"
            self.previous_screenshot = ScreenshotInfo(
                transformed=self.screenshot.transformed,
                screenshot=encode_file_base64(file_path=screenshot_path),
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
            result = await self.mobile_service.swipe_path(request)
            
            self.step_number += 1
            
            extra = result.extra.copy() if result.extra else {}
            extra["path"] = path
            extra["duration"] = duration
            extra["screenshot_path"] = screenshot_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        
        except Exception as e:
            logger.error(f"Error in swipe path operation: {e}")
            return {
                "success": False,
                "message": f"Swipe path failed: {str(e)}",
                "extra": {"error": str(e), "path": path, "duration": duration}
            }
    
    # ==================== SCROLL OPERATIONS ====================
    
    @ecp.action(
        name="scroll",
        description="Scroll on the mobile device in specified direction",
    )
    async def scroll(self, direction: str, distance: int = 500, **kwargs) -> Dict[str, Any]:
        """
        Scroll on the mobile device in specified direction.
        
        Args:
            direction (str) : Scroll direction ("up", "down", "left", "right")
            distance: Scroll distance in pixels
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            # DO NOT draw anything on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_scroll.png'
            screenshot = self.screenshot.screenshot
            screenshot_image = Image.open(io.BytesIO(decode_file_base64(screenshot)))
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot_image, self.step_number, screenshot_filename)
            screenshot_description = f"Action: Scroll {direction} by {distance} pixels"
            self.previous_screenshot = ScreenshotInfo(
                transformed=self.screenshot.transformed,
                screenshot=encode_file_base64(file_path=screenshot_path),
                screenshot_path=screenshot_path,
                screenshot_description=screenshot_description,
                transform_info=self.screenshot.transform_info
            )
            
            request = ScrollRequest(direction=direction, distance=distance)
            
            # Perform scroll
            result = await self.mobile_service.scroll(request)
            
            self.step_number += 1
            
            extra = result.extra.copy() if result.extra else {}
            extra["direction"] = direction
            extra["distance"] = distance
            extra["screenshot_path"] = screenshot_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
        
        except Exception as e:
            logger.error(f"Error in scroll operation: {e}")
            return {
                "success": False,
                "message": f"Scroll failed: {str(e)}",
                "extra": {"error": str(e), "direction": direction, "distance": distance}
            }
        
    @ecp.action(
        name="screenshot",
        description="Take a screenshot of the mobile device",
    )
    async def taske_screenshot(self, **kwargs) -> Dict[str, Any]:
        """Take a screenshot of the mobile device.
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            # DO NOT capture the screenshot here, just return the screenshot path
            screenshot_path = self.screenshot.screenshot_path if self.screenshot else None
            
            return {
                "success": True,
                "message": f"Screenshot taken successfully: {screenshot_path}." if screenshot_path else "Screenshot not available",
                "extra": {"screenshot_path": screenshot_path} if screenshot_path else {}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to get screenshot: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    @ecp.action(
        name="wait",
        description="Wait for a specified duration",
    )
    async def wait(self, duration: int, **kwargs) -> Dict[str, Any]:
        """
        Wait for a specified duration in seconds.
        
        Args:
            duration (int): Wait duration in seconds
        
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            await asyncio.sleep(int(duration))
            return {
                "success": True,
                "message": f"Waited for {duration} seconds",
                "extra": {"duration": duration}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Wait failed: {str(e)}",
                "extra": {"error": str(e), "duration": duration}
            }
        
    def transform_screenshot(self, screenshot: Image.Image) -> Image.Image:
        """Transform the screenshot to the target window size."""
        transformed_screenshot = self.screenshot_service.transform_screenshot(screenshot,
                                                                              target_width=self.target_window_width,
                                                                              target_height=self.target_window_height,
                                                                              pad_color=self.pad_color)
        return transformed_screenshot
    
    async def get_state(self, **kwargs) -> Dict[str, Any]:
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
            screenshot_data = mobile_device_state["screenshot"]
            screenshot_image = Image.open(io.BytesIO(decode_file_base64(screenshot_data)))
            source_width, source_height = screenshot_image.size
            transformed_screenshot = self.transform_screenshot(screenshot_image)
            screenshot_filename = f'step_{self.step_number:04d}_state.png'
            screenshot_path = await self.screenshot_service.store_screenshot(transformed_screenshot, self.step_number, screenshot_filename)
            screenshot_description = "A screenshot of the device at current step."
            
            self.screenshot = ScreenshotInfo(
                transformed=True,
                screenshot=encode_file_base64(file_path=screenshot_path),
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
    

"""Mobile Environment for AgentWorld - provides mobile device automation operations as an environment."""

from typing import Any, Dict, List, Union, Optional, Type
from pydantic import BaseModel, Field, ConfigDict

from src.environments.mobile.service import MobileService
from src.environments.mobile.types import (
    TapRequest,
    SwipeRequest,
    PressRequest,
    TypeTextRequest,
    KeyEventRequest,
    ScreenshotRequest,
    SwipePathRequest,
    SwipePathResult,
    ScrollRequest,
)
from src.logger import logger
from src.environments.protocol.server import ecp
from src.environments.protocol.environment import BaseEnvironment
from src.utils import dedent
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
        
        self.mobile_service = MobileService(
            base_dir=base_dir,
            device_id=device_id,
            fps=fps,
            bitrate=bitrate,
            chunk_duration=chunk_duration
        )
        
        self.is_connected = False
        
    async def initialize(self) -> None:
        """Initialize the mobile environment."""
        await self.mobile_service.start()
        logger.info(f"| 📱 Mobile Environment initialized at: {self.base_dir}")
        
    async def cleanup(self) -> None:
        """Cleanup the mobile environment."""
        await self.mobile_service.stop()
        logger.info("| 📱 Mobile Environment cleanup completed")
    
    async def get_state(self) -> Dict[str, Any]:
        """Get the current state of the mobile device."""
        try:
            mobile_device_state = await self.mobile_service.get_device_state()
            
            state = dedent(f"""
                <info>
                Device Info: {mobile_device_state.device_info}
                </info>
            """)
            
            screenshot_path = mobile_device_state.screenshot_path
            screenshot_description = "Current screenshot of the mobile device."
            screenshots = [ScreenshotInfo(screenshot_path=screenshot_path, screenshot_description=screenshot_description)]
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
            x: X coordinate for tap
            y: Y coordinate for tap
            
        Returns:
            TapResult: Result of the tap operation
        """
        try:
            request = TapRequest(x=x, y=y)
            result = await self.mobile_service.tap(request)
            logger.info(f"Tap operation: {result.success} - {result.message}")
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
            start_x: Start X coordinate
            start_y: Start Y coordinate
            end_x: End X coordinate
            end_y: End Y coordinate
            duration: Swipe duration in milliseconds
            
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
            result = await self.mobile_service.swipe(request)
            logger.info(f"Swipe operation: {result.success} - {result.message}")
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
            x: X coordinate for press
            y: Y coordinate for press
            duration: Press duration in milliseconds
            
        Returns:
            PressResult: Result of the press operation
        """
        try:
            request = PressRequest(x=x, y=y, duration=duration)
            result = await self.mobile_service.press(request)
            logger.info(f"Press operation: {result.success} - {result.message}")
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
            text: Text to input
            
        Returns:
            TypeTextResult: Result of the type operation
        """
        try:
            request = TypeTextRequest(text=text)
            result = await self.mobile_service.type_text(request)
            logger.info(f"Type operation: {result.success} - {result.message}")
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
            keycode: Android keycode to press
            
        Returns:
            KeyEventResult: Result of the key event operation
        """
        try:
            request = KeyEventRequest(keycode=keycode)
            result = await self.mobile_service.key_event(request)
            logger.info(f"Key event operation: {result.success} - {result.message}")
            return result.message
        except Exception as e:
            logger.error(f"Error in key event operation: {e}")
            return f"Key event failed: {e}"
    
    @ecp.action(
        name="screenshot",
        description="Take a screenshot of the mobile device",
        type="Mobile Environment",
    )
    async def screenshot(self, save_path: Optional[str] = None) -> str:
        """
        Take a screenshot of the mobile device.
        
        Args:
            save_path: Optional path to save screenshot
            
        Returns:
            ScreenshotResult: Result of the screenshot operation
        """
        try:
            request = ScreenshotRequest(save_path=save_path)
            result = await self.mobile_service.take_screenshot(request)
            logger.info(f"Screenshot operation: {result.success} - {result.message}")
            return result.message
        
        except Exception as e:
            logger.error(f"Error in screenshot operation: {e}")
            return f"Screenshot failed: {e}"
    
    # ==================== ADVANCED OPERATIONS ====================
    
    @ecp.action(
        name="swipe_path",
        description="Swipe along a path of coordinates on the mobile device",
        type="Mobile Environment",
    )
    async def swipe_path(self, path: List[List[int]], duration: int = 300) -> SwipePathResult:
        """
        Swipe along a path of coordinates.
        
        Args:
            path: List of [x, y] coordinates
            duration: Total swipe duration in milliseconds
            
        Returns:
            SwipePathResult: Result of the swipe path operation
        """
        try:
            request = SwipePathRequest(path=path, duration=duration)
            result = await self.mobile_service.swipe_path(request)
            logger.info(f"Swipe path operation: {result.success} - {result.message}")
            return result
        except Exception as e:
            logger.error(f"Error in swipe path operation: {e}")
            return SwipePathResult(success=False, message=f"Swipe path failed: {e}")
    
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
            direction: Scroll direction ("up", "down", "left", "right")
            distance: Scroll distance in pixels
            
        Returns:
            str: Result message
        """
        try:
            request = ScrollRequest(direction=direction, distance=distance)
            result = await self.mobile_service.scroll(request)
            logger.info(f"Scroll operation: {result.success} - {result.message}")
            return result.message
        except Exception as e:
            logger.error(f"Error in scroll operation: {e}")
            return f"Scroll failed: {e}"
    
    # ==================== SYSTEM OPERATIONS ====================
    @ecp.action(
        name="wake_up",
        description="Wake up the mobile device",
        type="Mobile Environment",
    )
    async def wake_up(self) -> bool:
        """
        Wake up the mobile device.
        
        Returns:
            bool: True if successful
        """
        try:
            await self.mobile_service.adb.wake_up()
            logger.info("Device wake up")
            return True
        except Exception as e:
            logger.error(f"Error in wake up: {e}")
            return False
    
    @ecp.action(
        name="unlock_screen",
        description="Unlock the mobile device screen",
        type="Mobile Environment",
    )
    async def unlock_screen(self) -> bool:
        """
        Unlock the mobile device screen.
        
        Returns:
            bool: True if successful
        """
        try:
            await self.mobile_service.adb.unlock_screen()
            logger.info("Screen unlock")
            return True
        except Exception as e:
            logger.error(f"Error in unlock screen: {e}")
            return False
    
    @ecp.action(
        name="open_app",
        description="Open an app on the mobile device",
        type="Mobile Environment",
    )
    async def open_app(self, package_name: str) -> bool:
        """
        Open an app on the mobile device.
        
        Args:
            package_name: App package name
            
        Returns:
            bool: True if successful
        """
        try:
            await self.mobile_service.adb.open_app(package_name)
            logger.info(f"Open app: {package_name}")
            return True
        except Exception as e:
            logger.error(f"Error in open app: {e}")
            return False
    
    @ecp.action(
        name="close_app",
        description="Close an app on the mobile device",
        type="Mobile Environment",
    )
    async def close_app(self, package_name: str) -> bool:
        """
        Close an app on the mobile device.
        
        Args:
            package_name: App package name
            
        Returns:
            bool: True if successful
        """
        try:
            await self.mobile_service.adb.close_app(package_name)
            logger.info(f"Close app: {package_name}")
            return True
        except Exception as e:
            logger.error(f"Error in close app: {e}")
            return False

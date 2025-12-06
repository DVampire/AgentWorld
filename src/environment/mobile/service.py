"""Mobile device service using ADB and Scrcpy."""

import asyncio
import os
import time
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import base64
import urllib.parse

from src.environment.mobile.types import (
    MobileDeviceInfo,
    TapRequest, 
    SwipeRequest,
    PressRequest, 
    TypeTextRequest,
    KeyEventRequest,
    ScrollRequest,
    ScreenshotRequest, 
    SwipePathRequest,
)
from src.environment.protocol.types import ActionResult

# Import the three components
from src.environment.mobile.adb import AdbDriver
from src.environment.mobile.scrcpy import ScrcpyDriver
from src.environment.mobile.minicap import MinicapDriver
from src.utils import dedent
from src.logger import logger


class MobileService:
    """Mobile device service using ADB and Scrcpy for device control and screen capture."""
    
    def __init__(
        self,
        base_dir: str = "./workdir/mobile_agent",
        device_id: Optional[str] = None,
        fps: int = 2,
        bitrate: int = 50000000,
        chunk_duration: int = 60,
    ):
        """
        Initialize the mobile service.
        
        Args:
            base_dir: Base directory for mobile agent work
            device_id: Target device ID (defaults to first connected device)
            fps: Frame rate for screen capture
            bitrate: Video bitrate
            chunk_duration: Video chunk duration in seconds
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.video_save_path = self.base_dir / "videos"
        self.video_save_path.mkdir(parents=True, exist_ok=True)
        self.video_save_name = "mobile_record"
        
        self.device_id = device_id
        self.fps = fps
        self.bitrate = bitrate
        self.chunk_duration = chunk_duration
        
        # Initialize components (following adb_scrcpy.py pattern)
        self.adb = None
        self.scrcpy = None
        self.minicap = None
        
        # Device info
        self.device_info: Optional[MobileDeviceInfo] = None
        self.is_connected = False
        self.is_recording = False
        self.recording_path: Optional[str] = None
        
        # Screen properties (like adb_scrcpy.py)
        self.screen_density = 0
        self.screen_info = {
            "width": 0,
            "height": 0,
            "rotate": 0
        }

    async def start(self) -> bool:
        """Start the mobile service and connect to device."""
        try:
            # Initialize ADB bridge (like adb_scrcpy.py)
            self.adb = AdbDriver(self.device_id)
            
            if not self.adb.device:
                logger.error("Failed to connect to ADB device")
                return False
            
            # Initialize Scrcpy bridge (like adb_scrcpy.py)
            self.scrcpy = ScrcpyDriver(device=self.device_id)
            
            # Initialize Minicap for screen capture (like adb_scrcpy.py)
            self.minicap = MinicapDriver(
                device_id=self.device_id,
                video_save_path=self.video_save_path,
                video_save_name=self.video_save_name,
                fps=self.fps,
                chunk_duration=self.chunk_duration,
                video_with_reasoning=True
            )
            
            # Start video recording if path is specified (like adb_scrcpy.py)
            if self.video_save_path:
                self.minicap.start_record()
                self.is_recording = True
                self.recording_path = os.path.join(self.video_save_path, f"{self.video_save_name}.mp4")
            
            # Get device screen properties (like adb_scrcpy.py)
            self.screen_density = await self.adb.get_screen_density()
            self.screen_info = await self.adb.get_screen_info()
            
            # Create device info
            self.device_info = MobileDeviceInfo(
                device_id=self.device_id or "default",
                screen_width=self.screen_info["width"],
                screen_height=self.screen_info["height"],
                screen_rotate=self.screen_info["rotate"],
                screen_density=self.screen_density,
                is_connected=True
            )
            
            self.is_connected = True
            
            # Wait for initialization (like adb_scrcpy.py)
            await asyncio.sleep(0.5)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start mobile service: {e}")
            return False
    
    async def stop(self) -> None:
        """Stop the mobile service and cleanup resources."""
        try:
            # Close components (like adb_scrcpy.py)
            if self.scrcpy:
                self.scrcpy.close()
            if self.adb:
                del self.adb
            if self.minicap:
                self.minicap.close()
        except Exception as e:
            logger.error(f"Error stopping mobile service: {e}")
        finally:
            self.is_connected = False
            self.is_recording = False
    
    async def tap(self, action: TapRequest) -> ActionResult:
        """Perform a tap action on the device."""
        try:
            if not self.is_connected:
                return ActionResult(
                    success=False,
                    message="Device not connected",
                    extra={"error": "Device not connected"}
                )
            
            # Use ADB for simple tap (like adb_scrcpy.py)
            await self.adb.tap(action.x, action.y)
            await asyncio.sleep(0.5) # wait for the device to respond
            
            screenshot = await self._take_screenshot()
            screenshot_description = f"A screenshot of the device after tapping at ({action.x}, {action.y})"
            
            return ActionResult(
                success=True,
                message=f"Tapped at ({action.x}, {action.y})",
                extra={
                    "screenshot": screenshot,
                    "screenshot_description": screenshot_description,
                    "x": action.x,
                    "y": action.y
                }
            )
        
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Tap failed: {e}",
                extra={"error": str(e), "x": action.x, "y": action.y}
            )
    
    async def swipe(self, action: SwipeRequest) -> ActionResult:
        """Perform a swipe action on the device."""
        try:
            if not self.is_connected:
                return ActionResult(
                    success=False,
                    message="Device not connected",
                    extra={"error": "Device not connected"}
                )
            
            # Use ADB for swipe (like adb_scrcpy.py)
            await self.adb.swipe(
                action.start_x,
                action.start_y,
                action.end_x,
                action.end_y,
                action.duration
            )
            
            await asyncio.sleep(0.5) # wait for the device to respond
            
            # Take screenshot
            screenshot = await self._take_screenshot()
            screenshot_description = f"A screenshot of the device after swipe from ({action.start_x}, {action.start_y}) to ({action.end_x}, {action.end_y}) for {action.duration}ms"
            
            return ActionResult(
                success=True,
                message=f"Swiped from ({action.start_x}, {action.start_y}) to ({action.end_x}, {action.end_y})",
                extra={
                    "screenshot": screenshot,
                    "screenshot_description": screenshot_description,
                    "start_x": action.start_x,
                    "start_y": action.start_y,
                    "end_x": action.end_x,
                    "end_y": action.end_y,
                    "duration": action.duration
                }
            )
        
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Swipe failed: {e}",
                extra={"error": str(e), "start_x": action.start_x, "start_y": action.start_y, "end_x": action.end_x, "end_y": action.end_y, "duration": action.duration}
            )

    async def press(self, action: PressRequest) -> ActionResult:
        """Perform a long press action on the device."""
        try:
            if not self.is_connected:
                return ActionResult(
                    success=False,
                    message="Device not connected",
                    extra={"error": "Device not connected"}
                )
            
            # Use ADB for long press (like adb_scrcpy.py - swipe to same position)
            await self.adb.press(action.x, action.y, action.duration)
            
            await asyncio.sleep(0.5) # wait for the device to respond
            
            # Take screenshot
            screenshot = await self._take_screenshot()
            screenshot_description = f"A screenshot of the device after long pressing at ({action.x}, {action.y}) for {action.duration}ms"
            
            return ActionResult(
                success=True,
                message=f"Long pressed at ({action.x}, {action.y}) for {action.duration}ms",
                extra={
                    "screenshot": screenshot,
                    "screenshot_description": screenshot_description,
                    "x": action.x,
                    "y": action.y,
                    "duration": action.duration
                }
            )
        
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Long press failed: {e}",
                extra={"error": str(e), "x": action.x, "y": action.y, "duration": action.duration}
            )
    
    async def type_text(self, action: TypeTextRequest) -> ActionResult:
        """Input text on the device."""
        try:
            if not self.is_connected:
                return ActionResult(
                    success=False,
                    message="Device not connected",
                    extra={"error": "Device not connected"}
                )
            
            # Use ADB for text input (like adb_scrcpy.py)
            text = urllib.parse.quote(action.text)
            await self.adb.type_text(text)
            
            await asyncio.sleep(0.5) # wait for the device to respond
            
            # Take screenshot
            screenshot = await self._take_screenshot()
            screenshot_description = f"A screenshot of the device after inputting text: {action.text}"
            
            return ActionResult(
                success=True,
                message=f"Input text: {action.text}",
                extra={
                    "screenshot": screenshot,
                    "screenshot_description": screenshot_description,
                    "text": action.text
                }
            )
        
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Text input failed: {e}",
                extra={"error": str(e), "text": action.text}
            )
    
    async def key_event(self, action: KeyEventRequest) -> ActionResult:
        """Press a key on the device."""
        try:
            if not self.is_connected:
                return ActionResult(
                    success=False,
                    message="Device not connected",
                    extra={"error": "Device not connected"}
                )
            
            # Use ADB for key press (like adb_scrcpy.py)
            await self.adb.key_event(action.keycode)
            
            await asyncio.sleep(0.5) # wait for the device to respond
            
            # Take screenshot
            screenshot = await self._take_screenshot()
            screenshot_description = f"A screenshot of the device after pressing key: {action.keycode}"
            
            return ActionResult(
                success=True,
                message=f"Pressed key: {action.keycode}",
                extra={
                    "screenshot": screenshot,
                    "screenshot_description": screenshot_description,
                    "keycode": action.keycode
                }
            )
                        
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Key press failed: {e}",
                extra={"error": str(e), "keycode": action.keycode}
            )
    
    async def swipe_path(self, action: SwipePathRequest) -> ActionResult:
        """Perform a swipe along a path."""
        try:
            if not self.is_connected:
                return ActionResult(
                    success=False,
                    message="Device not connected",
                    extra={"error": "Device not connected"}
                )
            
            if len(action.path) < 2:
                return ActionResult(
                    success=False,
                    message="Path must have at least 2 points",
                    extra={"error": "Path must have at least 2 points", "path": action.path}
                )
            
            # Calculate duration per segment
            segment_duration = action.duration // (len(action.path) - 1)
            
            # Perform swipe along path
            for i in range(len(action.path) - 1):
                start_point = action.path[i]
                end_point = action.path[i + 1]
                
                await self.adb.swipe(
                    start_point[0],
                    start_point[1],
                    end_point[0],
                    end_point[1],
                    segment_duration
                )
                
                # Small delay between segments
                await asyncio.sleep(0.5)
            
            # Take screenshot
            screenshot = await self._take_screenshot()
            screenshot_description = f"A screenshot of the device after swiping along path with {len(action.path)} points"
            
            return ActionResult(
                success=True,
                message=f"Swiped along path with {len(action.path)} points",
                extra={
                    "screenshot": screenshot,
                    "screenshot_description": screenshot_description,
                    "path": action.path,
                    "duration": action.duration
                }
            )
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Swipe path failed: {e}",
                extra={"error": str(e), "path": action.path, "duration": action.duration}
            )
        
    async def scroll(self, action: ScrollRequest) -> ActionResult:
        """Perform a scroll action on the device."""
        try:
            if not self.is_connected:
                return ActionResult(
                    success=False,
                    message="Device not connected",
                    extra={"error": "Device not connected"}
                )
            
            await self.adb.scroll(action.direction, action.distance)
            
            await asyncio.sleep(0.5) # wait for the device to respond
            
            screenshot = await self._take_screenshot()
            screenshot_description = f"A screenshot of the device after scrolling {action.direction} by {action.distance} pixels"
            
            return ActionResult(
                success=True,
                message=f"Scrolled {action.direction} by {action.distance} pixels",
                extra={
                    "screenshot": screenshot,
                    "screenshot_description": screenshot_description,
                    "direction": action.direction,
                    "distance": action.distance
                }
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Scroll failed: {e}",
                extra={"error": str(e), "direction": action.direction, "distance": action.distance}
            )
    
    async def take_screenshot(self, action: ScreenshotRequest) -> ActionResult:
        """Take a screenshot of the device."""
        try:
            if not self.is_connected:
                return ActionResult(
                    success=False,
                    message="Device not connected",
                    extra={"error": "Device not connected"}
                )
            
            screenshot_data = await self._take_screenshot(save_path=action.save_path)
            
            return ActionResult(
                success=True,
                message="Screenshot taken",
                extra={
                    "screenshot": screenshot_data,
                    "save_path": action.save_path
                }
            )
            
        except Exception as e:
            return ActionResult(
                success=False,
                message=f"Screenshot failed: {e}",
                extra={"error": str(e), "save_path": action.save_path}
            )
    
    async def _take_screenshot(self, save_path: Optional[str] = None) -> str:
        """Take a screenshot and save it."""
        if save_path is None:
            timestamp = int(time.time() * 1000)
            save_path = self.base_dir / f"screenshot_{timestamp}.png"
        
        # Use Minicap for high-quality screenshots (like adb_scrcpy.py)
        try:
            screenshot_data = self.minicap.get_screenshot_bytes()
            
            # Save screenshot
            screenshot_data = base64.b64encode(screenshot_data).decode('utf-8')
            return screenshot_data
            
        except Exception as e:
            # Fallback to ADB screenshot
            screenshot = await self.adb.get_screenshot()
            screenshot_data = base64.b64encode(screenshot.tobytes()).decode('utf-8')
            return screenshot_data
    
    async def get_state(self) -> Dict[str, Any]:
        """Get current device state."""
        
        screenshot = await self._take_screenshot()
        
        state = dedent(f"""
            <info>
            Device ID: {self.device_id}
            Screen Width: {self.screen_info["width"]}
            Screen Height: {self.screen_info["height"]}
            Screen Rotate: {self.screen_info["rotate"]}
            Screen Density: {self.screen_density}
            Is Connected: {self.is_connected}
            </info>
        """
        )
        extra = {
            "screenshot": screenshot,
            "is_recording": self.is_recording,
            "recording_path": self.recording_path
        }
        
        return {
            "state": state,
            "extra": extra
        }
    
    async def pause_recording(self) -> None:
        """Pause video recording (like adb_scrcpy.py)."""
        if self.minicap and self.is_recording:
            self.minicap.pause()
    
    async def resume_recording(self) -> None:
        """Resume video recording (like adb_scrcpy.py)."""
        if self.minicap and self.is_recording:
            self.minicap.unpause()
    
    # Additional methods following adb_scrcpy.py pattern
    def get_screen_info(self) -> Dict[str, Any]:
        """Get screen size (like adb_scrcpy.py)."""
        return self.screen_info
    
    def get_screen_density(self) -> int:
        """Get screen density (like adb_scrcpy.py)."""
        return self.screen_density
    
    async def get_screenshot(self, interface: str = "auto"):
        """Get screenshot with interface selection (like adb_scrcpy.py)."""
        if interface == "auto" or interface == "minicap":
            return self.minicap.get_screenshot()
        elif interface == "scrcpy":
            return await self.scrcpy.get_screenshot()
        elif interface == "adb":
            return await self.adb.get_screenshot()
        else:
            raise NotImplementedError(f"Interface {interface} not supported")

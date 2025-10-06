"""Mobile device service using ADB and Scrcpy."""

import asyncio
import os
import time
from typing import Optional, Tuple
from pathlib import Path

from src.environments.mobile.types import (
    MobileDeviceInfo, TapRequest, TapResult,
    SwipeRequest, SwipeResult, PressRequest, PressResult,
    TypeTextRequest, TypeTextResult, KeyEventRequest, KeyEventResult, ScrollRequest, ScrollResult,
    ScreenshotRequest, ScreenshotResult, SwipePathRequest, SwipePathResult,
    MobileDeviceState
)

# Import the three components
from src.environments.mobile.adb import AdbDriver
from src.environments.mobile.scrcpy import ScrcpyDriver
from src.environments.mobile.minicap import MinicapDriver


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
        self.screen_size = (0, 0)
        
        # Screenshot paths
        self.current_screenshot_path: Optional[str] = None
        self.previous_screenshot_path: Optional[str] = None
    
    async def start(self) -> bool:
        """Start the mobile service and connect to device."""
        try:
            # Initialize ADB bridge (like adb_scrcpy.py)
            self.adb = AdbDriver(self.device_id)
            
            if not self.adb.device:
                print("Failed to connect to ADB device")
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
            self.screen_size = await self.adb.get_screen_size()
            
            # Create device info
            self.device_info = MobileDeviceInfo(
                device_id=self.device_id or "default",
                screen_width=self.screen_size[0],
                screen_height=self.screen_size[1],
                screen_density=self.screen_density,
                is_connected=True
            )
            
            self.is_connected = True
            
            # Wait for initialization (like adb_scrcpy.py)
            await asyncio.sleep(0.5)
            
            return True
            
        except Exception as e:
            print(f"Failed to start mobile service: {e}")
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
            print(f"Error stopping mobile service: {e}")
        finally:
            self.is_connected = False
            self.is_recording = False
    
    async def tap(self, action: TapRequest) -> TapResult:
        """Perform a tap action on the device."""
        try:
            if not self.is_connected:
                return TapResult(success=False, message="Device not connected")
            
            # Use ADB for simple tap (like adb_scrcpy.py)
            await self.adb.tap(action.x, action.y)
            
            # Take screenshot
            screenshot_path = await self._take_screenshot()
            
            return TapResult(
                success=True,
                message=f"Tapped at ({action.x}, {action.y})",
                screenshot_path=screenshot_path
            )
            
        except Exception as e:
            return TapResult(success=False, message=f"Tap failed: {e}")
    
    async def swipe(self, action: SwipeRequest) -> SwipeResult:
        """Perform a swipe action on the device."""
        try:
            if not self.is_connected:
                return SwipeResult(success=False, message="Device not connected")
            
            # Use ADB for swipe (like adb_scrcpy.py)
            await self.adb.swipe(
                action.start_x,
                action.start_y,
                action.end_x,
                action.end_y,
                action.duration
            )
            
            # Take screenshot
            screenshot_path = await self._take_screenshot()
            
            return SwipeResult(
                success=True,
                message=f"Swiped from ({action.start_x}, {action.start_y}) to ({action.end_x}, {action.end_y})",
                screenshot_path=screenshot_path
            )
            
        except Exception as e:
            return SwipeResult(success=False, message=f"Swipe failed: {e}")
    
    async def press(self, action: PressRequest) -> PressResult:
        """Perform a long press action on the device."""
        try:
            if not self.is_connected:
                return PressResult(success=False, message="Device not connected")
            
            # Use ADB for long press (like adb_scrcpy.py - swipe to same position)
            await self.adb.press(action.x, action.y, action.duration)
            
            # Take screenshot
            screenshot_path = await self._take_screenshot()
            
            return PressResult(
                success=True,
                message=f"Long pressed at ({action.x}, {action.y}) for {action.duration}ms",
                screenshot_path=screenshot_path
            )
            
        except Exception as e:
            return PressResult(success=False, message=f"Long press failed: {e}")
    
    async def type_text(self, action: TypeTextRequest) -> TypeTextResult:
        """Input text on the device."""
        try:
            if not self.is_connected:
                return TypeTextResult(success=False, message="Device not connected")
            
            # Use ADB for text input (like adb_scrcpy.py)
            await self.adb.type_text(action.text)
            
            # Take screenshot
            screenshot_path = await self._take_screenshot()
            
            return TypeTextResult(
                success=True,
                message=f"Input text: {action.text}",
                screenshot_path=screenshot_path
            )
            
        except Exception as e:
            return TypeTextResult(success=False, message=f"Text input failed: {e}")
    
    async def key_event(self, action: KeyEventRequest) -> KeyEventResult:
        """Press a key on the device."""
        try:
            if not self.is_connected:
                return KeyEventResult(success=False, message="Device not connected")
            
            # Use ADB for key press (like adb_scrcpy.py)
            await self.adb.key_event(action.keycode)
            
            # Take screenshot
            screenshot_path = await self._take_screenshot()
            
            return KeyEventResult(
                success=True,
                message=f"Pressed key: {action.keycode}",
                screenshot_path=screenshot_path
            )
            
        except Exception as e:
            return KeyEventResult(success=False, message=f"Key press failed: {e}")
    
    async def swipe_path(self, action: SwipePathRequest) -> SwipePathResult:
        """Perform a swipe along a path."""
        try:
            if not self.is_connected:
                return SwipePathResult(success=False, message="Device not connected")
            
            if len(action.path) < 2:
                return SwipePathResult(success=False, message="Path must have at least 2 points")
            
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
                await asyncio.sleep(0.1)
            
            # Take screenshot
            screenshot_path = await self._take_screenshot()
            
            return SwipePathResult(
                success=True,
                message=f"Swiped along path with {len(action.path)} points",
                screenshot_path=screenshot_path
            )
            
        except Exception as e:
            return SwipePathResult(success=False, message=f"Swipe path failed: {e}")
        
    async def scroll(self, action: ScrollRequest) -> ScrollResult:
        """Perform a scroll action on the device."""
        try:
            if not self.is_connected:
                return ScrollResult(success=False, message="Device not connected")
            
            await self.adb.scroll(action.direction, action.distance)
            
            return ScrollResult(success=True, message=f"Scrolled {action.direction} by {action.distance} pixels")
            
        except Exception as e:
            return ScrollResult(success=False, message=f"Scroll failed: {e}")
    
    async def take_screenshot(self, action: ScreenshotRequest) -> ScreenshotResult:
        """Take a screenshot of the device."""
        try:
            if not self.is_connected:
                return ScreenshotResult(success=False, message="Device not connected")
            
            screenshot_path = await self._take_screenshot(save_path=action.save_path)
            
            return ScreenshotResult(
                success=True,
                message="Screenshot taken",
                screenshot_path=screenshot_path
            )
            
        except Exception as e:
            return ScreenshotResult(success=False, message=f"Screenshot failed: {e}")
    
    async def _take_screenshot(self, save_path: Optional[str] = None) -> str:
        """Take a screenshot and save it."""
        if save_path is None:
            timestamp = int(time.time() * 1000)
            save_path = self.base_dir / f"screenshot_{timestamp}.png"
        
        # Use Minicap for high-quality screenshots (like adb_scrcpy.py)
        try:
            screenshot_data = self.minicap.get_screenshot_bytes()
            
            # Save screenshot
            with open(save_path, 'wb') as f:
                f.write(screenshot_data)
        except Exception as e:
            print(f"Screenshot failed, falling back to ADB: {e}")
            # Fallback to ADB screenshot
            screenshot = await self.adb.get_screenshot()
            if screenshot:
                screenshot.save(save_path)
        
        # Update screenshot paths
        self.previous_screenshot_path = self.current_screenshot_path
        self.current_screenshot_path = str(save_path)
        
        return str(save_path)
    
    async def get_device_state(self) -> MobileDeviceState:
        """Get current device state."""
        return MobileDeviceState(
            device_info=self.device_info or MobileDeviceInfo(
                device_id="unknown",
                screen_width=0,
                screen_height=0,
                screen_density=0.0,
                is_connected=False
            ),
            screenshot_path=self.current_screenshot_path,
            is_recording=self.is_recording,
            recording_path=self.recording_path
        )
    
    async def pause_recording(self) -> None:
        """Pause video recording (like adb_scrcpy.py)."""
        if self.minicap and self.is_recording:
            self.minicap.pause()
    
    async def resume_recording(self) -> None:
        """Resume video recording (like adb_scrcpy.py)."""
        if self.minicap and self.is_recording:
            self.minicap.unpause()
    
    # Additional methods following adb_scrcpy.py pattern
    def get_screen_size(self) -> Tuple[int, int]:
        """Get screen size (like adb_scrcpy.py)."""
        return self.screen_size
    
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

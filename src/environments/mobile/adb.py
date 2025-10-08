"""ADB bridge for device control."""

import asyncio
import urllib.parse
from typing import Optional, Tuple, Dict, Any
from PIL.Image import Image
import adbutils


class AdbDriver:
    """
    ADB bridge for controlling Android devices via ADB (Android Debug Bridge).

    This class encapsulates common ADB commands such as input simulation, tapping, swiping, key events, and screenshot capture.
    It allows interaction with connected Android devices through Python.

    Attributes:
        device: ADB device instance for communication.
        device_id: Device ID or serial number to target specific devices.
    """

    def __init__(self, device_id: Optional[str] = None, serial_number: Optional[str] = None):
        """
        Initialize the ADBBridge instance.

        Args:
            device_id: Device ID to target specific devices. Defaults to None.
            serial_number: Device serial number (overrides device_id). Defaults to None.
        """
        self.device_id = device_id
        self.serial_number = serial_number
        self.device = None
        self._init_device()

    def _init_device(self):
        """
        Initialize and connect to the specified Android device.

        Raises:
            Exception: If device connection fails.
        """
        try:
            if self.serial_number:
                self.device = adbutils.adb.device(serial=self.serial_number)
            elif self.device_id:
                self.device = adbutils.adb.device(serial=self.device_id)
            else:
                devices = adbutils.adb.device_list()
                if devices:
                    self.device = devices[0]
                else:
                    raise Exception("No ADB devices connected")
        except Exception as e:
            print(f"ADB connection failed: {e}")
            self.device = None

    def device_list(self) -> list:
        """
        Get list of connected ADB devices.

        Returns:
            list: List of ADBDevice instances representing connected devices.
        """
        try:
            import adbutils
            return adbutils.adb.device_list()
        except Exception as e:
            print(f"Device listing failed: {e}")
            return []

    def check_activity(self, activity_name: str) -> bool:
        """
        Check if current activity contains specified name.

        Args:
            activity_name: Target activity name or keyword to search for.

        Returns:
            bool: True if activity found, else False.
        """
        result = self.shell("dumpsys activity activities")
        return activity_name in result if result else False

    # ==================== BASIC INPUT ACTIONS ====================
    
    async def tap(self, x: int, y: int):
        """Tap at specified coordinates."""
        await self.shell(f"input tap {x} {y}")
        
    async def key_event(self, keycode: int):
        """Press a key by keycode.
        
        Args:
            keycode: Android keycode to press.
             - https://developer.android.com/reference/android/view/KeyEvent.html
        """
        await self.shell(f"input keyevent {keycode}")

    async def press(self, x: int, y: int, duration: int = 1000):
        """Long press at specified coordinates."""
        await self.shell(f"input swipe {x} {y} {x} {y} {duration}")

    async def swipe(self, start_x: int, start_y: int, end_x: int, end_y: int, duration: int = 300):
        """Swipe gesture from start to end coordinates."""
        await self.shell(f"input swipe {start_x} {start_y} {end_x} {end_y} {duration}")

    async def type_text(self, text: str):
        """Type text on device."""
        encoded_text = urllib.parse.quote(text)
        await self.shell(f"input text {encoded_text}")

    async def clear(self):
        """Clear current text input field."""
        await self.shell("input keyevent KEYCODE_CTRL_A")
        await self.shell("input keyevent KEYCODE_DEL")
        
    # ==================== SCROLL ACTIONS ====================
    async def scroll(self, direction: str, distance: int = 500):
        """
        Scroll on screen in specified direction.
        
        Args:
            direction: Scroll direction ("up", "down", "left", "right")
            distance: Scroll distance in pixels
        """
        screen_size = await self.get_screen_size()
        x, y = screen_size[0] // 2, screen_size[1] // 2
        
        if direction.lower() == "up":
            await self.shell(f"input swipe {x} {y} {x} {y - distance} 300")
        elif direction.lower() == "down":
            await self.shell(f"input swipe {x} {y} {x} {y + distance} 300")
        elif direction.lower() == "left":
            await self.shell(f"input swipe {x} {y} {x - distance} {y} 300")
        elif direction.lower() == "right":
            await self.shell(f"input swipe {x} {y} {x + distance} {y} 300")
        else:
            raise ValueError(f"Invalid scroll direction: {direction}. Use 'up', 'down', 'left', or 'right'")

    # ==================== SYSTEM ACTIONS ====================
    async def wake_up(self):
        """Wake up the device."""
        await self.shell("input keyevent KEYCODE_WAKEUP")

    async def unlock_screen(self):
        """Unlock the screen."""
        await self.shell("input keyevent KEYCODE_WAKEUP")
        await self.shell("input swipe 500 1000 500 500")  # Swipe up to unlock

    async def open_app(self, package_name: str):
        """Open an app by package name."""
        await self.shell(f"monkey -p {package_name} -c android.intent.category.LAUNCHER 1")

    async def close_app(self, package_name: str):
        """Close an app by package name."""
        await self.shell(f"am force-stop {package_name}")

    async def get_screenshot(self) -> Optional[Image]:
        """
        Capture device screenshot.

        Returns:
            Image.Image: PIL Image object of the screenshot, or None on error.
        """
        try:
            return await asyncio.get_event_loop().run_in_executor(None, self.device.screenshot)
        except Exception as e:
            print(f"Screenshot failed: {e}")
            return None

    async def get_screen_info(self) -> Dict[str, Any]:
        """
        Get device screen resolution.

        Returns:
            Dict[str, Any]: (width, height) in pixels, or (0,0) on error.
        """
        
        # adb shell dumpsys input | grep 'SurfaceOrientation'
        result = await self.shell("dumpsys input | grep 'SurfaceOrientation'")
        
        rotate = 0
        if "SurfaceOrientation: 0" in result: # portrait
            rotate = 0
        elif "SurfaceOrientation: 1" in result: # landscape
            rotate = 90
        elif "SurfaceOrientation: 2" in result: # portrait upside down
            rotate = 180
        elif "SurfaceOrientation: 3" in result: # landscape upside down
            rotate = -90
            
        # adb shell wm size
        result = await self.shell("wm size")
        if not result:
            return {
                "width": 0,
                "height": 0,
                "rotate": rotate
            }
        parts = result.strip().split(": ")[-1].split("x")
        screen_size = tuple(map(int, parts))
        
        if rotate == 0 or rotate == 180:
            return {
                "width": screen_size[0],
                "height": screen_size[1],
                "rotate": rotate
            }
        elif rotate == 90 or rotate == -90:
            return {
                "width": screen_size[1],
                "height": screen_size[0],
                "rotate": rotate
            }
            
        return {
            "width": 0,
            "height": 0,
            "rotate": rotate
        }

    async def get_screen_density(self) -> int:
        """
        Get device screen density (DPI).

        Returns:
            int: Screen density value in DPI, or 0 on error.
        """
        result = await self.shell("wm density")
        if not result:
            return 0
        return int(result.strip().split(": ")[-1])

    # ==================== DEVICE INFO ====================
    async def get_device_info(self) -> dict:
        """Get basic device information."""
        return {
            'model': await self.shell("getprop ro.product.model"),
            'version': await self.shell("getprop ro.build.version.release"),
            'screen_info': await self.get_screen_info(),
            'screen_density': await self.get_screen_density()
        }

    async def get_current_activity(self) -> str:
        """Get current activity name."""
        result = await self.shell("dumpsys activity activities | grep -E 'mResumedActivity|mFocusedActivity'")
        if result:
            lines = result.strip().split('\n')
            for line in lines:
                if 'mResumedActivity' in line or 'mFocusedActivity' in line:
                    parts = line.split()
                    for part in parts:
                        if '/' in part and '.' in part:
                            return part
        return ""

    async def shell(self, command: str) -> Optional[str]:
        """
        Execute ADB shell command asynchronously.

        Args:
            command: Shell command to execute.

        Returns:
            str or None: Command output or None on failure.
        """
        try:
            return await asyncio.get_event_loop().run_in_executor(None, self.device.shell, command)
        except Exception as e:
            print(f"Command '{command}' failed: {e}")
            return None

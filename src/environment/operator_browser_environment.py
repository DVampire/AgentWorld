"""Operator Browser Environment for AgentWorld - provides browser automation as an environment."""

from tkinter import NO
from typing import Any, Dict, List, Union, Optional, Type
from pydantic import BaseModel, Field, ConfigDict
import os
from PIL import Image
import io

from src.environment.operator_browser.service import OperatorBrowserService
from src.environment.operator_browser.types import (
    ClickRequest,
    DoubleClickRequest,
    ScrollRequest,
    TypeRequest,
    WaitRequest,
    MoveRequest,
    KeypressRequest,
    DragRequest,
)
from src.logger import logger
from src.utils import assemble_project_path, encode_file_base64, decode_file_base64, make_file_url
from src.utils import dedent, ScreenshotService
from src.environment.types import Environment, ScreenshotInfo
from src.environment.server import environment_manager
from src.registry import ENVIRONMENT

@ENVIRONMENT.register_module(force=True)
class OperatorBrowserEnvironment(Environment):
    """Operator Browser Environment that provides browser automation as an environment interface."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="operator_browser", description="The name of the Operator Browser environment.")
    description: str = Field(default="OpenAI Operator compatible browser environment for web automation", description="The description of the Operator Browser environment.")
    metadata: Dict[str, Any] = Field(default={
        "has_vision": True,
        "additional_rules": {
            "state": "The state of the browser environment including current URL, title, and viewport.",
        }
    }, description="The metadata of the Operator Browser environment.")
    require_grad: bool = Field(default=False, description="Whether the environment requires gradients")
    
    def __init__(
        self,
        base_dir: str = None,
        headless: bool = False,
        viewport: Optional[Dict[str, int]] = None,
        require_grad: bool = False,
        **kwargs
    ):
        """
        Initialize the Operator browser environment.
        
        Args:
            base_dir: Base directory for screenshots and logs
            headless: Whether to run browser in headless mode
            viewport: Browser viewport size
        """
        super().__init__(**kwargs)
        self.base_dir = assemble_project_path(base_dir)
        self.headless = headless
        self.viewport = viewport or {"width": 1024, "height": 768}
        
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Initialize the browser service
        self.operator_browser_service = OperatorBrowserService(
            base_dir=self.base_dir,
            headless=self.headless,
            viewport=self.viewport
        )
        
        # Initialize step counter for screenshots
        self.step_number = 0
        self.screenshot: ScreenshotInfo = None
        self.previous_screenshot: ScreenshotInfo = None
        self.screenshot_service = ScreenshotService(base_dir=self.base_dir)
    
    async def initialize(self) -> None:
        """Initialize the Operator Browser environment."""
        await self.operator_browser_service.start()
        logger.info(f"| 🌐 Operator Browser Environment initialized at: {self.base_dir}")
        
    async def cleanup(self) -> None:
        """Cleanup the Operator Browser environment."""
        await self.operator_browser_service.stop()
        logger.info("| 🧹 Operator Browser Environment cleanup completed")
    
    @environment_manager.action(
        name="click",
        description="Click at specified coordinates on the page",
    )
    async def click(self, x: int, y: int, button: str = "left", **kwargs) -> Dict[str, Any]:
        """Click at specified coordinates on the page.
        
        Args:
            x (int): X coordinate to click
            y (int): Y coordinate to click
            button (str): Mouse button to click (left, right, middle)
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = ClickRequest(x=x, y=y, button=button)
            
            # Draw a cursor on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_click.png'
            screenshot = self.screenshot.screenshot
            screenshot_image = Image.open(io.BytesIO(decode_file_base64(screenshot)))
            screenshot_image = await self.screenshot_service.draw_cursor(screenshot_image, x, y)
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot_image, self.step_number, screenshot_filename)
            screenshot_description = f"Action: Click at ({x}, {y}) with {button} button"
            self.previous_screenshot = ScreenshotInfo(
                transformed=False,
                screenshot=encode_file_base64(file_path=screenshot_path),
                screenshot_path=screenshot_path,
                screenshot_description=screenshot_description,
                transform_info=None
            )
            self.step_number += 1
            
            result = await self.operator_browser_service.click(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["x"] = x
            extra["y"] = y
            extra["button"] = button
            extra["screenshot_path"] = screenshot_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
                
        except Exception as e:
            logger.error(f"| ❌ Click action failed: {e}")
            return {
                "success": False,
                "message": f"Click action failed: {str(e)}",
                "extra": {"error": str(e), "x": x, "y": y, "button": button}
            }
    
    @environment_manager.action(
        name="double_click",
        description="Double click at specified coordinates on the page",
    )
    async def double_click(self, x: int, y: int, button: str = "left", **kwargs) -> Dict[str, Any]:
        """Double click at specified coordinates on the page.
        
        Args:
            x (int): X coordinate to double click
            y (int): Y coordinate to double click
            button (str): Mouse button to double click (left, right, middle)
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = DoubleClickRequest(x=x, y=y, button=button)
            
            # Draw a cursor on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_double_click.png'
            screenshot = self.screenshot.screenshot
            screenshot_image = Image.open(io.BytesIO(decode_file_base64(screenshot)))
            screenshot_image = await self.screenshot_service.draw_cursor(screenshot_image, x, y)
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot_image, self.step_number, screenshot_filename)
            screenshot_description = f"Action: Double click at ({x}, {y}) with {button} button"
            self.previous_screenshot = ScreenshotInfo(
                transformed=False,
                screenshot=encode_file_base64(file_path=screenshot_path),
                screenshot_path=screenshot_path,
                screenshot_description=screenshot_description,
                transform_info=None
            )
            self.step_number += 1
            
            result = await self.operator_browser_service.double_click(request)
            
            extra = result.extra.copy() if result.extra else {}
            extra["x"] = x
            extra["y"] = y
            extra["button"] = button
            extra["screenshot_path"] = screenshot_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
                
        except Exception as e:
            logger.error(f"| ❌ Double click action failed: {e}")
            return {
                "success": False,
                "message": f"Double click action failed: {str(e)}",
                "extra": {"error": str(e), "x": x, "y": y, "button": button}
            }
    
    @environment_manager.action(
        name="scroll",
        description="Scroll at specified coordinates with given offsets",
    )
    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int, **kwargs) -> Dict[str, Any]:
        """Scroll at specified coordinates with given offsets.
        
        Args:
            x (int): X coordinate to scroll at
            y (int): Y coordinate to scroll at
            scroll_x (int): Horizontal scroll offset
            scroll_y (int): Vertical scroll offset
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = ScrollRequest(x=x, y=y, scroll_x=scroll_x, scroll_y=scroll_y)
            result = await self.operator_browser_service.scroll(request)
            
            # Draw scroll on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_scroll.png'
            screenshot = self.screenshot.screenshot
            screenshot_image = Image.open(io.BytesIO(decode_file_base64(screenshot)))
            screenshot_image = await self.screenshot_service.draw_scroll(screenshot_image, x, y, scroll_x, scroll_y)
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot_image, self.step_number, screenshot_filename)
            screenshot_description = f"Action: Scroll at ({x}, {y}) with offset ({scroll_x}, {scroll_y})"
            self.previous_screenshot = ScreenshotInfo(
                transformed=False,
                screenshot=encode_file_base64(file_path=screenshot_path),
                screenshot_path=screenshot_path,
                screenshot_description=screenshot_description,
                transform_info=None
            )
            self.step_number += 1
            
            extra = result.extra.copy() if result.extra else {}
            extra["x"] = x
            extra["y"] = y
            extra["scroll_x"] = scroll_x
            extra["scroll_y"] = scroll_y
            extra["screenshot_path"] = screenshot_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
                
        except Exception as e:
            logger.error(f"| ❌ Scroll action failed: {e}")
            return {
                "success": False,
                "message": f"Scroll action failed: {str(e)}",
                "extra": {"error": str(e), "x": x, "y": y, "scroll_x": scroll_x, "scroll_y": scroll_y}
            }
    
    @environment_manager.action(
        name="type",
        description="Type text at the current cursor position",
    )
    async def type_text(self, text: str, **kwargs) -> Dict[str, Any]:
        """Type text at the current cursor position.
        
        Args:
            text (str): Text to type
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = TypeRequest(text=text)
            result = await self.operator_browser_service.type(request)
            
            # DO NOT draw anything on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_type.png'
            screenshot = self.screenshot.screenshot
            screenshot_image = Image.open(io.BytesIO(decode_file_base64(screenshot)))
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot_image, self.step_number, screenshot_filename)
            screenshot_description = f"Action: Type text: {text}"
            self.previous_screenshot = ScreenshotInfo(
                transformed=False,
                screenshot=encode_file_base64(file_path=screenshot_path),
                screenshot_path=screenshot_path,
                screenshot_description=screenshot_description,
                transform_info=None
            )
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
            logger.error(f"| ❌ Type action failed: {e}")
            return {
                "success": False,
                "message": f"Type action failed: {str(e)}",
                "extra": {"error": str(e), "text": text}
            }
    
    @environment_manager.action(
        name="wait",
        description="Wait for specified milliseconds",
    )
    async def wait(self, ms: int, **kwargs) -> Dict[str, Any]:
        """Wait for specified milliseconds.
        
        Args:
            ms (int): Number of milliseconds to wait
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = WaitRequest(ms=ms)
            result = await self.operator_browser_service.wait(request)
            
            self.last_action_result = result
            
            # DO NOT draw anything on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_wait.png'
            screenshot = self.screenshot.screenshot
            screenshot_image = Image.open(io.BytesIO(decode_file_base64(screenshot)))
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot_image, self.step_number, screenshot_filename)
            screenshot_description = f"Action: Wait for {ms}ms"
            self.previous_screenshot = ScreenshotInfo(
                transformed=False,
                screenshot=encode_file_base64(file_path=screenshot_path),
                screenshot_path=screenshot_path,
                screenshot_description=screenshot_description,
                transform_info=None
            )
            self.step_number += 1
            
            extra = result.extra.copy() if result.extra else {}
            extra["ms"] = ms
            extra["screenshot_path"] = screenshot_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
                
        except Exception as e:
            logger.error(f"| ❌ Wait action failed: {e}")
            return {
                "success": False,
                "message": f"Wait action failed: {str(e)}",
                "extra": {"error": str(e), "ms": ms}
            }
    
    @environment_manager.action(
        name="move",
        description="Move mouse to specified coordinates",
    )
    async def move(self, x: int, y: int, **kwargs) -> Dict[str, Any]:
        """Move mouse to specified coordinates.
        
        Args:
            x (int): X coordinate to move to
            y (int): Y coordinate to move to
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = MoveRequest(x=x, y=y)
            result = await self.operator_browser_service.move(request)
            
            # Draw cursor on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_move.png'
            screenshot = self.screenshot.screenshot
            screenshot_image = Image.open(io.BytesIO(decode_file_base64(screenshot)))
            screenshot_image = await self.screenshot_service.draw_cursor(screenshot_image, x, y)
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot_image, self.step_number, screenshot_filename)
            screenshot_description = f"Action: Move to ({x}, {y})"
            self.previous_screenshot = ScreenshotInfo(
                transformed=False,
                screenshot=encode_file_base64(file_path=screenshot_path),
                screenshot_path=screenshot_path,
                screenshot_description=screenshot_description,
                transform_info=None
            )
            self.step_number += 1
            
            extra = result.extra.copy() if result.extra else {}
            extra["x"] = x
            extra["y"] = y
            extra["screenshot_path"] = screenshot_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
                
        except Exception as e:
            logger.error(f"| ❌ Move action failed: {e}")
            return {
                "success": False,
                "message": f"Move action failed: {str(e)}",
                "extra": {"error": str(e), "x": x, "y": y}
            }
    
    @environment_manager.action(
        name="keypress",
        description="Press specified keys",
    )
    async def keypress(self, keys: List[str], **kwargs) -> Dict[str, Any]:
        """Press specified keys.
        
        Args:
            keys (List[str]): List of keys to press
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = KeypressRequest(keys=keys)
            result = await self.operator_browser_service.keypress(request)
            
            # DO NOT draw anything on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_keypress.png'
            screenshot = self.screenshot.screenshot
            screenshot_image = Image.open(io.BytesIO(decode_file_base64(screenshot)))
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot_image, self.step_number, screenshot_filename)
            screenshot_description = f"Action: Keypress: {keys}"
            self.previous_screenshot = ScreenshotInfo(
                transformed=False,
                screenshot=encode_file_base64(file_path=screenshot_path),
                screenshot_path=screenshot_path,
                screenshot_description=screenshot_description,
                transform_info=None
            )
            self.step_number += 1
            
            extra = result.extra.copy() if result.extra else {}
            extra["keys"] = keys
            extra["screenshot_path"] = screenshot_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
                
        except Exception as e:
            logger.error(f"| ❌ Keypress action failed: {e}")
            return {
                "success": False,
                "message": f"Keypress action failed: {str(e)}",
                "extra": {"error": str(e), "keys": keys}
            }
    
    @environment_manager.action(
        name="drag",
        description="Drag mouse along specified path",
    )
    async def drag(self, path: List[List[int]], **kwargs) -> Dict[str, Any]:
        """Drag mouse along specified path.
        
        Args:
            path (List[List[int]]): Path to drag, e.g., [[x1, y1], [x2, y2]]
            
        Returns:
            Dict with success, message, and extra fields
        """
        try:
            request = DragRequest(path=path)
            result = await self.operator_browser_service.drag(request)
            
            # Draw path on the screenshot
            screenshot_filename = f'step_{self.step_number:04d}_drag.png'
            screenshot = self.screenshot.screenshot
            screenshot_image = Image.open(io.BytesIO(decode_file_base64(screenshot)))
            screenshot_image = await self.screenshot_service.draw_path(screenshot_image, path)
            screenshot_path = await self.screenshot_service.store_screenshot(screenshot_image, self.step_number, screenshot_filename)
            screenshot_description = f"Action: Drag along path with {len(path)} points"
            # Use safe defaults if self.screenshot is None
            transformed = self.screenshot.transformed if self.screenshot else False
            transform_info = self.screenshot.transform_info if self.screenshot else None
            
            self.previous_screenshot = ScreenshotInfo(
                transformed=transformed,
                screenshot=encode_file_base64(file_path=screenshot_path),
                screenshot_path=screenshot_path,
                screenshot_description=screenshot_description,
                transform_info=transform_info
            )
            self.step_number += 1
            
            extra = result.extra.copy() if result.extra else {}
            extra["path"] = path
            extra["screenshot_path"] = screenshot_path
            
            return {
                "success": result.success,
                "message": result.message,
                "extra": extra
            }
                
        except Exception as e:
            logger.error(f"| ❌ Drag action failed: {e}")
            return {
                "success": False,
                "message": f"Drag action failed: {str(e)}",
                "extra": {"error": str(e), "path": path}
            }
        
    async def get_state(self, **kwargs) -> Dict[str, Any]:
        """Get the current state of the browser environment.
        
        Returns:
            Dict containing browser state information
        """
        
        try:
            state = {}
            browser_state = await self.operator_browser_service.get_state()
            
            state = dedent(f"""
                <info>
                Current URL: {browser_state["url"]}
                Current Title: {browser_state["title"]}
                Current Tabs: {browser_state["tabs"]}
                Current Page Info: {browser_state["page_info"]}
                </info>
                """)

            if "screenshot" in browser_state and browser_state["screenshot"]:
                screenshot_data = browser_state["screenshot"]
                screenshot_image = Image.open(io.BytesIO(decode_file_base64(screenshot_data)))
                screenshot_filename = f'step_{self.step_number:04d}_state.png'
                screenshot_path = await self.screenshot_service.store_screenshot(screenshot_image, self.step_number, screenshot_filename)
                screenshot_description = "A screenshot of the browser environment at current step."
                
                self.screenshot = ScreenshotInfo(
                    transformed=False,
                    screenshot=browser_state["screenshot"],
                    screenshot_path=screenshot_path,
                    screenshot_description=screenshot_description,
                    transform_info=None
                )
            
                if not self.previous_screenshot:
                    self.previous_screenshot = self.screenshot
                
                screenshots = [
                    self.previous_screenshot,
                    self.screenshot,
                ]
                
            else:
                screenshots = []
            
            extra = {
                "step_number": self.step_number,
                "environment": "operator_browser_environment",
                "headless": self.headless,
                "viewport": self.viewport,
                "screenshots": screenshots,
                "base_dir": self.base_dir,
                "url": browser_state["url"],
                "title": browser_state["title"],
                "tabs": browser_state["tabs"],
                "page_info": browser_state["page_info"],
            }
            
            self.step_number += 1
            
            return {
                "state": state,
                "extra": extra,
            }
        except Exception as e:
            logger.error(f"| ❌ Error getting browser state: {e}")
            return {
                "state": "Failed to get browser state",
                "extra": dict(error=str(e)),
            }

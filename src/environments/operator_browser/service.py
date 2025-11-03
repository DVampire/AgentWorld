"""OpenAI Computer Use API compatible browser implementation."""

import base64
from typing import Dict, Any, Optional
import os
import asyncio

# Enable debug mode for observe_debug decorators
os.environ['DEBUG'] = '1'
os.environ['BROWSER_USE_DEBUG'] = '1'
os.environ['LMNR_LOGGING_LEVEL'] = 'debug'

from src.logger import logger
from src.environments.browser import Browser 
from src.environments.protocol.types import ActionResult
from src.environments.operator_browser.types import (
    ClickRequest,
    DoubleClickRequest,
    ScrollRequest,
    TypeRequest,
    WaitRequest,
    MoveRequest,
    KeypressRequest,
    DragRequest,
)
from src.environments.browser.browser.session import DEFAULT_BROWSER_PROFILE

class OperatorBrowserService:
    """Browser implementation compatible with OpenAI Operator Browser API."""
    
    def __init__(self, 
                 base_dir: str,
                 headless: bool = True, 
                 viewport: Dict[str, int] = None
                 ):
        """Initialize the browser.
        
        Args:
            headless: Whether to run browser in headless mode (default: True for API)
            viewport: Browser viewport size
        """
        self.base_dir = base_dir
        self.headless = headless
        self.viewport = viewport or {"width": 1024, "height": 768}
        self.browser: Optional[Browser] = None
        
        
    async def start(self):
        """Start the browser with OpenAI Computer Use API compatible settings."""
        try:
            self.browser = Browser(
                browser_profile=DEFAULT_BROWSER_PROFILE,
                headless=self.headless,
                viewport=self.viewport,
                window_size=self.viewport,
                highlight_elements=False,
            )
            
            # Add timeout for browser start
            await self.browser.start()
            
            self.page = await self.browser.get_current_page()
            
            await self.page.goto("https://www.google.com")
            
            await asyncio.sleep(2) # Wait for the browser to be ready
            
            logger.info("| 🌐 Operator started successfully")
            
        except Exception as e:
            logger.error(f"| ❌ Failed to start browser: {e}")
            raise
    
    async def stop(self):
        """Stop the browser."""
        try:
            if self.browser:
                await self.browser.stop()
                self.browser = None
                self.page = None
                
            logger.info("| 🛑 Operator stopped")
            
        except Exception as e:
            logger.error(f"| ❌ Error stopping browser: {e}")
    
    async def execute(self, 
                      action: Dict[str, Any]) -> ActionResult:
        """Execute an action on the current page.
        
        Args:
            action: The action to execute
            
        Returns:
            Action result
        """
        try:
            if action.type == "click":
                action = ClickRequest(**action)
                response = await self.click(action)
                return response 
            elif action.type == "double_click":
                action = DoubleClickRequest(**action)
                response = await self.double_click(action)
                return response
            elif action.type == "scroll":
                action = ScrollRequest(**action)
                response = await self.scroll(action)
                return response
            elif action.type == "type":
                action = TypeRequest(**action)
                response = await self.type(action)
                return response
            elif action.type == "wait":
                action = WaitRequest(**action)
                response = await self.wait(action)
                return response
            elif action.type == "move":
                action = MoveRequest(**action)
                response = await self.move(action)
                return response
            elif action.type == "keypress":
                action = KeypressRequest(**action)
                response = await self.keypress(action)
                return response
            elif action.type == "drag":
                action = DragRequest(**action)
                response = await self.drag(action)
                return response
            else:
                raise ValueError(f"Invalid action type: {action.type}")
        except Exception as e:
            logger.error(f"| ❌ Failed to execute action: {e}")
            return ActionResult(
                success=False,
                message=f"Failed to execute action: {e}",
                extra={"error": str(e)}
            )
    
    async def click(self, action: ClickRequest) -> ActionResult:
        """Click on the current page.
        
        Args:
            action: The action to execute
            
        Returns:
            ActionResult with screenshot and description in extra
        """
        try:
            if not self.browser or not self.page:
                return ActionResult(
                    success=False,
                    message="Browser not available",
                    extra={"error": "Browser not available"}
                )
            
            mouse = await self.page.mouse
            
            # Use BrowserSession's click functionality
            await mouse.click(action.x, action.y, button=action.button, click_count=1)
            
            browser_state = await self.browser.get_browser_state_summary(include_screenshot=True)
            screenshot = browser_state.screenshot
            screenshot_description = f"A screenshot of the current page after clicking at ({action.x}, {action.y}) with {action.button} button."
            
            return ActionResult(
                success=True,
                message=f"Clicked at ({action.x}, {action.y}) with {action.button} button",
                extra={
                    "screenshot": screenshot,
                    "screenshot_description": screenshot_description,
                    "x": action.x,
                    "y": action.y,
                    "button": action.button
                }
            )
            
        except Exception as e:
            logger.error(f"| ❌ Failed to click: {e}")
            return ActionResult(
                success=False,
                message=f"Failed to click: {e}",
                extra={"error": str(e), "x": action.x, "y": action.y, "button": action.button}
            )
        
    async def double_click(self, action: DoubleClickRequest) -> ActionResult:
        """Double click on the current page.
        
        Args:
            action: The action to execute
            
        Returns:
            ActionResult with screenshot and description in extra
        """
        try:
            if not self.browser or not self.page:
                return ActionResult(
                    success=False,
                    message="Browser not available",
                    extra={"error": "Browser not available"}
                )
            
            mouse = await self.page.mouse
            
            await mouse.click(action.x, action.y, button="left", click_count=2)
            
            # Take a screenshot of the current page
            browser_state = await self.browser.get_browser_state_summary(include_screenshot=True)
            screenshot = browser_state.screenshot
            screenshot_description = f"A screenshot of the current page after double clicking at ({action.x}, {action.y})."
            
            return ActionResult(
                success=True,
                message=f"Double clicked at ({action.x}, {action.y})",
                extra={
                    "screenshot": screenshot,
                    "screenshot_description": screenshot_description,
                    "x": action.x,
                    "y": action.y
                }
            )
        except Exception as e:
            logger.error(f"| ❌ Failed to double click: {e}")
            return ActionResult(
                success=False,
                message=f"Failed to double click: {e}",
                extra={"error": str(e), "x": action.x, "y": action.y}
            )
        
    async def scroll(self, action: ScrollRequest) -> ActionResult:
        """Scroll on the current page.
        
        Args:
            action: The action to execute
            
        Returns:
            ActionResult with screenshot and description in extra
        """
        try:
            if not self.browser or not self.page:
                return ActionResult(
                    success=False,
                    message="Browser not available",
                    extra={"error": "Browser not available"}
                )
            
            mouse = await self.page.mouse
            
            await mouse.scroll(action.x, action.y, action.scroll_x, action.scroll_y)
            
            # Take a screenshot of the current page
            browser_state = await self.browser.get_browser_state_summary(include_screenshot=True)
            screenshot = browser_state.screenshot
            screenshot_description = f"A screenshot of the current page after scrolling at ({action.x}, {action.y}) with {action.scroll_x} and {action.scroll_y}."
            
            return ActionResult(
                success=True,
                message=f"Scrolled at ({action.x}, {action.y}) with {action.scroll_x} and {action.scroll_y}",
                extra={
                    "screenshot": screenshot,
                    "screenshot_description": screenshot_description,
                    "x": action.x,
                    "y": action.y,
                    "scroll_x": action.scroll_x,
                    "scroll_y": action.scroll_y
                }
            )
            
        except Exception as e:
            logger.error(f"| ❌ Failed to scroll: {e}")
            return ActionResult(
                success=False,
                message=f"Failed to scroll: {e}",
                extra={"error": str(e), "x": action.x, "y": action.y, "scroll_x": action.scroll_x, "scroll_y": action.scroll_y}
            )
        
    async def type(self, action: TypeRequest) -> ActionResult:
        """Type on the current page.
        
        Args:
            action: The action to execute
            
        Returns:
            ActionResult with screenshot and description in extra
        """
        try:
            if not self.browser or not self.page:
                return ActionResult(
                    success=False,
                    message="Browser not available",
                    extra={"error": "Browser not available"}
                )
            
            # Type text at the current focused element
            keyboard = await self.page.keyboard
            
            await keyboard.type(action.text)
            
            # Take a screenshot of the current page
            browser_state = await self.browser.get_browser_state_summary(include_screenshot=True)
            screenshot = browser_state.screenshot
            screenshot_description = f"A screenshot of the current page after typing {action.text}."
            
            return ActionResult(
                success=True,
                message=f"Typed {action.text}",
                extra={
                    "screenshot": screenshot,
                    "screenshot_description": screenshot_description,
                    "text": action.text
                }
            )
        
        except Exception as e:
            logger.error(f"| ❌ Failed to type: {e}")
            return ActionResult(
                success=False,
                message=f"Failed to type: {e}",
                extra={"error": str(e), "text": action.text}
            )
    
    async def wait(self, action: WaitRequest) -> ActionResult:
        """Wait for the current page.
        
        Args:
            action: The action to execute
            
        Returns:
            ActionResult with screenshot and description in extra
        """
        try:
            if not self.browser or not self.page:
                return ActionResult(
                    success=False,
                    message="Browser not available",
                    extra={"error": "Browser not available"}
                )
            
            await asyncio.sleep(int(action.ms / 1000.0))  # Convert ms to seconds
            
            # Take a screenshot of the current page
            browser_state = await self.browser.get_browser_state_summary(include_screenshot=True)
            screenshot = browser_state.screenshot
            screenshot_description = f"A screenshot of the current page after waiting for {action.ms} ms."
            
            return ActionResult(
                success=True,
                message=f"Waited for {action.ms} ms",
                extra={
                    "screenshot": screenshot,
                    "screenshot_description": screenshot_description,
                    "ms": action.ms
                }
            )
        
        except Exception as e:
            logger.error(f"| ❌ Failed to wait: {e}")
            return ActionResult(
                success=False,
                message=f"Failed to wait: {e}",
                extra={"error": str(e), "ms": action.ms}
            )
    
    async def move(self, action: MoveRequest) -> ActionResult:
        """Move the current page.
        
        Args:
            action: The action to execute
            
        Returns:
            ActionResult with screenshot and description in extra
        """
        try:
            if not self.browser or not self.page:
                return ActionResult(
                    success=False,
                    message="Browser not available",
                    extra={"error": "Browser not available"}
                )
            
            mouse = await self.page.mouse
            
            await mouse.move(action.x, action.y)
            
            # Take a screenshot of the current page
            browser_state = await self.browser.get_browser_state_summary(include_screenshot=True)
            screenshot = browser_state.screenshot
            screenshot_description = f"A screenshot of the current page after moving to ({action.x}, {action.y})."
            
            return ActionResult(
                success=True,
                message=f"Moved to ({action.x}, {action.y})",
                extra={
                    "screenshot": screenshot,
                    "screenshot_description": screenshot_description,
                    "x": action.x,
                    "y": action.y
                }
            )
        except Exception as e:
            logger.error(f"| ❌ Failed to move: {e}")
            return ActionResult(
                success=False,
                message=f"Failed to move: {e}",
                extra={"error": str(e), "x": action.x, "y": action.y}
            )
        
    async def keypress(self, action: KeypressRequest) -> ActionResult:
        """Press a key on the current page.
        
        Args:
            action: The action to execute
            
        Returns:
            ActionResult with screenshot and description in extra
        """
        try:
            if not self.browser or not self.page:
                return ActionResult(
                    success=False,
                    message="Browser not available",
                    extra={"error": "Browser not available"}
                )
            
            keyboard = await self.page.keyboard
            
            await keyboard.press(action.keys)
            
            # Take a screenshot of the current page
            browser_state = await self.browser.get_browser_state_summary(include_screenshot=True)
            screenshot = browser_state.screenshot
            screenshot_description = f"A screenshot of the current page after pressing {action.keys}."
            
            return ActionResult(
                success=True,
                message=f"Pressed {action.keys}",
                extra={
                    "screenshot": screenshot,
                    "screenshot_description": screenshot_description,
                    "keys": action.keys
                }
            )
        
        except Exception as e:
            logger.error(f"| ❌ Failed to keypress: {e}")
            return ActionResult(
                success=False,
                message=f"Failed to keypress: {e}",
                extra={"error": str(e), "keys": action.keys}
            )
        
    async def drag(self, action: DragRequest) -> ActionResult:
        """Drag the current page.
        
        Args:
            action: The action to execute
            
        Returns:
            ActionResult with screenshot and description in extra
        """
        try:
            if not self.browser or not self.page:
                return ActionResult(
                    success=False,
                    message="Browser not available",
                    extra={"error": "Browser not available"}
                )
            
            mouse = await self.page.mouse
            
            await mouse.drag(action.path)
            
            # Take a screenshot of the current page
            browser_state = await self.browser.get_browser_state_summary(include_screenshot=True)
            screenshot = browser_state.screenshot
            screenshot_description = f"A screenshot of the current page after dragging {action.path}."
            
            return ActionResult(
                success=True,
                message=f"Dragged {action.path}",
                extra={
                    "screenshot": screenshot,
                    "screenshot_description": screenshot_description,
                    "path": action.path
                }
            )
        
        except Exception as e:
            logger.error(f"| ❌ Failed to drag: {e}")
            return ActionResult(
                success=False,
                message=f"Failed to drag: {e}",
                extra={"error": str(e), "path": action.path}
            )
    
    async def get_state(self) -> Dict[str, Any]:
        """Take a screenshot of the current page (Operator compatible).
            
        Returns:
            Base64 encoded screenshot string or bytes if save_path provided
        """
        try:
            if not self.browser:
                return {}
            
            # Add timeout to prevent infinite hang
            browser_state = await self.browser.get_browser_state_summary(include_screenshot=True)
            screenshot = browser_state.screenshot
            screenshot_description = "A screenshot of the current page at current step."
            
            state = {
                "url": browser_state.url,
                "title": browser_state.title,
                "tabs": browser_state.tabs,
                "page_info": browser_state.page_info,
                "screenshot": screenshot,
                "screenshot_description": screenshot_description
            }
            
            return state
        
        except Exception as e:
            logger.error(f"| ❌ Screenshot failed: {e}")
            return {
                "url": None,
                "title": None,
                "tabs": None,
                "page_info": None,
                "screenshot": None,
                "screenshot_description": None
            }
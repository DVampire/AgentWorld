"""OpenAI Computer Use API compatible browser implementation."""

import base64
from typing import Dict, Any, Optional
import os
import asyncio

from src.logger import logger
from src.environments.cdp_browser import Browser 
from src.environments.operator_browser.types import (
    ClickRequest,
    ClickResult,
    DoubleClickRequest,
    DoubleClickResult,
    ScrollRequest,
    ScrollResult,
    TypeRequest,
    TypeResult,
    WaitRequest,
    WaitResult,
    MoveRequest,
    MoveResult,
    KeypressRequest,
    KeypressResult,
    DragRequest,
    DragResult,
)
from src.environments.cdp_browser.browser.session import DEFAULT_BROWSER_PROFILE

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
        self.viewport = viewport or {"width": 1280, "height": 720}
        self.browser: Optional[Browser] = None
        
        
    async def start(self):
        """Start the browser with OpenAI Computer Use API compatible settings."""
        try:
            # Create browser session using default profile (like playwright service)
            self.browser = Browser(
                browser_profile=DEFAULT_BROWSER_PROFILE,
                headless=self.headless,
                viewport=self.viewport,
                window_size=self.viewport,
                highlight_elements=False,
                # record_video_dir=os.path.join(self.base_dir, "videos"),
                # record_video_framerate=30,
                # record_video_size=self.viewport,
            )
            await self.browser.start()
            
            self.page = await self.browser.get_current_page()
            
            await self.page.goto("https://www.google.com")
            
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
                      action: Dict[str, Any]) -> str:
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
                return response.message
            elif action.type == "double_click":
                action = DoubleClickRequest(**action)
                response = await self.double_click(action)
                return response.message
            elif action.type == "scroll":
                action = ScrollRequest(**action)
                response = await self.scroll(action)
                return response.message
            elif action.type == "type":
                action = TypeRequest(**action)
                response = await self.type(action)
                return response.message
            elif action.type == "wait":
                action = WaitRequest(**action)
                response = await self.wait(action)
                return response.message
            elif action.type == "move":
                action = MoveRequest(**action)
                response = await self.move(action)
                return response.message
            elif action.type == "keypress":
                action = KeypressRequest(**action)
                response = await self.keypress(action)
                return response.message
            elif action.type == "drag":
                action = DragRequest(**action)
                response = await self.drag(action)
                return response.message
            else:
                raise ValueError(f"Invalid action type: {action.type}")
        except Exception as e:
            logger.error(f"| ❌ Failed to execute action: {e}")
            return f"Failed to execute action: {e}"
    
    async def click(self, action: ClickRequest) -> ClickResult:
        """Click on the current page.
        
        Args:
            action: The action to execute
            
        Returns:
            Click result
        """
        try:
            if not self.browser or not self.page:
                return ClickResult(success=False, message="Browser not available")
            
            mouse = await self.page.mouse
            
            # Use BrowserSession's click functionality
            await mouse.click(action.x, action.y, button=action.button, click_count=1)
            
            browser_state = await self.browser.get_browser_state_summary(include_screenshot=True)
            screenshot = browser_state.screenshot
            screenshot_description = f"A screenshot of the current page after clicking at ({action.x}, {action.y}) with {action.button} button."
            
            result = ClickResult(success=True, 
                              message=f"Clicked at ({action.x}, {action.y}) with {action.button} button", 
                              screenshot=screenshot, 
                              screenshot_description=screenshot_description
                              )
            return result
            
        except Exception as e:
            logger.error(f"| ❌ Failed to click: {e}")
            return ClickResult(success=False, message=f"Failed to click: {e}", screenshot=None, screenshot_description=None)
        
    async def double_click(self, action: DoubleClickRequest) -> DoubleClickResult:
        """Double click on the current page.
        
        Args:
            action: The action to execute
            
        Returns:
            Double click result
        """
        try:
            if not self.browser or not self.page:
                return DoubleClickResult(success=False, message="Browser not available")
            
            mouse = await self.page.mouse
            
            await mouse.click(action.x, action.y, button=action.button, click_count=2)
            
            # Take a screenshot of the current page
            browser_state = await self.browser.get_browser_state_summary(include_screenshot=True)
            screenshot = browser_state.screenshot
            screenshot_description = f"A screenshot of the current page after double clicking at ({action.x}, {action.y}) with {action.button} button."
            
            result = DoubleClickResult(success=True, 
                                      message=f"Double clicked at ({action.x}, {action.y}) with {action.button} button", 
                                      screenshot=screenshot, 
                                      screenshot_description=screenshot_description)
            return result
        except Exception as e:
            logger.error(f"| ❌ Failed to double click: {e}")
            return DoubleClickResult(success=False, message=f"Failed to double click: {e}", screenshot=None, screenshot_description=None)
        
    async def scroll(self, action: ScrollRequest) -> ScrollResult:
        """Scroll on the current page.
        
        Args:
            action: The action to execute
            
        Returns:
            Scroll result
        """
        try:
            if not self.browser or not self.page:
                return ScrollResult(success=False, message="Browser not available")
            
            mouse = await self.page.mouse
            
            await mouse.scroll(action.x, action.y, action.scroll_x, action.scroll_y)
            
            # Take a screenshot of the current page
            browser_state = await self.browser.get_browser_state_summary(include_screenshot=True)
            screenshot = browser_state.screenshot
            screenshot_description = f"A screenshot of the current page after scrolling at ({action.x}, {action.y}) with {action.scroll_x} and {action.scroll_y}."
            
            result = ScrollResult(success=True, 
                                  message=f"Scrolled at ({action.x}, {action.y}) with {action.scroll_x} and {action.scroll_y}", 
                                  screenshot=screenshot, 
                                  screenshot_description=screenshot_description)
            return result
            
        except Exception as e:
            logger.error(f"| ❌ Failed to scroll: {e}")
            return ScrollResult(success=False, message=f"Failed to scroll: {e}", screenshot=None, screenshot_description=None)
        
    async def type(self, action: TypeRequest) -> TypeResult:
        """Type on the current page.
        
        Args:
            action: The action to execute
            
        Returns:
            Type result
        """
        try:
            if not self.browser or not self.page:
                return TypeResult(success=False, message="Browser not available")
            
            # Type text at the current focused element
            keyboard = await self.page.keyboard
            
            await keyboard.type(action.text)
            
            # Take a screenshot of the current page
            browser_state = await self.browser.get_browser_state_summary(include_screenshot=True)
            screenshot = browser_state.screenshot
            screenshot_description = f"A screenshot of the current page after typing {action.text}."
            # Save the screenshot of the current page
            result = TypeResult(success=True, 
                                message=f"Typed {action.text}", 
                                screenshot=screenshot, 
                                screenshot_description=screenshot_description)
            return result
        
        except Exception as e:
            logger.error(f"| ❌ Failed to type: {e}")
            return TypeResult(success=False, message=f"Failed to type: {e}", screenshot=None, screenshot_description=None)
    
    async def wait(self, action: WaitRequest) -> WaitResult:
        """Wait for the current page.
        
        Args:
            action: The action to execute
            
        Returns:
            Wait result
        """
        try:
            if not self.browser or not self.page:
                return WaitResult(success=False, message="Browser not available")
            
            await asyncio.sleep(int(action.ms / 1000.0))  # Convert ms to seconds
            
            # Take a screenshot of the current page
            browser_state = await self.browser.get_browser_state_summary(include_screenshot=True)
            screenshot = browser_state.screenshot
            screenshot_description = f"A screenshot of the current page after waiting for {action.ms} ms."
            # Save the screenshot of the current page
            result = WaitResult(success=True, 
                                message=f"Waited for {action.ms} ms", 
                                screenshot=screenshot, 
                                screenshot_description=screenshot_description)
            return result
        
        except Exception as e:
            logger.error(f"| ❌ Failed to wait: {e}")
            return WaitResult(success=False, message=f"Failed to wait: {e}", screenshot=None, screenshot_description=None)
    
    async def move(self, action: MoveRequest) -> MoveResult:
        """Move the current page.
        
        Args:
            action: The action to execute
            
        Returns:
            Move result
        """
        try:
            if not self.browser or not self.page:
                return MoveResult(success=False, message="Browser not available")
            
            mouse = await self.page.mouse
            
            await mouse.move(action.x, action.y)
            
            # Take a screenshot of the current page
            browser_state = await self.browser.get_browser_state_summary(include_screenshot=True)
            screenshot = browser_state.screenshot
            screenshot_description = f"A screenshot of the current page after moving to ({action.x}, {action.y})."
            # Save the screenshot of the current page
            result = MoveResult(success=True, 
                                message=f"Moved to ({action.x}, {action.y})", 
                                screenshot=screenshot, 
                                screenshot_description=screenshot_description)
            return result
        except Exception as e:
            logger.error(f"| ❌ Failed to move: {e}")
            return MoveResult(success=False, message=f"Failed to move: {e}", screenshot=None, screenshot_description=None)
        
    async def keypress(self, action: KeypressRequest) -> KeypressResult:
        """Press a key on the current page.
        
        Args:
            action: The action to execute
            
        Returns:
            Keypress result
        """
        try:
            if not self.browser or not self.page:
                return KeypressResult(success=False, message="Browser not available")
            
            keyboard = await self.page.keyboard
            
            await keyboard.press(action.keys)
            
            # Take a screenshot of the current page
            browser_state = await self.browser.get_browser_state_summary(include_screenshot=True)
            screenshot = browser_state.screenshot
            screenshot_description = f"A screenshot of the current page after pressing {action.keys}."
            
            result = KeypressResult(success=True, 
                                message=f"Pressed {action.keys}", 
                                screenshot=screenshot, 
                                screenshot_description=screenshot_description)
            return result
        
        except Exception as e:
            logger.error(f"| ❌ Failed to keypress: {e}")
            return KeypressResult(success=False, message=f"Failed to keypress: {e}", screenshot=None, screenshot_description=None)
        
    async def drag(self, action: DragRequest) -> DragResult:
        """Drag the current page.
        
        Args:
            action: The action to execute
            
        Returns:
            Drag result
        """
        try:
            if not self.browser or not self.page:
                return DragResult(success=False, message="Browser not available")
            
            mouse = await self.page.mouse
            
            await mouse.drag(action.path)
            
            # Take a screenshot of the current page
            browser_state = await self.browser.get_browser_state_summary(include_screenshot=True)
            screenshot = browser_state.screenshot
            screenshot_description = f"A screenshot of the current page after dragging {action.path}."
            result = DragResult(success=True, 
                                message=f"Dragged {action.path}", 
                                screenshot=screenshot, 
                                screenshot_description=screenshot_description)
            return result
        
        except Exception as e:
            logger.error(f"| ❌ Failed to drag: {e}")
            return DragResult(success=False, message=f"Failed to drag: {e}", screenshot=None, screenshot_description=None)
    
    async def get_state(self) -> Dict[str, Any]:
        """Take a screenshot of the current page (Operator compatible).
            
        Returns:
            Base64 encoded screenshot string or bytes if save_path provided
        """
        try:
            if not self.browser:
                return {}
            browser_state = await self.browser.get_browser_state_summary(include_screenshot=True)
            screenshot = browser_state.screenshot
            screenshot_description = f"A screenshot of the current page at current step."
            
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
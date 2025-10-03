"""OpenAI Computer Use API compatible browser implementation."""

import base64
from typing import Dict, Any, Optional, Union

from src.logger import logger
from src.environments.playwright import Browser 
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
from src.environments.playwright.browser.session import DEFAULT_BROWSER_PROFILE
from src.environments.playwright.browser.profile import ViewportSize

class OperatorBrowserService:
    """Browser implementation compatible with OpenAI Operator Browser API."""
    
    def __init__(self, headless: bool = True, viewport: Dict[str, int] = None):
        """Initialize the browser.
        
        Args:
            headless: Whether to run browser in headless mode (default: True for API)
            viewport: Browser viewport size
        """
        self.headless = headless
        self.viewport = viewport or {"width": 1280, "height": 720}
        self.browser: Optional[Browser] = None
        
    async def start(self):
        """Start the browser with OpenAI Computer Use API compatible settings."""
        try:
            # Create browser session using default profile (like playwright service)
            DEFAULT_BROWSER_PROFILE.window_size = ViewportSize(width=self.viewport["width"], height=self.viewport["height"])
            self.browser = Browser(
                browser_profile=DEFAULT_BROWSER_PROFILE,
                headless=self.headless,
                viewport=self.viewport,
                highlight_elements=False,
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
            
            return ClickResult(success=True, message=f"Clicked at ({action.x}, {action.y}) with {action.button} button")
        except Exception as e:
            logger.error(f"| ❌ Failed to click: {e}")
            return ClickResult(success=False, message=f"Failed to click: {e}")
        
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
            
            return DoubleClickResult(success=True, message=f"Double clicked at ({action.x}, {action.y}) with {action.button} button")
        except Exception as e:
            logger.error(f"| ❌ Failed to double click: {e}")
            return DoubleClickResult(success=False, message=f"Failed to double click: {e}")
        
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
            return ScrollResult(success=True, message=f"Scrolled at ({action.x}, {action.y}) with {action.scroll_x} and {action.scroll_y}")
        except Exception as e:
            logger.error(f"| ❌ Failed to scroll: {e}")
            return ScrollResult(success=False, message=f"Failed to scroll: {e}")
        
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
            
            return TypeResult(success=True, message=f"Typed {action.text}")
        except Exception as e:
            logger.error(f"| ❌ Failed to type: {e}")
            return TypeResult(success=False, message=f"Failed to type: {e}")
    
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
            
            await self.page.wait_for_timeout(action.ms)
            
            return WaitResult(success=True, message=f"Waited for {action.ms} ms")
        except Exception as e:
            logger.error(f"| ❌ Failed to wait: {e}")
            return WaitResult(success=False, message=f"Failed to wait: {e}")
    
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
            return MoveResult(success=True, message=f"Moved to ({action.x}, {action.y})")
        except Exception as e:
            logger.error(f"| ❌ Failed to move: {e}")
            return MoveResult(success=False, message=f"Failed to move: {e}")
        
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
            
            return KeypressResult(success=True, message=f"Pressed {action.keys}")
        except Exception as e:
            logger.error(f"| ❌ Failed to keypress: {e}")
            return KeypressResult(success=False, message=f"Failed to keypress: {e}")
        
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
            return DragResult(success=True, message=f"Dragged {action.path}")
        except Exception as e:
            logger.error(f"| ❌ Failed to drag: {e}")
            return DragResult(success=False, message=f"Failed to drag: {e}")
    
            
    async def get_state(self, screenshot_path: str = None) -> Dict[str, Any]:
        """Take a screenshot of the current page (Operator compatible).
        
        Args:
            screenshot_path: Optional path to save screenshot file
            
        Returns:
            Base64 encoded screenshot string or bytes if save_path provided
        """
        try:
            if not self.browser:
                return {}
            
            # Use BrowserSession's built-in screenshot method
            browser_state = await self.browser.get_browser_state_summary(include_screenshot=True)
            
            if screenshot_path:
                with open(screenshot_path, "wb") as f:
                    f.write(base64.b64decode(browser_state.screenshot))
            
            state = {
                "url": browser_state.url,
                "title": browser_state.title,
                "tabs": [tab.model_dump() for tab in browser_state.tabs],
                "screenshot": browser_state.screenshot,
                "screenshot_path": screenshot_path,
                "page_info": browser_state.page_info.model_dump(),
            }
            
            return state
            
        except Exception as e:
            logger.error(f"| ❌ Screenshot failed: {e}")
            return {}
    
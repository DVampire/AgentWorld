"""OpenAI Computer Use API compatible browser implementation."""

import base64
from typing import Dict, Any, Optional, Union

from playwright.async_api import async_playwright, Browser, Page, BrowserContext

from src.logger import logger
from src.environments.openai_browser.types import (
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

class OpenAIBrowserService:
    """Browser implementation compatible with OpenAI Computer Use API."""
    
    def __init__(self, headless: bool = True, viewport: Dict[str, int] = None):
        """Initialize the browser.
        
        Args:
            headless: Whether to run browser in headless mode (default: True for API)
            viewport: Browser viewport size
        """
        self.headless = headless
        self.viewport = viewport or {"width": 1280, "height": 720}
        self.browser: Optional[Browser] = None
        self.context: Optional[BrowserContext] = None
        self.page: Optional[Page] = None
        self.playwright = None
        
    async def start(self):
        """Start the browser with OpenAI Computer Use API compatible settings."""
        try:
            self.playwright = await async_playwright().start()
            
            # Launch browser with Computer Use API compatible settings
            self.browser = await self.playwright.chromium.launch(
                headless=self.headless,
            args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-web-security",
                    "--disable-features=VizDisplayCompositor",
                    "--disable-extensions",
                    "--disable-plugins",
                    "--disable-images",  # Faster loading for API use
                    "--disable-javascript",  # Can be enabled per page if needed
                ]
            )
            
            # Create context with Computer Use API settings
            self.context = await self.browser.new_context(
                viewport=self.viewport,
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                java_script_enabled=True,
                accept_downloads=False,
                has_touch=False,
                is_mobile=False,
                locale="en-US",
                timezone_id="America/New_York"
            )
            
            self.page = await self.context.new_page()
            
            # Set additional page settings for Computer Use API
            await self.page.set_extra_http_headers({
                "Accept-Language": "en-US,en;q=0.9"
            })
            
            logger.info("| 🌐 OpenAI Computer Use Browser started successfully")
            
        except Exception as e:
            logger.error(f"| ❌ Failed to start browser: {e}")
            raise
    
    async def stop(self):
        """Stop the browser."""
        try:
            if self.page:
                await self.page.close()
            if self.context:
                await self.context.close()
            if self.browser:
                await self.browser.close()
            if self.playwright:
                await self.playwright.stop()
                
            logger.info("| 🛑 OpenAI Browser stopped")
            
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
            response = await self.page.mouse.click(action.x, action.y, button=action.button)
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
            response = await self.page.mouse.dblclick(action.x, action.y, button=action.button)
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
            response = await self.page.mouse.scroll(action.x, action.y, action.scroll_x, action.scroll_y)
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
            response = await self.page.type(action.text)
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
            await self.page.mouse.move(action.x, action.y)
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
            await self.page.keyboard.press(action.keys)
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
            await self.page.mouse.drag(action.path)
            return DragResult(success=True, message=f"Dragged {action.path}")
        except Exception as e:
            logger.error(f"| ❌ Failed to drag: {e}")
            return DragResult(success=False, message=f"Failed to drag: {e}")
            
    async def screenshot(self, full_page: bool = False, save_path: Optional[str] = None) -> Union[str, bytes]:
        """Take a screenshot of the current page (Computer Use API compatible).
        
        Args:
            full_page: Whether to capture the full page
            save_path: Optional path to save screenshot file
            
        Returns:
            Base64 encoded screenshot string or bytes if save_path provided
        """
        try:
            # Wait for page to be stable before taking screenshot
            await self.page.wait_for_load_state("networkidle", timeout=5000)
            
            # Take screenshot with Computer Use API compatible settings
            screenshot_options = {
                "full_page": full_page,
                "type": "png",
                "animations": "disabled",  # Disable animations for consistent screenshots
                "caret": "hide",  # Hide text cursor
                "scale": "css"  # Use CSS scaling
            }
            
            if save_path:
                screenshot_options["path"] = save_path
                screenshot_b64 = await self.page.screenshot(**screenshot_options)
                logger.info(f"| 📸 Screenshot saved to: {save_path}")
                return screenshot_b64
            else:
                screenshot_bytes = await self.page.screenshot(**screenshot_options)
                screenshot_b64 = base64.b64encode(screenshot_bytes).decode('utf-8')
                logger.info(f"| 📸 Screenshot captured: {len(screenshot_b64)} characters")
                return screenshot_b64
            
        except Exception as e:
            logger.error(f"| ❌ Screenshot failed: {e}")
            return "" if not save_path else ""
    
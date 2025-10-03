"""Operator Browser Environment for AgentWorld - provides browser automation as an environment."""

from typing import Any, Dict, List, Union, Optional, Type
from pydantic import BaseModel, Field, ConfigDict
import os

from src.environments.operator_browser.service import OperatorBrowserService
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
from src.logger import logger
from src.utils import assemble_project_path
from src.utils import dedent
from src.environments.protocol.server import ecp
from src.environments.protocol.environment import BaseEnvironment
from src.environments.protocol.types import EnvironmentState

@ecp.environment()
class OperatorBrowserEnvironment(BaseEnvironment):
    """Operator Browser Environment that provides browser automation as an environment interface."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="operator_browser", description="The name of the Operator Browser environment.")
    type: str = Field(default="Operator Browser Environment", description="The type of the Operator Browser environment.")
    description: str = Field(default="OpenAI Operator compatible browser environment for web automation", description="The description of the Operator Browser environment.")
    args_schema: Type[BaseModel] = Field(default=None, description="The args schema of the Operator Browser environment.")
    metadata: Dict[str, Any] = Field(default={
        "has_vision": True,
        "additional_rules": {
            "state": "The state of the browser environment including current URL, title, and viewport.",
        }
    }, description="The metadata of the Operator Browser environment.")
    
    def __init__(
        self,
        base_dir: str = None,
        headless: bool = False,
        viewport: Optional[Dict[str, int]] = None,
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
        self.viewport = viewport or {"width": 1280, "height": 720}
        
        os.makedirs(self.base_dir, exist_ok=True)
        
        # Initialize the browser service
        self.operator_browser_service = OperatorBrowserService(
            headless=self.headless,
            viewport=self.viewport
        )
        
        # Initialize step counter for screenshots
        self.step_number = 0
    
    async def initialize(self) -> None:
        """Initialize the Operator Browser environment."""
        await self.operator_browser_service.start()
        logger.info(f"| 🌐 Operator Browser Environment initialized at: {self.base_dir}")
        
    async def cleanup(self) -> None:
        """Cleanup the Operator Browser environment."""
        await self.operator_browser_service.stop()
        logger.info("| 🧹 Operator Browser Environment cleanup completed")
    
    async def get_state(self) -> EnvironmentState:
        """Get the current state of the browser environment.
        
        Returns:
            Dict containing browser state information
        """
        
        try:
            screenshot_path = os.path.join(self.base_dir, f"step_{self.step_number:04d}.png")
            screenshot_b64 = await self.operator_browser_service.screenshot(save_path=screenshot_path)
            self.step_number += 1
            
            state = dedent(f"""
                <info>
                Current URL: {self.operator_browser_service.page.url if self.operator_browser_service.page else None}
                Current Title: {self.operator_browser_service.page.title() if self.operator_browser_service.page else None}
                Browser Ready: {self.operator_browser_service.browser is not None}
                Page Ready: {self.operator_browser_service.page is not None}
                Viewport: {self.viewport}
                </info>
                <img src={screenshot_path}>
            """)
            
            extra = {
                "environment": "operator_browser_environment",
                "headless": self.headless,
                "viewport": self.viewport,
                "screenshot": screenshot_b64,
                "screenshot_path": screenshot_path,
                "base_dir": self.base_dir,
                "browser_ready": self.operator_browser_service.browser is not None,
                "page_ready": self.operator_browser_service.page is not None,
                "current_url": self.operator_browser_service.page.url if self.operator_browser_service.page else None,
                "current_title": self.operator_browser_service.page.title() if self.operator_browser_service.page else None,
            }
            
            return EnvironmentState(
                state=state,
                extra=extra,
            )
        except Exception as e:
            logger.error(f"| ❌ Error getting browser state: {e}")
            return EnvironmentState(
                state="Failed to get browser state",
                extra=dict(error=str(e)),
            )
    
    @ecp.action(
        name="click",
        type="Operator Browser Environment",
        description="Click at specified coordinates on the page",
    )
    async def click(self, x: int, y: int, button: str = "left") -> str:
        """Click at specified coordinates on the page.
        
        Args:
            x (int): X coordinate to click
            y (int): Y coordinate to click
            button (str): Mouse button to click (left, right, middle)
            
        Returns:
            str: Result message
        """
        try:
            request = ClickRequest(x=x, y=y, button=button)
            result = await self.operator_browser_service.click(request)
            
            if result.success:
                return f"✅ Clicked at ({x}, {y}) with {button} button"
            else:
                return f"❌ Click failed: {result.message}"
                
        except Exception as e:
            logger.error(f"| ❌ Click action failed: {e}")
            return f"❌ Click action failed: {str(e)}"
    
    @ecp.action(
        name="double_click",
        description="Double click at specified coordinates on the page",
        type="Operator Browser Environment",
    )
    async def double_click(self, x: int, y: int, button: str = "left") -> str:
        """Double click at specified coordinates on the page.
        
        Args:
            x (int): X coordinate to double click
            y (int): Y coordinate to double click
            button (str): Mouse button to double click (left, right, middle)
            
        Returns:
            str: Result message
        """
        try:
            request = DoubleClickRequest(x=x, y=y, button=button)
            result = await self.operator_browser_service.double_click(request)
            
            if result.success:
                return f"✅ Double clicked at ({x}, {y}) with {button} button"
            else:
                return f"❌ Double click failed: {result.message}"
                
        except Exception as e:
            logger.error(f"| ❌ Double click action failed: {e}")
            return f"❌ Double click action failed: {str(e)}"
    
    @ecp.action(
        name="scroll",
        description="Scroll at specified coordinates with given offsets",
        type="Operator Browser Environment",
    )
    async def scroll(self, x: int, y: int, scroll_x: int, scroll_y: int) -> str:
        """Scroll at specified coordinates with given offsets.
        
        Args:
            x (int): X coordinate to scroll at
            y (int): Y coordinate to scroll at
            scroll_x (int): Horizontal scroll offset
            scroll_y (int): Vertical scroll offset
            
        Returns:
            str: Result message
        """
        try:
            request = ScrollRequest(x=x, y=y, scroll_x=scroll_x, scroll_y=scroll_y)
            result = await self.operator_browser_service.scroll(request)
            
            if result.success:
                return f"✅ Scrolled at ({x}, {y}) with offsets ({scroll_x}, {scroll_y})"
            else:
                return f"❌ Scroll failed: {result.message}"
                
        except Exception as e:
            logger.error(f"| ❌ Scroll action failed: {e}")
            return f"❌ Scroll action failed: {str(e)}"
    
    @ecp.action(
        name="type",
        description="Type text at the current cursor position",
        type="Operator Browser Environment",
    )
    async def type_text(self, text: str) -> str:
        """Type text at the current cursor position.
        
        Args:
            text (str): Text to type
            
        Returns:
            str: Result message
        """
        try:
            request = TypeRequest(text=text)
            result = await self.operator_browser_service.type(request)
            
            if result.success:
                return f"✅ Typed text: {text}"
            else:
                return f"❌ Type failed: {result.message}"
                
        except Exception as e:
            logger.error(f"| ❌ Type action failed: {e}")
            return f"❌ Type action failed: {str(e)}"
    
    @ecp.action(
        name="wait",
        description="Wait for specified milliseconds",
        type="Operator Browser Environment",
    )
    async def wait(self, ms: int) -> str:
        """Wait for specified milliseconds.
        
        Args:
            ms (int): Number of milliseconds to wait
            
        Returns:
            str: Result message
        """
        try:
            request = WaitRequest(ms=ms)
            result = await self.operator_browser_service.wait(request)
            
            if result.success:
                return f"✅ Waited for {ms} ms"
            else:
                return f"❌ Wait failed: {result.message}"
                
        except Exception as e:
            logger.error(f"| ❌ Wait action failed: {e}")
            return f"❌ Wait action failed: {str(e)}"
    
    @ecp.action(
        name="move",
        description="Move mouse to specified coordinates",
        type="Operator Browser Environment",
    )
    async def move(self, x: int, y: int) -> str:
        """Move mouse to specified coordinates.
        
        Args:
            x (int): X coordinate to move to
            y (int): Y coordinate to move to
            
        Returns:
            str: Result message
        """
        try:
            request = MoveRequest(x=x, y=y)
            result = await self.operator_browser_service.move(request)
            
            if result.success:
                return f"✅ Moved mouse to ({x}, {y})"
            else:
                return f"❌ Move failed: {result.message}"
                
        except Exception as e:
            logger.error(f"| ❌ Move action failed: {e}")
            return f"❌ Move action failed: {str(e)}"
    
    @ecp.action(
        name="keypress",
        description="Press specified keys",
        type="Operator Browser Environment",
    )
    async def keypress(self, keys: List[str]) -> str:
        """Press specified keys.
        
        Args:
            keys (List[str]): List of keys to press
            
        Returns:
            str: Result message
        """
        try:
            request = KeypressRequest(keys=keys)
            result = await self.operator_browser_service.keypress(request)
            
            if result.success:
                return f"✅ Pressed keys: {keys}"
            else:
                return f"❌ Keypress failed: {result.message}"
                
        except Exception as e:
            logger.error(f"| ❌ Keypress action failed: {e}")
            return f"❌ Keypress action failed: {str(e)}"
    
    @ecp.action(
        name="drag",
        description="Drag mouse along specified path",
        type="Operator Browser Environment",
    )
    async def drag(self, path: List[List[int]]) -> str:
        """Drag mouse along specified path.
        
        Args:
            path (List[List[int]]): Path to drag, e.g., [[x1, y1], [x2, y2]]
            
        Returns:
            str: Result message
        """
        try:
            request = DragRequest(path=path)
            result = await self.operator_browser_service.drag(request)
            
            if result.success:
                return f"✅ Dragged along path: {path}"
            else:
                return f"❌ Drag failed: {result.message}"
                
        except Exception as e:
            logger.error(f"| ❌ Drag action failed: {e}")
            return f"❌ Drag action failed: {str(e)}"

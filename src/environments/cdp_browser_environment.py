"""Playwright Environment for AgentWorld - provides browser automation operations as an environment."""

from typing import Any, Dict, List, Union, Optional, Type
from pydantic import BaseModel, Field, ConfigDict

from src.environments.browser.service import CDPBrowserService
from src.environments.browser.types import (
    SearchGoogleRequest,
    GoToUrlRequest,
    GoBackRequest,
    WaitRequest,
    ClickElementRequest,
    InputTextRequest,
    ScrollRequest,
    SendKeysRequest,
    ScrollToTextRequest,
    GetDropdownOptionsRequest,
    SelectDropdownOptionRequest,
    UploadFileRequest,
    SwitchTabRequest,
    CloseTabRequest,
    ExtractStructuredDataRequest,
    ExecuteJsRequest,
    ScreenshotRequest,
    StoreScreenshotRequest,
    GetScreenshotRequest,
    BrowserStateRequest
)
from src.logger import logger
from src.utils import assemble_project_path
from src.environments.protocol.server import ecp
from src.environments.protocol.environment import BaseEnvironment

@ecp.environment()
class CDPBrowserEnvironment(BaseEnvironment):
    """Playwright Environment that provides browser automation operations as an environment interface."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="cdp_browser", description="The name of the CDP Browser environment.")
    type: str = Field(default="CDP Browser", description="The type of the CDP Browser environment.")
    description: str = Field(default="CDP Browser automation environment for web interactions", description="The description of the CDP Browser environment.")
    args_schema: Type[BaseModel] = Field(default=None, description="The args schema of the Playwright environment.")
    metadata: Dict[str, Any] = Field(default={
        "has_vision": True,
        "additional_rules": {
            "state": "The state of the browser environment including current URL, title, and available elements.",
        }
    }, description="The metadata of the CDP Browser environment.")
    
    def __init__(
        self,
        base_dir: str = None,
        auto_start: bool = True,
        **kwargs
    ):
        """
        Initialize the CDP Browser environment.
        
        Args:
            base_dir (str): Base directory for storing screenshots and other files
            auto_start (bool): Whether to automatically start the browser session
        """
        super().__init__(**kwargs)
        
        self.base_dir = assemble_project_path(base_dir)
        self.auto_start = auto_start
        
        # Initialize playwright service
        self.playwright_service = CDPBrowserService(
            base_dir=self.base_dir
        )
        
    async def initialize(self) -> None:
        """Initialize the CDP Browser environment."""
        if self.auto_start:
            await self.playwright_service.start()
        logger.info(f"| 🌐 Playwright Environment initialized with base_dir: {self.base_dir}")
        
    async def cleanup(self) -> None:
        """Cleanup the CDP Browser environment."""
        await self.playwright_service.close()
        logger.info("| 🌐 Playwright Environment cleaned up")
    
    @ecp.action(name="search_google",
                type="CDP Browser",
                description="Search Google with a query.")
    async def search_google(self, query: str) -> str:
        """Search Google with a query.
        
        Args:
            query (str): The search query.
            
        Returns:
            str: The search result message.
        """
        request = SearchGoogleRequest(query=query)
        result = await self.playwright_service.search_google(request)
        
        if result.success:
            return result.message
        else:
            return f"Search failed: {result.message}"
    
    @ecp.action(name="go_to_url",
                type="CDP Browser", 
                description="Navigate to a URL.")
    async def go_to_url(self, url: str, new_tab: bool = False) -> str:
        """Navigate to a URL.
        
        Args:
            url (str): The URL to navigate to.
            new_tab (bool): Whether to open in a new tab.
            
        Returns:
            str: The navigation result message.
        """
        request = GoToUrlRequest(url=url, new_tab=new_tab)
        result = await self.playwright_service.go_to_url(request)
        
        if result.success:
            return result.message
        else:
            return f"Navigation failed: {result.message}"
    
    @ecp.action(name="go_back",
                type="CDP Browser",
                description="Go back to the previous page.")
    async def go_back(self) -> str:
        """Go back to the previous page.
        
        Returns:
            str: The navigation result message.
        """
        request = GoBackRequest()
        result = await self.playwright_service.go_back(request)
        
        if result.success:
            return result.message
        else:
            return f"Go back failed: {result.message}"
    
    @ecp.action(name="wait",
                    type="CDP Browser",
                description="Wait for a specified number of seconds.")
    async def wait(self, seconds: int = 3) -> str:
        """Wait for a specified number of seconds.
        
        Args:
            seconds (int): Number of seconds to wait (max 30).
            
        Returns:
            str: The wait result message.
        """
        request = WaitRequest(seconds=seconds)
        result = await self.playwright_service.wait(request)
        
        if result.success:
            return result.message
        else:
            return f"Wait failed: {result.message}"
    
    @ecp.action(name="click_element",
                type="CDP Browser",
                description="Click an element by its index.")
    async def click_element(self, index: int, while_holding_ctrl: bool = False) -> str:
        """Click an element by its index.
        
        Args:
            index (int): The index of the element to click.
            while_holding_ctrl (bool): Whether to hold Ctrl while clicking.
            
        Returns:
            str: The click result message.
        """
        request = ClickElementRequest(index=index, while_holding_ctrl=while_holding_ctrl)
        result = await self.playwright_service.click_element_by_index(request)
        
        if result.success:
            return result.message
        else:
            return f"Click failed: {result.message}"
    
    @ecp.action(name="input_text",
                type="CDP Browser",
                description="Input text into an element.")
    async def input_text(self, 
                         index: int, 
                         text: str, 
                         clear_existing: bool = True,
                         has_sensitive_data: bool = False,
                         sensitive_data: Optional[Dict[str, Any]] = None) -> str:
        """Input text into an element.
        
        Args:
            index (int): The index of the input element.
            text (str): The text to input.
            clear_existing (bool): Whether to clear existing text.
            has_sensitive_data (bool): Whether the input contains sensitive data.
            sensitive_data (Optional[Dict[str, Any]]): Sensitive data mapping.
            
        Returns:
            str: The input result message.
        """
        request = InputTextRequest(
            index=index,
            text=text,
            clear_existing=clear_existing,
            has_sensitive_data=has_sensitive_data,
            sensitive_data=sensitive_data
        )
        result = await self.playwright_service.input_text(request)
        
        if result.success:
            return result.message
        else:
            return f"Input failed: {result.message}"
    
    @ecp.action(name="scroll",
                type="CDP Browser",
                description="Scroll the page.")
    async def scroll(self, 
                     down: bool = True, 
                     num_pages: float = 1.0, 
                     frame_element_index: Optional[int] = None) -> str:
        """Scroll the page.
        
        Args:
            down (bool): Whether to scroll down (True) or up (False).
            num_pages (float): Number of pages to scroll.
            frame_element_index (Optional[int]): Index of the frame element to scroll.
            
        Returns:
            str: The scroll result message.
        """
        request = ScrollRequest(
            down=down,
            num_pages=num_pages,
            frame_element_index=frame_element_index
        )
        result = await self.playwright_service.scroll(request)
        
        if result.success:
            return result.message
        else:
            return f"Scroll failed: {result.message}"
    
    @ecp.action(name="send_keys",
                type="CDP Browser",
                description="Send special keys to the page.")
    async def send_keys(self, keys: str) -> str:
        """Send special keys to the page.
        
        Args:
            keys (str): The keys to send (e.g., 'Escape', 'Control+o').
            
        Returns:
            str: The send keys result message.
        """
        request = SendKeysRequest(keys=keys)
        result = await self.playwright_service.send_keys(request)
        
        if result.success:
            return result.message
        else:
            return f"Send keys failed: {result.message}"
    
    @ecp.action(name="scroll_to_text",
                type="CDP Browser",
                description="Scroll to a specific text on the page.")
    async def scroll_to_text(self, text: str) -> str:
        """Scroll to a specific text on the page.
        
        Args:
            text (str): The text to scroll to.
            
        Returns:
            str: The scroll result message.
        """
        request = ScrollToTextRequest(text=text)
        result = await self.playwright_service.scroll_to_text(request)
        
        if result.success:
            return result.message
        else:
            return f"Scroll to text failed: {result.message}"
    
    @ecp.action(name="get_dropdown_options",
                type="CDP Browser",
                description="Get options from a dropdown element.")
    async def get_dropdown_options(self, index: int) -> str:
        """Get options from a dropdown element.
        
        Args:
            index (int): The index of the dropdown element.
            
        Returns:
            str: The dropdown options or error message.
        """
        request = GetDropdownOptionsRequest(index=index)
        result = await self.playwright_service.get_dropdown_options(request)
        
        if result.success:
            options_text = ", ".join(result.options) if result.options else "No options found"
            return f"Dropdown options: {options_text}"
        else:
            return f"Get dropdown options failed: {result.message}"
    
    @ecp.action(name="select_dropdown_option",
                    type="CDP Browser",
                description="Select an option from a dropdown element.")
    async def select_dropdown_option(self, index: int, text: str) -> str:
        """Select an option from a dropdown element.
        
        Args:
            index (int): The index of the dropdown element.
            text (str): The exact text of the option to select.
            
        Returns:
            str: The selection result message.
        """
        request = SelectDropdownOptionRequest(index=index, text=text)
        result = await self.playwright_service.select_dropdown_option(request)
        
        if result.success:
            return result.message
        else:
            return f"Select dropdown option failed: {result.message}"
    
    @ecp.action(name="upload_file",
                type="CDP Browser",
                description="Upload a file to an element.")
    async def upload_file(self, 
                          index: int, 
                          path: str, 
                          available_file_paths: Optional[List[str]] = None) -> str:
        """Upload a file to an element.
        
        Args:
            index (int): The index of the file input element.
            path (str): The path to the file to upload.
            available_file_paths (Optional[List[str]]): List of available file paths.
            
        Returns:
            str: The upload result message.
        """
        request = UploadFileRequest(
            index=index,
            path=path,
            available_file_paths=available_file_paths
        )
        result = await self.playwright_service.upload_file_to_element(request)
        
        if result.success:
            return result.message
        else:
            return f"Upload failed: {result.message}"
    
    @ecp.action(name="switch_tab",
                type="CDP Browser",
                description="Switch to a different tab.")
    async def switch_tab(self, tab_id: str) -> str:
        """Switch to a different tab.
        
        Args:
            tab_id (str): The ID of the tab to switch to.
            
        Returns:
            str: The switch result message.
        """
        request = SwitchTabRequest(tab_id=tab_id)
        result = await self.playwright_service.switch_tab(request)
        
        if result.success:
            return result.message
        else:
            return f"Switch tab failed: {result.message}"
    
    @ecp.action(name="close_tab",
                type="CDP Browser",
                description="Close a tab.")
    async def close_tab(self, tab_id: str) -> str:
        """Close a tab.
        
        Args:
            tab_id (str): The ID of the tab to close.
            
        Returns:
            str: The close result message.
        """
        request = CloseTabRequest(tab_id=tab_id)
        result = await self.playwright_service.close_tab(request)
        
        if result.success:
            return result.message
        else:
            return f"Close tab failed: {result.message}"
    
    @ecp.action(name="extract_structured_data",
                type="CDP Browser",
                description="Extract structured data from the current page.")
    async def extract_structured_data(self, 
                                     query: str,
                                     extract_links: bool = False,
                                     start_from_char: int = 0) -> str:
        """Extract structured data from the current page.
        
        Args:
            query (str): The query for data extraction.
            extract_links (bool): Whether to extract links.
            start_from_char (int): Start character position for extraction.
            
        Returns:
            str: The extracted data or error message.
        """
        request = ExtractStructuredDataRequest(
            query=query,
            extract_links=extract_links,
            start_from_char=start_from_char
        )
        result = await self.playwright_service.extract_structured_data(request)
        
        if result.success:
            return result.extracted_content or "No content extracted"
        else:
            return f"Extraction failed: {result.message}"
    
    @ecp.action(name="execute_js",
                type="CDP Browser",
                description="Execute JavaScript code on the page.")
    async def execute_js(self, code: str) -> str:
        """Execute JavaScript code on the page.
        
        Args:
            code (str): The JavaScript code to execute.
            
        Returns:
            str: The execution result or error message.
        """
        request = ExecuteJsRequest(code=code)
        result = await self.playwright_service.execute_js(request)
        
        if result.success:
            return result.extracted_content or "JavaScript executed successfully"
        else:
            return f"JavaScript execution failed: {result.message}"
    
    @ecp.action(name="screenshot",
                type="CDP Browser",
                description="Take a screenshot of the current page.")
    async def screenshot(self, 
                         highlight_elements: bool = False,
                         path: Optional[str] = None) -> str:
        """Take a screenshot of the current page.
        
        Args:
            highlight_elements (bool): Whether to highlight interactive elements.
            path (Optional[str]): Path to save the screenshot.
            
        Returns:
            str: The screenshot result message.
        """
        request = ScreenshotRequest(
            highlight_elements=highlight_elements,
            path=path
        )
        result = await self.playwright_service.screenshot(request)
        
        if result.success:
            if result.screenshot_path:
                return f"Screenshot saved to: {result.screenshot_path}"
            else:
                return "Screenshot taken successfully"
        else:
            return f"Screenshot failed: {result.message}"
    
    @ecp.action(name="store_screenshot",
                type="CDP Browser",
                description="Take and store a screenshot with step number.")
    async def store_screenshot(self, 
                               step_number: int, 
                               highlight_elements: bool = False) -> str:
        """Take and store a screenshot with step number.
        
        Args:
            step_number (int): The step number for the screenshot.
            highlight_elements (bool): Whether to highlight interactive elements.
            
        Returns:
            str: The screenshot storage result message.
        """
        request = StoreScreenshotRequest(
            step_number=step_number,
            highlight_elements=highlight_elements
        )
        result = await self.playwright_service.store_screenshot(request)
        
        if result.success:
            return f"Screenshot stored for step {step_number}: {result.screenshot_path}"
        else:
            return f"Store screenshot failed: {result.message}"
    
    @ecp.action(name="get_screenshot",
                type="CDP Browser",
                description="Get a screenshot from disk.")
    async def get_screenshot(self, screenshot_path: str) -> str:
        """Get a screenshot from disk.
        
        Args:
            screenshot_path (str): The path to the screenshot file.
            
        Returns:
            str: The screenshot retrieval result message.
        """
        request = GetScreenshotRequest(screenshot_path=screenshot_path)
        result = await self.playwright_service.get_screenshot_from_disk(request)
        
        if result.success:
            return f"Screenshot retrieved from: {screenshot_path}"
        else:
            return f"Get screenshot failed: {result.message}"
    
    async def get_state(self) -> Dict[str, Any]:
        """Get the current state of the CDP Browser environment.
        
        Returns:
            Dict[str, Any]: The browser state information including URL, title, tabs, elements, and environment config.
        """
        try:
            request = BrowserStateRequest(include_screenshot=False)
            result = await self.playwright_service.state(request)
            
            if result.success:
                state = result.state
                return {
                    "base_dir": self.base_dir,
                    "auto_start": self.auto_start,
                    # Browser state from service
                    "url": state.get('url', 'Unknown'),
                    "title": state.get('title', 'Unknown'),
                    "active_tab_id": state.get('active_tab_id'),
                    "tabs": state.get('tabs', []),
                    "tabs_count": len(state.get('tabs', [])),
                    "interactive_elements": state.get('interactive_elements', []),
                    "interactive_elements_count": len(state.get('interactive_elements', [])),
                    "dom_tree": state.get('dom_tree'),
                    "selector_map": state.get('selector_map', {}),
                    "viewport": state.get('viewport'),
                    "console_logs": state.get('console_logs', []),
                    "network_logs": state.get('network_logs', []),
                    "errors": state.get('errors', []),
                    "warnings": state.get('warnings', []),
                    "performance_metrics": state.get('performance_metrics')
                }
            else:
                return {
                    "error": result.message,
                    "base_dir": self.base_dir,
                    "auto_start": self.auto_start
                }
        except Exception as e:
            logger.error(f"Failed to get CDP Browser state: {e}")
            return {
                "error": str(e),
                "base_dir": self.base_dir,
                "auto_start": self.auto_start
            }

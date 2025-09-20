import asyncio
import enum
import json
import logging
import os
from typing import Any, Generic, TypeVar
from dataclasses import dataclass, asdict

from sympy import N

try:
    from lmnr import Laminar  # type: ignore
except ImportError:
    Laminar = None  # type: ignore
from pydantic import BaseModel

# Suppress specific CDP error logs that are common and don't affect functionality
cdp_logger = logging.getLogger('cdp_use.client')
cdp_logger.setLevel(logging.CRITICAL)  # Only show critical errors, suppress common CDP errors

from src.environments.playwright.browser.session import BrowserSession
from src.environments.playwright.browser.events import (
    ClickElementEvent,
    CloseTabEvent,
    GetDropdownOptionsEvent,
    GoBackEvent,
    NavigateToUrlEvent,
    ScrollEvent,
    ScrollToTextEvent,
    SelectDropdownOptionEvent,
    SendKeysEvent,
    SwitchTabEvent,
    TypeTextEvent,
    UploadFileEvent,
)
from src.environments.playwright.browser.views import BrowserError
from src.environments.playwright.dom.service import EnhancedDOMTreeNode
from src.environments.playwright.observability import observe_debug
from src.environments.playwright.utils import _log_pretty_url
from src.environments.playwright.screenshots.service import ScreenshotService
from src.environments.playwright.browser.session import DEFAULT_BROWSER_PROFILE

logger = logging.getLogger(__name__)

# Import EnhancedDOMTreeNode and rebuild event models that have forward references to it
# This must be done after all imports are complete
ClickElementEvent.model_rebuild()
TypeTextEvent.model_rebuild()
ScrollEvent.model_rebuild()
UploadFileEvent.model_rebuild()

Context = TypeVar('Context')
T = TypeVar('T', bound=BaseModel)


class ActionResult(BaseModel):
    extracted_content: str | None = None
    error: str | None = None
    long_term_memory: str | None = None
    metadata: dict[str, Any] | None = None
    include_extracted_content_only_once: bool = False
    include_in_memory: bool = False
    is_done: bool = False
    success: bool | None = None
    attachments: list[str] = []


def _detect_sensitive_key_name(text: str, sensitive_data: dict[str, str | dict[str, str]] | None) -> str | None:
    """Detect which sensitive key name corresponds to the given text value."""
    if not sensitive_data or not text:
        return None

    # Collect all sensitive values and their keys
    for domain_or_key, content in sensitive_data.items():
        if isinstance(content, dict):
            # New format: {domain: {key: value}}
            for key, value in content.items():
                if value and value == text:
                    return key
        elif content:  # Old format: {key: value}
            if content == text:
                return domain_or_key

    return None


def handle_browser_error(e: BrowserError) -> ActionResult:
    if e.long_term_memory is not None:
        if e.short_term_memory is not None:
            return ActionResult(
                extracted_content=e.short_term_memory, error=e.long_term_memory, include_extracted_content_only_once=True
            )
        else:
            return ActionResult(error=e.long_term_memory)
    # Fallback to original error handling if long_term_memory is None
    logger.warning(
        '⚠️ A BrowserError was raised without long_term_memory - always set long_term_memory when raising BrowserError to propagate right messages to LLM.'
    )
    raise e


class PlaywrightService(Generic[Context]):
    """Playwright service that mirrors browser_use Tools functionality."""

    def __init__(
        self,
        *,
        screenshots_dir: str | None = None,
    ):
        self._session = BrowserSession(
            browser_profile=DEFAULT_BROWSER_PROFILE,
        )
        self._screenshot_service = ScreenshotService(screenshots_dir or "workdir") if screenshots_dir else None

    async def search_google(self, query: str) -> ActionResult:
        """Search the query in Google, the query should be a search query like humans search in Google, concrete and not vague or super long."""
        search_url = f'https://www.google.com/search?q={query}&udm=14'

        # Check if there's already a tab open on Google or agent's about:blank
        use_new_tab = True
        try:
            tabs = await self._session.get_tabs()
            # Get last 4 chars of browser session ID to identify agent's tabs
            browser_session_label = str(self._session.id)[-4:]
            logger.debug(f'Checking {len(tabs)} tabs for reusable tab (browser_session_label: {browser_session_label})')

            for i, tab in enumerate(tabs):
                logger.debug(f'Tab {i}: url="{tab.url}", title="{tab.title}"')
                # Check if tab is on Google domain
                if tab.url and tab.url.strip('/').lower() in ('https://www.google.com', 'https://google.com'):
                    # Found existing Google tab, navigate in it
                    logger.debug(f'Found existing Google tab at index {i}: {tab.url}, reusing it')

                    # Switch to this tab first if it's not the current one
                    if self._session.agent_focus and tab.target_id != self._session.agent_focus.target_id:
                        try:
                            switch_event = self._session.event_bus.dispatch(SwitchTabEvent(target_id=tab.target_id))
                            await switch_event
                            await switch_event.event_result(raise_if_none=False)
                        except Exception as e:
                            logger.warning(f'Failed to switch to existing Google tab: {e}, will use new tab')
                            continue

                    use_new_tab = False
                    break
                # Check if it's an agent-owned about:blank page (has "Starting agent XXXX..." title)
                # IMPORTANT: about:blank is also used briefly for new tabs the agent is trying to open, dont take over those!
                elif tab.url == 'about:blank' and tab.title:
                    # Check if this is our agent's about:blank page with DVD animation
                    # The title should be "Starting agent XXXX..." where XXXX is the browser_session_label
                    if browser_session_label in tab.title:
                        # This is our agent's about:blank page
                        logger.debug(f'Found agent-owned about:blank tab at index {i} with title: "{tab.title}", reusing it')

                        # Switch to this tab first
                        if self._session.agent_focus and tab.target_id != self._session.agent_focus.target_id:
                            try:
                                switch_event = self._session.event_bus.dispatch(SwitchTabEvent(target_id=tab.target_id))
                                await switch_event
                                await switch_event.event_result()
                            except Exception as e:
                                logger.warning(f'Failed to switch to agent-owned tab: {e}, will use new tab')
                                continue

                        use_new_tab = False
                        break
        except Exception as e:
            logger.debug(f'Could not check for existing tabs: {e}, using new tab')

        # Dispatch navigation event
        try:
            # Ensure browser session is ready
            if not self._session.agent_focus:
                logger.warning("Browser session not ready, waiting for initialization...")
                await asyncio.sleep(1.0)
                
            event = self._session.event_bus.dispatch(
                NavigateToUrlEvent(
                    url=search_url,
                    new_tab=use_new_tab,
                )
            )
            await event
            await event.event_result(raise_if_any=True, raise_if_none=False)
            
            # Wait for page to be ready and CDP session to be stable
            await asyncio.sleep(1.0)
            
            # Additional check to ensure page is loaded
            try:
                # Try to get current URL to verify page is ready
                current_url = await self._session.get_current_page_url()
                logger.debug(f"Google search page loaded successfully, current URL: {current_url}")
            except Exception as e:
                logger.warning(f"Could not verify Google search page readiness: {e}")
                # Continue anyway as the navigation might still be successful
            
            memory = f"Searched Google for '{query}'"
            msg = f'🔍  {memory}'
            logger.info(msg)
            return ActionResult(extracted_content=memory, long_term_memory=memory)
        except Exception as e:
            logger.error(f'Failed to search Google: {e}')
            return ActionResult(error=f'Failed to search Google for "{query}": {str(e)}')

    async def go_to_url(self, url: str, new_tab: bool = False) -> ActionResult:
        """Navigate to URL, set new_tab=True to open in new tab, False to navigate in current tab"""
        try:
            # Ensure browser session is ready
            if not self._session.agent_focus:
                logger.warning("Browser session not ready, waiting for initialization...")
                await asyncio.sleep(1.0)
                
            # Dispatch navigation event
            event = self._session.event_bus.dispatch(NavigateToUrlEvent(url=url, new_tab=new_tab))
            await event
            await event.event_result(raise_if_any=True, raise_if_none=False)

            # Wait for page to be ready and CDP session to be stable
            await asyncio.sleep(1.0)
            
            # Additional check to ensure page is loaded
            try:
                # Try to get current URL to verify page is ready
                current_url = await self._session.get_current_page_url()
                logger.debug(f"Page loaded successfully, current URL: {current_url}")
            except Exception as e:
                logger.warning(f"Could not verify page readiness: {e}")
                # Continue anyway as the navigation might still be successful

            if new_tab:
                memory = f'Opened new tab with URL {url}'
                msg = f'🔗  Opened new tab with url {url}'
            else:
                memory = f'Navigated to {url}'
                msg = f'🔗 {memory}'

            logger.info(msg)
            return ActionResult(extracted_content=msg, long_term_memory=memory)
        except Exception as e:
            error_msg = str(e)
            # Always log the actual error first for debugging
            self._session.logger.error(f'❌ Navigation failed: {error_msg}')

            # Check if it's specifically a RuntimeError about CDP client
            if isinstance(e, RuntimeError) and 'CDP client not initialized' in error_msg:
                self._session.logger.error('❌ Browser connection failed - CDP client not properly initialized')
                return ActionResult(error=f'Browser connection error: {error_msg}')
            # Check for network-related errors
            elif any(
                err in error_msg
                for err in [
                    'ERR_NAME_NOT_RESOLVED',
                    'ERR_INTERNET_DISCONNECTED',
                    'ERR_CONNECTION_REFUSED',
                    'ERR_TIMED_OUT',
                    'net::',
                ]
            ):
                site_unavailable_msg = f'Navigation failed - site unavailable: {url}'
                self._session.logger.warning(f'⚠️ {site_unavailable_msg} - {error_msg}')
                return ActionResult(error=site_unavailable_msg)
            else:
                # Return error in ActionResult instead of re-raising
                return ActionResult(error=f'Navigation failed: {str(e)}')

    async def go_back(self) -> ActionResult:
        """Go back"""
        try:
            event = self._session.event_bus.dispatch(GoBackEvent())
            await event
            memory = 'Navigated back'
            msg = f'🔙  {memory}'
            logger.info(msg)
            return ActionResult(extracted_content=memory)
        except Exception as e:
            logger.error(f'Failed to dispatch GoBackEvent: {type(e).__name__}: {e}')
            error_msg = f'Failed to go back: {str(e)}'
            return ActionResult(error=error_msg)

    async def wait(self, seconds: int = 3) -> ActionResult:
        """Wait for x seconds (default 3) (max 30 seconds). This can be used to wait until the page is fully loaded."""
        # Cap wait time at maximum 30 seconds
        # Reduce the wait time by 3 seconds to account for the llm call which takes at least 3 seconds
        # So if the model decides to wait for 5 seconds, the llm call took at least 3 seconds, so we only need to wait for 2 seconds
        # Note by Mert: the above doesnt make sense because we do the LLM call right after this or this could be followed by another action after which we would like to wait
        # so I revert this.
        actual_seconds = min(max(seconds - 3, 0), 30)
        memory = f'Waited for {seconds} seconds'
        logger.info(f'🕒 waited for {actual_seconds} seconds + 3 seconds for LLM call')
        await asyncio.sleep(actual_seconds)
        return ActionResult(extracted_content=memory, long_term_memory=memory)

    async def click_element_by_index(self, index: int, while_holding_ctrl: bool = False) -> ActionResult:
        """Click element by index. Only indices from your browser_state are allowed. Never use an index that is not inside your current browser_state. Set while_holding_ctrl=True to open any resulting navigation in a new tab."""
        # Dispatch click event with node
        try:
            assert index != 0, (
                'Cannot click on element with index 0. If there are no interactive elements use scroll(), wait(), refresh(), etc. to troubleshoot'
            )

            # Look up the node from the selector map
            node = await self._session.get_element_by_index(index)
            if node is None:
                raise ValueError(f'Element index {index} not found in browser state')

            event = self._session.event_bus.dispatch(
                ClickElementEvent(node=node, while_holding_ctrl=while_holding_ctrl or False)
            )
            await event
            # Wait for handler to complete and get any exception or metadata
            click_metadata = await event.event_result(raise_if_any=True, raise_if_none=False)
            memory = 'Clicked element'

            if while_holding_ctrl:
                memory += ' and opened in new tab'

            # Check if a new tab was opened (from watchdog metadata)
            elif isinstance(click_metadata, dict) and click_metadata.get('new_tab_opened'):
                memory += ' - which opened a new tab'

            msg = f'🖱️ {memory}'
            logger.info(msg)

            # Include click coordinates in metadata if available
            return ActionResult(
                extracted_content=memory,
                metadata=click_metadata if isinstance(click_metadata, dict) else None,
            )
        except BrowserError as e:
            if 'Cannot click on <select> elements.' in str(e):
                try:
                    return await self.get_dropdown_options(index=index)
                except Exception as dropdown_error:
                    logger.error(
                        f'Failed to get dropdown options as shortcut during click_element_by_index on dropdown: {type(dropdown_error).__name__}: {dropdown_error}'
                    )
                return ActionResult(error='Can not click on select elements.')

            return handle_browser_error(e)
        except Exception as e:
            error_msg = f'Failed to click element {index}: {str(e)}'
            return ActionResult(error=error_msg)

    async def input_text(
        self,
        index: int,
        text: str,
        clear_existing: bool = True,
        has_sensitive_data: bool = False,
        sensitive_data: dict[str, str | dict[str, str]] | None = None,
    ) -> ActionResult:
        """Input text into an input interactive element. Only input text into indices that are inside your current browser_state. Never input text into indices that are not inside your current browser_state."""
        # Look up the node from the selector map
        node = await self._session.get_element_by_index(index)
        if node is None:
            raise ValueError(f'Element index {index} not found in browser state')

        # Dispatch type text event with node
        try:
            # Detect which sensitive key is being used
            sensitive_key_name = None
            if has_sensitive_data and sensitive_data:
                sensitive_key_name = _detect_sensitive_key_name(text, sensitive_data)

            event = self._session.event_bus.dispatch(
                TypeTextEvent(
                    node=node,
                    text=text,
                    clear_existing=clear_existing,
                    is_sensitive=has_sensitive_data,
                    sensitive_key_name=sensitive_key_name,
                )
            )
            await event
            input_metadata = await event.event_result(raise_if_any=True, raise_if_none=False)

            # Create message with sensitive data handling
            if has_sensitive_data:
                if sensitive_key_name:
                    msg = f'Input {sensitive_key_name} into element {index}.'
                    log_msg = f'Input <{sensitive_key_name}> into element {index}.'
                else:
                    msg = f'Input sensitive data into element {index}.'
                    log_msg = f'Input <sensitive> into element {index}.'
            else:
                msg = f"Input '{text}' into element {index}."
                log_msg = msg

            logger.debug(log_msg)

            # Include input coordinates in metadata if available
            return ActionResult(
                extracted_content=msg,
                long_term_memory=msg,
                metadata=input_metadata if isinstance(input_metadata, dict) else None,
            )
        except BrowserError as e:
            return handle_browser_error(e)
        except Exception as e:
            # Log the full error for debugging
            logger.error(f'Failed to dispatch TypeTextEvent: {type(e).__name__}: {e}')
            error_msg = f'Failed to input text into element {index}: {e}'
            return ActionResult(error=error_msg)

    async def scroll(self, down: bool = True, num_pages: float = 1.0, frame_element_index: int | None = None) -> ActionResult:
        """Scroll the page by specified number of pages (set down=True to scroll down, down=False to scroll up, num_pages=number of pages to scroll like 0.5 for half page, 10.0 for ten pages, etc.). 
        Default behavior is to scroll the entire page. This is enough for most cases.
        Optional if there are multiple scroll containers, use frame_element_index parameter with an element inside the container you want to scroll in. For that you must use indices that exist in your browser_state (works well for dropdowns and custom UI components). 
        Instead of scrolling step after step, use a high number of pages at once like 10 to get to the bottom of the page.
        If you know where you want to scroll to, use scroll_to_text instead of this tool.
        """
        try:
            # Look up the node from the selector map if index is provided
            # Special case: index 0 means scroll the whole page (root/body element)
            node = None
            if frame_element_index is not None and frame_element_index != 0:
                node = await self._session.get_element_by_index(frame_element_index)
                if node is None:
                    # Element does not exist
                    msg = f'Element index {frame_element_index} not found in browser state'
                    return ActionResult(error=msg)

            # Dispatch scroll event with node - the complex logic is handled in the event handler
            # Convert pages to pixels (assuming 1000px per page as standard viewport height)
            pixels = int(num_pages * 1000)
            event = self._session.event_bus.dispatch(
                ScrollEvent(direction='down' if down else 'up', amount=pixels, node=node)
            )
            await event
            await event.event_result(raise_if_any=True, raise_if_none=False)
            direction = 'down' if down else 'up'

            # If index is 0 or None, we're scrolling the page
            target = (
                'the page'
                if frame_element_index is None or frame_element_index == 0
                else f'element {frame_element_index}'
            )

            if num_pages == 1.0:
                long_term_memory = f'Scrolled {direction} {target} by one page'
            else:
                long_term_memory = f'Scrolled {direction} {target} by {num_pages} pages'

            msg = f'🔍 {long_term_memory}'
            logger.info(msg)
            return ActionResult(extracted_content=msg, long_term_memory=long_term_memory)
        except Exception as e:
            logger.error(f'Failed to dispatch ScrollEvent: {type(e).__name__}: {e}')
            error_msg = 'Failed to execute scroll action.'
            return ActionResult(error=error_msg)

    async def send_keys(self, keys: str) -> ActionResult:
        """Send strings of special keys to use e.g. Escape, Backspace, Insert, PageDown, Delete, Enter, or Shortcuts such as `Control+o`, `Control+Shift+T`"""
        # Dispatch send keys event
        try:
            event = self._session.event_bus.dispatch(SendKeysEvent(keys=keys))
            await event
            await event.event_result(raise_if_any=True, raise_if_none=False)
            memory = f'Sent keys: {keys}'
            msg = f'⌨️  {memory}'
            logger.info(msg)
            return ActionResult(extracted_content=memory, long_term_memory=memory)
        except Exception as e:
            logger.error(f'Failed to dispatch SendKeysEvent: {type(e).__name__}: {e}')
            error_msg = f'Failed to send keys: {str(e)}'
            return ActionResult(error=error_msg)

    async def scroll_to_text(self, text: str) -> ActionResult:
        """Scroll to a text in the current page. This helps you to be efficient. Prefer this tool over scrolling step by step."""
        # Dispatch scroll to text event
        event = self._session.event_bus.dispatch(ScrollToTextEvent(text=text))

        try:
            # The handler returns None on success or raises an exception if text not found
            await event.event_result(raise_if_any=True, raise_if_none=False)
            memory = f'Scrolled to text: {text}'
            msg = f'🔍  {memory}'
            logger.info(msg)
            return ActionResult(extracted_content=memory, long_term_memory=memory)
        except Exception as e:
            # Text not found
            msg = f"Text '{text}' not found or not visible on page"
            logger.info(msg)
            return ActionResult(
                extracted_content=msg,
                long_term_memory=f"Tried scrolling to text '{text}' but it was not found",
            )

    async def get_dropdown_options(self, index: int) -> ActionResult:
        """Get list of values for a dropdown input field. Only works on dropdown-style form elements (<select>, Semantic UI/aria-labeled select, etc.). Do not use this tool for none dropdown elements."""
        # Look up the node from the selector map
        node = await self._session.get_element_by_index(index)
        if node is None:
            raise ValueError(f'Element index {index} not found in browser state')

        # Dispatch GetDropdownOptionsEvent to the event handler
        event = self._session.event_bus.dispatch(GetDropdownOptionsEvent(node=node))
        dropdown_data = await event.event_result(timeout=3.0, raise_if_none=True, raise_if_any=True)

        if not dropdown_data:
            raise ValueError('Failed to get dropdown options - no data returned')

        # Use structured memory from the handler
        return ActionResult(
            extracted_content=dropdown_data['short_term_memory'],
            long_term_memory=dropdown_data['long_term_memory'],
            include_extracted_content_only_once=True,
        )

    async def select_dropdown_option(self, index: int, text: str) -> ActionResult:
        """Select dropdown option by exact text from any dropdown type (native <select>, ARIA menus, or custom dropdowns). Searches target element and children to find selectable options."""
        # Look up the node from the selector map
        node = await self._session.get_element_by_index(index)
        if node is None:
            raise ValueError(f'Element index {index} not found in browser state')

        # Dispatch SelectDropdownOptionEvent to the event handler
        event = self._session.event_bus.dispatch(SelectDropdownOptionEvent(node=node, text=text))
        selection_data = await event.event_result()

        if not selection_data:
            raise ValueError('Failed to select dropdown option - no data returned')

        # Check if the selection was successful
        if selection_data.get('success') == 'true':
            # Extract the message from the returned data
            msg = selection_data.get('message', f'Selected option: {text}')
            return ActionResult(
                extracted_content=msg,
                include_in_memory=True,
                long_term_memory=f"Selected dropdown option '{text}' at index {index}",
            )
        else:
            # Handle structured error response
            # TODO: raise BrowserError instead of returning ActionResult
            if 'short_term_memory' in selection_data and 'long_term_memory' in selection_data:
                return ActionResult(
                    extracted_content=selection_data['short_term_memory'],
                    long_term_memory=selection_data['long_term_memory'],
                    include_extracted_content_only_once=True,
                )
            else:
                # Fallback to regular error
                error_msg = selection_data.get('error', f'Failed to select option: {text}')
                return ActionResult(error=error_msg)

    async def upload_file_to_element(
        self, index: int, path: str, available_file_paths: list[str] | None = None
    ) -> ActionResult:
        """Upload file to interactive element with file path"""
        # Check if file is in available_file_paths (user-provided or downloaded files)
        # For remote browsers (is_local=False), we allow absolute remote paths even if not tracked locally
        if available_file_paths and path not in available_file_paths:
            # Also check if it's a recently downloaded file that might not be in available_file_paths yet
            downloaded_files = self._session.downloaded_files
            if path not in downloaded_files:
                # If browser is remote, allow passing a remote-accessible absolute path
                if not self._session.is_local:
                    pass
                else:
                    msg = f'File path {path} is not available. Upload files must be in available_file_paths or downloaded_files.'
                    logger.error(f'❌ {msg}')
                    return ActionResult(error=msg)

        # For local browsers, ensure the file exists on the local filesystem
        if self._session.is_local:
            if not os.path.exists(path):
                msg = f'File {path} does not exist'
                return ActionResult(error=msg)

        # Get the selector map to find the node
        selector_map = await self._session.get_selector_map()
        if index not in selector_map:
            msg = f'Element with index {index} does not exist.'
            return ActionResult(error=msg)

        node = selector_map[index]

        # Helper function to find file input near the selected element
        def find_file_input_near_element(
            node: EnhancedDOMTreeNode, max_height: int = 3, max_descendant_depth: int = 3
        ) -> EnhancedDOMTreeNode | None:
            """Find the closest file input to the selected element."""

            def find_file_input_in_descendants(n: EnhancedDOMTreeNode, depth: int) -> EnhancedDOMTreeNode | None:
                if depth < 0:
                    return None
                if self._session.is_file_input(n):
                    return n
                for child in n.children_nodes or []:
                    result = find_file_input_in_descendants(child, depth - 1)
                    if result:
                        return result
                return None

            current = node
            for _ in range(max_height + 1):
                # Check the current node itself
                if self._session.is_file_input(current):
                    return current
                # Check all descendants of the current node
                result = find_file_input_in_descendants(current, max_descendant_depth)
                if result:
                    return result
                # Check all siblings and their descendants
                if current.parent_node:
                    for sibling in current.parent_node.children_nodes or []:
                        if sibling is current:
                            continue
                        if self._session.is_file_input(sibling):
                            return sibling
                        result = find_file_input_in_descendants(sibling, max_descendant_depth)
                        if result:
                            return result
                current = current.parent_node
                if not current:
                    break
            return None

        # Try to find a file input element near the selected element
        file_input_node = find_file_input_near_element(node)

        # If not found near the selected element, fallback to finding the closest file input to current scroll position
        if file_input_node is None:
            logger.info(
                f'No file upload element found near index {index}, searching for closest file input to scroll position'
            )

            # Get current scroll position
            cdp_session = await self._session.get_or_create_cdp_session()
            try:
                scroll_info = await cdp_session.cdp_client.send.Runtime.evaluate(
                    params={'expression': 'window.scrollY || window.pageYOffset || 0'}, session_id=cdp_session.session_id
                )
                current_scroll_y = scroll_info.get('result', {}).get('value', 0)
            except Exception:
                current_scroll_y = 0

            # Find all file inputs in the selector map and pick the closest one to scroll position
            closest_file_input = None
            min_distance = float('inf')

            for idx, element in selector_map.items():
                if self._session.is_file_input(element):
                    # Get element's Y position
                    if element.absolute_position:
                        element_y = element.absolute_position.y
                        distance = abs(element_y - current_scroll_y)
                        if distance < min_distance:
                            min_distance = distance
                            closest_file_input = element

            if closest_file_input:
                file_input_node = closest_file_input
                logger.info(f'Found file input closest to scroll position (distance: {min_distance}px)')
            else:
                msg = 'No file upload element found on the page'
                logger.error(msg)
                return ActionResult(error=msg)

        # Dispatch upload file event with the file input node
        try:
            event = self._session.event_bus.dispatch(UploadFileEvent(node=file_input_node, file_path=path))
            await event
            await event.event_result(raise_if_any=True, raise_if_none=False)
            msg = f'Successfully uploaded file to index {index}'
            logger.info(f'📁 {msg}')
            return ActionResult(
                extracted_content=msg,
                long_term_memory=f'Uploaded file {path} to element {index}',
            )
        except Exception as e:
            logger.error(f'Failed to upload file: {e}')
            return ActionResult(error=f'Failed to upload file: {e}')

    async def switch_tab(self, tab_id: str) -> ActionResult:
        """Switch tab"""
        # Dispatch switch tab event
        try:
            target_id = await self._session.get_target_id_from_tab_id(tab_id)

            event = self._session.event_bus.dispatch(SwitchTabEvent(target_id=target_id))
            await event
            new_target_id = await event.event_result(raise_if_any=True, raise_if_none=False)
            assert new_target_id, 'SwitchTabEvent did not return a TargetID for the new tab that was switched to'
            memory = f'Switched to Tab with ID {new_target_id[-4:]}'
            logger.info(f'🔄  {memory}')
            return ActionResult(extracted_content=memory, long_term_memory=memory)
        except Exception as e:
            logger.error(f'Failed to switch tab: {type(e).__name__}: {e}')
            return ActionResult(error=f'Failed to switch to tab {tab_id}.')

    async def close_tab(self, tab_id: str) -> ActionResult:
        """Close an existing tab"""
        # Dispatch close tab event
        try:
            target_id = await self._session.get_target_id_from_tab_id(tab_id)
            cdp_session = await self._session.get_or_create_cdp_session()
            target_info = await cdp_session.cdp_client.send.Target.getTargetInfo(
                params={'targetId': target_id}, session_id=cdp_session.session_id
            )
            tab_url = target_info['targetInfo']['url']
            event = self._session.event_bus.dispatch(CloseTabEvent(target_id=target_id))
            await event
            await event.event_result(raise_if_any=True, raise_if_none=False)
            memory = f'Closed tab # {tab_id} ({_log_pretty_url(tab_url)})'
            logger.info(f'🗑️  {memory}')
            return ActionResult(
                extracted_content=memory,
                long_term_memory=memory,
            )
        except Exception as e:
            logger.error(f'Failed to close tab: {e}')
            return ActionResult(error=f'Failed to close tab {tab_id}.')

    async def extract_structured_data(
        self,
        query: str,
        extract_links: bool = False,
        start_from_char: int = 0,
    ) -> ActionResult:
        """This tool sends the markdown of the current page with the query to an LLM to extract structured, semantic data (e.g. product description, price, all information about XYZ) from the markdown of the current webpage based on a query.
Only use when:
- You are sure that you are on the right page for the query
- You know exactly the information you need to extract from the page
- You did not previously call this tool on the same page
You can not use this tool to:
- Get interactive elements like buttons, links, dropdowns, menus, etc.
- If you previously asked extract_structured_data on the same page with the same query, you should not call it again.

Set extract_links=True only if your query requires extracting links/URLs from the page.
Use start_from_char to start extraction from a specific character position (use if extraction was previously truncated and you want more content).

If this tool does not return the desired outcome, do not call it again, use scroll_to_text or scroll to find the desired information.
"""
        # Constants
        MAX_CHAR_LIMIT = 30000

        # Extract clean markdown using the new method
        try:
            content, content_stats = await self.extract_clean_markdown(extract_links=extract_links)
        except Exception as e:
            raise RuntimeError(f'Could not extract clean markdown: {type(e).__name__}')

        # Original content length for processing
        final_filtered_length = content_stats['final_filtered_chars']

        if start_from_char > 0:
            if start_from_char >= len(content):
                return ActionResult(
                    error=f'start_from_char ({start_from_char}) exceeds content length ({len(content)}). Content has {final_filtered_length} characters after filtering.'
                )
            content = content[start_from_char:]
            content_stats['started_from_char'] = start_from_char

        # Smart truncation with context preservation
        truncated = False
        if len(content) > MAX_CHAR_LIMIT:
            # Try to truncate at a natural break point (paragraph, sentence)
            truncate_at = MAX_CHAR_LIMIT

            # Look for paragraph break within last 500 chars of limit
            paragraph_break = content.rfind('\n\n', MAX_CHAR_LIMIT - 500, MAX_CHAR_LIMIT)
            if paragraph_break > 0:
                truncate_at = paragraph_break
            else:
                # Look for sentence break within last 200 chars of limit
                sentence_break = content.rfind('.', MAX_CHAR_LIMIT - 200, MAX_CHAR_LIMIT)
                if sentence_break > 0:
                    truncate_at = sentence_break + 1

            content = content[:truncate_at]
            truncated = True
            next_start = (start_from_char or 0) + truncate_at
            content_stats['truncated_at_char'] = truncate_at
            content_stats['next_start_char'] = next_start

        # Add content statistics to the result
        original_html_length = content_stats['original_html_chars']
        initial_markdown_length = content_stats['initial_markdown_chars']
        chars_filtered = content_stats['filtered_chars_removed']

        stats_summary = f"""Content processed: {original_html_length:,} HTML chars → {initial_markdown_length:,} initial markdown → {final_filtered_length:,} filtered markdown"""
        if start_from_char > 0:
            stats_summary += f' (started from char {start_from_char:,})'
        if truncated:
            stats_summary += f' → {len(content):,} final chars (truncated, use start_from_char={content_stats["next_start_char"]} to continue)'
        elif chars_filtered > 0:
            stats_summary += f' (filtered {chars_filtered:,} chars of noise)'

        # For now, return the extracted content directly since we don't have LLM integration
        # In a full implementation, this would send to an LLM for structured extraction
        current_url = await self._session.get_current_page_url()
        extracted_content = f'<url>\n{current_url}\n</url>\n<query>\n{query}\n</query>\n<result>\n{content}\n</result>'

        # Simple memory handling
        MAX_MEMORY_LENGTH = 1000
        if len(extracted_content) < MAX_MEMORY_LENGTH:
            memory = extracted_content
            include_extracted_content_only_once = False
        else:
            memory = f'Extracted content from {current_url} for query: {query}\nContent length: {len(content)} characters.'
            include_extracted_content_only_once = True

        logger.info(f'📄 {memory}')
        return ActionResult(
            extracted_content=extracted_content,
            include_extracted_content_only_once=include_extracted_content_only_once,
            long_term_memory=memory,
        )

    async def execute_js(self, code: str) -> ActionResult:
        """This JavaScript code gets executed with Runtime.evaluate and 'returnByValue': True, 'awaitPromise': True

SYNTAX RULES - FAILURE TO FOLLOW CAUSES "Uncaught at line 0" ERRORS:
- ALWAYS wrap your code in IIFE: (function(){ ... })() or (async function(){ ... })() for async code
- ALWAYS add try-catch blocks to prevent execution errors
- ALWAYS use proper semicolons and valid JavaScript syntax
- NEVER write multiline code without proper IIFE wrapping
- ALWAYS validate elements exist before accessing them

EXAMPLES:
Use this tool when other tools do not work on the first try as expected or when a more general tool is needed, e.g. for filling a form all at once, hovering, dragging, extracting only links, extracting content from the page, press and hold, hovering, clicking on coordinates, zooming, use this if the user provides custom selectors which you can otherwise not interact with ....
You can also use it to explore the website.
- Write code to solve problems you could not solve with other tools.
- Don't write comments in here, no human reads that.
- Write only valid js code.
- use this to e.g. extract + filter links, convert the page to json into the format you need etc...


- limit the output otherwise your context will explode
- think if you deal with special elements like iframes / shadow roots etc
- Adopt your strategy for React Native Web, React, Angular, Vue, MUI pages etc.
- e.g. with  synthetic events, keyboard simulation, shadow DOM, etc.

PROPER SYNTAX EXAMPLES:
CORRECT: (function(){ try { const el = document.querySelector('#id'); return el ? el.value : 'not found'; } catch(e) { return 'Error: ' + e.message; } })()
CORRECT: (async function(){ try { await new Promise(r => setTimeout(r, 100)); return 'done'; } catch(e) { return 'Error: ' + e.message; } })()

WRONG: const el = document.querySelector('#id'); el ? el.value : '';
WRONG: document.querySelector('#id').value
WRONG: Multiline code without IIFE wrapping

SHADOW DOM ACCESS EXAMPLE:
(function(){
    try {
        const hosts = document.querySelectorAll('*');
        for (let host of hosts) {
            if (host.shadowRoot) {
                const el = host.shadowRoot.querySelector('#target');
                if (el) return el.textContent;
            }
        }
        return 'Not found';
    } catch(e) {
        return 'Error: ' + e.message;
    }
})()

## Return values:
- Async functions (with await, promises, timeouts) are automatically handled
- Returns strings, numbers, booleans, and serialized objects/arrays
- Use JSON.stringify() for complex objects: JSON.stringify(Array.from(document.querySelectorAll('a')).map(el => el.textContent.trim()))

"""
        # Execute JavaScript with proper error handling and promise support

        cdp_session = await self._session.get_or_create_cdp_session()

        try:
            # Always use awaitPromise=True - it's ignored for non-promises
            result = await cdp_session.cdp_client.send.Runtime.evaluate(
                params={'expression': code, 'returnByValue': True, 'awaitPromise': True},
                session_id=cdp_session.session_id,
            )

            # Check for JavaScript execution errors
            if result.get('exceptionDetails'):
                exception = result['exceptionDetails']
                error_msg = f'JavaScript execution error: {exception.get("text", "Unknown error")}'
                if 'lineNumber' in exception:
                    error_msg += f' at line {exception["lineNumber"]}'
                msg = f'Code: {code}\n\nError: {error_msg}'
                logger.info(msg)
                return ActionResult(error=msg)

            # Get the result data
            result_data = result.get('result', {})

            # Check for wasThrown flag (backup error detection)
            if result_data.get('wasThrown'):
                msg = f'Code: {code}\n\nError: JavaScript execution failed (wasThrown=true)'
                logger.info(msg)
                return ActionResult(error=msg)

            # Get the actual value
            value = result_data.get('value')

            # Handle different value types
            if value is None:
                # Could be legitimate null/undefined result
                result_text = str(value) if 'value' in result_data else 'undefined'
            elif isinstance(value, (dict, list)):
                # Complex objects - should be serialized by returnByValue
                try:
                    result_text = json.dumps(value, ensure_ascii=False)
                except (TypeError, ValueError):
                    # Fallback for non-serializable objects
                    result_text = str(value)
            else:
                # Primitive values (string, number, boolean)
                result_text = str(value)

            # Apply length limit with better truncation
            if len(result_text) > 20000:
                result_text = result_text[:19950] + '\n... [Truncated after 20000 characters]'
            msg = f'Code: {code}\n\nResult: {result_text}'
            logger.info(msg)
            return ActionResult(extracted_content=f'Code: {code}\n\nResult: {result_text}')

        except Exception as e:
            # CDP communication or other system errors
            error_msg = f'Code: {code}\n\nError: Failed to execute JavaScript: {type(e).__name__}: {e}'
            logger.info(error_msg)
            return ActionResult(error=error_msg)

    @observe_debug(ignore_input=True, ignore_output=True, name='extract_clean_markdown')
    async def extract_clean_markdown(
        self, extract_links: bool = False
    ) -> tuple[str, dict[str, Any]]:
        """Extract clean markdown from the current page.

        Args:
            extract_links: Whether to preserve links in markdown

        Returns:
            tuple: (clean_markdown_content, content_statistics)
        """
        import re

        # Get HTML content from current page
        cdp_session = await self._session.get_or_create_cdp_session()
        try:
            body_id = await cdp_session.cdp_client.send.DOM.getDocument(session_id=cdp_session.session_id)
            page_html_result = await cdp_session.cdp_client.send.DOM.getOuterHTML(
                params={'backendNodeId': body_id['root']['backendNodeId']}, session_id=cdp_session.session_id
            )
            page_html = page_html_result['outerHTML']
            current_url = await self._session.get_current_page_url()
        except Exception as e:
            raise RuntimeError(f"Couldn't extract page content: {e}")

        original_html_length = len(page_html)

        # Use html2text for clean markdown conversion
        import html2text

        h = html2text.HTML2Text()
        h.ignore_links = not extract_links
        h.ignore_images = True
        h.ignore_emphasis = False
        h.body_width = 0  # Don't wrap lines
        h.unicode_snob = True
        h.skip_internal_links = True
        content = h.handle(page_html)

        initial_markdown_length = len(content)

        # Minimal cleanup - html2text already does most of the work
        content = re.sub(r'%[0-9A-Fa-f]{2}', '', content)  # Remove any remaining URL encoding

        # Apply light preprocessing to clean up excessive whitespace
        content, chars_filtered = self._preprocess_markdown_content(content)

        final_filtered_length = len(content)

        # Content statistics
        stats = {
            'url': current_url,
            'original_html_chars': original_html_length,
            'initial_markdown_chars': initial_markdown_length,
            'filtered_chars_removed': chars_filtered,
            'final_filtered_chars': final_filtered_length,
        }

        return content, stats

    def _preprocess_markdown_content(self, content: str, max_newlines: int = 3) -> tuple[str, int]:
        """
        Light preprocessing of html2text output - minimal cleanup since html2text is already clean.

        Args:
            content: Markdown content from html2text to lightly filter
            max_newlines: Maximum consecutive newlines to allow

        Returns:
            tuple: (filtered_content, chars_filtered)
        """
        import re

        original_length = len(content)

        # Compress consecutive newlines (4+ newlines become max_newlines)
        content = re.sub(r'\n{4,}', '\n' * max_newlines, content)

        # Remove lines that are only whitespace or very short (likely artifacts)
        lines = content.split('\n')
        filtered_lines = []
        for line in lines:
            stripped = line.strip()
            # Keep lines with substantial content (html2text output is already clean)
            if len(stripped) > 2:
                filtered_lines.append(line)

        content = '\n'.join(filtered_lines)
        content = content.strip()

        chars_filtered = original_length - len(content)
        return content, chars_filtered

    # ----- lifecycle -----
    @observe_debug(ignore_input=True, ignore_output=True, name='pw_service_start')
    async def start(self) -> None:
        """Start the browser session with proper initialization checks"""
        await self._session.start()
        
        # Wait for browser to be fully initialized
        max_retries = 10
        for i in range(max_retries):
            if self._session.agent_focus:
                logger.debug("Browser session initialized successfully")
                break
            logger.debug(f"Waiting for browser initialization... ({i+1}/{max_retries})")
            await asyncio.sleep(0.5)
        else:
            logger.warning("Browser session may not be fully initialized")
        
        # Additional wait to ensure CDP session is stable
        await asyncio.sleep(0.5)

    @observe_debug(ignore_input=True, ignore_output=True, name='pw_service_stop')
    async def stop(self) -> None:
        await self._session.stop()

    async def kill(self) -> None:
        await self._session.kill()

    async def close(self) -> None:
        await self.stop()


    async def screenshot(
        self,
        *,
        path: str | None = None,
        full_page: bool = False,
        format: str = 'png',
        quality: int | None = None,
        highlight_elements: bool = False,
        filter_highlight_ids: bool = True,
        step_number: int | None = None,
    ) -> bytes | str:
        """Take a screenshot with optional element highlighting
        
        Args:
            path: File path to save screenshot (optional)
            full_page: Whether to capture full page (default: False)
            format: Image format (default: 'png')
            quality: Image quality (optional)
            highlight_elements: Whether to highlight interactive elements with bounding boxes and numbers (default: False)
            filter_highlight_ids: Whether to filter element IDs in highlights (default: True)
            step_number: Step number for automatic screenshot naming (optional)
        """
        if highlight_elements:
            # Use browser-use's built-in highlighting by getting browser state with screenshot
            # This automatically applies highlighting if highlight_elements is enabled
            import base64
            summary = await self._session.get_browser_state_summary(include_screenshot=True)
            
            if summary.screenshot:
                # Convert base64 screenshot to bytes
                highlighted_data = base64.b64decode(summary.screenshot)
                
                # Use ScreenshotService if available and no specific path provided
                if self._screenshot_service and not path and step_number is not None:
                    screenshot_path = await self._screenshot_service.store_screenshot(summary.screenshot, step_number)
                    return screenshot_path
                elif path:
                    # Save to specific path
                    from pathlib import Path
                    Path(path).write_bytes(highlighted_data)
                    return str(Path(path).resolve())
                else:
                    # Return raw data
                    return highlighted_data
            else:
                # Fallback to regular screenshot if highlighting failed
                logger.warning("Could not get highlighted screenshot from browser state, falling back to regular screenshot")
                data = await self._session.take_screenshot(path=path, full_page=full_page, format=format, quality=quality)
                if path:
                    from pathlib import Path
                    return str(Path(path).resolve())
                return data
        else:
            # Regular screenshot without highlighting
            data = await self._session.take_screenshot(path=path, full_page=full_page, format=format, quality=quality)
            
            # Use ScreenshotService if available and no specific path provided
            if self._screenshot_service and not path and step_number is not None:
                # Convert bytes to base64 for ScreenshotService
                import base64
                screenshot_b64 = base64.b64encode(data).decode('utf-8')
                screenshot_path = await self._screenshot_service.store_screenshot(screenshot_b64, step_number)
                return screenshot_path
            elif path:
                from pathlib import Path
                return str(Path(path).resolve())
            else:
                return data

    async def state(self, *, include_screenshot: bool = False) -> Any:
        """Get current browser state"""
        summary = await self._session.get_browser_state_summary(include_screenshot=include_screenshot)
        return summary

    async def current_url(self) -> str:
        """Get current page URL"""
        return await self._session.get_current_page_url()

    async def current_title(self) -> str:
        """Get current page title"""
        return await self._session.get_current_page_title()

    async def tabs(self) -> list[dict[str, Any]]:
        """Get all tabs"""
        infos = await self._session.get_tabs()
        return [info.model_dump() for info in infos]


    # ----- screenshot service methods -----
    async def store_screenshot(self, step_number: int, *, highlight_elements: bool = False) -> str:
        """Take a screenshot and store it using ScreenshotService"""
        if not self._screenshot_service:
            raise ValueError("ScreenshotService not initialized. Please provide screenshots_dir in constructor.")
        
        return await self.screenshot(step_number=step_number, highlight_elements=highlight_elements)
    
    async def get_screenshot_from_disk(self, screenshot_path: str) -> str | None:
        """Get screenshot from disk as base64"""
        if not self._screenshot_service:
            return None
        return await self._screenshot_service.get_screenshot(screenshot_path)



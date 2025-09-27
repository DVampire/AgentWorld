"""Configuration for Browser Environment."""

# Environment rules for the browser environment
environment_rules = """<environment_browser>

<state>
The browser environment provides web automation capabilities for the agent to interact with web pages.
Current state includes current URL, open tabs, page content, and browser navigation history.
Browser supports all standard web operations: navigation, interaction, and data extraction.
</state>

<vision>
Browser can capture screenshots and extract DOM content for visual analysis.
</vision>

<interaction>
The agent can perform operations through 4 main tool categories:

1. NAVIGATION_OPERATIONS: Web page navigation
   - goto: Navigate to a specific URL
   - back: Go back to the previous page in history
   - forward: Go forward to the next page in history
   - refresh: Refresh the current page

2. INTERACTION_OPERATIONS: Page element interaction
   - click: Click on elements by ID or selector
   - type: Type text into input fields and text areas
   - scroll: Scroll the page in specified direction and distance
   - send_keys: Send keyboard keys (Enter, Tab, Ctrl+A, etc.)

3. DATA_EXTRACTION_OPERATIONS: Page content extraction
   - get_dom: Extract the complete DOM structure of current page
   - screenshot: Capture screenshot of current page (saved to file)
   - get_page_content: Get simplified, readable page content

4. TAB_MANAGEMENT_OPERATIONS: Browser tab management
   - open_tab: Open new browser tab with optional URL
   - close_tab: Close specified tab by ID
   - switch_tab: Switch to specified tab by ID
   - list_tabs: List all open tabs with their status

All browser operations are executed asynchronously and return success/failure status.
Browser environment automatically handles page loading, element waiting, and error recovery.
Supports both real browser automation (via browser-use) and mock mode for testing.
</interaction>

</environment_browser>
"""

# Browser Environment Configuration
environment = dict(
    type="BrowserEnvironment",
    headless=False,  # Use headless mode for server environments
    window_size=(1920, 1080),
    downloads_dir="workdir/browser/downloads",
    max_pages=5,
    chrome_args=[
        '--disable-blink-features=AutomationControlled',
        '--disable-web-security',
        '--no-sandbox',
        '--disable-dev-shm-usage',
    ],
)

controller = dict(
    type="BrowserController",
    environment=environment,
    environment_rules=environment_rules,
)


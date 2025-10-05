class PlaywrightError(Exception):
    """Base exception for Playwright operations."""
    pass


class LLMException(PlaywrightError):
    """Exception for LLM-related errors."""
    def __init__(self, status_code, message):
        self.status_code = status_code
        self.message = message
        super().__init__(f'Error {status_code}: {message}')


class BrowserError(PlaywrightError):
    """Exception for browser-related errors."""
    pass


class NavigationError(PlaywrightError):
    """Exception for navigation-related errors."""
    pass


class ElementError(PlaywrightError):
    """Exception for element-related errors."""
    pass


class ScreenshotError(PlaywrightError):
    """Exception for screenshot-related errors."""
    pass


class JavaScriptError(PlaywrightError):
    """Exception for JavaScript execution errors."""
    pass

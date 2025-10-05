import os
from typing import TYPE_CHECKING

from src.environments.cdp_browser.logging_config import setup_logging

# Only set up logging if not in MCP mode or if explicitly requested
if os.environ.get('BROWSER_USE_SETUP_LOGGING', 'true').lower() != 'false':
	from src.environments.cdp_browser.config import CONFIG

	# Get log file paths from config/environment
	debug_log_file = getattr(CONFIG, 'BROWSER_USE_DEBUG_LOG_FILE', None)
	info_log_file = getattr(CONFIG, 'BROWSER_USE_INFO_LOG_FILE', None)

	# Set up logging with file handlers if specified
	logger = setup_logging(debug_log_file=debug_log_file, info_log_file=info_log_file)
else:
	import logging

	logger = logging.getLogger('playwright')

# Monkeypatch BaseSubprocessTransport.__del__ to handle closed event loops gracefully
from asyncio import base_subprocess

_original_del = base_subprocess.BaseSubprocessTransport.__del__


def _patched_del(self):
	"""Patched __del__ that handles closed event loops without throwing noisy red-herring errors like RuntimeError: Event loop is closed"""
	try:
		# Check if the event loop is closed before calling the original
		if hasattr(self, '_loop') and self._loop and self._loop.is_closed():
			# Event loop is closed, skip cleanup that requires the loop
			return
		_original_del(self)
	except RuntimeError as e:
		if 'Event loop is closed' in str(e):
			# Silently ignore this specific error
			pass
		else:
			raise


base_subprocess.BaseSubprocessTransport.__del__ = _patched_del


# Type stubs for lazy imports - fixes linter warnings
if TYPE_CHECKING:
	from src.environments.cdp_browser.browser import BrowserProfile, BrowserSession
	from src.environments.cdp_browser.browser import BrowserSession as Browser
	from src.environments.cdp_browser.dom.service import DomService

# Import exceptions directly (not lazy)
from .exceptions import (
	PlaywrightError,
	LLMException,
	BrowserError,
	NavigationError,
	ElementError,
	ScreenshotError,
	JavaScriptError,
)


# Lazy imports mapping - only import when actually accessed
_LAZY_IMPORTS = {
	'BrowserSession': ('src.environments.cdp_browser.browser', 'BrowserSession'),
	'Browser': ('src.environments.cdp_browser.browser', 'BrowserSession'),  # Alias for BrowserSession
	'BrowserProfile': ('src.environments.cdp_browser.browser', 'BrowserProfile'),
	'DomService': ('src.environments.cdp_browser.dom.service', 'DomService'),
}


def __getattr__(name: str):
	"""Lazy import mechanism - only import modules when they're actually accessed."""
	if name in _LAZY_IMPORTS:
		module_path, attr_name = _LAZY_IMPORTS[name]
		try:
			from importlib import import_module

			module = import_module(module_path)
			if attr_name is None:
				# For modules like 'models', return the module itself
				attr = module
			else:
				attr = getattr(module, attr_name)
			# Cache the imported attribute in the module's globals
			globals()[name] = attr
			return attr
		except ImportError as e:
			raise ImportError(f'Failed to import {name} from {module_path}: {e}') from e

	raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


__all__ = [
	'BrowserSession',
	'Browser',  # Alias for BrowserSession
	'BrowserProfile',
	'DomService',
	# Exceptions
	'PlaywrightError',
	'LLMException',
	'BrowserError',
	'NavigationError',
	'ElementError',
	'ScreenshotError',
	'JavaScriptError',
]

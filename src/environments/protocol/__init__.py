"""Environment Context Protocol (ECP)

A protocol for managing environments and their capabilities.
"""

from .types import EnvironmentInfo, ActionInfo, ScreenshotInfo, ActionResult
from .server import ECPServer, ecp
from .context import EnvironmentContextManager

__all__ = [
    "ECPServer",
    "ecp",
    "EnvironmentContextManager",
    "environment_context_manager",
    # Types
    "EnvironmentInfo",
    "ActionInfo",
    "ScreenshotInfo",
    "ActionResult",
]
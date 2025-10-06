"""Environment Context Protocol (ECP)

A protocol for managing environments and their capabilities.
"""

from .types import EnvironmentInfo, ActionInfo, EnvironmentState, ScreenshotInfo, ActionResult
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
    "EnvironmentState",
    "ScreenshotInfo",
    "ActionResult",
]
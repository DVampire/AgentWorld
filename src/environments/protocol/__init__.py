"""Environment Context Protocol (ECP)

A protocol for managing environments and their capabilities.
"""

from .types import *
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
    "ECPRequest",
    "ECPResponse",
    "ECPError",
]
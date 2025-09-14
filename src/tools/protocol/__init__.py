"""Tool Context Protocol (TCP)

A protocol for managing tools and their capabilities.
"""

from .types import *
from .server import TCPServer, tcp

__all__ = [
    "TCPServer",
    "tcp",
    # Types
    "TCPRequest",
    "TCPResponse", 
    "TCPError",
    "ToolInfo",
]

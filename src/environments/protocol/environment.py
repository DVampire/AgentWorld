"""
Abstract base class for environments in the Environment Context Protocol.
"""

from abc import ABC
from abc import abstractmethod
from typing import Any, Dict


class BaseEnvironment(ABC):
    """Base abstract class for ECP environments"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    @abstractmethod
    async def get_state(self) -> Dict[str, Any]:
        """Get the state of the environment"""
        raise NotImplementedError("Get state method not implemented")

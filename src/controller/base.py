from typing import Any, Dict, List
from abc import ABC, abstractmethod

class BaseController(ABC):
    """Base class for all controllers."""
    
    def __init__(self, **kwargs):
        super(BaseController, self).__init__()
"""Transformation server for protocol conversions.

This server handles transformations between ECP, TCP, and ACP protocols.
"""

from typing import Any, List, Optional

from src.logger import logger
from src.transformation.types import (
    TransformationType,
    T2ERequest,
    T2EResponse,
    E2TRequest,
    E2TResponse,
    T2ARequest,
    T2AResponse,
    E2ARequest,
    E2AResponse,
    A2TRequest,
    A2TResponse,
    A2ERequest,
    A2EResponse,
)
from src.transformation.t2e_transformer import T2ETransformer
from src.transformation.t2a_transformer import T2ATransformer
from src.transformation.e2t_transformer import E2TTransformer
from src.transformation.e2a_transformer import E2ATransformer
from src.transformation.a2t_transformer import A2TTransformer
from src.transformation.a2e_transformer import A2ETransformer


class TransformationServer:
    """Server for handling protocol transformations between ECP, TCP, and ACP."""
    
    def __init__(self):
        """Initialize the transformation server."""
        # Initialize all transformers
        self.t2e_transformer = T2ETransformer()
        self.t2a_transformer = T2ATransformer()
        self.e2t_transformer = E2TTransformer()
        self.e2a_transformer = E2ATransformer(self.e2t_transformer)  # E2A depends on E2T
        self.a2t_transformer = A2TTransformer()
        self.a2e_transformer = A2ETransformer()
        
        logger.info("| 🔄 Transformation Server initialized")
    
    async def transform(self, 
                        type: str,
                        env_names: Optional[List[str]] = None,
                        tool_names: Optional[List[str]] = None,
                        agent_names: Optional[List[str]] = None,
                        ) -> Any:
        """Perform a protocol transformation.
        
        Args:
            type: Transformation type (t2e, t2a, e2t, e2a, a2t, a2e)
            env_names: List of environment names (for e2t, e2a, a2e)
            tool_names: List of tool names (for t2e, t2a, a2t)
            agent_names: List of agent names (for t2a, e2a, a2t, a2e)
            
        Returns:
            Transformation response
        """
        try:
            logger.info(f"| 🔄 Starting transformation: {type}")
            
            # Route to appropriate transformation method
            if type == TransformationType.E2T.value:
                request = E2TRequest(
                    type=type,
                    env_names=env_names or []
                )
                result = await self.e2t_transformer.transform(request)
            elif type == TransformationType.T2E.value:
                request = T2ERequest(
                    type=type,
                    tool_names=tool_names or []
                )
                result = await self.t2e_transformer.transform(request)
            elif type == TransformationType.T2A.value:
                request = T2ARequest(
                    type=type,
                    tool_names=tool_names or []
                )
                result = await self.t2a_transformer.transform(request)
            elif type == TransformationType.E2A.value:
                request = E2ARequest(
                    type=type,
                    env_names=env_names or []
                )
                result = await self.e2a_transformer.transform(request)
            elif type == TransformationType.A2T.value:
                request = A2TRequest(
                    type=type,
                    agent_names=agent_names or []
                )
                result = await self.a2t_transformer.transform(request)
            elif type == TransformationType.A2E.value:
                request = A2ERequest(
                    type=type,
                    agent_names=agent_names or []
                )
                result = await self.a2e_transformer.transform(request)
            else:
                raise ValueError(f"Unknown transformation type: {type}")
            
            logger.info(f"| ✅ Transformation completed: {type}")
            return result
            
        except Exception as e:
            logger.error(f"| ❌ Transformation failed: {e}")
            return {
                "success": False,
                "message": f"Transformation failed: {str(e)}"
            }


transformation = TransformationServer()

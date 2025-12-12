"""Tool to Agent (T2A) Transformer.

Converts TCP tools to ACP agents.
"""

from src.logger import logger
from src.tool.server import tcp
from src.transformation.types import T2ARequest, T2AResponse


class T2ATransformer:
    """Transformer for converting TCP tools to ACP agents."""
    
    async def transform(self, request: T2ARequest) -> T2AResponse:
        """Convert TCP tools to ACP agents.
        
        Args:
            request (T2ARequest): T2ARequest instance
            
        Returns:
            T2AResponse: T2AResponse
        """
        try:
            logger.info("| 🔧 TCP to ACP transformation")
            
            for tool_name in request.tool_names:
                tool_info = await tcp.get_info(tool_name)
                if tool_info:
                    await tcp.register(tool_info)
                else:
                    logger.warning(f"| ⚠️ Tool {tool_name} not found in TCP")
                    
            return T2AResponse(
                success=True,
                message="TCP to ACP transformation completed",
            )
            
        except Exception as e:
            logger.error(f"| ❌ TCP to ACP transformation failed: {e}")
            return T2AResponse(
                success=False,
                message="TCP to ACP transformation failed: " + str(e)
            )

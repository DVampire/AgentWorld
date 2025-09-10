"""ECP Client

Client implementation for the Environment Context Protocol.
"""

import asyncio
from typing import Any, Dict, List, Optional, Callable
import json

from src.environments.protocol.types import (
    ECPRequest, 
    ECPResponse,
    ECPNotification,
    ECPError,
    EnvironmentInfo,
    ActionInfo, 
    ActionResult, 
    EnvironmentAction
)


class ECPClient:
    """ECP Client for interacting with ECP servers"""
    
    def __init__(self, server: Optional[Any] = None):
        self.server = server
        self._notification_handlers: Dict[str, Callable] = {}
    
    async def send_request(self, method: str, params: Optional[Dict[str, Any]] = None) -> ECPResponse:
        """Send a request to the ECP server
        
        Args:
            method: Request method
            params: Request parameters
            
        Returns:
            ECPResponse: Server response
        """
        request = ECPRequest(method=method, params=params or {})
        
        if self.server:
            return await self.server.handle_request(request)
        else:
            # Mock response for testing
            return ECPResponse(
                id=request.id,
                result={"message": "No server connected"}
            )
    
    async def list_environments(self) -> List[EnvironmentInfo]:
        """List all environments
        
        Returns:
            List[EnvironmentInfo]: List of environments
        """
        response = await self.send_request("list_environments")
        if response.error:
            raise Exception(f"Error listing environments: {response.error.message}")
        
        environments_data = response.result.get("environments", [])
        return [EnvironmentInfo(**env_data) for env_data in environments_data]
    
    async def get_environment(self, name: str) -> EnvironmentInfo:
        """Get environment information
        
        Args:
            name: Environment name
            
        Returns:
            EnvironmentInfo: Environment information
        """
        response = await self.send_request("get_environment", {"name": name})
        if response.error:
            raise Exception(f"Error getting environment: {response.error.message}")
        
        env_data = response.result.get("environment")
        return EnvironmentInfo(**env_data)
    
    async def create_environment(self, name: str, env_type: str, **kwargs) -> EnvironmentInfo:
        """Create a new environment
        
        Args:
            name: Environment name
            env_type: Environment type
            **kwargs: Additional environment parameters
            
        Returns:
            EnvironmentInfo: Created environment information
        """
        params = {"name": name, "type": env_type, **kwargs}
        response = await self.send_request("create_environment", params)
        if response.error:
            raise Exception(f"Error creating environment: {response.error.message}")
        
        env_data = response.result.get("environment")
        return EnvironmentInfo(**env_data)
    
    async def remove_environment(self, name: str) -> bool:
        """Remove an environment
        
        Args:
            name: Environment name
            
        Returns:
            bool: True if removed successfully
        """
        response = await self.send_request("remove_environment", {"name": name})
        if response.error:
            raise Exception(f"Error removing environment: {response.error.message}")
        
        return response.result.get("success", False)
    
    async def list_actions(self, environment_name: str) -> List[ActionInfo]:
        """List actions for an environment
        
        Args:
            environment_name: Environment name
            
        Returns:
            List[ActionInfo]: List of actions
        """
        response = await self.send_request("list_actions", {"environment": environment_name})
        if response.error:
            raise Exception(f"Error listing actions: {response.error.message}")
        
        actions_data = response.result.get("actions", [])
        return [ActionInfo(**action_data) for action_data in actions_data]
    
    async def execute_action(self, environment_name: str, operation: str, args: Optional[Dict[str, Any]] = None) -> ActionResult:
        """Execute an action in an environment
        
        Args:
            environment_name: Environment name
            operation: Operation name
            args: Operation arguments
            
        Returns:
            ActionResult: Action execution result
        """
        params = {
            "environment": environment_name,
            "operation": operation,
            "args": args or {}
        }
        
        response = await self.send_request("execute_action", params)
        if response.error:
            raise Exception(f"Error executing action: {response.error.message}")
        
        result_data = response.result.get("result")
        return ActionResult(**result_data)
    
    async def get_environment_status(self, name: str) -> Dict[str, Any]:
        """Get environment status
        
        Args:
            name: Environment name
            
        Returns:
            Dict[str, Any]: Environment status
        """
        response = await self.send_request("get_environment_status", {"name": name})
        if response.error:
            raise Exception(f"Error getting environment status: {response.error.message}")
        
        return response.result.get("status", {})
    
    def register_notification_handler(self, method: str, handler: Callable):
        """Register a notification handler
        
        Args:
            method: Notification method
            handler: Handler function
        """
        self._notification_handlers[method] = handler
    
    async def handle_notification(self, notification: ECPNotification):
        """Handle a notification from the server
        
        Args:
            notification: ECP notification
        """
        handler = self._notification_handlers.get(notification.method)
        if handler:
            await handler(notification.params or {})

# Environment Context Protocol (ECP)

A simplified protocol for managing environments and their actions as tools, inspired by MCP (Model Context Protocol).

## Overview

ECP provides a unified interface for managing different types of environments (file systems, databases, APIs, etc.) where each environment's actions are exposed as standardized tools.

## Key Features

- **Unified Interface**: All environments follow the same standard interface
- **Action as Tools**: Each environment action is exposed as a tool
- **Async Support**: Built for modern asynchronous Python applications
- **Type Safety**: Full type hints and validation using Pydantic
- **Extensible**: Easy to add new environment types

## Architecture

```
ECP Server
├── Environment Registry
│   ├── File System Environment
│   ├── Database Environment
│   └── API Environment
└── Action Execution Engine
```

## Core Components

### 1. Types (`types.py`)
- `ECPRequest/ECPResponse`: Request/response structures
- `EnvironmentInfo`: Environment metadata
- `ActionInfo`: Action (tool) definitions
- `ActionResult`: Action execution results
- `EnvironmentAction`: Action to execute

### 2. Base Environment (`environment.py`)
- `BaseEnvironment`: Abstract base class for all environments
- Standard methods: `initialize()`, `get_actions()`, `execute_action()`

### 3. Registry (`registry.py`)
- `EnvironmentRegistry`: Manages environment lifecycle
- Environment creation, registration, and execution

### 4. Server (`server.py`)
- `ECPServer`: Handles ECP requests and manages environments
- Built-in request handlers for common operations

### 5. Client (`client.py`)
- `ECPClient`: Client interface for interacting with ECP servers
- High-level methods for environment and action management

## Usage Example

```python
import asyncio
from src.environments.protocol import ECPServer, ECPClient
from src.environments.protocol.implementations.file_system_ecp import FileSystemECPEnvironment

async def main():
    # Create ECP server
    server = ECPServer()
    server.register_environment_type("file_system", FileSystemECPEnvironment)
    
    # Create client
    client = ECPClient(server)
    
    # Create environment
    env = await client.create_environment(
        name="my_fs",
        env_type="file_system",
        base_dir="/tmp"
    )
    
    # List available actions
    actions = await client.list_actions("my_fs")
    print(f"Available actions: {[a.name for a in actions]}")
    
    # Execute action
    result = await client.execute_action(
        "my_fs",
        "write_file",
        {"file_path": "test.txt", "content": "Hello World"}
    )
    
    print(f"Action result: {result.success}")

asyncio.run(main())
```

## Available Methods

### Server Methods
- `list_environments()`: List all environments
- `get_environment(name)`: Get environment info
- `create_environment(name, type, **kwargs)`: Create new environment
- `remove_environment(name)`: Remove environment
- `list_actions(environment)`: List environment actions
- `execute_action(environment, operation, args)`: Execute action
- `get_environment_status(name)`: Get environment status

### Environment Actions (File System Example)
- `read_file`: Read file content
- `write_file`: Write content to file
- `delete_file`: Delete a file
- `copy_file`: Copy file
- `move_file`: Move file
- `create_directory`: Create directory
- `delete_directory`: Delete directory
- `list_directory`: List directory contents
- `search_files`: Search for files
- `get_file_info`: Get file information

## Creating Custom Environments

1. Inherit from `BaseEnvironment`:

```python
from src.environments.protocol import BaseEnvironment, ActionInfo, ActionResult, EnvironmentAction

class MyEnvironment(BaseEnvironment):
    def __init__(self, name: str, **kwargs):
        super().__init__(name, "my_type", "My custom environment")
        # Initialize your environment
    
    async def initialize(self) -> bool:
        # Initialize your environment
        self.status = "ready"
        return True
    
    async def get_actions(self) -> List[ActionInfo]:
        return [
            ActionInfo(
                name="my_action",
                description="My custom action",
                parameters={
                    "type": "object",
                    "properties": {
                        "param1": {"type": "string", "description": "Parameter 1"}
                    },
                    "required": ["param1"]
                }
            )
        ]
    
    async def execute_action(self, action: EnvironmentAction) -> ActionResult:
        if action.operation == "my_action":
            # Execute your action
            return ActionResult(success=True, result="Action completed")
        else:
            return ActionResult(success=False, error="Unknown action")
```

2. Register with the server:

```python
server.register_environment_type("my_type", MyEnvironment)
```

## Running the Demo

```bash
cd src/environments/protocol/examples
python ecp_demo.py
```

## Benefits

1. **Standardization**: All environments follow the same interface
2. **Tool Integration**: Actions are automatically available as tools
3. **Async Support**: Built for modern Python applications
4. **Type Safety**: Full type checking and validation
5. **Extensibility**: Easy to add new environment types
6. **Protocol-Based**: Similar to MCP for standardized communication

## Comparison with MCP

| Feature | MCP | ECP |
|---------|-----|-----|
| Purpose | Model Context Protocol | Environment Context Protocol |
| Focus | General tool integration | Environment management |
| Actions | Tools | Environment actions as tools |
| Transport | Multiple (HTTP, WebSocket, etc.) | In-process (extensible) |
| Complexity | Full protocol implementation | Simplified for environments |

ECP is designed to be simpler and more focused on environment management while maintaining the benefits of a standardized protocol approach.

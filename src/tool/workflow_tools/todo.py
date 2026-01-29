"""Todo tool for managing todo.md file with task decomposition and step tracking.

This is a coroutine-safe version that uses per-call_id caching and locking.
"""

import json
import uuid
import os
import asyncio
import re
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, ConfigDict

from src.tool.types import Tool, ToolResponse, ToolExtra
from src.utils import assemble_project_path
from src.utils import file_lock
from src.registry import TOOL
from src.logger import logger


class Step(BaseModel):
    """Step model for todo management."""
    id: str = Field(description="Unique step ID")
    name: str = Field(description="Step name/description")
    parameters: Optional[Dict[str, Any]] = Field(default=None, description="Step parameters")
    status: str = Field(default="pending", description="Step status: pending, success, failed")
    result: Optional[str] = Field(default=None, description="Step result (1-3 sentences)")
    priority: str = Field(default="medium", description="Step priority: high, medium, low")
    category: Optional[str] = Field(default=None, description="Step category")
    created_at: str = Field(description="Creation timestamp")
    updated_at: Optional[str] = Field(default=None, description="Last update timestamp")


class TodoState(BaseModel):
    """State for a single todo instance."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    call_id: str = Field(description="Unique identifier for this todo instance")
    todo_file: str = Field(description="Path to the todo.md file")
    steps_file: str = Field(description="Path to the steps JSON file")
    steps: List[Step] = Field(default_factory=list, description="List of steps")


_TODO_TOOL_DESCRIPTION = """Todo tool for managing a todo.md file with task decomposition and step tracking.
When using this tool, only provide parameters that are relevant to the specific operation you are performing. Do not include unnecessary parameters.

IMPORTANT: You MUST provide a `call_id` parameter to identify your todo list. Use the same call_id across multiple calls to maintain state.

Available operations:
1. add: Add a new step to the todo list at the end or after a specific step.
    - call_id (REQUIRED): Unique identifier for the todo list.
    - task: The description of the step.
    - priority: The priority of the step.
    - category: The category of the step.
    - parameters: Optional parameters for the step.
    - after_step_id: Optional step ID to insert after (if not provided, adds to end).
2. complete: Mark step as completed (success or failed).
    - call_id (REQUIRED): Unique identifier for the todo list.
    - step_id: The ID of the step to complete.
    - status: Completion status: "success" or "failed".
    - result: Result description (1-3 sentences).
3. update: Update step information.
    - call_id (REQUIRED): Unique identifier for the todo list.
    - step_id: The ID of the step to update.
    - task: New step description.
    - parameters: New step parameters.
4. list: List all steps with their status.
    - call_id (REQUIRED): Unique identifier for the todo list.
5. clear: Clear completed steps.
    - call_id (REQUIRED): Unique identifier for the todo list.
6. show: Show the complete todo.md file content.
    - call_id (REQUIRED): Unique identifier for the todo list.
7. export: Export todo.md to a specified path.
    - call_id (REQUIRED): Unique identifier for the todo list.
    - export_path: The target path to export the todo.md file.
8. cleanup: Clean up and remove the todo from cache (call when done with the todo list).
    - call_id (REQUIRED): Unique identifier for the todo list.

Example: {"name": "todo", "args": {"action": "add", "call_id": "task_123", "task": "Task description", "priority": "high", "category": "work"}}
Example: {"name": "todo", "args": {"action": "complete", "call_id": "task_123", "step_id": "step_1", "status": "success", "result": "Completed successfully"}}

The todo.md file is maintained in the base directory and follows a structured format for task management.
"""


@TOOL.register_module(force=True)
class TodoTool(Tool):
    """A coroutine-safe tool for managing todo.md files with task decomposition and step tracking."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = "todo"
    description: str = _TODO_TOOL_DESCRIPTION
    metadata: Dict[str, Any] = Field(default={}, description="The metadata of the tool")
    require_grad: bool = Field(default=False, description="Whether the tool requires gradients")
    
    # Configuration parameters
    base_dir: str = Field(
        default="workdir/todo",
        description="The base directory for saving todo files."
    )
    
    def __init__(
        self, 
        base_dir: Optional[str] = None, 
        require_grad: bool = False,
        **kwargs
    ):
        """Initialize the todo tool."""
        super().__init__(require_grad=require_grad, **kwargs)
        
        if base_dir is not None:
            self.base_dir = assemble_project_path(base_dir)
        else:
            self.base_dir = assemble_project_path(self.base_dir)
            
        if self.base_dir is not None:
            os.makedirs(self.base_dir, exist_ok=True)
        
        # Per-call_id cache and locks for concurrent safety
        # Key: call_id (str), Value: TodoState instance
        self._todo_cache: Dict[str, TodoState] = {}
        # Key: call_id (str), Value: asyncio.Lock for that todo
        self._todo_locks: Dict[str, asyncio.Lock] = {}
        # Lock for managing the cache dictionaries themselves
        self._cache_lock = asyncio.Lock()
    
    def _get_todo_file_path(self, call_id: str) -> str:
        """Generate a fixed todo file path based on call_id."""
        safe_id = re.sub(r'[^\w\s-]', '', call_id).strip().replace(' ', '_')
        if not safe_id:
            safe_id = "todo"
        return os.path.join(self.base_dir, f"{safe_id}_todo.md")
    
    def _get_steps_file_path(self, call_id: str) -> str:
        """Generate a fixed steps file path based on call_id."""
        safe_id = re.sub(r'[^\w\s-]', '', call_id).strip().replace(' ', '_')
        if not safe_id:
            safe_id = "todo"
        return os.path.join(self.base_dir, f"{safe_id}_steps.json")
    
    async def _get_or_create_todo(self, call_id: str) -> tuple[TodoState, asyncio.Lock]:
        """Get or create a TodoState instance for the given call_id with proper locking.
        
        Args:
            call_id: The unique identifier for the call
            
        Returns:
            tuple[TodoState, asyncio.Lock]: The todo state instance and its lock
        """
        async with self._cache_lock:
            # Get or create lock for this call_id
            if call_id not in self._todo_locks:
                self._todo_locks[call_id] = asyncio.Lock()
            
            # Get or create todo state for this call_id
            if call_id not in self._todo_cache:
                todo_file = self._get_todo_file_path(call_id)
                steps_file = self._get_steps_file_path(call_id)
                
                # Load existing steps if file exists
                steps = []
                if os.path.exists(steps_file):
                    try:
                        with open(steps_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            steps = [Step(**step_data) for step_data in data]
                    except Exception:
                        steps = []
                
                self._todo_cache[call_id] = TodoState(
                    call_id=call_id,
                    todo_file=todo_file,
                    steps_file=steps_file,
                    steps=steps
                )
                logger.info(f"| 📝 Created new todo cache for id: {call_id} (file_path: {todo_file})")
            else:
                logger.info(f"| 📂 Using existing todo cache for id: {call_id} (steps: {len(self._todo_cache[call_id].steps)})")
            
            return self._todo_cache[call_id], self._todo_locks[call_id]
    
    async def _cleanup_todo(self, call_id: str):
        """Remove todo from cache after completion."""
        async with self._cache_lock:
            if call_id in self._todo_cache:
                del self._todo_cache[call_id]
                logger.info(f"| 🧹 Removed todo from cache: {call_id}")
            if call_id in self._todo_locks:
                del self._todo_locks[call_id]
    
    async def __call__(
        self, 
        action: str,
        call_id: str,
        task: Optional[str] = None, 
        step_id: Optional[str] = None,
        status: Optional[str] = None,
        result: Optional[str] = None,
        priority: Optional[str] = "medium",
        category: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        after_step_id: Optional[str] = None,
        export_path: Optional[str] = None,
        **kwargs
    ) -> ToolResponse:
        """Execute a todo action asynchronously with coroutine safety.

        Args:
            action (str): One of add, complete, update, list, clear, show, export, cleanup.
            call_id (str): REQUIRED: Unique identifier for the todo list. Use the same call_id across multiple calls to maintain state.
            task (Optional[str]): Step description for add/update.
            step_id (Optional[str]): Step identifier for complete/update actions.
            status (Optional[str]): Completion status for complete action.
            result (Optional[str]): Result summary for complete action.
            priority (Optional[str]): Priority for add action.
            category (Optional[str]): Category label for add action.
            parameters (Optional[Dict[str, Any]): Arbitrary metadata for steps.
            after_step_id (Optional[str]): Insert new step after the specified ID.
            export_path (Optional[str]): Target path for export action.
        """
        try:
            if not call_id:
                return ToolResponse(
                    success=False,
                    message="call_id is required. Please provide a unique identifier for the todo list."
                )
            
            if not self.base_dir:
                return ToolResponse(
                    success=False,
                    message="Cannot create todo: base_dir is not set."
                )
            
            # Handle cleanup action
            if action == "cleanup":
                await self._cleanup_todo(call_id)
                return ToolResponse(
                    success=True,
                    message=f"✅ Cleaned up todo cache for call_id: {call_id}"
                )
            
            # Get or create todo state and its lock
            todo_state, todo_lock = await self._get_or_create_todo(call_id)
            
            # Use the lock to ensure coroutine-safe access to this specific todo
            async with todo_lock:
                logger.info(f"| 📝 TodoTool action: {action} (call_id: {call_id}, todo_file: {todo_state.todo_file}, steps: {len(todo_state.steps)})")
                
                if action == "add":
                    return await self._add_step(todo_state, task, priority, category, parameters, after_step_id, step_id)
                if action == "complete":
                    return await self._complete_step(todo_state, step_id, status, result)
                if action == "update":
                    return await self._update_step(todo_state, step_id, task, parameters)
                if action == "list":
                    return await self._list_steps(todo_state)
                if action == "clear":
                    return await self._clear_completed(todo_state)
                if action == "show":
                    return await self._show_todo_file(todo_state)
                if action == "export":
                    return await self._export_todo_file(todo_state, export_path)
                
                return ToolResponse(
                    success=False,
                    message=(
                        f"Unknown action: {action}. "
                        "Available actions: add, complete, update, list, clear, show, export, cleanup"
                    ),
                )
                
        except Exception as e:
            logger.error(f"| ❌ Error in TodoTool: {e}")
            import traceback
            return ToolResponse(
                success=False, 
                message=f"Error executing todo action '{action}': {str(e)}\n{traceback.format_exc()}"
            )
    
    async def _save_steps(self, todo_state: TodoState) -> None:
        """Save steps to JSON file."""
        try:
            async with file_lock(todo_state.steps_file):
                with open(todo_state.steps_file, 'w', encoding='utf-8') as f:
                    json.dump([step.model_dump() for step in todo_state.steps], f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"| ❌ Error saving steps: {e}")
    
    async def _sync_to_markdown(self, todo_state: TodoState) -> None:
        """Sync steps list to markdown file."""
        try:
            async with file_lock(todo_state.todo_file):
                content = "# Todo List\n\n"
                
                for step in todo_state.steps:
                    priority_emoji = {
                        "high": "🔴", 
                        "medium": "🟡", 
                        "low": "🟢"
                    }.get(step.priority, "🟡")
                    
                    status_emoji = {
                        "pending": "⏳",
                        "success": "✅",
                        "failed": "❌"
                    }.get(step.status, "⏳")
                    
                    category_text = f" [{step.category}]" if step.category else ""
                    
                    # Create step line
                    if step.status == "pending":
                        checkbox = "[ ]"
                    else:
                        checkbox = "[x]"
                    
                    step_line = f"- {checkbox} **{step.id}** {priority_emoji} {status_emoji} {step.name}{category_text}"
                    
                    if step.parameters:
                        step_line += f" *(params: {json.dumps(step.parameters)})*"
                    
                    step_line += f" *(created: {step.created_at}*"
                    
                    if step.updated_at:
                        step_line += f", updated: {step.updated_at}"
                    
                    if step.result:
                        step_line += f", result: {step.result}"
                    
                    step_line += ")"
                    content += step_line + "\n"
                
                with open(todo_state.todo_file, 'w', encoding='utf-8') as f:
                    f.write(content)
        except Exception as e:
            logger.error(f"| ❌ Error syncing to markdown: {e}")
    
    def _generate_step_id(self) -> str:
        """Generate a unique step ID using UUID + timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{timestamp}_{unique_id}"
    
    async def _add_step(
        self, 
        todo_state: TodoState,
        task: str, 
        priority: str = "medium",
        category: Optional[str] = None, 
        parameters: Optional[Dict[str, Any]] = None, 
        after_step_id: Optional[str] = None,
        step_id: Optional[str] = None
    ) -> ToolResponse:
        """Add a new step to the todo list."""
        if not task:
            return ToolResponse(success=False, message="Error: Step description is required for add action")
        
        # Use provided step_id or generate a new one
        if step_id is None:
            step_id = self._generate_step_id()
        else:
            # Check if step_id already exists
            for step in todo_state.steps:
                if step.id == step_id:
                    return ToolResponse(success=False, message=f"Error: Step ID {step_id} already exists.")
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        new_step = Step(
            id=step_id,
            name=task,
            parameters=parameters,
            status="pending",
            priority=priority,
            category=category,
            created_at=timestamp
        )
        
        # Insert step at the right position
        if after_step_id:
            # Try to find by ID first
            insert_index = -1
            for i, step in enumerate(todo_state.steps):
                if step.id == after_step_id:
                    insert_index = i + 1
                    break
            
            # If not found by ID, try to find by numeric index (0-based)
            if insert_index == -1:
                try:
                    index = int(after_step_id)
                    if 0 <= index < len(todo_state.steps):
                        insert_index = index + 1
                except ValueError:
                    pass
            
            if insert_index == -1:
                # Not found, add to end but return warning
                todo_state.steps.append(new_step)
                await self._save_steps(todo_state)
                await self._sync_to_markdown(todo_state)
                return ToolResponse(
                    success=True, 
                    message=f"⚠️ Step {after_step_id} not found. Added step {step_id} to end of list: {task} (priority: {priority})"
                )
            
            # Insert at the found position
            todo_state.steps.insert(insert_index, new_step)
        else:
            # Add to end of list
            todo_state.steps.append(new_step)
        
        # Save and sync
        await self._save_steps(todo_state)
        await self._sync_to_markdown(todo_state)
        
        message = f"✅ Added step {step_id}: {task} (priority: {priority})"
        if after_step_id:
            message = f"✅ Added step {step_id} after {after_step_id}: {task} (priority: {priority})"
        
        logger.info(f"| {message}")
        return ToolResponse(
            success=True, 
            message=message, 
            extra=ToolExtra(
                file_path=todo_state.todo_file,
                data={
                    "step_id": step_id,
                    "after_step_id": after_step_id,
                    "task": task,
                    "priority": priority,
                    "category": category,
                    "parameters": parameters
                }
            )
        )
    
    async def _complete_step(
        self, 
        todo_state: TodoState, 
        step_id: str, 
        status: str, 
        result: Optional[str] = None
    ) -> ToolResponse:
        """Mark step as completed."""
        if not step_id:
            return ToolResponse(success=False, message="Error: Step ID is required for complete action")
        
        if status not in ["success", "failed"]:
            return ToolResponse(success=False, message="Error: Status must be 'success' or 'failed'")
        
        # Find the step
        step = None
        for s in todo_state.steps:
            if s.id == step_id:
                step = s
                break
        
        if not step:
            return ToolResponse(success=False, message=f"Error: Step {step_id} not found")
        
        if step.status != "pending":
            return ToolResponse(success=False, message=f"Step {step_id} is already completed with status: {step.status}")
        
        # Update step
        step.status = status
        step.result = result
        step.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Save and sync
        await self._save_steps(todo_state)
        await self._sync_to_markdown(todo_state)
        
        return ToolResponse(
            success=True, 
            message=f"✅ Completed step {step_id} with status: {status}",
            extra=ToolExtra(
                file_path=todo_state.todo_file,
                data={
                    "step_id": step_id,
                    "status": status,
                    "result": result,
                    "updated_at": step.updated_at,
                    "step_name": step.name
                }
            )
        )
    
    async def _update_step(
        self, 
        todo_state: TodoState, 
        step_id: str, 
        task: Optional[str] = None, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> ToolResponse:
        """Update step information."""
        if not step_id:
            return ToolResponse(success=False, message="Error: Step ID is required for update action")
        
        # Find the step
        step = None
        for s in todo_state.steps:
            if s.id == step_id:
                step = s
                break
        
        if not step:
            return ToolResponse(success=False, message=f"Error: Step {step_id} not found")
        
        # Update step fields
        if task:
            step.name = task
        if parameters is not None:
            step.parameters = parameters
        
        step.updated_at = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Save and sync
        await self._save_steps(todo_state)
        await self._sync_to_markdown(todo_state)
        
        return ToolResponse(
            success=True, 
            message=f"✅ Updated step {step_id}",
            extra=ToolExtra(
                file_path=todo_state.todo_file,
                data={
                    "step_id": step_id,
                    "updated_fields": {
                        "task": task if task else None,
                        "parameters": parameters if parameters is not None else None
                    },
                    "updated_at": step.updated_at,
                    "step_name": step.name
                }
            )
        )
    
    async def _list_steps(self, todo_state: TodoState) -> ToolResponse:
        """List all steps with their status."""
        if not todo_state.steps:
            return ToolResponse(success=False, message="No steps found. Use 'add' action to create your first step.")
        
        result = "📋 Todo Steps:\n\n"
        for step in todo_state.steps:
            status_emoji = {
                "pending": "⏳",
                "success": "✅",
                "failed": "❌"
            }.get(step.status, "⏳")
            
            priority_emoji = {
                "high": "🔴", 
                "medium": "🟡", 
                "low": "🟢"
            }.get(step.priority, "🟡")
            
            category_text = f" [{step.category}]" if step.category else ""
            result += f"**{step.id}** {priority_emoji} {status_emoji} {step.name}{category_text}\n"
            
            if step.parameters:
                result += f"  Parameters: {json.dumps(step.parameters)}\n"
            
            if step.result:
                result += f"  Result: {step.result}\n"
            
            result += f"  Created: {step.created_at}"
            if step.updated_at:
                result += f", Updated: {step.updated_at}"
            result += "\n\n"
        
        return ToolResponse(
            success=True, 
            message=result,
            extra=ToolExtra(
                file_path=todo_state.todo_file,
                data={
                    "total_steps": len(todo_state.steps),
                    "steps": [step.model_dump() for step in todo_state.steps],
                    "pending_count": len([s for s in todo_state.steps if s.status == "pending"]),
                    "completed_count": len([s for s in todo_state.steps if s.status in ["success", "failed"]])
                }
            )
        )
    
    async def _clear_completed(self, todo_state: TodoState) -> ToolResponse:
        """Remove all completed steps from the todo list."""
        completed_steps = [step for step in todo_state.steps if step.status in ["success", "failed"]]
        
        if not completed_steps:
            return ToolResponse(success=False, message="No completed steps to remove")
        
        # Remove completed steps
        todo_state.steps = [step for step in todo_state.steps if step.status == "pending"]
        
        # Save and sync
        await self._save_steps(todo_state)
        await self._sync_to_markdown(todo_state)
        
        return ToolResponse(
            success=True, 
            message=f"✅ Removed {len(completed_steps)} completed step(s)",
            extra=ToolExtra(
                file_path=todo_state.todo_file,
                data={
                    "removed_count": len(completed_steps),
                    "removed_steps": [step.model_dump() for step in completed_steps],
                    "remaining_steps": len(todo_state.steps)
                }
            )
        )
    
    async def _show_todo_file(self, todo_state: TodoState) -> ToolResponse:
        """Show the complete todo.md file content."""
        if not os.path.exists(todo_state.todo_file):
            return ToolResponse(success=False, message="No todo file found. Use 'add' action to create your first step.")
        
        async with file_lock(todo_state.todo_file):
            with open(todo_state.todo_file, 'r', encoding='utf-8') as f:
                content = f.read()
        
        return ToolResponse(
            success=True, 
            message=f"📄 Todo.md content:\n\n```markdown\n{content}\n```",
            extra=ToolExtra(
                file_path=todo_state.todo_file,
                data={
                    "content": content,
                    "file_size": len(content),
                    "total_steps": len(todo_state.steps)
                }
            )
        )
    
    async def _export_todo_file(self, todo_state: TodoState, export_path: str) -> ToolResponse:
        """Export todo.md file to a specified path."""
        if not export_path:
            return ToolResponse(success=False, message="Error: Export path is required for export action")
        
        try:
            # Ensure the todo.md file is up to date
            await self._sync_to_markdown(todo_state)
            
            if not os.path.exists(todo_state.todo_file):
                return ToolResponse(success=False, message="No todo file found. Use 'add' action to create your first step.")
            
            # Create parent directories if they don't exist
            export_dir = os.path.dirname(export_path)
            if export_dir:
                os.makedirs(export_dir, exist_ok=True)
            
            # Read the current todo.md content
            async with file_lock(todo_state.todo_file):
                with open(todo_state.todo_file, 'r', encoding='utf-8') as f:
                    content = f.read()
            
            # Write to the export path
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return ToolResponse(
                success=True, 
                message=f"✅ Successfully exported todo.md to: {export_path}",
                extra=ToolExtra(
                    file_path=[todo_state.todo_file, export_path],
                    data={
                        "source_file": todo_state.todo_file,
                        "export_path": export_path,
                        "file_size": len(content),
                        "total_steps": len(todo_state.steps)
                    }
                )
            )
            
        except Exception as e:
            return ToolResponse(success=False, message=f"Error exporting todo.md: {str(e)}")
    
    def get_todo_content(self, call_id: str) -> str:
        """Get the content of the todo.md file for a specific call_id.
        
        Note: This is a synchronous method for compatibility. For async usage,
        prefer using the 'show' action.
        """
        if call_id not in self._todo_cache:
            return "[Current todo.md is empty, fill it with your plan when applicable]"
        
        todo_state = self._todo_cache[call_id]
        if not os.path.exists(todo_state.todo_file):
            return "[Current todo.md is empty, fill it with your plan when applicable]"
        
        try:
            with open(todo_state.todo_file, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception:
            return "[Current todo.md is empty, fill it with your plan when applicable]"

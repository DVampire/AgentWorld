"""Todo tool for managing todo.md file with task decomposition and step tracking."""

import json
import uuid
import os
from datetime import datetime
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
import tempfile

from src.tool.types import Tool, ToolResponse
from src.utils import assemble_project_path
from src.utils import file_lock

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


_TODO_TOOL_DESCRIPTION = """Todo tool for managing a todo.md file with task decomposition and step tracking.
When using this tool, only provide parameters that are relevant to the specific operation you are performing. Do not include unnecessary parameters.

Available operations:
1. add: Add a new step to the todo list at the end or after a specific step.
    - task: The description of the step.
    - priority: The priority of the step.
    - category: The category of the step.
    - parameters: Optional parameters for the step.
    - after_step_id: Optional step ID to insert after (if not provided, adds to end).
2. complete: Mark step as completed (success or failed).
    - step_id: The ID of the step to complete.
    - status: Completion status: "success" or "failed".
    - result: Result description (1-3 sentences).
3. update: Update step information.
    - step_id: The ID of the step to update.
    - task: New step description.
    - parameters: New step parameters.
4. list: List all steps with their status.
5. clear: Clear completed steps.
6. show: Show the complete todo.md file content.
7. export: Export todo.md to a specified path.
    - export_path: The target path to export the todo.md file.

Input format: JSON string with 'action' and action-specific parameters.
Example: {"name": "todo", "args": {"action": "add", "task": "Task description", "priority": "high", "category": "work"}}

The todo.md file is maintained in the current working directory and follows a structured format for task management.
"""


class TodoTool(Tool):
    """A tool for managing a todo.md file with task decomposition and step tracking."""
    
    name: str = "todo"
    description: str = _TODO_TOOL_DESCRIPTION
    
    todo_file: Optional[str] = None
    steps_file: Optional[str] = None
    steps: Optional[List[Step]] = None
    
    def __init__(self, **kwargs):
        """A tool for managing a todo.md file with task decomposition and step tracking."""
        super().__init__(**kwargs)
        # Use temporary file for todo.md
        temp_dir = assemble_project_path(tempfile.mkdtemp())
        self.todo_file = assemble_project_path(os.path.join(temp_dir, "todo.md"))
        self.steps_file = assemble_project_path(os.path.join(temp_dir, "todo_steps.json"))
        self.steps = []
        self._load_steps()
    
    async def __call__(
        self, 
        action: str, 
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
        """
        Execute a todo action asynchronously.

        Args:
            action (str): One of add, complete, update, list, clear, show, export.
            task (Optional[str]): Step description for add/update.
            step_id (Optional[str]): Step identifier for complete/update actions.
            status (Optional[str]): Completion status for complete action.
            result (Optional[str]): Result summary for complete action.
            priority (Optional[str]): Priority for add action.
            category (Optional[str]): Category label for add action.
            parameters (Optional[Dict[str, Any]]): Arbitrary metadata for steps.
            after_step_id (Optional[str]): Insert new step after the specified ID.
            export_path (Optional[str]): Target path for export action.
        """
        try:
            if action == "add":
                return await self._add_step(task, priority, category, parameters, after_step_id)
            if action == "complete":
                return await self._complete_step(step_id, status, result)
            if action == "update":
                return await self._update_step(step_id, task, parameters)
            if action == "list":
                return await self._list_steps()
            if action == "clear":
                return await self._clear_completed()
            if action == "show":
                return await self._show_todo_file()
            if action == "export":
                return await self._export_todo_file(export_path)
            return ToolResponse(
                success=False,
                message=(
                    f"Unknown action: {action}. "
                    "Available actions: add, complete, update, list, clear, show, export"
                ),
            )
        except Exception as e:
            return ToolResponse(success=False, message=f"Error executing todo action: {str(e)}")
    
    def _load_steps(self) -> None:
        """Load steps from JSON file."""
        try:
            # Note: Not using file_lock here since __init__ cannot be async
            # This is safe during initialization as there's no concurrency
            if os.path.exists(self.steps_file):
                with open(self.steps_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.steps = [Step(**step_data) for step_data in data]
            else:
                self.steps = []
        except Exception:
            self.steps = []
    
    async def _save_steps(self) -> None:
        """Save steps to JSON file."""
        try:
            async with file_lock(self.steps_file):
                with open(self.steps_file, 'w', encoding='utf-8') as f:
                    json.dump([step.model_dump() for step in self.steps], f, indent=2, ensure_ascii=False)
        except Exception:
            pass
    
    async def _sync_to_markdown(self) -> None:
        """Sync steps list to markdown file."""
        try:
            async with file_lock(self.todo_file):
                content = "# Todo List\n\n"
                
                for step in self.steps:
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
                
                with open(self.todo_file, 'w', encoding='utf-8') as f:
                    f.write(content)
        except Exception:
            pass
    
    def _generate_step_id(self) -> str:
        """Generate a unique step ID using UUID + timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]  # Use first 8 characters of UUID
        return f"{timestamp}_{unique_id}"
    
    async def _add_step(self, 
                        task: str, 
                        priority: str = "medium",
                        category: Optional[str] = None, 
                        parameters: Optional[Dict[str, Any]] = None, 
                        after_step_id: Optional[str] = None
                        ) -> ToolResponse:
        """Add a new step to the todo list."""
        if not task:
            return ToolResponse(success=False, message="Error: Step description is required for add action")
        
        # Create new step
        step_id = self._generate_step_id()
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
            # Find the position to insert after
            insert_index = -1
            for i, step in enumerate(self.steps):
                if step.id == after_step_id:
                    insert_index = i + 1
                    break
            
            if insert_index == -1:
                return ToolResponse(success=False, message=f"Error: Step {after_step_id} not found. Adding to end of list.")
            
            # Insert at the found position
            self.steps.insert(insert_index, new_step)
        else:
            # Add to end of list
            self.steps.append(new_step)
        
        # Save and sync
        await self._save_steps()
        await self._sync_to_markdown()
        
        if after_step_id:
            return ToolResponse(success=True, message=f"✅ Added step {step_id} after {after_step_id}: {task} (priority: {priority})")
        else:
            return ToolResponse(success=True, message=f"✅ Added step {step_id}: {task} (priority: {priority})")
    
    async def _complete_step(self, step_id: str, status: str, result: Optional[str] = None) -> ToolResponse:
        """Mark step as completed."""
        if not step_id:
            return ToolResponse(success=False, message="Error: Step ID is required for complete action")
        
        if status not in ["success", "failed"]:
            return ToolResponse(success=False, message="Error: Status must be 'success' or 'failed'")
        
        # Find the step
        step = None
        for s in self.steps:
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
        await self._save_steps()
        await self._sync_to_markdown()
        
        return ToolResponse(success=True, message=f"✅ Completed step {step_id} with status: {status}")
    
    async def _update_step(self, step_id: str, task: Optional[str] = None, parameters: Optional[Dict[str, Any]] = None) -> ToolResponse:
        """Update step information."""
        if not step_id:
            return ToolResponse(success=False, message="Error: Step ID is required for update action")
        
        # Find the step
        step = None
        for s in self.steps:
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
        await self._save_steps()
        await self._sync_to_markdown()
        
        return ToolResponse(success=True, message=f"✅ Updated step {step_id}")
    
    async def _list_steps(self) -> ToolResponse:
        """List all steps with their status."""
        if not self.steps:
            return ToolResponse(success=False, message="No steps found. Use 'add' action to create your first step.")
        
        result = "📋 Todo Steps:\n\n"
        for step in self.steps:
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
        
        return ToolResponse(success=True, message=result) 
    
    async def _clear_completed(self) -> ToolResponse:
        """Remove all completed steps from the todo list."""
        completed_steps = [step for step in self.steps if step.status in ["success", "failed"]]
        
        if not completed_steps:
            return ToolResponse(success=False, message="No completed steps to remove")
        
        # Remove completed steps
        self.steps = [step for step in self.steps if step.status == "pending"]
        
        # Save and sync
        await self._save_steps()
        await self._sync_to_markdown()
        
        return ToolResponse(success=True, message=f"✅ Removed {len(completed_steps)} completed step(s)")
    
    async def _show_todo_file(self) -> ToolResponse:
        """Show the complete todo.md file content."""
        if not os.path.exists(self.todo_file):
            return ToolResponse(success=False, message="No todo file found. Use 'add' action to create your first step.")
        
        with open(self.todo_file, 'r', encoding='utf-8') as f:
            content = f.read()
        return ToolResponse(success=True, message=f"📄 Todo.md content:\n\n```markdown\n{content}\n```")
    
    async def _export_todo_file(self, export_path: str) -> ToolResponse:
        """Export todo.md file to a specified path."""
        if not export_path:
            return ToolResponse(success=False, message="Error: Export path is required for export action")
        
        try:
            # Ensure the todo.md file is up to date
            await self._sync_to_markdown()
            
            if not os.path.exists(self.todo_file):
                return ToolResponse(success=False, message="No todo file found. Use 'add' action to create your first step.")
            
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            
            # Read the current todo.md content
            with open(self.todo_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Write to the export path
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return ToolResponse(success=True, message=f"✅ Successfully exported todo.md to: {export_path}")
            
        except Exception as e:
            return ToolResponse(success=False, message=f"Error exporting todo.md: {str(e)}")
    
    def get_todo_content(self) -> str:
        """Get the content of the todo.md file."""
        if not os.path.exists(self.todo_file):
            return "[Current todo.md is empty, fill it with your plan when applicable]"
        with open(self.todo_file, 'r', encoding='utf-8') as f:
            todo_contents = f.read()
        return todo_contents
    
    async def export_todo_file(self, export_path: str):
        """Export todo.md file to a specified path."""
        if not export_path:
            return
        try:
            # Ensure the todo.md file is up to date
            await self._sync_to_markdown()
            
            if not os.path.exists(self.todo_file):
                return
            
            # Create parent directories if they don't exist
            os.makedirs(os.path.dirname(export_path), exist_ok=True)
            
            # Read the current todo.md content
            with open(self.todo_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Write to the export path
            with open(export_path, 'w', encoding='utf-8') as f:
                f.write(content)
        except Exception as e:
            return
    

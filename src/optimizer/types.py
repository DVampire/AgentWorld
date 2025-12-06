"""
Base optimizer module.
Contains shared logic for all optimizers, including variable extraction and cache management.
"""

from typing import List, Dict, Tuple, Optional, Any
from abc import ABC, abstractmethod

from src.logger import logger


class BaseOptimizer(ABC):
    """Base optimizer that provides shared functionality such as variable extraction and cache management."""
    
    def __init__(self, agent):
        self.agent = agent
        self.optimizable_vars = []
        self.var_mapping = {}  # Mapping from variable to prompt object.
        self.prompt_mapping = {}  # Mapping from variable to prompt object (used by certain optimizers).
    
    def find_prompt_objects_with_variables(self) -> List[Tuple[Any, str]]:
        """
        Find all prompt objects on the agent that contain `Variable` instances.

        The method traverses the prompt manager to discover prompt objects.

        Returns:
            List[Tuple[prompt_obj, prompt_name]]: A list of (prompt object, prompt name) pairs.
        """
        prompt_objects = []
        
        # Search through the prompt manager for prompt objects.
        if hasattr(self.agent, 'prompt_manager'):
            pm = self.agent.prompt_manager
            
            # SystemPrompt.
            if hasattr(pm, 'system_prompt') and hasattr(pm.system_prompt, 'prompt'):
                prompt_objects.append((pm.system_prompt, 'prompt_manager.system_prompt'))
            
            # AgentMessagePrompt.
            if hasattr(pm, 'agent_message_prompt') and hasattr(pm.agent_message_prompt, 'prompt'):
                prompt_objects.append((pm.agent_message_prompt, 'prompt_manager.agent_message_prompt'))
        
        return prompt_objects
    
    def _extract_from_variable_recursive(self, var, parent_name: str = ""):
        """
        Recursively extract variables with `require_grad=True`.

        Args:
            var: Variable instance.
            parent_name: Parent variable name used during recursion.

        Returns:
            List[orig_var]: List of optimizable variables.
        """
        result = []
        
        # Check whether the current variable should be optimized.
        if hasattr(var, 'require_grad') and var.require_grad:
            result.append(var)
        
        # Recursively process child variables.
        if hasattr(var, 'variables'):
            if isinstance(var.variables, list):
                for child in var.variables:
                    if hasattr(child, 'require_grad'):
                        result.extend(self._extract_from_variable_recursive(
                            child, f"{parent_name}.{var.name}" if parent_name else var.name
                        ))
            elif hasattr(var.variables, 'require_grad'):
                result.extend(self._extract_from_variable_recursive(
                    var.variables, f"{parent_name}.{var.name}" if parent_name else var.name
                ))
        
        return result
    
    def extract_optimizable_variables(self) -> Tuple[List[Any], Dict[Any, Any]]:
        """
        Extract optimizable variables (`require_grad=True`) from all prompt objects on the agent.

        This is a generic extraction method; subclasses may override it to return a custom structure.

        Returns:
            Tuple[List[orig_var], Dict[orig_var -> prompt_obj]]:
                (List of optimizable variables, mapping from variable to owning prompt object.)
        """
        all_optimizable_vars = []
        prompt_mapping = {}  # orig_var -> prompt_obj
        
        prompt_objects = self.find_prompt_objects_with_variables()
        
        for prompt_obj, prompt_name in prompt_objects:
            if hasattr(prompt_obj, 'prompt'):
                prompt_var = prompt_obj.prompt
                optimizable_vars = self._extract_from_variable_recursive(prompt_var)
                all_optimizable_vars.extend(optimizable_vars)
                
                # Record which prompt object owns each variable.
                for orig_var in optimizable_vars:
                    prompt_mapping[orig_var] = prompt_obj
        
        logger.info(f"| 📊 Found {len(all_optimizable_vars)} optimizable variables from {len(prompt_objects)} prompt object(s):")
        for orig_var in all_optimizable_vars:
            prompt_obj = prompt_mapping.get(orig_var)
            prompt_name = getattr(prompt_obj, '__class__', type(prompt_obj)).__name__ if prompt_obj else 'unknown'
            var_name = orig_var.name if hasattr(orig_var, 'name') else 'unknown'
            var_desc = orig_var.description if hasattr(orig_var, 'description') else f"Prompt module: {var_name}"
            logger.info(f"|   - [{prompt_name}] {var_name}: {var_desc}")
        
        return all_optimizable_vars, prompt_mapping
    
    def clear_prompt_caches(self, vars_to_clear: Optional[List[Any]] = None):
        """
        Clear the cached prompts for any prompt objects that contain the given variables.

        Args:
            vars_to_clear: List of variables whose prompt caches should be cleared.
                If None, all recorded variables are considered.

        Reloading details:
        - After clearing the cache, when the agent calls `prompt_obj.get_message()` (typically inside `_get_messages()`),
          if `reload=False` and `message` is `None`, the prompt automatically re-renders (`prompt.render()`).
          At that moment the updated variable values are applied.
        - Relevant locations:
          * `ToolCallingAgent._get_messages()` -> `prompt_manager.get_system_message()`
          * `SystemPrompt.get_message()` -> if `message` is `None`, `prompt.render(modules)` runs
        """
        if vars_to_clear is None:
            vars_to_clear = self.optimizable_vars
        
        # Collect all prompt objects whose caches should be cleared (deduplicated).
        prompt_objects_to_clear = set()
        
        # Look up prompt objects using `var_mapping` (used by Reflection-based optimizers).
        for orig_var in vars_to_clear:
            if orig_var in self.var_mapping:
                prompt_obj = self.var_mapping[orig_var]
                prompt_objects_to_clear.add(prompt_obj)
        
        # Look up prompt objects using `prompt_mapping` (used by TextGrad-based optimizers).
        # Subclasses can override this method for specialized handling.
        
        # Clear the cache on each prompt object.
        for prompt_obj in prompt_objects_to_clear:
            # Both `SystemPrompt` and `AgentMessagePrompt` expose a `message` attribute.
            if hasattr(prompt_obj, 'message'):
                prompt_obj.message = None
                prompt_name = getattr(prompt_obj, '__class__', type(prompt_obj)).__name__
                logger.debug(f"| 🗑️ Cleared cache for {prompt_name}")
        
        if prompt_objects_to_clear:
            logger.info(f"| 🗑️ Cleared cache for {len(prompt_objects_to_clear)} prompt object(s)")
    
    def get_variable_value(self, var: Any) -> str:
        """
        Return the current value of the variable as a string.

        Args:
            var: Variable instance.

        Returns:
            str: Variable value.
        """
        if hasattr(var, 'get_value'):
            return var.get_value()
        elif hasattr(var, 'variables'):
            return str(var.variables)
        elif hasattr(var, 'value'):
            return str(var.value)
        else:
            return str(var)
    
    def set_variable_value(self, var: Any, value: str):
        """
        Assign a new value to the variable.

        Args:
            var: Variable instance.
            value: New value.
        """
        if hasattr(var, 'variables'):
            var.variables = value
        elif hasattr(var, 'value'):
            var.value = value
        else:
            raise ValueError(f"Cannot set value for variable {type(var)}")
    
    @abstractmethod
    async def optimize(
        self,
        task: str,
        files: Optional[List[str]] = None,
        optimization_steps: int = 3,
        **kwargs
    ):
        """
        Execute the optimization routine (abstract; subclasses must implement).

        Args:
            task: Task description.
            files: Optional list of attachment paths.
            optimization_steps: Number of optimization iterations.
            **kwargs: Additional optimizer-specific parameters.
        """
        pass
    
    def close(self):
        """Close the optimizer and release resources."""
        pass


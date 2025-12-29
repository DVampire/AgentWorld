from typing import List, Optional, Any, Dict
from pydantic import ConfigDict, Field

from src.logger import logger
from src.optimizer.types import Optimizer, Variable
from src.model import model_manager
from src.memory import memory_manager
from src.tool import tcp


class ReflectionOptimizer(Optimizer):
    """Optimizer that improves agent prompts using the Reflection method."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    prompt_name: str = Field(default="reflection_optimizer", description="The name of the prompt")
    model_name: str = Field(default="openrouter/gpt-4o", description="The name of the model")
    memory_name: Optional[str] = Field(default=None, description="Name of the optimizer memory system for recording optimization history")
    
    def __init__(self, 
                 workdir: str,
                 prompt_name: str = "reflection_optimizer",
                 model_name: str = "openrouter/gpt-4o", 
                 memory_name: Optional[str] = "optimizer_memory_system",
                 **kwargs
                 ):
        """
        Initialize the optimizer.

        Args:
            agent: Agent instance.
            model_name: Model name for optimization.
            memory_name: Optional name of the optimizer memory system for recording optimization history.
        """
        super().__init__( 
                         workdir=workdir, 
                         prompt_name=prompt_name,
                         model_name=model_name,
                         memory_name=memory_name,
                         **kwargs)
        self.workdir = workdir
        if model_name:
            self.model_name = model_name
        if prompt_name:
            self.prompt_name = prompt_name
        self.memory_name = memory_name
        
        
    async def get_trainable_variables(self, agent: Optional[Any] = None) -> Dict[str, Any]:
        """
        Get trainable variables from prompt and tools only.
        
        Returns:
            Dict[str, Variable]: Dictionary mapping variable names to Variable objects.
        """
        # Lazy import to avoid circular dependency
        from src.prompt import prompt_manager
        
        variables: Dict[str, Any] = {}
        
        # Get trainable variables from prompt (returns Dict[str, Variable])
        if agent and hasattr(agent, 'prompt_name'):
            prompt_name = agent.prompt_name
            prompt_variables_dict = await prompt_manager.get_trainable_variables(prompt_name=prompt_name)
            variables.update(prompt_variables_dict)
        
        # Get trainable variables from tools (returns Dict[str, Variable])
        tool_variables_dict = await tcp.get_trainable_variables()
        variables.update(tool_variables_dict)
        
        return variables
    
    async def _format_variables(self, variables: Dict[str, Any]) -> str:
        """
        Format variables for context.
        
        Args:
            variables (Dict[str, Any]): Dictionary of variables.
        """
        
        variables_text = ""
        
        # Step1: Format prompt variables
        prompt_variables_text = "<prompt_variables>\n"
        prompt_variables = {k: v for k, v in variables.items() if isinstance(v, Variable) and (v.type == "system_prompt" or v.type == "agent_message_prompt")}
        for prompt_index, (prompt_name, prompt_variable) in enumerate(prompt_variables.items()):
            prompt_variables_text += f"<prompt_variable_{prompt_index:04d}>\n"
            prompt_variables_text += f"Name: {prompt_name}\n"
            prompt_variables_text += f"Description: {prompt_variable.description}\n"
            
            # Format sub-variables if they exist
            if isinstance(prompt_variable.variables, dict):
                sub_vars = {k: v for k, v in prompt_variable.variables.items() if isinstance(v, Variable)}
                if sub_vars:
                    prompt_variables_text += "<sub_variables>\n"
                    for sub_index, (sub_name, sub_var) in enumerate(sub_vars.items()):
                        prompt_variables_text += f"  <sub_variable_{sub_index:04d}>\n"
                        prompt_variables_text += f"    Name: {sub_name}\n"
                        prompt_variables_text += f"    Description: {sub_var.description}\n"
                        try:
                            sub_value = sub_var.get_value() if hasattr(sub_var, 'get_value') else str(sub_var)
                        except Exception as e:
                            sub_value = f"<Error getting value: {e}>"
                        prompt_variables_text += f"    ```text\n{sub_value}\n```\n"
                        prompt_variables_text += f"  </sub_variable_{sub_index:04d}>\n"
                    prompt_variables_text += "</sub_variables>\n"
            
            prompt_variables_text += f"</prompt_variable_{prompt_index:04d}>\n"
        prompt_variables_text += "</prompt_variables>\n"
        variables_text += prompt_variables_text
        
        # Step2: Format tool variables
        tool_variables_text = "<tool_variables>\n"
        tool_variables = {k: v for k, v in variables.items() if v.type == "tool_code"}
        for index, (tool_name, tool_variable) in enumerate(tool_variables.items()):
            
            tool_variables_text += f"<tool_variable_{index:04d}>\n"
            tool_variables_text += f"Name: {tool_name}\n"
            tool_variables_text += f"Description: {tool_variable.description}\n"
            tool_variables_text += f"```python\n{tool_variable.get_value()}\n```\n"
            tool_variables_text += f"</tool_variable_{index:04d}>\n"
            
        tool_variables_text += "</tool_variables>\n"
        variables_text += tool_variables_text
        
        # Step3: Format solution variable
        solution_variable_text = "<solution_variable>\n"
        solution_variables = {k: v for k, v in variables.items() if v.type == "solution"}
        for solution_index, (solution_name, solution_variable) in enumerate(solution_variables.items()):
            solution_variable_text += f"<solution_variable_{solution_index:04d}>\n"
            solution_variable_text += f"Name: {solution_name}\n"
            solution_variable_text += f"Description: {solution_variable.description}\n"
            solution_variable_text += f"```text\n{solution_variable.get_value()}\n```\n"
            solution_variable_text += f"</solution_variable_{solution_index:04d}>\n"
        solution_variable_text += "</solution_variable>\n"
        variables_text += solution_variable_text
        
        return variables_text
        
    
    async def _generate_reflection(self, task: str, variables: Dict[str, Any], execution_result: str) -> str:
        """
        Generate the reflection analysis for all variables.

        Args:
            task (str): Task description.
            variables (Dict[str, Any]): Dictionary of variables.
            execution_result (str): Agent execution result.
        Returns:
            str: Reflection analysis identifying which variables to optimize and how.
        """
        # Lazy import to avoid circular dependency
        from src.prompt import prompt_manager
        
        # Ensure prompt_manager is initialized
        if not hasattr(prompt_manager, 'prompt_context_manager'):
            await prompt_manager.initialize()
        
        variables_text = await self._format_variables(variables)
 
        system_modules = {}
        agent_message_modules = {
            "task": task,
            "variables": variables_text,
            "execution_result": execution_result,
        }
        messages = await prompt_manager.get_messages(
            prompt_name=f"{self.prompt_name}_reflection",
            system_modules=system_modules,
            agent_modules=agent_message_modules,
        )
        
        logger.info(f"| 🤔 Generating reflection analysis for all variables...")
        
        try:
            response = await model_manager(model=self.model_name, messages=messages)
            reflection_text = response.message if hasattr(response, 'message') else str(response)
            
            logger.info(f"| ✅ Reflection analysis generated ({len(reflection_text)} chars)")
            logger.info(f"| Reflection analysis:\n{reflection_text}\n")
            
            return reflection_text
        except Exception as e:
            logger.error(f"| ❌ Error generating reflection: {e}")
            raise
    
    async def _improve_variables(self, task: str, variables: List, reflection_analysis: str, variable_mapping: Dict) -> Dict[str, str]:
        """
        Improve variables based on reflection analysis. May improve multiple variables simultaneously.
        Uses different optimization logic based on variable types.

        Args:
            task (str): Task description.
            variables: List of Variable objects to potentially improve.
            reflection_analysis (str): Reflection analysis output.
            variable_mapping: Mapping from variable name to Variable object.

        Returns:
            Dict[str, str]: Dictionary mapping variable names to improved values.
        """
        # Lazy import to avoid circular dependency
        from src.prompt import prompt_manager
        
        # Ensure prompt_manager is initialized
        if not hasattr(prompt_manager, 'prompt_context_manager'):
            await prompt_manager.initialize()
        
        # Format all variables for context
        all_variables_text = await self._format_all_variables_by_type(variables)
        
        system_modules = {}
        agent_message_modules = {
            "task": task,
            "current_variable": all_variables_text,
            "reflection_analysis": reflection_analysis
        }
        messages = await prompt_manager.get_messages(
            prompt_name=f"{self.prompt_name}_improvement",
            system_modules=system_modules,
            agent_modules=agent_message_modules,
        )
        
        logger.info(f"| ✨ Generating improved variables (may improve multiple variables)...")
        
        try:
            response = await model_manager(model=self.model_name, messages=messages)
            improved_text = response.message if hasattr(response, 'message') else str(response)
            
            # Strip potential Markdown code fences.
            improved_text = improved_text.strip()
            if improved_text.startswith("```"):
                # Remove code block markers.
                lines = improved_text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines[-1].strip() == "```":
                    lines = lines[:-1]
                improved_text = "\n".join(lines).strip()
            
            logger.info(f"| ✅ Improved variables generated ({len(improved_text)} chars)")
            logger.info(f"| Improved variables:\n{improved_text}\n")
            
            # Parse the improved text to extract variable-specific improvements
            # The LLM should return improvements in a structured format
            # For now, we'll try to parse it, but this might need refinement based on actual LLM output
            # TODO: Consider using structured output format or JSON for better parsing
            improved_variables = await self._parse_improved_variables(improved_text, variables, variable_mapping)
            
            return improved_variables
        except Exception as e:
            logger.error(f"| ❌ Error improving variables: {e}")
            raise
    
    async def _parse_improved_variables(self, improved_text: str, variables: List, variable_mapping: Dict) -> Dict[str, str]:
        """
        Parse improved text to extract variable-specific improvements.
        Handles hierarchical variables (e.g., system_prompt with sub-variables).
        
        Args:
            improved_text: Text containing improved variables.
            variables: List of original Variable objects.
            variable_mapping: Mapping from variable name to Variable object.
            
        Returns:
            Dict[str, str]: Dictionary mapping variable names to improved values.
        """
        from src.optimizer.types import Variable
        import re
        
        improved_variables = {}
        
        def collect_all_variable_names(var, collected=None, parent_path=""):
            """Recursively collect all variable names including nested ones."""
            if collected is None:
                collected = {}
            
            var_name = getattr(var, 'name', 'unknown')
            full_path = f"{parent_path}.{var_name}" if parent_path else var_name
            collected[full_path] = var
            collected[var_name] = var  # Also add short name for backward compatibility
            
            # Recursively collect sub-variables
            if isinstance(var.variables, list):
                for sub_var in var.variables:
                    if isinstance(sub_var, Variable):
                        collect_all_variable_names(sub_var, collected, full_path)
            elif isinstance(var.variables, Variable):
                collect_all_variable_names(var.variables, collected, full_path)
            
            return collected
        
        # Collect all variable names (including nested ones)
        all_var_map = {}
        for var in variables:
            collect_all_variable_names(var, all_var_map)
        
        # Try to parse improved text with hierarchical structure support
        # Pattern 1: [variable_name] (type: type) followed by content or sub-variables
        # Match variables at any indentation level
        for var_path, var in all_var_map.items():
            var_name = getattr(var, 'name', '')
            var_type = getattr(var, 'type', '')
            
            # Pattern for variable with sub-variables: [name] (type: type) ... Sub-variables: ... Content: ...
            # Pattern for leaf variable: [name] (type: type) ... Content: ...
            
            # Try to match variable section
            # Look for [var_name] followed by optional type, description, and then Content: or Sub-variables:
            pattern = rf'\[{re.escape(var_name)}\][\s\S]*?(?:\(type:\s*{re.escape(var_type)}\)[\s\S]*?)?(?:Description:[\s\S]*?)?(?:Sub-variables:[\s\S]*?)?Content:\s*([\s\S]*?)(?=\n\s*\[|\n===|$)'
            match = re.search(pattern, improved_text, re.IGNORECASE | re.MULTILINE)
            
            if match:
                content = match.group(1).strip()
                # Remove indentation from content lines
                lines = content.split('\n')
                dedented_lines = []
                for line in lines:
                    # Remove leading spaces (up to 2 levels of indentation)
                    dedented_line = line.lstrip(' ')
                    dedented_lines.append(dedented_line)
                improved_variables[var_name] = '\n'.join(dedented_lines)
        
        # If no variable-specific improvements found, try to match by type sections
        if not improved_variables:
            for var_type in set(getattr(var, 'type', 'unknown') for var in variables):
                # Look for === TYPE === section
                pattern = rf'===\s*{re.escape(var_type.upper())}\s*===[\s\S]*?\[([^\]]+)\][\s\S]*?Content:\s*([\s\S]*?)(?=\n\s*\[|\n===|$)'
                matches = re.finditer(pattern, improved_text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    var_name = match.group(1).strip()
                    content = match.group(2).strip()
                    if var_name in all_var_map:
                        improved_variables[var_name] = content
        
        # If still no improvements found, apply to all top-level variables
        if not improved_variables:
            logger.warning("| ⚠️ Could not parse variable-specific improvements, applying to all top-level variables")
            for var in variables:
                var_name = getattr(var, 'name', 'unknown')
                # Only apply to variables that don't have sub-variables (leaf nodes)
                has_sub_vars = False
                if isinstance(var.variables, list):
                    has_sub_vars = any(isinstance(sub_var, Variable) for sub_var in var.variables)
                elif isinstance(var.variables, Variable):
                    has_sub_vars = True
                
                if not has_sub_vars:
                    improved_variables[var_name] = improved_text
        
        return improved_variables
    
    async def optimize(
        self,
        agent: Any,
        task: str,
        files: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Optimize the agent prompt using the Reflection approach.

        Args:
            agent: Agent instance.
            task: Task description to optimize for.
            files: Optional list of attachments.
        """
        # Use optimization_steps if provided, otherwise use self.max_steps
        optimization_steps = self.max_steps
        
        # Initialize optimizer memory session if available
        memory_name = self.memory_name
        session_id = None
        if memory_name:
            try:
                import uuid
                agent_name = getattr(agent, 'name', 'unknown_agent')
                session_id = f"opt_session_{uuid.uuid4().hex[:8]}"
                await memory_manager.start_session(
                    memory_name=memory_name,
                    session_id=session_id,
                    agent_name=agent_name,
                    description=task
                )
            except Exception as e:
                logger.warning(f"| ⚠️ Failed to initialize optimizer memory: {e}")
                memory_name = None
                session_id = None
                
        # Run agent once to get initial solution
        logger.info(f"| 🚀 Running agent to get initial solution...")
        initial_agent_response = await agent(task=task, files=files)
        initial_agent_response_extra_data = initial_agent_response.extra.data if initial_agent_response.extra and initial_agent_response.extra.data else None
        initial_agent_result = initial_agent_response_extra_data['final_result']
        initial_agent_reasoning = initial_agent_response_extra_data['final_reasoning']
        initial_solution = f"Result: {initial_agent_result}\nReasoning: {initial_agent_reasoning}" if initial_agent_reasoning else f"Result: {initial_agent_result}"
        
        # Create solution variable
        solution_variable = Variable(
            name="solution",
            type="solution",
            description="The solution to the task before optimization.",
            require_grad=True,
            variables=initial_solution
        )
        logger.info(f"| ✅ Initial solution obtained and encapsulated as variable")
        
        # Get trainable variables
        logger.info(f"| 📊 Getting trainable variables...")
        trainable_variables = await self.get_trainable_variables(agent=agent)
        # Add solution variable to trainable variables
        trainable_variables["solution"] = solution_variable
        
        # Run the optimization loop.
        for opt_step in range(optimization_steps):
            logger.info(f"| Reflection Optimization Step {opt_step + 1}/{optimization_steps}")
            
            try:
                # Step1: Generate reflection analysis for all variables.
                reflection_analysis = await self._generate_reflection(
                    task=task,
                    variables=trainable_variables,
                    execution_result=solution_variable.get_value(),
                )
                print(reflection_analysis)
                exit()
                
                # Step2: Improve variables based on reflection (may improve multiple variables).
                improved_variables = await self._improve_variables(
                    task=task,
                    variables=trainable_variables,
                    reflection_analysis=reflection_analysis,
                )
                
                # Step3: Update variables based on improved variables.
                # TODO
                
                # Step4: Run agent with improved variables.
                agent_response = await agent(task=task, files=files)
                agent_response_extra_data = agent_response.extra.data if agent_response.extra and agent_response.extra.data else None
                improved_agent_result = agent_response_extra_data['final_result']
                improved_agent_reasoning = agent_response_extra_data['final_reasoning']
                improved_solution_variable = trainable_variables['solution']
                improved_solution_variable.value = f"Result: {improved_agent_result}\nReasoning: {improved_agent_reasoning}" if improved_agent_reasoning else f"Result: {improved_agent_result}"
                
                logger.info(f"| ✅ Optimization step {opt_step + 1} completed\n")
                
            except Exception as e:
                logger.error(f"| ❌ Error in optimization step {opt_step + 1}: {e}")
                import traceback
                logger.error(f"| Traceback: {traceback.format_exc()}")
                # Continue with the next iteration.
                continue
        
        # End optimization memory session if available
        if memory_name and session_id:
            try:
                await memory_manager.end_session(memory_name=memory_name, session_id=session_id)
                logger.info(f"| 📝 Ended optimization memory session: {session_id}")
            except Exception as e:
                logger.warning(f"| ⚠️ Failed to end optimization memory session: {e}")
        
        logger.info(f"| ✅ Reflection optimization completed!")
        logger.info(f"| {'='*60}")
        logger.info(f"| Reflection optimization completed!")
        logger.info(f"| {'='*60}")


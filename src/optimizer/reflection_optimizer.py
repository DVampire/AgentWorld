from typing import List, Optional, Any, Dict, Union
from pydantic import ConfigDict, Field
from pydantic import BaseModel

from src.logger import logger
from src.optimizer.types import Optimizer, Variable
from src.model import model_manager
from src.memory import EventType

class LeafVariable(BaseModel):
    name: str = Field(description="Name of the leaf variable")
    variables: str = Field(description="Leaf value (no further nesting allowed)")

class ImprovedVariable(BaseModel):
    name: str = Field(description="The name of the variable")
    variables: Optional[Union[str, Dict[str, LeafVariable]]] = Field(default=None,description=("Either a direct string value or a mapping of names to leaf variables. Leaf variables must not contain nested objects."))

class ImprovedVariables(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    variables: Dict[str, ImprovedVariable] = Field(default={}, description="The variables to improve")

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
        from src.tool import tcp
        
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
        
        current_variables_text = await self._format_variables(variables)
 
        system_modules = {}
        agent_message_modules = {
            "task": task,
            "current_variables": current_variables_text,
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
    
    async def _improve_variables(self, task: str, variables: Dict[str, Variable], reflection_analysis: str) -> ImprovedVariables:
        """
        Improve variables based on reflection analysis. May improve multiple variables simultaneously.
        Uses different optimization logic based on variable types.

        Args:
            task (str): Task description.
            variables: List of Variable objects to potentially improve.
            reflection_analysis (str): Reflection analysis output.
            variable_mapping: Mapping from variable name to Variable object.

        Returns:
            {
                "variables": {
                    # prompt variables
                    "tool_calling_system_prompt": {
                        "name": "tool_calling_system_prompt",
                        "variables": {
                            "agent_context_rules": {
                                "name": "agent_context_rules",
                                "variables": "You are a helpful assistant."
                            },
                            "tool_context_rules": {
                                "name": "tool_context_rules",
                                "variables": "You can use the following tools: {tools}"
                            }
                    }
                    # tool variables
                    "bash_tool": {
                        "name": "tool_calling_tool_code",
                        "variables": "tool_code"
                    }
                }
            }
        """
        # Lazy import to avoid circular dependency
        from src.prompt import prompt_manager
        
        # Ensure prompt_manager is initialized
        if not hasattr(prompt_manager, 'prompt_context_manager'):
            await prompt_manager.initialize()

        # Format all variables for context
        current_variables_text = await self._format_variables(variables)
        
        system_modules = {}
        agent_message_modules = {
            "task": task,
            "current_variables": current_variables_text,
            "reflection_analysis": reflection_analysis
        }
        messages = await prompt_manager.get_messages(
            prompt_name=f"{self.prompt_name}_improvement",
            system_modules=system_modules,
            agent_modules=agent_message_modules,
        )
        
        logger.info(f"| ✨ Generating improved variables (may improve multiple variables)...")
        
        try:
            response = await model_manager(model=self.model_name, messages=messages, response_format=ImprovedVariables)
            improved_variables: ImprovedVariables = response.extra.parsed_model
            return improved_variables
        except Exception as e:
            logger.error(f"| ❌ Error improving variables: {e}")
            raise
    
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
        
        # Lazy import to avoid circular dependency
        from src.prompt import prompt_manager
        from src.tool import tcp
        from src.environment import ecp
        from src.agent import acp
        from src.memory import memory_manager
        
        # Use optimization_steps if provided, otherwise use self.max_steps
        optimization_steps = self.max_steps
        
        # Initialize optimizer memory session if available
        memory_name = self.memory_name
        session_id = None
        task_id = None
        if memory_name:
            try:
                import uuid
                from datetime import datetime
                agent_name = getattr(agent, 'name', 'unknown_agent')
                session_id = f"opt_session_{uuid.uuid4().hex[:8]}"
                task_id = f"opt_task_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                await memory_manager.start_session(
                    memory_name=memory_name,
                    session_id=session_id,
                    agent_name=agent_name,
                    description=task
                )
                
                # Add optimization task start event
                await memory_manager.add_event(
                    memory_name=memory_name,
                    step_number=0,
                    event_type=EventType.TASK_START,
                    data=dict(task=task, optimization_steps=optimization_steps),
                    agent_name=agent_name,
                    task_id=task_id,
                    session_id=session_id
                )
            except Exception as e:
                logger.warning(f"| ⚠️ Failed to initialize optimizer memory: {e}")
                memory_name = None
                session_id = None
                task_id = None
                
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
                
                # Step2: Improve variables based on reflection (may improve multiple variables).
                improved_variables = await self._improve_variables(
                    task=task,
                    variables=trainable_variables,
                    reflection_analysis=reflection_analysis,
                )
                
                # Step3: Update variables based on improved variables.
                for variable_name, variable_value in improved_variables.variables.items():
                    variable_type = trainable_variables[variable_name].type
                    
                    if variable_type == "system_prompt" or variable_type == "agent_message_prompt":
                        await prompt_manager.set_variables(
                            prompt_name=variable_name,
                            variable_updates=variable_value
                        )
                    elif variable_type == "tool_code":
                        await tcp.set_variable(variable_name=variable_name, variable_value=variable_value)
                    elif variable_type == "solution":
                        # Variable model uses `variables` to hold the actual content/value.
                        trainable_variables[variable_name].variables = variable_value
                    elif variable_type == "environment_code":
                        await ecp.set_variables(variable_name=variable_name, variable_value=variable_value)
                    elif variable_type == "agent_code":
                        await acp.set_variables(variable_name=variable_name, variable_value=variable_value)
                    elif variable_type == "memory_code":
                        await memory_manager.set_variables(variable_name=variable_name, variable_value=variable_value)
                    else:
                        continue
                
                # Step4: Run agent with improved variables.
                agent_response = await agent(task=task, files=files)
                agent_response_extra_data = agent_response.extra.data if agent_response.extra and agent_response.extra.data else None
                improved_agent_result = agent_response_extra_data['final_result']
                improved_agent_reasoning = agent_response_extra_data['final_reasoning']
                improved_solution_variable = trainable_variables['solution']
                improved_solution_variable.variables = f"Result: {improved_agent_result}\nReasoning: {improved_agent_reasoning}" if improved_agent_reasoning else f"Result: {improved_agent_result}"
                
                # Step5: Record optimization step to memory for learning
                if memory_name and session_id:
                    try:
                        # Prepare execution result string
                        execution_result_str = f"Result: {improved_agent_result}\nReasoning: {improved_agent_reasoning}" if improved_agent_reasoning else f"Result: {improved_agent_result}"
                        
                        # Build comprehensive event data for optimization learning
                        import json
                        event_data = {
                            "task": task,
                            "reflection_analysis": reflection_analysis,
                            "execution_result": execution_result_str,
                            "variable_changes": {}
                        }
                        
                        # Record variable changes (before and after)
                        for var_name, improved_value in improved_variables.variables.items():
                            before_var = trainable_variables.get(var_name)
                            before_value = ""
                            if before_var:
                                if hasattr(before_var, 'get_value'):
                                    try:
                                        before_value = before_var.get_value()
                                    except:
                                        before_value = str(before_var.variables) if hasattr(before_var, 'variables') else str(before_var)
                                elif hasattr(before_var, 'variables'):
                                    before_value = str(before_var.variables)
                                else:
                                    before_value = str(before_var)
                            
                            var_type = before_var.type if before_var and hasattr(before_var, 'type') else ""
                            
                            # Format improved value
                            if isinstance(improved_value, dict):
                                after_value = json.dumps(improved_value, indent=2, ensure_ascii=False)
                            else:
                                after_value = str(improved_value)
                            
                            event_data["variable_changes"][var_name] = {
                                "type": var_type,
                                "before": before_value,
                                "after": after_value
                            }
                        
                        await memory_manager.add_event(
                            memory_name=memory_name,
                            step_number=opt_step + 1,
                            event_type=EventType.OPTIMIZATION_STEP,
                            data=event_data,
                            agent_name=getattr(agent, 'name', 'unknown_agent'),
                            task_id=task_id,
                            session_id=session_id
                        )
                        logger.debug(f"| 📝 Recorded optimization step {opt_step + 1} to memory")
                    except Exception as e:
                        logger.warning(f"| ⚠️ Failed to record optimization step to memory: {e}")
                
                # Step6: Update trainable variables with improved variables.
                trainable_variables = await self.get_trainable_variables(agent=agent)
                trainable_variables["solution"] = improved_solution_variable
                
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
                # Add optimization task end event
                await memory_manager.add_event(
                    memory_name=memory_name,
                    step_number=optimization_steps + 1,
                    event_type=EventType.TASK_END,
                    data=dict(
                        task=task,
                        optimization_steps=optimization_steps,
                        completed=True
                    ),
                    agent_name=getattr(agent, 'name', 'unknown_agent'),
                    task_id=task_id,
                    session_id=session_id
                )
                
                await memory_manager.end_session(memory_name=memory_name, session_id=session_id)
                logger.info(f"| 📝 Ended optimization memory session: {session_id}")
            except Exception as e:
                logger.warning(f"| ⚠️ Failed to end optimization memory session: {e}")
        
        logger.info(f"| ✅ Reflection optimization completed!")
        logger.info(f"| {'='*60}")
        logger.info(f"| Reflection optimization completed!")
        logger.info(f"| {'='*60}")


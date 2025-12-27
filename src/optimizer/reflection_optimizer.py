from typing import List, Optional
from pydantic import ConfigDict, Field

from src.logger import logger
from src.optimizer.types import Optimizer
from src.model import model_manager

class ReflectionOptimizer(Optimizer):
    """Optimizer that improves agent prompts using the Reflection method."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    prompt_name: str = Field(default="reflection_optimizer", description="The name of the prompt")
    model_name: str = Field(default="openrouter/gpt-4o", description="The name of the model")
    
    def __init__(self, agent, model_name: str = "openrouter/gpt-4o", ):
        """
        Initialize the optimizer.

        Args:
            agent: Agent instance.
        """
        super().__init__(agent=agent)
        if model_name:
            self.model_name = model_name
        
    
    async def _generate_reflection(self, task: str, prompt_text: str, execution_result: str) -> str:
        """
        Generate the reflection analysis.

        Args:
            task (str): Task description.
            prompt_text (str): Current prompt text.
            execution_result (str): Agent execution result.

        Returns:
            str: Reflection analysis.
        """
        # Lazy import to avoid circular dependency
        from src.prompt import prompt_manager
        
        # Ensure prompt_manager is initialized
        if not hasattr(prompt_manager, 'prompt_context_manager'):
            await prompt_manager.initialize()
        
        system_modules = {}
        agent_message_modules = {
            "task": task,
            "variable_to_improve": prompt_text,
            "execution_result": str(execution_result)[:2000]  # Clamp length.
        }
        messages = await prompt_manager.get_messages(
            prompt_name=f"{self.prompt_name}_reflection",
            system_modules=system_modules,
            agent_modules=agent_message_modules,
        )
        
        logger.info(f"| 🤔 Generating reflection analysis...")
        
        try:
            response = await model_manager(model=self.model_name, messages=messages)
            reflection_text = response.content if hasattr(response, 'content') else str(response)
            
            logger.info(f"| ✅ Reflection analysis generated ({len(reflection_text)} chars)")
            logger.info(f"| Reflection analysis:\n{reflection_text}\n")
            
            return reflection_text
        except Exception as e:
            logger.error(f"| ❌ Error generating reflection: {e}")
            raise
    
    async def _improve_prompt(self, task: str, current_prompt: str, reflection_analysis: str) -> str:
        """
        Improve the prompt using the reflection analysis.

        Args:
            task (str): Task description.
            current_prompt (str): Current prompt text.
            reflection_analysis (str): Reflection analysis output.

        Returns:
            str: Improved prompt.
        """
        # Lazy import to avoid circular dependency
        from src.prompt import prompt_manager
        
        # Ensure prompt_manager is initialized
        if not hasattr(prompt_manager, 'prompt_context_manager'):
            await prompt_manager.initialize()
            
        system_modules = {}
        agent_message_modules = {
            "task": task,
            "current_variable": current_prompt,
            "reflection_analysis": reflection_analysis
        }
        messages = await prompt_manager.get_messages(
            prompt_name=f"{self.prompt_name}_improvement",
            system_modules=system_modules,
            agent_modules=agent_message_modules,
        )
        
        logger.info(f"| ✨ Generating improved prompt...")
        
        try:
            response = await model_manager(model=self.model_name, messages=messages)
            improved_text = response.content if hasattr(response, 'content') else str(response)
            
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
            
            logger.info(f"| ✅ Improved prompt generated ({len(improved_text)} chars)")
            logger.info(f"| Improved prompt:\n{improved_text}\n")
            
            return improved_text
        except Exception as e:
            logger.error(f"| ❌ Error improving prompt: {e}")
            raise
    
    async def optimize(
        self,
        task: str,
        files: Optional[List[str]] = None,
        optimization_steps: int = 3
    ):
        """
        Optimize the agent prompt using the Reflection approach.

        Args:
            task: Task description to optimize for.
            files: Optional list of attachments.
            optimization_steps: Number of optimization iterations (execute -> reflect -> improve per iteration).
        """
        logger.info(f"| 🚀 Starting Reflection optimization...")
        logger.info(f"| 📋 Task: {task}")
        logger.info(f"| 🔄 Optimization steps: {optimization_steps}")
        logger.info(f"| {'='*60}")
        logger.info(f"| Reflection optimization started")
        logger.info(f"| {'='*60}")
        logger.info(f"| Task: {task}")
        logger.info(f"| Optimization iterations: {optimization_steps}\n")
        
        # 1. Extract optimizable variables.
        logger.info(f"| 📊 Extracting optimizable variables...")
        self.optimizable_vars, self.var_mapping = self.extract_optimizable_variables()
        
        if not self.optimizable_vars:
            logger.warning("| ⚠️ No optimizable variables found (require_grad=True). Skipping optimization.")
            return
        
        # 3. Run the optimization loop.
        for opt_step in range(optimization_steps):
            logger.info(f"\n| {'='*60}")
            logger.info(f"| Reflection Optimization Step {opt_step + 1}/{optimization_steps}")
            logger.info(f"| {'='*60}\n")
            
            try:
                # 3.1 Execute the task.
                logger.info(f"| 🔄 Executing task with current prompt...")
                
                agent_result = await self.agent(task=task, files=files)
                
                logger.info(f"| ✅ Task execution completed")
                logger.info(f"| 📄 Result: {str(agent_result)[:200]}...")
                logger.info(f"| Execution result: {str(agent_result)[:500]}...\n")
                
                # 3.2 Reflect on and improve each optimizable variable.
                for orig_var in self.optimizable_vars:
                    var_name = orig_var.name if hasattr(orig_var, 'name') else 'unknown'
                    var_desc = orig_var.description if hasattr(orig_var, 'description') else f"Prompt module: {var_name}"
                    
                    logger.info(f"\n| 📝 Optimizing variable: {var_name}")
                    logger.info(f"| 📋 Description: {var_desc}")
                    
                    # Retrieve the current variable value.
                    current_value = self.get_variable_value(orig_var)
                    
                    # 3.2.1 Generate reflection analysis.
                    reflection_analysis = await self._generate_reflection(
                        task=task,
                        prompt_text=current_value,
                        execution_result=agent_result,
                    )
                    
                    # 3.2.2 Improve the prompt based on the reflection.
                    improved_value = await self._improve_prompt(
                        task=task,
                        current_prompt=current_value,
                        reflection_analysis=reflection_analysis,
                    )
                    
                    # 3.2.3 Update the variable value.
                    self.set_variable_value(orig_var, improved_value)
                    logger.info(f"| ✅ Variable {var_name} updated")
                
                # 3.3 Clear caches.
                self.clear_prompt_caches()
                
                logger.info(f"| ✅ Optimization step {opt_step + 1} completed\n")
                
            except Exception as e:
                logger.error(f"| ❌ Error in optimization step {opt_step + 1}: {e}")
                import traceback
                logger.error(f"| Traceback: {traceback.format_exc()}")
                # Continue with the next iteration.
                continue
        
        logger.info(f"| ✅ Reflection optimization completed!")
        logger.info(f"| {'='*60}")
        logger.info(f"| Reflection optimization completed!")
        logger.info(f"| {'='*60}")


async def optimize_agent_with_reflection(
    agent,
    task: str,
    files: Optional[List[str]] = None,
    optimization_steps: int = 3
):
    """
    Convenience wrapper that optimizes an agent prompt using Reflection.

    Args:
        agent: Agent instance (must support `__call__` method).
        task: Task description.
        files: Optional list of attachments.
        optimization_steps: Number of optimization iterations (execute -> reflect -> improve).
    """
    optimizer = ReflectionOptimizer(agent)
    try:
        await optimizer.optimize(task, files, optimization_steps)
    finally:
        optimizer.close()
    
    return optimizer


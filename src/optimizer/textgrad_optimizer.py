"""
TextGrad optimizer module.
Contains the optimizer that tunes agent prompts using TextGrad.
"""

from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

from src.logger import logger
import src.optimizer.textgrad as tg
from src.optimizer.types import BaseOptimizer


class TextGradOptimizer(BaseOptimizer):
    """Optimizer that leverages TextGrad to improve agent prompts."""
    
    def __init__(self, agent):
        super().__init__(agent)
        self.optimizable_tg_vars = []  # List of textgrad.Variable instances.
        self.tg_var_mapping = {}  # Mapping from TextGrad variable to original variable (TextGrad specific).
    
    def extract_optimizable_variables(self) -> Tuple[List[tg.Variable], Dict, Dict]:
        """
        Extract optimizable variables (`require_grad=True`) from all agent prompts
        and convert them into `textgrad.Variable` instances.

        Overrides the base implementation to accommodate TextGrad-specific needs.

        Returns:
            Tuple[List[tg.Variable], Dict, Dict]:
                (List of TextGrad variables, variable mapping, prompt mapping)
                - var_mapping: tg_var -> orig_var
                - prompt_mapping: tg_var -> prompt_obj
        """
        # Use the base class extraction routine to obtain the original variables.
        # Note: the base method returns (List[orig_var], Dict[orig_var -> prompt_obj]).
        all_optimizable_vars, orig_prompt_mapping = super().extract_optimizable_variables()
        
        # Convert to `textgrad.Variable`.
        all_optimizable_var_pairs = []
        tg_prompt_mapping = {}  # tg_var -> prompt_obj
        
        for orig_var in all_optimizable_vars:
            # Retrieve the current variable value.
            var_value = self.get_variable_value(orig_var)
            var_desc = orig_var.description if hasattr(orig_var, 'description') else f"Prompt module: {orig_var.name if hasattr(orig_var, 'name') else 'unknown'}"
            
            # Create the TextGrad variable.
            tg_var = tg.Variable(
                value=var_value,
                requires_grad=True,
                role_description=var_desc
            )
            all_optimizable_var_pairs.append((tg_var, orig_var))
            
            # Record which prompt object owns each TextGrad variable.
            if orig_var in orig_prompt_mapping:
                tg_prompt_mapping[tg_var] = orig_prompt_mapping[orig_var]
        
        # Separate the mappings between TextGrad and original variables.
        tg_vars = [tg_var for tg_var, _ in all_optimizable_var_pairs]
        tg_var_mapping = {tg_var: orig_var for tg_var, orig_var in all_optimizable_var_pairs}
        
        # Update `prompt_mapping` so it indexes by TextGrad variables.
        self.prompt_mapping = tg_prompt_mapping
        
        logger.info(f"| 📊 Converted to {len(tg_vars)} TextGrad variables")
        
        return tg_vars, tg_var_mapping, tg_prompt_mapping
    
    def clear_prompt_caches(self, tg_vars: Optional[List[tg.Variable]] = None):
        """
        Clear caches for prompt objects that contain the specified TextGrad variables.

        Args:
            tg_vars: List of TextGrad variables whose prompt caches should be cleared.
                If None, all tracked variables are used.

        Reloading details:
        - After the cache is cleared, when the agent calls `prompt_obj.get_message()` (typically in `_get_messages()`),
          if `reload=False` and `message` is `None`, the prompt automatically re-renders (`prompt.render()`),
          applying the updated variable values.
        - Relevant locations:
          * `ToolCallingAgent._get_messages()` -> `prompt_manager.get_system_message()`
          * `SystemPrompt.get_message()` -> if `message` is `None`, it invokes `prompt.render(modules)`
        """
        if tg_vars is None:
            tg_vars = self.optimizable_tg_vars
        
        # Collect the prompt objects that require cache clearing (deduplicated).
        prompt_objects_to_clear = set()
        for tg_var in tg_vars:
            if tg_var in self.prompt_mapping:
                prompt_obj = self.prompt_mapping[tg_var]
                prompt_objects_to_clear.add(prompt_obj)
        
        # Clear the cache on each prompt object.
        for prompt_obj in prompt_objects_to_clear:
            # Both `SystemPrompt` and `AgentMessagePrompt` expose a `message` attribute.
            if hasattr(prompt_obj, 'message'):
                prompt_obj.message = None
                prompt_name = getattr(prompt_obj, '__class__', type(prompt_obj)).__name__
                logger.debug(f"| 🗑️ Cleared cache for {prompt_name}")
        
        if prompt_objects_to_clear:
            logger.info(f"| 🗑️ Cleared cache for {len(prompt_objects_to_clear)} prompt object(s)")
    
    def define_loss_function(self, agent_result: Any, task: str, max_steps: int) -> tg.TextLoss:
        """
        Define the loss function based on the agent's execution result.

        Args:
            agent_result: Agent execution result.
            task: Original task description.
            max_steps: Maximum number of steps.

        Returns:
            tg.TextLoss: Loss function object.
        """
        # Construct evaluation instructions based on task completion status.
        if agent_result and "success" in str(agent_result).lower():
            # Task completed successfully.
            eval_instruction = (
                f"The agent successfully completed the task: '{task}'. "
                f"The prompt worked well. Identify any remaining areas for improvement "
                f"to make the prompt even more effective and clear."
            )
        elif agent_result and "error" in str(agent_result).lower():
            # Task failed.
            eval_instruction = (
                f"The agent failed to complete the task: '{task}'. "
                f"Result: {str(agent_result)[:200]}. "
                f"Critically analyze what went wrong and provide feedback on how to improve the prompt "
                f"to help the agent better understand and execute the task."
            )
        else:
            # Partial completion or uncertain outcome.
            eval_instruction = (
                f"Evaluate the agent's performance on task: '{task}'. "
                f"Result: {str(agent_result)[:200]}. "
                f"Provide critical feedback on how to improve the prompt to make it clearer and more actionable."
            )
        
        # Create the TextLoss object (standard TextGrad API).
        loss_fn = tg.TextLoss(eval_instruction)
        
        return loss_fn
    
    async def optimize(
        self,
        task: str,
        files: Optional[List[str]] = None,
        optimization_steps: int = 3,
        optimizer_model: str = "gpt-4o"
    ):
        """
        Optimize the agent prompt using TextGrad.

        Args:
            task: Task description.
            files: Optional list of attachments.
            optimization_steps: Number of optimization steps.
            optimizer_model: Model identifier or engine used for optimization.
        """
        # Initialize the log file.
        logger.info(f"| 📋 Task: {task}")
        logger.info(f"| 📂 Files: {files}")
        
        try:
            # 1. Configure the TextGrad backward engine.
            optimizer_engine = tg.get_engine(optimizer_model)
            tg.set_backward_engine(optimizer_engine, override=True)
            
            # 2. Extract optimizable variables from all prompt objects.
            self.optimizable_tg_vars, self.tg_var_mapping, self.prompt_mapping = self.extract_optimizable_variables()
            
            if not self.optimizable_tg_vars:
                logger.warning("| ⚠️ No optimizable variables found. Skipping optimization.")
                return
            
            # 3. Create the optimizer (standard TextGrad API).
            # Provide explicit tags and stricter constraints to improve format adherence.
            optimizer = tg.TextualGradientDescent(
                parameters=self.optimizable_tg_vars,
                engine=optimizer_engine,
                verbose=1,
                constraints=[
                    "Keep the prompt clear and actionable",
                    "Maintain compatibility with the existing tool calling framework",
                    "Do not change the core agent identity",
                    "The prompt should work with the tool calling agent architecture",
                    "You MUST respond with ONLY the improved text between <IMPROVED_VARIABLE> and </IMPROVED_VARIABLE> tags",
                    "Do not include any explanation, feedback, or other text outside the tags"
                ],
                new_variable_tags=["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"]  # Explicitly specify tags.
            )
            
            logger.info(f"| 🔄 Starting TextGrad optimization with {optimization_steps} steps...")
            
            # 4. Iterate through optimization steps.
            for opt_step in range(optimization_steps):
                logger.info(f"\n| {'='*60}")
                logger.info(f"| Optimization Step {opt_step + 1}/{optimization_steps}")
                logger.info(f"| {'='*60}\n")
                
                # 4.1 Synchronize TextGrad variables back to the original variables (after the first step).
                if opt_step > 0:
                    for tg_var, orig_var in self.tg_var_mapping.items():
                        self.set_variable_value(orig_var, tg_var.value)
                    
                    # Clear caches on related prompt objects so updated variables are used.
                    # Reloading happens automatically on the next `get_message()` call (see `clear_prompt_caches` docstring).
                    self.clear_prompt_caches()
                
                # 4.2 Run the agent with the current prompts.
                logger.info(f"| 🚀 Running agent with current prompts...")
                agent_result = await self.agent.ainvoke(task=task, files=files)
                logger.info(f"| 📋 Agent result: {str(agent_result)[:200]}...")
                
                # 4.3 Define the loss function based on the execution result.
                loss_fn = self.define_loss_function(agent_result, task, self.agent.max_steps)
                
                # 4.4 Compute the loss and perform backpropagation.
                logger.info(f"| 📉 Computing loss and gradients...")
                
                # Create the response variable that represents the agent output.
                response_var = tg.Variable(
                    value=str(agent_result)[:1000],  # Limit the length.
                    requires_grad=True,
                    role_description="Agent execution result"
                )
                
                # Compute the loss.
                loss = loss_fn(response_var)
                logger.info(f"| 📊 Loss: {loss.value[:200]}...")
                
                # Manually add loss feedback to prompt variables (they are detached from the computation graph).
                # We therefore create gradients for each optimizable variable explicitly.
                loss_feedback = loss.value  # Use loss value as feedback.
                
                for tg_var in self.optimizable_tg_vars:
                    # Create a gradient variable.
                    gradient_var = tg.Variable(
                        value=loss_feedback,
                        requires_grad=False,
                        role_description=f"Feedback for {tg_var.role_description} based on agent performance"
                    )
                    tg_var.gradients.add(gradient_var)
                    logger.info(f"| 📈 Added gradient for {tg_var.role_description[:50]}...")
                
                # 4.5 Perform the optimizer step (update prompts).
                logger.info(f"| ✨ Updating prompts with TextGrad...")
                try:
                    optimizer.step()
                except IndexError as e:
                    logger.error(f"| ❌ Optimizer step failed: {e}")
                    logger.warning(f"| ⚠️ LLM response may not have followed the required format. Trying with a stronger model or retry...")
                    # Consider adding retry logic or switching to a stronger model here.
                    raise
                
                logger.info(f"| ✅ Optimization step {opt_step + 1} completed\n")
                
                # 4.6 Sync optimized values back to the original variables.
                for tg_var in self.optimizable_tg_vars:
                    if tg_var in self.tg_var_mapping:
                        orig_var = self.tg_var_mapping[tg_var]
                        self.set_variable_value(orig_var, tg_var.value)
                
                # Clear caches on related prompt objects to ensure the next call uses updated values.
                # Reloading happens automatically on the next `get_message()` call (see `clear_prompt_caches` docstring).
                self.clear_prompt_caches()
            
            logger.info(f"| 🎉 Optimization completed!")
            
            # 5. Output a summary of the final optimized variables.
            logger.info(f"| 📊 Final optimized variables (summary):")
            for tg_var in self.optimizable_tg_vars:
                logger.info(f"|   - {tg_var.role_description[:60]}: {tg_var.value[:150]}...")
        
        finally:
            # Close the optimization log file.
            logger.info(f"\n{'='*70}")
            logger.info(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*70}")
            logger.info(f"| 📝 Optimization log saved and closed")
    
    def get_optimized_variables(self) -> List[tg.Variable]:
        """
        Retrieve the list of optimized TextGrad variables.

        Returns:
            List[tg.Variable]: Final optimized variables.
        """
        return self.optimizable_tg_vars


# Convenience wrapper to preserve backward compatibility.
async def optimize_agent_with_textgrad(
    agent,
    task: str,
    files: Optional[List[str]] = None,
    optimization_steps: int = 3,
    optimizer_model: str = "gpt-4o"
):
    """
    Convenience function that optimizes an agent prompt using TextGrad.

    Args:
        agent: Agent instance.
        task: Task description.
        files: Optional list of attachments.
        optimization_steps: Number of optimization steps.
        optimizer_model: Model identifier or engine name for optimization.
    """
    optimizer = TextGradOptimizer(agent)
    await optimizer.optimize(task, files, optimization_steps, optimizer_model)
    return optimizer


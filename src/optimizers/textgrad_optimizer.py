"""
TextGrad optimizer module.
Used to optimize Agent prompts using TextGrad.
"""

from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any


from src.optimizers import textgrad as tg
from src.logger import logger


class TextGradOptimizer:
    """Optimizer that uses TextGrad to optimize Agent prompts."""
    
    def __init__(self, agent):
        """
        Initialize optimizer.
        
        Args:
            agent: Agent instance
        """
        self.agent = agent
        self.optimizable_tg_vars = []
        self.var_mapping = {}  # tg_var -> orig_var
        self.prompt_mapping = {}  # tg_var -> prompt_obj (prompt object containing this variable)
    
    def find_prompt_objects_with_variables(self) -> List[Tuple[Any, str]]:
        """
        Find all prompt objects in the Agent that contain Variable objects.
        
        Returns:
            List[Tuple[prompt_obj, prompt_name]]: List of (prompt object, name) tuples
        """
        prompt_objects = []
        
        # Find all prompt objects from prompt_manager
        if hasattr(self.agent, 'prompt_manager'):
            pm = self.agent.prompt_manager
            
            # SystemPrompt
            if hasattr(pm, 'system_prompt') and hasattr(pm.system_prompt, 'prompt'):
                prompt_objects.append((pm.system_prompt, 'system_prompt'))
            
            # AgentMessagePrompt
            if hasattr(pm, 'agent_message_prompt') and hasattr(pm.agent_message_prompt, 'prompt'):
                prompt_objects.append((pm.agent_message_prompt, 'agent_message_prompt'))
        
        # Can be extended: find other objects that may contain Variables
        # For example: if there are other types of prompt objects, add them here
        
        return prompt_objects
    
    def extract_optimizable_variables(self) -> Tuple[List[tg.Variable], Dict, Dict]:
        """
        Extract optimizable variables (require_grad=True) from all prompt objects in the Agent
        and convert them to textgrad.Variable format.
        
        Returns:
            Tuple[List[tg.Variable], Dict, Dict]: 
                (list of optimizable variables, variable mapping dict, prompt object mapping dict)
                - var_mapping: tg_var -> orig_var
                - prompt_mapping: tg_var -> prompt_obj (prompt object containing this variable)
        """
        optimizable_vars = []
        
        # Find all prompt objects containing Variables
        prompt_objects = self.find_prompt_objects_with_variables()
        
        def extract_from_variable(var, parent_name=""):
            """Recursively extract all variables with require_grad=True and convert them to textgrad.Variable."""
            result = []
            
            # Check if current variable requires gradient
            if hasattr(var, 'require_grad') and var.require_grad:
                # Get current value of variable
                var_value = var.get_value() if hasattr(var, 'get_value') else str(var.variables)
                var_desc = var.description if hasattr(var, 'description') else f"Prompt module: {var.name}"
                
                # Create textgrad.Variable
                tg_var = tg.Variable(
                    value=var_value,
                    requires_grad=True,
                    role_description=var_desc
                )
                result.append((tg_var, var))  # Save original variable for later updates
            
            # Recursively process child variables
            if isinstance(var.variables, list):
                for child in var.variables:
                    if hasattr(child, 'require_grad'):
                        result.extend(extract_from_variable(child, f"{parent_name}.{var.name}"))
            elif hasattr(var.variables, 'require_grad'):
                result.extend(extract_from_variable(var.variables, f"{parent_name}.{var.name}"))
            
            return result
        
        # Extract variables from all prompt objects
        all_optimizable_var_pairs = []
        prompt_mapping = {}  # tg_var -> prompt_obj
        
        for prompt_obj, prompt_name in prompt_objects:
            if hasattr(prompt_obj, 'prompt'):
                prompt_var = prompt_obj.prompt
                optimizable_var_pairs = extract_from_variable(prompt_var)
                all_optimizable_var_pairs.extend(optimizable_var_pairs)
                
                # Record which prompt object each variable belongs to
                for tg_var, orig_var in optimizable_var_pairs:
                    prompt_mapping[tg_var] = prompt_obj
        
        # Separate textgrad variables and original variable mapping
        tg_vars = [tg_var for tg_var, _ in all_optimizable_var_pairs]
        var_mapping = {tg_var: orig_var for tg_var, orig_var in all_optimizable_var_pairs}
        
        logger.info(f"| 📊 Found {len(tg_vars)} optimizable variables from {len(prompt_objects)} prompt object(s):")
        for tg_var, orig_var in all_optimizable_var_pairs:
            prompt_obj = prompt_mapping.get(tg_var)
            prompt_name = getattr(prompt_obj, '__class__', type(prompt_obj)).__name__ if prompt_obj else 'unknown'
            logger.info(f"|   - [{prompt_name}] {orig_var.name if hasattr(orig_var, 'name') else 'unknown'}: {tg_var.role_description}")
        
        return tg_vars, var_mapping, prompt_mapping
    
    def clear_prompt_caches(self, tg_vars: Optional[List[tg.Variable]] = None):
        """
        Clear cache for all prompt objects containing optimized variables.
        
        Args:
            tg_vars: List of variables to clear cache for (if None, clear cache for all recorded variables)
        
        Reload location notes:
        - After clearing cache, when Agent calls prompt object's get_message() method (usually in _get_messages()),
          if reload=False and message is None, it will automatically re-render the prompt (call prompt.render()),
          at which point the updated variable values will be used
        - Specific locations:
          * ToolCallingAgent._get_messages() -> prompt_manager.get_system_message()
          * SystemPrompt.get_message() -> if message is None, will execute prompt.render(modules)
        """
        if tg_vars is None:
            tg_vars = self.optimizable_tg_vars
        
        # Collect all prompt objects that need cache clearing (deduplicate)
        prompt_objects_to_clear = set()
        for tg_var in tg_vars:
            if tg_var in self.prompt_mapping:
                prompt_obj = self.prompt_mapping[tg_var]
                prompt_objects_to_clear.add(prompt_obj)
        
        # Clear cache for each prompt object
        for prompt_obj in prompt_objects_to_clear:
            # SystemPrompt and AgentMessagePrompt both have message attribute
            if hasattr(prompt_obj, 'message'):
                prompt_obj.message = None
                prompt_name = getattr(prompt_obj, '__class__', type(prompt_obj)).__name__
                logger.debug(f"| 🗑️ Cleared cache for {prompt_name}")
        
        if prompt_objects_to_clear:
            logger.info(f"| 🗑️ Cleared cache for {len(prompt_objects_to_clear)} prompt object(s)")
    
    def define_loss_function(self, agent_result: Any, task: str, max_steps: int) -> tg.TextLoss:
        """
        Define loss function based on Agent execution results.
        
        Args:
            agent_result: Agent execution result
            task: Original task description
            max_steps: Maximum number of steps
        
        Returns:
            tg.TextLoss: Loss function object
        """
        # Evaluation instruction based on task completion status
        if agent_result and "success" in str(agent_result).lower():
            # Task completed successfully
            eval_instruction = (
                f"The agent successfully completed the task: '{task}'. "
                f"The prompt worked well. Identify any remaining areas for improvement "
                f"to make the prompt even more effective and clear."
            )
        elif agent_result and "error" in str(agent_result).lower():
            # Task failed
            eval_instruction = (
                f"The agent failed to complete the task: '{task}'. "
                f"Result: {str(agent_result)[:200]}. "
                f"Critically analyze what went wrong and provide feedback on how to improve the prompt "
                f"to help the agent better understand and execute the task."
            )
        else:
            # Partially completed or uncertain
            eval_instruction = (
                f"Evaluate the agent's performance on task: '{task}'. "
                f"Result: {str(agent_result)[:200]}. "
                f"Provide critical feedback on how to improve the prompt to make it clearer and more actionable."
            )
        
        # Create TextLoss object (textgrad standard API)
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
        Optimize Agent prompts using TextGrad.
        
        Args:
            task: Task description
            files: List of attached files
            optimization_steps: Number of optimization iterations
            optimizer_model: Model for optimization (string or engine object)
        """
        # Log optimization start
        logger.info("="*70)
        logger.info("TextGrad Optimization Log")
        logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Task: {task}")
        logger.info(f"Optimization steps: {optimization_steps}")
        logger.info("="*70)
        
        try:
            # 1. Set TextGrad backward propagation engine
            optimizer_engine = tg.get_engine(optimizer_model)
            tg.set_backward_engine(optimizer_engine, override=True)
            
            # 2. Extract optimizable variables (from all prompt objects)
            self.optimizable_tg_vars, self.var_mapping, self.prompt_mapping = self.extract_optimizable_variables()
            
            if not self.optimizable_tg_vars:
                logger.warning("| ⚠️ No optimizable variables found. Skipping optimization.")
                return
            
            # 3. Create optimizer (textgrad standard API)
            # Use more explicit tags and add more constraints to improve format compliance
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
                new_variable_tags=["<IMPROVED_VARIABLE>", "</IMPROVED_VARIABLE>"]  # Explicitly specify tags
            )
            
            logger.info(f"| 🔄 Starting TextGrad optimization with {optimization_steps} steps...")
            
            # 4. Iterative optimization
            for opt_step in range(optimization_steps):
                logger.info(f"\n| {'='*60}")
                logger.info(f"| Optimization Step {opt_step + 1}/{optimization_steps}")
                logger.info(f"| {'='*60}\n")
                
                # 4.1 Synchronize textgrad variables to original variables (if optimized values exist)
                if opt_step > 0:
                    for tg_var, orig_var in self.var_mapping.items():
                        if hasattr(orig_var, 'variables'):
                            orig_var.variables = tg_var.value
                    
                    # Clear cache for all related prompt objects to ensure updated variable values are used
                    # Reload will automatically happen when Agent calls get_message() next time (see clear_prompt_caches comments)
                    self.clear_prompt_caches()
                
                # 4.2 Run Agent with current prompts
                logger.info(f"| 🚀 Running agent with current prompts...")
                agent_result = await self.agent.ainvoke(task=task, files=files)
                logger.info(f"| 📋 Agent result: {str(agent_result)[:200]}...")
                
                # 4.3 Define loss function based on execution results
                loss_fn = self.define_loss_function(agent_result, task, self.agent.max_steps)
                
                # 4.4 Compute loss and perform backpropagation
                logger.info(f"| 📉 Computing loss and gradients...")
                
                # Create response variable for evaluation (represents agent output)
                response_var = tg.Variable(
                    value=str(agent_result)[:1000],  # Limit length
                    requires_grad=True,
                    role_description="Agent execution result"
                )
                
                # Compute loss
                loss = loss_fn(response_var)
                logger.info(f"| 📊 Loss: {loss.value[:200]}...")
                
                # Manually add loss feedback to prompt variables (they are independent, no computation graph connection)
                # We need to manually create gradients for each optimizable variable
                loss_feedback = loss.value  # Get loss value as feedback
                
                for tg_var in self.optimizable_tg_vars:
                    # Create gradient variable
                    gradient_var = tg.Variable(
                        value=loss_feedback,
                        requires_grad=False,
                        role_description=f"Feedback for {tg_var.role_description} based on agent performance"
                    )
                    tg_var.gradients.add(gradient_var)
                    logger.info(f"| 📈 Added gradient for {tg_var.role_description[:50]}...")
                
                # 4.5 Execute optimization step (update prompts)
                logger.info(f"| ✨ Updating prompts with TextGrad...")
                try:
                    optimizer.step()
                except IndexError as e:
                    logger.error(f"| ❌ Optimizer step failed: {e}")
                    logger.warning(f"| ⚠️ LLM response may not have followed the required format. Trying with a stronger model or retry...")
                    # Can add retry logic or use stronger model here
                    raise
                
                logger.info(f"| ✅ Optimization step {opt_step + 1} completed\n")
                
                # 4.6 Synchronize optimized values back to original variables
                for tg_var in self.optimizable_tg_vars:
                    if tg_var in self.var_mapping:
                        orig_var = self.var_mapping[tg_var]
                        if hasattr(orig_var, 'variables'):
                            orig_var.variables = tg_var.value
                
                # Clear cache for all related prompt objects to ensure updated variable values are used next time
                # Reload will automatically happen when Agent calls get_message() next time (see clear_prompt_caches comments)
                self.clear_prompt_caches()
            
            logger.info(f"| 🎉 Optimization completed!")
            
            # 5. Output final optimized variables summary
            logger.info(f"| 📊 Final optimized variables summary:")
            for tg_var in self.optimizable_tg_vars:
                logger.info(f"|   - {tg_var.role_description[:60]}: {tg_var.value[:150]}...")
        
        finally:
            logger.info(f"\n{'='*70}")
            logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*70}")
    
    def get_optimized_variables(self) -> List[tg.Variable]:
        """
        Get list of optimized variables.
        
        Returns:
            List[tg.Variable]: List of optimized variables
        """
        return self.optimizable_tg_vars


# Convenience function: for backward compatibility
async def optimize_agent_with_textgrad(
    agent,
    task: str,
    files: Optional[List[str]] = None,
    optimization_steps: int = 3,
    optimizer_model: str = "gpt-4o"
):
    """
    Optimize Agent prompts using TextGrad (convenience function).
    
    Args:
        agent: Agent instance
        task: Task description
        files: List of attached files
        optimization_steps: Number of optimization iterations
        optimizer_model: Model for optimization (string or engine object)
    """
    optimizer = TextGradOptimizer(agent)
    await optimizer.optimize(task, files, optimization_steps, optimizer_model)
    return optimizer


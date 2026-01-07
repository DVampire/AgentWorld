from os import name
from typing import List, Optional, Any, Dict, Union, Callable
from pydantic import ConfigDict, Field
from pydantic import BaseModel

from src.logger import logger
from src.optimizer.types import Optimizer, Variable
from src.model import model_manager
from src.memory import EventType
import math
import tiktoken


class LeafVariable(BaseModel):
    name: str = Field(description="Name of the leaf variable")
    variables: str = Field(description="Leaf value (no further nesting allowed)")


class ImprovedVariable(BaseModel):
    name: str = Field(description="The name of the variable")
    variables: Optional[Union[str, Dict[str, LeafVariable]]] = Field(default=None, description=(
        "Either a direct string value or a mapping of names to leaf variables. Leaf variables must not contain nested objects."))


class ImprovedVariables(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    variables: Dict[str, ImprovedVariable] = Field(default={}, description="The variables to improve")


class GrpoOptimizer(Optimizer):
    """Optimizer that improves agent prompts using the Reflection method."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    prompt_name: str = Field(default="reflection_optimizer", description="The name of the prompt")
    model_name: str = Field(default="openrouter/gpt-4o", description="The name of the model")
    memory_name: Optional[str] = Field(default=None,
                                       description="Name of the optimizer memory system for recording optimization history")
    benchmark_name: str = Field(default="aime24", description="The name of benchmark")
    num_candidates: int = Field(default=4, description="Number of candidates to generate per variable per step")
    clip_ratio: float = Field(default=0.2, description="Clipping ratio for GRPO")
    beta: float = Field(default=0.01, description="KL penalty coefficient")
    reward_fn: Optional[Callable[[str,str,str], Any]] = Field(default=None, description="Custom reward function for evaluating a single candidate")

    def __init__(self,
                 workdir: str,
                 prompt_name: str = "reflection_optimizer",
                 model_name: str = "openrouter/gpt-4o",
                 memory_name: Optional[str] = "optimizer_memory_system",
                 benchmark_name: str = "aime24",
                 num_candidates: int = 4,
                 clip_ratio: float = 0.2,
                 beta: float = 0.01,
                 reward_fn: Optional[Callable[[str,str,str], Any]] = None,
                 **kwargs
                 ):
        """
        Initialize the optimizer.

        Args:
            agent: Agent instance.
            model_name: Model name for optimization.
            memory_name: Optional name of the optimizer memory system for recording optimization history.
            reward_fn: Optional custom reward function that takes a single candidate text and returns a reward score.
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
        self.benchmark_name = benchmark_name
        # GRPO-like config
        self.num_candidates = num_candidates
        self.clip_ratio = clip_ratio
        self.beta = beta
        self.reward_fn = reward_fn

        # Initialize tokenizer for text similarity calculation
        self.tokenizer = tiktoken.encoding_for_model('gpt-4o')

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
        prompt_variables = {k: v for k, v in variables.items() if
                            isinstance(v, Variable) and (v.type == "system_prompt" or v.type == "agent_message_prompt")}
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

    async def _improve_variables(self, task: str, variables: Dict[str, Variable],
                                 reflection_analysis: str) -> ImprovedVariables:
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


    def _normalize_rewards(self, rewards: List[float]) -> List[float]:
        """Group-normalize rewards into advantages."""
        if len(rewards) <= 1:
            return [0.0] * len(rewards)
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std_reward = math.sqrt(variance) if variance > 1e-8 else 1.0
        return [(r - mean_reward) / std_reward for r in rewards]

    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into tokens for edit distance calculation.
        """
        token_ids = self.tokenizer.encode(text)
        # Use token ids as strings for comparison to avoid decoding overhead
        return [str(tid) for tid in token_ids]

    def _levenshtein_distance(self, tokens1: List[str], tokens2: List[str]) -> int:
        """
        Calculate Levenshtein edit distance between two token sequences.

        :param tokens1: First token sequence
        :param tokens2: Second token sequence
        :return: Edit distance (minimum number of insertions, deletions, or substitutions)
        """
        m, n = len(tokens1), len(tokens2)

        # Create DP table
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        # Initialize: empty sequence to tokens2
        for j in range(n + 1):
            dp[0][j] = j

        # Initialize: tokens1 to empty sequence
        for i in range(m + 1):
            dp[i][0] = i

        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if tokens1[i - 1] == tokens2[j - 1]:
                    # Tokens match, no operation needed
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    # Take minimum of three operations
                    dp[i][j] = min(
                        dp[i - 1][j] + 1,  # Delete token from tokens1
                        dp[i][j - 1] + 1,  # Insert token from tokens2
                        dp[i - 1][j - 1] + 1  # Substitute token
                    )

        return dp[m][n]

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two texts using token-level Levenshtein edit distance.
        """
        if not text1 or not text2:
            return 0.0

        tokens1 = self._tokenize_text(text1)
        tokens2 = self._tokenize_text(text2)

        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        distance = self._levenshtein_distance(tokens1, tokens2)
        max_len = max(len(tokens1), len(tokens2))
        similarity = 1.0 - (distance / max_len)
        return max(0.0, similarity)

    def _apply_clipping_and_kl(self, policy_ratios: List[float], advantages: List[float], kl_divs: List[float]) -> List[float]:
        """Apply GRPO/PPO-style clipping and KL penalty to produce objectives."""
        clipped_objectives = []
        for ratio, advantage, kl in zip(policy_ratios, advantages, kl_divs):
            unclipped = ratio * advantage
            if advantage >= 0:
                clipped_ratio = min(ratio, 1 + self.clip_ratio)
            else:
                clipped_ratio = max(ratio, 1 - self.clip_ratio)
            clipped_obj = clipped_ratio * advantage
            final = min(unclipped, clipped_obj) - self.beta * kl
            clipped_objectives.append(final)
        return clipped_objectives

    async def optimize(
            self,
            agent: Any,
            task: str,
            benchmark_task_id: str,
            files: Optional[List[str]] = None,
            **kwargs
    ):
        """
        Optimize the agent prompt using the Reflection approach.

        Args:
            agent: Agent instance.
            task: Task description to optimize for.
            files: Optional list of attachments.
            reward_fn: Optional custom reward function that takes a single candidate text and returns a reward score.
                      If None, uses the default heuristic evaluation.
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

        # Save initial policy representation for KL calculation (used throughout optimization)
        initial_policy_combined = {}
        for vn, v in trainable_variables.items():
            if not isinstance(v, Variable):
                continue
            initial_policy_combined[vn] = v.get_value() if hasattr(v, 'get_value') else str(v.variables)

        import json
        initial_policy_text = json.dumps(initial_policy_combined, ensure_ascii=False, sort_keys=True)
        logger.info(f"| 📊 Initial policy captured for KL calculations")

        # Run the optimization loop.
        for opt_step in range(optimization_steps):
            logger.info(f"| GRPO Optimization Step {opt_step + 1}/{optimization_steps}")

            try:
                # Capture "before" values for variables we may change, so we can log diffs later
                before_values: Dict[str, str] = {}
                for vn, v in trainable_variables.items():
                    if not isinstance(v, Variable):
                        continue
                    
                    before_values[vn] = v.get_value() if hasattr(v, 'get_value') else str(v.variables)

                # Get current reward before generating candidate
                current_solution = solution_variable.get_value()

                current_reward = await self.reward_fn(benchmark_name=self.benchmark_name, prediction=current_solution, task_id=benchmark_task_id)

                logger.info(f"| 📊 Current reward: {current_reward}")

                # Generate multiple candidate sets (each set may improve multiple variables)
                candidate_sets: List[ImprovedVariables] = []
                for cand_idx in range(self.num_candidates):
                    # Generate reflection analysis for all variables.
                    reflection_analysis = await self._generate_reflection(
                        task=task,
                        variables=trainable_variables,
                        execution_result=current_solution,
                    )

                    # Generate improved variables for each candidate set (each set may improve multiple variables)
                    improved_set = await self._improve_variables(
                        task=task,
                        variables=trainable_variables,
                        reflection_analysis=reflection_analysis,
                    )
                    if improved_set:
                        candidate_sets.append(improved_set)

                # Treat each candidate_set as a whole parameter set. For each candidate set,
                # temporarily apply it, run agent to get improved_solution, then score based on that solution.
                if not candidate_sets:
                    logger.warning("| ⚠️ No candidate sets generated; skipping variable updates this step")
                else:
                    improved_solutions: List[str] = []

                    # For each candidate set, temporarily apply it and run agent to get solution
                    for cand_idx, improved_set in enumerate(candidate_sets):
                        logger.info(f"| 🔄 Evaluating candidate set {cand_idx + 1}/{len(candidate_sets)}")

                        # Temporarily apply the candidate set
                        applied_updates = []
                        for variable_name, improved_entry in improved_set.variables.items():
                            try:
                                variable_type = trainable_variables[variable_name].type if variable_name in trainable_variables else ""
                                candidate_val = improved_entry.variables if hasattr(improved_entry, 'variables') else improved_entry
                                if isinstance(candidate_val, dict):
                                    applied_value = json.dumps(candidate_val, ensure_ascii=False)
                                else:
                                    applied_value = str(candidate_val)

                                # Store original value for rollback
                                original_value = None
                                if variable_type == "system_prompt" or variable_type == "agent_message_prompt":
                                    # Store current variable value from trainable_variables
                                    original_value = ("prompt", variable_name, trainable_variables[variable_name].variables)
                                    await prompt_manager.set_variables(
                                        prompt_name=variable_name,
                                        variable_updates={"variables": applied_value}
                                    )
                                    # Update Variable object for parameter tracking
                                    trainable_variables[variable_name].variables = applied_value
                                elif variable_type == "tool_code":
                                    original_value = ("tool", variable_name, trainable_variables[variable_name].variables)
                                    await tcp.set_variable(variable_name=variable_name, variable_value=applied_value)
                                    # Update Variable object for parameter tracking
                                    trainable_variables[variable_name].variables = applied_value
                                elif variable_type == "solution":
                                    original_value = ("solution", variable_name, trainable_variables[variable_name].variables)
                                    trainable_variables[variable_name].variables = applied_value
                                elif variable_type == "environment_code":
                                    original_value = ("env", variable_name, trainable_variables[variable_name].variables)
                                    await ecp.set_variables(variable_name=variable_name, variable_value=applied_value)
                                    # Update Variable object for parameter tracking
                                    trainable_variables[variable_name].variables = applied_value
                                elif variable_type == "agent_code":
                                    original_value = ("agent", variable_name, trainable_variables[variable_name].variables)
                                    await acp.set_variables(variable_name=variable_name, variable_value=applied_value)
                                    # Update Variable object for parameter tracking
                                    trainable_variables[variable_name].variables = applied_value
                                elif variable_type == "memory_code":
                                    original_value = ("memory", variable_name, trainable_variables[variable_name].variables)
                                    await memory_manager.set_variables(variable_name=variable_name, variable_value=applied_value)
                                    # Update Variable object for parameter tracking
                                    trainable_variables[variable_name].variables = applied_value

                                if original_value:
                                    applied_updates.append(original_value)

                            except Exception as e:
                                logger.warning(f"| ❌ Applying candidate for {variable_name} failed: {e}")

                        # Run agent with the temporarily applied candidate set
                        candidate_response = await agent(task=task, files=files)
                        candidate_response_extra_data = candidate_response.extra.data if candidate_response.extra and candidate_response.extra.data else None
                        candidate_result = candidate_response_extra_data.get('final_result') if candidate_response_extra_data else None
                        candidate_reasoning = candidate_response_extra_data.get('final_reasoning') if candidate_response_extra_data else None
                        improved_solution = f"Result: {candidate_result}\nReasoning: {candidate_reasoning}" if candidate_reasoning else f"Result: {candidate_result}"
                        improved_solutions.append(improved_solution)

                        # Rollback the temporary application
                        for update_type, var_name, original_val in applied_updates:
                            try:
                                if update_type == "prompt":
                                    await prompt_manager.set_variables(prompt_name=var_name, variable_updates={"variables": original_val})
                                    # Update Variable object for parameter tracking
                                    trainable_variables[var_name].variables = original_val
                                elif update_type == "tool":
                                    await tcp.set_variable(variable_name=var_name, variable_value=original_val)
                                    # Update Variable object for parameter tracking
                                    trainable_variables[var_name].variables = original_val
                                elif update_type == "solution":
                                    trainable_variables[var_name].variables = original_val
                                elif update_type == "env":
                                    await ecp.set_variables(variable_name=var_name, variable_value=original_val)
                                    # Update Variable object for parameter tracking
                                    trainable_variables[var_name].variables = original_val
                                elif update_type == "agent":
                                    await acp.set_variables(variable_name=var_name, variable_value=original_val)
                                    # Update Variable object for parameter tracking
                                    trainable_variables[var_name].variables = original_val
                                elif update_type == "memory":
                                    await memory_manager.set_variables(memory_name=var_name, variable_value=original_val)
                                    # Update Variable object for parameter tracking
                                    trainable_variables[var_name].variables = original_val
                            except Exception as e:
                                logger.warning(f"| ❌ Rollback failed for {var_name}: {e}")

                    # Ensure we have the right number of solutions
                    while len(improved_solutions) < len(candidate_sets):
                        logger.warning(f"| ❌ Error number of parameter texts: {len(improved_solutions)}")
                        baseline_solution = f"Result: {solution_variable.get_value()}"
                        improved_solutions.append(baseline_solution)

                    # Score candidate sets based on their improved solutions
                    rewards = [await self.reward_fn(benchmark_name=self.benchmark_name, prediction=solution, task_id=benchmark_task_id) for solution in improved_solutions]
                    logger.info(f"| ✨ Reward: {rewards}")

                    advantages = self._normalize_rewards(rewards)

                    # Calculate policy ratios and KL based on parameter changes (since reward is based on solution)
                    # Build parameter representations for baseline and each candidate set
                    combined_baseline = {}
                    for vn, bv in before_values.items():
                        combined_baseline[vn] = bv

                    import json
                    baseline_text = json.dumps(combined_baseline, ensure_ascii=False, sort_keys=True)

                    candidate_texts: List[str] = []
                    for improved_set in candidate_sets:
                        combined = combined_baseline.copy()
                        try:
                            for k, v in improved_set.variables.items():
                                entry = v.variables if hasattr(v, 'variables') else v
                                if isinstance(entry, dict):
                                    combined[k] = json.dumps(entry, ensure_ascii=False, sort_keys=True)
                                else:
                                    combined[k] = str(entry)
                        except Exception as e:
                            logger.warning(f"| ❌ Error building combined representation for a candidate set: {e}")
                        candidate_texts.append(json.dumps(combined, ensure_ascii=False, sort_keys=True))

                    # Ensure we have the right number of parameter texts
                    while len(candidate_texts) < len(improved_solutions):
                        logger.warning(f"| ❌ Error number of parameter texts: {len(candidate_texts)}")
                        candidate_texts.append(baseline_text)

                    # Calculate policy ratios and KL
                    # Policy ratios: computed relative to current step baseline (π_new / π_current)
                    # KL divergence: computed relative to initial policy (KL(π_initial || π_new))
                    policy_ratios = []
                    kl_divs = []
                    for params_text in candidate_texts:
                        # Policy ratio relative to current step baseline
                        ratio = self._calculate_text_similarity(baseline_text, params_text)
                        policy_ratios.append(ratio)

                        # KL divergence relative to initial policy using |log(policy_ratio)| surrogate
                        initial_policy_ratio = self._calculate_text_similarity(initial_policy_text, params_text)
                        kl_div = abs(math.log(max(initial_policy_ratio, 1e-8)))
                        kl_divs.append(kl_div)

                    grpo_objectives = self._apply_clipping_and_kl(policy_ratios, advantages, kl_divs)
                    logger.info(f"| ✨ Grpo Objectives: {grpo_objectives}")
                    best_idx = int(grpo_objectives.index(max(grpo_objectives)))
                    best_set = candidate_sets[best_idx]

                    logger.info(f"| ✨ Applying best candidate set (idx={best_idx}) across variables")
                    # Apply all variable updates from best_set atomically
                    for variable_name, improved_entry in best_set.variables.items():
                        try:
                            variable_type = trainable_variables[variable_name].type if variable_name in trainable_variables else ""
                            candidate_val = improved_entry.variables if hasattr(improved_entry, 'variables') else improved_entry
                            if isinstance(candidate_val, dict):
                                applied_value = json.dumps(candidate_val, ensure_ascii=False)
                            else:
                                applied_value = str(candidate_val)

                            if variable_type == "system_prompt" or variable_type == "agent_message_prompt":
                                await prompt_manager.set_variables(
                                    prompt_name=variable_name,
                                    variable_updates={"variables": applied_value}
                                )
                            elif variable_type == "tool_code":
                                await tcp.set_variable(variable_name=variable_name, variable_value=applied_value)
                            elif variable_type == "solution":
                                trainable_variables[variable_name].variables = applied_value
                            elif variable_type == "environment_code":
                                await ecp.set_variables(variable_name=variable_name, variable_value=applied_value)
                            elif variable_type == "agent_code":
                                await acp.set_variables(variable_name=variable_name, variable_value=applied_value)
                            elif variable_type == "memory_code":
                                await memory_manager.set_variables(variable_name=variable_name, variable_value=applied_value)
                            else:
                                # unknown type, skip
                                continue
                        except Exception as e:
                            logger.warning(f"| ❌ Applying candidate for {variable_name} failed: {e}")

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

                        # Record variable changes (before and after) using captured before_values
                        for var_name, before_raw in before_values.items():
                            before_var = trainable_variables.get(var_name)
                            before_value = ""
                            if before_raw is not None:
                                before_value = str(before_raw)

                            after_value = ""
                            if before_var:
                                try:
                                    # Prefer get_value if available
                                    after_raw = before_var.get_value() if hasattr(before_var, 'get_value') else getattr(before_var, 'variables', before_var)
                                except Exception:
                                    after_raw = getattr(before_var, 'variables', before_var)
                                if isinstance(after_raw, dict):
                                    after_value = json.dumps(after_raw, indent=2, ensure_ascii=False)
                                else:
                                    after_value = str(after_raw)

                            var_type = before_var.type if before_var and hasattr(before_var, 'type') else ""

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
        logger.info(f"| {'=' * 60}")
        logger.info(f"| Reflection optimization completed!")
        logger.info(f"| {'=' * 60}")


from abc import ABC, abstractmethod
from typing import List, Union, Dict, Tuple, Optional, Callable
from collections import defaultdict, deque
import random
import math
import tiktoken
from src.optimizers.textgrad.variable import Variable
from src.optimizers.textgrad import logger
from src.optimizers.textgrad.engine import EngineLM, get_engine
from src.optimizers.textgrad.config import validate_engine_or_get_default
from src.optimizers.textgrad.optimizer import Optimizer
from src.optimizers.textgrad.loss import MultiFieldEvaluation

DEFAULT_EVALUATION_SYSTEM_PROMPT = (
    "You are an expert mathematics competition grader. You carefully analyse "
    "candidate solutions, identify issues, and provide actionable feedback that "
    "helps improve the answer while keeping the required output format."
)

DEFAULT_EVALUATION_INSTRUCTION = (
    "Analyse the candidate solution to the contest problem. You will receive:\n"
    "1. The problem statement.\n"
    "2. The candidate's full reasoning enclosed in <think>...</think> plus a final "
    "answer in \\boxed{}.\n\n"
    "Your task:\n"
    "- Determine whether the boxed answer is logically correct based on the reasoning.\n"
    "- Recompute key steps when needed, but never reveal the exact numeric answer even "
    "if you know or infer it.\n"
    "- Point out precise reasoning or computation mistakes (if any).\n"
    "- Provide targeted guidance that helps amend the reasoning while keeping the "
    "<think> format intact.\n\n"
    "Respond using the following template:\n"
    "<VERDICT>correct|incorrect</VERDICT>\n"
    "<EXPLANATION>your detailed critique</EXPLANATION>\n"
    "<GUIDANCE>step-by-step instructions to improve the solution</GUIDANCE>"
)

DEFAULT_REFLECTION_SYSTEM_PROMPT = (
    "You are an expert mathematics tutor. Given a problem, a student's solution, "
    "and detailed feedback, you must produce an improved solution.\n\n"
    "Requirements:\n"
    "- Keep all reasoning strictly inside <think>...</think>.\n"
    "- Put ONLY the final numeric answer in a single \\boxed{VALUE} expression on the last line.\n"
    "- Do not explain the meta-instructions.\n"
)


class ReinforcePlusPlusTextualOptimizer(Optimizer):
    """
    REINFORCE++ style optimizer for text optimization.
    
    REINFORCE++ combines REINFORCE with PPO's clipping mechanism but without a critic network.
    Key features:
    1. Reward function: r(x, y) - β * Σ KL(i)  (reward model score minus KL penalty)
    2. Advantage function: A(s, a) = r(x, y) - β * Σ KL(i)  (direct reward minus KL penalty)
    3. Uses PPO clipping to limit policy updates: clip(r_t(θ), 1-ε, 1+ε)
    4. No value function (critic network) needed
    
    Reference: Jian Hu. "REINFORCE++: A Simple and Efficient Approach for Aligning Large Language Models."
    """
    
    def __init__(self, 
                 parameters: List[Variable],
                 initial_answer: str,
                 question_context: str,
                 engine: Union[EngineLM, str] = None,
                 reward_function = None,
                 clip_ratio: float = 0.2,
                 beta: float = 0.01,  # KL penalty coefficient
                 learning_rate: float = 0.1,
                 max_kl: float = 0.01,
                 verbose: int = 0,
                 tokenizer_model: str = "gpt-4o",
                 evaluation_model: str = 'gpt-4o',
                 evaluation_system_prompt: str = DEFAULT_EVALUATION_SYSTEM_PROMPT,
                 evaluation_instruction: str = DEFAULT_EVALUATION_INSTRUCTION,
                 reflection_system_prompt: str = DEFAULT_REFLECTION_SYSTEM_PROMPT,
                 num_reflection_steps: int = 3,
                 reflection_runner: Optional[Callable[[str, str], str]] = None):
        """
        Initialize REINFORCE++ Textual Optimizer.
        
        :param parameters: List of variables to optimize
        :param engine: LLM engine for generating candidates
        :param reward_function: Function to evaluate text quality (returns float)
        :param clip_ratio: PPO clipping ratio for policy updates (ε)
        :param beta: KL penalty coefficient (β)
        :param learning_rate: Learning rate for policy updates
        :param max_kl: Maximum KL divergence for early stopping
        :param verbose: Verbosity level
        :param new_variable_tags: Tags for parsing improved variables
        :param optimizer_system_prompt: System prompt for the optimizer
        :param kl_method: Method for KL calculation: "token_freq", "token_prob", "char_freq", "n_gram"
        :param tokenizer_model: Model name for tiktoken tokenizer (e.g., "gpt-4o", "cl100k_base")
        :param use_relative_kl: If True, sum per-token KL; if False, average KL
        """
        super().__init__(parameters)


        new_variable_tags = ["<CANDIDATE_VARIABLE>", "</CANDIDATE_VARIABLE>"]


        optimizer_system_prompt = self._get_default_reinforce_plus_plus_prompt()

        self.initial_answer = initial_answer
        self.engine = validate_engine_or_get_default(engine)
        self.reward_function = reward_function
        self.clip_ratio = clip_ratio
        self.beta = beta
        self.learning_rate = learning_rate
        self.max_kl = max_kl
        self.verbose = verbose
        self.new_variable_tags = new_variable_tags
        self.optimizer_system_prompt = optimizer_system_prompt
        self.evaluation_system_prompt = evaluation_system_prompt
        self.evaluation_instruction = evaluation_instruction
        self.reflection_system_prompt = reflection_system_prompt
        # Optional external reflection runner (e.g., tool-calling agent).
        # Signature: runner(system_prompt, prompt) -> response_text
        self.reflection_runner = reflection_runner

        # Tokenizer for token-level edit distance
        self.tokenizer = tiktoken.encoding_for_model(tokenizer_model)
        self.question_context = Variable(
            question_context,
            requires_grad=False,
            role_description="question or context for evaluation"
        )

        # Auto-build evaluation_module
        self.evaluation_module = self._build_evaluation_module(
            evaluation_model=evaluation_model,
            question_context=self.question_context,
            parameters=parameters,
        )
        if self.verbose:
            logger.info(f"Auto-built evaluation_module for reflection using {evaluation_model}")

        self.num_reflection_steps = num_reflection_steps

        # REINFORCE++ state tracking
        self.policy_history = defaultdict(list)  # Store policy history for each parameter
        self.reward_history = defaultdict(list)  # Store reward history (raw rewards)
        self.kl_divergences = defaultdict(list)  # Store KL surrogates (|log policy_ratio|)
        self.advantages = defaultdict(list)  # Store advantage estimates (reward - KL penalty)
        self.policy_ratios = defaultdict(list)  # Store policy ratios
        self.kl_penalties = defaultdict(list)  # Store KL penalties
        
        logger.info(f"REINFORCE++ Textual Optimizer initialized, β={beta}, ε={clip_ratio}")

    def _build_evaluation_module(self,
                                 evaluation_model: str,
                                 question_context: Union[Variable, str],
                                 parameters: List[Variable]) -> MultiFieldEvaluation:
        """
        Auto-build MultiFieldEvaluation module for reflection.

        :param evaluation_model: Model name for evaluation (e.g., "gpt-4o")
        :param question_context: Question or context variable
        :param parameters: List of parameters being optimized (to infer role descriptions)
        :return: MultiFieldEvaluation module
        """
        # Get evaluation engine from model name
        eval_engine = get_engine(evaluation_model)

        # Create Variable objects
        eval_instruction_var = Variable(
            self.evaluation_instruction,
            requires_grad=False,
            role_description="instructions for evaluating a solution"
        )

        eval_system_prompt_var = Variable(
            self.evaluation_system_prompt,
            requires_grad=False,
            role_description="system prompt for the evaluation LLM"
        )

        # Infer role descriptions from question_context and parameters
        question_role = question_context.get_role_description() or "question or context"

        # Get solution role description from first parameter
        solution_role = parameters[0].get_role_description() or "candidate solution"

        # Build MultiFieldEvaluation
        evaluation_module = MultiFieldEvaluation(
            evaluation_instruction=eval_instruction_var,
            role_descriptions=[
                question_role,
                solution_role,
            ],
            engine=eval_engine,
            system_prompt=eval_system_prompt_var,
        )

        return evaluation_module

    def _get_default_reinforce_plus_plus_prompt(self) -> str:
        """Get default REINFORCE++ system prompt."""
        return f"""You are a REINFORCE++ style text optimizer. Your task is to generate diverse, high-quality candidate solutions.

For each optimization step, you will:
1. Propose one improved candidate solution
2. Focus on quality while exploring alternative phrasing when helpful

Generate each candidate between <CANDIDATE_VARIABLE> and </CANDIDATE_VARIABLE> tags.
Be creative and explore different solution paths."""

########################################################################################################################

    def _generate_new_policy_via_reflection(self, parameter: Variable) -> str:
        """
        Generate a candidate solution through multiple rounds of reflection.
        Each candidate goes through num_reflection_steps rounds of evaluation → reflection → improvement.
        """
        question_text = str(self.question_context)
        current_solution = parameter.value

        # Perform multiple rounds of reflection
        for reflection_step in range(1, self.num_reflection_steps + 1):
            # Step 1: Evaluate current solution
            solution_var = Variable(
                current_solution,
                requires_grad=False,
                role_description=parameter.get_role_description()
            )

            evaluation_output = self.evaluation_module([self.question_context, solution_var])
            evaluation_text = evaluation_output.value

            # Step 2: Build reflection prompt (with different strategies for diversity)
            reflection_prompt = self._build_reflection_prompt(
                question_text=question_text,
                current_solution=current_solution,
                evaluation_text=evaluation_text,
            )

            # Step 3: Generate improved solution
            if self.reflection_runner is not None:
                # Use external runner (e.g., tool-calling agent)
                improved_text = self.reflection_runner(
                    self.reflection_system_prompt,
                    reflection_prompt,
                )
            else:
                # Default: call underlying engine directly
                improved_text = self.engine(
                    reflection_prompt,
                    system_prompt=self.reflection_system_prompt,
                )

            improved_text = str(improved_text).strip()

            # Clean up markdown code blocks if present
            if improved_text.startswith("```"):
                lines = improved_text.split("\n")
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                improved_text = "\n".join(lines).strip()

            if improved_text:
                current_solution = improved_text
            else:
                # If generation failed, keep current solution
                break

        return current_solution if current_solution else None


    async def _generate_new_policy_via_reflection_with_agent(self, parameter: Variable) -> str:
        """
        Generate a candidate solution through multiple rounds of reflection.
        Each candidate goes through num_reflection_steps rounds of evaluation → reflection → improvement.
        """
        question_text = str(self.question_context)
        current_solution = parameter.value

        # Perform multiple rounds of reflection
        for reflection_step in range(1, self.num_reflection_steps + 1):
            # Step 1: Evaluate current solution
            solution_var = Variable(
                current_solution,
                requires_grad=False,
                role_description=parameter.get_role_description()
            )

            evaluation_output = self.evaluation_module([self.question_context, solution_var])
            evaluation_text = evaluation_output.value

            # Step 2: Build reflection prompt (with different strategies for diversity)
            reflection_prompt = self._build_reflection_prompt(
                question_text=question_text,
                current_solution=current_solution,
                evaluation_text=evaluation_text,
            )

            # Step 3: Generate improved solution
            if self.reflection_runner is not None:
                improved_text = self.reflection_runner(
                    self.reflection_system_prompt,
                    reflection_prompt,
                )
            else:
                improved_text = self.engine(
                    reflection_prompt,
                    system_prompt=self.reflection_system_prompt,
                )

            improved_text = str(improved_text).strip()

            # Clean up markdown code blocks if present
            if improved_text.startswith("```"):
                lines = improved_text.split("\n")
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                improved_text = "\n".join(lines).strip()

            if improved_text:
                current_solution = improved_text
            else:
                # If generation failed, keep current solution
                break

        return current_solution if current_solution else None

    def _build_reflection_prompt(self, question_text: str, current_solution: str, evaluation_text: str,) -> str:
        """
        Build a reflection prompt that asks the model to improve the solution based on feedback.
        Uses random strategy selection to encourage diverse approaches.
        """

        return (
            "You are improving a student's solution to a math contest problem based on detailed feedback.\n\n"
            "=== Problem ===\n"
            f"{question_text}\n\n"
            "=== Current Solution ===\n"
            f"{current_solution}\n\n"
            "=== Feedback ===\n"
            f"{evaluation_text}\n\n"
            "Please produce an improved solution that:\n"
            "- Addresses all the feedback points mentioned above.\n"
            "- Keeps all reasoning clear and well-structured.\n"
            "- Ensures the final answer is correct and well-justified.\n\n"
            "Return ONLY the improved solution text."
        )

    def _generate_new_policy(self, parameter: Variable) -> str:
        """Generate a single new policy (candidate text) for a parameter."""
        prompt = self._build_new_policy_prompt(parameter)

        response = self.engine(prompt, system_prompt=self.optimizer_system_prompt)
        new_policy = self._parse_policies_from_response(response)

        return new_policy[0]

    
    def _build_new_policy_prompt(self, parameter: Variable) -> str:
        """Build prompt for generating a new policy candidate."""
        # Add randomness to encourage exploration
        exploration_hints = [
            "Try a completely different approach",
            "Use a more detailed step-by-step method", 
            "Try a more concise explanation",
            "Use a different mathematical notation",
            "Include more intermediate steps",
            "Try a visual or diagrammatic approach",
            "Use a different problem-solving strategy"
        ]
        
        import random
        selected_hints = random.sample(exploration_hints, min(3, len(exploration_hints)))
        
        return f"""Generate 1 improved candidate solution for the following:

Role: {parameter.get_role_description()}
Current value: {parameter.value}

IMPORTANT: The candidate should improve the current value. Consider these exploration strategies:
{chr(10).join(f"- {hint}" for hint in selected_hints)}

Generate one improved approach between {self.new_variable_tags[0]} and {self.new_variable_tags[1]} tags.
Ensure clarity and quality."""

    def _parse_policies_from_response(self, response: str) -> List[str]:
        """Parse policy candidates from LLM response."""
        candidates = []
        start_tag = self.new_variable_tags[0]
        end_tag = self.new_variable_tags[1]
        
        # Split by start tag and process each candidate
        parts = response.split(start_tag)
        for part in parts[1:]:  # Skip first part (before first start tag)
            if end_tag in part:
                candidate = part.split(end_tag)[0].strip()
                if candidate:
                    candidates.append(candidate)
        
        return candidates
    
    def _generate_fallback_policy(self, parameter: Variable) -> str:
        """Generate fallback policy text if main generation fails."""
        base_value = parameter.value
        # Simple variation
        return f"Improved version: {base_value}"

########################################################################################################################

    def _evaluate_policy_reward(self, parameter: Variable, policy_text: str) -> float:
        """Evaluate a single policy text using reward function (raw reward before KL penalty)."""
        if self.reward_function is None:
            # Default evaluation: use a simple heuristic
            return self._default_reward_estimate(parameter, [policy_text])[0]
        
        try:
            # Create temporary variable for evaluation
            temp_var = Variable(
                value=policy_text,
                role_description=parameter.role_description,
                requires_grad=False
            )
            reward = self.reward_function(temp_var)
            return float(reward)
        except Exception as e:
            logger.warning(f"Error evaluating candidate: {e}")
            return 0.0
    
    def _default_reward_estimate(self, parameter: Variable, policies: List[str]) -> List[float]:
        """Default reward estimate when no reward function is provided."""
        # Simple heuristic: longer, more detailed responses get higher rewards
        rewards = []
        for candidate in policies:
            # Basic heuristics for text quality
            length_score = min(len(candidate) / 100, 1.0)  # Normalize by 100 chars
            detail_score = candidate.count('.') + candidate.count('!') + candidate.count('?')
            detail_score = min(detail_score / 5, 1.0)  # Normalize by 5 sentences
            
            reward = (length_score + detail_score) / 2
            rewards.append(reward)
        
        return rewards
    
    def _calculate_kl_div(self, new_value: str) -> float:
        """Calculate KL penalty using policy_ratio as surrogate: β * |log(policy_ratio)|."""
        # safe_ratio = max(policy_ratio, 1e-8)  # Avoid log(0)
        # pseudo_kl = abs(math.log(safe_ratio))
        kl_div = self._calculate_text_similarity(self.initial_answer, new_value)
        return max(kl_div, 1e-8)
    
    def _calculate_advantage(self, parameter: Variable, reward: float, kl_penalty: float) -> float:
        """
        Calculate advantage using REINFORCE++ formula.
        A(s, a) = reward - KL_penalty
        """
        advantage = reward - kl_penalty
        self.advantages[parameter].append(advantage)
        return advantage


########################################################################################################################

    def _estimate_policy_ratio_from_similarity(self, parameter: Variable, old_value: str, new_value: str) -> float:
        """
        Calculate policy ratio π_new(a|s) / π_old(a|s).
        
        Uses token-level edit distance similarity for more accurate policy ratio estimation.
        """
        # Calculate similarity between old and new policies using token-level edit distance
        # This is more accurate than character-level Jaccard as it considers token order
        similarity = self._calculate_text_similarity(old_value, new_value)
        
        # Policy ratio is inversely related to similarity (more different = higher ratio)
        # But we want to avoid extreme ratios
        policy_ratio = 1.0 + (1.0 - similarity) * 0.5  # Scale between 1.0 and 1.5
        
        return policy_ratio
    
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
                if tokens1[i-1] == tokens2[j-1]:
                    # Tokens match, no operation needed
                    dp[i][j] = dp[i-1][j-1]
                else:
                    # Take minimum of three operations
                    dp[i][j] = min(
                        dp[i-1][j] + 1,      # Delete token from tokens1
                        dp[i][j-1] + 1,      # Insert token from tokens2
                        dp[i-1][j-1] + 1     # Substitute token
                    )
        
        return dp[m][n]
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into tokens for edit distance calculation.
        """
        token_ids = self.tokenizer.encode(text)
        # Use token ids as strings for comparison to avoid decoding overhead
        return [str(tid) for tid in token_ids]
    
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

########################################################################################################################

    def _apply_clipping(self, policy_ratio: float) -> float:
        """
        Apply REINFORCE++ clipping to a policy ratio.
        Clip ratio to [1-ε, 1+ε] where ε = clip_ratio
        """
        return max(1 - self.clip_ratio, min(1 + self.clip_ratio, policy_ratio))
    
    def _calculate_reinforce_plus_plus_objective(self, policy_ratio: float, clipped_ratio: float, advantage: float) -> float:
        """
        Calculate REINFORCE++ objective with clipping for a single sample.
        L^CLIP(θ) = min(r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t)
        """
        return min(clipped_ratio * advantage, policy_ratio * advantage)

########################################################################################################################

    def _select_and_apply_policy(self, 
                       parameter: Variable, 
                       new_policy: str,
                       reward: float, 
                       policy_ratio: float,
                       kl_penalty: float, 
                       advantage: float,
                       reward_old: float,
                       advantage_old: float) -> str:
        """
        Update policy by comparing candidate and old policy objectives.
        """
        old_value = parameter.value
        clipped_ratio = self._apply_clipping(policy_ratio)
        objective_candidate = self._calculate_reinforce_plus_plus_objective(policy_ratio, clipped_ratio, advantage)
        objective_old = advantage_old  # old policy ratio = 1, clipped =1
        
        # Choose based on objective
        if objective_candidate >= objective_old:
            chosen_value = new_policy
            chosen_reward = reward
            chosen_policy_ratio = policy_ratio
            chosen_kl_penalty = kl_penalty
            chosen_advantage = advantage
            chosen_clipped_ratio = clipped_ratio
            chosen_objective = objective_candidate
        else:
            chosen_value = old_value
            chosen_reward = reward_old
            chosen_policy_ratio = 1.0
            chosen_kl_penalty = 0.0
            chosen_advantage = advantage_old
            chosen_clipped_ratio = 1.0
            chosen_objective = objective_old
        
        # Use |log(policy_ratio)| as surrogate KL
        kl_div = abs(math.log(max(chosen_policy_ratio, 1e-8)))
        
        # Store metrics
        self.policy_history[parameter].append({
            'old_value': old_value,
            'new_value': chosen_value,
            'reward': chosen_reward,
            'kl_penalty': chosen_kl_penalty,
            'advantage': chosen_advantage,
            'policy_ratio': chosen_policy_ratio,
            'clipped_ratio': chosen_clipped_ratio,
            'reinforce_plus_plus_objective': chosen_objective,
            'kl_divergence': kl_div
        })
        
        self.reward_history[parameter].append(chosen_reward)
        self.kl_penalties[parameter].append(chosen_kl_penalty)
        self.policy_ratios[parameter].append(chosen_policy_ratio)
        self.kl_divergences[parameter].append(kl_div)
        
        # if kl_div > self.max_kl:
        #     logger.info(f"Early stopping due to high KL divergence: {kl_div}")
        #     return old_value
        
        return chosen_value
    
    def step(self):
        """Perform one REINFORCE++ optimization step (synchronous version)."""
        for parameter in self.parameters:
            if self.verbose:
                print(f"\n--- REINFORCE++ Step for {parameter.get_role_description()} ---")
            
            new_policy = self._generate_new_policy_via_reflection(parameter)
            reward = self._evaluate_policy_reward(parameter, new_policy)
            policy_ratio = self._estimate_policy_ratio_from_similarity(parameter, parameter.value, new_policy)
            kl_div = self._calculate_kl_div(new_policy)
            kl_penalty = self.beta*kl_div
            raw_advantage = self._calculate_advantage(parameter, reward, kl_penalty)

            # Evaluate current (old) policy for comparison
            reward_old = self._evaluate_policy_reward(parameter, parameter.value)
            raw_advantage_old = self._calculate_advantage(parameter, reward_old, 0.0)  # no KL penalty for old

            new_value = self._select_and_apply_policy(
                parameter=parameter,
                new_policy=new_policy,
                reward=reward,
                policy_ratio=policy_ratio,
                kl_penalty=kl_penalty,
                advantage=raw_advantage,
                reward_old=reward_old,
                advantage_old=raw_advantage_old
            )
            
            parameter.set_value(new_value)
            
            if self.verbose:
                print(f"Updated to: {new_value[:100]}...")
                print(f"Reward(chosen): {self.reward_history[parameter][-1]:.3f}, KL penalty: {self.kl_penalties[parameter][-1]:.3f}, Advantage: {self.policy_history[parameter][-1]['advantage']:.3f}")
    
    async def astep(self):
        """Perform one REINFORCE++ optimization step (async version for agent use)."""
        for parameter in self.parameters:
            if self.verbose:
                print(f"\n--- REINFORCE++ Step for {parameter.get_role_description()} ---")
            
            new_policy = await self._generate_new_policy_via_reflection_with_agent(parameter)
            reward = self._evaluate_policy_reward(parameter, new_policy)
            policy_ratio = self._estimate_policy_ratio_from_similarity(parameter, parameter.value, new_policy)
            kl_div = self._calculate_kl_div(new_policy)
            kl_penalty = self.beta*kl_div
            raw_advantage = self._calculate_advantage(parameter, reward, kl_penalty)

            # Evaluate current (old) policy for comparison
            reward_old = self._evaluate_policy_reward(parameter, parameter.value)
            raw_advantage_old = self._calculate_advantage(parameter, reward_old, 0.0)  # no KL penalty for old

            new_value = self._select_and_apply_policy(
                parameter=parameter,
                new_policy=new_policy,
                reward=reward,
                policy_ratio=policy_ratio,
                kl_penalty=kl_penalty,
                advantage=raw_advantage,
                reward_old=reward_old,
                advantage_old=raw_advantage_old
            )
            
            parameter.set_value(new_value)
            
            if self.verbose:
                print(f"Updated to: {new_value[:100]}...")
                print(f"Reward(chosen): {self.reward_history[parameter][-1]:.3f}, KL penalty: {self.kl_penalties[parameter][-1]:.3f}, Advantage: {self.policy_history[parameter][-1]['advantage']:.3f}")
    
    def get_statistics(self) -> Dict:
        """Get optimization statistics including REINFORCE++ metrics."""
        stats = {}
        for param in self.parameters:
            if param in self.reward_history:
                rewards = self.reward_history[param]
                advantages = self.advantages[param]
                policy_ratios = self.policy_ratios[param]
                kl_penalties = self.kl_penalties[param]
                
                stats[param.get_role_description()] = {
                    'avg_reward': sum(rewards) / len(rewards) if rewards else 0,
                    'max_reward': max(rewards) if rewards else 0,
                    'num_steps': len(self.policy_history[param]),
                    'avg_kl_divergence': sum(self.kl_divergences[param]) / len(self.kl_divergences[param]) if self.kl_divergences[param] else 0,
                    'avg_kl_penalty': sum(kl_penalties) / len(kl_penalties) if kl_penalties else 0,
                    'avg_advantage': sum(advantages) / len(advantages) if advantages else 0,
                    'avg_policy_ratio': sum(policy_ratios) / len(policy_ratios) if policy_ratios else 0,
                    'exploration_ratio': len(set(rewards)) / len(rewards) if rewards else 0,  # Diversity measure
                    'beta': self.beta,
                    'clip_ratio': self.clip_ratio
                }
        return stats


# Convenience function
def ReinforcePlusPlus(parameters: List[Variable], **kwargs) -> ReinforcePlusPlusTextualOptimizer:
    """Convenience function to create REINFORCE++ optimizer."""
    return ReinforcePlusPlusTextualOptimizer(parameters, **kwargs)


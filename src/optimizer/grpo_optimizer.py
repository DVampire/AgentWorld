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


class GRPOTextualOptimizer(Optimizer):
    """
    GRPO (Group Relative Policy Optimization) text optimizer.

    GRPO is a simplified variant of PPO with:
    1. Multiple candidate actions per step
    2. Reward-based evaluation for candidates
    3. Group-relative ranking as advantage (no value function)
    4. Policy updates based on relative advantage
    5. Clipping to prevent large updates

    Simplifications compared to PPO:
    - No value function estimation
    - Uses group-normalized rewards as advantages
    - Simpler update mechanism
    """

    def __init__(self,
                 parameters: List[Variable],
                 initial_answer: str,
                 question_context: str,
                 engine: Union[EngineLM, str] = None,
                 reward_function=None,
                 num_candidates: int = 4,
                 clip_ratio: float = 0.2,
                 beta: float = 0.01,  # KL penalty coefficient
                 learning_rate: float = 0.1,
                 entropy_coef: float = 0.01,
                 max_kl: float = 0.01,
                 verbose: int = 0,
                 tokenizer_model: str = 'gpt-4o',
                 evaluation_model: str = 'gpt-4o',
                 evaluation_system_prompt: str = DEFAULT_EVALUATION_SYSTEM_PROMPT,
                 evaluation_instruction: str = DEFAULT_EVALUATION_INSTRUCTION,
                 reflection_system_prompt: str = DEFAULT_REFLECTION_SYSTEM_PROMPT,
                 num_reflection_steps: int = 3,
                 reflection_runner: Optional[Callable[[str, str], str]] = None):
        """
        Initialize GRPO textual optimizer.

        :param parameters: Variables to optimize
        :param engine: LLM engine for generating candidates
        :param reward_function: Reward function to score candidates (float)
        :param num_candidates: Number of candidates per step
        :param clip_ratio: Clipping ratio for policy updates
        :param learning_rate: Learning rate for updates
        :param entropy_coef: Entropy coefficient for exploration
        :param max_kl: Max KL for early stopping
        :param verbose: Verbosity level
        :param new_variable_tags: Tags to parse improved variables
        :param evaluation_model: Model name for evaluation (e.g., "gpt-4o"). If provided with question_context, will auto-build evaluation_module for reflection
        :param question_context: Question or context variable for evaluation (optional). If provided with evaluation_model, enables reflection-based candidate generation
        :param num_reflection_steps: Number of reflection steps per candidate (default: 3)
        """
        super().__init__(parameters)

        # If no tags specified, use defaults
        new_variable_tags = ["<CANDIDATE_VARIABLE>", "</CANDIDATE_VARIABLE>"]

        # If no system prompt specified, use default
        optimizer_system_prompt = self._get_default_grpo_prompt(num_candidates)

        # Init config
        self.initial_answer = initial_answer
        self.engine = validate_engine_or_get_default(engine)
        self.reward_function = reward_function
        self.num_candidates = num_candidates
        self.clip_ratio = clip_ratio
        self.beta = beta
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
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

        # GRPO state tracking
        self.policy_history = defaultdict(list)
        self.reward_history = defaultdict(list)
        self.kl_divergences = defaultdict(list)
        self.normalized_rewards = defaultdict(list)
        self.policy_ratios = defaultdict(list)

        logger.info(f"GRPO optimizer initialized with num_candidates={num_candidates}")

########################################################################################################################
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

########################################################################################################################

    def _get_default_grpo_prompt(self, num_candidates: int) -> str:
        """Get default GRPO system prompt (English)."""
        return f"""You are a GRPO-style text optimizer. Your task is to generate one improved candidate solution per call.

For each optimization step:
1. Propose 1 improved candidate solution
2. Focus on quality while exploring alternative approaches when helpful
3. Consider different strategies, styles, or methodologies

Generate the candidate between <CANDIDATE_VARIABLE> and </CANDIDATE_VARIABLE> tags.
Be creative and explore different solution paths."""

    def _generate_candidates(self, parameter: Variable) -> List[str]:
        """Generate multiple candidate solutions by making multiple separate LLM calls."""
        candidates = []
        
        for i in range(self.num_candidates):
            try:
                # Generate one candidate per call
                candidate = self._generate_single_candidate(parameter, candidate_idx=i)
                if candidate:
                    candidates.append(candidate)
            except Exception as e:
                logger.warning(f"Error generating candidate {i+1}: {e}")
                # Continue to next candidate instead of failing completely
                continue
        
        # If we didn't get enough candidates, use fallback for the rest
        if len(candidates) < self.num_candidates:
            fallback_count = self.num_candidates - len(candidates)
            fallback_candidates = self._generate_fallback_candidates(parameter, count=fallback_count)
            candidates.extend(fallback_candidates)
        
        return candidates[:self.num_candidates]
    
    def _generate_single_candidate(self, parameter: Variable, candidate_idx: int = 0) -> str:
        """Generate a single candidate solution using reflection process."""
        # If evaluation_module and question_context are provided, use reflection
        if self.evaluation_module is not None and self.question_context is not None:
            return self._generate_candidate_via_reflection(parameter, candidate_idx)
        else:
            # Fallback to direct generation
            return self._generate_candidate_direct(parameter, candidate_idx)
    
    def _generate_candidate_via_reflection(self, parameter: Variable, candidate_idx: int = 0) -> str:
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
                candidate_idx=candidate_idx
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

    def _generate_candidate_direct(self, parameter: Variable, candidate_idx: int = 0) -> str:
        """Generate a candidate solution directly without reflection."""
        prompt = self._build_single_candidate_prompt(parameter, candidate_idx)
        
        try:
            response = self.engine(prompt, system_prompt=self.optimizer_system_prompt)
            candidate = self._parse_single_candidate(response)
            
            if not candidate:
                # If parsing failed, try to extract from response
                start_tag = self.new_variable_tags[0]
                end_tag = self.new_variable_tags[1]
                if start_tag in response and end_tag in response:
                    candidate = response.split(start_tag)[1].split(end_tag)[0].strip()
                else:
                    # If no tags, use the whole response as candidate
                    candidate = response.strip()
            
            return candidate if candidate else None
            
        except Exception as e:
            logger.error(f"Error generating single candidate: {e}")
            return None
    
    def _build_reflection_prompt(self, question_text: str, current_solution: str, evaluation_text: str,
                                 candidate_idx: int) -> str:
        """
        Build a reflection prompt that asks the model to improve the solution based on feedback.
        Uses random strategy selection to encourage diverse approaches.
        """
        # Different reflection strategies for diversity
        reflection_strategies = [
            "Focus on correcting any logical errors and improving the reasoning flow.",
            "Emphasize clarity and step-by-step explanation, making each step explicit.",
            "Pay special attention to mathematical rigor and completeness of the solution.",
            "Consider alternative approaches or methods that might be more elegant.",
            "Prioritize fixing computational mistakes and verifying numerical accuracy.",
            "Strengthen the logical connections between steps and improve overall coherence.",
        ]
        
        # Randomly select strategy to encourage diversity
        selected_strategy = reflection_strategies[candidate_idx % len(reflection_strategies)]

        return (
            "You are improving a student's solution to a math contest problem based on detailed feedback.\n\n"
            "=== Problem ===\n"
            f"{question_text}\n\n"
            "=== Current Solution ===\n"
            f"{current_solution}\n\n"
            "=== Feedback ===\n"
            f"{evaluation_text}\n\n"
            f"=== Instructions ===\n"
            f"{selected_strategy}\n\n"
            "Please produce an improved solution that:\n"
            "- Addresses all the feedback points mentioned above.\n"
            "- Keeps all reasoning clear and well-structured.\n"
            "- Ensures the final answer is correct and well-justified.\n\n"
            "Return ONLY the improved solution text."
        )

    def _build_single_candidate_prompt(self, parameter: Variable, candidate_idx: int = 0) -> str:
        """Build prompt for generating a single candidate solution."""
        # Different exploration hints for diversity across multiple calls
        exploration_hints = [
            "Try a completely different approach",
            "Use a more detailed step-by-step method",
            "Try a more concise explanation",
            "Use a different mathematical notation",
            "Include more intermediate steps",
            "Try a visual or diagrammatic approach",
            "Use a different problem-solving strategy",
            "Focus on a different aspect of the problem",
            "Use alternative reasoning paths"
        ]

        import random
        # Select a hint based on candidate_idx to ensure diversity across calls
        # Use modulo to cycle through hints if we have more candidates than hints
        selected_hint = exploration_hints[candidate_idx % len(exploration_hints)]
        
        # Add some randomness by occasionally picking a random hint
        if random.random() < 0.3:  # 30% chance to use random hint
            selected_hint = random.choice(exploration_hints)

        return f"""Generate 1 improved candidate solution for the following:

Role: {parameter.get_role_description()}
Current value: {parameter.value}

IMPORTANT: The candidate should improve the current value. Consider this exploration strategy:
- {selected_hint}

Generate one improved approach between {self.new_variable_tags[0]} and {self.new_variable_tags[1]} tags.
Ensure clarity and quality."""

    def _parse_single_candidate(self, response: str) -> str:
        """Parse a single candidate from LLM response."""
        start_tag = self.new_variable_tags[0]
        end_tag = self.new_variable_tags[1]

        # Look for the first candidate between tags
        if start_tag in response and end_tag in response:
            candidate = response.split(start_tag)[1].split(end_tag)[0].strip()
            if candidate:
                return candidate
        
        return None
    
    def _parse_candidates(self, response: str) -> List[str]:
        """Parse multiple candidates from LLM response (for backward compatibility)."""
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

    def _generate_fallback_candidates(self, parameter: Variable, count: int = None) -> List[str]:
        """Generate fallback candidates if main generation fails."""
        if count is None:
            count = self.num_candidates
        
        base_value = parameter.value
        candidates = []

        # Simple variations
        variations = [
            f"Revised version: {base_value}",
            f"Alternative approach: {base_value}",
            f"Improved version: {base_value}",
            f"Different strategy: {base_value}",
            f"Refined solution: {base_value}",
            f"Enhanced approach: {base_value}"
        ]

        candidates.extend(variations[:count])
        return candidates[:count]

    def _evaluate_candidates(self, parameter: Variable, candidates: List[str]) -> List[float]:
        """Evaluate candidates using reward function."""
        if self.reward_function is None:
            # Default evaluation: simple heuristic
            return self._default_evaluation(parameter, candidates)

        rewards = []
        for candidate in candidates:
            try:
                # Create temporary variable for evaluation
                temp_var = Variable(
                    value=candidate,
                    role_description=parameter.role_description,
                    requires_grad=False
                )
                reward = self.reward_function(temp_var)
                rewards.append(float(reward))
            except Exception as e:
                logger.warning(f"Error evaluating candidate: {e}")
                rewards.append(0.0)

        return rewards

    def _default_evaluation(self, parameter: Variable, candidates: List[str]) -> List[float]:
        """Default evaluation when no reward function is provided."""
        # Simple heuristic: longer and more detailed responses get higher rewards
        rewards = []
        for candidate in candidates:
            # Basic heuristics for text quality
            length_score = min(len(candidate) / 100, 1.0)  # Normalize by 100 chars
            detail_score = candidate.count('.') + candidate.count('!') + candidate.count('?')
            detail_score = min(detail_score / 5, 1.0)  # Normalize by 5 sentences

            reward = (length_score + detail_score) / 2
            rewards.append(reward)

        return rewards

    def _normalize_rewards(self, rewards: List[float]) -> List[float]:
        """
        Normalize rewards as advantages (GRPO key step).
        Group normalization: (r - mean(r)) / std(r)
        """
        if len(rewards) <= 1:
            return [0.0] * len(rewards)

        mean_reward = sum(rewards) / len(rewards)
        variance = sum((r - mean_reward) ** 2 for r in rewards) / len(rewards)
        std_reward = math.sqrt(variance) if variance > 1e-8 else 1.0

        # Normalize rewards
        normalized = [(r - mean_reward) / std_reward for r in rewards]
        return normalized

    def _calculate_policy_ratio(self, parameter: Variable, old_value: str, new_value: str) -> float:
        """
        Calculate policy ratio π_new(a|s) / π_old(a|s).
        In practice this would use real policy probabilities.
        """
        # Simplified ratio based on text similarity/quality

        # Similarity between old and new policies
        similarity = self._calculate_text_similarity(old_value, new_value)

        # Ratio inversely related to similarity (more different -> higher ratio), capped
        policy_ratio = 1.0 + (1.0 - similarity) * 0.5  # scale 1.0~1.5

        return policy_ratio

    def _calculate_kl_div(self, new_value: str) -> float:
        """Calculate KL penalty using policy_ratio as surrogate: β * |log(policy_ratio)|."""
        # safe_ratio = max(policy_ratio, 1e-8)  # Avoid log(0)
        # pseudo_kl = abs(math.log(safe_ratio))
        kl_div = self._calculate_text_similarity(self.initial_answer, new_value)
        return max(kl_div, 1e-8)

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

    def _apply_grpo_clipping_and_kl(self, parameter: Variable, policy_ratios: List[float], advantages: List[float], kl_divs) -> List[float]:
        """
        Apply GRPO clipping to policy ratios (same as PPO clipping).
        """
        clipped_objectives = []

        for ratio, advantage, kl_div in zip(policy_ratios, advantages, kl_divs):
            # unclipped objective
            unclipped_objective = ratio * advantage

            # clipped ratio
            if advantage >= 0:
                clipped_ratio = min(ratio, 1 + self.clip_ratio)
            else:
                clipped_ratio = max(ratio, 1 - self.clip_ratio)

            clipped_objective = clipped_ratio * advantage

            # conservative update
            final_objective = min(unclipped_objective, clipped_objective)
            clipped_objectives.append(final_objective - self.beta * kl_div)

        return clipped_objectives

    def _update_policy(self, parameter: Variable, candidates: List[str], rewards: List[float]) -> str:
        """Update policy based on GRPO algorithm."""
        old_value = parameter.value

        # Step 1: normalize rewards as advantages
        advantages = self._normalize_rewards(rewards)

        # Step 2: compute policy ratios and KL
        policy_ratios = []
        kl_divs = []
        for candidate in candidates:
            ratio = self._calculate_policy_ratio(parameter, old_value, candidate)
            policy_ratios.append(ratio)
            kl_div = self._calculate_kl_div(candidate)
            kl_divs.append(kl_div)

        # Step 3: apply GRPO clipping
        grpo_objectives = self._apply_grpo_clipping_and_kl(parameter, policy_ratios, advantages, kl_divs)

        # Step 4: pick best by GRPO objective
        best_idx = grpo_objectives.index(max(grpo_objectives))
        best_candidate = candidates[best_idx]



        # Step 6: store metrics
        self.policy_history[parameter].append({
            'old_value': old_value,
            'new_value': best_candidate,
            'rewards': rewards,
            'advantages': advantages,
            'policy_ratios': policy_ratios,
            'grpo_objectives': grpo_objectives,
            'kl_divergence': kl_div,
            'best_reward': rewards[best_idx],
            'best_advantage': advantages[best_idx]
        })

        self.reward_history[parameter].extend(rewards)
        self.normalized_rewards[parameter].extend(advantages)
        self.policy_ratios[parameter].extend(policy_ratios)
        self.kl_divergences[parameter].append(kl_div)

        return best_candidate

    def step(self):
        """Perform one GRPO optimization step."""
        for parameter in self.parameters:
            if self.verbose:
                print(f"\n--- GRPO step for {parameter.get_role_description()} ---")
                print(f"Current value: {parameter.value[:200]}...")

            # Generate candidates
            candidates = self._generate_candidates(parameter)
            if self.verbose:
                print(f"\n✅ Generated {len(candidates)} candidates")
                for i, cand in enumerate(candidates):
                    print(f"  Candidate {i + 1} (len {len(cand)}): {cand[:150]}...")

            # Evaluate candidates
            rewards = self._evaluate_candidates(parameter, candidates)
            if self.verbose:
                print(f"\n📊 Rewards: {[f'{r:.3f}' for r in rewards]}")
                print(f"   Best reward: {max(rewards):.3f}, Mean reward: {sum(rewards) / len(rewards):.3f}")

            # Update policy
            old_value = parameter.value
            new_value = self._update_policy(parameter, candidates, rewards)

            # Check if updated
            value_changed = (new_value != old_value)

            # Set parameter value
            parameter.set_value(new_value)

            if self.verbose:
                if value_changed:
                    print(f"\n✅ Parameter updated")
                    print(f"   New value: {new_value[:150]}...")
                else:
                    print(f"\n⚠️ Parameter unchanged (kept original or identical candidate)")
                print(f"   Best reward: {max(rewards):.3f}")

    def get_statistics(self) -> Dict:
        """Get optimization statistics including GRPO metrics."""
        stats = {}
        for param in self.parameters:
            if param in self.reward_history:
                rewards = self.reward_history[param]
                advantages = self.normalized_rewards[param]
                policy_ratios = self.policy_ratios[param]

                stats[param.get_role_description()] = {
                    'avg_reward': sum(rewards) / len(rewards) if rewards else 0,
                    'max_reward': max(rewards) if rewards else 0,
                    'min_reward': min(rewards) if rewards else 0,
                    'num_steps': len(self.policy_history[param]),
                    'avg_kl_divergence': sum(self.kl_divergences[param]) / len(self.kl_divergences[param]) if
                    self.kl_divergences[param] else 0,
                    'avg_advantage': sum(advantages) / len(advantages) if advantages else 0,
                    'avg_policy_ratio': sum(policy_ratios) / len(policy_ratios) if policy_ratios else 0,
                    'exploration_ratio': len(set(rewards)) / len(rewards) if rewards else 0,  # diversity metric
                    'reward_std': math.sqrt(
                        sum((r - sum(rewards) / len(rewards)) ** 2 for r in rewards) / len(rewards)) if len(
                        rewards) > 1 else 0
                }
        return stats


# Convenience function
def GRPO(parameters: List[Variable], **kwargs) -> GRPOTextualOptimizer:
    """Convenience factory for GRPO optimizer."""
    return GRPOTextualOptimizer(parameters, **kwargs)

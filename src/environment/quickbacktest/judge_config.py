
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class StrategyJudgeConfig:
    min_relative_improvement: float = 0.05
    plateau_rounds: int = 5
    success_plateau_rounds: int = 3
    eps: float = 1e-8


@dataclass
class StrategyJudgeState:
    best_excess_return: Optional[float] = None
    best_information_ratio: Optional[float] = None
    best_risk_to_reward_ratio: Optional[float] = None

    best_signal_combinations: List[str] = field(default_factory=list)
    best_strategy_name: Optional[str] = None

    plateau_count: int = 0
    success_plateau_count: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)


class StrategyJudge:
    """
    Expected input:
    {
        "original_hypothesis": "...",
        "hypothesis_true": true,
        "recommended_hypothesis": "...",
        "best_signal_combinations": ["signal_1+signal_3"],
        "best_strategy_name": "...",
        "metrics": {
            "excess_return": 0.061,
            "information_ratio": 0.836,
            "risk_to_reward_ratio": 0.62
        },
        "hypothesis_evidence": {
            "summary": "..."
        }
    }
    """

    def __init__(self, config: Optional[StrategyJudgeConfig] = None) -> None:
        self.config = config or StrategyJudgeConfig()
        self.state = StrategyJudgeState()

    def step(self, evaluation: Dict[str, Any]) -> Dict[str, Any]:
        metrics = evaluation["metrics"]

        curr_excess_return = float(metrics["excess_return"])
        curr_information_ratio = float(metrics["information_ratio"])
        curr_risk_to_reward_ratio = float(metrics["risk_to_reward_ratio"])

        hypothesis_true = bool(evaluation["hypothesis_true"])
        recommended_hypothesis = str(evaluation["recommended_hypothesis"])
        best_signal_combinations = evaluation["best_signal_combinations"]
        best_strategy_name = str(evaluation["best_strategy_name"])
        evidence_summary = str(evaluation["hypothesis_evidence"]["summary"])

        # ===== first round =====
        if self.state.best_excess_return is None:
            self.state.best_excess_return = curr_excess_return
            self.state.best_information_ratio = curr_information_ratio
            self.state.best_risk_to_reward_ratio = curr_risk_to_reward_ratio
            self.state.best_signal_combinations = list(best_signal_combinations)
            self.state.best_strategy_name = best_strategy_name
            self.state.history.append(evaluation)

            return {
                "decision": "continue",
                "reason": "first_round",
                "next_step_path": {
                    "next_hypothesis": recommended_hypothesis,
                    "best_signal_combinations": self.state.best_signal_combinations,
                    "best_strategy_name": self.state.best_strategy_name,
                    "path_reason": evidence_summary,
                },
                # "extra": {
                #     "current_excess_return": curr_excess_return,
                #     "current_information_ratio": curr_information_ratio,
                #     "current_risk_to_reward_ratio": curr_risk_to_reward_ratio,
                #     "best_excess_return": self.state.best_excess_return,
                #     "best_information_ratio": self.state.best_information_ratio,
                #     "best_risk_to_reward_ratio": self.state.best_risk_to_reward_ratio,
                #     "relative_excess_return_improvement": 0.0,
                #     "relative_information_ratio_improvement": 0.0,
                #     "relative_risk_to_reward_improvement": 0.0,
                #     "plateau_count": 0,
                #     "success_plateau_count": 0,
                # },
            }

        # ===== relative improvements =====
        excess_return_improve = self._relative_improvement(
            curr=curr_excess_return,
            best=self.state.best_excess_return,
            direction="higher_better",
        )

        information_ratio_improve = self._relative_improvement(
            curr=curr_information_ratio,
            best=self.state.best_information_ratio,
            direction="higher_better",
        )

        risk_to_reward_improve = self._relative_improvement(
            curr=curr_risk_to_reward_ratio,
            best=self.state.best_risk_to_reward_ratio,
            direction="lower_better",
        )

        excess_return_good = excess_return_improve >= self.config.min_relative_improvement
        information_ratio_good = information_ratio_improve >= self.config.min_relative_improvement
        risk_to_reward_good = risk_to_reward_improve >= self.config.min_relative_improvement

        any_good = excess_return_good or information_ratio_good or risk_to_reward_good

        # ===== update best metrics =====
        if excess_return_good:
            self.state.best_excess_return = curr_excess_return

        if information_ratio_good:
            self.state.best_information_ratio = curr_information_ratio

        if risk_to_reward_good:
            self.state.best_risk_to_reward_ratio = curr_risk_to_reward_ratio

        # ===== update best path if any metric improved =====
        if any_good:
            self.state.best_signal_combinations = list(best_signal_combinations)
            self.state.best_strategy_name = best_strategy_name

        # ===== plateau logic =====
        if any_good:
            self.state.plateau_count = 0
        else:
            self.state.plateau_count += 1

        # ===== success plateau logic =====
        if hypothesis_true and (not excess_return_good) and (not information_ratio_good) and (not risk_to_reward_good):
            self.state.success_plateau_count += 1
        else:
            self.state.success_plateau_count = 0

        self.state.history.append(evaluation)

        next_step_path = {
            "next_hypothesis": recommended_hypothesis,
            "best_signal_combinations": self.state.best_signal_combinations,
            "best_strategy_name": self.state.best_strategy_name,
            "path_reason": evidence_summary,
        }

        extra = {
            "current_excess_return": curr_excess_return,
            "current_information_ratio": curr_information_ratio,
            "current_risk_to_reward_ratio": curr_risk_to_reward_ratio,
            "best_excess_return": self.state.best_excess_return,
            "best_information_ratio": self.state.best_information_ratio,
            "best_risk_to_reward_ratio": self.state.best_risk_to_reward_ratio,
            "relative_excess_return_improvement": excess_return_improve,
            "relative_information_ratio_improvement": information_ratio_improve,
            "relative_risk_to_reward_improvement": risk_to_reward_improve,
            "plateau_count": self.state.plateau_count,
            "success_plateau_count": self.state.success_plateau_count,
        }

        if self.state.plateau_count >= self.config.plateau_rounds:
            return {
                "decision": "end",
                "reason": f"strategy_plateau_{self.config.plateau_rounds}_rounds",
                "next_step_path": next_step_path,
                "extra": extra,
            }

        if self.state.success_plateau_count >= self.config.success_plateau_rounds:
            return {
                "decision": "end",
                "reason": "hypothesis_true_and_strategy_plateau",
                "next_step_path": next_step_path,
                "extra": extra,
            }

        return {
            "decision": "continue",
            "reason": "still_improving_or_not_yet_plateau",
            "next_step_path": next_step_path,
            # "extra": extra,
        }

    def _relative_improvement(self, curr: float, best: float, direction: str) -> float:
        if direction == "higher_better":
            return (curr - best) / max(abs(best), self.config.eps)
        if direction == "lower_better":
            return (best - curr) / max(abs(best), self.config.eps)
        raise ValueError(f"Unknown direction: {direction}")

@dataclass
class JudgeConfig:
    """
    Judge stopping rules.

    Args:
        min_relative_improvement:
            Minimum relative improvement required to count as meaningful progress.
            Example: 0.05 means 5%.
        plateau_rounds:
            End if combo rank_ic fails to improve meaningfully for this many consecutive rounds.
        success_plateau_rounds:
            End earlier if all hypotheses are true and combo rank_ic is already on plateau
            for this many consecutive rounds.
        eps:
            Small constant to avoid division by zero.
    """
    min_relative_improvement: float = 0.05
    plateau_rounds: int = 5
    success_plateau_rounds: int = 3
    rank_ic_early_stop_threshold: float = 0.05
    eps: float = 1e-8


@dataclass
class JudgeState:
    """
    Persistent state across iterations.
    """
    best_combo_rank_ic: Optional[float] = None
    plateau_count: int = 0
    success_plateau_count: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)
    best_signal_combination = None
    best_combo_hypothesis:str = None


class SignalJudge:
    """
    Rule-based judge for signal iteration.

    Input JSON format:
    {
      "signals": [
        {
          "name": "...",
          "original_hypothesis": "...",
          "hypothesis_true": true/false,
          "recommended_hypothesis": "...",
          "rank_ic": 0.024
        },
        ...
      ],
      "combo": {
        "original_hypothesis": "...",
        "hypothesis_true": true/false,
        "recommended_hypothesis": "...",
        "rank_ic": 0.019
      }
    }

    Output:
    {
      "decision": "continue" | "end",
      "reason": "...",
      "combo_rank_ic": float,
      "best_combo_rank_ic": float,
      "relative_improvement_vs_best": float,
      "all_hypothesis_true": bool,
      "plateau_count": int,
      "success_plateau_count": int,
      "suggested_focus": "combo" | "prune_weak_signal" | "refine_combo",
      "weak_signals": [...]
    }
    """

    def __init__(self, config: Optional[JudgeConfig] = None) -> None:
        self.config = config or JudgeConfig()
        self.state = JudgeState()

    def step(self, result_json: Dict[str, Any]) -> Dict[str, Any]:
        combo = result_json["combo"]
        signals = result_json["signals"]

        combo_rank_ic = float(combo["rank_ic"])
        combo_true = bool(combo["hypothesis_true"])
        all_signal_true = all(bool(s["hypothesis_true"]) for s in signals)
        all_hypothesis_true = all_signal_true and combo_true
        combo_hypothesis = combo["original_hypothesis"]


        # weak_signals = [
        #     s["name"]
        #     for s in signals
        #     if (not bool(s["hypothesis_true"])) or float(s["rank_ic"]) <= 0
        # ]

        recommended_hypotheses = {
            s["name"]: s["recommended_hypothesis"]
            for s in signals
            if not bool(s["hypothesis_true"])
        }

        # First round initialization
        if self.state.best_combo_rank_ic is None:
            self.state.best_combo_rank_ic = combo_rank_ic
            self.state.best_signal_combination = [s["name"] for s in signals]
            self.state.history.append(result_json)
            self.state.best_combo_hypothesis = combo_hypothesis
            return {
                "decision": "continue",
                "reason": "first_round",
                # "combo_rank_ic": combo_rank_ic,
                # "best_combo_rank_ic": self.state.best_combo_rank_ic,
                # "relative_improvement_vs_best": 0.0,
                "all_hypothesis_true": all_hypothesis_true,
                # "plateau_count": self.state.plateau_count,
                # "success_plateau_count": self.state.success_plateau_count,
                "suggested_focus": self._suggest_focus(combo_true,all_hypothesis_true),
                # "weak_signals": weak_signals,
                "recommended_improvements": recommended_hypotheses,
                # "best_signal_combination": self.state.best_signal_combination,
                # "current_signals": [s["name"] for s in signals],
            }

        prev_best = self.state.best_combo_rank_ic
        rel_improve = (combo_rank_ic - prev_best) / max(abs(prev_best), self.config.eps)

        # Plateau logic
        if rel_improve >= self.config.min_relative_improvement:
            self.state.best_combo_rank_ic = combo_rank_ic
            self.state.best_signal_combination = [s["name"] for s in signals]
            self.state.best_combo_hypothesis = combo_hypothesis
            self.state.plateau_count = 0
        else:
            self.state.plateau_count += 1

        # Early-stop success plateau logic
        if all_hypothesis_true and rel_improve < self.config.min_relative_improvement:
            self.state.success_plateau_count += 1
        else:
            self.state.success_plateau_count = 0

        self.state.history.append(result_json)

        if all_hypothesis_true:
            return {
                "decision": "end",
                "reason": "all_hypothesis_true",
                "combo_rank_ic": combo_rank_ic,
                "best_combo_rank_ic": self.state.best_combo_rank_ic,
                "relative_improvement_vs_best": rel_improve,
                "all_hypothesis_true": all_hypothesis_true,
                # "plateau_count": self.state.plateau_count,
                # "success_plateau_count": self.state.success_plateau_count,
                # "suggested_focus": self._suggest_focus(combo_true, weak_signals, all_hypothesis_true),
                # "weak_signals": weak_signals,
                "recommended_improvements": recommended_hypotheses,
                "best_signal_combination": self.state.best_signal_combination,
                "best_combo_hypothesis": self.state.best_combo_hypothesis,
                # "current_signals": [s["name"] for s in signals],
            }
        
        if combo_rank_ic > self.config.rank_ic_early_stop_threshold:
            self.state.best_combo_rank_ic = combo_rank_ic
            self.state.best_combo_hypothesis = combo_hypothesis
            return {
                "decision": "end",
                "reason": f"combo_rank_ic > {self.config.rank_ic_early_stop_threshold}",
                "combo_rank_ic": combo_rank_ic,
                "best_combo_rank_ic": self.state.best_combo_rank_ic,
                "relative_improvement_vs_best": rel_improve,
                "all_hypothesis_true": all_hypothesis_true,
                # "plateau_count": self.state.plateau_count,
                # "success_plateau_count": self.state.success_plateau_count,
                # "suggested_focus": self._suggest_focus(combo_true, weak_signals, all_hypothesis_true),
                # "weak_signals": weak_signals,
                "recommended_improvements": recommended_hypotheses,
                "best_signal_combination": self.state.best_signal_combination,
                "best_combo_hypothesis": self.state.best_combo_hypothesis,
                # "current_signals": [s["name"] for s in signals],
            }

        # End conditions
        if self.state.plateau_count >= self.config.plateau_rounds:
            return {
                "decision": "end",
                "reason": f"combo_rank_ic_plateau_{self.config.plateau_rounds}_rounds",
                "combo_rank_ic": combo_rank_ic,
                "best_combo_rank_ic": self.state.best_combo_rank_ic,
                "relative_improvement_vs_best": rel_improve,
                "all_hypothesis_true": all_hypothesis_true,
                # "plateau_count": self.state.plateau_count,
                # "success_plateau_count": self.state.success_plateau_count,
                # "suggested_focus": self._suggest_focus(combo_true, weak_signals, all_hypothesis_true),
                # "weak_signals": weak_signals,
                "recommended_improvements": recommended_hypotheses,
                "best_signal_combination": self.state.best_signal_combination,
                "best_combo_hypothesis": self.state.best_combo_hypothesis,
                # "current_signals": [s["name"] for s in signals],
            }

        if self.state.success_plateau_count >= self.config.success_plateau_rounds:
            return {
                "decision": "end",
                "reason": "all_hypothesis_true_and_plateau",
                # "combo_rank_ic": combo_rank_ic,
                # "best_combo_rank_ic": self.state.best_combo_rank_ic,
                # "relative_improvement_vs_best": rel_improve,
                "all_hypothesis_true": all_hypothesis_true,
                # "plateau_count": self.state.plateau_count,
                # "success_plateau_count": self.state.success_plateau_count,
                # "suggested_focus": self._suggest_focus(combo_true, weak_signals, all_hypothesis_true),
                # "weak_signals": weak_signals,
                "best_combo_hypothesis": self.state.best_combo_hypothesis,
                "best_signal_combination": self.state.best_signal_combination,
                # "current_signals": [s["name"] for s in signals],

            }

        return {
            "decision": "continue",
            "reason": "still_improving_or_not_yet_plateau",
            # "combo_rank_ic": combo_rank_ic,
            # "best_combo_rank_ic": self.state.best_combo_rank_ic,
            # "relative_improvement_vs_best": rel_improve,
            "all_hypothesis_true": all_hypothesis_true,
            # "plateau_count": self.state.plateau_count,
            # "success_plateau_count": self.state.success_plateau_count,
            "suggested_focus": self._suggest_focus(combo_true, all_hypothesis_true),
            # "weak_signals": weak_signals,
            "recommended_improvements": recommended_hypotheses,
            # "best_signal_combination": self.state.best_signal_combination,
            # "current_signals": [s["name"] for s in signals],

        }
    


    @staticmethod
    def _suggest_focus(combo_true: bool, all_hypotheses_true: bool) -> str:
        """
        Lightweight, non-binding suggestion.
        Judge still remains mostly a stop/continue controller.
        """
        if not all_hypotheses_true:
            return "refine_hypotheses"
        # if not combo_true and weak_signals:
        #     return "prune_weak_signal"
        if not combo_true:
            return "refine_combo"
        return "combo"

"""AgentBenchmark Template"""

from src.environment.quickbacktest.base_types import BaseStrategyEvaluation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class AgentStrategyEvaluation(BaseStrategyEvaluation):
    """
    AgentStrategyEvaluation
    ======================

    This class defines **how backtest trade logs and fill logs are evaluated**.
    The evaluation backend is **pandas / numpy / matplotlib**.

    When coding, treat this class as a **benchmark analysis module**,
    NOT a trading strategy module.

    Describe the benchmark logic in the docstring of this class
    following the format:

    Example: module name: MyBenchmark -> class name: MyBenchmark

    Benchmark Logic Overview
      - trade_analysis: explain what trade-level evidence is used
      - fills_analysis: explain what fill-level evidence is used
      - plots_analysis: explain what plots are used to support the same hypothesis

    Keep the class name same as module name for dynamic loading.

    This class is NOT responsible for trading execution.
    Therefore, DO NOT implement:
      - _run
      - handle_signal
      - handle_stop_loss
      - handle_take_profit
      - any order execution logic

    ============================================================
    Parent Class Contract
    ============================================================

    BaseStrategyBenchmark already provides in __init__:

        self.trade_log_path
        self.fills_log_path
        self.base_dir
        self.trades_df
        self.fills_df

    BaseStrategyBenchmark also provides:

        self._save_plot(fig, name)
        self.run()

    Important:
    - self.base_dir is already a Path
    - all plots must be saved only by calling:
          self._save_plot(fig, name)
    - do NOT call fig.savefig(...) directly
    - do NOT override run()
    - do NOT override __init__ unless absolutely necessary
    - if you override __init__, call super().__init__(...) first

    ============================================================
    Available Data Schema
    ============================================================

    Trades data: self.trades_df
      Columns:
        - dt_open
        - dt_close
        - barlen
        - pnl
        - pnlcomm
        - commission
        - is_liq

    Fills data: self.fills_df
      Columns:
        - dt
        - ref
        - side
        - size
        - price
        - value
        - commission
        - reason
        - is_liq

    Meanings:
      - pnl: gross trade pnl before commission
      - pnlcomm: net trade pnl after commission
      - commission: transaction cost
      - is_liq: whether liquidation occurred
      - barlen: holding duration measured in bars
      - reason: fill reason recorded by strategy logic

    ============================================================
    Core Benchmark Requirement
    ============================================================

    This benchmark MUST evaluate exactly ONE single hypothesis.

    Do NOT build a generic dashboard.
    Do NOT compute every possible metric.
    Do NOT mix unrelated benchmark goals.

    All required methods must support the SAME hypothesis:
      - trade_analysis()
      - fills_analysis()
      - plots_analysis()

    Fixed Hypothesis for this template:
      "The strategy's raw edge is materially weakened by transaction costs."

    Therefore:
      - trade_analysis must compare gross profitability vs net profitability
      - fills_analysis must evaluate commission burden and execution intensity
      - plots_analysis must visually show cost erosion
      - every metric and every plot must support this same claim

    ============================================================
    Benchmark Outputs
    ============================================================

    trade_analysis() must return:
        {
            "hypothesis": str,
            "summary": str,
            "metrics": dict,
            "warnings": list
        }

    fills_analysis() must return:
        {
            "hypothesis": str,
            "summary": str,
            "metrics": dict,
            "warnings": list
        }

    plots_analysis() must return:
        {
            "hypothesis": str,
            "summary": str,
            "plot_paths": list,
            "warnings": list
        }

    ============================================================
    Data Handling Rules
    ============================================================

    - Convert datetime columns when needed:
        self.trades_df["dt_open"]
        self.trades_df["dt_close"]
        self.fills_df["dt"]

    - Prefer pnlcomm when discussing real profitability
    - Use pnl when comparing gross vs net edge
    - Use commission when analyzing cost burden
    - Use is_liq only if relevant to the chosen hypothesis
    - Use reason only if relevant to the chosen hypothesis

    - Write robust code
    - Handle empty DataFrames gracefully
    - Avoid division by zero
    - Handle missing / invalid values safely
    - Return warnings in a list instead of crashing where possible

    ============================================================
    Recommended Helper Methods
    ============================================================

    You MAY add helper methods such as:

      - _prepare_trades()
      - _prepare_fills()
      - _safe_div(a, b)
      - _safe_mean(series)
      - _safe_median(series)
      - _build_warning(msg)

    Add helpers only if they make the class cleaner.

    ============================================================
    Method Responsibility Boundaries
    ============================================================

    trade_analysis():
      - Responsible for trade-level numeric evidence
      - Should use self.trades_df only
      - Should quantify gross edge, net edge, and cost drag
      - Must NOT generate plots
      - Must NOT call fills_analysis or plots_analysis

    fills_analysis():
      - Responsible for fill-level numeric evidence
      - Should use self.fills_df only
      - Should quantify commission burden and execution intensity
      - Must NOT generate plots
      - Must NOT call trade_analysis or plots_analysis

    plots_analysis():
      - Responsible for visual evidence supporting the SAME hypothesis
      - Can use self.trades_df and/or self.fills_df
      - Must save all plots only via self._save_plot(fig, name)
      - Must NOT call trade_analysis or fills_analysis

    ============================================================
    Plotting Rules
    ============================================================

    Example valid plots:
      - cumulative pnl vs cumulative pnlcomm
      - histogram of pnlcomm
      - commission distribution
      - cumulative commission over time
      - fill reason count plot (only if useful for the same hypothesis)

    Important:
      - Do NOT create unrelated plots
      - Do NOT use seaborn
      - Use matplotlib only
      - Every plot must support the same cost-erosion hypothesis

    ============================================================
    Code Style Rules
    ============================================================

    - Write full runnable production-style code
    - Use only:
        pandas
        numpy
        matplotlib
    - Do not use seaborn
    - Do not print debug output
    - Do not write explanatory prose outside code
    - Do not include markdown fences in output
    - Prefer clear and compact implementation
    - Use type hints only if helpful
    - Keep the class directly usable

    IMPORTANT:
    - Do NOT override next() or prenext()
    - Do NOT implement _run()
    - This class is not driven by bar iteration
    - run() is already provided by BaseStrategyBenchmark
    """

    def trade_analysis(self):
        """
        Trade-level numeric analysis for the cost-erosion hypothesis.

        Purpose
        -------
        Evaluate whether transaction costs materially reduce the strategy's
        apparent raw profitability at the closed-trade level.

        Data Used
        ---------
        self.trades_df with columns:
          - dt_open
          - dt_close
          - barlen
          - pnl
          - pnlcomm
          - commission
          - is_liq

        Recommended Analysis Directions
        -------------------------------
        - total trade count
        - gross total pnl using pnl
        - net total pnl using pnlcomm
        - total commission
        - average gross pnl per trade
        - average net pnl per trade
        - gross win rate based on pnl
        - net win rate based on pnlcomm
        - cost drag = gross pnl - net pnl
        - cost drag ratio
        - expectancy before cost
        - expectancy after cost

        Important
        ---------
        - Focus only on trade-level evidence relevant to transaction-cost erosion
        - Prefer pnlcomm when discussing real profitability
        - Compare pnl vs pnlcomm explicitly
        - Do NOT generate plots here
        - Do NOT call fills_analysis() or plots_analysis()

        Returns
        -------
        dict
            {
                "hypothesis": str,
                "summary": str,
                "metrics": dict,
                "warnings": list
            }
        """
        raise NotImplementedError

    def fills_analysis(self):
        """
        Fill-level numeric analysis for the same cost-erosion hypothesis.

        Purpose
        -------
        Evaluate whether fill frequency and commission burden are large enough
        to weaken or consume the strategy's raw edge.

        Data Used
        ---------
        self.fills_df with columns:
          - dt
          - ref
          - side
          - size
          - price
          - value
          - commission
          - reason
          - is_liq

        Recommended Analysis Directions
        -------------------------------
        - total fill count
        - total fill commission
        - average commission per fill
        - median commission per fill
        - total traded value
        - average absolute fill value
        - commission as fraction of traded value
        - fill count by side
        - fill count by reason
        - liquidation fill ratio if relevant
        - execution intensity proxy if relevant

        Important
        ---------
        - Focus only on fill-level evidence relevant to transaction-cost erosion
        - Do NOT drift into unrelated topics such as slippage or latency
          unless those fields actually exist
        - Do NOT generate plots here
        - Do NOT call trade_analysis() or plots_analysis()

        Returns
        -------
        dict
            {
                "hypothesis": str,
                "summary": str,
                "metrics": dict,
                "warnings": list
            }
        """
        raise NotImplementedError

    def plots_analysis(self):
        """
        Visual analysis for the same cost-erosion hypothesis.

        Purpose
        -------
        Generate plots that visually demonstrate how transaction costs reduce
        raw strategy profitability.

        Data Used
        ---------
        self.trades_df and/or self.fills_df

        Recommended Plot Choices
        ------------------------
        - cumulative gross pnl vs cumulative net pnl
        - histogram of pnlcomm
        - histogram of commission per trade or per fill
        - cumulative commission over time
        - fill reason count chart if it helps explain cost generation

        Plot Saving Rules
        -----------------
        Every figure MUST be saved only with:

            self._save_plot(fig, name)

        Do NOT call fig.savefig(...) directly.
        Do NOT manually build file paths.

        Important
        ---------
        - Every plot must support the same cost-erosion hypothesis
        - Do NOT create unrelated charts
        - Do NOT call trade_analysis() or fills_analysis()

        Returns
        -------
        dict
            {
                "hypothesis": str,
                "summary": str,
                "plot_paths": list,
                "warnings": list
            }
        """
        raise NotImplementedError
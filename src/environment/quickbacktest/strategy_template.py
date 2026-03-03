"""AgentStrategy Template"""

from src.environment.quickbacktest.base_types import BaseStrategy
import pandas as pd
import talib as ta

class AgentStrategy(BaseStrategy):
    """
    AgentStrategy
    =============

    This class defines **how trading actions are executed**.
    The execution backend is **Backtrader**.
    When coding, always use tz-aware DatetimeIndex.

    Describe the strategy logic in the docstring of this class
    following the format 

    Example: module name: MyStrategy  -> class name: MyStrategy

    Strategy Logic Overview
      - handle_signal: explain entry and reversal logic
      - handle_stop_loss: explain risk exit logic
      - handle_take_profit: explain profit-taking logic

    Keep the class name same as module name for dynamic loading.


    All trading operations described here are ultimately translated
    into Backtrader orders (Market orders by default).

    Insights:
    - Reduce frequent trading by introducing time

    Example:
    def _run(self, symbol: str) -> None:

        current_time: str = bt.num2date(
            self.getdatabyname(symbol).datetime[0]
        ).strftime("%H:%M:%S")

        if current_time in ["04:30:00","11:30:00","18:30:00"]:

            self.handle_signal(symbol)

        elif current_time in ["23:55:00","07:55:00","15:55:00"]:
            self.rebalance(symbol)

        elif self.getpositionbyname(symbol).size == 0:
            pass

        else:
            self.handle_stop_loss(symbol)
            self.handle_take_profit(symbol)


    Data are predefined by BaseStrategy and include in __init___:
    call super().__init__() first to initialize BaseStrategy if you override __init__ # DO NOT INCLUDE ARGS
        self.signal_1: Dict = {d._name: d.signal_1 for d in self.datas}
        self.signal_2: Dict = {d._name: d.signal_2 for d in self.datas}
        self.signal_3: Dict = {d._name: d.signal_3 for d in self.datas}

    Data can be accessed using:
    self.signal_1[symbol][0], self.signal_2[symbol][0], self.signal_3[symbol][0]

    BaseStrategy only guarantees:
      self.signal_1 / self.signal_2 / self.signal_3

    Therefore, this strategy MUST NOT access:
      self.high / self.low / self.close / self.open ..

    ============================================================
    Trading Operations (Conceptual Definitions)
    ============================================================

    1) Open Position
       -------------
       Meaning:
         - Enter a new position from flat (no position)
         - Can be either long (> 0) or short (< 0)

       When to use:
         - No existing position
         - Entry conditions are satisfied

       Backtrader execution:
         - Uses self.buy(...) or self.sell(...)
         - Wrapped by BaseStrategy._open_position(...)
         - self._open_position(data, reason: str, action) , action is self.buy or self.sell
         Example:
          - self._open_position(data, f"{symbol} short open and your reason", self.sell)
          - self._open_position(data, f"{symbol} long open and your reason", self.buy)
         - Order type: Market

       Position change:
         - 0 → +size   (open long)
         - 0 → -size   (open short)

    ------------------------------------------------------------

    2) Reverse Position
       ----------------
       Meaning:
         - Close the current position and open the opposite position
         - Treated as one logical trading decision

       When to use:
         - Existing position is in the wrong direction
         - New signal strongly favors the opposite direction

       Backtrader execution:
         - Issues a close order, then a buy/sell in the opposite direction
         - Wrapped by BaseStrategy._close_and_reverse(...), smilar to _open_position

       Position change:
         - +size → -size
         - -size → +size

    ------------------------------------------------------------

    3) Close all Positions
       ----------
       Meaning:
         - Exit an existing position without reversing
         - Often used for take-profit or protective exits

       Backtrader execution:
         - Uses self._close_position(data, reason: str)

       Position change:
         - +size → 0
         - -size → 0


    ============================================================
    Method Responsibility Boundaries
    ============================================================

    handle_signal(symbol):
      - Responsible for entry and reversal decisions
      - Allowed operations:
          * Open position
          * Reverse position
      - Must NOT handle stop-loss or take-profit logic

    handle_stop_loss(symbol):
      - Responsible for risk-driven exits
      - Allowed operations:
          * Rebalance (close only)
          * Reverse position (forced reversal)
      - Must NOT introduce new entry logic

    handle_take_profit(symbol):
      - Responsible for profit-taking exits
      - Allowed operations:
          * Rebalance (close only)
          * Close only
      - Must NOT open or reverse positions

    ============================================================
    Execution Backend (Backtrader)
    ============================================================

    - Orders are executed via Backtrader's broker
    - Strategy logic is evaluated bar-by-bar
    - Position state is obtained via:
        self.getpositionbyname(symbol).size
    IMPORTANT:
    - Do NOT override next() or prenext()
    - _run(symbol) is called by BaseStrategy and driven by Backtrader
    """

    def handle_signal(self, symbol: str) -> None:
        """Entry and reversal logic (open / reverse positions)."""
        pass

    def handle_stop_loss(self, symbol: str) -> None:
        """Risk exit logic (rebalance or forced reversal)."""
        pass

    def handle_take_profit(self, symbol: str) -> None:
        """Profit-taking logic (close only / rebalance)."""
        pass

    def _run(self, symbol: str) -> None:
        """
        Per-bar execution coordinator for one symbol.

        Recommended execution order:
          1) handle_stop_loss   (highest priority)
          2) handle_take_profit
          3) handle_signal      (entry / reversal)

        This method is invoked by the BaseStrategy layer
        and ultimately driven by Backtrader's bar iteration.
        """
        pass





    

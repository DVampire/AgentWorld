from typing import List, Literal, Optional,Dict, Callable,Any
import numpy as np
import pandas as pd
import talib as ta
import backtrader as bt
from loguru import logger
from pathlib import Path
import csv



__all__ = ["BaseSignal","BaseStrategy"]


class BaseSignal:
    """
    Docstring for BaseSignal
    """

    REQUIRED = ("get_signals", "concat_signals")
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        missing = [m for m in cls.REQUIRED if m not in cls.__dict__]
        if missing:
            raise TypeError(
                f"{cls.__name__} must define methods: {', '.join(missing)}"
            )
    def __init__(self,ohlcv:pd.DataFrame)->None:
        required = {"code", "trade_time", "close", "volume"}
        missing = required - set(ohlcv.columns)
        if missing:
            raise ValueError(f"ohlcv is missing required columns: {sorted(missing)}")

        self.ohlcv: pd.DataFrame = ohlcv.copy()
        self.ohlcv["trade_time"] = pd.to_datetime(self.ohlcv["trade_time"])

        self.pivot_frame: pd.DataFrame = pd.pivot_table(
            self.ohlcv,
            index="trade_time",
            columns="code",
            values=["close", "volume","open", "high", "low","amount"],
        ).sort_index()

        self.close: pd.DataFrame = self.pivot_frame["close"]
        self.volume: pd.DataFrame = self.pivot_frame["volume"]
        self.open: pd.DataFrame = self.pivot_frame["open"]
        self.high: pd.DataFrame = self.pivot_frame["high"]
        self.low: pd.DataFrame = self.pivot_frame["low"]
        self.amount: pd.DataFrame = self.pivot_frame["amount"]

    def get_signals(self,**kwargs)->pd.DataFrame:
        """Get signal DataFrame"""
        pass


    def concat_signals(self,data:pd.DataFrame)->pd.DataFrame:
        """Concatenate signal to original data"""
        pass


    def fit(self) -> pd.DataFrame:
        return self.concat_signals(self.ohlcv)

    def calculate_rolling_vwap(
        self,
        window: int,
    ) -> pd.Series:
        """
        Calculate rolling VWAP (no daily reset, perps-friendly)

        Parameters
        ----------
        df : pd.DataFrame
            Time-indexed OHLCV data
        window : int
            Rolling window length (number of bars)
        price_col : str
            Price column used in VWAP
        volume_col : str
            Volume column used in VWAP

        Returns
        -------
        pd.Series
            Rolling VWAP aligned with df.index
        """

        pv = self.close * self.volume

        vwap = (
            pv.rolling(window=window, min_periods=1).sum()
            / self.volume.rolling(window=window, min_periods=1).sum()
        )

        return vwap
    

class BaseStrategy(bt.Strategy):
    """
    BaseStrategy template (single-asset, market orders, no cheat_on_open/close).

    Features:
    - Enforces subclass to implement: handle_signal / handle_stop_loss / handle_take_profit
    - Position sizing with leverage
    - Safe order gating for market orders (avoid canceling pending market orders)
    - Forced liquidation (strongly simplified perp-style):
        * If equity <= notional * maintenance_margin_rate => close position by market
        * Optional liquidation fee: notional * liq_fee_rate (deducted via broker.add_cash if available)
        * After liquidation triggers once, strategy stops trading (configurable via halt_after_liq)
    - Trade logging:
        * fills.csv  : every completed order execution (with reason)
        * trades.csv : every closed trade (round-trip, with optional liq flag)
    """

    params: Dict[str, Any] = dict(
        commission=0.01,      # used here as sizing buffer (NOT broker commission)
        hold_num=1,           # keep for compatibility; single-asset => usually 1
        leverage=1,
        verbose=False,

        # output
        log_dir="trade_logs",
        fills_csv="fills.csv",
        trades_csv="trades.csv",

        # liquidation
        enable_liquidation=True,
        maintenance_margin_rate=0.005,  # e.g., 0.5%
        liq_fee_rate=0.0,               # optional extra fee on liquidation, e.g., 0.0005 (5 bps)
        halt_after_liq=True,            # stop trading after a liquidation event
    )

    REQUIRED = ("handle_signal", "handle_stop_loss", "handle_take_profit")

    def __init__(self) -> None:
        # single-asset order handle
        self.order: Optional[bt.Order] = None

        # reverse state machine: close first, then open reverse after close fills
        self._pending_reverse: Optional[Dict[str, Any]] = None  # {"action": Callable, "size": float, "reason": str}

        # liquidation state
        self._liq_triggered: bool = False
        self._liq_pending: bool = False  # liquidation close order submitted but not completed yet

        # lines mapping (single asset still fine)
        self.signal_1: Dict[str, Any] = {d._name: d.signal_1 for d in self.datas}
        self.signal_2: Dict[str, Any] = {d._name: d.signal_2 for d in self.datas}
        self.signal_3: Dict[str, Any] = {d._name: d.signal_3 for d in self.datas}

        # trade records
        self.fill_records: list[Dict[str, Any]] = []
        self.trade_records: list[Dict[str, Any]] = []

        # order metadata (reason tagging)
        self._order_reason: Dict[int, str] = {}      # order.ref -> reason string
        self._order_is_liq: Dict[int, bool] = {}     # order.ref -> whether liquidation-driven

        self.log(
            f"策略初始化完成 - sizing_commission_buffer: {self.p.commission}",
            pd.Timestamp.now(),
            verbose=self.p.verbose,
        )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        missing = [m for m in cls.REQUIRED if m not in cls.__dict__]
        if missing:
            raise TypeError(f"{cls.__name__} must define methods: {', '.join(missing)}")

    # --------- required interface (must be implemented by subclass) ---------

    def handle_signal(self, symbol: str) -> None:
        raise NotImplementedError

    def handle_stop_loss(self, symbol: str) -> None:
        raise NotImplementedError

    def handle_take_profit(self, symbol: str) -> None:
        raise NotImplementedError

    # ------------------------------ utils ------------------------------

    def log(self, msg: str, current_dt: pd.Timestamp = None, verbose: bool = False) -> None:
        if current_dt is None:
            current_dt = self.datetime.datetime(0)
        if verbose:
            logger.info(f"{current_dt} {msg}")

    def _calculate_size(self, data: bt.LineSeries) -> float:
        """
        Perps-like position sizing:
        - alloc_cash = equity * (1 - commission_buffer) / hold_num
        - notional   = alloc_cash * leverage
        - size       = notional / price
        """
        price = float(data.close[0])
        equity = float(self.broker.getvalue())

        alloc_cash = equity * (1.0 - float(self.p.commission)) / float(self.p.hold_num)
        notional = alloc_cash * float(self.p.leverage)
        return notional / price if price > 0 else 0.0

    def _tag_order(self, order: bt.Order, reason: str, is_liq: bool = False) -> None:
        """Attach reason/liquidation flag to an order ref for later logging."""
        try:
            ref = int(order.ref)
        except Exception:
            return
        self._order_reason[ref] = reason
        self._order_is_liq[ref] = bool(is_liq)

    def _open_position(self, data: bt.LineSeries, reason: str, action: Callable) -> None:
        """
        Open position using market order.
        action: self.buy or self.sell
        """
        self.log(reason, verbose=self.p.verbose)
        size = self._calculate_size(data)
        o = action(data=data, size=size, exectype=bt.Order.Market)
        self.order = o
        self._tag_order(o, reason=reason, is_liq=False)

    def _close_and_reverse(self, data: bt.LineSeries, reason: str, new_action: Callable) -> None:
        """
        Reverse position safely:
        1) submit close market order
        2) after close is filled, submit reverse open market order in notify_order()
        """
        self.log(reason, verbose=self.p.verbose)

        size = self._calculate_size(data)
        self._pending_reverse = {"action": new_action, "size": size, "reason": reason}

        # close first
        o = self.close(data=data, exectype=bt.Order.Market)
        self.order = o
        self._tag_order(o, reason=f"{reason} | close_for_reverse", is_liq=False)


    def _close_position(self, data, reason):
        if self.order and self.order.status in [bt.Order.Submitted, bt.Order.Accepted]:
            return
        if not self.getposition(data).size:
            return
        self.log(reason, verbose=self.p.verbose)
        o = self.close(data=data, exectype=bt.Order.Market)
        self.order = o
        self._tag_order(o, reason=reason, is_liq=False)

    # ------------------------------ liquidation ------------------------------

    def _get_mark_like_price(self, data: bt.LineSeries) -> float:
        """
        Use data.close as a proxy for mark price.
        If your feed provides a 'mark' line, you can switch to it here.
        """
        return float(data.close[0])

    def _check_liquidation(self, data: bt.LineSeries) -> bool:
        """
        Forced liquidation rule (simplified):
        if equity <= notional * maintenance_margin_rate:
            submit market close; optionally deduct liquidation fee.
        Returns True if liquidation triggered (order submitted) this bar.
        """
        if not bool(self.p.enable_liquidation):
            return False

        if self._liq_triggered and bool(self.p.halt_after_liq):
            return False

        if self._liq_pending:
            return False

        pos = self.position
        if pos.size == 0:
            return False

        price = self._get_mark_like_price(data)
        notional = abs(float(pos.size)) * price
        equity = float(self.broker.getvalue())
        maint = notional * float(self.p.maintenance_margin_rate)

        if equity <= maint:
            # trigger liquidation
            self._liq_triggered = True
            self._liq_pending = True

            # optional liquidation fee (best-effort)
            liq_fee_rate = float(self.p.liq_fee_rate)
            if liq_fee_rate > 0:
                fee = notional * liq_fee_rate
                # Backtrader broker typically supports add_cash; if not, skip silently
                if hasattr(self.broker, "add_cash"):
                    try:
                        self.broker.add_cash(-fee)
                    except Exception:
                        pass

            reason = (
                f"FORCED_LIQUIDATION: equity({equity:.6f}) <= maint({maint:.6f}) "
                f"| notional={notional:.6f} mmr={float(self.p.maintenance_margin_rate):.6f}"
            )
            self.log(reason, verbose=True)

            # cancel any pending reverse intent
            self._pending_reverse = None

            # submit close
            o = self.close(data=data, exectype=bt.Order.Market)
            self.order = o
            self._tag_order(o, reason=reason, is_liq=True)
            return True

        return False

    # ------------------------------ engine hooks ------------------------------

    def prenext(self) -> None:
        self.next()

    def next(self) -> None:
        """
        Single-asset + market orders + no cheat:
        - Do NOT cancel pending orders each bar.
        - If there's a pending order (Submitted/Accepted), skip this bar.
        - Liquidation check runs BEFORE any strategy logic.
        """
        data = self.data
        symbol = data._name

        # If we already have a pending order, skip.
        if self.order and self.order.status in [bt.Order.Submitted, bt.Order.Accepted]:
            return

        # If liquidation has already happened and we halt trading, stop here.
        if self._liq_triggered and bool(self.p.halt_after_liq):
            return

        # 1) liquidation first
        if self._check_liquidation(data):
            return  # liquidation order submitted this bar

        # 2) normal strategy logic
        self._run(symbol)

    def _run(self, symbol: str) -> None:
        """
        Trade every bar based on:
        - stop loss
        - take profit
        - signal
        """
        _ = bt.num2date(self.getdatabyname(symbol).datetime[0]).strftime("%H:%M:%S")

        self.handle_stop_loss(symbol)
        self.handle_take_profit(symbol)
        self.handle_signal(symbol)

    def notify_order(self, order: bt.Order) -> None:
        """
        1) Record fills when Completed
        2) Clear self.order when order lifecycle ends
        3) If reverse pending and close filled -> open reverse
        4) Clear liquidation pending when liquidation close completes
        """
        if order.status in [order.Submitted, order.Accepted]:
            return

        # reason tags
        ref = int(order.ref)
        reason = self._order_reason.get(ref, "")
        is_liq = bool(self._order_is_liq.get(ref, False))

        if order.status == order.Completed:
            dt = bt.num2date(order.executed.dt)
            self.fill_records.append(
                {
                    "dt": dt.isoformat(sep=" "),
                    "ref": ref,
                    "side": "BUY" if order.isbuy() else "SELL",
                    "size": float(order.executed.size),
                    "price": float(order.executed.price),
                    "value": float(order.executed.value),
                    "commission": float(order.executed.comm),
                    "reason": reason,
                    "is_liq": int(is_liq),
                }
            )

        # order finished -> clear pointer
        if order.status in [order.Completed, order.Canceled, order.Rejected]:
            self.order = None

        # if this was liquidation close order and it finished
        if is_liq and order.status in [order.Completed, order.Canceled, order.Rejected]:
            self._liq_pending = False

        # reverse: after close fills, submit reverse open (only if not liquidated/halting)
        if (
            order.status == order.Completed
            and self._pending_reverse
            and not (self._liq_triggered and bool(self.p.halt_after_liq))
        ):
            pending = self._pending_reverse
            self._pending_reverse = None

            action = pending["action"]
            size = float(pending["size"])
            reason_open = f"{pending.get('reason', '')} | open_reverse"

            o = action(data=order.data, size=size, exectype=bt.Order.Market)
            self.order = o
            self._tag_order(o, reason=reason_open, is_liq=False)

    def notify_trade(self, trade: bt.Trade) -> None:
        """
        Record closed trades (round-trip).
        """
        if not trade.isclosed:
            return

        dt_open = bt.num2date(trade.dtopen)
        dt_close = bt.num2date(trade.dtclose)

        # If liquidation happened, mark trade as liquidated if close time is after trigger.
        # (Simplified flag: once liquidation triggers, any subsequent closed trade is tagged.)
        is_liq_trade = 1 if self._liq_triggered else 0

        self.trade_records.append(
            {
                "dt_open": dt_open.isoformat(sep=" "),
                "dt_close": dt_close.isoformat(sep=" "),
                "barlen": int(trade.barlen),
                "pnl": float(trade.pnl),
                "pnlcomm": float(trade.pnlcomm),
                "commission": float(trade.commission),
                "is_liq": is_liq_trade,
            }
        )

    def stop(self) -> None:
        """
        Persist records to CSV under p.log_dir.
        """
        outdir = Path(str(self.p.log_dir))
        outdir.mkdir(parents=True, exist_ok=True)

        # fills
        fills_path = outdir / str(self.p.fills_csv)
        cols = ["dt", "ref", "side", "size", "price", "value", "commission", "reason", "is_liq"]
        with fills_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            if self.fill_records:
                w.writerows(self.fill_records)
        

        # trades
        trades_path = outdir / str(self.p.trades_csv)
        cols = ["dt_open", "dt_close", "barlen", "pnl", "pnlcomm", "commission", "is_liq"]
        with trades_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            if self.trade_records:
                w.writerows(self.trade_records)


class BaseSignalBenchmark():
    """BaseBenchmark template for signal benchmarks."""
    
    REQUIRED = ("evaluate")

    def __init__(self,signal:BaseSignal,data:pd.DataFrame) -> None:
        self.signal = signal
        self.data = data
        self.combo_data =  self.signal(self.data).fit()
        self.pivot_frame: pd.DataFrame = pd.pivot_table(
            self.combo_data,
            index="trade_time",
            columns="code",
            values=["close", "volume","open", "high", "low","amount","signal","factor1","factor2"],
        ).sort_index()

        self.close: pd.DataFrame = self.pivot_frame["close"]
        self.volume: pd.DataFrame = self.pivot_frame["volume"]
        self.open: pd.DataFrame = self.pivot_frame["open"]
        self.high: pd.DataFrame = self.pivot_frame["high"]
        self.low: pd.DataFrame = self.pivot_frame["low"]
        self.amount: pd.DataFrame = self.pivot_frame["amount"]
        self.signal: pd.DataFrame = self.pivot_frame["signal"]
        self.factor1: pd.DataFrame = self.pivot_frame["factor1"]
        self.factor2: pd.DataFrame = self.pivot_frame["factor2"]


    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        missing = [m for m in cls.REQUIRED if m not in cls.__dict__]
        if missing:
            raise TypeError(
                f"{cls.__name__} must define methods: {', '.join(missing)}"
            )

    def evaluate(self):
        raise NotImplementedError



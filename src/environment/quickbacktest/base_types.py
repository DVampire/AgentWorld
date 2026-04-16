from typing import List, Literal, Optional,Dict, Callable,Any, Type
from ddgs import results
import numpy as np
import pandas as pd
import talib as ta
import backtrader as bt
from loguru import logger
from pathlib import Path
import csv
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod



__all__ = ["BaseSignal","BaseStrategy","SignalData","SignalRegistry","SignalPipeline","BaseStrategyEvaluation"]


class SignalData:
    """
    Shared market data container for all signals.

    Expected input
    --------------
    Long-format OHLCV DataFrame with columns:
        code, trade_time, open, high, low, close, volume, amount
    """

    REQUIRED_COLUMNS = {
        "code",
        "trade_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    }

    BASE_FIELDS = ["open", "high", "low", "close", "volume", "amount"]

    def __init__(self, ohlcv: pd.DataFrame) -> None:
        if not isinstance(ohlcv, pd.DataFrame):
            raise TypeError("ohlcv must be a pandas DataFrame")

        missing = self.REQUIRED_COLUMNS - set(ohlcv.columns)
        if missing:
            raise ValueError(f"ohlcv is missing required columns: {sorted(missing)}")

        self.ohlcv = ohlcv.copy()
        self.ohlcv["trade_time"] = pd.to_datetime(self.ohlcv["trade_time"])
        self.ohlcv = self.ohlcv.sort_values(["trade_time", "code"]).reset_index(drop=True)

        self.pivot_frame = pd.pivot_table(
            self.ohlcv,
            index="trade_time",
            columns="code",
            values=self.BASE_FIELDS,
        ).sort_index()

        self.open = self.pivot_frame["open"]
        self.high = self.pivot_frame["high"]
        self.low = self.pivot_frame["low"]
        self.close = self.pivot_frame["close"]
        self.volume = self.pivot_frame["volume"]
        self.amount = self.pivot_frame["amount"]


class BaseSignal(ABC):
    """
    Independent signal module.

    Rules
    -----
    1. Signal does NOT know its final slot name.
    2. Signal only defines:
       - logical name
       - compute()
    3. compute() must return wide-format DataFrame:
       - index   = trade_time
       - columns = code
    """

    name: str = ""

    def __init__(self, data: SignalData) -> None:
        if not isinstance(data, SignalData):
            raise TypeError("data must be an instance of SignalData")

        self.data = data

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                f"{self.__class__.__name__}.name must be a non-empty string"
            )
        self.open: pd.DataFrame = data.open
        self.high: pd.DataFrame = data.high
        self.low: pd.DataFrame = data.low
        self.close: pd.DataFrame = data.close
        self.volume: pd.DataFrame = data.volume
        self.amount: pd.DataFrame = data.amount
        
        self.ohlcv: pd.DataFrame = data.ohlcv




    @abstractmethod
    def compute(self, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def validate_wide_output(self, frame: pd.DataFrame) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("output must be a pandas DataFrame")

        if not frame.index.equals(self.data.close.index):
            raise ValueError("output index does not match data.close.index")

        if list(frame.columns) != list(self.data.close.columns):
            raise ValueError("output columns do not match data.close.columns")

    def wide_to_long(self, frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
        self.validate_wide_output(frame)
        out = frame.stack(dropna=False).rename(value_name).reset_index()
        out.columns = ["trade_time", "code", value_name]
        return out

    def attach_as(
        self,
        slot_name: str,
        df: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Attach this signal to a long-format DataFrame using pipeline-assigned slot_name.
        """
        base = self.data.ohlcv.copy() if df is None else df.copy()
        signal_wide = self.compute(**kwargs)
        signal_long = self.wide_to_long(signal_wide, slot_name)
        return base.merge(signal_long, on=["trade_time", "code"], how="left")

    # -------------------------
    # Common helpers
    # -------------------------
    def rolling_vwap(self, window: int, min_periods: int = 1) -> pd.DataFrame:
        if window <= 0:
            raise ValueError("window must be positive")

        pv = self.data.close * self.data.volume
        pv_sum = pv.rolling(window=window, min_periods=min_periods).sum()
        vol_sum = self.data.volume.rolling(window=window, min_periods=min_periods).sum()
        return pv_sum / vol_sum

    def rolling_zscore(
        self,
        frame: pd.DataFrame,
        window: int,
        min_periods: Optional[int] = None,
        ddof: int = 1,
    ) -> pd.DataFrame:
        if window <= 0:
            raise ValueError("window must be positive")

        if min_periods is None:
            min_periods = window

        mean_ = frame.rolling(window=window, min_periods=min_periods).mean()
        std_ = frame.rolling(window=window, min_periods=min_periods).std(ddof=ddof)
        return (frame - mean_) / std_


class SignalRegistry:
    """
    Fixed slot registry.

    Slots
    -----
    signal_1 ... signal_5

    Notes
    -----
    - Signals themselves are slot-agnostic.
    - Registry decides which signal goes into which slot.
    """

    SLOT_NAMES = [f"signal_{i}" for i in range(1, 6)]

    def __init__(self) -> None:
        self._slots: dict[str, Optional[BaseSignal]] = {
            slot: None for slot in self.SLOT_NAMES
        }

    def assign(self, slot_name: str, signal: BaseSignal, overwrite: bool = False) -> None:
        if slot_name not in self._slots:
            raise ValueError(f"invalid slot name: {slot_name}")

        if not isinstance(signal, BaseSignal):
            raise TypeError("signal must be an instance of BaseSignal")

        if self._slots[slot_name] is not None and not overwrite:
            raise ValueError(
                f"{slot_name} is already occupied by {self._slots[slot_name].__class__.__name__}; "
                f"use overwrite=True to replace it"
            )

        self._slots[slot_name] = signal

    def replace(self, slot_name: str, signal: BaseSignal) -> None:
        self.assign(slot_name=slot_name, signal=signal, overwrite=True)

    def remove(self, slot_name: str) -> None:
        if slot_name not in self._slots:
            raise ValueError(f"invalid slot name: {slot_name}")
        self._slots[slot_name] = None

    def clear(self) -> None:
        for slot in self.SLOT_NAMES:
            self._slots[slot] = None

    def get(self, slot_name: str) -> Optional[BaseSignal]:
        if slot_name not in self._slots:
            raise ValueError(f"invalid slot name: {slot_name}")
        return self._slots[slot_name]

    def items(self) -> list[tuple[str, Optional[BaseSignal]]]:
        return [(slot, self._slots[slot]) for slot in self.SLOT_NAMES]

    def active_signals(self) -> list[BaseSignal]:
        return [sig for _, sig in self.items() if sig is not None]

    def summary(self) -> pd.DataFrame:
        rows = []
        for slot in self.SLOT_NAMES:
            sig = self._slots[slot]
            rows.append(
                {
                    "slot": slot,
                    "occupied": sig is not None,
                    "signal_class": None if sig is None else sig.__class__.__name__,
                    "signal_name": None if sig is None else sig.name,
                }
            )
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        parts = []
        for slot in self.SLOT_NAMES:
            sig = self._slots[slot]
            desc = None if sig is None else f"{sig.__class__.__name__}({sig.name})"
            parts.append(f"{slot}={desc}")
        return f"SignalRegistry({', '.join(parts)})"


class SignalPipeline:
    """
    Production pipeline.

    Guarantees
    ----------
    1. Always outputs signal_1 ~ signal_5
    2. Missing slots are filled with -1
    3. Signals are renamed inside pipeline
    4. Optional VWAP column is appended
    5. Stable column order
    """

    SLOT_NAMES = [f"signal_{i}" for i in range(1, 6)]

    def __init__(self, registry: SignalRegistry) -> None:
        if not isinstance(registry, SignalRegistry):
            raise TypeError("registry must be an instance of SignalRegistry")

        self.registry = registry

        active = registry.active_signals()
        if not active:
            raise ValueError("registry has no active signals")

        self.data = active[0].data
        for sig in active:
            if sig.data is not self.data:
                raise ValueError("all active signals must share the same SignalData instance")

    def _attach_vwap(
        self,
        df: pd.DataFrame,
        window: int = 20,
        min_periods: int = 1,
        col_name: str = "vwap",
    ) -> pd.DataFrame:
        helper = self.registry.active_signals()[0]
        vwap_wide = helper.rolling_vwap(window=window, min_periods=min_periods)
        vwap_long = helper.wide_to_long(vwap_wide, col_name)
        return df.merge(vwap_long, on=["trade_time", "code"], how="left")

    def run(
        self,
        df: Optional[pd.DataFrame] = None,
        signal_kwargs: Optional[dict[str, dict]] = None,
        include_vwap: bool = True,
        vwap_kwargs: Optional[dict] = None,
    ) -> pd.DataFrame:
        signal_kwargs = signal_kwargs or {}
        vwap_kwargs = vwap_kwargs or {}

        result = self.data.ohlcv.copy() if df is None else df.copy()

        for slot in self.SLOT_NAMES:
            sig = self.registry.get(slot)

            if sig is None:
                result[slot] = -1
            else:
                kwargs = signal_kwargs.get(slot, {})
                result = sig.attach_as(slot_name=slot, df=result, **kwargs)
                result[slot] = result[slot].fillna(-1)

        if include_vwap:
            col_name = vwap_kwargs.get("col_name", "vwap")
            result = self._attach_vwap(
                df=result,
                window=vwap_kwargs.get("window", 20),
                min_periods=vwap_kwargs.get("min_periods", 1),
                col_name=col_name,
            )

        base_cols = [
            "code",
            "trade_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
        ]
        signal_cols = self.SLOT_NAMES
        extra_cols = [
            c for c in result.columns
            if c not in base_cols + signal_cols + ["vwap"]
        ]

        ordered_cols = base_cols + signal_cols + extra_cols
        if "vwap" in result.columns:
            ordered_cols.append("vwap")

        return result[ordered_cols]



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
    def _validate_perc(self, perc: float) -> float:
        perc = float(perc)
        if perc <= 0.0:
            raise ValueError(f"perc must be > 0, got {perc}")
        if perc > 1.0:
            raise ValueError(f"perc must be <= 1, got {perc}")
        return perc

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
        self.signal_4: Dict[str, Any] = {d._name: d.signal_4 for d in self.datas}
        self.signal_5: Dict[str, Any] = {d._name: d.signal_5 for d in self.datas}

        self.c = {d._name: d.close for d in self.datas}
        self.o = {d._name: d.open for d in self.datas}
        self.h = {d._name: d.high for d in self.datas}
        self.l = {d._name: d.low for d in self.datas}
        self.v = {d._name: d.volume for d in self.datas}
        self.a = {d._name: d.amount for d in self.datas}
        self.vwap = {d._name: d.vwap for d in self.datas}

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

    def _open_position(self, data: bt.LineSeries, reason: str, action: Callable, perc: float) -> None:
        """
        Open position using market order.
        action: self.buy or self.sell
        """
        self.log(reason, verbose=self.p.verbose)
        size = self._calculate_size(data) * self._validate_perc(perc)
        o = action(data=data, size=size, exectype=bt.Order.Market)
        self.order = o
        self._tag_order(o, reason=reason, is_liq=False)

    def _close_and_reverse(self, data: bt.LineSeries, reason: str, new_action: Callable, perc: float) -> None:
        """
        Reverse position safely:
        1) submit close market order
        2) after close is filled, submit reverse open market order in notify_order()
        """
        self.log(reason, verbose=self.p.verbose)

        size = self._calculate_size(data) * self._validate_perc(perc)
        self._pending_reverse = {"action": new_action, "size": size, "reason": reason}

        # close first
        o = self.close(data=data, exectype=bt.Order.Market)
        self.order = o
        self._tag_order(o, reason=f"{reason} | close_for_reverse", is_liq=False)


    def _close_position(self, data, reason, perc: float) -> None:
        if self.order and self.order.status in [bt.Order.Submitted, bt.Order.Accepted]:
            return

        pos_size = self.getposition(data).size
        if not pos_size:
            return

        self.log(reason, verbose=self.p.verbose)

        close_size = abs(pos_size) * perc
        o = self.close(data=data, exectype=bt.Order.Market, size=close_size)
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



class BaseSignalEvaluation:
    """Base benchmark template for signal evaluation."""

    REQUIRED = ("numeric_evaluation", "plot_evaluation")
    BASE_FIELDS = ["open", "high", "low", "close", "volume", "amount"]

    def __init__(self, data: pd.DataFrame, base_dir: str) -> None:
        self.combo_data = data.copy()
        self.base_dir = base_dir

        required_index_name = "trade_time"
        if self.combo_data.index.name != required_index_name:
            raise ValueError(
                f"combo_data index must be '{required_index_name}', "
                f"got {self.combo_data.index.name!r}"
            )

        signal_fields = [c for c in self.combo_data.columns if c.startswith("signal_")]
        pivot_values = self.BASE_FIELDS + signal_fields

        required_cols = {"code", *pivot_values}
        missing = required_cols - set(self.combo_data.columns)
        if missing:
            raise ValueError(f"combo_data is missing required columns: {sorted(missing)}")

        self.signal_fields = signal_fields

        # reset_index() so trade_time becomes a normal column for pivot
        self.pivot_frame: pd.DataFrame = (
            self.combo_data.reset_index()
            .pivot(index="trade_time", columns="code", values=pivot_values)
            .sort_index()
        )

        for field in pivot_values:
            setattr(self, field, self.pivot_frame[field])

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        missing = [m for m in cls.REQUIRED if m not in cls.__dict__]
        if missing:
            raise TypeError(f"{cls.__name__} must define methods: {', '.join(missing)}")

    def numeric_evaluation(self) -> dict[str, Any]:
        raise NotImplementedError

    def plot_evaluation(self, **kwargs) -> Any:
        raise NotImplementedError


class BaseStrategyEvaluation:

    trade_log_path = Path("trade_logs") / "trades.csv"
    fills_log_path = Path("trade_logs") / "fills.csv"
    REQUIRED = ("trade_analysis", "fills_analysis", "plots_analysis")


    def __init__(self, base_dir: str ,trade_log_path: Optional[Path] = None, fills_log_path: Optional[Path] = None):

        self.trade_log_path = trade_log_path or self.trade_log_path
        self.fills_log_path = fills_log_path or self.fills_log_path
        self.base_dir = Path(base_dir)

        if not self.trade_log_path.exists():
            raise FileNotFoundError(f"Trade log file not found: {self.trade_log_path}")
        if not self.fills_log_path.exists():
            raise FileNotFoundError(f"Fills log file not found: {self.fills_log_path}")

        self.trades_df = pd.read_csv(self.trade_log_path)
        self.fills_df = pd.read_csv(self.fills_log_path)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        missing = [m for m in cls.REQUIRED if m not in cls.__dict__]
        if missing:
            raise TypeError(f"{cls.__name__} must define methods: {', '.join(missing)}")
        
    def _save_plot(self, fig, name: str):

        path = self.base_dir/ "images" / f"{name}.png"

        fig.savefig(path, bbox_inches="tight", dpi=150)

        plt.close(fig)

        return str(path)
    
    def trade_analysis(self):
        """
        The method is defined as a placeholder for the trade analysis logic.

        The analysis should be designed to extract insights from the trade logs,
        such as win rate, average PnL, max drawdown, etc.

        The exact analysis should be determined by the hypothesis being tested.
        The output can be returned as a dictionary summarizing the trade analysis results.
        """
        raise NotImplementedError
    
    def fills_analysis(self):
        """
        The method is defined as a placeholder for the fills analysis logic.

        The analysis should be designed to extract insights from the fills logs,
        such as slippage, execution quality, fill latency, etc.

        The exact analysis should be determined by the hypothesis being tested.
        The output can be returned as a dictionary summarizing the fills analysis results.
        """
        raise NotImplementedError
    
    def plots_analysis(self):
        """
        The method is defined as a placeholder for the plots analysis logic.

        The plots should be designed specifically to provide visual evidence
        for the same single hypothesis being tested in `numeric_evaluation`.

        Example plots could include:
        - Equity curve with trade markers
        - Distribution of trade returns
        - Slippage over time
        - Cumulative PnL by signal

        The exact plots should be determined by the hypothesis being tested.
        """
        raise NotImplementedError

    def run(self):

        results = {}

        results["trade_analysis"] = self.trade_analysis()

        results["fills_analysis"] = self.fills_analysis()

        results["plots_analysis"] = self.plots_analysis()

        return results
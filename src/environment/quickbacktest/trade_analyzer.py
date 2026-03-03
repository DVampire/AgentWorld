# trade_analysis.py
# One-file trade analytics + plotting (matplotlib). No argparse.
# Output directory is PROVIDED externally.

from __future__ import annotations

from math import sqrt
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from scipy.stats import skew as _skew, kurtosis as _kurtosis
except Exception:
    _skew = None
    _kurtosis = None


# ============================================================
# Internal utilities
# ============================================================

def _normalize(x: pd.Series) -> pd.Series:
    """
    Min-max normalize to [0,1]. Safe for constant/all-NaN series.
    """
    x = pd.to_numeric(x, errors="coerce")
    xmin = x.min()
    xmax = x.max()
    if pd.isna(xmin) or pd.isna(xmax) or xmax == xmin:
        return x * 0.0
    return (x - xmin) / (xmax - xmin)


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _equity_curve(df: pd.DataFrame, initial: float) -> pd.Series:
    pnl = df["pnlcomm"].astype(float).to_numpy()
    eq = np.empty(len(pnl), dtype=float)
    cur = float(initial)
    for i, v in enumerate(pnl):
        cur += float(v)
        eq[i] = cur
    return pd.Series(eq, index=df.index, name="equity")


def _drawdown(eq: pd.Series) -> pd.Series:
    peak = eq.cummax()
    dd = eq - peak
    dd.name = "drawdown"
    return dd


def _profit_factor(pnl: pd.Series) -> float:
    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    return float(gp / gl) if gl > 0 else float("inf")


def _max_consecutive_losses(pnl: pd.Series) -> int:
    losses = (pnl < 0).astype(int).to_numpy()
    run = 0
    max_run = 0
    for x in losses:
        run = run + 1 if x else 0
        max_run = max(max_run, run)
    return int(max_run)


def _skewness(x: pd.Series) -> float:
    if _skew is None:
        return float("nan")
    return float(_skew(x.astype(float).to_numpy(), bias=False))


def _kurt(x: pd.Series) -> float:
    if _kurtosis is None:
        return float("nan")
    return float(_kurtosis(x.astype(float).to_numpy(), fisher=True, bias=False))


def _stats_block(s: pd.Series, prefix: str) -> Dict[str, Any]:
    """
    Common summary stats (mean/median/p25/p75/p90/p99/max).
    Returns {} if series is empty.
    """
    s = pd.to_numeric(s, errors="coerce").dropna()
    if len(s) == 0:
        return {}
    return {
        f"{prefix}mean": float(s.mean()),
        f"{prefix}median": float(s.median()),
        f"{prefix}p25": float(s.quantile(0.25)),
        f"{prefix}p75": float(s.quantile(0.75)),
        f"{prefix}p90": float(s.quantile(0.90)),
        f"{prefix}p99": float(s.quantile(0.99)),
        f"{prefix}max": float(s.max()),
        f"{prefix}count": int(len(s)),
    }


# ============================================================
# Core Analyzer
# ============================================================

@dataclass
class TradeAnalyzer:
    """
    Expected columns (trades.csv):
      - dt_open, dt_close (optional but recommended)
      - barlen (optional)
      - pnlcomm (required)
      - pnl, commission, is_liq (optional)
    """
    df: pd.DataFrame

    def __post_init__(self):
        if "pnlcomm" not in self.df.columns:
            raise ValueError("Missing required column: pnlcomm")
        self.df = self.df.copy().reset_index(drop=True)

        # best-effort numeric casting
        for col in ["barlen", "pnl", "pnlcomm", "commission", "is_liq"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        if "is_liq" in self.df.columns:
            self.df["is_liq"] = self.df["is_liq"].fillna(0).astype(int)

        # parse datetimes if present
        if "dt_open" in self.df.columns:
            self.df["dt_open"] = pd.to_datetime(self.df["dt_open"], errors="coerce")
        if "dt_close" in self.df.columns:
            self.df["dt_close"] = pd.to_datetime(self.df["dt_close"], errors="coerce")

        # create holding_seconds if possible
        # if "dt_open" in self.df.columns and "dt_close" in self.df.columns:
        #     dur = (self.df["dt_close"] - self.df["dt_open"]).dt.total_seconds()
        #     self.df["holding_seconds"] = pd.to_numeric(dur, errors="coerce")

    # ---------------- metrics ----------------

    def n_trades(self) -> int:
        return int(len(self.df))

    def win_rate(self) -> float:
        return float((self.df["pnlcomm"] > 0).mean())

    def avg_win(self) -> float:
        s = self.df.loc[self.df["pnlcomm"] > 0, "pnlcomm"]
        return float(s.mean()) if len(s) else float("nan")

    def avg_loss(self) -> float:
        s = self.df.loc[self.df["pnlcomm"] < 0, "pnlcomm"]
        return float(s.mean()) if len(s) else float("nan")

    def profit_factor(self) -> float:
        return _profit_factor(self.df["pnlcomm"].fillna(0.0))

    def expectancy(self) -> float:
        w = self.win_rate()
        aw = self.avg_win()
        al = self.avg_loss()
        if np.isnan(aw) or np.isnan(al):
            return float("nan")
        return float(w * aw + (1.0 - w) * al)

    def max_consecutive_losses(self) -> int:
        return _max_consecutive_losses(self.df["pnlcomm"].fillna(0.0))

    def pnl_skew(self) -> float:
        return _skewness(self.df["pnlcomm"].dropna())

    def pnl_kurtosis(self) -> float:
        return _kurt(self.df["pnlcomm"].dropna())

    def pnl_autocorr(self, lag: int = 1) -> float:
        return float(self.df["pnlcomm"].autocorr(lag))

    def liquidation_ratio(self) -> float:
        if "is_liq" not in self.df.columns:
            return 0.0
        return float(self.df["is_liq"].mean())

    # ---------------- equity / drawdown ----------------

    def equity_curve(self, initial: float = 1.0) -> pd.Series:
        return _equity_curve(self.df, initial=initial)

    def drawdown_series(self, initial: float = 1.0) -> pd.Series:
        return _drawdown(self.equity_curve(initial=initial))

    def max_drawdown(self, initial: float = 1.0) -> float:
        dd = self.drawdown_series(initial=initial)
        return float(dd.min()) if len(dd) else 0.0

    # ---------------- Kelly (heuristic) ----------------

    def kelly_fraction(self) -> float:
        w = self.win_rate()
        aw = self.avg_win()
        al = self.avg_loss()
        if np.isnan(aw) or np.isnan(al) or al == 0:
            return 0.0
        b = abs(aw / al)
        return float(w - (1.0 - w) / b) if b > 0 else 0.0

    # ---------------- Holding period stats ----------------

    def holding_period_stats(self) -> Dict[str, Any]:
        """
        Overall holding stats:
        - barlen-based holding
        - time-based holding (seconds/minutes/hours)
        """
        out: Dict[str, Any] = {}

        if "barlen" in self.df.columns:
            out.update(_stats_block(self.df["barlen"], "barlen_"))

        # if "holding_seconds" in self.df.columns:
        #     t = self.df["holding_seconds"]
        #     out.update(_stats_block(t, "hold_sec_"))

        #     # convenience means
        #     t_clean = pd.to_numeric(t, errors="coerce").dropna()
        #     if len(t_clean):
        #         out["hold_min_mean"] = float(t_clean.mean() / 60.0)
        #         out["hold_hour_mean"] = float(t_clean.mean() / 3600.0)

        return out

    def holding_period_stats_by_outcome(self) -> Dict[str, Any]:
        """
        Holding period grouped by outcome (win vs loss), for both:
        - barlen
        - holding_seconds (and minutes)
        """
        out: Dict[str, Any] = {}

        win_mask = self.df["pnlcomm"] > 0
        loss_mask = self.df["pnlcomm"] < 0

        # barlen grouped
        if "barlen" in self.df.columns:
            out.update(_stats_block(self.df.loc[win_mask, "barlen"], "barlen_win_"))
            out.update(_stats_block(self.df.loc[loss_mask, "barlen"], "barlen_loss_"))

        # # time grouped
        # if "holding_seconds" in self.df.columns:
        #     win_sec = self.df.loc[win_mask, "holding_seconds"]
        #     loss_sec = self.df.loc[loss_mask, "holding_seconds"]

        #     out.update(_stats_block(win_sec, "hold_sec_win_"))
        #     out.update(_stats_block(loss_sec, "hold_sec_loss_"))

        #     # convenience means in minutes
        #     w = pd.to_numeric(win_sec, errors="coerce").dropna()
        #     l = pd.to_numeric(loss_sec, errors="coerce").dropna()
        #     if len(w):
        #         out["hold_min_win_mean"] = float(w.mean() / 60.0)
        #     if len(l):
        #         out["hold_min_loss_mean"] = float(l.mean() / 60.0)

        return out

    # ---------------- Extreme trades ----------------

    def top_winners(self, n: int = 5) -> pd.DataFrame:
        return self.df.sort_values("pnlcomm", ascending=False).head(n).reset_index(drop=True)

    def top_losers(self, n: int = 5) -> pd.DataFrame:
        return self.df.sort_values("pnlcomm", ascending=True).head(n).reset_index(drop=True)

    # ---------------- report ----------------

    def report(self, initial: float = 1.0) -> Dict[str, Any]:
        rep = {
            "n_trades": self.n_trades(),
            "win_rate": self.win_rate(),
            "profit_factor": self.profit_factor(),
            # "expectancy": self.expectancy(),
            "max_consecutive_losses": self.max_consecutive_losses(),
            "liq_ratio": self.liquidation_ratio(),
            # "pnl_skew": self.pnl_skew(),
            # "pnl_kurtosis": self.pnl_kurtosis(),
            # "pnl_autocorr_lag1": self.pnl_autocorr(lag=1),
            "max_drawdown_trade_level": self.max_drawdown(initial=initial),
            # "kelly_fraction": self.kelly_fraction(),
        }

        # holding stats (overall + grouped)
        # rep.update(self.holding_period_stats())
        # rep.update(self.holding_period_stats_by_outcome())

        # top trades (values only in JSON; detailed saved separately)
        winners = self.top_winners(5)
        losers = self.top_losers(5)
        rep["top_5_profit_values"] = winners["pnlcomm"].tolist()
        rep["top_5_loss_values"] = losers["pnlcomm"].tolist()

        return rep

    # ============================================================
    # Plotting (saves to provided out_dir)
    # ============================================================

    def save_plots(
        self,
        out_dir: str | Path,
        initial: float = 1.0,
        rolling_window: int = 50,
        bins: int = 50,
        prefix: str = "",
        mark_liq: bool = True,
    ) -> Dict[str, Path]:
        """
        Save a standard set of plots into out_dir.

        Files:
          - {prefix}pnl_hist.png
          - {prefix}pnl_by_trade.png
          - {prefix}rolling_metrics.png
          - {prefix}holding_barlen_hist.png
          - {prefix}holding_minutes_hist.png
          - {prefix}holding_minutes_win_loss.png   (NEW)
          - {prefix}holding_barlen_win_loss.png    (NEW)
        """
        out_dir = _ensure_dir(out_dir)

        paths: Dict[str, Path] = {}
        pref = prefix

        liq_idx = None
        if mark_liq and "is_liq" in self.df.columns and self.df["is_liq"].sum() > 0:
            liq_idx = self.df.index[self.df["is_liq"] == 1].to_list()

        # pnl histogram
        x = self.df["pnlcomm"].dropna().astype(float).to_numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.hist(x, bins=bins)
        ax.set_title("PnL (pnlcomm) Histogram")
        ax.set_xlabel("pnlcomm")
        ax.set_ylabel("count")
        p = out_dir / f"{pref}pnl_hist.png"
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["pnl_hist"] = p

        # pnl by trade
        x = self.df["pnlcomm"].fillna(0.0).astype(float).to_numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(x)
        ax.axhline(0.0, linewidth=1)
        if liq_idx:
            for i in liq_idx:
                ax.axvline(i, linewidth=1)
        ax.set_title("PnL (pnlcomm) by Trade")
        ax.set_xlabel("Trade index")
        ax.set_ylabel("pnlcomm")
        p = out_dir / f"{pref}pnl_by_trade.png"
        fig.savefig(str(p), dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths["pnl_by_trade"] = p

            # # holding barlen histogram
            # if "barlen" in self.df.columns and self.df["barlen"].notna().any():
            #     bl = self.df["barlen"].dropna().astype(float).to_numpy()
            #     fig = plt.figure()
            #     ax = fig.add_subplot(111)
            #     ax.hist(bl, bins=bins)
            #     ax.set_title("Holding Period Histogram (barlen)")
            #     ax.set_xlabel("barlen")
            #     ax.set_ylabel("count")
            #     p = out_dir / f"{pref}holding_barlen_hist.png"
            #     fig.savefig(str(p), dpi=150, bbox_inches="tight")
            #     plt.close(fig)
            #     paths["holding_barlen_hist"] = p

        # holding grouped win/loss (barlen)
        if "barlen" in self.df.columns and self.df["barlen"].notna().any():
            win_mask = self.df["pnlcomm"] > 0
            loss_mask = self.df["pnlcomm"] < 0
            win_bl = self.df.loc[win_mask, "barlen"].dropna().astype(float).to_numpy()
            loss_bl = self.df.loc[loss_mask, "barlen"].dropna().astype(float).to_numpy()

            if len(win_bl) or len(loss_bl):
                fig = plt.figure()
                ax = fig.add_subplot(111)
                if len(win_bl):
                    ax.hist(win_bl, bins=bins, alpha=0.6, label="win (barlen)")
                if len(loss_bl):
                    ax.hist(loss_bl, bins=bins, alpha=0.6, label="loss (barlen)")
                ax.set_title("Holding Period by Outcome (barlen)")
                ax.set_xlabel("barlen")
                ax.set_ylabel("count")
                ax.legend()
                p = out_dir / f"{pref}holding_barlen_win_loss.png"
                fig.savefig(str(p), dpi=150, bbox_inches="tight")
                plt.close(fig)
                paths["holding_barlen_win_loss"] = p

        # rolling metrics: win rate / PF / normalized expectancy
        # pnl = self.df["pnlcomm"].fillna(0.0).astype(float)

        # roll_wr = pnl.rolling(rolling_window).apply(lambda s: (s > 0).mean(), raw=False)

        # def _pf(s: pd.Series) -> float:
        #     gp = s[s > 0].sum()
        #     gl = -s[s < 0].sum()
        #     return float(gp / gl) if gl > 0 else float("inf")

        # roll_pf = pnl.rolling(rolling_window).apply(_pf, raw=False)

        # def _exp(s: pd.Series) -> float:
        #     w = (s > 0).mean()
        #     aw = s[s > 0].mean()
        #     al = s[s < 0].mean()
        #     if np.isnan(aw) or np.isnan(al):
        #         return float("nan")
        #     return float(w * aw + (1.0 - w) * al)

        # roll_exp = _normalize(pnl.rolling(rolling_window).apply(_exp, raw=False))

        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(roll_wr.values, label="rolling win rate")
        # ax.plot(roll_pf.values, label="rolling PF")
        # ax.plot(roll_exp.values, label="normalized rolling expectancy")
        # if liq_idx:
        #     for i in liq_idx:
        #         ax.axvline(i, linewidth=1)
        # ax.set_title(f"Rolling Metrics (window={rolling_window})")
        # ax.set_xlabel("Trade index")
        # ax.set_ylabel("metric")
        # ax.legend()
        # p = out_dir / f"{pref}rolling_metrics.png"
        # fig.savefig(str(p), dpi=150, bbox_inches="tight")
        # plt.close(fig)
        # paths["rolling_metrics"] = p

        return paths


# ============================================================
# Convenience APIs
# ============================================================

def load_trades_csv(path: str | Path) -> TradeAnalyzer:
    df = pd.read_csv(path)
    return TradeAnalyzer(df)


def analyze_trades(
    trades_csv: str | Path,
    out_dir: str | Path,
    initial: float = 1.0,
    rolling_window: int = 50,
    prefix: str = "",
    mark_liq: bool = True,
) -> Dict[str, Any]:
    """
    One-call analysis:
      - Loads trades_csv
      - Computes report dict
      - Saves report.json and standard plots into out_dir
      - Saves detailed top5 winners/losers CSV
    """
    ta = load_trades_csv(trades_csv)
    out_dir = _ensure_dir(out_dir)

    report = ta.report(initial=initial)

    # import json
    # report_path = Path(out_dir) / f"{prefix}report.json"
    # with report_path.open("w", encoding="utf-8") as f:
    #     json.dump(report, f, ensure_ascii=False, indent=2)

    # plots
    ta.save_plots(
        out_dir=out_dir,
        initial=initial,
        rolling_window=rolling_window,
        bins=int(sqrt(len(ta.df))) if len(ta.df) > 0 else 10,
        prefix=prefix,
        mark_liq=mark_liq,
    )

    # save top 5 winners/losers detail
    # ta.top_winners(5).to_csv(Path(out_dir) / f"{prefix}top5_winners.csv", index=False)
    # ta.top_losers(5).to_csv(Path(out_dir) / f"{prefix}top5_losers.csv", index=False)

    return report
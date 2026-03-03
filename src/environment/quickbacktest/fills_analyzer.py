# fill_analyzer.py
# One-file FillAnalyzer (execution-level analytics + plots). No argparse.
# Output directory is PROVIDED externally.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt


# ============================================================
# Utilities
# ============================================================

def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _weighted_avg(values: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    v = np.asarray(values, dtype=float)
    s = w.sum()
    return float((v * w).sum() / s) if s > 0 else float("nan")


def _side_sign(side: str) -> int:
    # robust mapping: BUY -> +1, SELL -> -1
    s = str(side).strip().upper()
    return 1 if s == "BUY" else -1


# ============================================================
# FillAnalyzer
# ============================================================

@dataclass
class FillAnalyzer:
    """
    Expected columns (fills.csv):
      - dt (optional but recommended)
      - ref (optional)
      - side (BUY/SELL)
      - size (executed size; may be signed or unsigned depending on your logger)
      - price
      - value (optional)
      - commission
      - reason (optional)
      - is_liq (optional 0/1)

    Notes:
      - If your 'size' is always positive, we infer direction from 'side'.
      - If your 'size' already carries sign, we keep it but still use 'side' for grouping.
    """
    df: pd.DataFrame

    def __post_init__(self):
        required = {"side", "size", "price", "commission"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        self.df = self.df.copy().reset_index(drop=True)

        # datetime
        if "dt" in self.df.columns:
            self.df["dt"] = _to_datetime(self.df["dt"])

        # numeric
        for col in ["size", "price", "value", "commission", "is_liq", "ref"]:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")

        if "is_liq" in self.df.columns:
            self.df["is_liq"] = self.df["is_liq"].fillna(0).astype(int)

        # normalize side
        self.df["side"] = self.df["side"].astype(str).str.upper().str.strip()

        # derive signed_size for convenience
        # If size already has sign, keep it; otherwise apply side sign.
        raw_size = self.df["size"].astype(float)
        inferred = raw_size.copy()
        mask_unsigned = raw_size >= 0  # common in backtrader: executed.size is signed, but keep robust
        inferred[mask_unsigned] = inferred[mask_unsigned] * self.df.loc[mask_unsigned, "side"].map(_side_sign).astype(float)
        self.df["signed_size"] = inferred
        self.df["abs_size"] = np.abs(self.df["size"].astype(float))

        # derive notional if absent
        if "value" not in self.df.columns or self.df["value"].isna().all():
            self.df["notional"] = self.df["abs_size"] * self.df["price"].astype(float)
        else:
            self.df["notional"] = self.df["value"].astype(float).abs()

    # ========================================================
    # Core execution metrics
    # ========================================================

    def n_fills(self) -> int:
        return int(len(self.df))

    def total_volume(self) -> float:
        return float(self.df["abs_size"].sum())

    def total_notional(self) -> float:
        return float(self.df["notional"].sum())

    def total_commission(self) -> float:
        return float(self.df["commission"].sum())

    def avg_commission_per_fill(self) -> float:
        return float(self.df["commission"].mean())

    def commission_per_notional(self) -> float:
        denom = self.total_notional()
        return float(self.total_commission() / denom) if denom > 0 else float("nan")

    def vwap(self) -> float:
        # VWAP using abs_size as weights
        return _weighted_avg(self.df["price"].to_numpy(), self.df["abs_size"].to_numpy())

    def vwap_by_side(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for side in ["BUY", "SELL"]:
            sub = self.df[self.df["side"] == side]
            if len(sub) == 0:
                continue
            out[side] = _weighted_avg(sub["price"].to_numpy(), sub["abs_size"].to_numpy())
        return out

    def count_by_side(self) -> Dict[str, int]:
        return self.df["side"].value_counts().to_dict()

    def notional_by_side(self) -> Dict[str, float]:
        return self.df.groupby("side")["notional"].sum().to_dict()

    def avg_fill_size(self) -> float:
        return float(self.df["abs_size"].mean())

    def median_fill_size(self) -> float:
        return float(self.df["abs_size"].median())

    def max_fill_size(self) -> float:
        return float(self.df["abs_size"].max())

    def liquidation_fill_ratio(self) -> float:
        if "is_liq" not in self.df.columns:
            return 0.0
        return float(self.df["is_liq"].mean())

    def liquidation_fill_count(self) -> int:
        if "is_liq" not in self.df.columns:
            return 0
        return int(self.df["is_liq"].sum())

    def reason_counts(self, topn: int = 10) -> Dict[str, int]:
        if "reason" not in self.df.columns:
            return {}
        s = self.df["reason"].fillna("").astype(str)
        vc = s.value_counts()
        return vc.head(topn).to_dict()

    # ========================================================
    # Report
    # ========================================================

    def report(self) -> Dict[str, Any]:
        return {
            "n_fills": self.n_fills(),
            "count_by_side": self.count_by_side(),
            # "total_volume": self.total_volume(),
            # "total_notional": self.total_notional(),
            # "vwap": self.vwap(),
            # "vwap_by_side": self.vwap_by_side(),
            # "notional_by_side": self.notional_by_side(),
            "total_commission": self.total_commission(),
            # "avg_commission_per_fill": self.avg_commission_per_fill(),
            # "commission_per_notional": self.commission_per_notional(),
            # "avg_fill_size": self.avg_fill_size(),
            # "median_fill_size": self.median_fill_size(),
            # "max_fill_size": self.max_fill_size(),
            # "liq_fill_ratio": self.liquidation_fill_ratio(),
            # "liq_fill_count": self.liquidation_fill_count(),
            "top_reasons": self.reason_counts(topn=10),
        }

    # ========================================================
    # Plotting (saved to provided out_dir)
    # ========================================================

    def save_plots(
        self,
        out_dir: str | Path,
        bins: int = 50,
        prefix: str = "",
        mark_liq: bool = True,
    ) -> Dict[str, Path]:
        out_dir = _ensure_dir(out_dir)
        pref = prefix

        liq_idx = None
        if mark_liq and "is_liq" in self.df.columns and self.df["is_liq"].sum() > 0:
            liq_idx = self.df.index[self.df["is_liq"] == 1].to_list()

        paths: Dict[str, Path] = {}

        # # fill prices
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(self.df["price"].astype(float).to_numpy())
        # if liq_idx:
        #     for i in liq_idx:
        #         ax.axvline(i, linewidth=1)
        # ax.set_title("Fill Prices")
        # ax.set_xlabel("Fill index")
        # ax.set_ylabel("Price")
        # p = out_dir / f"{pref}fill_prices.png"
        # fig.savefig(str(p), dpi=150, bbox_inches="tight")
        # plt.close(fig)
        # paths["fill_prices"] = p

        # # commission per fill
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(self.df["commission"].astype(float).to_numpy())
        # if liq_idx:
        #     for i in liq_idx:
        #         ax.axvline(i, linewidth=1)
        # ax.set_title("Commission per Fill")
        # ax.set_xlabel("Fill index")
        # ax.set_ylabel("Commission")
        # p = out_dir / f"{pref}commission.png"
        # fig.savefig(str(p), dpi=150, bbox_inches="tight")
        # plt.close(fig)
        # paths["commission"] = p

        # # fill size histogram
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.hist(self.df["abs_size"].astype(float).to_numpy(), bins=bins)
        # ax.set_title("Fill Size Distribution (abs)")
        # ax.set_xlabel("abs(size)")
        # ax.set_ylabel("count")
        # p = out_dir / f"{pref}fill_size_hist.png"
        # fig.savefig(str(p), dpi=150, bbox_inches="tight")
        # plt.close(fig)
        # paths["fill_size_hist"] = p

        # # notional histogram
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.hist(self.df["notional"].astype(float).to_numpy(), bins=bins)
        # ax.set_title("Fill Notional Distribution")
        # ax.set_xlabel("notional")
        # ax.set_ylabel("count")
        # p = out_dir / f"{pref}fill_notional_hist.png"
        # fig.savefig(str(p), dpi=150, bbox_inches="tight")
        # plt.close(fig)
        # paths["fill_notional_hist"] = p

        # vwap by side bar
        # vwap_side = self.vwap_by_side()
        # if vwap_side:
        #     fig = plt.figure()
        #     ax = fig.add_subplot(111)
        #     sides = list(vwap_side.keys())
        #     vals = [vwap_side[s] for s in sides]
        #     ax.bar(sides, vals)
        #     ax.set_title("VWAP by Side")
        #     ax.set_xlabel("side")
        #     ax.set_ylabel("VWAP")
        #     ax.set_yscale("log")
        #     p = out_dir / f"{pref}vwap_by_side.png"
        #     fig.savefig(str(p), dpi=150, bbox_inches="tight")
        #     plt.close(fig)
        #     paths["vwap_by_side"] = p

        return paths


# ============================================================
# Convenience APIs
# ============================================================

def load_fills_csv(path: str | Path) -> FillAnalyzer:
    df = pd.read_csv(path)
    return FillAnalyzer(df)


def analyze_fills(
    fills_csv: str | Path,
    out_dir: str | Path,
    prefix: str = "",
    mark_liq: bool = True,
) -> Dict[str, Any]:
    """
    One-call:
      - Load fills.csv
      - Compute report dict
      - Save report.json + plots into out_dir
    """
    fa = load_fills_csv(fills_csv)
    out_dir = _ensure_dir(out_dir)

    report = fa.report()

    # import json
    # report_path = Path(out_dir) / f"{prefix}fill_report.json"
    # with report_path.open("w", encoding="utf-8") as f:
    #     json.dump(report, f, ensure_ascii=False, indent=2)

    fa.save_plots(out_dir=out_dir, bins=int(sqrt(len(fa.df))) if len(fa.df) > 0 else 10, prefix=prefix, mark_liq=mark_liq)

    return report
"""LLM Agent Trading Strategy.

This strategy uses an LLM to analyze market conditions and select the most appropriate
traditional strategy (TSMOM, Adaptive Trend Fusion, or ZScoreMR) for trading.
"""

import os
import sys
import json
import math
import numpy as np
import pandas as pd
import asyncio
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Any
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path
from mmengine import DictAction

# Enable nested event loops for calling async LLM from sync backtest loop
try:
    import nest_asyncio
    nest_asyncio.apply()
except ImportError:
    # nest_asyncio not available, will use fallback method
    pass

# Add project root to path
root = str(Path(__file__).resolve().parents[3])
sys.path.append(root)

from src.model.manager import model_manager
from src.message import HumanMessage, SystemMessage
from src.logger import logger
from src.config import config

PRICE_COL = "c"
VOL_COL = "v"
MAX_LEVERAGE = 5.0

# ============================================================
#          Traditional Strategy Classes
# ============================================================

class TSMOMStrategy:
    """Time Series Momentum Strategy with trend filtering."""
    
    def __init__(self):
        self.last_side = 0
        self.hold_bars = 0
    
    def _calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate SMA."""
        return series.rolling(window=period).mean()
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_pos: float,
        current_equity: float,
        min_hold_bars: int = 0,
        lookback: int = 50,
        threshold: float = 0.001,  # Default 0.1% threshold to filter noise
        trend_filter_period: int = 200,  # Use SMA200 for trend filtering
    ) -> float:
        """Generate trading signal with trend filtering.
        
        Args:
            threshold: Minimum return threshold to trigger signal (default 0.1% to filter noise)
            trend_filter_period: Period for trend SMA filter (default 200)
        """
        if df is None or df.empty or current_equity <= 0 or PRICE_COL not in df.columns:
            return 0.0
        
        closes = df[PRICE_COL].astype(float)
        
        # Need enough data for both lookback and trend filter
        min_required = max(lookback + 1, trend_filter_period)
        if len(df) < min_required:
            return 0.0
        
        p_now = closes.iloc[-1]
        p_past = closes.iloc[-(lookback + 1)]
        
        if np.isnan(p_now) or np.isnan(p_past) or p_past <= 0:
            return 0.0
        
        # Calculate momentum return
        ret = p_now / p_past - 1.0
        
        # Calculate trend filter (SMA)
        sma_trend = self._calculate_sma(closes, trend_filter_period).iloc[-1]
        if np.isnan(sma_trend):
            return 0.0
        
        # Determine trend direction
        # Price > SMA * 1.001: uptrend (allow long)
        # Price < SMA * 0.999: downtrend (allow short)
        trend_allows_long = p_now > sma_trend * 1.001
        trend_allows_short = p_now < sma_trend * 0.999
        
        # Generate raw signal based on momentum
        if ret > threshold:
            raw_side = 1.0
        elif ret < -threshold:
            raw_side = -1.0
        else:
            raw_side = 0.0
        
        # Apply trend filter: only trade in the direction of the trend
        if raw_side > 0 and not trend_allows_long:
            raw_side = 0.0  # Don't go long in downtrend
        elif raw_side < 0 and not trend_allows_short:
            raw_side = 0.0  # Don't go short in uptrend
        
        curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
        
        if curr_side == 0:
            self.hold_bars = 0
        else:
            if curr_side == self.last_side:
                self.hold_bars += 1
            else:
                self.hold_bars = 1
                self.last_side = curr_side
        
        if min_hold_bars > 0 and curr_side != 0:
            if raw_side != curr_side and self.hold_bars < min_hold_bars:
                return float(curr_side)
        
        return raw_side


class ZScoreMRStrategy:
    """Z-score Mean Reversion Strategy with trend filtering."""
    
    def __init__(self):
        self.last_side = 0
        self.hold_bars = 0
    
    def _calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate SMA."""
        return series.rolling(window=period).mean()
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_pos: float,
        current_equity: float,
        min_hold_bars: int = 0,
        window: int = 40,
        entry_z: float = 2.0,  # Increased default from 1.5 to 2.0 for better signal quality
        exit_z: float = 0.3,
        trend_filter_period: int = 200,  # Use SMA200 for trend filtering
        trend_strength_threshold: float = 0.005,  # 0.5% threshold for strong trend detection
    ) -> float:
        """Generate trading signal with trend filtering.
        
        Mean reversion works best in sideways/weak trend markets.
        In strong trends, mean reversion can be dangerous.
        
        Args:
            entry_z: Z-score threshold for entry (default 2.0, increased for better quality)
            trend_filter_period: Period for trend SMA filter (default 200)
            trend_strength_threshold: Threshold to detect strong trend (default 0.5%)
        """
        if df is None or df.empty or current_equity <= 0 or PRICE_COL not in df.columns:
            return 0.0
        
        closes = df[PRICE_COL].astype(float)
        
        # Need enough data for both window and trend filter
        min_required = max(window, trend_filter_period)
        if len(df) < min_required:
            return 0.0
        
        ma = closes.rolling(window).mean()
        std = closes.rolling(window).std(ddof=0)
        
        mu = ma.iloc[-1]
        sigma = std.iloc[-1]
        p = closes.iloc[-1]
        
        if np.isnan(mu) or np.isnan(sigma) or sigma < 1e-12:
            return 0.0
        
        # Calculate trend filter
        sma_trend = self._calculate_sma(closes, trend_filter_period).iloc[-1]
        if np.isnan(sma_trend):
            return 0.0
        
        # Detect trend strength: how far is price from SMA?
        trend_deviation = abs(p - sma_trend) / sma_trend if sma_trend > 0 else 0
        
        # Calculate z-score
        z = (p - mu) / sigma
        
        # Generate raw signal based on z-score
        if abs(z) < exit_z:
            raw_side = 0.0
        elif z > entry_z:
            raw_side = -1.0  # Price too high, mean revert short
        elif z < -entry_z:
            raw_side = 1.0   # Price too low, mean revert long
        else:
            raw_side = float(current_pos)
        
        # Apply trend filter: avoid mean reversion in strong trends
        # If trend is strong (price far from SMA), disable mean reversion trades
        if trend_deviation > trend_strength_threshold:
            # Strong trend detected: only allow mean reversion trades that align with trend
            # In uptrend (price > SMA), only allow long mean reversion (buy dips)
            # In downtrend (price < SMA), only allow short mean reversion (sell rallies)
            if p > sma_trend:
                # Uptrend: only allow long positions (buying dips)
                if raw_side < 0:
                    raw_side = 0.0  # Don't short in strong uptrend
            elif p < sma_trend:
                # Downtrend: only allow short positions (selling rallies)
                if raw_side > 0:
                    raw_side = 0.0  # Don't long in strong downtrend
        
        curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
        
        if curr_side == 0:
            self.hold_bars = 0
        else:
            if curr_side == self.last_side:
                self.hold_bars += 1
            else:
                self.hold_bars = 1
                self.last_side = curr_side
        
        if min_hold_bars > 0 and curr_side != 0:
            if (raw_side == 0.0 or np.sign(raw_side) != curr_side) and self.hold_bars < min_hold_bars:
                return float(curr_side)
        
        return raw_side


class AdaptiveTrendFusionStrategy:
    """Adaptive Trend Fusion Strategy (combination of multiple indicators)."""
    
    def __init__(self):
        self.last_side = 0
        self.hold_bars = 0
        self.last_ema20 = None
        self.last_ema50 = None
    
    def _calculate_ema(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate EMA."""
        return series.ewm(span=period, adjust=False).mean()
    
    def _calculate_sma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate SMA."""
        return series.rolling(window=period).mean()
    
    def _calculate_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ATR."""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def _calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate ADX."""
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr1 = high - low
        tr2 = np.abs(high - close.shift())
        tr3 = np.abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        atr_safe = atr.replace(0, np.nan)
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_safe)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_safe)
        
        di_sum = plus_di + minus_di
        di_sum = di_sum.replace(0, np.nan)
        dx = 100 * np.abs(plus_di - minus_di) / di_sum
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def _find_swing_points(self, high: pd.Series, low: pd.Series, lookback: int = 20) -> Tuple[Optional[float], Optional[float]]:
        """Find swing points."""
        if len(high) < lookback * 2:
            return None, None
        
        recent_high = high.iloc[-lookback:].max()
        recent_low = low.iloc[-lookback:].min()
        
        return recent_high, recent_low
    
    def _check_volume_surge(self, volumes: pd.Series, current_vol: float, lookback: int = 20, multiplier: float = 1.5) -> bool:
        """Check volume surge."""
        if len(volumes) < lookback:
            return False
        
        avg_vol = volumes.iloc[-lookback:].mean()
        if avg_vol <= 0:
            return False
        
        return current_vol >= avg_vol * multiplier
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        current_pos: float,
        current_equity: float,
        min_hold_bars: int = 0,
        sma_trend_period: int = 200,
        ema_fast: int = 20,
        ema_slow: int = 50,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        rsi_period: int = 14,
        rsi_oversold: float = 40.0,
        rsi_overbought: float = 60.0,
        swing_lookback: int = 20,
        vol_surge_lookback: int = 20,
        vol_surge_multiplier: float = 1.5,
        atr_period: int = 14,
    ) -> float:
        """Generate trading signal."""
        if df is None or df.empty or current_equity <= 0:
            return 0.0
        
        if PRICE_COL not in df.columns or VOL_COL not in df.columns:
            return 0.0
        
        max_period = max(sma_trend_period, ema_slow, adx_period, rsi_period, atr_period, swing_lookback)
        if len(df) < max_period + 10:
            return 0.0
        
        closes = df[PRICE_COL].astype(float)
        highs = df["h"].astype(float) if "h" in df.columns else closes
        lows = df["l"].astype(float) if "l" in df.columns else closes
        volumes = df[VOL_COL].astype(float)
        
        current_price = closes.iloc[-1]
        current_vol = volumes.iloc[-1]
        
        sma_trend = self._calculate_sma(closes, sma_trend_period).iloc[-1]
        ema_fast_val = self._calculate_ema(closes, ema_fast).iloc[-1]
        ema_slow_val = self._calculate_ema(closes, ema_slow).iloc[-1]
        
        rsi = self._calculate_rsi(closes, rsi_period).iloc[-1]
        adx = self._calculate_adx(highs, lows, closes, adx_period).iloc[-1]
        atr = self._calculate_atr(highs, lows, closes, atr_period).iloc[-1]
        
        if (np.isnan(sma_trend) or np.isnan(ema_fast_val) or np.isnan(ema_slow_val) or 
            np.isnan(rsi) or np.isnan(adx) or np.isnan(atr)):
            curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
            if min_hold_bars > 0 and curr_side != 0:
                if curr_side == self.last_side:
                    self.hold_bars += 1
                else:
                    self.hold_bars = 1
                    self.last_side = curr_side
                if self.hold_bars < min_hold_bars:
                    return float(curr_side)
            return 0.0
        
        adx_filter_active = adx <= adx_threshold
        
        trend_direction = 0
        if current_price > sma_trend * 1.001:
            trend_direction = 1
        elif current_price < sma_trend * 0.999:
            trend_direction = -1
        
        ema_cross_signal = 0
        if self.last_ema20 is not None and self.last_ema50 is not None:
            prev_fast_above_slow = self.last_ema20 > self.last_ema50
            curr_fast_above_slow = ema_fast_val > ema_slow_val
            
            if not prev_fast_above_slow and curr_fast_above_slow:
                ema_cross_signal = 1.0
            elif prev_fast_above_slow and not curr_fast_above_slow:
                ema_cross_signal = -1.0
        else:
            if ema_fast_val > ema_slow_val:
                ema_cross_signal = 1.0
            elif ema_fast_val < ema_slow_val:
                ema_cross_signal = -1.0
        
        self.last_ema20 = ema_fast_val
        self.last_ema50 = ema_slow_val
        
        momentum_signal = 0
        if not adx_filter_active:
            swing_high, swing_low = self._find_swing_points(highs, lows, swing_lookback)
            if swing_high is not None and swing_low is not None:
                volume_surge = self._check_volume_surge(volumes, current_vol, vol_surge_lookback, vol_surge_multiplier)
                
                if current_price > swing_high * 1.0005 and volume_surge:
                    momentum_signal = 1.0
                elif current_price < swing_low * 0.9995 and volume_surge:
                    momentum_signal = -1.0
        
        pullback_signal = 0
        if not adx_filter_active:
            price_to_ema20 = (current_price - ema_fast_val) / ema_fast_val if ema_fast_val > 0 else 0
            
            if trend_direction >= 0:
                if -0.005 <= price_to_ema20 <= 0.01 and rsi < rsi_oversold:
                    pullback_signal = 1.0
            
            if trend_direction <= 0:
                if -0.01 <= price_to_ema20 <= 0.005 and rsi > rsi_overbought:
                    pullback_signal = -1.0
        
        raw_side = 0.0
        
        if not adx_filter_active:
            if ema_cross_signal != 0:
                raw_side = ema_cross_signal
            elif momentum_signal != 0:
                raw_side = momentum_signal
            elif pullback_signal != 0:
                raw_side = pullback_signal
        
        if trend_direction == 1 and raw_side < 0:
            raw_side = 0.0
        elif trend_direction == -1 and raw_side > 0:
            raw_side = 0.0
        
        curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
        
        if curr_side == 0:
            self.hold_bars = 0
            self.last_side = 0
        else:
            if curr_side == self.last_side:
                self.hold_bars += 1
            else:
                self.hold_bars = 1
                self.last_side = curr_side
        
        if min_hold_bars > 0 and curr_side != 0:
            if (raw_side == 0.0 or np.sign(raw_side) != curr_side) and self.hold_bars < min_hold_bars:
                return float(curr_side)
        
        if adx_filter_active and curr_side == 0:
            return 0.0
        
        return raw_side


# ============================================================
#          Factor Calculation Functions
# ============================================================

def calculate_factors(df: pd.DataFrame, long_window: int = 200, short_window: int = 50) -> Dict[str, Any]:
    """Calculate minute-level factors across longer time scales.
    
    Args:
        df: DataFrame with price data
        long_window: Long-term window for trend analysis
        short_window: Short-term window for momentum
        
    Returns:
        Dictionary containing calculated factors
    """
    if df is None or df.empty or PRICE_COL not in df.columns:
        return {}
    
    closes = df[PRICE_COL].astype(float)
    volumes = df[VOL_COL].astype(float) if VOL_COL in df.columns else pd.Series()
    
    factors = {}
    
    # Price-based factors
    if len(closes) >= long_window:
        factors['long_ma'] = closes.rolling(long_window).mean().iloc[-1]
        factors['long_std'] = closes.rolling(long_window).std().iloc[-1]
        factors['price_to_long_ma'] = (closes.iloc[-1] - factors['long_ma']) / factors['long_ma'] if factors['long_ma'] > 0 else 0
    
    if len(closes) >= short_window:
        factors['short_ma'] = closes.rolling(short_window).mean().iloc[-1]
        factors['short_std'] = closes.rolling(short_window).std().iloc[-1]
        factors['price_to_short_ma'] = (closes.iloc[-1] - factors['short_ma']) / factors['short_ma'] if factors['short_ma'] > 0 else 0
    
    # Momentum factors
    if len(closes) >= 20:
        factors['momentum_20'] = (closes.iloc[-1] / closes.iloc[-20] - 1) if closes.iloc[-20] > 0 else 0
    if len(closes) >= 50:
        factors['momentum_50'] = (closes.iloc[-1] / closes.iloc[-50] - 1) if closes.iloc[-50] > 0 else 0
    if len(closes) >= 100:
        factors['momentum_100'] = (closes.iloc[-1] / closes.iloc[-100] - 1) if closes.iloc[-100] > 0 else 0
    
    # Volatility factors
    if len(closes) >= 20:
        returns = closes.pct_change().dropna()
        if len(returns) >= 20:
            factors['volatility_20'] = returns.iloc[-20:].std()
            factors['volatility_50'] = returns.iloc[-50:].std() if len(returns) >= 50 else factors['volatility_20']
    
    # Volume factors
    if len(volumes) >= 20:
        factors['volume_ma_20'] = volumes.rolling(20).mean().iloc[-1]
        factors['volume_ratio'] = volumes.iloc[-1] / factors['volume_ma_20'] if factors['volume_ma_20'] > 0 else 1
    
    # Trend strength
    if 'long_ma' in factors and 'short_ma' in factors:
        factors['trend_strength'] = (factors['short_ma'] - factors['long_ma']) / factors['long_ma'] if factors['long_ma'] > 0 else 0
    
    # Current price
    factors['current_price'] = closes.iloc[-1]
    factors['high_20'] = closes.iloc[-20:].max() if len(closes) >= 20 else closes.iloc[-1]
    factors['low_20'] = closes.iloc[-20:].min() if len(closes) >= 20 else closes.iloc[-1]
    
    return factors


# ============================================================
#          Pydantic Models for Response Format
# ============================================================

class TSMOMHyperparameters(BaseModel):
    """TSMOM strategy hyperparameters."""
    min_hold_bars: int = Field(
        default=5,
        description="Minimum bars to hold position before allowing exit/reverse. REQUIRED. "
        "For volatile markets: 10-20, normal markets: 5-15, stable trends: 3-10. NEVER set to 0.",
        ge=0,
        le=100
    )
    lookback: Optional[int] = Field(
        default=50,
        description="Lookback period for momentum calculation (default: 50, range: 20-200)",
        ge=20,
        le=200
    )
    threshold: Optional[float] = Field(
        default=0.0,
        description="Momentum threshold for signal generation (default: 0.0, range: 0.0-0.01)",
        ge=0.0,
        le=0.01
    )


class ZScoreMRHyperparameters(BaseModel):
    """Z-Score Mean Reversion strategy hyperparameters."""
    min_hold_bars: int = Field(
        default=5,
        description="Minimum bars to hold position before allowing exit/reverse. REQUIRED. "
        "For volatile markets: 10-20, normal markets: 5-15, stable trends: 3-10. NEVER set to 0.",
        ge=0,
        le=100
    )
    window: Optional[int] = Field(
        default=40,
        description="Window size for moving average and standard deviation (default: 40, range: 20-100)",
        ge=20,
        le=100
    )
    entry_z: Optional[float] = Field(
        default=1.5,
        description="Z-score threshold for entry signal (default: 1.5, range: 1.0-3.0)",
        ge=1.0,
        le=3.0
    )
    exit_z: Optional[float] = Field(
        default=None,
        description="Z-score threshold for exit signal (default: 20% of entry_z if not provided)",
        ge=0.1,
        le=1.0
    )


class AdaptiveTrendFusionHyperparameters(BaseModel):
    """Adaptive Trend Fusion strategy hyperparameters."""
    min_hold_bars: int = Field(
        default=5,
        description="Minimum bars to hold position before allowing exit/reverse. REQUIRED. "
        "For volatile markets: 10-20, normal markets: 5-15, stable trends: 3-10. NEVER set to 0.",
        ge=0,
        le=100
    )
    ema_fast: Optional[int] = Field(
        default=20,
        description="Fast EMA period (default: 20, range: 5-50)",
        ge=5,
        le=50
    )
    adx_threshold: Optional[float] = Field(
        default=25.0,
        description="ADX threshold for trend filter (default: 25.0, range: 15.0-35.0)",
        ge=15.0,
        le=35.0
    )
    rsi_threshold: Optional[float] = Field(
        default=50.0,
        description="RSI threshold for oversold/overbought detection (default: 50.0, range: 30.0-70.0)",
        ge=30.0,
        le=70.0
    )


class MarketAnalysis(BaseModel):
    """Market analysis response model."""
    trend: str = Field(description="Current market trend: bullish, bearish, or sideways")
    volatility: str = Field(description="Market volatility level: high, medium, or low")
    expected_movement: str = Field(description="Expected price movement in the next trading period")
    recommended_strategy: str = Field(description="Recommended trading strategy: tsmom, zscore_mr, or adaptive_trend_fusion")
    trading_duration_minutes: int = Field(description="How many minutes to trade with this strategy (should be 300-600 minutes for minute-level trading)", ge=300, le=600, default=450)
    strategy_hyperparameters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Core hyperparameters for the selected strategy. "
        "For 'tsmom': provide TSMOMHyperparameters with min_hold_bars (REQUIRED), lookback, threshold. "
        "For 'zscore_mr': provide ZScoreMRHyperparameters with min_hold_bars (REQUIRED), window, entry_z, exit_z. "
        "For 'adaptive_trend_fusion': provide AdaptiveTrendFusionHyperparameters with min_hold_bars (REQUIRED), ema_fast, adx_threshold, rsi_threshold. "
        "min_hold_bars is REQUIRED for all strategies to prevent excessive trading."
    )
    reasoning: str = Field(description="Reasoning for the recommendation, duration, and hyperparameters")


class TradingResultAnalysis(BaseModel):
    """Trading result analysis response model."""
    strategy_appropriate: bool = Field(description="Whether the strategy selection was appropriate")
    performance_analysis: str = Field(description="Analysis of what went well or poorly")
    adjustments: str = Field(description="What should be adjusted for the next trading period")
    lessons_learned: str = Field(description="Key lessons learned from this trading period")


class PerformanceAnalysis(BaseModel):
    """Performance analysis response model."""
    performance_assessment: str = Field(description="Overall performance assessment")
    strategy_effectiveness: str = Field(description="Analysis of strategy selection effectiveness")
    recommendations: str = Field(description="Recommendations for improvement")
    insights: str = Field(description="Key insights and lessons learned")


# ============================================================
#          LLM Agent Strategy
# ============================================================

class AgentStrategy:
    """LLM Agent Trading Strategy.
    
    This strategy uses an LLM to:
    1. Analyze minute-level factors and trends
    2. Predict future market conditions
    3. Select the most appropriate traditional strategy
    4. Execute trades and analyze results
    5. Iterate until trading ends
    """
    
    def __init__(
        self,
        model_name: str = "openai/gemini-3-flash-preview",
        data_path: Optional[str] = None,
        trading_horizon: int = 60,  # minutes to trade ahead
        min_hold_bars: int = 450,
    ):
        """Initialize the agent strategy.
        
        Args:
            model_name: Name of the LLM model to use
            data_path: Path to the data directory (default: workdir/crypto/crypto_binance_price_1min for minute-level trading)
            trading_horizon: Number of minutes to trade ahead
            min_hold_bars: Minimum bars to hold a position
        """
        self.model_name = model_name
        self.trading_horizon = trading_horizon
        self.min_hold_bars = min_hold_bars
        
        # Initialize traditional strategies
        self.tsmom = TSMOMStrategy()
        self.zscore_mr = ZScoreMRStrategy()
        self.adaptive_trend_fusion = AdaptiveTrendFusionStrategy()
        
        # Set data path
        if data_path is None:
            # Default to workdir/crypto/crypto_binance_price_1min for minute-level trading
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
            data_path = os.path.join(project_root, "workdir", "crypto", "crypto_binance_price_1min")
        self.data_path = data_path
        
        # Trading state
        self.current_strategy: Optional[str] = None
        self.trading_history: List[Dict[str, Any]] = []
        self.factor_history: List[Dict[str, Any]] = []
        
        # Strategy execution state
        self.active_strategy: Optional[str] = None  # Currently executing strategy
        self.strategy_start_bar_index: Optional[int] = None  # Bar index when strategy started
        self.strategy_duration_bars: Optional[int] = None  # Duration in bars
        self.strategy_hyperparameters: Dict[str, Any] = {}  # Hyperparameters for active strategy
        self.strategy_period_history: List[Dict[str, Any]] = []  # History of strategy periods
        self.strategy_start_equity: Optional[float] = None  # Equity when strategy started (for stop-loss)
        self.max_loss_threshold: float = -0.05  # Stop-loss threshold: -5% per strategy period
    
    def _load_data(self, symbol: str = "BTCUSDT") -> pd.DataFrame:
        """Load data from JSONL file.
        
        Args:
            symbol: Trading pair symbol (default: BTCUSDT)
            
        Returns:
            DataFrame with columns: t, o, h, l, c, v
        """
        file_path = os.path.join(self.data_path, f"{symbol}.jsonl")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        df = pd.read_json(file_path, lines=True)
        
        # Rename columns to match backtest format
        column_map = {
            "timestamp": "t",
            "open": "o",
            "high": "h",
            "low": "l",
            "close": "c",
            "volume": "v",
        }
        
        df = df.rename(columns=column_map)
        
        # Ensure timestamp is datetime
        if "t" in df.columns:
            df["t"] = pd.to_datetime(df["t"])
            df = df.sort_values("t").reset_index(drop=True)
        
        return df
    
    async def _analyze_market(self, df: pd.DataFrame, factors: Dict[str, Any], previous_period_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Use LLM to analyze market conditions and predict future trends.
        
        Args:
            df: Historical price data
            factors: Calculated factors
            previous_period_info: Optional information about the previous strategy period
            
        Returns:
            Dictionary containing market analysis and prediction
        """
        
        # Prepare data summary for LLM
        recent_data = df.tail(100) if len(df) >= 100 else df
        data_summary = {
            "current_price": factors.get('current_price', 0),
            "price_range_20": {
                "high": factors.get('high_20', 0),
                "low": factors.get('low_20', 0),
            },
            "momentum": {
                "20_period": factors.get('momentum_20', 0),
                "50_period": factors.get('momentum_50', 0),
                "100_period": factors.get('momentum_100', 0),
            },
            "trend": {
                "strength": factors.get('trend_strength', 0),
                "price_to_long_ma": factors.get('price_to_long_ma', 0),
                "price_to_short_ma": factors.get('price_to_short_ma', 0),
            },
            "volatility": {
                "20_period": factors.get('volatility_20', 0),
                "50_period": factors.get('volatility_50', 0),
            },
            "volume": {
                "ratio": factors.get('volume_ratio', 1),
            },
        }
        
        # Prepare prompt
        previous_period_text = ""
        if previous_period_info:
            previous_period_text = f"""
Previous Strategy Period Results:
{json.dumps(previous_period_info, indent=2)}

Consider the performance of the previous strategy period when selecting the next strategy.
If the previous strategy performed well, you may want to continue with it or a similar strategy.
If it performed poorly, consider switching to a different strategy that might be more suitable for current market conditions.
"""
        
        prompt = f"""You are a quantitative trading analyst. Analyze the following market data and provide insights.

**CRITICAL: RISK MANAGEMENT IS PARAMOUNT**
- Each strategy period has a stop-loss at -5%: if a strategy loses more than 5%, it will be automatically stopped
- Your goal is to preserve capital and avoid large losses
- If previous strategy lost money, you MUST be more conservative and consider switching strategies
- If previous strategy lost more than 5%, you MUST switch to a different strategy

Market Data Summary:
{json.dumps(data_summary, indent=2)}

Recent Price History (last 10 bars):
{recent_data[['t', 'c', 'v']].tail(10).to_string()}
{previous_period_text}

Please analyze:
1. Current market trend (bullish, bearish, or sideways)
2. Market volatility level (high, medium, low)
3. Expected price movement in the next period
4. Recommended trading strategy - **CRITICAL: Choose based on market characteristics**:

   **"tsmom" - Time Series Momentum Strategy (Trend-Following)**
   - **How it works**: Compares current price to price N bars ago. If price increased above threshold, go long; if decreased below threshold, go short.
   - **Best for**: Strong directional trends, trending markets with clear momentum
   - **Avoid when**: Sideways/choppy markets, high volatility without clear direction, mean-reverting conditions
   - **Key indicators**: High trend_strength, clear momentum_20 direction, stable volatility
   - **Core hyperparameters** (optional, max 3): 
     * lookback (int, default 50, range 20-200): Number of bars to look back. Higher for longer trends.
     * threshold (float, default 0.0, range 0.0-0.01): Minimum return to trigger signal. Higher = fewer but stronger signals.
     * min_hold_bars (int, REQUIRED, range 80-95% of trading_duration_minutes): Minimum bars to hold position to reduce whipsaws (e.g., 360-430 for 450 minutes duration)

   **"zscore_mr" - Z-Score Mean Reversion Strategy (Range-Bound Trading)**
   - **How it works**: Calculates Z-score of price relative to rolling mean/std. When price deviates significantly (high Z-score), trades against the deviation expecting mean reversion.
   - **Best for**: Range-bound markets, sideways/oscillating price action, low volatility periods, mean-reverting conditions
   - **Avoid when**: Strong trends, breakout markets, trending momentum
   - **Key indicators**: Low volatility, sideways trend_strength, price oscillating around mean
   - **Core hyperparameters** (optional, max 3):
     * window (int, default 40, range 20-100): Rolling window for mean/std calculation. Smaller = more sensitive.
     * entry_z (float, default 1.5, range 1.0-3.0): Z-score threshold to enter trade. Higher = fewer but stronger signals.
     * min_hold_bars (int, REQUIRED, range 80-95% of trading_duration_minutes): Minimum bars to hold position (e.g., 360-430 for 450 minutes duration)

   **"adaptive_trend_fusion" - Adaptive Trend Fusion Strategy (Multi-Indicator Combination)**
   - **How it works**: Combines EMA crossovers, ADX (trend strength), RSI (momentum), ATR (volatility), volume surges, and swing points for comprehensive trend analysis.
   - **Best for**: Mixed market conditions, when you need confirmation from multiple indicators, moderate volatility with some trend
   - **Avoid when**: Extreme market conditions (very high/low volatility), very choppy markets without clear signals
   - **Key indicators**: Moderate ADX (trend strength), RSI showing momentum, volume confirmation
   - **Core hyperparameters** (optional, max 4):
     * ema_fast (int, default 20, range 5-50): Fast EMA period. Smaller = more sensitive to short-term trends.
     * adx_threshold (float, default 25.0, range 15.0-35.0): Minimum ADX for trend confirmation. Higher = require stronger trends.
     * rsi_threshold (float, default 50.0, range 30.0-70.0): RSI level for oversold/overbought detection
     * min_hold_bars (int, REQUIRED, range 80-95% of trading_duration_minutes): Minimum bars to hold position (e.g., 360-430 for 450 minutes duration)
5. Trading duration in minutes (how long to use this strategy, MUST be between 300-600 minutes for minute-level trading. Recommended: 450 minutes)

**Strategy Selection Guidelines:**
- If market shows **strong directional trend** (high trend_strength, clear momentum) → Use "tsmom"
- If market is **sideways/oscillating** (low volatility, price bouncing around mean) → Use "zscore_mr"
- If market shows **moderate trend with mixed signals** (some trend but not strong, moderate volatility) → Use "adaptive_trend_fusion"
- **CRITICAL RISK MANAGEMENT**: 
  - If previous strategy period lost more than 5% (return < -0.05), **MUST switch to a different strategy**
  - If previous strategy lost money (return < 0), strongly consider switching to a different approach
  - **Avoid repeatedly using losing strategies**: If a strategy consistently underperforms, try a different one
  - **Prioritize capital preservation**: If uncertain, choose a more conservative strategy (e.g., zscore_mr for range-bound markets)

6. Hyperparameters (strategy_hyperparameters): Provide structured hyperparameters as a JSON object.
   **CRITICAL: min_hold_bars is REQUIRED and must be set for ALL strategies to prevent excessive trading.**
   
   For "tsmom" strategy, provide:
   {{
     "min_hold_bars": <int, REQUIRED, 80-95% of trading_duration_minutes (e.g., 360-430 for 450 minutes)>,
     "lookback": <int, optional, default 50, range 20-200>,
     "threshold": <float, optional, default 0.0, range 0.0-0.01>
   }}
   
   For "zscore_mr" strategy, provide:
   {{
     "min_hold_bars": <int, REQUIRED, 80-95% of trading_duration_minutes (e.g., 360-430 for 450 minutes)>,
     "window": <int, optional, default 40, range 20-100>,
     "entry_z": <float, optional, default 1.5, range 1.0-3.0>,
     "exit_z": <float, optional, default 20% of entry_z>
   }}
   
   For "adaptive_trend_fusion" strategy, provide:
   {{
     "min_hold_bars": <int, REQUIRED, 80-95% of trading_duration_minutes (e.g., 360-430 for 450 minutes)>,
     "ema_fast": <int, optional, default 20, range 5-50>,
     "adx_threshold": <float, optional, default 25.0, range 15.0-35.0>,
     "rsi_threshold": <float, optional, default 50.0, range 30.0-70.0>
   }}
   
   min_hold_bars guidelines:
   - **CRITICAL: min_hold_bars MUST be less than trading_duration_minutes** (e.g., if duration is 450, min_hold_bars should be < 450)
   - **IMPORTANT: min_hold_bars should be set to 80-95% of trading_duration_minutes to minimize trading frequency and reduce fees**
   - For 450 minutes duration: set min_hold_bars to 360-430 bars (80-95% of duration)
   - For 300 minutes duration: set min_hold_bars to 240-285 bars (80-95% of duration)
   - For 600 minutes duration: set min_hold_bars to 480-570 bars (80-95% of duration)
   - **NEVER set below 300 bars** - this is too small and will cause excessive trading
   - **NEVER set to 0** - this will cause excessive trading and high fees
   - **Recommended: set min_hold_bars to approximately 85-90% of trading_duration_minutes** (e.g., 380-405 for 450 minutes duration)
   
   Only include parameters you want to adjust from defaults. min_hold_bars must always be included.
7. Reasoning for your recommendation, duration, and hyperparameters

Respond in JSON format with keys: trend, volatility, expected_movement, recommended_strategy, trading_duration_minutes, strategy_hyperparameters (optional), reasoning.
"""
        
        messages = [
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=MarketAnalysis,
            )
            
            if response.success and response.extra and "parsed_model" in response.extra:
                # Extract parsed model from response (response_format ensures structured output)
                parsed_model: MarketAnalysis = response.extra["parsed_model"]
                
                # Convert hyperparameters to dict if it's a Pydantic model
                hyperparams = parsed_model.strategy_hyperparameters
                if hyperparams is None:
                    hyperparams = {}
                elif isinstance(hyperparams, BaseModel):
                    # If LLM returned a structured hyperparameter model, convert to dict
                    try:
                        hyperparams = hyperparams.model_dump(exclude_none=True)
                    except Exception as e:
                        logger.warning(f"Failed to convert hyperparameters model to dict: {e}, using empty dict")
                        hyperparams = {}
                elif isinstance(hyperparams, dict):
                    # Already a dict, use as is
                    pass
                elif isinstance(hyperparams, (list, tuple)):
                    # If it's a list/tuple, try to convert to dict
                    logger.warning(f"Hyperparameters is a list/tuple, converting to empty dict")
                    hyperparams = {}
                else:
                    # Fallback: try to convert to dict
                    try:
                        if hasattr(hyperparams, '__dict__'):
                            hyperparams = dict(hyperparams)
                        elif hasattr(hyperparams, 'model_dump'):
                            hyperparams = hyperparams.model_dump(exclude_none=True)
                        else:
                            logger.warning(f"Unknown hyperparameters type: {type(hyperparams)}, using empty dict")
                            hyperparams = {}
                    except Exception as e:
                        logger.warning(f"Failed to convert hyperparameters to dict: {e}, using empty dict")
                        hyperparams = {}
                
                return {
                    "trend": parsed_model.trend,
                    "volatility": parsed_model.volatility,
                    "expected_movement": parsed_model.expected_movement,
                    "recommended_strategy": parsed_model.recommended_strategy,
                    "trading_duration_minutes": parsed_model.trading_duration_minutes,
                    "strategy_hyperparameters": hyperparams,
                    "reasoning": parsed_model.reasoning,
                }
            else:
                # If response_format failed, log error and return fallback
                error_msg = response.message if hasattr(response, 'message') else 'No parsed_model in response'
                logger.warning(f"LLM analysis failed: {error_msg}")
                return {"recommended_strategy": "tsmom", "trading_duration_minutes": 450, "strategy_hyperparameters": {}, "reasoning": "Fallback to TSMOM"}
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return {"recommended_strategy": "tsmom", "trading_duration_minutes": 450, "reasoning": f"Error: {e}"}
    
    async def _select_strategy(self, analysis: Dict[str, Any]) -> str:
        """Select the appropriate strategy based on LLM analysis.
        
        Args:
            analysis: Market analysis from LLM
            
        Returns:
            Strategy name: "tsmom", "zscore_mr", or "adaptive_trend_fusion"
        """
        recommended = analysis.get("recommended_strategy", "tsmom").lower()
        
        if recommended == "zscore_mr" or recommended == "zscore":
            return "zscore_mr"
        elif recommended == "adaptive_trend_fusion" or recommended == "adaptive":
            return "adaptive_trend_fusion"
        else:
            return "tsmom"
    
    async def _analyze_trading_result(
        self,
        trading_result: Dict[str, Any],
        factors: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Use LLM to analyze trading results.
        
        Args:
            trading_result: Dictionary containing trading results
            factors: Current market factors
            
        Returns:
            Analysis dictionary
        """
        
        prompt = f"""You are a quantitative trading analyst. Analyze the following trading result.

Trading Result:
{json.dumps(trading_result, indent=2)}

Current Market Factors:
{json.dumps(factors, indent=2)}

Please analyze:
1. Was the strategy selection appropriate?
2. What went well or poorly?
3. What should be adjusted for the next trading period?
4. Any lessons learned?

Respond in JSON format with keys: strategy_appropriate, performance_analysis, adjustments, lessons_learned.
"""
        
        messages = [
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=TradingResultAnalysis,
            )
            
            if response.success and response.extra and "parsed_model" in response.extra:
                # Extract parsed model from response (response_format ensures structured output)
                parsed_model: TradingResultAnalysis = response.extra["parsed_model"]
                return {
                    "strategy_appropriate": parsed_model.strategy_appropriate,
                    "performance_analysis": parsed_model.performance_analysis,
                    "adjustments": parsed_model.adjustments,
                    "lessons_learned": parsed_model.lessons_learned,
                }
            else:
                # If response_format failed, log error and return fallback
                logger.warning(f"LLM result analysis failed: {response.message if hasattr(response, 'message') else 'No parsed_model in response'}")
                return {"strategy_appropriate": True, "performance_analysis": "Unable to analyze"}
        except Exception as e:
            logger.error(f"Error in LLM result analysis: {e}")
            return {"strategy_appropriate": True, "performance_analysis": f"Error: {e}"}
    
    async def generate_signal(
        self,
        df: pd.DataFrame,
        current_pos: float,
        current_equity: float,
        min_hold_bars: int = 0,
    ) -> float:
        """Generate trading signal using LLM agent.
        
        This is the main entry point for the backtest framework.
        
        Args:
            df: Historical price data
            current_pos: Current position
            current_equity: Current equity
            min_hold_bars: Minimum bars to hold (overrides instance default)
            
        Returns:
            Trading signal: -1.0 (short), 0.0 (flat), 1.0 (long)
        """
        return await self._generate_signal_async(df, current_pos, current_equity, min_hold_bars)
    
    async def _generate_signal_async(
        self,
        df: pd.DataFrame,
        current_pos: float,
        current_equity: float,
        min_hold_bars: int = 0,
    ) -> float:
        """Async implementation of signal generation with LLM-based strategy selection and execution periods.
        
        Logic:
        1. If a strategy is currently active and hasn't expired, use it directly
        2. If no active strategy or strategy expired:
           a. Analyze previous strategy period results (if any)
           b. Use LLM to select next strategy and duration
           c. Start new strategy period
        """
        if df is None or df.empty or current_equity <= 0:
            return 0.0
        
        if PRICE_COL not in df.columns:
            return 0.0
        
        # Get current bar index (use length of df as proxy)
        current_bar_index = len(df)
        
        # Check if we're in an active strategy period
        if (self.active_strategy is not None and 
            self.strategy_start_bar_index is not None and 
            self.strategy_duration_bars is not None):
            
            bars_elapsed = current_bar_index - self.strategy_start_bar_index
            
            # Check stop-loss: if loss exceeds threshold, end strategy period early
            if self.strategy_start_equity is not None and self.strategy_start_equity > 0:
                period_return = (current_equity - self.strategy_start_equity) / self.strategy_start_equity
                if period_return <= self.max_loss_threshold:
                    logger.warning(f"Strategy {self.active_strategy} hit stop-loss: {period_return:.4f} <= {self.max_loss_threshold:.4f}, ending period early after {bars_elapsed} bars")
                    self._end_strategy_period(df, current_pos, current_equity)
                    # Continue to select new strategy below
                elif bars_elapsed < self.strategy_duration_bars:
                    # Still in strategy period, use active strategy directly with stored hyperparameters
                    return self._generate_signal_with_strategy(
                        df, current_pos, current_equity, 
                        self.active_strategy, min_hold_bars,
                        hyperparameters=self.strategy_hyperparameters,
                    )
                else:
                    # Strategy period expired, need to analyze and select next strategy
                    logger.info(f"Strategy {self.active_strategy} period expired after {bars_elapsed} bars")
                    self._end_strategy_period(df, current_pos, current_equity)
            elif bars_elapsed < self.strategy_duration_bars:
                # Still in strategy period, use active strategy directly with stored hyperparameters
                return self._generate_signal_with_strategy(
                    df, current_pos, current_equity, 
                    self.active_strategy, min_hold_bars,
                    hyperparameters=self.strategy_hyperparameters,
                )
            else:
                # Strategy period expired, need to analyze and select next strategy
                logger.info(f"Strategy {self.active_strategy} period expired after {bars_elapsed} bars")
                self._end_strategy_period(df, current_pos, current_equity)
        
        # No active strategy or strategy expired - need LLM to select next strategy
        try:
            return await asyncio.wait_for(
                self._select_and_start_strategy_async(df, current_pos, current_equity, min_hold_bars),
                timeout=60.0
            )
        except asyncio.TimeoutError:
            logger.error("LLM call timed out after 60 seconds, falling back to heuristic")
            return await self._select_strategy_heuristic(df, current_pos, current_equity, min_hold_bars)
        except Exception as e:
            logger.error(f"Error calling LLM for strategy selection: {e}, falling back to heuristic")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return await self._select_strategy_heuristic(df, current_pos, current_equity, min_hold_bars)
    
    def _generate_signal_with_strategy(
        self,
        df: pd.DataFrame,
        current_pos: float,
        current_equity: float,
        strategy_name: str,
        min_hold_bars: int = 0,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Generate signal using a specific strategy with optional core hyperparameters.
        
        Args:
            df: Historical price data
            current_pos: Current position
            current_equity: Current equity
            strategy_name: Strategy name
            min_hold_bars: Minimum bars to hold
            hyperparameters: Optional dictionary of core hyperparameters to override defaults
        """
        hyperparams = hyperparameters or {}
        
        # min_hold_bars priority: hyperparameters > function parameter (if > 0) > instance default
        # This ensures traditional strategies always have a reasonable min_hold_bars to prevent excessive trading
        min_hold_bars_val = hyperparams.get("min_hold_bars")
        if min_hold_bars_val is None:
            # If not set in hyperparameters, use function parameter if > 0, otherwise use instance default
            if min_hold_bars > 0:
                min_hold_bars_val = min_hold_bars
            else:
                min_hold_bars_val = self.min_hold_bars
        
        if strategy_name == "tsmom":
            # Core parameters: lookback, threshold, min_hold_bars, trend_filter_period
            signal = self.tsmom.generate_signal(
                df, current_pos, current_equity,
                min_hold_bars=min_hold_bars_val,
                lookback=hyperparams.get("lookback", 50),
                threshold=hyperparams.get("threshold", 0.001),  # Default 0.1% threshold
                trend_filter_period=hyperparams.get("trend_filter_period", 200),  # Default SMA200
            )
        elif strategy_name == "zscore_mr":
            # Core parameters: window, entry_z, min_hold_bars (exit_z derived from entry_z if not provided)
            # trend_filter_period, trend_strength_threshold
            entry_z = hyperparams.get("entry_z", 2.0)  # Increased default from 1.5 to 2.0
            exit_z = hyperparams.get("exit_z", entry_z * 0.15)  # Default exit_z is 15% of entry_z
            signal = self.zscore_mr.generate_signal(
                df, current_pos, current_equity,
                min_hold_bars=min_hold_bars_val,
                window=hyperparams.get("window", 40),
                entry_z=entry_z,
                exit_z=exit_z,
                trend_filter_period=hyperparams.get("trend_filter_period", 200),  # Default SMA200
                trend_strength_threshold=hyperparams.get("trend_strength_threshold", 0.005),  # Default 0.5%
            )
        elif strategy_name == "adaptive_trend_fusion":
            # Core parameters: ema_fast, adx_threshold, rsi_threshold, min_hold_bars
            # ema_slow derived from ema_fast (default 2.5x ratio)
            ema_fast = hyperparams.get("ema_fast", 20)
            ema_slow = hyperparams.get("ema_slow", int(ema_fast * 2.5))
            rsi_threshold = hyperparams.get("rsi_threshold", 50.0)
            # Use rsi_threshold for both oversold and overbought (centered around 50)
            rsi_oversold = 100 - rsi_threshold
            rsi_overbought = rsi_threshold
            
            signal = self.adaptive_trend_fusion.generate_signal(
                df, current_pos, current_equity,
                min_hold_bars=min_hold_bars_val,
                sma_trend_period=200,  # Fixed
                ema_fast=ema_fast,
                ema_slow=ema_slow,
                adx_period=14,  # Fixed
                adx_threshold=hyperparams.get("adx_threshold", 25.0),
                rsi_period=14,  # Fixed
                rsi_oversold=rsi_oversold,
                rsi_overbought=rsi_overbought,
                swing_lookback=20,  # Fixed
                vol_surge_lookback=20,  # Fixed
                vol_surge_multiplier=1.5,  # Fixed
                atr_period=14,  # Fixed
            )
        else:
            signal = 0.0
        
        # Store trading decision
        self.trading_history.append({
            "timestamp": df["t"].iloc[-1] if "t" in df.columns else None,
            "strategy": strategy_name,
            "signal": signal,
            "current_pos": current_pos,
            "current_equity": current_equity,
            "in_period": True,
        })
        
        return signal
    
    def _end_strategy_period(
        self,
        df: pd.DataFrame,
        current_pos: float,
        current_equity: float,
    ):
        """End current strategy period and record results."""
        if self.active_strategy is None:
            return
        
        # Calculate period performance using strategy_start_equity
        if self.strategy_start_equity is not None and self.strategy_start_equity > 0:
            period_start_equity = self.strategy_start_equity
        else:
            # Fallback: try to get from previous period or trading history
            if self.strategy_period_history:
                last_period = self.strategy_period_history[-1]
                period_start_equity = last_period.get("end_equity", current_equity)
            elif self.trading_history:
                period_start_equity = self.trading_history[0].get("current_equity", current_equity)
            else:
                period_start_equity = current_equity
        
        period_return = (current_equity - period_start_equity) / period_start_equity if period_start_equity > 0 else 0
        
        # Extract equity history for this strategy period from trading_history
        # Get entries marked with "in_period": True, which are generated during this period
        period_equity_values = []
        
        if self.trading_history:
            # Get entries that belong to this period (marked with "in_period": True)
            for entry in self.trading_history:
                if "current_equity" in entry and entry.get("in_period", False):
                    period_equity_values.append(entry["current_equity"])
        
        # Always include start and end equity for accurate calculation
        if not period_equity_values:
            period_equity_values = [period_start_equity, current_equity]
        else:
            # Ensure start equity is first and end equity is last
            if period_equity_values[0] != period_start_equity:
                period_equity_values.insert(0, period_start_equity)
            if period_equity_values[-1] != current_equity:
                period_equity_values.append(current_equity)
        
        # Calculate performance metrics
        equity_series = pd.Series(period_equity_values)
        
        # Calculate return (already calculated above)
        # period_return is already calculated
        
        # Calculate Sharpe ratio
        if len(equity_series) > 1:
            ret = equity_series.pct_change().fillna(0.0)
            # For minute-level data: 1 day = 1440 bars
            bars_per_day = 1440.0
            annual_factor = math.sqrt(365.0 * bars_per_day)
            vol = float(ret.std(ddof=0))
            sharpe = (ret.mean() * annual_factor / vol) if vol > 0 else float("nan")
        else:
            sharpe = float("nan")
        
        # Calculate max drawdown
        if len(equity_series) > 1:
            cummax = equity_series.cummax()
            drawdown = (equity_series / cummax - 1.0).fillna(0.0)
            max_drawdown = float(drawdown.min())
        else:
            max_drawdown = 0.0
        
        # Count trades (signal changes) during this strategy period
        num_trades = 0
        num_buys = 0
        num_sells = 0
        
        if self.trading_history:
            # Get entries that belong to this period (marked with "in_period": True)
            period_entries = [entry for entry in self.trading_history if entry.get("in_period", False)]
            
            if len(period_entries) > 1:
                # Track previous signal to detect changes
                prev_signal = None
                for entry in period_entries:
                    signal = entry.get("signal", 0.0)
                    
                    if prev_signal is not None:
                        # Detect signal change
                        if signal != prev_signal:
                            # Classify trade type based on signal transition
                            if prev_signal == 0.0:
                                # Opening position
                                if signal > 0:
                                    num_buys += 1  # Opening long
                                    num_trades += 1
                                elif signal < 0:
                                    num_sells += 1  # Opening short
                                    num_trades += 1
                            elif signal == 0.0:
                                # Closing position
                                if prev_signal > 0:
                                    num_sells += 1  # Closing long (selling)
                                    num_trades += 1
                                elif prev_signal < 0:
                                    num_buys += 1  # Closing short (buying to cover)
                                    num_trades += 1
                            else:
                                # Reversing position (prev_signal != 0 and signal != 0)
                                # This counts as closing one position and opening another
                                if prev_signal > 0:
                                    num_sells += 1  # Closing long
                                elif prev_signal < 0:
                                    num_buys += 1  # Closing short
                                
                                if signal > 0:
                                    num_buys += 1  # Opening long
                                elif signal < 0:
                                    num_sells += 1  # Opening short
                                
                                num_trades += 2  # Two trades: close + open
                    
                    prev_signal = signal
        
        # Record period with performance metrics
        period_record = {
            "strategy": self.active_strategy,
            "start_bar": self.strategy_start_bar_index,
            "end_bar": len(df),
            "duration_bars": self.strategy_duration_bars,
            "start_equity": period_start_equity,
            "end_equity": current_equity,
            "return": period_return,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "num_trades": num_trades,
            "num_buys": num_buys,
            "num_sells": num_sells,
            "hyperparameters": self.strategy_hyperparameters.copy(),
        }
        self.strategy_period_history.append(period_record)
        
        # Calculate total return from start of backtest
        initial_equity = None
        if self.strategy_period_history:
            # Get initial equity from first period
            first_period = self.strategy_period_history[0]
            initial_equity = first_period.get("start_equity", current_equity)
        elif self.trading_history:
            # Fallback: get from first trading history entry
            initial_equity = self.trading_history[0].get("current_equity", current_equity)
        else:
            initial_equity = current_equity
        
        total_return = (current_equity - initial_equity) / initial_equity if initial_equity > 0 else 0.0
        
        logger.info(f"Ended strategy period: {self.active_strategy}, period_return: {period_return:.4f}, sharpe: {sharpe:.4f}, max_drawdown: {max_drawdown:.4f}")
        logger.info(f"Trades during period: total={num_trades}, buys={num_buys}, sells={num_sells}")
        logger.info(f"Current total equity: {current_equity:.2f}, total_return: {total_return:.4f} ({total_return*100:.2f}%)")
        
        # Clear active strategy
        self.active_strategy = None
        self.strategy_start_bar_index = None
        self.strategy_duration_bars = None
        self.strategy_hyperparameters = {}
        self.strategy_start_equity = None
    
    async def _select_and_start_strategy_async(
        self,
        df: pd.DataFrame,
        current_pos: float,
        current_equity: float,
        min_hold_bars: int = 0,
    ) -> float:
        """Use LLM to select next strategy and start execution period."""
        # Calculate factors
        factors = calculate_factors(df)
        if not factors:
            return 0.0
        
        # Store factor history
        self.factor_history.append({
            "timestamp": df["t"].iloc[-1] if "t" in df.columns else None,
            "factors": factors.copy(),
        })
        
        # Prepare previous period analysis if available
        previous_period_info = None
        if self.strategy_period_history:
            last_period = self.strategy_period_history[-1]
            previous_period_info = {
                "strategy": last_period["strategy"],
                "duration_bars": last_period["duration_bars"],
                "return": last_period.get("return", 0.0),
                "sharpe": last_period.get("sharpe", float("nan")),
                "max_drawdown": last_period.get("max_drawdown", 0.0),
                "num_trades": last_period.get("num_trades", 0),
                "num_buys": last_period.get("num_buys", 0),
                "num_sells": last_period.get("num_sells", 0),
            }
        
        # Calculate total return from start of backtest
        initial_equity = None
        if self.strategy_period_history:
            # Get initial equity from first period
            first_period = self.strategy_period_history[0]
            initial_equity = first_period.get("start_equity", current_equity)
        elif self.trading_history:
            # Fallback: get from first trading history entry
            initial_equity = self.trading_history[0].get("current_equity", current_equity)
        else:
            initial_equity = current_equity
        
        total_return = (current_equity - initial_equity) / initial_equity if initial_equity > 0 else 0.0
        
        # Use LLM to analyze market and select strategy
        logger.info("Calling LLM to analyze market and select strategy...")
        logger.info(f"Current total equity: {current_equity:.2f}, total_return: {total_return:.4f} ({total_return*100:.2f}%)")
        if previous_period_info:
            num_trades_prev = previous_period_info.get('num_trades', 0)
            num_buys_prev = previous_period_info.get('num_buys', 0)
            num_sells_prev = previous_period_info.get('num_sells', 0)
            logger.info(f"Previous strategy period: {previous_period_info['strategy']}, return: {previous_period_info['return']:.4f}, "
                       f"sharpe: {previous_period_info.get('sharpe', float('nan')):.4f}, "
                       f"max_drawdown: {previous_period_info.get('max_drawdown', 0.0):.4f}, "
                       f"trades: {num_trades_prev} (buys: {num_buys_prev}, sells: {num_sells_prev})")
        analysis = await self._analyze_market(df, factors, previous_period_info=previous_period_info)
        logger.info(f"LLM analysis result: trend={analysis.get('trend')}, volatility={analysis.get('volatility')}, "
                   f"recommended_strategy={analysis.get('recommended_strategy')}, "
                   f"trading_duration_minutes={analysis.get('trading_duration_minutes')}")
        logger.info(f"LLM reasoning: {analysis.get('reasoning', 'N/A')}")
        strategy_name = await self._select_strategy(analysis)
        trading_duration_minutes = analysis.get("trading_duration_minutes", 450)
        # Ensure duration is within valid range (300-600)
        trading_duration_minutes = max(300, min(600, trading_duration_minutes))
        hyperparameters = analysis.get("strategy_hyperparameters", {}) or {}
        
        # Convert minutes to bars
        # For minute-level data: 1 minute = 1 bar
        # For day-level data: 1 minute = 1/1440 bar (but we're doing minute-level trading)
        strategy_duration_bars = trading_duration_minutes
        
        # Ensure min_hold_bars is set in hyperparameters
        # If LLM didn't set it, use a reasonable default: 85-90% of strategy duration to minimize trading
        if "min_hold_bars" not in hyperparameters:
            # Default to 87.5% of strategy duration (close to 450 for 450 bars duration)
            default_min_hold_bars = max(300, min(int(strategy_duration_bars * 0.95), int(strategy_duration_bars * 0.875)))
            hyperparameters["min_hold_bars"] = default_min_hold_bars
            logger.info(f"LLM did not set min_hold_bars, using default: {default_min_hold_bars} (87.5% of {strategy_duration_bars} bars)")
        
        # Ensure min_hold_bars is less than strategy duration and within reasonable range (80-95% of duration)
        min_hold_bars_val = hyperparameters.get("min_hold_bars", self.min_hold_bars)
        min_min_hold_bars = max(300, int(strategy_duration_bars * 0.80))  # At least 80% of duration, but at least 300
        max_min_hold_bars = int(strategy_duration_bars * 0.95)  # At most 95% of duration
        
        if min_hold_bars_val >= strategy_duration_bars:
            # Cap min_hold_bars to be at most 95% of strategy duration
            hyperparameters["min_hold_bars"] = max_min_hold_bars
            logger.warning(f"min_hold_bars ({min_hold_bars_val}) >= strategy_duration_bars ({strategy_duration_bars}), "
                         f"capping to {hyperparameters['min_hold_bars']} (95% of duration)")
        elif min_hold_bars_val < min_min_hold_bars:
            # If too small, increase to at least 80% of strategy duration (but at least 300)
            hyperparameters["min_hold_bars"] = min_min_hold_bars
            logger.warning(f"min_hold_bars ({min_hold_bars_val}) is too small, increasing to {hyperparameters['min_hold_bars']} (80% of {strategy_duration_bars} bars)")
        elif min_hold_bars_val > max_min_hold_bars:
            # If too large, cap to 95% of strategy duration
            hyperparameters["min_hold_bars"] = max_min_hold_bars
            logger.warning(f"min_hold_bars ({min_hold_bars_val}) is too large, capping to {hyperparameters['min_hold_bars']} (95% of {strategy_duration_bars} bars)")
        
        # Start new strategy period
        current_bar_index = len(df)
        self.active_strategy = strategy_name
        self.strategy_start_bar_index = current_bar_index
        self.strategy_duration_bars = strategy_duration_bars
        self.strategy_hyperparameters = hyperparameters.copy()
        self.strategy_start_equity = current_equity  # Record starting equity for stop-loss
        
        logger.info(f"Starting new strategy period: {strategy_name} for {trading_duration_minutes} minutes ({strategy_duration_bars} bars)")
        logger.info(f"Strategy hyperparameters: {hyperparameters}")
        
        # Generate signal using selected strategy with hyperparameters
        signal = self._generate_signal_with_strategy(
            df, current_pos, current_equity, strategy_name, min_hold_bars,
            hyperparameters=hyperparameters,
        )
        
        # Store strategy selection decision
        self.trading_history.append({
            "timestamp": df["t"].iloc[-1] if "t" in df.columns else None,
            "strategy": strategy_name,
            "signal": signal,
            "current_pos": current_pos,
            "current_equity": current_equity,
            "in_period": False,  # This is the decision point
            "analysis": analysis,
            "strategy_duration_minutes": trading_duration_minutes,
        })
        
        self.current_strategy = strategy_name
        
        return signal
    
    async def _select_strategy_heuristic(
        self,
        df: pd.DataFrame,
        current_pos: float,
        current_equity: float,
        min_hold_bars: int = 0,
    ) -> float:
        """Fallback heuristic strategy selection when async LLM is not available."""
        # Calculate factors
        factors = calculate_factors(df)
        if not factors:
            return 0.0
        
        # Simple heuristic selection
        trend_strength = factors.get('trend_strength', 0)
        volatility = factors.get('volatility_20', 0)
        momentum_20 = factors.get('momentum_20', 0)
        
        if abs(trend_strength) > 0.01 and abs(momentum_20) > 0.005:
            strategy_name = "tsmom"
            duration_minutes = 450  # Default to 450 minutes (middle of 300-600 range)
        elif volatility < 0.02:
            strategy_name = "zscore_mr"
            duration_minutes = 450  # Default to 450 minutes (middle of 300-600 range)
        else:
            strategy_name = "adaptive_trend_fusion"
            duration_minutes = 450  # Default to 450 minutes (middle of 300-600 range)
        
        # Start strategy period
        current_bar_index = len(df)
        self.active_strategy = strategy_name
        self.strategy_start_bar_index = current_bar_index
        self.strategy_duration_bars = duration_minutes
        self.strategy_start_equity = current_equity  # Record starting equity for stop-loss
        
        # Use reasonable min_hold_bars for heuristic selection: 85-90% of strategy duration to minimize trading
        min_hold_bars_val = max(300, min(int(duration_minutes * 0.95), int(duration_minutes * 0.875)))
        if min_hold_bars_val >= duration_minutes:
            # Cap min_hold_bars to be at most 95% of strategy duration
            min_hold_bars_val = int(duration_minutes * 0.95)
            logger.warning(f"Heuristic: min_hold_bars would be >= duration ({duration_minutes}), "
                         f"capping to {min_hold_bars_val} (95% of duration)")
        heuristic_hyperparameters = {"min_hold_bars": min_hold_bars_val}
        
        return self._generate_signal_with_strategy(
            df, current_pos, current_equity, strategy_name, min_hold_bars,
            hyperparameters=heuristic_hyperparameters,
        )
    
    def get_trading_history(self) -> List[Dict[str, Any]]:
        """Get trading history.
        
        Returns:
            List of trading decisions
        """
        return self.trading_history.copy()
    
    def get_factor_history(self) -> List[Dict[str, Any]]:
        """Get factor calculation history.
        
        Returns:
            List of factor calculations
        """
        return self.factor_history.copy()
    
    async def analyze_trading_performance(
        self,
        performance_metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Analyze trading performance using LLM.
        
        Args:
            performance_metrics: Optional performance metrics (e.g., total_return, sharpe_ratio)
            
        Returns:
            Analysis dictionary
        """
        
        # Prepare summary
        summary = {
            "total_decisions": len(self.trading_history),
            "strategy_usage": {},
            "performance_metrics": performance_metrics or {},
        }
        
        # Count strategy usage
        for decision in self.trading_history:
            strategy = decision.get("strategy", "unknown")
            summary["strategy_usage"][strategy] = summary["strategy_usage"].get(strategy, 0) + 1
        
        prompt = f"""You are a quantitative trading analyst. Analyze the following trading performance.

Trading Summary:
{json.dumps(summary, indent=2)}

Recent Trading Decisions (last 10):
{json.dumps(self.trading_history[-10:], indent=2) if len(self.trading_history) > 0 else "No trading history"}

Please provide:
1. Overall performance assessment
2. Strategy selection effectiveness
3. Recommendations for improvement
4. Key insights and lessons learned

Respond in JSON format with keys: performance_assessment, strategy_effectiveness, recommendations, insights.
"""
        
        messages = [
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await model_manager(
                model=self.model_name,
                messages=messages,
                response_format=PerformanceAnalysis,
            )
            
            if response.success and response.extra and "parsed_model" in response.extra:
                # Extract parsed model from response (response_format ensures structured output)
                parsed_model: PerformanceAnalysis = response.extra["parsed_model"]
                analysis = {
                    "performance_assessment": parsed_model.performance_assessment,
                    "strategy_effectiveness": parsed_model.strategy_effectiveness,
                    "recommendations": parsed_model.recommendations,
                    "insights": parsed_model.insights,
                }
                return {
                    "summary": summary,
                    "llm_analysis": analysis,
                }
            else:
                # If response_format failed, log error and return fallback
                logger.warning(f"LLM performance analysis failed: {response.message if hasattr(response, 'message') else 'No parsed_model in response'}")
                return {
                    "summary": summary,
                    "llm_analysis": {"error": "Analysis failed"},
                }
        except Exception as e:
            logger.error(f"Error in LLM performance analysis: {e}")
            return {
                "summary": summary,
                "llm_analysis": {"error": str(e)},
            }
    
    def reset(self):
        """Reset trading state (useful for new backtest runs)."""
        self.trading_history.clear()
        self.factor_history.clear()
        self.current_strategy = None
        self.active_strategy = None
        self.strategy_start_bar_index = None
        self.strategy_duration_bars = None
        self.strategy_hyperparameters = {}
        self.strategy_period_history.clear()
        self.strategy_start_equity = None
        self.tsmom = TSMOMStrategy()
        self.zscore_mr = ZScoreMRStrategy()
        self.adaptive_trend_fusion = AdaptiveTrendFusionStrategy()


# ============================================================
#          Baseline Function for Backtest Framework
# ============================================================

# Global agent instance (will be initialized on first call)
_agent_instance: Optional[AgentStrategy] = None


async def agent_baseline(
    df: pd.DataFrame,
    current_pos: float,
    current_equity: float,
    min_hold_bars: int = 0,
    model_name: str = "openai/gemini-3-flash-preview",
    data_path: Optional[str] = None,
    trading_horizon: int = 60,
) -> float:
    """Baseline function for LLM agent strategy.
    
    This function is called by the backtest framework.
    
    Args:
        df: Historical price data
        current_pos: Current position
        current_equity: Current equity
        min_hold_bars: Minimum bars to hold position
        model_name: LLM model name
        data_path: Path to data directory
        trading_horizon: Trading horizon in minutes
        
    Returns:
        Trading signal: -1.0 (short), 0.0 (flat), 1.0 (long)
    """
    global _agent_instance
    
    # Initialize agent instance if needed
    if _agent_instance is None:
        _agent_instance = AgentStrategy(
            model_name=model_name,
            data_path=data_path,
            trading_horizon=trading_horizon,
        )
    
    return await _agent_instance.generate_signal(df, current_pos, current_equity, min_hold_bars)


def get_agent_instance() -> Optional[AgentStrategy]:
    """Get the global agent instance (for accessing history and analysis).
    
    Returns:
        AgentStrategy instance or None if not initialized
    """
    return _agent_instance


def reset_agent_instance():
    """Reset the global agent instance (useful for new backtest runs)."""
    global _agent_instance
    if _agent_instance is not None:
        _agent_instance.reset()
    _agent_instance = None


# ============================================================
#          Data Loading Functions
# ============================================================

def load_data_from_jsonl(
    coin: str = "BTC",
    interval: str = "1m",
    lookback: int = 0,
    data_path: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load crypto price data from JSONL files in workdir.
    
    Args:
        coin: Coin symbol (e.g., "BTC")
        interval: K-line interval (e.g., "1day", "1min")
        lookback: Number of candles to load (0 = all)
        data_path: Path to data directory (default: workdir/crypto/crypto_binance_price_{interval})
        start_time: Start time filter (ISO format string, e.g., "2024-01-01T00:00:00")
        end_time: End time filter (ISO format string, e.g., "2024-12-31T23:59:59")
    
    Returns:
        DataFrame with columns: t, o, h, l, c, v
    """
    # Map interval to directory name
    interval_map = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1hour",
        "1d": "1day",
        "1day": "1day",
    }
    
    # Get project root
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..'))
    
    # Build file path
    if data_path is None:
        interval_dir = interval_map.get(interval, interval)
        data_path = os.path.join(
            project_root,
            "workdir",
            "crypto",
            f"crypto_binance_price_{interval_dir}",
        )
    
    symbol = f"{coin}USDT"
    file_path = os.path.join(data_path, f"{symbol}.jsonl")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Load jsonl file
    df = pd.read_json(file_path, lines=True)
    
    # Rename columns to match backtest format
    column_map = {
        "timestamp": "t",
        "open": "o",
        "high": "h",
        "low": "l",
        "close": "c",
        "volume": "v",
    }
    
    df = df.rename(columns=column_map)
    
    # Ensure timestamp is datetime
    df["t"] = pd.to_datetime(df["t"])
    
    # Sort by timestamp
    df = df.sort_values("t").reset_index(drop=True)
    
    # Apply time filters
    if start_time:
        start_dt = pd.to_datetime(start_time)
        df = df[df["t"] >= start_dt].reset_index(drop=True)
        logger.info(f"Filtered data: start_time >= {start_time}, remaining {len(df)} rows")
    
    if end_time:
        end_dt = pd.to_datetime(end_time)
        df = df[df["t"] <= end_dt].reset_index(drop=True)
        logger.info(f"Filtered data: end_time <= {end_time}, remaining {len(df)} rows")
    
    # Apply lookback limit (take last N rows) - only if no time filters specified
    if lookback > 0 and len(df) > lookback and not start_time and not end_time:
        df = df.tail(lookback).reset_index(drop=True)
    
    # Ensure numeric columns are float
    for col in ["o", "h", "l", "c", "v"]:
        df[col] = df[col].astype(float)
    
    return df


# ============================================================
#          Backtest Functions
# ============================================================

async def run_agent_backtest(
    df: pd.DataFrame,
    agent_strategy: AgentStrategy,
    initial_equity: float = 1000.0,
    max_leverage: float = MAX_LEVERAGE,
    taker_fee_rate: float = 0.00045,
    slippage_bps: float = 1.0,
    price_col: str = PRICE_COL,
    start_index: int = 200,  # Need enough data for factor calculation
    min_hold_bars: int = 0,
) -> Dict[str, Any]:
    """
    Run backtest with agent strategy.
    
    Args:
        df: DataFrame with price data
        agent_strategy: AgentStrategy instance
        initial_equity: Initial equity
        max_leverage: Maximum leverage
        taker_fee_rate: Taker fee rate
        slippage_bps: Slippage in basis points
        price_col: Price column name
        start_index: Starting index for backtest
        min_hold_bars: Minimum bars to hold position
    
    Returns:
        Dictionary containing backtest results
    """
    df = df.reset_index(drop=True).copy()
    if price_col not in df.columns:
        raise ValueError(f"price_col '{price_col}' not in df columns")
    
    closes = df[price_col].astype(float).values
    times = pd.to_datetime(df["t"]).values if "t" in df.columns else df.index.to_numpy()
    
    equity: float = float(initial_equity)
    pos_frac: float = 0.0       # Position fraction ∈ [-1,1]
    position: float = 0.0       # Actual position size
    last_price: float = float(closes[0])
    
    slippage = slippage_bps / 10000.0
    
    equity_list = []
    pos_frac_list = []
    trades = []
    
    last_trade_equity: float = equity
    realized_pnl_cum: float = 0.0
    
    for i in range(len(df)):
        price = float(closes[i])
        time = times[i]
        
        # ---------- 1. Mark-to-market (unrealized PnL) ----------
        # 永续合约盈亏计算：
        # - 做多（position > 0）：价格上涨时赚钱，价格下跌时亏钱
        # - 做空（position < 0）：价格下跌时赚钱，价格上涨时亏钱
        # 公式：pnl = (price - last_price) * position
        #   做多：position > 0, price上涨 → pnl > 0 ✓
        #   做空：position < 0, price下跌 → pnl > 0 ✓
        pnl = (price - last_price) * position
        equity += pnl
        last_price = price
        
        # ---------- 2. Call strategy for new position fraction ----------
        if i >= start_index and price > 0 and equity > 0:
            new_frac = float(await agent_strategy.generate_signal(
                df.iloc[: i + 1], pos_frac, equity, min_hold_bars=min_hold_bars
            ))
            new_frac = float(max(-1.0, min(1.0, new_frac)))  # Limit to [-1,1]
        else:
            new_frac = pos_frac
        
        # ---------- 3. Rebalance if position changed ----------
        if abs(new_frac - pos_frac) > 1e-6 and price > 0 and equity > 0:
            equity_before_trade = equity
            
            # 计算目标持仓量（永续合约）：
            # - new_frac = 1.0 → target_position > 0（做多）
            # - new_frac = -1.0 → target_position < 0（做空）
            # - new_frac = 0.0 → target_position = 0（空仓）
            target_position = new_frac * equity * max_leverage / price
            delta = target_position - position
            
            if abs(delta) > 1e-9:
                side = "BUY" if delta > 0 else "SELL"
                # Add slippage
                fill = price * (1 + slippage if delta > 0 else 1 - slippage)
                
                notional = abs(delta) * fill
                fee = notional * taker_fee_rate
                
                # Segment PnL (from last trade to this trade, excluding current fee)
                segment_pnl = equity_before_trade - last_trade_equity
                
                # Realized PnL = segment PnL - current fee
                realized_pnl = segment_pnl - fee
                realized_pnl_cum += realized_pnl
                
                equity_after_trade = equity_before_trade - fee
                
                # Classify trade action
                if abs(pos_frac) < 1e-9 and abs(new_frac) > 1e-9:
                    action = "OPEN_LONG" if new_frac > 0 else "OPEN_SHORT"
                elif abs(new_frac) < 1e-9 and abs(pos_frac) > 1e-9:
                    action = "CLOSE"
                elif pos_frac * new_frac < 0:
                    action = "REVERSE"
                else:
                    action = "ADJUST"
                
                # Get current strategy info
                current_strategy = agent_strategy.active_strategy or agent_strategy.current_strategy or "unknown"
                
                trades.append({
                    "index": i,
                    "time": pd.Timestamp(time).isoformat() if hasattr(time, 'isoformat') else str(time),
                    "price": fill,
                    "old_frac": pos_frac,
                    "new_frac": new_frac,
                    "old_pos": position,
                    "new_pos": target_position,
                    "delta_pos": delta,
                    "side": side,
                    "action": action,
                    "fee": fee,
                    "segment_pnl": segment_pnl,
                    "realized_pnl": realized_pnl,
                    "realized_pnl_cum": realized_pnl_cum,
                    "equity_before": equity_before_trade,
                    "equity_after": equity_after_trade,
                    "strategy": current_strategy,
                })
                
                equity = equity_after_trade
                last_trade_equity = equity_after_trade
                position = target_position
                pos_frac = new_frac
        
        equity_list.append(equity)
        pos_frac_list.append(pos_frac)
    
    equity_series = pd.Series(
        equity_list,
        index=df["t"] if "t" in df.columns else df.index,
        name="equity",
    )
    pos_frac_series = pd.Series(pos_frac_list, index=equity_series.index, name="pos_frac")
    trades_df = pd.DataFrame(trades)
    
    stats = _calc_backtest_stats(equity_series, trades_df)
    
    return {
        "equity_curve": equity_series,
        "pos_frac_series": pos_frac_series,
        "trades": trades_df,
        "stats": stats,
        "trading_history": agent_strategy.get_trading_history(),
        "factor_history": agent_strategy.get_factor_history(),
        "strategy_period_history": agent_strategy.strategy_period_history,
    }


def _calc_backtest_stats(
    equity_curve: pd.Series,
    trades: Optional[pd.DataFrame] = None,
) -> Dict[str, float]:
    """Calculate backtest statistics."""
    stats: Dict[str, float] = {}
    
    if equity_curve is None or len(equity_curve) == 0:
        return stats
    
    first = float(equity_curve.iloc[0])
    last = float(equity_curve.iloc[-1])
    ret = equity_curve.pct_change().fillna(0.0)
    
    total_return = (last / first - 1.0) if first > 0 else float("nan")
    
    # Calculate annualization factor based on bar frequency
    # For minute-level data: 1 day = 1440 bars (24 hours * 60 minutes)
    # For day-level data: 1 day = 1 bar
    # We assume minute-level trading by default
    bars_per_day = 1440.0
    annual_factor = math.sqrt(365.0 * bars_per_day)
    
    vol = float(ret.std(ddof=0))
    sharpe = (ret.mean() * annual_factor / vol) if vol > 0 else float("nan")
    
    cummax = equity_curve.cummax()
    drawdown = (equity_curve / cummax - 1.0).fillna(0.0)
    max_dd = float(drawdown.min())
    
    stats.update({
        "final_equity": last,
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
    })
    
    if trades is not None and not trades.empty:
        num_trades = int(len(trades))
        total_fees = float(trades["fee"].sum()) if "fee" in trades.columns else 0.0
        avg_fee_per_trade = float(total_fees / num_trades) if num_trades > 0 else 0.0
        
        num_long_trades = 0
        num_short_trades = 0
        if "action" in trades.columns:
            num_long_trades = int((trades["action"] == "OPEN_LONG").sum())
            num_short_trades = int((trades["action"] == "OPEN_SHORT").sum())
        
        total_realized_pnl = 0.0
        if "realized_pnl" in trades.columns:
            total_realized_pnl = float(trades["realized_pnl"].sum())
        
        total_equity_pnl = last - first
        unrealized_pnl = total_equity_pnl - total_realized_pnl
        
        win_rate = float("nan")
        if "realized_pnl" in trades.columns and "action" in trades.columns:
            closed = trades[trades["action"].isin(["CLOSE", "REVERSE", "ADJUST"])]
            if not closed.empty:
                wins = (closed["realized_pnl"] > 0).sum()
                win_rate = float(wins / len(closed))
        
        stats.update({
            "num_trades": num_trades,
            "total_fees": total_fees,
            "avg_fee_per_trade": avg_fee_per_trade,
            "num_long_trades": num_long_trades,
            "num_short_trades": num_short_trades,
            "total_realized_pnl": total_realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "win_rate": win_rate,
        })
    
    return stats


# ============================================================
#          Save Trading History Functions
# ============================================================

def save_trading_results(
    results: Dict[str, Any],
    output_path: str,
    include_history: bool = True,
):
    """
    Save trading results to JSON file.
    
    Args:
        results: Backtest results dictionary
        output_path: Path to save JSON file
        include_history: Whether to include full trading history
    """
    output_data = {
        "timestamp": datetime.now().isoformat(),
        "stats": results["stats"],
        "trades": results["trades"].to_dict("records") if not results["trades"].empty else [],
        "strategy_periods": results.get("strategy_period_history", []),
    }
    
    if include_history:
        output_data["trading_history"] = results.get("trading_history", [])
        output_data["factor_history"] = results.get("factor_history", [])
    
    # Convert numpy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        elif pd.isna(obj):
            return None
        return obj
    
    output_data = convert_numpy_types(output_data)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"Trading results saved to {output_path}")


# ============================================================
#          Visualization Functions
# ============================================================

def plot_backtest_results(
    results: Dict[str, Any],
    title: str = "Agent Strategy Backtest Results",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot backtest results including equity curve, drawdown, and position fraction.
    
    Args:
        results: Backtest results dictionary
        title: Plot title
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    equity_curve = results["equity_curve"]
    pos_frac_series = results["pos_frac_series"]
    trades_df = results["trades"]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(title, fontsize=16)
    
    # 1. Equity curve
    ax1 = axes[0]
    ax1.plot(equity_curve.index, equity_curve.values, label="Equity", linewidth=2)
    ax1.set_title("Equity Curve")
    ax1.set_ylabel("Equity")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Mark trades
    if not trades_df.empty and "time" in trades_df.columns:
        trade_times = pd.to_datetime(trades_df["time"])
        trade_prices = trades_df["equity_after"].values
        ax1.scatter(trade_times, trade_prices, c='red', s=20, alpha=0.5, label="Trades")
    
    # 2. Drawdown
    ax2 = axes[1]
    cummax = equity_curve.cummax()
    drawdown = (equity_curve / cummax - 1.0) * 100
    ax2.fill_between(equity_curve.index, drawdown.values, 0, alpha=0.3, color='red')
    ax2.plot(equity_curve.index, drawdown.values, color='red', linewidth=1)
    ax2.set_title("Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True, alpha=0.3)
    
    # 3. Position fraction
    ax3 = axes[2]
    ax3.plot(pos_frac_series.index, pos_frac_series.values, label="Position Fraction", linewidth=1)
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=0.5)
    ax3.set_title("Position Fraction")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Position Fraction")
    ax3.set_ylim(-1.1, 1.1)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Color code by strategy if available
    if not trades_df.empty and "strategy" in trades_df.columns:
        for strategy in trades_df["strategy"].unique():
            strategy_trades = trades_df[trades_df["strategy"] == strategy]
            if not strategy_trades.empty:
                trade_times = pd.to_datetime(strategy_trades["time"])
                trade_positions = strategy_trades["new_frac"].values
                ax3.scatter(trade_times, trade_positions, label=f"{strategy} trades", s=30, alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_strategy_periods(
    results: Dict[str, Any],
    title: str = "Strategy Periods",
    save_path: Optional[str] = None,
    show: bool = True,
):
    """
    Plot strategy periods over time.
    
    Args:
        results: Backtest results dictionary
        title: Plot title
        save_path: Optional path to save figure
        show: Whether to display the plot
    """
    equity_curve = results["equity_curve"]
    strategy_periods = results.get("strategy_period_history", [])
    
    if not strategy_periods:
        logger.warning("No strategy periods to plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_title(title)
    
    # Plot equity curve
    ax.plot(equity_curve.index, equity_curve.values, label="Equity", linewidth=2, color='black', alpha=0.3)
    
    # Color code by strategy
    strategy_colors = {
        "tsmom": "blue",
        "zscore_mr": "green",
        "adaptive_trend_fusion": "orange",
    }
    
    for period in strategy_periods:
        strategy = period.get("strategy", "unknown")
        start_bar = period.get("start_bar", 0)
        end_bar = period.get("end_bar", len(equity_curve))
        return_pct = period.get("return", 0) * 100
        
        # Get time range
        if start_bar < len(equity_curve) and end_bar <= len(equity_curve):
            start_time = equity_curve.index[start_bar] if start_bar < len(equity_curve.index) else equity_curve.index[0]
            end_time = equity_curve.index[min(end_bar - 1, len(equity_curve.index) - 1)]
            
            color = strategy_colors.get(strategy, "gray")
            ax.axvspan(start_time, end_time, alpha=0.2, color=color, label=f"{strategy} ({return_pct:.2f}%)")
    
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Strategy periods plot saved to {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


# ============================================================
#          Main Execution Function
# ============================================================

async def run_full_agent_backtest(
    coin: str = "BTC",
    interval: str = "1m",
    lookback: int = 0,
    initial_equity: float = 1000.0,
    model_name: str = "openai/gemini-3-flash-preview",
    data_path: Optional[str] = None,
    output_dir: str = "workdir/agent_backtest",
    save_results: bool = True,
    plot_results: bool = True,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    **backtest_kwargs,
) -> Dict[str, Any]:
    """
    Complete workflow: load data, run backtest, save results, and plot.
    
    Args:
        coin: Coin symbol
        interval: K-line interval
        lookback: Number of candles to load (0 = all)
        initial_equity: Initial equity
        model_name: LLM model name
        data_path: Path to data directory
        output_dir: Output directory for results
        save_results: Whether to save results to JSON
        plot_results: Whether to plot results
        start_time: Start time filter (ISO format string, e.g., "2024-01-01T00:00:00")
        end_time: End time filter (ISO format string, e.g., "2024-12-31T23:59:59")
        **backtest_kwargs: Additional arguments for run_agent_backtest
    
    Returns:
        Backtest results dictionary
    """
    # 1. Load data
    logger.info(f"Loading data for {coin} {interval}...")
    if start_time:
        logger.info(f"Start time filter: {start_time}")
    if end_time:
        logger.info(f"End time filter: {end_time}")
    df = load_data_from_jsonl(
        coin=coin, 
        interval=interval, 
        lookback=lookback, 
        data_path=data_path,
        start_time=start_time,
        end_time=end_time,
    )
    logger.info(f"Loaded {len(df)} candles from {df['t'].iloc[0]} to {df['t'].iloc[-1]}")
    
    # 2. Initialize agent strategy
    logger.info("Initializing agent strategy...")
    agent = AgentStrategy(
        model_name=model_name,
        data_path=data_path,
        trading_horizon=60,
    )
    
    # 3. Run backtest
    logger.info("Running backtest...")
    results = await run_agent_backtest(
        df=df,
        agent_strategy=agent,
        initial_equity=initial_equity,
        **backtest_kwargs,
    )
    
    # 4. Save results
    if save_results:
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f"agent_backtest_{coin}_{interval}_{timestamp}.json")
        save_trading_results(results, json_path)
    
    # 5. Plot results
    if plot_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(output_dir, f"agent_backtest_{coin}_{interval}_{timestamp}.png")
        plot_backtest_results(
            results,
            title=f"Agent Strategy Backtest - {coin} {interval}",
            save_path=plot_path,
            show=False,
        )
        
        # Plot strategy periods
        periods_plot_path = os.path.join(output_dir, f"strategy_periods_{coin}_{interval}_{timestamp}.png")
        plot_strategy_periods(
            results,
            title=f"Strategy Periods - {coin} {interval}",
            save_path=periods_plot_path,
            show=False,
        )
    
    logger.info("Backtest completed!")
    logger.info(f"Final equity: {results['stats']['final_equity']:.2f}")
    logger.info(f"Total return: {results['stats']['total_return']:.4f}")
    logger.info(f"Sharpe ratio: {results['stats']['sharpe']:.4f}")
    logger.info(f"Max drawdown: {results['stats']['max_drawdown']:.4f}")
    
    return results


# ============================================================
#          Main Entry Point
# ============================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run LLM Agent Trading Strategy Backtest")
    parser.add_argument(
        "--config",
        default=os.path.join(root, "configs", "base.py"),
        help="config file path"
    )
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    
    # Backtest-specific arguments (can be overridden via cfg-options)
    parser.add_argument("--coin", type=str, default="BTC", help="Coin symbol (e.g., BTC)")
    parser.add_argument("--interval", type=str, default="1m", help="K-line interval (e.g., 1m)")
    parser.add_argument("--lookback", type=int, default=0, help="Number of candles to load (0 = all)")
    parser.add_argument("--initial-equity", type=float, default=2000.0, help="Initial equity")
    parser.add_argument("--model-name", type=str, default="openrouter/gemini-3-flash-preview", help="LLM model name")
    parser.add_argument("--data-path", type=str, default="/Users/wentaozhang/workspace/RA/AgentWorld/workdir/crypto/crypto_binance_price_1min", help="Path to data directory")
    parser.add_argument("--output-dir", type=str, default="workdir/agent_backtest", help="Output directory")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to JSON")
    parser.add_argument("--no-plot", action="store_true", help="Don't plot results")
    parser.add_argument("--start-index", type=int, default=200, help="Starting index for backtest")
    parser.add_argument("--min-hold-bars", type=int, default=450, help="Minimum bars to hold position")
    parser.add_argument("--start-time", type=str, default="2025-12-17 00:00:00", help="Start time filter (ISO format, e.g., '2024-01-01 00:00:00')")
    parser.add_argument("--end-time", type=str, default="2025-12-21 23:59:59", help="End time filter (ISO format, e.g., '2024-12-31 23:59:59')")
    
    args = parser.parse_args()
    return args


async def main():
    """Main entry point for running agent backtest."""
    args = parse_args()
    
    # Initialize configuration and logger
    config.initialize(config_path=args.config, args=args)
    logger.initialize(config=config)
    logger.info(f"| Config: {config.pretty_text}")
    
    # Initialize model manager
    await model_manager.initialize()
    logger.info(f"| Model manager initialized: {await model_manager.list()}")
    
    # Get configuration values (from config file or command line args)
    # Use getattr with defaults for config values
    coin = getattr(config, 'coin', None) or args.coin or "BTC"
    interval = getattr(config, 'interval', None) or args.interval or "1m"
    lookback = getattr(config, 'lookback', None) if hasattr(config, 'lookback') else (args.lookback if args.lookback is not None else 0)
    initial_equity = getattr(config, 'initial_equity', None) if hasattr(config, 'initial_equity') else (args.initial_equity if args.initial_equity is not None else 2000.0)
    model_name = getattr(config, 'model_name', None) or args.model_name or "openai/gemini-3-flash-preview"
    data_path = getattr(config, 'data_path', None) or args.data_path
    output_dir = getattr(config, 'output_dir', None) or args.output_dir or "workdir/agent_backtest"
    start_index = getattr(config, 'start_index', None) if hasattr(config, 'start_index') else (args.start_index if args.start_index is not None else 200)
    min_hold_bars = getattr(config, 'min_hold_bars', None) if hasattr(config, 'min_hold_bars') else (args.min_hold_bars if args.min_hold_bars is not None else 450)
    start_time = getattr(config, 'start_time', None) if hasattr(config, 'start_time') else (args.start_time if args.start_time else None)
    end_time = getattr(config, 'end_time', None) if hasattr(config, 'end_time') else (args.end_time if args.end_time else None)
    save_results = not args.no_save
    plot_results = not args.no_plot
    
    logger.info("Starting agent backtest...")
    logger.info(f"| Coin: {coin}, Interval: {interval}, Lookback: {lookback}")
    logger.info(f"| Initial Equity: {initial_equity}, Model: {model_name}")
    if start_time:
        logger.info(f"| Start Time: {start_time}")
    if end_time:
        logger.info(f"| End Time: {end_time}")
    
    # Run backtest
    results = await run_full_agent_backtest(
        coin=coin,
        interval=interval,
        lookback=lookback,
        initial_equity=initial_equity,
        model_name=model_name,
        data_path=data_path,
        output_dir=output_dir,
        save_results=save_results,
        plot_results=plot_results,
        start_index=start_index,
        min_hold_bars=min_hold_bars,
        start_time=start_time,
        end_time=end_time,
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Backtest Summary")
    print("="*60)
    print(f"Final Equity: {results['stats']['final_equity']:.2f}")
    print(f"Total Return: {results['stats']['total_return']:.4f} ({results['stats']['total_return']*100:.2f}%)")
    print(f"Sharpe Ratio: {results['stats']['sharpe']:.4f}")
    print(f"Max Drawdown: {results['stats']['max_drawdown']:.4f} ({results['stats']['max_drawdown']*100:.2f}%)")
    if 'num_trades' in results['stats']:
        print(f"Number of Trades: {results['stats']['num_trades']}")
        print(f"Win Rate: {results['stats']['win_rate']:.4f} ({results['stats']['win_rate']*100:.2f}%)")
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())
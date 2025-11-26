"""
Momentum and Mean Reversion Hybrid Strategy.

This strategy implements:
1. Momentum Factor - Short window returns (3-10 minutes)
2. Mean Reversion Factor - Short-term SMA deviation (15 minutes)
3. Signal Logic - Combined momentum and mean reversion signals
4. Stop Loss / Take Profit - Fixed or ATR-based risk management
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging

# Try to import logger, fallback to standard logging if not available
try:
    from src.logger import logger
except ImportError:
    logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class FactorSignals:
    """Factor signals output."""
    momentum_signal: SignalType
    mean_reversion_signal: SignalType
    combined_signal: SignalType
    momentum_value: float
    mean_reversion_value: float
    confidence: float  # 0-1, higher means more confident


@dataclass
class RiskManagement:
    """Risk management parameters."""
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    stop_loss_pct: Optional[float] = None  # Percentage-based stop loss
    take_profit_pct: Optional[float] = None  # Percentage-based take profit
    use_atr_stop: bool = False  # Use ATR-based stop loss
    atr_multiplier: float = 2.0  # ATR multiplier for stop loss


class MomentumMeanReversionStrategy:
    """
    Momentum and Mean Reversion Hybrid Strategy.
    
    This strategy combines:
    - Momentum factor: Short window returns (3, 5, 10 minutes) to capture rapid price movements
    - Mean reversion factor: Deviation from 15-minute SMA for stable baseline reference
    - Combined signal logic: Executes trades when both factors agree
    - Risk management: Fixed or ATR-based stop loss and take profit
    """
    
    def __init__(
        self,
        momentum_windows: List[int] = [3, 5, 10],
        sma_window: int = 15,
        deviation_threshold: float = 0.0015,  # 0.15% default
        momentum_threshold: float = 0.0008,  # 0.08% default
        use_atr_stop: bool = True,
        atr_multiplier: float = 2.0,
        fixed_stop_loss_pct: Optional[float] = 0.0025,  # 0.25% default
        fixed_take_profit_pct: Optional[float] = 0.004,  # 0.4% default
    ):
        """
        Initialize the Momentum and Mean Reversion Hybrid Strategy.
        
        Optimized parameters for minute-level trading:
        - Short momentum windows (3, 5, 10 minutes) to capture rapid price movements
        - Medium SMA window (15 minutes) for stable mean reversion reference
        - Tight thresholds (0.08-0.15%) to filter noise while maintaining sensitivity
        
        Args:
            momentum_windows: List of time windows (minutes) for momentum calculation [3, 5, 10]
                              Short windows (3-10 min) capture minute-level trends effectively
            sma_window: Window for SMA calculation (minutes), default 15 for stable baseline
            deviation_threshold: Threshold for mean reversion signal (0.0015 = 0.15%)
                                Adjusted for minute-level volatility
            momentum_threshold: Threshold for momentum signal (0.0008 = 0.08%)
                               Lower threshold for faster signal generation
            use_atr_stop: Whether to use ATR-based stop loss (recommended for volatility)
            atr_multiplier: Multiplier for ATR-based stop loss (2.0 = 2x ATR)
            fixed_stop_loss_pct: Fixed stop loss percentage (0.0025 = 0.25%, tight for minute trading)
            fixed_take_profit_pct: Fixed take profit percentage (0.004 = 0.4%, conservative for quick exits)
        """
        self.momentum_windows = momentum_windows
        self.sma_window = sma_window
        self.deviation_threshold = deviation_threshold
        self.momentum_threshold = momentum_threshold
        self.use_atr_stop = use_atr_stop
        self.atr_multiplier = atr_multiplier
        self.fixed_stop_loss_pct = fixed_stop_loss_pct
        self.fixed_take_profit_pct = fixed_take_profit_pct
        
        logger.info(f"| 📊 MomentumMeanReversionStrategy initialized (optimized for minute-level trading):")
        logger.info(f"|   - Momentum windows: {momentum_windows} minutes (short windows for rapid trend capture)")
        logger.info(f"|   - SMA window: {sma_window} minutes (stable baseline for mean reversion)")
        logger.info(f"|   - Deviation threshold: {deviation_threshold*100:.2f}% (adjusted for minute volatility)")
        logger.info(f"|   - Momentum threshold: {momentum_threshold*100:.2f}% (sensitive to minute movements)")
        logger.info(f"|   - Use ATR stop: {use_atr_stop} (volatility-adaptive risk management)")
        logger.info(f"|   - Fixed stop loss: {fixed_stop_loss_pct*100 if fixed_stop_loss_pct else None}% (tight for minute trading)")
        logger.info(f"|   - Fixed take profit: {fixed_take_profit_pct*100 if fixed_take_profit_pct else None}% (quick exits)")
    
    def calculate_momentum_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum factor using short window returns (optimized for minute-level data).
        
        Uses multiple short windows (3, 5, 10 minutes) to capture rapid price movements
        typical in minute-level trading. The average across windows provides a robust signal.
        
        Formula: momentum = (close_t - close_{t-n}) / close_{t-n}
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            DataFrame with momentum values for each window (momentum_3m, momentum_5m, momentum_10m)
            and average momentum (momentum_avg)
        """
        df = df.copy()
        
        for n in self.momentum_windows:
            if len(df) < n + 1:
                df[f"momentum_{n}m"] = np.nan
                continue
            
            # Calculate momentum: (close_t - close_{t-n}) / close_{t-n}
            close_t = df["close"]
            close_t_n = df["close"].shift(n)
            
            momentum = (close_t - close_t_n) / close_t_n
            df[f"momentum_{n}m"] = momentum
        
        # Calculate average momentum across all windows
        momentum_cols = [f"momentum_{n}m" for n in self.momentum_windows]
        df["momentum_avg"] = df[momentum_cols].mean(axis=1)
        
        return df
    
    def calculate_mean_reversion_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate mean reversion factor using deviation from short-term SMA (15-minute window).
        
        The 15-minute SMA provides a stable reference point for minute-level price movements.
        When price deviates significantly from this baseline, mean reversion opportunities arise.
        
        Formula: deviation = (close_t - sma_15m) / sma_15m
        
        Args:
            df: DataFrame with 'close' column
            
        Returns:
            DataFrame with SMA (sma_short) and deviation percentage (deviation)
        """
        df = df.copy()
        
        if len(df) < self.sma_window:
            df["sma_short"] = np.nan
            df["deviation"] = np.nan
            return df
        
        # Calculate short-term SMA
        df["sma_short"] = talib.SMA(df["close"], timeperiod=self.sma_window)
        
        # Calculate deviation from SMA
        df["deviation"] = (df["close"] - df["sma_short"]) / df["sma_short"]
        
        return df
    
    def generate_signals(
        self, 
        df: pd.DataFrame,
        current_price: Optional[float] = None
    ) -> FactorSignals:
        """
        Generate trading signals based on momentum and mean reversion factors.
        
        Signal Logic:
        - Momentum signal: 
          * BUY if momentum > threshold
          * SELL if momentum < -threshold
          * HOLD otherwise
        - Mean reversion signal:
          * BUY if price is below SMA by more than threshold (oversold)
          * SELL if price is above SMA by more than threshold (overbought)
          * HOLD otherwise
        - Combined signal:
          * BUY if both signals agree on BUY
          * SELL if both signals agree on SELL
          * HOLD otherwise
        
        Args:
            df: DataFrame with calculated factors
            current_price: Current price (if None, uses last close price)
            
        Returns:
            FactorSignals object with signals and values
        """
        if df.empty:
            return FactorSignals(
                momentum_signal=SignalType.HOLD,
                mean_reversion_signal=SignalType.HOLD,
                combined_signal=SignalType.HOLD,
                momentum_value=0.0,
                mean_reversion_value=0.0,
                confidence=0.0
            )
        
        # Get latest values
        latest = df.iloc[-1]
        
        # Get current price
        if current_price is None:
            current_price = latest["close"]
        
        # Calculate momentum signal
        momentum_value = latest.get("momentum_avg", 0.0)
        if pd.isna(momentum_value):
            momentum_value = 0.0
        
        if momentum_value > self.momentum_threshold:
            momentum_signal = SignalType.BUY
        elif momentum_value < -self.momentum_threshold:
            momentum_signal = SignalType.SELL
        else:
            momentum_signal = SignalType.HOLD
        
        # Calculate mean reversion signal
        deviation_value = latest.get("deviation", 0.0)
        if pd.isna(deviation_value):
            deviation_value = 0.0
        
        if deviation_value < -self.deviation_threshold:
            # Price is below SMA (oversold) -> BUY
            mean_reversion_signal = SignalType.BUY
        elif deviation_value > self.deviation_threshold:
            # Price is above SMA (overbought) -> SELL
            mean_reversion_signal = SignalType.SELL
        else:
            mean_reversion_signal = SignalType.HOLD
        
        # Combine signals
        if momentum_signal == mean_reversion_signal and momentum_signal != SignalType.HOLD:
            combined_signal = momentum_signal
            # High confidence when both agree
            confidence = 0.8 + min(abs(momentum_value) / self.momentum_threshold, abs(deviation_value) / self.deviation_threshold) * 0.2
            confidence = min(confidence, 1.0)
        elif momentum_signal != SignalType.HOLD and mean_reversion_signal == SignalType.HOLD:
            # Only momentum signal
            combined_signal = SignalType.HOLD
            confidence = abs(momentum_value) / self.momentum_threshold * 0.5
        elif mean_reversion_signal != SignalType.HOLD and momentum_signal == SignalType.HOLD:
            # Only mean reversion signal
            combined_signal = SignalType.HOLD
            confidence = abs(deviation_value) / self.deviation_threshold * 0.5
        else:
            # Both neutral or conflicting
            combined_signal = SignalType.HOLD
            confidence = 0.0
        
        return FactorSignals(
            momentum_signal=momentum_signal,
            mean_reversion_signal=mean_reversion_signal,
            combined_signal=combined_signal,
            momentum_value=momentum_value,
            mean_reversion_value=deviation_value,
            confidence=confidence
        )
    
    def calculate_risk_management(
        self,
        df: pd.DataFrame,
        entry_price: float,
        signal: SignalType,
        current_price: Optional[float] = None
    ) -> RiskManagement:
        """
        Calculate risk management parameters (stop loss and take profit).
        
        Args:
            df: DataFrame with OHLCV data
            entry_price: Entry price for the position
            signal: Trading signal (BUY or SELL)
            current_price: Current price (if None, uses last close price)
            
        Returns:
            RiskManagement object with stop loss and take profit prices
        """
        if df.empty:
            return RiskManagement()
        
        if current_price is None:
            current_price = df["close"].iloc[-1]
        
        stop_loss_price = None
        take_profit_price = None
        stop_loss_pct = None
        take_profit_pct = None
        
        # Calculate ATR if needed
        atr_value = None
        if self.use_atr_stop and len(df) >= 14:
            try:
                atr_value = talib.ATR(df["high"], df["low"], df["close"], timeperiod=14).iloc[-1]
            except Exception:
                atr_value = None
        
        if signal == SignalType.BUY:
            # Long position
            if self.use_atr_stop and atr_value is not None:
                # ATR-based stop loss
                stop_loss_price = entry_price - (atr_value * self.atr_multiplier)
                take_profit_price = entry_price + (atr_value * self.atr_multiplier * 1.5)  # 1.5x for take profit
            else:
                # Fixed percentage-based
                if self.fixed_stop_loss_pct:
                    stop_loss_price = entry_price * (1 - self.fixed_stop_loss_pct)
                    stop_loss_pct = self.fixed_stop_loss_pct
                if self.fixed_take_profit_pct:
                    take_profit_price = entry_price * (1 + self.fixed_take_profit_pct)
                    take_profit_pct = self.fixed_take_profit_pct
        elif signal == SignalType.SELL:
            # Short position
            if self.use_atr_stop and atr_value is not None:
                # ATR-based stop loss
                stop_loss_price = entry_price + (atr_value * self.atr_multiplier)
                take_profit_price = entry_price - (atr_value * self.atr_multiplier * 1.5)  # 1.5x for take profit
            else:
                # Fixed percentage-based
                if self.fixed_stop_loss_pct:
                    stop_loss_price = entry_price * (1 + self.fixed_stop_loss_pct)
                    stop_loss_pct = self.fixed_stop_loss_pct
                if self.fixed_take_profit_pct:
                    take_profit_price = entry_price * (1 - self.fixed_take_profit_pct)
                    take_profit_pct = self.fixed_take_profit_pct
        
        return RiskManagement(
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            use_atr_stop=self.use_atr_stop,
            atr_multiplier=self.atr_multiplier
        )
    
    def _generate_market_description(
        self,
        signals: FactorSignals,
        current_price: float,
        risk_management: Optional[RiskManagement] = None
    ) -> str:
        """
        Generate natural language description of market conditions based on strategy signals.
        
        Args:
            signals: FactorSignals object with trading signals
            current_price: Current market price
            risk_management: Optional RiskManagement object
            
        Returns:
            Natural language description of market conditions and trading recommendation
        """
        momentum_value = signals.momentum_value
        deviation_value = signals.mean_reversion_value
        confidence = signals.confidence
        
        # Build market description based on signal values
        description_parts = []
        
        # Describe momentum condition
        if abs(momentum_value) < self.momentum_threshold * 0.5:
            description_parts.append("The market shows weak momentum with minimal price movement.")
        elif momentum_value > self.momentum_threshold * 2:
            description_parts.append("The market exhibits strong upward momentum with significant price appreciation.")
        elif momentum_value > self.momentum_threshold:
            description_parts.append("The market shows positive momentum with moderate upward price movement.")
        elif momentum_value < -self.momentum_threshold * 2:
            description_parts.append("The market exhibits strong downward momentum with significant price decline.")
        elif momentum_value < -self.momentum_threshold:
            description_parts.append("The market shows negative momentum with moderate downward price movement.")
        
        # Describe mean reversion condition (using 15-minute SMA)
        if abs(deviation_value) < self.deviation_threshold * 0.5:
            description_parts.append("The price is trading near its 15-minute moving average, indicating balanced market conditions.")
        elif deviation_value > self.deviation_threshold * 2:
            description_parts.append(f"The price is significantly above its 15-minute moving average (deviation: {deviation_value:.2%}), suggesting potential overbought conditions.")
        elif deviation_value > self.deviation_threshold:
            description_parts.append(f"The price is moderately above its 15-minute moving average (deviation: {deviation_value:.2%}), indicating slight overvaluation.")
        elif deviation_value < -self.deviation_threshold * 2:
            description_parts.append(f"The price is significantly below its 15-minute moving average (deviation: {deviation_value:.2%}), suggesting potential oversold conditions.")
        elif deviation_value < -self.deviation_threshold:
            description_parts.append(f"The price is moderately below its 15-minute moving average (deviation: {deviation_value:.2%}), indicating slight undervaluation.")
        
        # Describe signal alignment
        if signals.momentum_signal == signals.mean_reversion_signal and signals.momentum_signal != SignalType.HOLD:
            description_parts.append("Both momentum and mean reversion indicators align, suggesting a strong trading opportunity.")
        elif signals.momentum_signal != SignalType.HOLD and signals.mean_reversion_signal == SignalType.HOLD:
            description_parts.append("Momentum indicator provides a signal, but mean reversion suggests neutral conditions.")
        elif signals.mean_reversion_signal != SignalType.HOLD and signals.momentum_signal == SignalType.HOLD:
            description_parts.append("Mean reversion indicator provides a signal, but momentum suggests neutral conditions.")
        elif signals.momentum_signal != SignalType.HOLD and signals.mean_reversion_signal != SignalType.HOLD and signals.momentum_signal != signals.mean_reversion_signal:
            # Signals conflict (one BUY, one SELL)
            description_parts.append(f"Momentum and mean reversion indicators conflict (momentum: {signals.momentum_signal.value}, mean reversion: {signals.mean_reversion_signal.value}), indicating mixed market signals and high uncertainty.")
        else:
            description_parts.append("Both indicators suggest neutral market conditions with no clear directional bias.")
        
        # Add confidence assessment
        if confidence > 0.8:
            description_parts.append(f"The signal has high confidence ({confidence:.0%}), indicating strong conviction.")
        elif confidence > 0.5:
            description_parts.append(f"The signal has moderate confidence ({confidence:.0%}), suggesting reasonable conviction.")
        elif confidence > 0.2:
            description_parts.append(f"The signal has low confidence ({confidence:.0%}), indicating weak conviction.")
        elif confidence == 0.0 and signals.momentum_signal != SignalType.HOLD and signals.mean_reversion_signal != SignalType.HOLD and signals.momentum_signal != signals.mean_reversion_signal:
            # Signal conflict
            description_parts.append("The conflicting signals result in zero confidence, making trading inadvisable at this time.")
        else:
            description_parts.append("The signal has very low confidence, suggesting high uncertainty.")
        
        # Add trading recommendation
        if signals.combined_signal == SignalType.BUY:
            recommendation = f"Recommendation: Consider opening a LONG position at current price of {current_price:.2f}."
            if risk_management:
                if risk_management.stop_loss_price:
                    recommendation += f" Set stop loss at {risk_management.stop_loss_price:.2f} to limit potential losses."
                if risk_management.take_profit_price:
                    recommendation += f" Set take profit at {risk_management.take_profit_price:.2f} to secure potential gains."
        elif signals.combined_signal == SignalType.SELL:
            recommendation = f"Recommendation: Consider opening a SHORT position at current price of {current_price:.2f}."
            if risk_management:
                if risk_management.stop_loss_price:
                    recommendation += f" Set stop loss at {risk_management.stop_loss_price:.2f} to limit potential losses."
                if risk_management.take_profit_price:
                    recommendation += f" Set take profit at {risk_management.take_profit_price:.2f} to secure potential gains."
        else:
            recommendation = f"Recommendation: HOLD - Maintain current position or wait for clearer signals. Current price: {current_price:.2f}."
        
        description_parts.append(recommendation)
        
        # Combine all parts
        full_description = " ".join(description_parts)
        
        return full_description
    
    def analyze(
        self,
        df: pd.DataFrame,
        current_price: Optional[float] = None
    ) -> str:
        """
        Complete analysis: calculate factors, generate signals, and return natural language description.
        
        Args:
            df: DataFrame with OHLCV data (must have 'close', 'high', 'low' columns)
            current_price: Current price (if None, uses last close price)
            
        Returns:
            Natural language description of market conditions and trading recommendation
        """
        if df.empty:
            return "Unable to analyze market conditions: No historical data available. Please ensure sufficient market data is provided."

        # Calculate factors
        df_with_factors = df.copy()
        df_with_factors = self.calculate_momentum_factor(df_with_factors)
        df_with_factors = self.calculate_mean_reversion_factor(df_with_factors)

        # Generate signals
        signals = self.generate_signals(df_with_factors, current_price)

        # Get current price
        if current_price is None:
            current_price = df_with_factors["close"].iloc[-1]

        # Calculate risk management if signal is not HOLD
        risk_management = None
        if signals.combined_signal != SignalType.HOLD:
            entry_price = current_price
            risk_management = self.calculate_risk_management(
                df_with_factors,
                entry_price,
                signals.combined_signal,
                current_price
            )

        # Generate natural language description
        description = self._generate_market_description(
            signals,
            current_price,
            risk_management
        )

        return description


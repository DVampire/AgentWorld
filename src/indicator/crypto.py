from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from pandas import DataFrame
from copy import deepcopy

from src.registry import INDICATOR
from src.logger import logger
from src.indicator.types import Indicator

EPS = 1e-12

@INDICATOR.register_module(force=True)
class Crypto(Indicator):
    """
    Crypto indicator for regime-based trading strategies.
    
    This indicator computes various technical factors used in crypto trading strategies,
    including moving averages, momentum, volatility, and trend indicators.
    Referenced from: tests/hl_backtest/regime_based.py
    """

    def __init__(self, windows: List[int] = None, level: str = None):
        super().__init__()
        
        self.windows: List[int] = windows if windows is not None else [5, 10, 20, 30, 60]
        self.level = level if level is not None else '1day'

        self.indicators_name: List[str] = []
        
        # Moving averages
        for window in self.windows:
            self.indicators_name.extend([
                f'sma_{window}',
                f'ema_{window}',
            ])
        
        # Momentum factors
        for window in self.windows:
            self.indicators_name.extend([
                f'momentum_{window}',
                f'zscore_{window}',
            ])
        
        # Volatility and trend factors
        self.indicators_name.extend([
            'rsi_14',
            'atr_14',
            'adx_14',
        ])
        
        # Swing points
        for window in [10, 20, 30]:
            self.indicators_name.extend([
                f'swing_high_{window}',
                f'swing_low_{window}',
            ])
        
        # Volume factors
        for window in self.windows:
            self.indicators_name.extend([
                f'volume_surge_{window}',
                f'volume_ma_{window}',
            ])
        
        self.indicators_name = list(sorted(self.indicators_name))

    async def _sma(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """Calculate Simple Moving Average (SMA)"""
        df = deepcopy(df)
        factors_info = {}
        
        windows = windows if windows is not None else self.windows
        for window in windows:
            col_name = f'sma_{window}'
            df[col_name] = df['close'].rolling(window=window).mean()
            factors_info[col_name] = f'close.rolling({window}).mean()'
        
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _ema(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """Calculate Exponential Moving Average (EMA)"""
        df = deepcopy(df)
        factors_info = {}
        
        windows = windows if windows is not None else self.windows
        for window in windows:
            col_name = f'ema_{window}'
            df[col_name] = df['close'].ewm(span=window, adjust=False).mean()
            factors_info[col_name] = f'close.ewm(span={window}, adjust=False).mean()'
        
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _momentum(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """Calculate momentum: (P_t / P_{t-k}) - 1"""
        df = deepcopy(df)
        factors_info = {}
        
        windows = windows if windows is not None else self.windows
        for window in windows:
            col_name = f'momentum_{window}'
            df[col_name] = df['close'] / df['close'].shift(window) - 1.0
            factors_info[col_name] = f'close / close.shift({window}) - 1'
        
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _zscore(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """Calculate Z-score: (P - MA) / std"""
        df = deepcopy(df)
        factors_info = {}
        
        windows = windows if windows is not None else self.windows
        for window in windows:
            col_name = f'zscore_{window}'
            ma = df['close'].rolling(window=window).mean()
            std = df['close'].rolling(window=window).std(ddof=0)
            df[col_name] = (df['close'] - ma) / (std + EPS)
            factors_info[col_name] = f'(close - close.rolling({window}).mean()) / close.rolling({window}).std()'
        
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _rsi(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """Calculate Relative Strength Index (RSI)"""
        df = deepcopy(df)
        factors_info = {}
        
        period = 14
        col_name = f'rsi_{period}'
        
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + EPS)
        df[col_name] = 100 - (100 / (1 + rs))
        
        factors_info[col_name] = f'RSI({period})'
        df = df[[col_name]].copy()
        return df, factors_info

    async def _atr(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """Calculate Average True Range (ATR)"""
        df = deepcopy(df)
        factors_info = {}
        
        period = 14
        col_name = f'atr_{period}'
        
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df[col_name] = true_range.rolling(window=period).mean()
        
        factors_info[col_name] = f'ATR({period})'
        df = df[[col_name]].copy()
        return df, factors_info

    async def _adx(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """Calculate Average Directional Index (ADX)"""
        df = deepcopy(df)
        factors_info = {}
        
        period = 14
        col_name = f'adx_{period}'
        
        # Calculate +DM and -DM
        plus_dm = df['high'].diff()
        minus_dm = -df['low'].diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # True Range
        tr1 = df['high'] - df['low']
        tr2 = np.abs(df['high'] - df['close'].shift())
        tr3 = np.abs(df['low'] - df['close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Smooth ATR
        atr = tr.rolling(window=period).mean()
        atr_safe = atr.replace(0, np.nan)
        
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_safe)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_safe)
        
        # DX and ADX
        di_sum = plus_di + minus_di
        di_sum = di_sum.replace(0, np.nan)
        dx = 100 * np.abs(plus_di - minus_di) / di_sum
        df[col_name] = dx.rolling(window=period).mean()
        
        factors_info[col_name] = f'ADX({period})'
        df = df[[col_name]].copy()
        return df, factors_info

    async def _swing_points(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """Find swing high and low points"""
        df = deepcopy(df)
        factors_info = {}
        
        lookback_windows = [10, 20, 30]
        for lookback in lookback_windows:
            high_col = f'swing_high_{lookback}'
            low_col = f'swing_low_{lookback}'
            
            # Calculate rolling max and min
            df[high_col] = df['high'].rolling(window=lookback * 2).max()
            df[low_col] = df['low'].rolling(window=lookback * 2).min()
            
            # Normalize by current price
            df[high_col] = df[high_col] / (df['close'] + EPS)
            df[low_col] = df[low_col] / (df['close'] + EPS)
            
            factors_info[high_col] = f'high.rolling({lookback * 2}).max() / close'
            factors_info[low_col] = f'low.rolling({lookback * 2}).min() / close'
        
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _volume_surge(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """Calculate volume surge indicator"""
        df = deepcopy(df)
        factors_info = {}
        
        windows = windows if windows is not None else self.windows
        multiplier = 1.5
        
        for window in windows:
            col_name = f'volume_surge_{window}'
            avg_vol = df['volume'].rolling(window=window).mean()
            df[col_name] = df['volume'] / (avg_vol + EPS) >= multiplier
            df[col_name] = df[col_name].astype(float)
            factors_info[col_name] = f'(volume >= volume.rolling({window}).mean() * {multiplier})'
        
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def _volume_ma(self, df: DataFrame, windows: List[int] = None, level: str = '1day') -> Tuple[DataFrame, Dict[str, str]]:
        """Calculate volume moving average"""
        df = deepcopy(df)
        factors_info = {}
        
        windows = windows if windows is not None else self.windows
        for window in windows:
            col_name = f'volume_ma_{window}'
            df[col_name] = df['volume'].rolling(window=window).mean() / (df['volume'] + EPS)
            factors_info[col_name] = f'volume.rolling({window}).mean() / volume'
        
        df = df[[col for col in factors_info.keys()]].copy()
        return df, factors_info

    async def __call__(self, df: pd.DataFrame, windows: List[int] = None, level: str = None) -> pd.DataFrame:
        """
        Run the crypto factors computation on the given DataFrame.

        :param df: Input DataFrame containing crypto price data.
        :param windows: List of window sizes for rolling calculations (if applicable).
        :param level: Level of detail for time-based factors.
        :return: DataFrame with computed factors.
        """
        assert 'timestamp' in df.columns, "DataFrame must contain a 'timestamp' column."
        assert 'open' in df.columns, "DataFrame must contain an 'open' column."
        assert 'high' in df.columns, "DataFrame must contain a 'high' column."
        assert 'low' in df.columns, "DataFrame must contain a 'low' column."
        assert 'close' in df.columns, "DataFrame must contain a 'close' column."
        assert 'volume' in df.columns, "DataFrame must contain a 'volume' column."

        windows = windows if windows is not None else self.windows
        level = level if level is not None else self.level

        factor_methods = {
            'sma': self._sma,
            'ema': self._ema,
            'momentum': self._momentum,
            'zscore': self._zscore,
            'rsi': self._rsi,
            'atr': self._atr,
            'adx': self._adx,
            'swing_points': self._swing_points,
            'volume_surge': self._volume_surge,
            'volume_ma': self._volume_ma,
        }

        factors_info = {}
        result_df = pd.DataFrame(index=df.index)
        result_df['timestamp'] = df['timestamp']

        for factor_name, method in factor_methods.items():
            factor_df, factor_info = await method(df, windows=windows, level=level)
            logger.info(f"Computed crypto factor: {list(factor_info.keys())} with windows: {windows} and level: {level}")
            result_df = pd.concat([result_df, factor_df], axis=1)
            factors_info.update(factor_info)

        result_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        result_df = result_df.fillna(0)

        assert sorted(factors_info.keys()) == self.indicators_name, \
            f"Factor names do not match. Expected: {self.indicators_name}, Got: {sorted(factors_info.keys())}"

        return result_df


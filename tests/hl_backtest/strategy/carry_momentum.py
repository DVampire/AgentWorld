"""Carry + Momentum Multi-Factor Strategy.

This strategy implements a robust multi-factor framework combining:
1. Carry/Basis/Funding Rate factor (return source)
2. Momentum factor (quality confirmation)
3. Volatility management (robustness core)
4. Trend filtering (directional gate)

Based on academic research and institutional practices for crypto perpetual trading.
"""

import numpy as np
import pandas as pd
from .base import PRICE_COL, VOL_COL


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """计算简单移动平均线（SMA）"""
    return series.rolling(window=period).mean()


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """计算平均真实波幅（ATR）"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr

# ==== 最小持仓计数器 ====
CM_LAST_SIDE = 0
CM_HOLD_BARS = 0


def carry_momentum_baseline(
    df: pd.DataFrame,
    current_pos: float,
    current_equity: float,
    MIN_HOLD_BARS: int = 0,
    # Carry/Basis/Funding Rate 参数（收益来源因子）
    FUNDING_RATE: float = 0.0,  # 当前 funding rate (需要从外部传入，通常是 8h rate)
    FUNDING_THRESHOLD: float = 0.00005,  # Funding rate 阈值 (0.005% = 0.00005)
    BASIS: float = 0.0,  # 当前基差 (perp - spot) / spot (需要从外部传入)
    BASIS_THRESHOLD: float = 0.0003,  # 基差阈值 (0.03% = 0.0003)
    USE_PRICE_MOMENTUM_AS_CARRY: bool = True,  # 如果数据缺失，使用价格动量作为 carry 代理
    # 趋势过滤参数（方向闸门）
    TREND_MA_PERIOD: int = 200,  # 趋势判断 SMA 周期
    TREND_FILTER_ENABLED: bool = True,  # 是否启用趋势过滤
    TREND_THRESHOLD: float = 0.001,  # 趋势判断阈值（价格相对 SMA 的偏离）
    # 动量因子参数（质量确认）
    MOMENTUM_LOOKBACK: int = 20,  # 动量回看周期
    MOMENTUM_THRESHOLD: float = 0.002,  # 动量阈值 (0.2% = 0.002)
    MOMENTUM_WEIGHT: float = 0.3,  # 动量因子权重（0-1）
    # 波动率管理参数（稳健核心）
    VOL_PERIOD: int = 20,  # 波动率计算周期
    VOL_THRESHOLD_HIGH: float = 0.08,  # 高波动阈值 (8% = 0.08)，超过此值降仓位
    VOL_THRESHOLD_LOW: float = 0.02,  # 低波动阈值 (2% = 0.02)，低于此值减少交易
    VOL_CONTROL_ENABLED: bool = True,  # 是否启用波动率控制
    VOL_SCALING_ENABLED: bool = True,  # 是否启用波动率仓位缩放
    # 风险控制参数
    MAX_LEVERAGE: float = 3.0,  # 最大杠杆
    MIN_CARRY_SCORE: float = 0.3,  # 最小 carry 得分才允许交易（0-1）
) -> float:
    """
    Carry + Momentum 多因子策略
    
    策略框架：
    1. **方向闸门（趋势因子）**：只做顺势方向
       - 价格 > SMA200 → 只允许做多
       - 价格 < SMA200 → 只允许做空
    
    2. **收益来源（Carry/基差/资金费率因子）**：决定"值不值得参与"
       - Funding > 0 或 Basis > 0 → 做空 perp（赚 carry）
       - Funding < 0 或 Basis < 0 → 做多 perp（赚 carry）
       - 如果数据缺失，使用价格动量作为代理
    
    3. **质量确认（动量因子）**：避免逆势抄底摸顶
       - 动量与趋势方向一致 → 加分
       - 动量与趋势方向相反 → 降权或禁止
    
    4. **稳健核心（波动率管理）**：决定仓位大小与是否暂停
       - 波动率 > 高阈值 → 降仓位或暂停交易
       - 波动率 < 低阈值 → 减少交易频率
       - 波动率适中 → 正常仓位
    
    参数说明：
    - FUNDING_RATE: 当前 funding rate（8h rate，需要从外部传入）
    - FUNDING_THRESHOLD: Funding rate 阈值
    - BASIS: 当前基差 (perp - spot) / spot（需要从外部传入）
    - MOMENTUM_LOOKBACK: 动量回看周期（默认 20）
    - MOMENTUM_WEIGHT: 动量因子权重（0-1，默认 0.3）
    - VOL_THRESHOLD_HIGH: 高波动阈值，超过此值降仓位
    - VOL_THRESHOLD_LOW: 低波动阈值，低于此值减少交易
    - MIN_CARRY_SCORE: 最小 carry 得分才允许交易（0-1）
    """
    global CM_LAST_SIDE, CM_HOLD_BARS
    
    # 基础检查
    if df is None or df.empty or current_equity <= 0:
        return 0.0
    
    if PRICE_COL not in df.columns:
        return 0.0
    
    # 检查数据长度
    max_period = max(TREND_MA_PERIOD, VOL_PERIOD, MOMENTUM_LOOKBACK)
    if len(df) < max_period + 10:
        return 0.0
    
    # 提取数据
    closes = df[PRICE_COL].astype(float)
    highs = df["h"].astype(float) if "h" in df.columns else closes
    lows = df["l"].astype(float) if "l" in df.columns else closes
    volumes = df[VOL_COL].astype(float) if VOL_COL in df.columns else pd.Series([1.0] * len(df))
    
    current_price = closes.iloc[-1]
    
    # ========== 1. 获取 Carry 数据 ==========
    # 尝试从 DataFrame 读取 funding_rate 和 basis（如果存在）
    if "funding_rate" in df.columns:
        current_funding_rate = float(df["funding_rate"].iloc[-1])
    else:
        current_funding_rate = FUNDING_RATE
    
    if "basis" in df.columns:
        current_basis = float(df["basis"].iloc[-1])
    else:
        current_basis = BASIS
    
    # 如果 funding_rate 和 basis 都是 0（数据缺失），使用价格动量作为替代
    if USE_PRICE_MOMENTUM_AS_CARRY and current_funding_rate == 0.0 and current_basis == 0.0:
        if len(closes) >= MOMENTUM_LOOKBACK:
            price_change = (closes.iloc[-1] - closes.iloc[-MOMENTUM_LOOKBACK]) / closes.iloc[-MOMENTUM_LOOKBACK]
            # 价格快速上涨 → 正 funding（做空信号）
            current_funding_rate = price_change * 0.1
    
    # ========== 2. 计算技术指标 ==========
    # 趋势指标
    sma_trend = calculate_sma(closes, TREND_MA_PERIOD).iloc[-1]
    
    # 波动率指标（使用 ATR 归一化）
    atr = calculate_atr(highs, lows, closes, VOL_PERIOD).iloc[-1]
    realized_vol = atr / current_price if current_price > 0 else 0.0
    
    # 动量指标
    if len(closes) >= MOMENTUM_LOOKBACK:
        momentum = (closes.iloc[-1] - closes.iloc[-MOMENTUM_LOOKBACK]) / closes.iloc[-MOMENTUM_LOOKBACK]
    else:
        momentum = 0.0
    
    # 检查指标有效性
    if np.isnan(sma_trend) or np.isnan(realized_vol) or np.isnan(momentum):
        # 指标无效时，如果有仓位且未达到最小持仓，保持仓位
        curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
        if MIN_HOLD_BARS > 0 and curr_side != 0:
            if curr_side == CM_LAST_SIDE:
                CM_HOLD_BARS += 1
            else:
                CM_HOLD_BARS = 1
                CM_LAST_SIDE = curr_side
            if CM_HOLD_BARS < MIN_HOLD_BARS:
                return float(curr_side)
        return 0.0
    
    # ========== 3. 波动率管理（稳健核心）==========
    # 高波动时降仓位或暂停交易
    if VOL_CONTROL_ENABLED and realized_vol > VOL_THRESHOLD_HIGH:
        # 高波动时，如果有仓位且未达到最小持仓，保持仓位；否则平仓
        curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
        if MIN_HOLD_BARS > 0 and curr_side != 0:
            if curr_side == CM_LAST_SIDE:
                CM_HOLD_BARS += 1
            else:
                CM_HOLD_BARS = 1
                CM_LAST_SIDE = curr_side
            if CM_HOLD_BARS < MIN_HOLD_BARS:
                return float(curr_side)
        return 0.0  # 高波动时平仓
    
    # 低波动时减少交易频率（但不禁止）
    low_vol_penalty = 1.0
    if VOL_CONTROL_ENABLED and realized_vol < VOL_THRESHOLD_LOW:
        low_vol_penalty = 0.5  # 降低信号强度
    
    # 波动率仓位缩放因子（波动率越低，仓位越大）
    vol_scaling = 1.0
    if VOL_SCALING_ENABLED:
        # 使用反比例缩放：vol 越低，scaling 越大（但不超过 1.5）
        if realized_vol > 0:
            vol_scaling = min(1.5, VOL_THRESHOLD_LOW / max(realized_vol, 0.001))
        else:
            vol_scaling = 1.0
    
    # ========== 4. 趋势过滤（方向闸门）==========
    trend_direction = 0  # 0: 无限制, 1: 只做多, -1: 只做空
    
    if TREND_FILTER_ENABLED:
        price_to_sma = (current_price - sma_trend) / sma_trend if sma_trend > 0 else 0
        if price_to_sma > TREND_THRESHOLD:
            trend_direction = 1  # 趋势向上，只允许做多
        elif price_to_sma < -TREND_THRESHOLD:
            trend_direction = -1  # 趋势向下，只允许做空
    
    # ========== 5. Carry 因子（收益来源）==========
    carry_score = 0.0  # -1 到 1 的得分
    carry_signal = 0.0
    
    # Funding rate 得分
    if abs(current_funding_rate) > FUNDING_THRESHOLD:
        funding_score = np.clip(current_funding_rate / 0.001, -1.0, 1.0)  # 归一化到 [-1, 1]
    else:
        funding_score = 0.0
    
    # Basis 得分
    if abs(current_basis) > BASIS_THRESHOLD:
        basis_score = np.clip(current_basis / 0.001, -1.0, 1.0)  # 归一化到 [-1, 1]
    else:
        basis_score = 0.0
    
    # 综合 carry 得分（取平均值）
    if funding_score != 0.0 or basis_score != 0.0:
        if funding_score != 0.0 and basis_score != 0.0:
            carry_score = (funding_score + basis_score) / 2.0
        elif funding_score != 0.0:
            carry_score = funding_score
        else:
            carry_score = basis_score
    
    # Carry 信号：carry_score > 0 表示做空 perp，< 0 表示做多 perp
    if abs(carry_score) >= MIN_CARRY_SCORE:
        carry_signal = -carry_score  # 注意：carry 为正 → 做空 perp
    
    # ========== 6. 动量因子（质量确认）==========
    momentum_signal = 0.0
    
    if abs(momentum) > MOMENTUM_THRESHOLD:
        # 动量方向
        if momentum > 0:
            momentum_signal = 1.0
        else:
            momentum_signal = -1.0
    
    # ========== 7. 多因子综合 ==========
    raw_side = 0.0
    
    # 基础信号：carry 信号
    if carry_signal != 0.0:
        raw_side = carry_signal
        
        # 动量确认：如果动量与 carry 方向一致，加分；如果相反，降权
        if momentum_signal != 0.0:
            if np.sign(carry_signal) == np.sign(momentum_signal):
                # 方向一致，增强信号
                raw_side = carry_signal * (1.0 + MOMENTUM_WEIGHT)
            else:
                # 方向相反，降权或禁止
                raw_side = carry_signal * (1.0 - MOMENTUM_WEIGHT)
                # 如果降权后信号太弱，禁止交易
                if abs(raw_side) < MIN_CARRY_SCORE:
                    raw_side = 0.0
    
    # 应用波动率惩罚
    raw_side = raw_side * low_vol_penalty
    
    # 归一化到 [-1, 1]
    raw_side = np.clip(raw_side, -1.0, 1.0)
    
    # ========== 8. 趋势方向过滤 ==========
    if trend_direction == 1 and raw_side < 0:
        # 趋势向上，不允许做空
        raw_side = 0.0
    elif trend_direction == -1 and raw_side > 0:
        # 趋势向下，不允许做多
        raw_side = 0.0
    
    # ========== 9. 最小持仓周期逻辑 ==========
    curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
    
    if curr_side == 0:
        CM_HOLD_BARS = 0
        CM_LAST_SIDE = 0
    else:
        if curr_side == CM_LAST_SIDE:
            CM_HOLD_BARS += 1
        else:
            CM_HOLD_BARS = 1
            CM_LAST_SIDE = curr_side
    
    # 检查最小持仓周期
    if MIN_HOLD_BARS > 0 and curr_side != 0:
        # 如果新信号与当前持仓方向不同（包括平仓），且未达到最小持仓周期
        if (raw_side == 0.0 or np.sign(raw_side) != curr_side) and CM_HOLD_BARS < MIN_HOLD_BARS:
            return float(curr_side)  # 保持当前仓位
    
    return raw_side


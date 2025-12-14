"""Funding Rate Arbitrage Mixed Strategy.

This strategy combines funding rate arbitrage with trend filtering, basis trading,
volatility control, and risk management for professional-grade execution.
"""

import numpy as np
import pandas as pd
from .base import PRICE_COL, VOL_COL

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """计算指数移动平均线（EMA）"""
    return series.ewm(span=period, adjust=False).mean()


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
FA_LAST_SIDE = 0
FA_HOLD_BARS = 0


def funding_arb_baseline(
    df: pd.DataFrame,
    current_pos: float,
    current_equity: float,
    MIN_HOLD_BARS: int = 0,
    # Funding Rate 参数
    FUNDING_RATE: float = 0.0,  # 当前 funding rate (需要从外部传入，通常是 8h rate)
    FUNDING_THRESHOLD: float = 0.0001,  # Funding rate 阈值 (0.01% = 0.0001)
    # 基差参数
    BASIS: float = 0.0,  # 当前基差 (perp - spot) / spot (需要从外部传入)
    BASIS_THRESHOLD: float = 0.0005,  # 基差阈值 (0.05% = 0.0005)
    # 趋势过滤参数
    TREND_MA_PERIOD: int = 200,  # 趋势判断 SMA 周期
    TREND_FILTER_ENABLED: bool = True,  # 是否启用趋势过滤
    # 波动率控制参数
    VOL_PERIOD: int = 20,  # 波动率计算周期
    VOL_THRESHOLD: float = 0.05,  # 波动率阈值 (5% = 0.05)
    VOL_CONTROL_ENABLED: bool = True,  # 是否启用波动率控制
    # 风险控制参数
    MAX_LEVERAGE: float = 3.0,  # 最大杠杆（Funding 套利通常用较低杠杆）
    STOP_ON_FUNDING_FLIP: bool = True,  # Funding 翻转时是否平仓
    # 信号优先级
    REQUIRE_BOTH_FUNDING_AND_BASIS: bool = False,  # 是否要求同时满足 funding 和 basis（默认 False，因为数据可能缺失）
) -> float:
    """
    Funding Rate 套利混合策略
    
    策略组合：
    1. Funding Rate 信号：funding > 0 做空 perp，funding < 0 做多 perp
    2. 基差过滤：basis > threshold 时做空 perp，basis < -threshold 时做多 perp
    3. 趋势过滤：只在顺趋势方向进行套利
       - funding > 0 且 price < MA200 → 做空 perp（顺趋势）
       - funding < 0 且 price > MA200 → 做多 perp（顺趋势）
    4. 波动率控制：高波动时减少或禁止开仓
    5. 风险控制：最小持仓周期、Funding 翻转止损
    
    参数说明：
    - FUNDING_RATE: 当前 funding rate（8h rate，需要从外部传入）
    - FUNDING_THRESHOLD: Funding rate 阈值，低于此值不交易
    - BASIS: 当前基差 (perp - spot) / spot（需要从外部传入）
    - BASIS_THRESHOLD: 基差阈值
    - TREND_MA_PERIOD: 趋势判断 SMA 周期（默认 200）
    - VOL_THRESHOLD: 波动率阈值，高于此值不交易
    - MIN_HOLD_BARS: 最小持仓 bar 数，防止频繁调仓
    - REQUIRE_BOTH_FUNDING_AND_BASIS: 是否要求同时满足 funding 和 basis 条件
    """
    global FA_LAST_SIDE, FA_HOLD_BARS
    
    # 基础检查
    if df is None or df.empty or current_equity <= 0:
        return 0.0
    
    if PRICE_COL not in df.columns:
        return 0.0
    
    # 检查数据长度
    max_period = max(TREND_MA_PERIOD, VOL_PERIOD)
    if len(df) < max_period + 10:
        return 0.0
    
    # 提取数据
    closes = df[PRICE_COL].astype(float)
    highs = df["h"].astype(float) if "h" in df.columns else closes
    lows = df["l"].astype(float) if "l" in df.columns else closes
    volumes = df[VOL_COL].astype(float) if VOL_COL in df.columns else pd.Series([1.0] * len(df))
    
    current_price = closes.iloc[-1]
    
    # 尝试从 DataFrame 读取 funding_rate 和 basis（如果存在）
    # 如果 DataFrame 中有这些列，优先使用；否则使用传入的参数
    if "funding_rate" in df.columns:
        current_funding_rate = float(df["funding_rate"].iloc[-1])
    else:
        current_funding_rate = FUNDING_RATE
    
    if "basis" in df.columns:
        current_basis = float(df["basis"].iloc[-1])
    else:
        current_basis = BASIS
    
    # 如果 funding_rate 和 basis 都是 0（数据缺失），使用价格动量作为替代信号
    # 这是一个简化的模拟：如果价格快速上涨，可能意味着 funding 为正（perp 溢价）
    if current_funding_rate == 0.0 and current_basis == 0.0:
        # 使用短期价格变化作为 funding rate 的代理
        if len(closes) >= 20:
            price_change_20 = (closes.iloc[-1] - closes.iloc[-20]) / closes.iloc[-20]
            # 将价格变化转换为 funding rate 的近似值（需要调整系数）
            # 这里使用一个简化的映射：价格快速上涨 → 正 funding（做空信号）
            current_funding_rate = price_change_20 * 0.1  # 调整系数，使信号更敏感
    
    # ========== 1. 计算技术指标 ==========
    # 趋势指标
    sma_trend = calculate_sma(closes, TREND_MA_PERIOD).iloc[-1]
    
    # 波动率指标（使用 ATR 归一化）
    atr = calculate_atr(highs, lows, closes, VOL_PERIOD).iloc[-1]
    realized_vol = atr / current_price if current_price > 0 else 0.0
    
    # 检查指标有效性
    if np.isnan(sma_trend) or np.isnan(realized_vol):
        # 指标无效时，如果有仓位且未达到最小持仓，保持仓位
        curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
        if MIN_HOLD_BARS > 0 and curr_side != 0:
            if curr_side == FA_LAST_SIDE:
                FA_HOLD_BARS += 1
            else:
                FA_HOLD_BARS = 1
                FA_LAST_SIDE = curr_side
            if FA_HOLD_BARS < MIN_HOLD_BARS:
                return float(curr_side)
        return 0.0
    
    # ========== 2. 波动率控制 ==========
    if VOL_CONTROL_ENABLED and realized_vol > VOL_THRESHOLD:
        # 高波动时，如果有仓位且未达到最小持仓，保持仓位；否则平仓
        curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
        if MIN_HOLD_BARS > 0 and curr_side != 0:
            if curr_side == FA_LAST_SIDE:
                FA_HOLD_BARS += 1
            else:
                FA_HOLD_BARS = 1
                FA_LAST_SIDE = curr_side
            if FA_HOLD_BARS < MIN_HOLD_BARS:
                return float(curr_side)
        return 0.0  # 高波动时平仓
    
    # ========== 3. Funding Rate 信号 ==========
    funding_signal = 0.0
    
    if abs(current_funding_rate) > FUNDING_THRESHOLD:
        if current_funding_rate > 0:
            # Funding 为正 → 做空 perp（因为 perp 溢价，预期会收敛）
            funding_signal = -1.0
        elif current_funding_rate < 0:
            # Funding 为负 → 做多 perp（因为 perp 折价，预期会收敛）
            funding_signal = 1.0
    
    # ========== 4. 基差信号 ==========
    basis_signal = 0.0
    
    if abs(current_basis) > BASIS_THRESHOLD:
        if current_basis > 0:
            # 基差为正 → perp 溢价 → 做空 perp
            basis_signal = -1.0
        elif current_basis < 0:
            # 基差为负 → perp 折价 → 做多 perp
            basis_signal = 1.0
    
    # ========== 5. 趋势过滤 ==========
    trend_direction = 0  # 0: 无限制, 1: 只做多, -1: 只做空
    
    if TREND_FILTER_ENABLED:
        if current_price > sma_trend * 1.001:
            trend_direction = 1  # 趋势向上，只允许做多
        elif current_price < sma_trend * 0.999:
            trend_direction = -1  # 趋势向下，只允许做空
    
    # ========== 6. 信号综合 ==========
    raw_side = 0.0
    
    # 检查是否同时满足 funding 和 basis 条件
    if REQUIRE_BOTH_FUNDING_AND_BASIS:
        # 要求 funding 和 basis 信号方向一致
        if funding_signal != 0 and basis_signal != 0:
            if funding_signal == basis_signal:
                raw_side = funding_signal
            else:
                # 信号冲突，不交易
                raw_side = 0.0
        else:
            # 缺少其中一个信号，不交易
            # 但如果两个都是 0（数据缺失），尝试使用 funding 信号（如果存在）
            if funding_signal != 0:
                raw_side = funding_signal
            elif basis_signal != 0:
                raw_side = basis_signal
            else:
                raw_side = 0.0
    else:
        # 只需要满足其中一个条件
        if funding_signal != 0:
            raw_side = funding_signal
        elif basis_signal != 0:
            raw_side = basis_signal
    
    # ========== 7. 趋势方向过滤 ==========
    if trend_direction == 1 and raw_side < 0:
        # 趋势向上，不允许做空（funding 套利通常做空 perp）
        # 但如果是 funding 套利，可以考虑：如果 funding > 0 且趋势向上，可能不适合做空
        # 这里选择平仓或保持空仓
        raw_side = 0.0
    elif trend_direction == -1 and raw_side > 0:
        # 趋势向下，不允许做多
        raw_side = 0.0
    
    # ========== 8. Funding 翻转检测（风险控制）==========
    if STOP_ON_FUNDING_FLIP and current_pos != 0:
        # 如果当前有仓位，检查 funding 是否翻转
        curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
        
        # 如果 funding 信号与当前持仓方向相反，考虑平仓
        if curr_side > 0 and funding_signal < 0:
            # 持有多头，但 funding 信号转为做空 → 可能翻转
            raw_side = 0.0  # 平仓
        elif curr_side < 0 and funding_signal > 0:
            # 持有空头，但 funding 信号转为做多 → 可能翻转
            raw_side = 0.0  # 平仓
    
    # ========== 9. 最小持仓周期逻辑 ==========
    curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
    
    if curr_side == 0:
        FA_HOLD_BARS = 0
        FA_LAST_SIDE = 0
    else:
        if curr_side == FA_LAST_SIDE:
            FA_HOLD_BARS += 1
        else:
            FA_HOLD_BARS = 1
            FA_LAST_SIDE = curr_side
    
    # 检查最小持仓周期
    if MIN_HOLD_BARS > 0 and curr_side != 0:
        # 如果新信号与当前持仓方向不同（包括平仓），且未达到最小持仓周期
        if (raw_side == 0.0 or np.sign(raw_side) != curr_side) and FA_HOLD_BARS < MIN_HOLD_BARS:
            return float(curr_side)  # 保持当前仓位
    
    # ========== 10. 杠杆限制 ==========
    # 注意：这里返回的是仓位比例，实际杠杆控制应该在回测引擎中处理
    # 但我们可以根据波动率调整仓位大小
    if raw_side != 0 and VOL_CONTROL_ENABLED:
        # 根据波动率调整仓位（高波动时减少仓位）
        vol_adjustment = min(1.0, VOL_THRESHOLD / max(realized_vol, 1e-6))
        # 这里我们只返回方向，实际仓位调整应该在回测引擎中处理
        # 但可以通过返回较小的绝对值来表示减少仓位
        # 为了简化，这里仍然返回 -1.0 或 1.0，实际仓位调整在回测引擎中处理
    
    return raw_side


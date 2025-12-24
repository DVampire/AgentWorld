import numpy as np
import pandas as pd
from .base import PRICE_COL, VOL_COL


# ==== 最小持仓计数器 ====
LT_LAST_SIDE = 0
LT_HOLD_BARS = 0
LT_LAST_EMA20 = None
LT_LAST_EMA50 = None

# ============================================================
#         技术指标辅助函数（用于 livetrading_baseline）
# ============================================================

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """计算指数移动平均线（EMA）"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """计算简单移动平均线（SMA）"""
    return series.rolling(window=period).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """计算相对强弱指标（RSI）"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """计算平均真实波幅（ATR）"""
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=period).mean()
    return atr


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """计算平均趋向指标（ADX）- 简化版本"""
    # 计算 +DI 和 -DI
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - close.shift())
    tr3 = np.abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # 平滑处理
    atr = tr.rolling(window=period).mean()
    # 避免除零
    atr_safe = atr.replace(0, np.nan)
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_safe)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_safe)
    
    # DX 和 ADX（避免除零）
    di_sum = plus_di + minus_di
    di_sum = di_sum.replace(0, np.nan)
    dx = 100 * np.abs(plus_di - minus_di) / di_sum
    adx = dx.rolling(window=period).mean()
    
    return adx


def find_swing_points(high: pd.Series, low: pd.Series, lookback: int = 20) -> tuple:
    """找到最近的结构高低点（swing points）"""
    if len(high) < lookback * 2:
        return None, None
    
    recent_high = high.iloc[-lookback:].max()
    recent_low = low.iloc[-lookback:].min()
    
    return recent_high, recent_low


def check_volume_surge(volumes: pd.Series, current_vol: float, lookback: int = 20, multiplier: float = 1.5) -> bool:
    """检查成交量是否放大"""
    if len(volumes) < lookback:
        return False
    
    avg_vol = volumes.iloc[-lookback:].mean()
    if avg_vol <= 0:
        return False
    
    return current_vol >= avg_vol * multiplier


# ============================================================
#         Live Trading Baseline（组合策略集）
# ============================================================

def livetrading_baseline(
    df: pd.DataFrame,
    current_pos: float,
    current_equity: float,
    MIN_HOLD_BARS: int = 0,
    # 趋势主干参数
    SMA_TREND_PERIOD: int = 200,  # 主趋势 SMA
    EMA_FAST: int = 20,            # 快线 EMA
    EMA_SLOW: int = 50,            # 慢线 EMA
    # ADX 过滤参数
    ADX_PERIOD: int = 14,
    ADX_THRESHOLD: float = 25.0,
    # RSI 回调参数
    RSI_PERIOD: int = 14,
    RSI_OVERSOLD: float = 40.0,    # 多头回调入场
    RSI_OVERBOUGHT: float = 60.0,  # 空头回调入场
    # 动量突破参数
    SWING_LOOKBACK: int = 20,      # 结构高低点回看
    VOL_SURGE_LOOKBACK: int = 20,  # 成交量回看
    VOL_SURGE_MULTIPLIER: float = 1.5,  # 成交量放大倍数
    # ATR 止损参数（用于风控，这里只做信号，实际止损在回测引擎中处理）
    ATR_PERIOD: int = 14,
) -> float:
    """
    Live Trading Baseline：组合策略集
    
    策略组合：
    1. 趋势主干：200 SMA 判断大方向 + EMA 20/50 交叉触发
    2. 动量爆发：放量突破结构高低点
    3. 回调入场：EMA 20 回调 + RSI 过滤
    4. 市况过滤：ADX > 25 才允许趋势/动量策略运行
    5. 风控：ATR × 2 止损（信号层面，实际止损在回测引擎）
    
    参数说明：
    - MIN_HOLD_BARS: 最小持仓 bar 数，防止频繁调仓
    - SMA_TREND_PERIOD: 主趋势 SMA 周期（默认 200）
    - EMA_FAST/SLOW: 趋势驱动 EMA 周期（默认 20/50）
    - ADX_THRESHOLD: ADX 阈值，低于此值停用趋势策略（默认 25）
    - RSI_OVERSOLD/OVERBOUGHT: RSI 回调入场阈值
    - VOL_SURGE_MULTIPLIER: 成交量放大倍数阈值
    """
    global LT_LAST_SIDE, LT_HOLD_BARS, LT_LAST_EMA20, LT_LAST_EMA50
    
    # 基础检查
    if df is None or df.empty or current_equity <= 0:
        return 0.0
    
    if PRICE_COL not in df.columns or VOL_COL not in df.columns:
        return 0.0
    
    # 检查数据长度
    max_period = max(SMA_TREND_PERIOD, EMA_SLOW, ADX_PERIOD, RSI_PERIOD, ATR_PERIOD, SWING_LOOKBACK)
    if len(df) < max_period + 10:  # 额外缓冲
        return 0.0
    
    # 提取数据
    closes = df[PRICE_COL].astype(float)
    highs = df["h"].astype(float) if "h" in df.columns else closes
    lows = df["l"].astype(float) if "l" in df.columns else closes
    volumes = df[VOL_COL].astype(float)
    
    current_price = closes.iloc[-1]
    current_vol = volumes.iloc[-1]
    
    # ========== 1. 计算技术指标 ==========
    sma_trend = calculate_sma(closes, SMA_TREND_PERIOD).iloc[-1]
    ema_fast = calculate_ema(closes, EMA_FAST).iloc[-1]
    ema_slow = calculate_ema(closes, EMA_SLOW).iloc[-1]
    
    rsi = calculate_rsi(closes, RSI_PERIOD).iloc[-1]
    adx = calculate_adx(highs, lows, closes, ADX_PERIOD).iloc[-1]
    atr = calculate_atr(highs, lows, closes, ATR_PERIOD).iloc[-1]
    
    # 检查指标有效性
    if (np.isnan(sma_trend) or np.isnan(ema_fast) or np.isnan(ema_slow) or 
        np.isnan(rsi) or np.isnan(adx) or np.isnan(atr)):
        # 指标无效时，如果有仓位且未达到最小持仓，保持仓位
        curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
        if MIN_HOLD_BARS > 0 and curr_side != 0:
            if curr_side == LT_LAST_SIDE:
                LT_HOLD_BARS += 1
            else:
                LT_HOLD_BARS = 1
                LT_LAST_SIDE = curr_side
            if LT_HOLD_BARS < MIN_HOLD_BARS:
                return float(curr_side)
        return 0.0
    
    # ========== 2. ADX 市况过滤 ==========
    # ADX <= 25：震荡市，停用趋势/动量策略
    # 注意：这里不直接返回，而是设置一个标志，让后续逻辑处理
    adx_filter_active = adx <= ADX_THRESHOLD
    
    # ========== 3. 趋势主干：200 SMA 判断大方向 ==========
    trend_direction = 0  # 0: 无趋势限制, 1: 只做多, -1: 只做空
    
    if current_price > sma_trend * 1.001:  # 略高于 SMA，避免噪音
        trend_direction = 1  # 只允许做多
    elif current_price < sma_trend * 0.999:  # 略低于 SMA
        trend_direction = -1  # 只允许做空
    
    # ========== 4. EMA 交叉信号 ==========
    ema_cross_signal = 0
    
    # 检测交叉（需要前一根的数据）
    if LT_LAST_EMA20 is not None and LT_LAST_EMA50 is not None:
        prev_fast_above_slow = LT_LAST_EMA20 > LT_LAST_EMA50
        curr_fast_above_slow = ema_fast > ema_slow
        
        if not prev_fast_above_slow and curr_fast_above_slow:
            # 上穿：做多信号
            ema_cross_signal = 1.0
        elif prev_fast_above_slow and not curr_fast_above_slow:
            # 下穿：做空信号
            ema_cross_signal = -1.0
    else:
        # 首次运行，根据当前 EMA 位置给出初始信号
        if ema_fast > ema_slow:
            ema_cross_signal = 1.0
        elif ema_fast < ema_slow:
            ema_cross_signal = -1.0
    
    # 更新 EMA 历史值
    LT_LAST_EMA20 = ema_fast
    LT_LAST_EMA50 = ema_slow
    
    # ========== 5. 动量爆发：放量突破 ==========
    momentum_signal = 0
    
    # ADX 过滤：震荡市停用动量策略
    if not adx_filter_active:
        swing_high, swing_low = find_swing_points(highs, lows, SWING_LOOKBACK)
        if swing_high is not None and swing_low is not None:
            volume_surge = check_volume_surge(volumes, current_vol, VOL_SURGE_LOOKBACK, VOL_SURGE_MULTIPLIER)
            
            # 向上突破
            if current_price > swing_high * 1.0005 and volume_surge:  # 0.05% 突破阈值
                momentum_signal = 1.0
            # 向下突破
            elif current_price < swing_low * 0.9995 and volume_surge:
                momentum_signal = -1.0
    
    # ========== 6. 回调入场：EMA 20 回调 + RSI ==========
    pullback_signal = 0
    
    # ADX 过滤：震荡市停用回调策略
    if not adx_filter_active:
        # 计算价格相对 EMA20 的位置
        price_to_ema20 = (current_price - ema_fast) / ema_fast if ema_fast > 0 else 0
        
        # 多头回调：价格接近或略低于 EMA20，且 RSI < 40，且趋势向上
        if trend_direction >= 0:  # 允许做多或中性
            if -0.005 <= price_to_ema20 <= 0.01 and rsi < RSI_OVERSOLD:  # 价格在 EMA20 附近或略低
                pullback_signal = 1.0
        
        # 空头回调：价格接近或略高于 EMA20，且 RSI > 60，且趋势向下
        if trend_direction <= 0:  # 允许做空或中性
            if -0.01 <= price_to_ema20 <= 0.005 and rsi > RSI_OVERBOUGHT:
                pullback_signal = -1.0
    
    # ========== 7. 信号综合 ==========
    raw_side = 0.0
    
    # ADX 过滤：震荡市时，EMA 交叉信号也停用
    if not adx_filter_active:
        # 优先级：EMA 交叉 > 动量突破 > 回调入场
        if ema_cross_signal != 0:
            raw_side = ema_cross_signal
        elif momentum_signal != 0:
            raw_side = momentum_signal
        elif pullback_signal != 0:
            raw_side = pullback_signal
    
    # ========== 8. 趋势方向过滤 ==========
    if trend_direction == 1 and raw_side < 0:
        # 趋势向上，不允许做空
        raw_side = 0.0
    elif trend_direction == -1 and raw_side > 0:
        # 趋势向下，不允许做多
        raw_side = 0.0
    
    # ========== 9. 最小持仓周期逻辑（必须在所有信号计算之后）==========
    # 先更新持仓计数器
    curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
    
    if curr_side == 0:
        LT_HOLD_BARS = 0
        LT_LAST_SIDE = 0
    else:
        if curr_side == LT_LAST_SIDE:
            LT_HOLD_BARS += 1
        else:
            LT_HOLD_BARS = 1
            LT_LAST_SIDE = curr_side
    
    # 然后检查最小持仓周期：这个检查必须在所有信号计算之后，包括 ADX 过滤、趋势过滤等导致的平仓信号
    if MIN_HOLD_BARS > 0 and curr_side != 0:
        # 如果新信号与当前持仓方向不同（包括平仓 raw_side == 0），且未达到最小持仓周期
        # 注意：raw_side == 0.0 表示平仓信号，np.sign(raw_side) != curr_side 表示反向信号
        if (raw_side == 0.0 or np.sign(raw_side) != curr_side) and LT_HOLD_BARS < MIN_HOLD_BARS:
            return float(curr_side)  # 保持当前仓位，忽略平仓/反向信号
    
    # ADX 过滤：如果震荡市，返回空仓信号（但需要先检查最小持仓周期）
    # 注意：如果当前有仓位且未达到最小持仓周期，已经在上面保持仓位了
    if adx_filter_active and curr_side == 0:
        return 0.0
    
    return raw_side
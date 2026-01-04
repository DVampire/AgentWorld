import numpy as np
import pandas as pd

# 使用 data_store 导出的 1m K 线，默认列名：
# - "t": 时间戳
# - "c": 收盘价
# - "v": 成交量
PRICE_COL = "c"
VOL_COL = "v"
MAX_LEVERAGE = 5.0

LAST_BAND = 0   # 可以给其它策略用
_SIGNAL_HISTORY = []

# ==== 最小持仓计数器（每个 baseline 一套） ====
MA_LAST_SIDE = 0
MA_HOLD_BARS = 0

Z_LAST_SIDE = 0
Z_HOLD_BARS = 0

TS_LAST_SIDE = 0
TS_HOLD_BARS = 0

LT_LAST_SIDE = 0
LT_HOLD_BARS = 0
LT_LAST_EMA20 = None
LT_LAST_EMA50 = None

FA_LAST_SIDE = 0
FA_HOLD_BARS = 0

CM_LAST_SIDE = 0
CM_HOLD_BARS = 0


def reset_signal_history() -> None:
    """清空信号历史和所有持仓计数器（在每次回测前调用）。"""
    global MA_LAST_SIDE, MA_HOLD_BARS, Z_LAST_SIDE, Z_HOLD_BARS, TS_LAST_SIDE, TS_HOLD_BARS
    global LT_LAST_SIDE, LT_HOLD_BARS, LT_LAST_EMA20, LT_LAST_EMA50
    global FA_LAST_SIDE, FA_HOLD_BARS, CM_LAST_SIDE, CM_HOLD_BARS

    _SIGNAL_HISTORY.clear()
    
    MA_LAST_SIDE = 0
    MA_HOLD_BARS = 0
    Z_LAST_SIDE = 0
    Z_HOLD_BARS = 0
    TS_LAST_SIDE = 0
    TS_HOLD_BARS = 0
    LT_LAST_SIDE = 0
    LT_HOLD_BARS = 0
    LT_LAST_EMA20 = None
    LT_LAST_EMA50 = None
    FA_LAST_SIDE = 0
    FA_HOLD_BARS = 0
    CM_LAST_SIDE = 0
    CM_HOLD_BARS = 0


def get_signal_df() -> pd.DataFrame:
    """将当前缓存的信号历史导出为 DataFrame。"""
    if not _SIGNAL_HISTORY:
        return pd.DataFrame()
    df = pd.DataFrame(_SIGNAL_HISTORY)
    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
    return df


def band_from_signal(x: float) -> int:
    if x >= 0.4:
        return 1
    if x <= -0.4:
        return -1
    return 0


# ============================================================
#             几个常见的基线（Baseline）策略
# ============================================================

def bh_baseline(
    df: pd.DataFrame,
    current_pos: float,
    current_equity: float,
    MIN_HOLD_BARS: int = 0,   # 为了接口统一，这里加上但实际上 BH 始终满仓
) -> float:
    """
    基线 1：Buy & Hold（永远做多）

    - 有数据且权益>0 就一直返回 +1.0（满仓多）
    - 用作最简单的方向性 Beta benchmark
    """
    if df is None or df.empty or current_equity <= 0:
        return 0.0
    return 1.0


def ma_crossover_baseline(
    df: pd.DataFrame,
    current_pos: float,
    current_equity: float,
    MIN_HOLD_BARS: int = 0,
    FAST_WINDOW: int = 20,
    SLOW_WINDOW: int = 60,
) -> float:
    """
    基线 2：MA 交叉趋势策略（1m）

    - 快线 > 慢线 → +1.0（多）
    - 快线 < 慢线 → -1.0（空）
    - 其余 → 0.0（空仓）

    MIN_HOLD_BARS:
      - 当前有仓位时，如果尚未持有足够 bar 数，则不允许平仓 / 反向，
        直接维持 current_pos。
    """
    global MA_LAST_SIDE, MA_HOLD_BARS

    if df is None or df.empty or current_equity <= 0 or PRICE_COL not in df.columns:
        return 0.0

    if len(df) < max(FAST_WINDOW, SLOW_WINDOW):
        return 0.0

    closes = df[PRICE_COL].astype(float)
    fast_ma = closes.rolling(FAST_WINDOW).mean().iloc[-1]
    slow_ma = closes.rolling(SLOW_WINDOW).mean().iloc[-1]

    if np.isnan(fast_ma) or np.isnan(slow_ma):
        return 0.0

    # 原始目标方向（不考虑最小持仓）
    if fast_ma > slow_ma * 1.0001:
        raw_side = 1.0
    elif fast_ma < slow_ma * 0.9999:
        raw_side = -1.0
    else:
        raw_side = 0.0

    # === 最小持仓周期逻辑 ===
    curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)

    if curr_side == 0:
        # 空仓 → 重置计数
        MA_HOLD_BARS = 0
    else:
        # 有仓位
        if curr_side == MA_LAST_SIDE:
            MA_HOLD_BARS += 1
        else:
            MA_HOLD_BARS = 1
            MA_LAST_SIDE = curr_side

    if MIN_HOLD_BARS > 0 and curr_side != 0:
        # 有仓 & 未达到最小持仓 → 即便信号想平仓/反向，也先 hold
        if raw_side != curr_side and MA_HOLD_BARS < MIN_HOLD_BARS:
            return float(curr_side)

    return raw_side


def zscore_mr_baseline(
    df: pd.DataFrame,
    current_pos: float,
    current_equity: float,
    MIN_HOLD_BARS: int = 0,
    WINDOW: int = 40,
    ENTRY_Z: float = 1.5,
    EXIT_Z: float = 0.3,
) -> float:
    """
    基线 3：Z-score 均值回归（1m）

    - z = (P - MA) / std
    - z >  ENTRY_Z  → -1.0（做空）
    - z < -ENTRY_Z  → +1.0（做多）
    - |z| < EXIT_Z  → 0.0（平仓）

    MIN_HOLD_BARS:
      - 持仓期间未达到 bar 数，信号即使满足平仓或反手也先 hold。
    """
    global Z_LAST_SIDE, Z_HOLD_BARS

    if df is None or df.empty or current_equity <= 0 or PRICE_COL not in df.columns:
        return 0.0

    if len(df) < WINDOW:
        return 0.0

    closes = df[PRICE_COL].astype(float)
    ma = closes.rolling(WINDOW).mean()
    std = closes.rolling(WINDOW).std(ddof=0)

    mu = ma.iloc[-1]
    sigma = std.iloc[-1]
    p = closes.iloc[-1]

    if np.isnan(mu) or np.isnan(sigma) or sigma < 1e-12:
        return 0.0

    z = (p - mu) / sigma

    # 原始方向
    if abs(z) < EXIT_Z:
        raw_side = 0.0
    elif z > ENTRY_Z:
        raw_side = -1.0
    elif z < -ENTRY_Z:
        raw_side = 1.0
    else:
        # 中间区域：保持现有仓位
        raw_side = float(current_pos)

    # === 最小持仓周期逻辑 ===
    curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)

    if curr_side == 0:
        Z_HOLD_BARS = 0
    else:
        if curr_side == Z_LAST_SIDE:
            Z_HOLD_BARS += 1
        else:
            Z_HOLD_BARS = 1
            Z_LAST_SIDE = curr_side

    if MIN_HOLD_BARS > 0 and curr_side != 0:
        # 有仓 & 未达到最小持仓 → 不允许从有仓 → 空仓 或翻向
        if (raw_side == 0.0 or np.sign(raw_side) != curr_side) and Z_HOLD_BARS < MIN_HOLD_BARS:
            return float(curr_side)

    return raw_side


def tsmom_baseline(
    df: pd.DataFrame,
    current_pos: float,
    current_equity: float,
    MIN_HOLD_BARS: int = 0,
    LOOKBACK: int = 50,
    THRESHOLD: float = 0.0,
) -> float:
    """
    基线 4：时间序列动量（TSMOM，1m）

    - ret = P_t / P_{t-k} - 1
    - ret >  THRESHOLD → +1.0（多）
    - ret < -THRESHOLD → -1.0（空）
    - |ret| <= THRESHOLD → 0.0（空仓）

    MIN_HOLD_BARS:
      - 同样在有仓时要求至少持有这么多个 bar 才能平仓/反向。
    """
    global TS_LAST_SIDE, TS_HOLD_BARS

    if df is None or df.empty or current_equity <= 0 or PRICE_COL not in df.columns:
        return 0.0

    if len(df) <= LOOKBACK:
        return 0.0

    closes = df[PRICE_COL].astype(float)
    p_now = closes.iloc[-1]
    p_past = closes.iloc[-(LOOKBACK + 1)]

    if np.isnan(p_now) or np.isnan(p_past) or p_past <= 0:
        return 0.0

    ret = p_now / p_past - 1.0

    # 原始方向
    if ret > THRESHOLD:
        raw_side = 1.0
    elif ret < -THRESHOLD:
        raw_side = -1.0
    else:
        raw_side = 0.0

    # === 最小持仓逻辑 ===
    curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)

    if curr_side == 0:
        TS_HOLD_BARS = 0
    else:
        if curr_side == TS_LAST_SIDE:
            TS_HOLD_BARS += 1
        else:
            TS_HOLD_BARS = 1
            TS_LAST_SIDE = curr_side

    if MIN_HOLD_BARS > 0 and curr_side != 0:
        if raw_side != curr_side and TS_HOLD_BARS < MIN_HOLD_BARS:
            return float(curr_side)

    return raw_side


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


# ============================================================
#         Funding Rate Arbitrage Baseline
# ============================================================

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
    4. 波动率控制：高波动时减少或禁止开仓
    5. 风险控制：最小持仓周期、Funding 翻转止损
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
    if "funding_rate" in df.columns:
        current_funding_rate = float(df["funding_rate"].iloc[-1])
    else:
        current_funding_rate = FUNDING_RATE
    
    if "basis" in df.columns:
        current_basis = float(df["basis"].iloc[-1])
    else:
        current_basis = BASIS
    
    # 如果 funding_rate 和 basis 都是 0（数据缺失），使用价格动量作为替代信号
    if current_funding_rate == 0.0 and current_basis == 0.0:
        if len(closes) >= 20:
            price_change_20 = (closes.iloc[-1] - closes.iloc[-20]) / closes.iloc[-20]
            current_funding_rate = price_change_20 * 0.1
    
    # ========== 1. 计算技术指标 ==========
    sma_trend = calculate_sma(closes, TREND_MA_PERIOD).iloc[-1]
    atr = calculate_atr(highs, lows, closes, VOL_PERIOD).iloc[-1]
    realized_vol = atr / current_price if current_price > 0 else 0.0
    
    # 检查指标有效性
    if np.isnan(sma_trend) or np.isnan(realized_vol):
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
    
    # ========== 3. Funding Rate 信号 ==========
    funding_signal = 0.0
    if abs(current_funding_rate) > FUNDING_THRESHOLD:
        if current_funding_rate > 0:
            funding_signal = -1.0
        elif current_funding_rate < 0:
            funding_signal = 1.0
    
    # ========== 4. 基差信号 ==========
    basis_signal = 0.0
    if abs(current_basis) > BASIS_THRESHOLD:
        if current_basis > 0:
            basis_signal = -1.0
        elif current_basis < 0:
            basis_signal = 1.0
    
    # ========== 5. 趋势过滤 ==========
    trend_direction = 0
    if TREND_FILTER_ENABLED:
        if current_price > sma_trend * 1.001:
            trend_direction = 1
        elif current_price < sma_trend * 0.999:
            trend_direction = -1
    
    # ========== 6. 信号综合 ==========
    raw_side = 0.0
    if REQUIRE_BOTH_FUNDING_AND_BASIS:
        if funding_signal != 0 and basis_signal != 0:
            if funding_signal == basis_signal:
                raw_side = funding_signal
    else:
        if funding_signal != 0:
            raw_side = funding_signal
        elif basis_signal != 0:
            raw_side = basis_signal
    
    # ========== 7. 趋势方向过滤 ==========
    if trend_direction == 1 and raw_side < 0:
        raw_side = 0.0
    elif trend_direction == -1 and raw_side > 0:
        raw_side = 0.0
    
    # ========== 8. Funding 翻转检测 ==========
    if STOP_ON_FUNDING_FLIP and current_pos != 0:
        curr_side = 1 if current_pos > 0 else (-1 if current_pos < 0 else 0)
        if curr_side > 0 and funding_signal < 0:
            raw_side = 0.0
        elif curr_side < 0 and funding_signal > 0:
            raw_side = 0.0
    
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
    
    if MIN_HOLD_BARS > 0 and curr_side != 0:
        if (raw_side == 0.0 or np.sign(raw_side) != curr_side) and FA_HOLD_BARS < MIN_HOLD_BARS:
            return float(curr_side)
    
    return raw_side


# ============================================================
#         Carry + Momentum Multi-Factor Baseline
# ============================================================

def carry_momentum_baseline(
    df: pd.DataFrame,
    current_pos: float,
    current_equity: float,
    MIN_HOLD_BARS: int = 0,
    # Carry/Basis/Funding Rate 参数
    FUNDING_RATE: float = 0.0,
    FUNDING_THRESHOLD: float = 0.00005,  # 降低阈值，允许更多交易机会
    BASIS: float = 0.0,
    BASIS_THRESHOLD: float = 0.0003,  # 降低阈值
    USE_PRICE_MOMENTUM_AS_CARRY: bool = True,  # 启用价格动量作为 carry 代理，确保有信号
    # 趋势过滤参数
    TREND_MA_PERIOD: int = 200,
    TREND_FILTER_ENABLED: bool = True,
    TREND_THRESHOLD: float = 0.001,  # 降低阈值，允许更多交易机会
    # 动量因子参数
    MOMENTUM_LOOKBACK: int = 20,
    MOMENTUM_THRESHOLD: float = 0.002,  # 降低阈值，允许更多交易机会
    MOMENTUM_WEIGHT: float = 0.3,  # 降低动量权重，更依赖 carry 信号
    # 波动率管理参数
    VOL_PERIOD: int = 20,
    VOL_THRESHOLD_HIGH: float = 0.08,
    VOL_THRESHOLD_LOW: float = 0.02,
    VOL_CONTROL_ENABLED: bool = True,
    VOL_SCALING_ENABLED: bool = True,
    # 风险控制参数
    MAX_LEVERAGE: float = 3.0,
    MIN_CARRY_SCORE: float = 0.1,  # 进一步降低最小 carry 得分，确保能够交易
) -> float:
    """
    Carry + Momentum 多因子策略
    
    策略框架：
    1. 方向闸门（趋势因子）：只做顺势方向
    2. 收益来源（Carry/基差/资金费率因子）：决定"值不值得参与"
    3. 质量确认（动量因子）：避免逆势抄底摸顶
    4. 稳健核心（波动率管理）：决定仓位大小与是否暂停
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
            # 使用价格变化作为 funding rate 的代理
            # 价格快速上涨 → 正 funding（做空信号）
            # 增加系数，使信号更容易触发
            current_funding_rate = price_change * 0.2
    
    # ========== 2. 计算技术指标 ==========
    sma_trend = calculate_sma(closes, TREND_MA_PERIOD).iloc[-1]
    atr = calculate_atr(highs, lows, closes, VOL_PERIOD).iloc[-1]
    realized_vol = atr / current_price if current_price > 0 else 0.0
    
    if len(closes) >= MOMENTUM_LOOKBACK:
        momentum = (closes.iloc[-1] - closes.iloc[-MOMENTUM_LOOKBACK]) / closes.iloc[-MOMENTUM_LOOKBACK]
    else:
        momentum = 0.0
    
    # 检查指标有效性
    if np.isnan(sma_trend) or np.isnan(realized_vol) or np.isnan(momentum):
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
    
    # ========== 3. 波动率管理 ==========
    if VOL_CONTROL_ENABLED and realized_vol > VOL_THRESHOLD_HIGH:
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
    
    low_vol_penalty = 1.0
    if VOL_CONTROL_ENABLED and realized_vol < VOL_THRESHOLD_LOW:
        low_vol_penalty = 0.5
    
    vol_scaling = 1.0
    if VOL_SCALING_ENABLED:
        if realized_vol > 0:
            vol_scaling = min(1.5, VOL_THRESHOLD_LOW / max(realized_vol, 0.001))
        else:
            vol_scaling = 1.0
    
    # ========== 4. 趋势过滤 ==========
    trend_direction = 0
    if TREND_FILTER_ENABLED:
        price_to_sma = (current_price - sma_trend) / sma_trend if sma_trend > 0 else 0
        if price_to_sma > TREND_THRESHOLD:
            trend_direction = 1
        elif price_to_sma < -TREND_THRESHOLD:
            trend_direction = -1
    
    # ========== 5. Carry 因子 ==========
    carry_score = 0.0
    carry_signal = 0.0
    
    # 计算 funding_score
    if abs(current_funding_rate) > FUNDING_THRESHOLD:
        # 归一化到 [-1, 1]，使用更大的归一化因子，使信号更容易达到阈值
        funding_score = np.clip(current_funding_rate / 0.0005, -1.0, 1.0)  # 从 0.001 改为 0.0005
    else:
        funding_score = 0.0
    
    # 计算 basis_score
    if abs(current_basis) > BASIS_THRESHOLD:
        basis_score = np.clip(current_basis / 0.0005, -1.0, 1.0)  # 从 0.001 改为 0.0005
    else:
        basis_score = 0.0
    
    # 综合 carry_score
    if funding_score != 0.0 or basis_score != 0.0:
        if funding_score != 0.0 and basis_score != 0.0:
            carry_score = (funding_score + basis_score) / 2.0
        elif funding_score != 0.0:
            carry_score = funding_score
        else:
            carry_score = basis_score
    
    # 生成 carry_signal：降低阈值要求，或者直接使用 carry_score（如果足够大）
    # 如果 carry_score 的绝对值大于 MIN_CARRY_SCORE，或者如果使用价格动量且价格变化足够大
    if abs(carry_score) >= MIN_CARRY_SCORE:
        carry_signal = -carry_score
    elif USE_PRICE_MOMENTUM_AS_CARRY and abs(carry_score) > 0.05:  # 如果使用价格动量，降低阈值到 0.05
        carry_signal = -carry_score
    
    # ========== 6. 动量因子 ==========
    momentum_signal = 0.0
    if abs(momentum) > MOMENTUM_THRESHOLD:
        if momentum > 0:
            momentum_signal = 1.0
        else:
            momentum_signal = -1.0
    
    # ========== 7. 多因子综合 ==========
    raw_side = 0.0
    
    # 基础信号：carry 信号
    if carry_signal != 0.0:
        raw_side = carry_signal
        
        # 动量确认：如果动量与 carry 方向一致，增强信号；如果相反，降权
        if momentum_signal != 0.0:
            if np.sign(carry_signal) == np.sign(momentum_signal):
                # 方向一致，增强信号
                raw_side = carry_signal * (1.0 + MOMENTUM_WEIGHT)
            else:
                # 方向相反，降权（但不完全禁止）
                raw_side = carry_signal * (1.0 - MOMENTUM_WEIGHT)
                # 如果降权后信号太弱，禁止交易
                if abs(raw_side) < MIN_CARRY_SCORE:
                    raw_side = 0.0
        # 如果没有动量信号，仍然使用 carry 信号（允许基于 carry 单独交易）
    
    # 应用波动率惩罚
    raw_side = raw_side * low_vol_penalty
    raw_side = np.clip(raw_side, -1.0, 1.0)
    
    # ========== 8. 趋势方向过滤 ==========
    if trend_direction == 1 and raw_side < 0:
        raw_side = 0.0
    elif trend_direction == -1 and raw_side > 0:
        raw_side = 0.0
    
    # ========== 9. 最小持仓周期逻辑（必须在所有信号计算之后）==========
    # 先更新持仓计数器
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
    
    # 将 raw_side 转换为离散信号（-1, 0, 1），与 livetrading 保持一致
    # 这样可以避免因为 raw_side 的小数变化导致频繁交易
    discrete_side = 0.0
    if raw_side > 0.01:  # 使用小阈值避免浮点误差
        discrete_side = 1.0
    elif raw_side < -0.01:
        discrete_side = -1.0
    else:
        discrete_side = 0.0
    
    # 然后检查最小持仓周期：这个检查必须在所有信号计算之后
    if MIN_HOLD_BARS > 0 and curr_side != 0:
        # 如果新信号与当前持仓方向不同（包括平仓 discrete_side == 0），且未达到最小持仓周期
        # 注意：discrete_side == 0.0 表示平仓信号，np.sign(discrete_side) != curr_side 表示反向信号
        if (discrete_side == 0.0 or np.sign(discrete_side) != curr_side) and CM_HOLD_BARS < MIN_HOLD_BARS:
            return float(curr_side)  # 保持当前仓位，忽略平仓/反向信号
    
    # 返回离散信号，而不是连续的小数值
    return discrete_side

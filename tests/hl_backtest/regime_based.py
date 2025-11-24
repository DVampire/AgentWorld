import numpy as np
import pandas as pd

PRICE_COL = "c"
VOL_COL   = "v"
MAX_LEVERAGE = 5.0

SIGNAL_HISTORY = []
LAST_POS_SIGN = 0
HOLD_BARS     = 0

LAST_BAND     = 0   # 新增：记录上一根 K 的 band

def reset_signal_history():
    """在回测前清空 signal 缓存（可选，在 main 里用）。"""
    global SIGNAL_HISTORY
    SIGNAL_HISTORY = []
    LAST_BAND = 0


def get_signal_df() -> pd.DataFrame:
    """将缓存的 signal 历史转换为 DataFrame."""
    if not SIGNAL_HISTORY:
        return pd.DataFrame()
    df = pd.DataFrame(SIGNAL_HISTORY)
    # 如果有 time 字段，按时间排序
    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
    return df
# ======================================
# 基础：band 分档（你的系统已有）
# ======================================
def band_from_signal(x: float) -> int:
    if x >= 1.2:
        return 2
    if x >= 0.4:
        return 1
    if x <= -1.2:
        return -2
    if x <= -0.4:
        return -1
    return 0



# ======================================
# 1) OBV 计算
# ======================================
def compute_obv(df: pd.DataFrame) -> pd.Series:
    closes = df[PRICE_COL].astype(float)
    vols   = df[VOL_COL].astype(float)

    obv = [0.0]
    for i in range(1, len(df)):
        if closes.iloc[i] > closes.iloc[i-1]:
            obv.append(obv[-1] + vols.iloc[i])
        elif closes.iloc[i] < closes.iloc[i-1]:
            obv.append(obv[-1] - vols.iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)



# ======================================
# 组合策略（纯量价）
# ======================================





def hl_strategy(df: pd.DataFrame, current_pos: float, current_equity: float) -> float:
    """
    Hyperliquid 15m 中频趋势跟随策略 · 专业 V3

    特点：
      - 只做趋势，不做均值回归
      - 使用 MA20 / MA80 + 趋势偏离度 + 8h / 24h TSMOM 判断趋势
      - price 明显高于 MA80 只做多，明显低于 MA80 只做空
      - 持仓至少 24 根 K（6 小时），期间不允许平仓
      - 满足最少持仓后，仅在动量出现“明确反转”时才平仓
      - 不反手：只从有仓 → 平仓，不直接多翻空
    """
    global LAST_POS_SIGN, HOLD_BARS

    # ======================
    # 1. 参数配置
    # ======================
    FAST_MA_WINDOW  = 20     # MA20：约 5 小时
    SLOW_MA_WINDOW  = 80     # MA80：约 20 小时（≈1天）
    MOM8_WINDOW     = 32     # 8 小时动量
    MOM24_WINDOW    = 96     # 24 小时动量
    VOL_WINDOW      = 32     # 8 小时波动率

    # 趋势偏离度阈值（相对 MA80）
    DISP_TH = 0.0025         # 价格相对 MA80 偏离 >0.25% 才认为有明显趋势方向

    # 入场动量阈值（trend must be strong）
    MOM8_ENTER   = 0.0030    # 8h 累积涨跌 > 0.3%
    MOM24_ENTER  = 0.0060    # 24h 累积涨跌 > 0.6%

    # 退出阈值：真正反转才离场（注意是“反向动量”）
    MOM8_REV   = 0.0015      # 8h 反向动量 > 0.15%
    MOM24_REV  = 0.0025      # 24h 反向动量 > 0.25%

    VOL_TH       = 0.0010    # 8h std <0.1% → 不开新仓（太平）
    MIN_HOLD_BARS = 70  # 最少持仓 70 根 

    MIN_LEN = max(FAST_MA_WINDOW, SLOW_MA_WINDOW, MOM24_WINDOW, VOL_WINDOW) + 5
    if len(df) < MIN_LEN or current_equity <= 0:
        return 0.0

    df = df.copy()
    closes = df[PRICE_COL].astype(float)
    vols   = df[VOL_COL].astype(float)

    # ======================
    # 2. 基础因子计算
    # ======================

    ret = closes.pct_change()
    df["ret"] = ret

    ma_fast = closes.rolling(FAST_MA_WINDOW).mean()
    ma_slow = closes.rolling(SLOW_MA_WINDOW).mean()

    tsmom8  = ret.rolling(MOM8_WINDOW).sum()
    tsmom24 = ret.rolling(MOM24_WINDOW).sum()

    vol8h   = ret.rolling(VOL_WINDOW).std(ddof=0)

    price_last   = float(closes.iloc[-1])
    ma_fast_last = float(ma_fast.iloc[-1])
    ma_slow_last = float(ma_slow.iloc[-1])
    mom8_last    = float(tsmom8.iloc[-1])
    mom24_last   = float(tsmom24.iloc[-1])
    vol_last     = float(vol8h.iloc[-1])

    if any(np.isnan(x) for x in (ma_fast_last, ma_slow_last, mom8_last, mom24_last, vol_last)):
        return 0.0

    # ======================
    # 3. 更新当前持仓时长
    # ======================
    curr_sign = 0
    if current_pos > 0:
        curr_sign = 1
    elif current_pos < 0:
        curr_sign = -1

    if curr_sign == 0:
        HOLD_BARS = 0
        LAST_POS_SIGN = 0
    else:
        if curr_sign == LAST_POS_SIGN:
            HOLD_BARS += 1
        else:
            HOLD_BARS = 1
            LAST_POS_SIGN = curr_sign

    # ======================
    # 4. 趋势方向判定（方向锁定 + 偏离度 + 动量）
    # ======================

    diff_ma = (price_last - ma_slow_last) / ma_slow_last

    # 方向锁定：price 相对 MA80 的偏离度决定“牛 / 熊”
    strong_above = diff_ma > DISP_TH   # 明显在 MA80 上方
    strong_below = diff_ma < -DISP_TH  # 明显在 MA80 下方

    # 快慢均线方向一致
    up_bias   = strong_above and (ma_fast_last > ma_slow_last)
    down_bias = strong_below and (ma_fast_last < ma_slow_last)

    # 动量方向：8h & 24h 同向
    up_mom   = (mom8_last > 0) and (mom24_last > 0)
    down_mom = (mom8_last < 0) and (mom24_last < 0)

    has_vol  = (vol_last > VOL_TH)

    long_trend  = up_bias   and up_mom   and has_vol
    short_trend = down_bias and down_mom and has_vol

    # ======================
    # 5. 进入 / 退出逻辑（注意：退出逻辑必须在 MIN_HOLD 之后才触发）
    # ======================
    continuous_signal = 0.0

    if current_pos == 0.0:
        # -------- 空仓：只有趋势非常明确时才入场 --------
        if long_trend and (mom8_last > MOM8_ENTER) and (mom24_last > MOM24_ENTER):
            strength = max(mom8_last / MOM8_ENTER, mom24_last / MOM24_ENTER)
            continuous_signal = float(np.clip(strength, 1.0, 3.0))   # 1~3 → band=1/2
        elif short_trend and (mom8_last < -MOM8_ENTER) and (mom24_last < -MOM24_ENTER):
            strength = min(mom8_last / MOM8_ENTER, mom24_last / MOM24_ENTER)  # 负数
            continuous_signal = float(np.clip(strength, -3.0, -1.0))          # -1~-3 → band=-1/-2
        else:
            continuous_signal = 0.0

    else:
        # -------- 有仓：先看 MIN_HOLD_BARS，再看 exit --------
        if HOLD_BARS < MIN_HOLD_BARS:
            # 没到最少持仓时长 → 强制继续持有
            continuous_signal = 1.0 if current_pos > 0 else -1.0
        else:
            # 已经满足最少持仓要求 → 可以考虑退出
            if current_pos > 0:
                # 多头：只有在出现“空头动量”时才平仓
                exit_cond = (not long_trend) or (mom8_last < -MOM8_REV) or (mom24_last < -MOM24_REV)
                continuous_signal = 0.0 if exit_cond else 1.0
            else:
                # 空头：只有在出现“多头动量”时才平仓
                exit_cond = (not short_trend) or (mom8_last > MOM8_REV) or (mom24_last > MOM24_REV)
                continuous_signal = 0.0 if exit_cond else -1.0

    continuous_signal = float(np.clip(continuous_signal, -3.0, 3.0))

    # ======================
    # 6. 映射到 band，并做仓位 sizing
    # ======================

    # 6. 映射到 band
    band = band_from_signal(continuous_signal)

    global LAST_BAND

    # ======== 关键修改：band 没变就不调仓 ========
    # 注意：对于刚开回测那段，current_pos 一定是 0，可以正常开第一笔
    if band == LAST_BAND:
        # band 没有变化 → 不想调整仓位 → 直接维持 current_pos
        target_pos = current_pos
    else:
        # band 发生变化 → 才重新计算目标仓位
        if price_last <= 0:
            return 0.0

        max_notional = current_equity * MAX_LEVERAGE
        band_abs = abs(band)
        if band_abs == 0:
            use_notional = 0.0
        elif band_abs == 1:
            use_notional = max_notional * 0.3
        else:
            use_notional = max_notional * 0.6

        pos_abs = use_notional / price_last

        if band > 0:
            target_pos = pos_abs
        elif band < 0:
            target_pos = -pos_abs
        else:
            target_pos = 0.0

    # 更新 LAST_BAND
    LAST_BAND = band


    # ======================
    # 7. 日志与记录
    # ======================
    last_time = df["T"].iloc[-1] if "T" in df.columns else None
    if long_trend:
        trend_str = "UP"
    elif short_trend:
        trend_str = "DOWN"
    else:
        trend_str = "NONE"

    print(
        f"\n[T={last_time}] price={price_last:.2f} | trend={trend_str} | "
        f"ma_fast={ma_fast_last:.2f} | ma_slow={ma_slow_last:.2f} | "
        f"mom8h={mom8_last:.4f} | mom24h={mom24_last:.4f} | vol8h={vol_last:.4f} | "
        f"hold_bars={HOLD_BARS} | cont={continuous_signal:.3f} → band={band:+.1f}"
    )

    SIGNAL_HISTORY.append(
        {
            "time": last_time,
            "price": price_last,
            "trend": trend_str,
            "ma_fast": ma_fast_last,
            "ma_slow": ma_slow_last,
            "mom8h": mom8_last,
            "mom24h": mom24_last,
            "vol8h": vol_last,
            "hold_bars": HOLD_BARS,
            "continuous_signal": continuous_signal,
            "band": band,
            "equity_dummy": float(current_equity),
        }
    )

    return float(target_pos)

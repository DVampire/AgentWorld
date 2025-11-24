from typing import Tuple

import numpy as np
import pandas as pd
import math


# ====== 全局 signal 缓存 ======
SIGNAL_HISTORY = []  # 每次 hl_strategy 调用都会往这里 append 一条 dict


def reset_signal_history():
    """在回测前清空 signal 缓存（可选，在 main 里用）。"""
    global SIGNAL_HISTORY
    SIGNAL_HISTORY = []


def get_signal_df() -> pd.DataFrame:
    """将缓存的 signal 历史转换为 DataFrame."""
    if not SIGNAL_HISTORY:
        return pd.DataFrame()
    df = pd.DataFrame(SIGNAL_HISTORY)
    # 如果有 time 字段，按时间排序
    if "time" in df.columns:
        df = df.sort_values("time").reset_index(drop=True)
    return df


# ====== 策略参数：可以按你之前那套来改 ======
PRICE_COL = "c"

MA_WINDOW       = 30
ROLL_WINDOW     = 60




ZSCORE_WINDOW   = 10
CORR_WINDOW  = 10  # ret 与 vol 相关性窗口


MOM_WINDOW = 20


VOL_WINDOW      = 30


# 波动率 + Amihud 择时
RV_WINDOW       = 10     # realized volatility 窗口
AMIHUD_WINDOW   = 10     # Amihud illiquidity 窗口
    # return 计算的最小长度



RISK_FRACTION = 0.3
MAX_LEVERAGE  = 5.0
MIN_SIGNAL_STEP = 0.3      # 死区阈值，防止频繁换仓


# ====== 工具函数 ======

def rolling_z(series: pd.Series, window: int) -> pd.Series:
    """rolling z-score: (x - mean) / std"""
    mean = series.rolling(window).mean()
    std = series.rolling(window).std(ddof=0)
    return (series - mean) / std


# ====== 因子：动量 + Roll ======

def compute_mean_reversion(df: pd.DataFrame):
    """
    均值回归 + 成交量放大 + roll 成本
    输入 df 至少包含:
        'c': close 价格
        'v': volume 成交量
    返回:
        dev_vol_z:   成交量加权“偏离均值” z-score（主因子，均值回归）
        roll_z:      roll 成本 z-score（辅助抑制）
        last_close:  最新收盘价
    """
    closes  = df["c"].astype(float)
    volumes = df["v"].astype(float)
    # ========= 1. 价格相对均线偏离 =========
    ma = closes.rolling(MA_WINDOW).mean()
    df["dev"] = closes / ma - 1.0   # 高于均值为正，低于均值为负

    # ========= 2. Roll 成本（沿用你的定义，偏反转）=========
    df["dp"] = closes.diff()
    cov_dp = df["dp"].rolling(ROLL_WINDOW).cov(df["dp"].shift(1))
    df["roll"] = -2 * cov_dp.abs()

    # ========= 3. 成交量归一化 =========
    vol_mean = volumes.rolling(VOL_WINDOW).mean()
    df["vol_norm"] = volumes / (vol_mean + 1e-8)
    df["vol_norm"] = df["vol_norm"].clip(lower=0.0, upper=5.0)

    # ========= 4. 成交量加权偏离（高量时更信号）=========
    df["dev_vol"] = df["dev"] * df["vol_norm"]

    # ========= 5. 统一做 z-score =========
    df["dev_vol_z"] = rolling_z(df["dev_vol"], ZSCORE_WINDOW)
    df["roll_z"]    = rolling_z(df["roll"],    ZSCORE_WINDOW)

    df = df.dropna()
    if df.empty:
        raise ValueError("Not enough data for mean reversion factors")

    last = df.iloc[-1]

    return (
        float(last["dev_vol_z"]),  # 主因子
        float(last["roll_z"]),     # 辅助因子
        float(last["c"]),          # 最新价格
    )


# ====== 因子：波动择时（占位，可以换成 Amihud） ======

def compute_dev_vol_z(df: pd.DataFrame) -> float:
    """
    这里用 log return 的 rolling std 作为波动率，再做 z-score。
    后面你如果想换成 Amihud，只要改这个函数内部即可。
    """
    df = df.copy()
    closes = df[PRICE_COL]
    df["ret"] = np.log(closes).diff()
    vol = df["ret"].rolling(VOL_WINDOW).std(ddof=0)
    df["dev_vol_z"] = rolling_z(vol, ZSCORE_WINDOW)
    df = df.dropna()
    if df.empty:
        return 0.0
    return float(df["dev_vol_z"].iloc[-1])


def compute_log_return_volume_corr(df: pd.DataFrame) -> float:
    df = df.copy()
    closes = df[PRICE_COL]
    # === 1. 计算 log return ===
    df["ret"] = np.log(closes).diff()
    vols   = df["v"].astype(float)

    obv = compute_obv(df)
    slope= np.sign(obv.diff().rolling(5).mean().iloc[-1])
    # # === 2. 成交量 z-score（VOL_WINDOW） ===
    # vol_mean = vols.rolling(VOL_WINDOW).mean()
    # vol_std  = vols.rolling(VOL_WINDOW).std(ddof=0)
    # df["vol_z"] = (vols - vol_mean) / vol_std.replace(0, np.nan)
    tp = (df["h"] + df["l"] + df["c"]) / 3
    # vwap = (tp * vols).cumsum() / vols.cumsum()
    # corr_pv_series = rolling_z(vwap.rolling(CORR_WINDOW).corr(vols),ZSCORE_WINDOW)
    # corr_pv = float(corr_pv_series.iloc[-1])  # ret-volume corr
    # corr_rv_series = rolling_z(df["ret"].rolling(CORR_WINDOW).corr(vols),ZSCORE_WINDOW)
    # corr_rv = float(corr_rv_series.iloc[-1]) 
    df["dev_corr"] = -(closes/tp-1).rolling(10).corr(vols.shift(1))


    # df["mom_z"]  = rolling_z(df["mom"],  ZSCORE_WINDOW)



    df = df.dropna()
    if df.empty:
        # 你可以按自己习惯改这里（return None / 抛异常）
        raise ValueError("Error")

    last = df.iloc[-1]

    return (
        float(last["dev_corr"]),
        float(slope),
        float(last["c"])
    )


    

# ====== 连续信号 -> 档位 ======

def band_from_signal(x: float) -> int:
    """
    连续信号 -> 档位:
      [-inf, -1.0) -> -2
      [-1.0, -0.3) -> -1
      [-0.3, 0.3)  ->  0
      [0.3, 1.0)   ->  1
      [1.0, +inf)  ->  2
    阈值你可以按自己习惯调。
    """
    if x >= 1.0:
        return 2
    if x >= 0.3:
        return 1
    if x <= -1.0:
        return -2
    if x <= -0.3:
        return -1
    return 0


# ====== 策略入口：给回测引擎调用 ======

def hl_strategy(df: pd.DataFrame, current_pos: float, current_equity: float) -> float:
    """
    策略大脑：
    - 输入：截至当前的历史 df、当前持仓、当前权益
    - 输出：目标持仓（张数，>0 做多，<0 做空）
    """
    # dev_vol_z, roll_z, last_price = compute_mean_reversion(df)
    # dev_vol_z = compute_dev_vol_z(df)

    # ========= 信号逻辑（示例） =========
    # if abs(dev_vol_z) > 2.0:
    #     # 波动太高，观望
    #     continuous_signal = 0.0
    # else:
    #     # 简单 mean-reversion：动量越高越想做空，动量越低越想做多
    # 波动适中时放大一点
    # vol_scale = max(0.0, 1.5 - abs(dev_vol_z))
    # continuous_signal *= vol_scale

    dev_vol_z, roll_z, last_price = compute_mean_reversion(df)
    # 限幅避免极端值
    # dev_vol_z = max(min(dev_vol_z, 5.0), -5.0)

    # 均值回归方向：dev_vol_z 为正 → 高于均值 → 做空，故取 -dev_vol_z
    alpha_signal = -math.tanh(dev_vol_z)

    # 2) 波动率 + Amihud 择时（只缩放，不“强化”）
    closes = df["c"]
    rets   = closes.pct_change()

    # realized volatility
    rv = rets.rolling(RV_WINDOW).std()
    rv_z = rolling_z(rv, ZSCORE_WINDOW)

    # Amihud illiquidity: 平均 |return| / volume
    amihud = (rets.abs() / (df["v"] + 1e-8)).rolling(AMIHUD_WINDOW).mean()
    amihud_z = rolling_z(amihud, ZSCORE_WINDOW)

    # 取最后一个值（前面已保证有足够数据）
    last_rv_z = float(rv_z.dropna().iloc[-1])
    last_amihud_z = float(amihud_z.dropna().iloc[-1])

    # 综合“波动 + 流动性风险”
    # vol_risk = max(0.0, 0.7 * last_rv_z + 1.0 * last_amihud_z)

    vol_risk = max(0.0, 0.3 * last_rv_z + 0.2 * last_amihud_z)
    vol_gate = max(0.35, 1 / (1 + vol_risk))
    # vol_gate = 1.0 / (1.0 + vol_risk)  # 风险越高 gate 越小，0~1

    # 3) roll 风险过滤（轻量）
    roll_gate = 1.0 / (1.0 + abs(roll_z))

    # 4) 最终连续信号：均值回归 × 波动择时 × roll 过滤
    continuous_signal = -alpha_signal
    # continuous_signal = max(-3.0, min(3.0, continuous_signal))

    
    band = band_from_signal(continuous_signal)

    # ====== 日志（和你之前风格类似） ======
    print(
        f"\n[FACTORS] price={last_price:.6f} | dev_vol_z={dev_vol_z:.3f} | roll_z={roll_z:.3f}"
    )
    print(f"[SIGNAL]  cont={continuous_signal:.3f} → band={band:+.1f}")

    # ====== 仓位大小：按 band 控制杠杆 ======
    if last_price <= 0 or current_equity <= 0:
        target_pos = 0.0
    else:
        max_notional = current_equity * MAX_LEVERAGE

        band_abs = abs(band)
        if band_abs == 0:
            use_notional = 0.0
        elif band_abs == 1:
            use_notional = max_notional * 0.25
        else:  # 2
            use_notional = max_notional * 0.5

        pos_abs = use_notional / last_price
        if band > 0:
            target_pos = pos_abs
        elif band < 0:
            target_pos = -pos_abs
        else:
            target_pos = 0.0

    # ====== 把本次调用的因子 & 信号存起来，用于画图 ======
    # time 用 df 最后一行的收盘时间 T（如果有）
    time_val = df["T"].iloc[-1] if "T" in df.columns else None
    SIGNAL_HISTORY.append(
        {
            "time": time_val,
            "price": last_price,
            "roll_z": roll_z,
            "dev_vol_z": dev_vol_z,
            "continuous_signal": continuous_signal,
            "band": band,
            "equity_dummy": float(current_equity),  # 可选：存一下当时权益
        }
    )

    return target_pos



def compute_obv(df: pd.DataFrame) -> pd.Series:
    closes = df["c"].astype(float)
    vols   = df["v"].astype(float)

    obv = [0]
    for i in range(1, len(df)):
        if closes.iloc[i] > closes.iloc[i-1]:
            obv.append(obv[-1] + vols.iloc[i])
        elif closes.iloc[i] < closes.iloc[i-1]:
            obv.append(obv[-1] - vols.iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=df.index)




def hl_strategy_2(df: pd.DataFrame, current_pos: float, current_equity: float) -> float:
    """
    只用量价指标作为 signal 的极简策略（适配你的回测引擎）：

    - 输入:
        df: 截至当前的历史K线（至少包含 'c','v','T'）
        current_pos: 当前持仓张数
        current_equity: 当前账户权益
    - 输出:
        target_pos: 目标持仓张数（float）
    """

    min_len = max(VOL_WINDOW, CORR_WINDOW) + 5
    if len(df) < min_len:
        return 0.0

    # === 5. 构造连续信号 ===
    # 基本信号：signed_vol
    # 叠加一点 corr_rv（量价同向时放大，反向时抑制）
    dev_corr,slope,last_price = compute_log_return_volume_corr(df)


    continuous_signal = slope


    # 裁剪，防止极端
    continuous_signal = float(np.clip(continuous_signal, -3.0, 3.0))

    # === 6. 映射到档位 band ===
    band = band_from_signal(continuous_signal)


    # print(
    #     f"\n[FACTORS] price={last_price:.6f} | corr_rv={corr_rv:.3f} | corr_pv={corr_pv:.3f}"
    # )
    # print(f"[SIGNAL]  cont={continuous_signal:.3f} → band={band:+.1f}")

    # ====== 仓位大小：按 band 控制杠杆 ======
    if last_price <= 0 or current_equity <= 0:
        target_pos = 0.0
    else:
        max_notional = current_equity * MAX_LEVERAGE

        band_abs = abs(band)
        if band_abs == 0:
            use_notional = 0.0
        elif band_abs == 1:
            use_notional = max_notional * 0.25
        else:  # 2
            use_notional = max_notional * 0.5

        pos_abs = use_notional / last_price
        if band > 0:
            target_pos = pos_abs
        elif band < 0:
            target_pos = -pos_abs
        else:
            target_pos = 0.0

    # ====== 把本次调用的因子 & 信号存起来，用于画图 ======
    # time 用 df 最后一行的收盘时间 T（如果有）
    time_val = df["T"].iloc[-1] if "T" in df.columns else None
    SIGNAL_HISTORY.append(
        {
            "time": time_val,
            "price": last_price,
            "continuous_signal": continuous_signal,
            "band": band,
            "equity_dummy": float(current_equity),  # 可选：存一下当时权益
        }
    )

    return target_pos


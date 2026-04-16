import pandas as pd
import backtrader as bt
import empyrical as ep
from typing import Union,Tuple
import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "get_strategy_return",
    "get_strategy_cumulative_return",
    "get_strategy_maxdrawdown",
    "get_strategy_sharpe_ratio",
]


def get_strategy_return(
    strat: bt.Cerebro,
) -> pd.Series:
    """
    获取策略的收益率数据。

    参数：
        - strat: bt.Cerebro 对象，策略对象。


    返回：
        - pd.Series 对象，策略的收益率数据。
    """
    return pd.Series(strat.analyzers.getbyname("time_return").get_analysis())

def get_strategy_cumulative_return(
    strat: bt.Cerebro,
    starting_value: int = 0,
) -> pd.Series:
    """
    获取策略的累计收益率数据。

    参数：
        - strat: bt.Cerebro 对象，策略对象。


    返回：
        - pd.Series 对象，策略的累计收益率数据。
    """
    return ep.cum_returns(get_strategy_return(strat), starting_value=starting_value)


def get_strategy_maxdrawdown(
    strat: bt.Cerebro,
) -> float:
    """
    获取策略的最大回撤。

    参数：
        - strat: bt.Cerebro 对象，策略对象。
    返回：
        - float，策略的最大回撤。

    """
    return ep.max_drawdown(get_strategy_return(strat))


def get_strategy_sharpe_ratio(
    strat: bt.Cerebro,
    risk_free: float = 0.0,
    annulization: int = 365,
) -> float:
    """
    获取策略的夏普比率。

    参数：
        - strat: bt.Cerebro 对象，策略对象。
        - risk_free: float，无风险利率，默认为0.0。
        - period: int，年化周期，默认为252。
    返回：
        - float,策略的夏普比率。
    """
    return ep.sharpe_ratio(
        get_strategy_return(strat),
        risk_free=risk_free,
        annualization=annulization,
    )
def plot_cumulative_return(
    strat: bt.Cerebro,
    minute_benchmark: pd.Series = None,
    ax: plt.Axes = None,
    figure: Tuple[int, int] = (16, 4),
    title: str = "",
) -> plt.Axes:
    """
    绘制累积收益图。

    参数：
        - strat: bt.Cerebro 对象，策略对象。
        - minute_benchmark: pd.Series 对象，分钟级别的基准收益率数据，默认为 None。
        - ax: plt.Axes 对象，用于绘制图形的坐标轴对象，默认为 None。
        - figure: Tuple[int, int] 对象，图形的尺寸，默认为 (16, 4)。
        - title: str 对象，图形的标题，默认为空字符串。

    返回：
        - plt.Axes 对象，绘制累积收益图的坐标轴对象。
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figure)

    if title == "":
        title = "Cumulative Return"

    returns: pd.Series = get_strategy_return(strat)
    ep.cum_returns(returns, 1).plot(ax=ax, label="strategy", color="red")

    if minute_benchmark is not None:
        bench = minute_benchmark.resample("D").last().dropna()
        (bench / bench.iloc[0]).plot(color="darkgray", label="benchmark", ax=ax)

    plt.title(title)
    ax.grid()
    ax.axhline(1, color="black", ls="-")
    ax.legend()
    return ax


def get_strategy_win_rate(
    strat: bt.Cerebro,
) -> pd.DataFrame:
    """
    获取策略的胜率

    参数：
        - strat: bt.Cerebro 对象，策略对象。
    返回：
        - pd.DataFrame，策略的交易记录。
    """    
    ta_analyzer = strat.analyzers.getbyname('trades')
    ta_data = ta_analyzer.get_analysis()
    records = []
    
    closed = ta_data.get('total', {}).get('closed', 0)
    win = ta_data.get('won', {}).get('total', 0)
    loss = ta_data.get('lost', {}).get('total', 0)

    results = {
        'closed': closed,
        'win': win,
        'loss': loss,
        'win_rate': win / closed if closed > 0 else 0,
    }
    return pd.DataFrame([results])


def get_strategy_total_commission(
    strat: bt.Cerebro,
) -> float:
    """
    获取策略的总手续费

    参数：
        - strat: bt.Cerebro 对象，策略对象。
    返回：
        - float，策略的总手续费。
    """    
    
    tc_analyzer = strat.analyzers.getbyname('total_commission')
    total_commission = tc_analyzer.get_analysis().get('total_commission', 0.0)
    return total_commission


def get_excess_return(
    strat: bt.Cerebro,
    minute_benchmark: pd.Series,
    benchmark_is_return: bool = False,   # False=price/close; True=return
) -> float:
    """
    返回策略相对 buy&hold 的超额收益（float）：
        excess = (1+R_strat_total)/(1+R_bh_total) - 1

    参数
    - strat: bt.Cerebro（已 run 过，或 get_strategy_return 内部会 run）
    - minute_benchmark: 1m 基准序列（价格/close 或 return）
    - benchmark_is_return: minute_benchmark 是否已经是 return 序列

    返回
    - float: 超额收益
    """
    # 1) 拿到策略 1m return（建议 get_strategy_return 返回 pd.Series 或 dict）
    strat_ret = get_strategy_return(strat)
    sr = pd.Series(strat_ret).sort_index()

    # 防御：去掉 NaN
    sr = sr.dropna()
    if sr.empty:
        return 0.0

    # 2) 确定比较窗口：用策略 return 的首尾时间戳
    t0, t1 = sr.index[0], sr.index[-1]

    t0 = pd.Timestamp(t0).tz_localize("UTC")
    t1 = pd.Timestamp(t1).tz_localize("UTC")
    # 3) 基准：对齐到同一窗口 & 1m 网格
    bm = minute_benchmark.sort_index()

    if benchmark_is_return:
        br = bm.resample("T").ffill()
        br = br.loc[t0:t1].dropna()
        if br.empty:
            return 0.0
        # 累计 buy&hold return（复利）
        R_bh = (1.0 + br).prod() - 1.0
    else:
        # minute_benchmark 是价格/close
        px = bm.resample("T").ffill()
        px = px.loc[t0:t1].dropna()
        
        if px.empty:
            return 0.0
        # buy&hold 累计收益
        R_bh = (px.iloc[-1] / px.iloc[0]) - 1.0

    # 4) 策略累计收益（复利）
    R_strat = (1.0 + sr).prod() - 1.0

    # 5) 相对超额（ratio-form）
    excess = (1.0 + R_strat) / (1.0 + R_bh) - 1.0
    return float(excess)



def get_relative_equity_curve(
    strat: bt.Cerebro,
    minute_benchmark: pd.Series,
    benchmark_is_return: bool = False,
) -> pd.DataFrame:
    """
    计算策略 vs Buy&Hold 的【相对净值曲线】

    返回 DataFrame（index=时间）:
        Ws       : 策略净值
        Wb       : 基准净值（buy&hold）
        W_rel    : 相对净值 = Ws / Wb
        outperf  : 相对超越幅度 = W_rel - 1
    """

    # -------------------------------------------------
    # 1) 策略 1m return → Series
    # -------------------------------------------------
    strat_ret = get_strategy_return(strat)  # {dt: ret} 或 Series
    sr = pd.Series(strat_ret).sort_index().dropna()
    if sr.empty:
        return pd.DataFrame()

    t0, t1 = sr.index[0], sr.index[-1]

    t0 = pd.Timestamp(t0).tz_localize("UTC")
    t1 = pd.Timestamp(t1).tz_localize("UTC")

    # 策略净值
    Ws = (1.0 + sr).cumprod()

    # -------------------------------------------------
    # 2) 基准：构造 buy&hold 的“净值曲线”
    # -------------------------------------------------
    bm = minute_benchmark.sort_index()

    if benchmark_is_return:
        # benchmark 已是 return
        br = bm.resample("T").ffill().loc[t0:t1].dropna()
        if br.empty:
            return pd.DataFrame()

        Wb = (1.0 + br).cumprod()

    else:
        # benchmark 是价格 / close
        px = bm.resample("T").ffill().loc[t0:t1].dropna()
        if px.empty:
            return pd.DataFrame()

        # buy&hold 净值 = 价格归一化
        Wb = px / px.iloc[0]

    # -------------------------------------------------
    # 3) 严格时间对齐（非常关键）
    # -------------------------------------------------
    idx = Ws.index.intersection(Wb.index)
    Ws = Ws.reindex(idx)
    Wb = Wb.reindex(idx)

    # -------------------------------------------------
    # 4) 相对净值曲线
    # -------------------------------------------------
    W_rel = Ws / Wb
    outperf = W_rel - 1.0

    return pd.DataFrame({
        "Ws": Ws,
        "Wb": Wb,
        "W_rel": W_rel,
        "outperf": outperf,
    })

def path_outperformance_score(W_rel: pd.Series, mode: str = "min") -> float:
    """
    W_rel: 相对净值曲线 (Ws/Wb), index=时间
    mode:
      - "min"            : min(W_rel)  (>=1 表示全程跑赢)
      - "max_shortfall"  : max(0, 1-min(W_rel)) (0 表示全程跑赢, 越大越差)
      - "violation_area" : sum(max(0, 1-W_rel)) (0 表示全程跑赢, 越大越差)
      - "rel_mdd"        : relative max drawdown on W_rel (越小越好)
    """
    w = W_rel.dropna()
    if w.empty:
        return float("nan")

    if mode == "min":
        return float(w.min())

    if mode == "max_shortfall":
        return float(max(0.0, 1.0 - w.min()))

    if mode == "violation_area":
        return float(np.maximum(0.0, 1.0 - w).sum())

    if mode == "rel_mdd":
        peak = w.cummax()
        dd = 1.0 - w / peak
        return float(dd.max())

    raise ValueError(f"unknown mode={mode}")
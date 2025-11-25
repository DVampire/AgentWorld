from data_store import ensure_history_csv
from strategy import PRICE_COL, MAX_LEVERAGE
from regime_based import hl_strategy,get_signal_df,reset_signal_history
from backtest import run_backtest
import matplotlib.pyplot as plt
import os

# ====== 基本配置 ======
COIN = "BTC"
INTERVAL = "15m"

LOOKBACK_CANDLES = 6000
USE_TESTNET = False

INITIAL_EQUITY = 1000.0
TAKER_FEE_RATE = 0.00005
SLIPPAGE_BPS = 1.0


def main():
    print(f"[INIT] ensure history for {COIN} {INTERVAL}, lookback={LOOKBACK_CANDLES}")
    df = ensure_history_csv(
        COIN,
        INTERVAL,
        lookback_candles=LOOKBACK_CANDLES,
        data_dir="data",
        use_testnet=USE_TESTNET,
        price_col=PRICE_COL,
        local=False
    )
    print(f"[DATA] {len(df)} candles | {df['t'].iloc[0]} -> {df['t'].iloc[-1]}")

    reset_signal_history()

    result = run_backtest(
        df,
        strategy_fn=hl_strategy,
        initial_equity=INITIAL_EQUITY,
        max_leverage=MAX_LEVERAGE,
        taker_fee_rate=TAKER_FEE_RATE,
        slippage_bps=SLIPPAGE_BPS,
        price_col=PRICE_COL,
        start_index=100,
        interval=INTERVAL
    )

    print("\n=== Backtest Stats ===")
    for k, v in result["stats"].items():
        print(f"{k:15s}: {v: .4f}")

    print("\n=== Last 5 equity points ===")
    print(result["equity_curve"].tail())

    print("\n=== Last 5 trades ===")
    print(result["trades"].tail())

    # ==========================================================
    # 导出 CSV
    # ==========================================================
    EXPORT_DIR = "exports"
    os.makedirs(EXPORT_DIR, exist_ok=True)

    signal_df = get_signal_df()
    signal_df.to_csv(f"{EXPORT_DIR}/signals.csv", index=False)
    print(f"[EXPORT] signals saved to {EXPORT_DIR}/signals.csv")

    result["equity_curve"].to_csv(f"{EXPORT_DIR}/equity_curve.csv", index=True)
    print(f"[EXPORT] equity saved to {EXPORT_DIR}/equity_curve.csv")

    result["trades"].to_csv(f"{EXPORT_DIR}/trades.csv", index=False)
    print(f"[EXPORT] trades saved to {EXPORT_DIR}/trades.csv")

    # ==========================================================
    # 作图
    # ==========================================================
    if not signal_df.empty:
        equity_curve = result["equity_curve"]

        fig, (ax1, ax2,ax3) = plt.subplots(3, 1, sharex=True, figsize=(12, 12))

        ax1.plot(signal_df["time"], signal_df["continuous_signal"], label="continuous_signal")
        ax1.step(signal_df["time"], signal_df["band"], where="post", label="band")
        ax1.set_ylabel("Signal / Band")
        ax1.legend(loc="upper left")
        ax1.grid(True)

        ax2.plot(equity_curve.index, equity_curve.values, label="equity")
        ax2.set_ylabel("Equity")
        ax2.set_xlabel("Time")
        ax2.legend(loc="upper left")
        ax2.grid(True)


        ax3.plot(signal_df["time"], signal_df["price"].values, label="price")
        ax3.set_ylabel("Price")
        ax3.set_xlabel("Time")
        ax3.legend(loc="upper left")
        ax3.grid(True)

        plt.tight_layout()
        plt.show()
    else:
        print("[PLOT] signal_df is empty, skip plotting.")


if __name__ == "__main__":
    main()

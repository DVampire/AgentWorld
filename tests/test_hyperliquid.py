import json
import time
import threading
import websocket

WS_URL = "wss://api.hyperliquid-testnet.xyz/ws"

def on_open(ws):
    print("✅ Connected to Hyperliquid WebSocket")

    subs = [
        # 1m K线 (OHLCV)
        {
            "method": "subscribe",
            "subscription": {
                "type": "candle",
                "coin": "BTC",
                "interval": "1m"
            }
        },
        # 实时成交
        {
            "method": "subscribe",
            "subscription": {
                "type": "trades",
                "coin": "BTC"
            }
        },
        # L2 order book
        {
            "method": "subscribe",
            "subscription": {
                "type": "l2Book",
                "coin": "BTC"
            }
        }
    ]

    for s in subs:
        ws.send(json.dumps(s))
        time.sleep(0.1)


def handle_candle(data):
    """
    data: Candle[] 数组
    Candle 定义见官方文档：
    t/T: 开/收时间 (ms), s: symbol, i: interval,
    o/c/h/l: 价格, v: volume, n: trades 数量
    """
    if not isinstance(data, list):
        print("[CANDLE] Unexpected format:", data)
        return

    for c in data:
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(c["t"] / 1000))
        print(
            f"[CANDLE] {c['s']} {c['i']} {ts} "
            f"O:{c['o']} H:{c['h']} L:{c['l']} C:{c['c']} V:{c['v']} N:{c['n']}"
        )


def handle_trades(data):
    """
    data: WsTrade[] 数组
    """
    if not isinstance(data, list):
        print("[TRADES] Unexpected format:", data)
        return

    for t in data:
        ts = time.strftime("%H:%M:%S", time.gmtime(t["time"] / 1000))
        print(
            f"[TRADE] {t['coin']} {ts} "
            f"{t['side']} px:{t['px']} sz:{t['sz']}"
        )


def handle_l2book(data):
    """
    data: WsBook:
    {
      "coin": str,
      "levels": [ [WsLevel...], [WsLevel...] ], # [bids, asks]
      "time": int
    }
    WsLevel: { px: string, sz: string, n: int }
    """
    if not isinstance(data, dict):
        print("[L2BOOK] Unexpected format:", data)
        return

    levels = data.get("levels")
    if not levels or len(levels) != 2:
        print("[L2BOOK] Missing levels:", data)
        return

    bids, asks = levels
    best_bid = bids[0] if bids else None
    best_ask = asks[0] if asks else None

    if best_bid and best_ask:
        print(
            f"[L2BOOK] {data.get('coin')} "
            f"Bid:{best_bid['px']} ({best_bid['sz']})  "
            f"Ask:{best_ask['px']} ({best_ask['sz']})"
        )
    else:
        print("[L2BOOK] No BBO:", data)


def on_message(ws, message):
    try:
        msg = json.loads(message)
    except Exception as e:
        print("❌ JSON parse error:", e, message)
        return

    channel = msg.get("channel")
    data = msg.get("data")

    if channel == "subscriptionResponse":
        # 订阅确认
        print("🟢 Subscribed:", data.get("subscription"))
        return

    if channel == "candle":
        handle_candle(data)
    elif channel == "trades":
        handle_trades(data)
    elif channel == "l2Book":
        handle_l2book(data)
    else:
        # 你也可以暂时忽略其它 channel
        # print("ℹ️ Other message:", msg)
        pass


def on_error(ws, error):
    print("❌ Error:", error)


def on_close(ws, status_code, close_msg):
    print("🔴 Connection closed:", status_code, close_msg)


def run_ws():
    ws = websocket.WebSocketApp(
        WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    # 自动发 ping 保活
    ws.run_forever(ping_interval=30, ping_timeout=10)


if __name__ == "__main__":
    t = threading.Thread(target=run_ws)
    t.daemon = True
    t.start()

    # 简单阻塞主线程，方便看输出
    while True:
        time.sleep(1)

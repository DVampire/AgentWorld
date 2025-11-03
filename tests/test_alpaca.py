import os
from dotenv import load_dotenv
load_dotenv(verbose=True)
from alpaca.data.historical import CryptoHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime
import asyncio
from alpaca.data.live import StockDataStream, CryptoDataStream  

# no keys required for crypto data
client = CryptoHistoricalDataClient(
    api_key=os.getenv("ALPACA_PAPER_TRAING_API_KEY"),
    secret_key=os.getenv("ALPACA_PAPER_TRAING_SECRET_KEY")
)

request_params = CryptoBarsRequest(
                    symbol_or_symbols=["BTC/USD"],
                    timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                    start=datetime.strptime("2025-01-01", '%Y-%m-%d'),
                    end=datetime.strptime("2025-01-31", '%Y-%m-%d')
                )

bars = client.get_crypto_bars(request_params)

print(bars)

stream = CryptoDataStream(
    api_key=os.getenv("ALPACA_PAPER_TRAING_API_KEY"),
    secret_key=os.getenv("ALPACA_PAPER_TRAING_SECRET_KEY"),
)

async def bars_handler(data):
    print("Crypto Bars:", data)
    
async def subscribe_quotes(data):
    print("Crypto Quotes:", data)

stream.subscribe_quotes(subscribe_quotes, "BTC/USD")

try:
    stream.run()
except KeyboardInterrupt:
    print("Stopped by user")
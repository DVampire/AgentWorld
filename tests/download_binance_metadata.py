"""Download Binance cryptocurrency metadata and statistics.

This script demonstrates what data Binance API can provide, similar to the stock data
structure in exp.json. Note that Binance provides trading data, not company fundamentals.

Binance can provide:
- 24hr ticker statistics (price, volume, changes, etc.)
- Exchange info (symbol, base asset, quote asset, trading rules)
- Historical OHLCV data

Binance CANNOT provide (because cryptocurrencies are not companies):
- Company name, CEO, industry, sector
- Company description, website, address
- Beta, market cap (in traditional sense)
- Full-time employees, IPO date
"""

import sys
from pathlib import Path
import json
from typing import Dict, Any, List

# Add project root to path
root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

try:
    from binance.spot import Spot
except ImportError:
    try:
        from binance import Spot
    except ImportError:
        print("❌ binance-connector not installed. Install with: pip install binance-connector")
        sys.exit(1)


def get_24hr_ticker(symbol: str) -> Dict[str, Any]:
    """Get 24hr ticker statistics for a symbol.
    
    This provides similar data to exp.json's price, changes, volAvg, range fields.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
    
    Returns:
        Dictionary with 24hr ticker statistics
    """
    client = Spot()
    ticker = client.ticker_24hr(symbol=symbol)
    return ticker


def get_exchange_info(symbol: str) -> Dict[str, Any]:
    """Get exchange trading rules and symbol information.
    
    This provides basic symbol information similar to exp.json's symbol, exchange fields.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
    
    Returns:
        Dictionary with exchange info for the symbol
    """
    client = Spot()
    exchange_info = client.exchange_info()
    
    # Find the specific symbol
    for s in exchange_info.get('symbols', []):
        if s['symbol'] == symbol:
            return s
    
    return {}


def get_price(symbol: str) -> Dict[str, Any]:
    """Get current price for a symbol.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
    
    Returns:
        Dictionary with current price
    """
    client = Spot()
    price = client.ticker_price(symbol=symbol)
    return price


def format_binance_data(symbol: str) -> Dict[str, Any]:
    """Format Binance data into a structure similar to exp.json.
    
    Note: Many fields from exp.json are not available for cryptocurrencies.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
    
    Returns:
        Dictionary with formatted data
    """
    # Get data from Binance
    ticker_24hr = get_24hr_ticker(symbol)
    exchange_info = get_exchange_info(symbol)
    price_data = get_price(symbol)
    
    # Extract base and quote assets
    base_asset = exchange_info.get('baseAsset', '')
    quote_asset = exchange_info.get('quoteAsset', '')
    
    # Format similar to exp.json structure
    formatted_data = {
        "symbol": symbol,
        "price": float(price_data.get('price', 0)),
        "baseAsset": base_asset,
        "quoteAsset": quote_asset,
        "exchange": "Binance",
        "exchangeShortName": "BINANCE",
        "currency": quote_asset,  # Usually USDT, BUSD, etc.
        
        # 24hr statistics (similar to exp.json fields)
        "priceChange": float(ticker_24hr.get('priceChange', 0)),
        "priceChangePercent": float(ticker_24hr.get('priceChangePercent', 0)),
        "weightedAvgPrice": float(ticker_24hr.get('weightedAvgPrice', 0)),
        "prevClosePrice": float(ticker_24hr.get('prevClosePrice', 0)),
        "lastPrice": float(ticker_24hr.get('lastPrice', 0)),
        "bidPrice": float(ticker_24hr.get('bidPrice', 0)),
        "askPrice": float(ticker_24hr.get('askPrice', 0)),
        "openPrice": float(ticker_24hr.get('openPrice', 0)),
        "highPrice": float(ticker_24hr.get('highPrice', 0)),
        "lowPrice": float(ticker_24hr.get('lowPrice', 0)),
        "volume": float(ticker_24hr.get('volume', 0)),
        "quoteVolume": float(ticker_24hr.get('quoteVolume', 0)),
        "openTime": ticker_24hr.get('openTime'),
        "closeTime": ticker_24hr.get('closeTime'),
        "count": ticker_24hr.get('count', 0),
        
        # Range (similar to exp.json)
        "range": f"{ticker_24hr.get('lowPrice', 0)}-{ticker_24hr.get('highPrice', 0)}",
        
        # Trading rules from exchange_info
        "status": exchange_info.get('status', ''),
        "baseAssetPrecision": exchange_info.get('baseAssetPrecision', 0),
        "quotePrecision": exchange_info.get('quotePrecision', 0),
        "orderTypes": exchange_info.get('orderTypes', []),
        "icebergAllowed": exchange_info.get('icebergAllowed', False),
        "ocoAllowed": exchange_info.get('ocoAllowed', False),
        "isSpotTradingAllowed": exchange_info.get('isSpotTradingAllowed', False),
        "isMarginTradingAllowed": exchange_info.get('isMarginTradingAllowed', False),
        
        # Fields NOT available from Binance (cryptocurrency-specific limitations)
        "companyName": None,  # Cryptocurrencies don't have companies
        "beta": None,  # Not applicable to cryptocurrencies
        "volAvg": None,  # Can calculate from historical data
        "mktCap": None,  # Can calculate: price * circulating supply (need external API)
        "lastDiv": None,  # Cryptocurrencies don't pay dividends
        "changes": float(ticker_24hr.get('priceChangePercent', 0)),  # Similar to exp.json
        "ceo": None,  # Not applicable
        "sector": None,  # Not applicable
        "industry": None,  # Not applicable
        "country": None,  # Cryptocurrencies are global
        "fullTimeEmployees": None,  # Not applicable
        "phone": None,  # Not applicable
        "address": None,  # Not applicable
        "city": None,  # Not applicable
        "state": None,  # Not applicable
        "zip": None,  # Not applicable
        "description": None,  # Would need external API (CoinGecko, CoinMarketCap)
        "website": None,  # Would need external API
        "ipoDate": None,  # Would need external API (listing date)
        "image": None,  # Would need external API
    }
    
    return formatted_data


def main():
    """Main function to demonstrate Binance data structure."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Binance cryptocurrency metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Get metadata for BTCUSDT
  python tests/download_binance_metadata.py --symbol BTCUSDT
  
  # Get metadata for multiple symbols
  python tests/download_binance_metadata.py --symbols BTCUSDT ETHUSDT BNBUSDT
  
  # Save to JSON file
  python tests/download_binance_metadata.py --symbol BTCUSDT --output binance_metadata.json
        """
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        help='Single trading pair symbol (e.g., BTCUSDT)'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        nargs='+',
        help='Multiple trading pair symbols (e.g., BTCUSDT ETHUSDT BNBUSDT)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output JSON file path (default: print to stdout)'
    )
    
    args = parser.parse_args()
    
    if not args.symbol and not args.symbols:
        parser.error("Must provide either --symbol or --symbols")
    
    symbols = []
    if args.symbol:
        symbols.append(args.symbol.upper())
    if args.symbols:
        symbols.extend([s.upper() for s in args.symbols])
    
    print(f"\n{'='*60}")
    print(f"Binance Cryptocurrency Metadata")
    print(f"{'='*60}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"{'='*60}\n")
    
    results = {}
    
    for symbol in symbols:
        print(f"📥 Fetching data for {symbol}...")
        try:
            data = format_binance_data(symbol)
            results[symbol] = data
            print(f"✅ Successfully fetched data for {symbol}")
        except Exception as e:
            print(f"❌ Error fetching data for {symbol}: {e}")
            results[symbol] = {"error": str(e)}
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
        print(f"\n💾 Saved to: {output_path}")
    else:
        print("\n" + "="*60)
        print("Formatted Data (similar to exp.json structure):")
        print("="*60)
        print(json.dumps(results, indent=4, ensure_ascii=False))
    
    print("\n" + "="*60)
    print("Note: Binance provides trading data, not company fundamentals.")
    print("For cryptocurrency metadata (description, website, market cap, etc.),")
    print("consider using CoinGecko API or CoinMarketCap API.")
    print("="*60)


if __name__ == "__main__":
    main()


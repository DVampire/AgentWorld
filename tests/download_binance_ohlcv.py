"""Download Binance cryptocurrency OHLCV data.

This script downloads OHLCV (Open, High, Low, Close, Volume) data from Binance
for specified symbols and time intervals (1min, 5min, 1day).

Example usage:
    python tests/download_binance_ohlcv.py --symbol BTCUSDT --interval 1m --days 1
    python tests/download_binance_ohlcv.py --symbol ETHUSDT --interval 5m --days 30
    python tests/download_binance_ohlcv.py --symbol BTCUSDT --interval 1d --days 365
"""
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List
import pandas as pd

# Add project root to path
root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

try:
    from binance.spot import Spot
except ImportError:
    try:
        # Alternative import path
        from binance import Spot
    except ImportError:
        print("❌ binance-connector not installed. Install with: pip install binance-connector")
        sys.exit(1)


def download_klines(
    symbol: str,
    interval: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000
) -> List[dict]:
    """Download klines (OHLCV) data from Binance.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT')
        interval: Kline interval ('1m', '5m', '1d', etc.)
        start_time: Start time (default: None, will use end_time - limit intervals)
        end_time: End time (default: None, will use current time)
        limit: Maximum number of klines to fetch per request (max 1000)
    
    Returns:
        List of kline dictionaries
    """
    client = Spot()
    
    # Convert datetime to milliseconds timestamp
    if end_time:
        end_time_ms = int(end_time.timestamp() * 1000)
    else:
        end_time_ms = int(datetime.now().timestamp() * 1000)
    
    if start_time:
        start_time_ms = int(start_time.timestamp() * 1000)
    else:
        start_time_ms = None
    
    all_klines = []
    current_start_time = start_time_ms
    
    print(f"📥 Downloading {symbol} {interval} data...")
    
    # Use a set to track downloaded timestamps to avoid duplicates
    seen_timestamps = set()
    
    while True:
        try:
            # Fetch klines
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            # Binance returns klines in chronological order (oldest first)
            # Use startTime for pagination
            if current_start_time:
                params['startTime'] = current_start_time
            
            # Also set endTime to limit the range
            params['endTime'] = end_time_ms
            
            klines = client.klines(**params)
            
            if not klines:
                break
            
            # Filter out duplicates and add new klines
            new_klines = []
            for kline in klines:
                kline_time = kline[0]  # open_time in ms
                if kline_time not in seen_timestamps:
                    seen_timestamps.add(kline_time)
                    new_klines.append(kline)
            
            if not new_klines:
                # No new data, we're done
                break
            
            all_klines.extend(new_klines)
            
            # Get the newest kline's close time (last element in the list)
            newest_kline = new_klines[-1]
            newest_close_time = newest_kline[6]  # close_time in ms
            
            # If we've reached or passed the end_time, stop
            if newest_close_time >= end_time_ms:
                # Filter to only include klines before end_time
                all_klines = [k for k in all_klines if k[0] < end_time_ms]
                break
            
            # If we got fewer than limit, we've reached the end
            if len(klines) < limit:
                break
            
            # Set next start_time to be just after the newest kline's close time
            current_start_time = newest_close_time + 1
            
            print(f"   Downloaded {len(all_klines)} klines...", end='\r')
            
        except Exception as e:
            print(f"\n❌ Error downloading data: {e}")
            break
    
    # Final filter by time range
    if start_time_ms:
        all_klines = [k for k in all_klines if k[0] >= start_time_ms]
    if end_time_ms:
        all_klines = [k for k in all_klines if k[0] < end_time_ms]
    
    # Sort by open_time to ensure chronological order
    all_klines.sort(key=lambda x: x[0])
    
    print(f"\n✅ Downloaded {len(all_klines)} unique klines")
    return all_klines


def klines_to_dataframe(klines: List[list]) -> pd.DataFrame:
    """Convert Binance klines format to pandas DataFrame.
    
    Binance kline format:
    [
        open_time (ms),
        open (str),
        high (str),
        low (str),
        close (str),
        volume (str),
        close_time (ms),
        quote_volume (str),
        trades (int),
        taker_buy_base_volume (str),
        taker_buy_quote_volume (str),
        ignore
    ]
    
    Args:
        klines: List of kline lists from Binance API
    
    Returns:
        DataFrame with columns: timestamp, open_time, close_time, open, high, low, close, volume, quote_volume, trades
    """
    if not klines:
        return pd.DataFrame()
    
    data = []
    for kline in klines:
        data.append({
            'timestamp': datetime.fromtimestamp(kline[0] / 1000),
            'open_time': kline[0],
            'close_time': kline[6],
            'open': float(kline[1]),
            'high': float(kline[2]),
            'low': float(kline[3]),
            'close': float(kline[4]),
            'volume': float(kline[5]),
            'quote_volume': float(kline[7]),
            'trades': int(kline[8])
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    return df


def save_to_csv(df: pd.DataFrame, symbol: str, interval: str, output_dir: Path):
    """Save DataFrame to CSV file.
    
    Args:
        df: DataFrame to save
        symbol: Trading pair symbol
        interval: Kline interval
        output_dir: Output directory path
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename: SYMBOL_INTERVAL_YYYYMMDD_HHMMSS.csv
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol}_{interval}_{timestamp}.csv"
    filepath = output_dir / filename
    
    df.to_csv(filepath)
    print(f"💾 Saved to: {filepath}")
    print(f"   Rows: {len(df)}, Columns: {len(df.columns)}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Binance OHLCV data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download last 7 days of 1-minute data for BTCUSDT
  python tests/download_binance_ohlcv.py --symbol BTCUSDT --interval 1m --days 7
  
  # Download last 30 days of 5-minute data for ETHUSDT
  python tests/download_binance_ohlcv.py --symbol ETHUSDT --interval 5m --days 30
  
  # Download last 365 days of daily data for BTCUSDT
  python tests/download_binance_ohlcv.py --symbol BTCUSDT --interval 1d --days 365
  
  # Download data for specific date range
  python tests/download_binance_ohlcv.py --symbol BTCUSDT --interval 1m --start 2024-01-01 --end 2024-01-31
        """
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        required=True,
        help='Trading pair symbol (e.g., BTCUSDT, ETHUSDT)'
    )
    
    parser.add_argument(
        '--interval',
        type=str,
        required=True,
        choices=['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
        help='Kline interval (1m, 5m, 1d, etc.)'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        help='Number of days to download (from now backwards). Mutually exclusive with --start/--end'
    )
    
    parser.add_argument(
        '--start',
        type=str,
        help='Start date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS). Mutually exclusive with --days'
    )
    
    parser.add_argument(
        '--end',
        type=str,
        help='End date (YYYY-MM-DD or YYYY-MM-DD HH:MM:SS). Default: current time'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='workdir/binance_data',
        help='Output directory for CSV files (default: workdir/binance_data)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.days and (args.start or args.end):
        parser.error("--days cannot be used with --start/--end")
    
    # Calculate time range
    end_time = datetime.now()
    if args.end:
        try:
            if len(args.end) == 10:  # YYYY-MM-DD
                end_time = datetime.strptime(args.end, "%Y-%m-%d")
            else:  # YYYY-MM-DD HH:MM:SS
                end_time = datetime.strptime(args.end, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            parser.error(f"Invalid end date format: {args.end}. Use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")
    
    start_time = None
    if args.days:
        start_time = end_time - timedelta(days=args.days)
    elif args.start:
        try:
            if len(args.start) == 10:  # YYYY-MM-DD
                start_time = datetime.strptime(args.start, "%Y-%m-%d")
            else:  # YYYY-MM-DD HH:MM:SS
                start_time = datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
        except ValueError:
            parser.error(f"Invalid start date format: {args.start}. Use YYYY-MM-DD or YYYY-MM-DD HH:MM:SS")
    
    # Download data
    print(f"\n{'='*60}")
    print(f"Binance OHLCV Data Downloader")
    print(f"{'='*60}")
    print(f"Symbol: {args.symbol}")
    print(f"Interval: {args.interval}")
    if start_time:
        print(f"Start: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")
    
    klines = download_klines(
        symbol=args.symbol.upper(),
        interval=args.interval,
        start_time=start_time,
        end_time=end_time
    )
    
    if not klines:
        print("❌ No data downloaded")
        return
    
    # Convert to DataFrame
    df = klines_to_dataframe(klines)
    
    # Display summary
    print(f"\n📊 Data Summary:")
    print(f"   Period: {df.index[0]} to {df.index[-1]}")
    print(f"   Total klines: {len(df)}")
    print(f"   Open range: {df['open'].min():.2f} - {df['open'].max():.2f}")
    print(f"   Close range: {df['close'].min():.2f} - {df['close'].max():.2f}")
    print(f"   Total volume: {df['volume'].sum():.2f}")
    
    # Save to CSV
    output_dir = Path(args.output_dir)
    save_to_csv(df, args.symbol.upper(), args.interval, output_dir)
    
    print(f"\n✅ Done!")


if __name__ == "__main__":
    main()


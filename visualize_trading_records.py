#!/usr/bin/env python3
"""
Trading Records Visualization Script
Visualize trading records: price curves, profit curves, and buy/sell point annotations
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

def load_trading_data(csv_path):
    """Load trading records data"""
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def create_trading_visualization(df, save_path=None):
    """Create trading records visualization chart"""
    
    # Create figure and subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True)
    
    # 1. Price Chart
    ax1.plot(df['timestamp'], df['close'], 'b-', linewidth=2, label='Close Price', alpha=0.8)
    ax1.set_ylabel('Price ($)', fontsize=12)
    ax1.set_title('Stock Price Movement with Trading Profit', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Mark BUY points (green upward triangles)
    buy_points = df[df['action_label'] == 'BUY']
    if not buy_points.empty:
        ax1.scatter(buy_points['timestamp'], buy_points['close'], 
                   c='green', marker='^', s=100, label='BUY', zorder=5, alpha=0.8)
    
    # Mark SELL points (red downward triangles)
    sell_points = df[df['action_label'] == 'SELL']
    if not sell_points.empty:
        ax1.scatter(sell_points['timestamp'], sell_points['close'], 
                   c='red', marker='v', s=100, label='SELL', zorder=5, alpha=0.8)
    
    # 2. Profit Chart
    ax2.plot(df['timestamp'], df['total_profit'], 'purple', linewidth=2, label='Cumulative Profit', alpha=0.8)
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Calculate and add Buy & Hold benchmark curve
    # Buy & Hold: Buy at first day, hold until last day - shows daily return progression
    start_price = df['close'].iloc[0]
    buy_hold_returns = ((df['close'] - start_price) / start_price) * 100
    final_buy_hold_return = buy_hold_returns.iloc[-1]
    
    ax2.plot(df['timestamp'], buy_hold_returns, 'orange', linestyle='--', alpha=0.7, 
             label=f'Buy & Hold: {final_buy_hold_return:.2f}%')
    
    ax2.set_ylabel('Cumulative Profit (%)', fontsize=12)
    ax2.set_xlabel('Time', fontsize=12)
    ax2.set_title('Trading Profit Trend', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Mark trading points on profit chart as well
    if not buy_points.empty:
        ax2.scatter(buy_points['timestamp'], buy_points['total_profit'], 
                   c='green', marker='^', s=100, label='BUY', zorder=5, alpha=0.8)
    
    if not sell_points.empty:
        ax2.scatter(sell_points['timestamp'], sell_points['total_profit'], 
                   c='red', marker='v', s=100, label='SELL', zorder=5, alpha=0.8)
    
    # Format x-axis time display
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # Adjust layout
    plt.tight_layout()
    
    # Add statistics
    final_profit = df['total_profit'].iloc[-1]
    max_profit = df['total_profit'].max()
    min_profit = df['total_profit'].min()
    
    # Add statistics text to the chart (top right corner)
    stats_text = f'Final Profit: {final_profit:.2f}%\nMax Profit: {max_profit:.2f}%\nMin Profit: {min_profit:.2f}%\nBuy & Hold: {final_buy_hold_return:.2f}%'
    ax2.text(0.98, 0.98, stats_text, transform=ax2.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Save the chart
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Chart saved to: {save_path}")
    
    return fig

def print_trading_summary(df):
    """Print trading summary information"""
    print("\n" + "="*50)
    print("Trading Records Summary")
    print("="*50)
    
    # Basic statistics
    total_trades = len(df[df['action_label'].isin(['BUY', 'SELL'])])
    buy_count = len(df[df['action_label'] == 'BUY'])
    sell_count = len(df[df['action_label'] == 'SELL'])
    hold_count = len(df[df['action_label'] == 'HOLD'])
    
    print(f"Total trades: {total_trades}")
    print(f"BUY count: {buy_count}")
    print(f"SELL count: {sell_count}")
    print(f"HOLD count: {hold_count}")
    
    # Profit statistics
    final_profit = df['total_profit'].iloc[-1]
    max_profit = df['total_profit'].max()
    min_profit = df['total_profit'].min()
    
    print(f"\nProfit Statistics:")
    print(f"Final profit: {final_profit:.2f}%")
    print(f"Max profit: {max_profit:.2f}%")
    print(f"Min profit: {min_profit:.2f}%")
    
    # Price statistics
    start_price = df['close'].iloc[0]
    end_price = df['close'].iloc[-1]
    price_change = end_price - start_price
    price_change_pct = (price_change / start_price) * 100
    
    print(f"\nPrice Statistics:")
    print(f"Start price: ${start_price:.2f}")
    print(f"End price: ${end_price:.2f}")
    print(f"Price change: ${price_change:.2f} ({price_change_pct:.2f}%)")
    print(f"Buy & Hold return: {price_change_pct:.2f}%")

def main():
    """Main function"""
    # Data file path
    csv_path = "workdir/interday_trading/trading_records.csv"
    
    try:
        # Load data
        print("Loading trading data...")
        df = load_trading_data(csv_path)
        print(f"Successfully loaded {len(df)} trading records")
        
        # Create visualization
        print("Generating visualization chart...")
        fig = create_trading_visualization(df, save_path="trading_visualization.png")
        
        # Print summary information
        print_trading_summary(df)
        
        print("\nVisualization completed!")
        
    except FileNotFoundError:
        print(f"Error: File not found {csv_path}")
        print("Please ensure the file path is correct")
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()

# Factor Strategy Module

This module contains multiple factor strategies for trading. Each strategy is implemented in its own file.

## Available Strategies

### Momentum and Mean Reversion Hybrid Strategy

A minute-level factor strategy that combines momentum and mean reversion factors for trading.

**File**: `momentum_mean_reversion_strategy.py`  
**Class**: `MomentumMeanReversionStrategy`

## Features

### 1. Momentum Factor (动量因子)
- Uses very short window returns (3, 5, 10 minutes) optimized for minute-level trading
- Formula: `momentum = (close_t - close_{t-n}) / close_{t-n}`
- Calculates momentum for multiple short windows (3, 5, 10 minutes) and averages them
- Short windows (3-10 min) capture rapid price movements typical in minute-level trading
- Signal: BUY if momentum > threshold (0.08%), SELL if momentum < -threshold

### 2. Mean Reversion Factor (均值回归因子)
- Uses medium-term SMA (15 minutes) for stable baseline
- Formula: `deviation = (close_t - sma_15m) / sma_15m`
- 15-minute SMA provides stable reference point for minute-level price movements
- Signal: BUY if price is below SMA by >0.15% (oversold), SELL if price is above SMA by >0.15% (overbought)

### 3. Combined Signal Logic
- BUY: Both momentum and mean reversion signals agree on BUY
- SELL: Both momentum and mean reversion signals agree on SELL
- HOLD: Signals conflict or both are neutral

### 4. Risk Management
- **Fixed Stop Loss/Take Profit**: Tight percentages optimized for minute trading
  - Stop Loss: 0.25% (tight to limit losses in fast-moving minute markets)
  - Take Profit: 0.4% (quick exits to capture profits before reversals)
- **ATR-based Stop Loss**: Dynamic stop loss using ATR multiplier (2.0x ATR, adapts to volatility)

## Usage

### Basic Usage

```python
from src.tools.factor_strategy import MomentumMeanReversionStrategy
import pandas as pd

# Initialize strategy with optimized defaults for minute-level trading
strategy = MomentumMeanReversionStrategy()

# Prepare data (must have 'close', 'high', 'low' columns)
df = pd.DataFrame({
    'close': [...],
    'high': [...],
    'low': [...],
    'open': [...],
    'volume': [...]
})

# Run analysis - returns natural language description for LLM
result = strategy.analyze(df)
print(result)
```

## Parameters (Optimized for Minute-Level Trading)

- `momentum_windows` (List[int]): Time windows for momentum calculation in minutes. 
  - **Default: [3, 5, 10]** - Short windows (3-10 min) capture rapid price movements in minute trading
  
- `sma_window` (int): Window for SMA calculation in minutes.
  - **Default: 15** - Medium window provides stable baseline for mean reversion
  
- `deviation_threshold` (float): Threshold for mean reversion signal.
  - **Default: 0.0015 (0.15%)** - Adjusted for minute-level volatility
  
- `momentum_threshold` (float): Threshold for momentum signal.
  - **Default: 0.0008 (0.08%)** - Lower threshold for faster signal generation
  
- `use_atr_stop` (bool): Whether to use ATR-based stop loss. **Default: True**
  - Recommended for minute trading as it adapts to volatility
  
- `atr_multiplier` (float): Multiplier for ATR-based stop loss. **Default: 2.0**
  - 2x ATR provides reasonable stop distance for minute markets
  
- `fixed_stop_loss_pct` (Optional[float]): Fixed stop loss percentage.
  - **Default: 0.0025 (0.25%)** - Tight stop loss for minute trading
  
- `fixed_take_profit_pct` (Optional[float]): Fixed take profit percentage.
  - **Default: 0.004 (0.4%)** - Quick profit taking before reversals

## Example

See `tests/test_momentum_mean_reversion_strategy.py` for a complete example with real-time API data.

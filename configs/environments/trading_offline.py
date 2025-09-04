symbol = "AAPL"
if_norm = False
if_use_future = False
if_use_temporal = True
if_norm_temporal = False
history_timestamps = 32
future_timestamps = 0
start_timestamp = "2015-05-01"
split_timestamp = "2023-05-01"
end_timestamp = "2025-05-01"
level = "1day"
initial_amount = float(1e5)
transaction_cost_pct = float(1e-4)
gamma = 0.99
record_max_len = 32
valid_action_max_len = 8
single_text_max_tokens = 1024
single_text_min_tokens = 256
daily_sample_texts = 2

dataset = dict(
    type="SingleAssetDataset",
    symbol=symbol,
    data_path="datasets/exp",
    enabled_data_configs = [
        {
            "asset_name": "exp",
            "source": "fmp",
            "data_type": "price",
            "level": "1day",
        },
        {
            "asset_name": "exp",
            "source": "fmp",
            "data_type": "feature",
            "level": "1day",
        },
        {
            "asset_name": "exp",
            "source": "fmp",
            "data_type": "news",
            "level": "1day",
        },
        {
            "asset_name": "exp",
            "source": "alpaca",
            "data_type": "news",
            "level": "1day",
        }
    ],
    if_norm=if_norm,
    if_use_future=if_use_future,
    if_use_temporal=if_use_temporal,
    if_norm_temporal=if_norm_temporal,
    scaler_cfg = dict(
        type="WindowedScaler"
    ),
    history_timestamps = history_timestamps,
    future_timestamps = future_timestamps,
    start_timestamp=start_timestamp,
    end_timestamp=end_timestamp,
    level=level
)

environments_rules = """<environment_trading_offline>

<state>
The environment state includes:
1. Name: Asset name, Symbol: Asset symbol
2. Price: Price information of the asset
3. News: News information of the asset
4. Record: Trading record of the asset
5. History Valid Action: Valid action of the asset
6. Current State: Current price, cash, and position.

Trading record fields:
1. `timestamp`: the timestamp of the record
2. `close`: Close price
3. `high`: High price
4. `low`: Low price
5. `open`: Open price
6. `volume`: Volume of the asset traded
7. `price`: Current price (adj_close price)
8. `cash`: Current cash
9. `position`: Current position
10. `pre_value`: Previous total value, `value = cash + position * price`
11. `action`: Action taken, `BUY`, `SELL`, or `HOLD`
12. `post_value`: Current total value
13. `ret`: Return, `ret = (post_value - pre_value) / pre_value`
</state>

<vision>
No vision available.
</vision>

<interaction>
Available trading actions:
- BUY: Buy shares with all available cash.
- SELL: Sell existing position shares.  
- HOLD: Maintain current position.
</interaction>

</environment_trading_offline>"""

environment = dict(
    type="TradingOfflineEnvironment",
    mode="test",
    dataset=None,
    initial_amount=initial_amount,
    transaction_cost_pct=transaction_cost_pct,
    start_timestamp=split_timestamp,
    end_timestamp=end_timestamp,
    history_timestamps=history_timestamps,
    future_timestamps=future_timestamps,
    gamma=gamma,
    record_max_len=record_max_len,
    valid_action_max_len=valid_action_max_len,
    single_text_max_tokens=single_text_max_tokens,
    single_text_min_tokens=single_text_min_tokens,
    daily_sample_texts=daily_sample_texts,
)

controller = dict(
    type="TradingOfflineController",
    dataset=dataset,
    environment=environment,
    environments_rules=environments_rules,
)
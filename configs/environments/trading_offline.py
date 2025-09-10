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

environment = dict(
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
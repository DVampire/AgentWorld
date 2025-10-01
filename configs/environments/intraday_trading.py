symbol = "AAPL"
if_norm = False
if_use_future = False
if_use_temporal = True
if_norm_temporal = False
history_timestamps =  1
future_timestamps = 1
start_timestamp = "2015-05-01"
split_timestamp = "2023-05-01"
end_timestamp = "2025-05-01"
level = "1min"
initial_amount = float(1e5)
transaction_cost_pct = float(1e-4)
gamma = 0.99
valid_review_actions = 5
valid_review_trends = 60

dataset = dict(
    type="SingleAssetDataset",
    symbol=symbol,
    data_path="datasets/exp",
    enabled_data_configs = [
        {
            "asset_name": "exp",
            "source": "fmp",
            "data_type": "price",
            "level": "1min",
        },
        {
            "asset_name": "exp",
            "source": "fmp",
            "data_type": "feature",
            "level": "1min",
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
    base_dir=None,
    mode="test",
    dataset=None,
    dataset_cfg=dataset,
    initial_amount=initial_amount,
    transaction_cost_pct=transaction_cost_pct,
    start_timestamp=split_timestamp,
    end_timestamp=end_timestamp,
    history_timestamps=history_timestamps,
    future_timestamps=future_timestamps,
    gamma=gamma,
    valid_review_actions=valid_review_actions,
    valid_review_trends=valid_review_trends
)
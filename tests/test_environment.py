from math import e
import os
import sys
import numpy as np
from typing import Any
from dotenv import load_dotenv
load_dotenv(verbose=True)

from pathlib import Path

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.config import config
from src.logger import logger
from src.registry import DATASETS, ENVIRONMENTS
from src.utils import TradingRecords

def test_trading_offline_environment():
    symbol = "AAPL"
    history_timestamps = 5
    future_timestamps = 1
    start_timestamp = "2015-05-01"
    split_timestamp = "2023-05-01"
    end_timestamp = "2025-05-01"

    dataset = dict(
        type="SingleAssetDataset",
        symbol=symbol,
        data_path="datasets/exp",
        enabled_data_configs=[
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
        if_norm=False,
        if_use_future=False,
        if_use_temporal=True,
        if_norm_temporal=False,
        scaler_cfg=dict(
            type="WindowedScaler"
        ),
        history_timestamps=history_timestamps,
        future_timestamps=future_timestamps,
        start_timestamp=start_timestamp,
        end_timestamp=end_timestamp,
    )

    env_cfg: dict[str, Any] = dict(
        type='TradingOfflineEnvironment',
        mode = "test",
        dataset=None,
        initial_amount=float(1e5),
        transaction_cost_pct=float(1e-4),
        history_timestamps=history_timestamps,
        step_timestamps=1,
        future_timestamps=future_timestamps,
        start_timestamp=split_timestamp,
        end_timestamp=end_timestamp,
        gamma=0.99,
        record_max_len=32,
        valid_action_max_len=8,
        single_text_max_tokens=1024,
        single_text_min_tokens=256,
        daily_sample_texts=2,
    )

    dataset = DATASETS.build(dataset)

    env_cfg.update(
        dict(
            dataset=dataset,
        )
    )

    record = TradingRecords()

    environment = ENVIRONMENTS.build(env_cfg)

    state, info = environment.reset()

    record.add(
        dict(
            timestamp=info["timestamp"],
            price=info["price"],
            cash=info["cash"],
            position=info["position"],
            value=info["value"],
        ),
    )

    for step in range(500):
        action = np.random.choice([0, 1, 2])
        next_state, reward, done, truncted, info = environment.step(action)

        if step == 10:
            print(next_state['prompt'])
            exit()

        record.add(
            dict(
                action=info["action"],
                action_label=info["action_label"],
                ret=info["ret"],
                total_profit=info["total_profit"],
                timestamp=info["timestamp"],  # next timestamp
                price=info["price"],  # next price
                cash=info["cash"],  # next cash
                position=info["position"],  # next position
                value=info["value"],  # next value
            ),
        )

        if "final_info" in info:
            break

    record.add(
        dict(
            action=info["action"],
            action_label=info["action_label"],
            ret=info["ret"],
            total_profit=info["total_profit"],
        )
    )

    print(record.to_dataframe())
    
    
def test_file_system_environment():
    environment = dict(
        type="FileSystemEnvironment",
        base_dir="workdir/file_system",
        create_default_files=True,
        max_file_size=1024 * 1024,  # 1MB
    )
    
    environment = ENVIRONMENTS.build(environment)
    state, info = environment.reset()
    print(state)
    print(info)
    
    state, reward, done, truncated, info = environment.step(
        operation="write",
        filename="test.txt",
        content="Hello, world!",
    )
    print(state)
    print(info)
    
if __name__ == "__main__":
    # test_trading_offline_environment()
    test_file_system_environment()
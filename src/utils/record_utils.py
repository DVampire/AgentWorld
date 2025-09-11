import numpy as np
from typing import Dict, Any, Optional
from dataclasses import dataclass
import pandas as pd

@dataclass
class Record():
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    price: float
    cash: float
    position: int
    pre_value: Optional[float]
    action: Optional[str]
    post_value: Optional[float]
    ret: Optional[float]

class TradingRecords():
    def __init__(self):
        self.data = dict(
            # state (action-before)
            timestamp = [],
            price = [],
            position = [],
            cash = [],
            value = [],

            # action (action-after)
            action = [],
            action_label = [],
            ret = [],
            total_profit=[],
        )

    def add(self, info: Dict[str, Any]):
        """
        Add a new record to the trading records.
        :param info: A dictionary containing the trading information.
        """
        for key, value in info.items():
            self.data[key].append(value)

    def to_dataframe(self):
        """
        Convert the trading records to a pandas DataFrame.
        :return: A pandas DataFrame containing the trading records.
        """
        df = pd.DataFrame(self.data, index=range(len(self.data['timestamp'])))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        df.set_index('timestamp', inplace=True)
        return df

class PortfolioRecords():
    def __init__(self):
        self.data = dict(
            # state (action-before)
            timestamp = [],
            price = [],
            position = [],
            cash = [],
            value = [],

            # action (action-after)
            action = [],
            ret = [],
            total_profit = [],
        )

    def add(self, info: Dict[str, Any]):
        """
        Add a new record to the portfolio records.
        :param info: A dictionary containing the portfolio information.
        """
        for key, value in info.items():
            self.data[key].append(value)

    def to_dataframe(self):
        """
        Convert the portfolio records to a pandas DataFrame.
        :return: A pandas DataFrame containing the portfolio records.
        """
        df = pd.DataFrame(self.data, index=range(len(self.data['timestamp'])))
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['timestamp'] = df['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d %H:%M:%S'))
        df.set_index('timestamp', inplace=True)
        return df
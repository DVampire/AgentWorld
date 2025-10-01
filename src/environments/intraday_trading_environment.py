import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
from pandas import DataFrame
from typing import Any, Dict, Type
import random
import os
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ConfigDict

from src.utils import TradingRecords
from src.utils import get_start_end_timestamp
from src.environments.protocol import ecp
from src.logger import logger
from src.utils import dedent
from src.environments.protocol.environment import BaseEnvironment
from src.supports.metric import ARR, SR, MDD, SOR, CR, VOL
from src.supports.registry import DATASETS
from src.utils import get_token_count

_STATE_RULES = """The state of the intraday trading environment will be provided with the following information:
1. Info: Information of the intraday trading environment.
    - Timestamp: Current timestamp (minute-level).
    - Prices: Current prices. (close, high, low, open, volume)
    - Current cash: Current cash.
    - Current position: Current position.
    - Current profit: Current profit.
2. News: Current news. (title, summary)
    - If no news available, it will be None.
    - The news will be a list of news with the following information: timestamp, title, summary.
3. History Valid Actions: History of intraday trading valid actions. 
    - If no valid actions available, it will be None.
    - The valid actions will be a table with the following columns:
        - `timestamp` is the timestamp of the action.
        - `price` is the close price of the action.
        - `cash` is the cash of the action.
        - `position` is the position of the action.
        - `value` is the value of the action. value = cash + position * price
        - `action` is the action label of the action. (e.g., BUY, SELL, HOLD)
        - `profit` is the result total profit of the action. profit = (value - initial_amount) / initial_amount * 100
4. History Price Trends: History of intraday price trends (minute-level).
    - If no price trends available, it will be None.
    - The price trend will be a table with the following columns:
        - `timestamp` is the timestamp.
        - `close` is the close price.
        - `high` is the high price.
        - `low` is the low price.
        - `open` is the open price.
        - `volume` is the volume.
"""

_INTERACTION_RULES = """Interaction guidelines for intraday trading:
1. If you DO NOT have enough current cash, you CAN NOT execute the `BUY` action.
2. If you DO NOT have enough current position, you CAN NOT execute the `SELL` action.
3. Intraday trading focuses on minute-level price movements and short-term opportunities.
4. All positions must be closed by the end of trading day (automatic liquidation if not closed).
"""

@ecp.environment()
class IntradayTradingEnvironment(BaseEnvironment):
    """Intraday Trading Environment that provides minute-level trading operations."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: str = Field(default="intraday_trading", description="The name of the intraday trading environment.")
    type: str = Field(default="Intraday Trading", description="The type of the intraday trading environment.")
    description: str = Field(default="Intraday trading environment for minute-level trading", description="The description of the intraday trading environment.")
    args_schema: Type[BaseModel] = Field(default=None, description="The args schema of the intraday trading environment.")
    metadata: Dict[str, Any] = Field(default={
        "has_vision": False,
        "additional_rules": {
            "state": _STATE_RULES,
            "interaction": _INTERACTION_RULES,
        }
    }, description="The metadata of the intraday trading environment.")
    
    def __init__(
        self,
        base_dir: str = None,
        mode: str = "train",
        dataset: Any = None,
        dataset_cfg: Dict[str, Any] = None,
        initial_amount: float = 1e3,
        transaction_cost_pct: float = 1e-3,
        history_timestamps: int = 1,  # 1 minute history
        step_timestamps: int = 1,
        future_timestamps: int = 1,
        start_timestamp='2023-01-01',
        end_timestamp='2023-12-31',
        gamma: float = 0.99,
        valid_review_actions: int = 10,  # More actions for intraday
        valid_review_trends: int = 60,   # 60 minutes trend window
        auto_close_eod: bool = True,     # Auto close positions at end of day
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.base_dir = base_dir
        self.mode = mode
        self.dataset = dataset
        self.dataset_cfg = dataset_cfg
        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.history_timestamps = history_timestamps
        self.step_timestamps = step_timestamps
        self.future_timestamps = future_timestamps
        self.gamma = gamma
        self.valid_review_actions = valid_review_actions
        self.valid_review_trends = valid_review_trends
        self.auto_close_eod = auto_close_eod
        
    async def initialize(self) -> None:
        """Initialize the intraday trading environment."""
        
        # Initialize dataset
        if self.dataset is None:
            self.dataset = DATASETS.build(self.dataset_cfg)
        
        self.symbol = self.dataset.symbol
        self.level = self.dataset.level
        self.level_format = self.dataset.level_format

        asset_info = self.dataset.asset_info

        self.asset_info = dict(
            asset_symbol=asset_info['symbol'],
            asset_name=asset_info['companyName'],
            asset_exchange=asset_info['exchange'],
            asset_sector=asset_info['sector'],
            asset_industry=asset_info['industry'],
            asset_description=asset_info['description'],
        )
        
        symbol_info = dict(
            symbol=self.asset_info['asset_symbol'],
            exchange=self.asset_info['asset_exchange'],
        )
        self.metrics_functions = dict(
            ARR=ARR(level=self.level.value, symbol_info=symbol_info),
            SR=SR(level=self.level.value, symbol_info=symbol_info),
            MDD=MDD(level=self.level.value, symbol_info=symbol_info),
            SOR=SOR(level=self.level.value, symbol_info=symbol_info),
            CR=CR(level=self.level.value, symbol_info=symbol_info),
            VOL=VOL(level=self.level.value, symbol_info=symbol_info),
        )
        self.metrics = dict()

        self.start_timestamp, self.end_timestamp = get_start_end_timestamp(
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            level=self.level
        )

        self.res_info = self._init_features()
        self.timestamp_info = self.res_info['timestamp_info']

        self.features_df = self.res_info['features_df']
        self.original_prices_df = self.res_info['original_prices_df']
        self.news_df = self.res_info['news_df']

        self.action_labels = ['SELL', 'HOLD', 'BUY']  # 0, 1, 2
        self.action_dim = len(self.action_labels)

        self.trading_records = TradingRecords()

        state, info = self.reset()
        
        logger.info(f"| 📈 Intraday Trading Environment initialized at: {self.base_dir}")

    def _init_features(self):

        timestamp_info = {}
        asset_meta_info = self.dataset.asset_meta_info['items']
        for key, value in asset_meta_info.items():
            start_timestamp = value["history_info"]["start_timestamp"]
            end_timestamp = value["history_info"]["end_timestamp"]

            if (end_timestamp >= self.start_timestamp
                    and end_timestamp <= self.end_timestamp):
                timestamp_info[key] = {
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                }

        self.timestamp_min_index = min(timestamp_info.keys())
        self.timestamp_max_index = max(timestamp_info.keys())
        self.timestamp_min = timestamp_info[self.timestamp_min_index]["start_timestamp"]
        self.timestamp_max = timestamp_info[self.timestamp_max_index]["end_timestamp"]

        self.num_timestamps = self.timestamp_max_index - self.timestamp_min_index + 1
        assert self.num_timestamps == len(
            timestamp_info), f"num_timestamps {self.num_timestamps} != len(data_info) {len(timestamp_info)}"

        features_df = self.dataset.asset_data["features"]
        prices_df = self.dataset.asset_data["prices"]
        times_df = self.dataset.asset_data["times"]
        original_prices_df = self.dataset.asset_data["original_prices"]
        labels_df = self.dataset.asset_data["labels"]
        news_df = self.dataset.asset_data["news"]

        res_info = dict(
            timestamp_info=timestamp_info,
            features_df=features_df,
            prices_df=prices_df,
            original_prices_df=original_prices_df,
            times_df=times_df,
            labels_df=labels_df,
            news_df=news_df,
        )

        return res_info

    def _get_dataitem(self,
                      df: DataFrame,
                      start_timestamp: str,
                      end_timestamp: str):
        df = deepcopy(df)
        df = df[(start_timestamp <= df.index) & (df.index <= end_timestamp)]
        return df

    def _init_timestamp_index(self):
        if self.mode == "train":
            timestamp_index = random.randint(self.timestamp_min_index,
                                             self.timestamp_min_index + 3 * (self.num_timestamps // 4))
        else:
            timestamp_index = self.timestamp_min_index
        return timestamp_index

    def get_timestamp_string(self, timestamp_index: int):
        end_timestamp = self.timestamp_info[timestamp_index]["end_timestamp"]
        end_timestamp_string = end_timestamp.strftime(self.level_format.value)
        return end_timestamp_string

    def get_value(self,
                  cash: float,
                  postition: int,
                  price: float):
        value = cash + postition * price
        return value

    def get_price(self, timestamp_index: int):

        timestamp_info = self.timestamp_info[timestamp_index]
        start_timestamp = timestamp_info["start_timestamp"]
        end_timestamp = timestamp_info["end_timestamp"]
        original_prices_df = self._get_dataitem(self.original_prices_df,
                                       start_timestamp,
                                       end_timestamp)

        prices = original_prices_df.iloc[-1].to_dict()

        # close, high, low, open, volume
        close, high, low, open, volume = (prices["close"],
                                          prices["high"],
                                          prices["low"],
                                          prices["open"],
                                          prices["volume"])
        price = close
        
        res = {
            "close": close,
            "high": high,
            "low": low,
            "open": open,
            "volume": volume,
            "price": price,
        }

        return res

    def get_state_data(self, timestamp_index: int):
        timestamp_info = self.timestamp_info[timestamp_index]
        timestamp_string = self.get_timestamp_string(timestamp_index)

        start_timestamp = timestamp_info['start_timestamp']
        end_timestamp = timestamp_info['end_timestamp']

        prices = self._get_dataitem(self.original_prices_df, start_timestamp, end_timestamp)
        prices = prices if not prices.empty else None
        news = self._get_dataitem(self.news_df, start_timestamp, end_timestamp)
        news = news if not news.empty else None
        
        # Get valid records
        records_df = self.trading_records.to_dataframe()
        records_df.index = pd.to_datetime(records_df.index)
        current_date = pd.to_datetime(end_timestamp).date()
        records_df = records_df[records_df.index.date == current_date]
        records_df.index = records_df.index.strftime('%Y-%m-%d %H:%M:%S')
        if records_df.empty:
            review_trends = review_actions = None
        else:
            review_trends = records_df.tail(self.valid_review_trends)
            review_trends = review_trends if not review_trends.empty else None
            review_actions = records_df[records_df["action_label"] != "HOLD"].tail(self.valid_review_actions)
            review_actions = review_actions if not review_actions.empty else None
        
        state = {
            "timestamp": timestamp_string,
            "prices": prices,
            "news": news,
            "review_actions": review_actions,
            "review_trends": review_trends,
        }

        return state

    def eval_buy_position(self,
                          cash: float,
                          price: float):
        # evaluate buy position
        # price * position + price * position * transaction_cost_pct <= cash
        # position <= cash / price / (1 + transaction_cost_pct)
        return int(np.floor(cash / price / (1 + self.transaction_cost_pct)))

    def eval_sell_position(self,
                           position: int):
        # evaluate sell position
        return int(position)

    def buy(self,
            cash: float,
            position: int,
            price: float,
            amount: int):

        # evaluate buy position
        eval_buy_postion = self.eval_buy_position(price=price, cash=cash)

        # predict buy position
        buy_position = int(np.floor((1.0 * np.abs(amount)) * eval_buy_postion))

        cash = cash - (buy_position * price * (1 + self.transaction_cost_pct))
        position = position + buy_position
        value = self.get_value(cash=cash, postition=position, price=price)

        if buy_position == 0:
            action_label = "HOLD"
            action = self.action_labels.index("HOLD")
        else:
            action_label = "BUY"
            action = self.action_labels.index("BUY")

        res_info = {
            "cash": cash,
            "position": position,
            "value": value,
            "action": action,
            "action_label": action_label
        }

        return res_info

    def sell(self,
             cash: float,
             position: int,
             price: float,
             amount: int):

        # evaluate sell position
        eval_sell_postion = self.eval_sell_position(position=position)

        # predict sell position
        sell_position = int(np.floor((1.0 * np.abs(amount)) * eval_sell_postion))

        cash = cash + (sell_position * price * (1 - self.transaction_cost_pct))
        position = position - sell_position
        value = self.get_value(cash=cash, postition=position, price=price)

        if sell_position == 0:
            action_label = "HOLD"
            action = self.action_labels.index("HOLD")
        else:
            action_label = "SELL"
            action = self.action_labels.index("SELL")

        res_info = {
            "cash": cash,
            "position": position,
            "value": value,
            "action": action,
            "action_label": action_label
        }

        return res_info

    def hold(self,
             cash: float,
             position: int,
             price: float,
             amount: int):

        value = self.get_value(cash=cash, postition=position, price=price)

        action_label = "HOLD"
        action = self.action_labels.index("HOLD")

        res_info = {
            "cash": cash,
            "position": position,
            "value": value,
            "action": action,
            "action_label": action_label
        }

        return res_info
        
    async def cleanup(self) -> None:
        """Cleanup the intraday trading environment."""
        logger.info("| 🧹 Intraday Trading Environment cleanup completed")

    def reset(self, **kwargs):
        self.timestamp_index = self._init_timestamp_index()
        self.timestamp_string = self.get_timestamp_string(timestamp_index=self.timestamp_index)
        self.price_info = self.get_price(timestamp_index=self.timestamp_index)
        self.close = self.price_info['close']
        self.high = self.price_info['high']
        self.low = self.price_info['low']
        self.open = self.price_info['open']
        self.volume = self.price_info['volume']
        self.price = self.price_info['price']

        self.ret = 0.0
        self.cash = self.initial_amount
        self.position = 0
        self.discount = 1.0
        self.pre_value = self.value = self.initial_amount
        self.value = self.initial_amount
        self.total_return = 0.0
        self.total_profit = 0.0
        self.action = 1
        self.action_label = 'HOLD'
        self.done = False

        info = dict(
            timestamp=self.timestamp_string,
            ret=self.ret,
            close=self.close,
            high=self.high,
            low=self.low,
            open=self.open,
            volume=self.volume,
            price=self.price,
            cash=self.cash,
            position=self.position,
            discount=self.discount,
            pre_value=self.pre_value,
            value=self.value,
            total_profit=self.total_profit,
            total_return=self.total_return,
            action=self.action,
            action_label=self.action_label,
            done=self.done,
        )
        
        self.trading_records.add(
            dict(
                timestamp=info["timestamp"],
                close=info["close"],
                high=info["high"],
                low=info["low"],
                open=info["open"],
                volume=info["volume"],
                price=info["price"],
                cash=info["cash"],
                position=info["position"],
                value=info["value"],
            )
        )

        # after init record, get the state
        state = self.get_state_data(timestamp_index=self.timestamp_index)
        
        self.state = state
        self.info = info
        
        return state, info

    def _extract_action(self, action: str):
        for index, label in enumerate(self.action_labels):
            if label == action:
                return index
        return 1 # HOLD

    def _step(self, action: str):

        action = self._extract_action(action)

        action = action - 1  # modify the action to -1, 0, 1

        if action > 0:
            res_info = self.buy(cash=self.cash,
                                position=self.position,
                                price=self.price,
                                amount=action)
        elif action < 0:
            res_info = self.sell(cash=self.cash,
                                 position=self.position,
                                 price=self.price,
                                 amount=action)
        else:
            res_info = self.hold(cash=self.cash,
                                 position=self.position,
                                 price=self.price,
                                 amount=action)

        self.cash = res_info['cash']
        self.position = res_info['position']
        self.value = res_info['value']
        self.action = res_info['action']
        self.action_label = res_info['action_label']

        ret = (self.value - self.pre_value) / (self.pre_value + 1e-6)

        self.ret = ret
        self.discount *= 0.99
        self.total_return += self.discount * ret
        self.total_profit = (self.value - self.initial_amount) / self.initial_amount * 100
        reward = ret

        # next timestamp
        self.timestamp_index = self.timestamp_index + 1
        if self.timestamp_index < self.timestamp_max_index:
            self.done = False
            self.truncted = False
        else:
            self.done = True
            self.truncted = True

        self.timestamp_string = self.get_timestamp_string(timestamp_index=self.timestamp_index)
        self.price_info = self.get_price(timestamp_index=self.timestamp_index)
        self.close = self.price_info['close']
        self.high = self.price_info['high']
        self.low = self.price_info['low']
        self.open = self.price_info['open']
        self.volume = self.price_info['volume']
        self.price = self.price_info['price']

        info = dict(
            timestamp=self.timestamp_string,
            ret=self.ret,
            close=self.close,
            high=self.high,
            low=self.low,
            open=self.open,
            volume=self.volume,
            price=self.price,
            cash=self.cash,
            position=self.position,
            discount=self.discount,
            pre_value=self.pre_value,
            value=self.value,
            total_profit=self.total_profit,
            total_return=self.total_return,
            action=self.action,
            action_label=self.action_label,
            done=self.done,
        )
        
        # add the trading record
        self.trading_records.add(
            dict(
                action=info["action"],
                action_label=info["action_label"],
                ret=info["ret"],
                total_profit=info["total_profit"],
                timestamp=info["timestamp"], # next timestamp
                close=info["close"], # next close
                high=info["high"], # next high
                low=info["low"], # next low
                open=info["open"], # next open
                volume=info["volume"], # next volume
                price=info["price"],  # next price
                cash=info["cash"],  # next cash
                position=info["position"],  # next position
                value=info["value"],  # next value
            ),
        )
        
        if self.done:
            self.trading_records.add(
                dict(
                    action=1,
                    action_label="HOLD",
                    ret=0.0,
                    total_profit=info['total_profit'],
                )
            )
            
        # after update record, get the state
        state = self.get_state_data(timestamp_index=self.timestamp_index)
        
        self.state = state
        self.info = info
        
        # update the pre_value
        self.pre_value = self.value

        return state, reward, self.done, self.truncted, info
    
    @ecp.action(name = "step",
                type = "Intraday Trading",
                description = "Step the intraday trading environment.")
    async def step(self, action: str) -> str:
        """Step the intraday trading environment.
        
        Args:
            action (str): The action to take. Should be `BUY`, `SELL` or `HOLD`.

        Returns:
            str: The state of the intraday trading environment.
        """
        state, reward, done, truncted, info = self._step(action)
        
        if not done:
            res = dedent(f"""
                <info>
                Name: {self.asset_info['asset_name']}
                Symbol: {self.asset_info['asset_symbol']}
                Start timestamp: {self.start_timestamp}
                End timestamp: {self.end_timestamp}
                Current timestamp: {self.state['timestamp']}
                Environment status: running (intraday)
                </info>
                <action>
                Expected executed action of assistant: {action}
                Actual executed action because of cash or position constraint: {info['action_label']}
                Action result: The action return is {info['ret'] * 100:.2f}%.
                </action>
                <result>
                Total profit: {info['total_profit']:.2f}%
                </result>
                """)
            return res
        
        else:
            rets = np.array(self.trading_records.data['ret'])
            for metric_name, metric in self.metrics_functions.items():
                self.metrics[metric_name] = metric(rets)
                
            metrics_string = f"**Metric | Value**\n"
            for metric_name, metric_value in self.metrics.items():
                metrics_string += f"{metric_name} | {metric_value:.4f}\n"
            
            df = self.trading_records.to_dataframe()
            df.to_csv(os.path.join(self.base_dir, "intraday_trading_records.csv"), index=True)
            logger.info(f"| 📊 Intraday trading records saved to {os.path.join(self.base_dir, 'intraday_trading_records.csv')}")
            
            res = dedent(f"""
                <info>
                Name: {self.asset_info['asset_name']}
                Symbol: {self.asset_info['asset_symbol']}
                Start timestamp: {self.start_timestamp}
                End timestamp: {self.end_timestamp}
                Current timestamp: {self.state['timestamp']}
                Environment status: done (intraday session complete)
                </info>
                <action>
                Expected executed action of assistant: {action}
                Actual executed action because of cash or position constraint: {info['action_label']}
                Action result: The action return is {info['ret'] * 100:.2f}%.
                </action>
                <result>
                Total profit: {info['total_profit']:.2f}%
                Intraday trading metrics: 
                {metrics_string}
                Trading records saved to {os.path.join(self.base_dir, 'intraday_trading_records.csv')}
                </result>
                """) 
            return res
    
    async def get_state(self) -> Dict[str, Any]:
        """Get the current state of the Trading Offline environment."""
        try:
            timestamp = self.state["timestamp"]
            timestamp_string = f"{timestamp}"
            
            cash = self.info["cash"]
            position = self.info["position"]
            profit = self.info["total_profit"]
            
            prices = self.state["prices"] # close, high, low, open, volume
            if prices is not None:
                close = prices["close"][-1]
                high = prices["high"][-1]
                low = prices["low"][-1]
                open = prices["open"][-1]
                volume = prices["volume"][-1]
                prices_string = f"close={close:.2f}, high={high:.2f}, low={low:.2f}, open={open:.2f}, volume={volume:02d}"
            else:
                prices_string = "No prices available"
            
            # 1. info string
            info_string = dedent(f"""
                <info>
                Timestamp: {timestamp_string}
                Prices: {prices_string}
                Current cash: {cash:.2f}
                Current position: {position:04d}
                Current profit: {profit:.2f}%
                </info>
            """)

            # 2. news string
            news = self.state["news"]
            has_news = news is not None
            if has_news:
                news_string = "<news>"
                news = news[['title', 'summary']]
                news['title'] = news['title'].apply(lambda x: x.replace("\n", ""))
                news['summary'] = news['summary'].apply(lambda x: x.replace("\n", ""))
                
                for index, row in news.iterrows():
                    news_string += f"Timestamp: {index}\n"
                    news_string += f"Title: {row['title']}\n"
                    news_string += f"Summary: {row['summary']}\n"
                    news_string += "\n"
                news_string += "</news>"
            else:
                news_string = "<news>No news available</news>"
            
            # 3. review actions string
            review_actions = self.state["review_actions"]
            if review_actions is not None:
                review_actions_string = "<history_trading_actions>"
                review_actions = review_actions[['price', 'cash', 'position', 'value', 'action_label', 'total_profit']]
                review_actions = review_actions.rename(columns={
                    "action_label": "action",
                    "total_profit": "profit",
                })
                review_actions = review_actions.round(2)
                review_actions_string += review_actions.to_markdown(index=True)
                review_actions_string += "</history_trading_actions>"
            else:
                review_actions_string = "<history_trading_actions>No valid actions available</history_trading_actions>"
            
            # 4. review trends string
            review_trends = self.state["review_trends"]
            if review_trends is not None:
                review_trends_string = "<history_price_trends>"
                review_trends = review_trends[['close', 'high', 'low', 'open', 'volume']]
                review_trends = review_trends.round(2)
                review_trends_string += review_trends.to_markdown(index=True)
                review_trends_string += "</history_price_trends>"
            else:
                review_trends_string = "<history_price_trends>No price trend available</history_price_trends>"
            
            state = dedent(f"""
                {info_string}
                {news_string}
                {review_actions_string}
                {review_trends_string}
            """)
            
            token_count = get_token_count(state)
            logger.info(f"| 🔢 Token count: {token_count}")
            
            extra = {
                "timestamp": timestamp,
                "prices": prices,
                "prices_string": prices_string,
                "news": news,
                "news_string": news_string,
                "has_news": has_news,
                "review_actions": review_actions,
                "review_actions_string": review_actions_string,
                "review_trends": review_trends,
                "review_trends_string": review_trends_string,
                "token_count": token_count,
            }
            
            return {
                "state": state,
                "extra": extra,
            }
            
        except Exception as e:
            logger.error(f"Failed to get trading state: {e}")
            return {
                "state": "Failed to get trading state",
                "extra": {
                    "error": str(e),
                },
            }


import warnings
warnings.filterwarnings("ignore")
from copy import deepcopy
from pandas import DataFrame
from typing import Any, Optional, Dict
import random
import numpy as np
import pandas as pd
from dataclasses import asdict

from src.utils import TradingRecords, Record
from src.utils import get_token_count
from src.utils import get_start_end_timestamp
from src.environments.protocol import ecp
from src.logger import logger
from src.utils import dedent

from src.environments.protocol.environment import BaseEnvironment
from src.supports.metric import ARR, SR, MDD, SOR, CR, VOL

_STATE_RULES = """
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
"""


def sample_news(df: pd.DataFrame, sample_texts: int = 2):
    """
    Sample news from the news_df.
    :param news_df: DataFrame of news
    :param sample_texts: number of texts to sample
    :return: sampled news
    """
    if len(df) == 0:
        return None
    else:
        df = df.reset_index(drop=False)
        df['date'] = df['timestamp'].apply(lambda x: x.strftime('%Y-%m-%d'))
        df = df.groupby('date').apply(lambda x: x.sample(n=min(sample_texts, len(x)), random_state=0)).reset_index(drop=True)
        df.drop(columns=['date'], inplace=True)
        df.set_index('timestamp', inplace=True)
        return df

def convert_dataframe_to_markdown(
        price: pd.DataFrame,
        news: pd.DataFrame,
        record: pd.DataFrame,
        valid_action: pd.DataFrame,
    ):

    price_string = price.to_markdown(index=False)

    news_string = f"**Timestamp | Title | Content**\n"
    if news is None:
        news_string += f"**No news available**\n"
    else:
        for row in news.iterrows():
            timestamp = row[0]
            values = row[1]

            content = values["summary"] if "summary" in values else values["content"]

            if content is not None:
                content = content.replace('\n', '')
            title = row[1]['title']
            news_string += f"{timestamp.strftime('%Y-%m-%d %H:%M:%S')} | {title} | {content}\n"

    record_string = record.to_markdown(index=False)

    valid_action_string = valid_action.to_markdown(index=False)

    res_strings = dict(
        price=price_string,
        news=news_string,
        record=record_string,
        valid_action=valid_action_string,
    )

    return res_strings

@ecp.environment(name = "trading_offline",
                 type = "Trading Offline",
                 description = "Trading offline environment for trading",
                 has_vision = False,
                 additional_rules = {
                    "state": _STATE_RULES,
                 })
class TradingOfflineEnvironment(BaseEnvironment):
    def __init__(
        self,
        *args,
        mode: str = "train",
        dataset: Any = None,
        initial_amount: float = 1e3,
        transaction_cost_pct: float = 1e-3,
        history_timestamps: int = 5,
        step_timestamps: int = 1,
        future_timestamps: int = 1,
        start_timestamp='2008-04-01',
        end_timestamp='2021-04-01',
        gamma: float = 0.99,
        record_max_len: int = 5,
        valid_action_max_len: int = 8,
        single_text_max_tokens: int = 1024,
        single_text_min_tokens: int = 256,
        daily_sample_texts: int = 2,
        **kwargs,
    ):
        super(TradingOfflineEnvironment, self).__init__()

        self.mode = mode
        self.dataset = dataset
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

        self.initial_amount = initial_amount
        self.transaction_cost_pct = transaction_cost_pct

        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.start_timestamp, self.end_timestamp = get_start_end_timestamp(
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
            level=self.level
        )

        self.history_timestamps = history_timestamps
        self.step_timestamps = step_timestamps
        self.future_timestamps = future_timestamps
        self.gamma = gamma
        self.record_max_len = record_max_len
        self.valid_action_max_len = valid_action_max_len
        self.single_text_max_tokens = single_text_max_tokens
        self.single_text_min_tokens = single_text_min_tokens
        self.daily_sample_texts = daily_sample_texts

        self.res_info = self._init_features()
        self.timestamp_info = self.res_info['timestamp_info']

        self.features_df = self.res_info['features_df']
        self.original_prices_df = self.res_info['original_prices_df']
        self.news_df = self.res_info['news_df']

        self.action_labels = ['SELL', 'HOLD', 'BUY']  # 0, 1, 2
        self.action_dim = len(self.action_labels)

        self.record_df = pd.DataFrame() # record the trading history
        self.valid_action_df = pd.DataFrame() # record the valid action
        self.trading_records = TradingRecords()

        self.state, self.info = self.reset()
        self.trading_records.add(
            dict(
                timestamp=self.info["timestamp"],
                price=self.info["price"],
                cash=self.info["cash"],
                position=self.info["position"],
                value=self.info["value"],
            )
        )
        

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

        return price

    def get_price_full_df(self):
        start_timestamp_index = self.timestamp_min_index
        end_timestamp_index = self.timestamp_max_index

        start_timestamp = self.timestamp_info[start_timestamp_index]["end_timestamp"]
        end_timestamp = self.timestamp_info[end_timestamp_index]["end_timestamp"]

        original_prices_df = self._get_dataitem(self.original_prices_df,
                                       start_timestamp,
                                       end_timestamp)
        return original_prices_df

    def get_price_full(self, timestamp_index: int):

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

        return close, high, low, open, volume

    def get_state_data(self, timestamp_index: int):
        timestamp_info = self.timestamp_info[timestamp_index]

        start_timestamp = timestamp_info['start_timestamp']
        end_timestamp = timestamp_info['end_timestamp']

        price = self._get_dataitem(self.original_prices_df, start_timestamp, end_timestamp)
        news = self._get_dataitem(self.news_df, start_timestamp, end_timestamp)

        record = self.record_df
        valid_action = self.valid_action_df

        sampled_news = sample_news(df=news, sample_texts=self.daily_sample_texts)

        # convert to markdown
        strings = convert_dataframe_to_markdown(
            price=price,
            news=sampled_news,
            record=record,
            valid_action=valid_action,
        )
        price_string = strings['price']
        news_string = strings['news']
        record_string = strings['record']
        valid_action_string = strings['valid_action']
        
        prompt = dedent(f"""
            <symbol>
            Name: {self.asset_info['asset_name']}
            Symbol: {self.asset_info['asset_symbol']}
            </symbol>
            <price>
            {price_string}
            </price>
            <news>
            {news_string}
            </news>
            <record>
            {record_string}
            </record>
            <history_valid_action>
            {valid_action_string}
            </history_valid_action>
            <current_state>
            Today is {end_timestamp.strftime('%Y-%m-%d %H:%M:%S')}, and the current price, cash, and position are {self.price:.2f}, {self.cash:.2f}, and {self.position:04d}.
            </current_state>
            <environment_status>
            The environment status is {'done' if self.done else 'running'}.
            </environment_status>
            """)
        
        prompt_token_nums = get_token_count(prompt)

        state = dict(
            timestamp=end_timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            prompt=prompt,
            prompt_token_nums=prompt_token_nums,
        )
        state.update(self.asset_info)

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

    def _init_record(self):

        timestamp_info = self.timestamp_info[self.timestamp_index]

        start_timestamp = timestamp_info['start_timestamp']
        end_timestamp = timestamp_info['end_timestamp']

        price = self._get_dataitem(self.original_prices_df, start_timestamp, end_timestamp)

        rows = list(price.iterrows())

        for row in rows[:-1]:
            timestamp = row[0]
            timestamp_string = timestamp.strftime(self.level_format.value)
            close, high, low, open, volume = row[1].values
            self.last_record = Record(
                timestamp=timestamp_string,
                open=open,
                high=high,
                low=low,
                close=close,
                volume=volume,
                price=self.price,
                cash=self.cash,
                position=self.position,
                pre_value=self.initial_amount,
                action='HOLD',
                post_value=self.initial_amount,
                ret=0.0,
            )
            self.record_df = pd.concat([self.record_df, pd.DataFrame([asdict(self.last_record)])], ignore_index=True)

        # last record, because the action is not predicted, so the pre_value, action, post_value, ret are None
        timestamp = rows[-1][0]
        timestamp_string = timestamp.strftime(self.level_format.value)
        close, high, low, open, volume = rows[-1][1].values
        self.last_record = Record(
            timestamp=timestamp_string,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
            price=self.price,
            cash=self.cash,
            position=self.position,
            pre_value=None,
            action=None,
            post_value=None,
            ret=None,
        )
        self.record_df = pd.concat([self.record_df, pd.DataFrame([asdict(self.last_record)])], ignore_index=True)

        self.valid_action_df = self.record_df[self.record_df['action'] != 'HOLD']

    def _add_record(self, record):
        self.record_df = pd.concat([self.record_df, pd.DataFrame([asdict(record)])], ignore_index=True)

        record_max_len = min(self.record_max_len, len(self.record_df))
        self.record_df = self.record_df[-record_max_len:]
        self.valid_action_df = self.record_df[self.record_df['action'] != 'HOLD']

        valid_action_max_len = min(self.valid_action_max_len, len(self.valid_action_df))
        self.valid_action_df = self.valid_action_df[-valid_action_max_len:]

    def _update_record(self, record):

        last_record = self.record_df.iloc[-1]

        last_record['pre_value'] = record.pre_value
        last_record['action'] = record.action
        last_record['post_value'] = record.post_value
        last_record['ret'] = record.ret

        self.record_df.iloc[-1] = last_record

        self.valid_action_df = self.record_df[self.record_df['action'] != 'HOLD']
        valid_action_max_len = min(self.valid_action_max_len, len(self.valid_action_df))
        self.valid_action_df = self.valid_action_df[-valid_action_max_len:]

    def reset(self, **kwargs):
        self.timestamp_index = self._init_timestamp_index()
        self.timestamp_string = self.get_timestamp_string(timestamp_index=self.timestamp_index)
        self.price = self.get_price(timestamp_index=self.timestamp_index)

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

        # init record
        self._init_record()

        # after init record, get the state
        self.state = self.get_state_data(timestamp_index=self.timestamp_index)

        info = dict(
            timestamp=self.timestamp_string,
            ret=self.ret,
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

        return self.state, info

    def _extract_action(self, action: str):
        for index, label in enumerate(self.action_labels):
            if label == action:
                return index
        return 1 # HOLD

    def step(self, action: str):

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

        # update record
        self.last_record.pre_value = self.pre_value
        self.last_record.action = self.action_label
        self.last_record.post_value = self.value
        self.last_record.ret = ret
        self._update_record(self.last_record)

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
        self.price = self.get_price(timestamp_index=self.timestamp_index)

        # next record
        close, high, low, open, volume = self.get_price_full(timestamp_index=self.timestamp_index)
        self.last_record = Record(
            timestamp=self.timestamp_string,
            open=open,
            high=high,
            low=low,
            close=close,
            volume=volume,
            price=self.price,
            cash=self.cash,
            position=self.position,
            pre_value=None,
            action=None,
            post_value=None,
            ret=None,
        )
        self._add_record(self.last_record)

        # after update record, get the state
        self.state = self.get_state_data(timestamp_index=self.timestamp_index)

        info = dict(
            timestamp=self.timestamp_string,
            ret=self.ret,
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

        # update the pre_value
        self.pre_value = self.value

        return self.state, reward, self.done, self.truncted, info
    
    async def initialize(self) -> None:
        """Initialize the trading offline environment."""
        logger.info(f"| 💰 Trading Offline Environment initialized")
    
    @ecp.action(name = "step",
                type = "Trading Offline",
                description = "Step the trading environment.")
    async def step(self, action: str) -> str:
        """Step the trading environment.
        
        Args:
            action (str): The action to take. Should be `BUY`, `SELL` or `HOLD`.

        Returns:
            str: The state of the trading environment.
        """
        state, reward, done, truncted, info = self.step(action)
        self.trading_records.add(
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
        
        if not done:
            logger.info(f"| Prompt Token Numbers: {state['prompt_token_nums']}")
        
            res = dedent(f"""
                <info>
                Name: {self.asset_info['asset_name']}
                Symbol: {self.asset_info['asset_symbol']}
                Start timestamp: {self.start_timestamp}
                End timestamp: {self.end_timestamp}
                Current timestamp: {info['timestamp']}
                Environment status: running
                </info>
                <action>
                Expected executed action of assistant: {action}
                Actual executed action because of cash or position constraint: {info['action_label']}
                </action>
                <result>
                Total profit: {info['total_profit']:.4f}%
                Reward: {reward:.4f}
                </result>
                """)
            
            return res
        
        else:
            self.trading_records.add(
                dict(
                    action=1,
                    action_label="HOLD",
                    ret=0.0,
                    total_profit=info['total_profit'],
                )
            )
            
            rets = np.array(self.trading_records.data['ret'])
            for metric_name, metric in self.metrics_functions.items():
                self.metrics[metric_name] = metric(rets)
                
            metrics_string = f"**Metric | Value**\n"
            for metric_name, metric_value in self.metrics.items():
                metrics_string += f"{metric_name} | {metric_value:.4f}\n"
                
            res = dedent(f"""
                <info>
                Name: {self.asset_info['asset_name']}
                Symbol: {self.asset_info['asset_symbol']}
                Start timestamp: {self.start_timestamp}
                End timestamp: {self.end_timestamp}
                Current timestamp: {info['timestamp']}
                Environment status: done
                </info>
                <action>
                Expected executed action of assistant: {action}
                Actual executed action because of cash or position constraint: {info['action_label']}
                </action>
                <result>
                Total profit: {info['total_profit']:.4f}%
                Reward: {reward:.4f}
                Trading metrics: 
                {metrics_string}
                </result>
                """) 
            return res
        
    @ecp.action(name = "save",
                type = "Trading Offline",
                description = "Save the trading records.")
    async def save(self, file_path: str) -> str:
        """Save the trading records.
        
        Args:
            file_path (str): The absolute path to save the trading records.

        Returns:
            str: The message of the trading records saved successfully.
        """
        df = self.trading_records.to_dataframe()
        df.to_csv(file_path, index=False)
        return f"Trading records saved successfully to {file_path}."
    
    async def get_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "state": self.state['prompt']
        }
        return state
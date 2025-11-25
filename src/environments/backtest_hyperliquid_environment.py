"""
回测环境：模拟 HyperliquidEnvironment，使用本地缓存的历史数据进行回测
"""

import asyncio
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, List, Union, Type
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd

from src.logger import logger
from src.utils import dedent
from src.environments.protocol.environment import BaseEnvironment
from src.environments.protocol import ecp


class BacktestPosition(BaseModel):
    """回测持仓信息"""
    symbol: str
    side: str  # "long" or "short"
    size: float  # 持仓数量（张数）
    entry_price: float  # 开仓价格
    leverage: int = 1
    unrealized_pnl: float = 0.0


class BacktestOrder(BaseModel):
    """回测订单信息"""
    order_id: str
    symbol: str
    side: str  # "buy" or "sell"
    qty: float
    price: float
    order_type: str  # "Market" or "Limit"
    status: str  # "filled", "pending", "cancelled"
    filled_qty: float = 0.0
    filled_price: float = 0.0
    timestamp: datetime
    fee: float = 0.0


@ecp.environment()
class BacktestHyperliquidEnvironment(BaseEnvironment):
    """
    回测环境：模拟 HyperliquidEnvironment，使用本地数据库中的历史数据进行回测
    
    特点：
    - 从本地数据库读取历史K线数据
    - 按时间顺序逐步推进，模拟实时交易
    - 支持 LLM agent 进行交易决策
    - 记录所有交易和持仓变化
    
    重要：环境名称设置为 "hyperliquid"，这样 Agent 可以通过相同的名称访问
    （与 online trading 的环境名称一致，Agent 代码不需要修改）
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")
    
    name: str = Field(default="hyperliquid", description="回测环境名称（与online trading一致）")
    type: str = Field(default="Hyperliquid Trading", description="回测交易环境（与online trading一致）")
    description: str = Field(default="Hyperliquid trading environment for backtest with historical data", description="回测环境描述")
    args_schema: Type[BaseModel] = Field(default=None, description="The args schema of the backtest environment.")
    metadata: Dict[str, Any] = Field(default={
        "has_vision": False,
        "additional_rules": {
            "state": "The state of the Hyperliquid trading environment including account information, positions, and market data.",
        }
    }, description="The metadata of the backtest environment.")
    
    def __init__(
        self,
        db_path: Union[str, Path],
        symbol: str = "BTC",
        initial_equity: float = 1000.0,
        max_leverage: float = 5.0,
        taker_fee_rate: float = 0.0005,
        slippage_bps: float = 1.0,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ):
        """
        初始化回测环境
        
        Args:
            db_path: 数据库路径
            symbol: 交易币种（如 "BTC"）
            initial_equity: 初始资金
            max_leverage: 最大杠杆
            taker_fee_rate: 手续费率
            slippage_bps: 滑点（基点）
            start_date: 回测开始日期（格式: 'YYYY-MM-DD HH:MM:SS'）
            end_date: 回测结束日期（格式: 'YYYY-MM-DD HH:MM:SS'）
        """
        super().__init__(**kwargs)
        
        self.db_path = Path(db_path)
        self.symbol = symbol
        self.initial_equity = initial_equity
        self.max_leverage = max_leverage
        self.taker_fee_rate = taker_fee_rate
        self.slippage_bps = slippage_bps
        self.start_date = start_date
        self.end_date = end_date
        
        # 回测状态
        self.current_index = 0
        self.historical_data: pd.DataFrame = None
        self.current_time: Optional[datetime] = None
        self.current_price: Optional[float] = None
        
        # 账户状态
        self.equity = initial_equity
        self.base_equity = initial_equity  # 基础权益（已实现盈亏）
        self.positions: Dict[str, BacktestPosition] = {}  # symbol -> position
        self.orders: List[BacktestOrder] = []
        self.trades_history: List[Dict[str, Any]] = []
        
        # 统计信息（用于模拟 HyperliquidEnvironment 的 metrics）
        self.initial_account_value = None  # 将在第一次调用时初始化
        self.account_value = None
        self.max_account_value = None
        self.total_profit = 0.0
        self.equity_curve: List[Dict[str, Any]] = []
        self.max_equity = initial_equity
        self.max_drawdown = 0.0
        
        # 订单ID计数器
        self._order_id_counter = 0
        
    async def initialize(self) -> None:
        """初始化回测环境，加载历史数据"""
        logger.info(f"| 🚀 初始化回测环境: {self.symbol}")
        logger.info(f"| 📊 数据库路径: {self.db_path}")
        
        # 从数据库读取数据
        await self._load_historical_data()
        
        if self.historical_data is None or len(self.historical_data) == 0:
            raise ValueError(f"无法加载历史数据: {self.symbol}")
        
        logger.info(f"| ✅ 加载了 {len(self.historical_data)} 条历史数据")
        logger.info(f"| 📅 时间范围: {self.historical_data.iloc[0]['timestamp_utc']} 到 {self.historical_data.iloc[-1]['timestamp_utc']}")
        
        # 初始化到第一条数据
        await self._advance_to_index(0)
        
    async def _load_historical_data(self) -> None:
        """从数据库加载历史数据"""
        import sqlite3
        
        table_name = f"data_{self.symbol}_candle"
        
        conn = sqlite3.connect(self.db_path)
        
        # 构建查询
        query = f"SELECT * FROM {table_name}"
        conditions = []
        params = []
        
        if self.start_date:
            conditions.append("timestamp_utc >= ?")
            params.append(self.start_date)
        
        if self.end_date:
            conditions.append("timestamp_utc <= ?")
            params.append(self.end_date)
        
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        
        query += " ORDER BY timestamp ASC, open_time ASC"
        
        # 读取数据
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if len(df) > 0:
            # 转换时间格式
            df['timestamp_utc'] = pd.to_datetime(df['timestamp_utc'])
            df['close_time_utc'] = pd.to_datetime(df['close_time_utc'])
            
            # 确保数据列名匹配回测需要的格式
            # 回测需要: t (开盘时间), T (收盘时间), o, h, l, c, v
            df['t'] = pd.to_datetime(df['open_time_utc'])
            df['T'] = pd.to_datetime(df['close_time_utc'])
            df['o'] = df['open']
            df['h'] = df['high']
            df['l'] = df['low']
            df['c'] = df['close']
            df['v'] = df['volume']
            
            self.historical_data = df.reset_index(drop=True)
        else:
            self.historical_data = pd.DataFrame()
    
    async def _advance_to_index(self, index: int) -> None:
        """推进到指定的数据索引"""
        if index < 0 or index >= len(self.historical_data):
            return
        
        self.current_index = index
        row = self.historical_data.iloc[index]
        
        # 更新当前时间和价格
        self.current_time = pd.to_datetime(row['T'])  # 使用收盘时间
        self.current_price = float(row['close'])
        
        # 更新持仓的未实现盈亏
        await self._update_unrealized_pnl()
        
        # 更新账户权益
        await self._update_equity()
        
        # 记录权益曲线
        self.equity_curve.append({
            "time": self.current_time,
            "equity": self.equity,
            "price": self.current_price,
            "index": index
        })
    
    async def _update_unrealized_pnl(self) -> None:
        """更新持仓的未实现盈亏"""
        for symbol, position in self.positions.items():
            if position.side == "long":
                position.unrealized_pnl = (self.current_price - position.entry_price) * position.size
            else:  # short
                position.unrealized_pnl = (position.entry_price - self.current_price) * position.size
    
    async def _update_equity(self) -> None:
        """更新账户权益"""
        # 计算未实现盈亏
        total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
        
        # 总权益 = 基础权益（初始资金 + 已实现盈亏） + 未实现盈亏
        self.equity = self.base_equity + total_unrealized_pnl
        
        # 更新最大权益和最大回撤
        if self.equity > self.max_equity:
            self.max_equity = self.equity
        
        current_drawdown = (self.max_equity - self.equity) / self.max_equity if self.max_equity > 0 else 0.0
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
    
    async def advance_to_next(self) -> bool:
        """推进到下一个时间点，返回是否还有更多数据"""
        next_index = self.current_index + 1
        if next_index >= len(self.historical_data):
            return False
        
        await self._advance_to_index(next_index)
        return True
    
    def _safe_float_format(self, value: Any, default: float = 0.0) -> str:
        """安全格式化浮点数（模拟 HyperliquidEnvironment）"""
        try:
            if value is None:
                return f"{float(default):.2f}"
            if isinstance(value, (int, float)):
                return f"{float(value):.2f}"
            if isinstance(value, str):
                return f"{float(value):.2f}"
            else:
                return f"{float(default):.2f}"
        except (ValueError, TypeError):
            return f"{float(default):.2f}"
    
    async def get_state(self) -> Dict[str, Any]:
        """获取环境状态（完全模拟 HyperliquidEnvironment 的格式）"""
        from src.utils import dedent
        import json
        
        try:
            # 获取账户信息（完全模拟 HyperliquidEnvironment）
            account_result = await self.get_account()
            account_result_extra = account_result.get("extra", {})
            account_info = account_result_extra.get("account", {})
            metrics = account_result_extra.get("metrics", {})
            
            account_string = dedent(f"""
                Timestamp: {account_info.get("time", "N/A")}
                Account Value: ${self._safe_float_format(account_info.get("margin_summary", {}).get("accountValue", 0))},
                Total Profit: ${self._safe_float_format(metrics.get("total_profit", 0))} ({self._safe_float_format(metrics.get("profit_percentage", 0))}%),
                Current Drawdown: ${self._safe_float_format(metrics.get("current_drawdown", 0))} ({self._safe_float_format(metrics.get("current_drawdown_percentage", 0))}%),
                Max Drawdown: ${self._safe_float_format(metrics.get("max_drawdown", 0))} ({self._safe_float_format(metrics.get("max_drawdown_percentage", 0))}%),
                Period Return: ${self._safe_float_format(metrics.get("period_return", 0))} ({self._safe_float_format(metrics.get("period_return_percentage", 0))}%),
            """)
            account_string = dedent(f"""
                <account>
                {account_string}
                </account>
            """)
            
            # 获取持仓信息（完全模拟 HyperliquidEnvironment）
            positions_result = await self.get_positions()
            positions_result_extra = positions_result.get("extra", {})
            positions_list = positions_result_extra.get("positions", [])
            positions_string = ""
            for position in positions_list:
                return_on_equity = position.get('return_on_equity', 0)
                try:
                    return_on_equity_float = float(return_on_equity)
                except (ValueError, TypeError):
                    return_on_equity_float = 0.0
                return_on_equity_pct = self._safe_float_format(return_on_equity_float)
                
                position_side = "LONG" if float(position.get('position_amt', 0)) > 0 else "SHORT"
                
                positions_string += f"Symbol: {position.get('symbol', 'N/A')}, Position Side: {position_side}, Trade Type: {position.get('trade_type', 'perpetual')}, Leverage: {position.get('leverage', '1')}, Position Amount: {position.get('position_amt', '0')}, Entry Price: {position.get('entry_price', '0')}, Current Price: {position.get('mark_price', '0')}, Return on Equity: {return_on_equity_pct}%\n"
            
            positions_string = dedent(f"""
                <positions>
                {positions_string}
                </positions>
            """)
            
            # 获取订单信息（从本地维护的订单列表中获取）
            orders_result = await self.get_orders()
            orders_result_extra = orders_result.get("extra", {})
            orders_list = orders_result_extra.get("orders", [])
            orders_string = ""
            for order in orders_list:
                quantity_str = self._safe_float_format(order.get('quantity', 0))
                price_str = self._safe_float_format(order.get('price', 0)) if order.get('price') else "N/A"
                orders_string += f"Order ID: {order.get('order_id', 'N/A')}, Symbol: {order.get('symbol', 'N/A')}, Trade Type: {order.get('trade_type', 'perpetual')}, Order Type: {order.get('type', 'N/A')}, Order Side: {order.get('side', 'N/A')}, Quantity: {quantity_str}, Price: {price_str}, Status: {order.get('status', 'N/A')}\n"
            
            if not orders_string:
                orders_string = "No orders found."
            
            orders_string = dedent(f"""
                <orders>
                {orders_string}
                </orders>
            """)
            
            # 获取市场数据（完全模拟 HyperliquidEnvironment 的格式）
            data_result = await self.get_data(limit=30)
            data_result_extra = data_result.get("extra", {})
            
            candles = {}
            indicators = {}
            for symbol, data in data_result_extra.get("data", {}).items():
                candles[symbol] = data.get("candles", data.get("candle", []))  # 支持两种格式
                indicators[symbol] = data.get("indicators", [])  # 从数据库读取的指标数据
            
            candles_string = ""
            for symbol, candles_list in candles.items():
                if not candles_list:
                    continue
                
                # 创建表格（完全模拟 HyperliquidEnvironment 的格式）
                symbol_string = f"Symbol: {symbol}. History {len(candles_list)} minutes candles data.\n"
                symbol_string += "| Timestamp           | Open | High | Low | Close | Volume | Trade Count |\n"
                symbol_string += "|---------------------|------|------|-----|-------|--------|-------------|\n"
                
                # 添加表格行
                for candle in candles_list:
                    timestamp = candle.get("timestamp_local", candle.get("timestamp_utc", ""))
                    open_val = self._safe_float_format(candle.get("open"))
                    high_val = self._safe_float_format(candle.get("high"))
                    low_val = self._safe_float_format(candle.get("low"))
                    close_val = self._safe_float_format(candle.get("close"))
                    volume_val = self._safe_float_format(candle.get("volume"))
                    trade_count_val = self._safe_float_format(candle.get("trade_count", 0))
                    symbol_string += f"| {timestamp:<19} | {open_val:>10} | {high_val:>10} | {low_val:>10} | {close_val:>10} | {volume_val:>10} | {trade_count_val:>11} |\n"
                
                candles_string += symbol_string + "\n"
            
            # 格式化指标数据（与在线环境一致）
            indicators_string = ""
            for symbol, indicators_list in indicators.items():
                if not indicators_list:
                    continue
                
                # 获取指标名称（从第一条记录中提取）
                if indicators_list and isinstance(indicators_list, list) and len(indicators_list) > 0:
                    first_indicator = indicators_list[0]
                    indicator_names = [k for k in first_indicator.keys() 
                                     if k not in ['timestamp', 'timestamp_utc', 'timestamp_local', 'symbol', 'id', 'created_at']]
                    
                    if not indicator_names:
                        continue
                    
                    # 创建表格头部
                    symbol_string = f"Symbol: {symbol}. History {len(indicators_list)} minutes indicators data.\n"
                    header_cols = ["Timestamp           "] + [name.upper() for name in indicator_names]
                    symbol_string += "| " + " | ".join(header_cols) + " |\n"
                    
                    # 创建分隔行
                    separator_cols = ["---------------------"] + ["-" * max(12, len(name.upper())) for name in indicator_names]
                    separator = "| " + " | ".join(separator_cols) + " |\n"
                    symbol_string += separator
                    
                    # 添加数据行
                    for indicator in indicators_list:
                        timestamp = indicator.get("timestamp_local", indicator.get("timestamp_utc", ""))
                        row_values = [str(timestamp)]
                        
                        # 按顺序添加指标值
                        for indicator_name in indicator_names:
                            indicator_value = indicator.get(indicator_name, None)
                            if indicator_value is not None:
                                row_values.append(self._safe_float_format(indicator_value))
                            else:
                                row_values.append("")
                        
                        # 格式化行
                        formatted_row = f"| {row_values[0]:<19} |"
                        for val in row_values[1:]:
                            formatted_row += f" {val:>12} |"
                        formatted_row += "\n"
                        symbol_string += formatted_row
                    
                    indicators_string += symbol_string + "\n"
            
            data_string = dedent(f"""
                <data>
                <candles>
                {candles_string}
                </candles>
                <indicators>
                {indicators_string}
                </indicators>
                </data>
            """)
            
            # 组合完整的状态字符串（完全模拟 HyperliquidEnvironment）
            # 注意：在线环境使用 account_info（字典），但这里我们需要使用 account_string（字符串）来匹配格式
            state = dedent(f"""
                <state>
                {account_string}
                {orders_string}
                {data_string}
                </state>
            """)
            
            return {
                "state": state,
                "extra": {
                    "account": account_result,
                    "positions": positions_result,
                    "orders": orders_result,  # 使用实际的 orders_result，而不是硬编码
                    "input": data_result,  # 使用完整的 data_result，包含 extra
                }
            }
        except Exception as e:
            logger.error(f"Failed to get backtest state: {e}")
            import traceback
            traceback.print_exc()
            return {
                "state": f"Failed to get backtest state: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def get_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: Optional[int] = 30
    ) -> Dict[str, Any]:
        """获取K线数据和指标（模拟 HyperliquidEnvironment 的接口）"""
        try:
            # 当前时间点的索引
            end_idx = self.current_index + 1
            
            # 确定起始索引
            if limit:
                start_idx = max(0, end_idx - limit)
            else:
                start_idx = 0
            
            # 获取数据切片
            data_slice = self.historical_data.iloc[start_idx:end_idx].copy()
            
            # 转换为 Hyperliquid 格式
            candles = []
            for _, row in data_slice.iterrows():
                candle = {
                    "timestamp": int(row['timestamp']),
                    "timestamp_utc": row['timestamp_utc'].strftime('%Y-%m-%d %H:%M:%S'),
                    "timestamp_local": row['timestamp_utc'].strftime('%Y-%m-%d %H:%M:%S'),  # 添加本地时间
                    "open_time": int(row['open_time']),
                    "open_time_utc": row['open_time_utc'],
                    "close_time": int(row['close_time']),
                    "close_time_utc": row['close_time_utc'].strftime('%Y-%m-%d %H:%M:%S'),
                    "open": float(row['open']),
                    "high": float(row['high']),
                    "low": float(row['low']),
                    "close": float(row['close']),
                    "volume": float(row['volume']),
                    "trade_count": float(row['trade_count']) if 'trade_count' in row else None
                }
                candles.append(candle)
            
            # 从数据库读取指标数据（如果存在）
            indicators = []
            try:
                conn = sqlite3.connect(self.db_path)
                symbol_upper = self.symbol.upper()
                table_name = f"data_{symbol_upper}_indicators"
                
                # 检查指标表是否存在
                check_table = f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
                table_exists = conn.execute(check_table).fetchone()
                
                if table_exists:
                    # 获取对应时间戳的指标数据
                    timestamps = [int(row['timestamp']) for _, row in data_slice.iterrows()]
                    if timestamps:
                        placeholders = ','.join(['?' for _ in timestamps])
                        query = f"SELECT * FROM {table_name} WHERE symbol = ? AND timestamp IN ({placeholders}) ORDER BY timestamp ASC"
                        params = [symbol_upper] + timestamps
                        indicator_df = pd.read_sql_query(query, conn, params=params)
                        
                        if not indicator_df.empty:
                            # 转换为字典列表
                            for _, ind_row in indicator_df.iterrows():
                                indicator_dict = {
                                    "timestamp": int(ind_row['timestamp']),
                                    "timestamp_utc": ind_row.get('timestamp_utc', ''),
                                    "timestamp_local": ind_row.get('timestamp_local', ind_row.get('timestamp_utc', '')),
                                    "symbol": ind_row.get('symbol', symbol_upper)
                                }
                                # 添加所有指标列（排除非指标列）
                                exclude_cols = ['id', 'timestamp', 'timestamp_utc', 'timestamp_local', 'symbol', 'created_at']
                                for col in indicator_df.columns:
                                    if col not in exclude_cols:
                                        value = ind_row.get(col)
                                        if pd.notna(value):
                                            indicator_dict[col] = float(value)
                                indicators.append(indicator_dict)
                conn.close()
            except Exception as e:
                logger.warning(f"Failed to load indicators from database: {e}")
                # 如果读取指标失败，继续使用空列表
            
            return {
                "success": True,
                "message": f"Retrieved {len(candles)} candle records and {len(indicators)} indicator records for {self.symbol}",
                "extra": {
                    "data": {
                        self.symbol: {
                            "candles": candles,  # 改为 "candles" 以匹配在线环境的格式
                            "indicators": indicators  # 添加指标数据
                        }
                    },
                    "symbols": [self.symbol],
                    "data_type": "candle",
                    "row_count": len(candles)
                }
            }
        except Exception as e:
            logger.error(f"Error getting data: {e}")
            return {
                "success": False,
                "message": f"Failed to get data: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def get_account(self) -> Dict[str, Any]:
        """获取账户信息（完全模拟 HyperliquidEnvironment 的返回格式）"""
        from src.utils import dedent
        import json
        
        try:
            # 更新权益
            await self._update_equity()
            
            # 计算指标（模拟 HyperliquidEnvironment 的 _calculate_matrics）
            account_value = self.equity
            if self.initial_account_value is None:
                self.initial_account_value = account_value
                self.account_value = account_value
                self.max_account_value = account_value
            
            pre_account_value = self.account_value
            self.account_value = account_value
            
            total_profit = account_value - self.initial_account_value
            profit_percentage = (total_profit / self.initial_account_value * 100) if self.initial_account_value > 0 else 0.0
            
            if account_value > self.max_account_value:
                self.max_account_value = account_value
            
            current_drawdown = self.max_account_value - account_value
            current_drawdown_percentage = (current_drawdown / self.max_account_value * 100) if self.max_account_value > 0 else 0.0
            
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
            
            max_drawdown_percentage = (self.max_drawdown / self.max_account_value * 100) if self.max_account_value > 0 else 0.0
            period_return = account_value - pre_account_value
            period_return_percentage = (period_return / pre_account_value * 100) if pre_account_value > 0 else 0.0
            
            metrics = {
                "current_account_value": account_value,
                "initial_account_value": self.initial_account_value,
                "max_account_value": self.max_account_value,
                "total_profit": total_profit,
                "profit_percentage": profit_percentage,
                "current_drawdown": current_drawdown,
                "current_drawdown_percentage": current_drawdown_percentage,
                "max_drawdown": self.max_drawdown,
                "max_drawdown_percentage": max_drawdown_percentage,
                "period_return": period_return,
                "period_return_percentage": period_return_percentage,
            }
            
            # 构建资产持仓信息（模拟 Hyperliquid 的格式）
            asset_positions = []
            for symbol, pos in self.positions.items():
                asset_positions.append({
                    "type": "oneWay",
                    "position": {
                        "coin": symbol,
                        "szi": str(pos.size),
                        "leverage": {"type": "cross", "value": pos.leverage},
                        "entryPx": str(pos.entry_price),
                        "positionValue": str(pos.size * self.current_price),
                        "unrealizedPnl": str(pos.unrealized_pnl),
                        "returnOnEquity": str(pos.unrealized_pnl / (pos.size * pos.entry_price / pos.leverage) * 100) if pos.size != 0 else "0",
                        "liquidationPx": "0",  # 简化处理
                        "marginUsed": str(abs(pos.size * pos.entry_price / pos.leverage)),
                        "maxLeverage": int(self.max_leverage),
                    }
                })
            
            # 构建账户信息（模拟 Hyperliquid 的格式）
            account = {
                "margin_summary": {
                    "accountValue": str(account_value),
                    "totalNtlPos": str(sum(abs(pos.size * self.current_price) for pos in self.positions.values())),
                    "totalRawUsd": str(sum(pos.unrealized_pnl for pos in self.positions.values())),
                    "totalMarginUsed": str(sum(abs(pos.size * pos.entry_price / pos.leverage) for pos in self.positions.values()))
                },
                "crossMarginSummary": {
                    "accountValue": str(account_value),
                    "totalNtlPos": str(sum(abs(pos.size * self.current_price) for pos in self.positions.values())),
                    "totalRawUsd": str(sum(pos.unrealized_pnl for pos in self.positions.values())),
                    "totalMarginUsed": str(sum(abs(pos.size * pos.entry_price / pos.leverage) for pos in self.positions.values()))
                },
                "asset_positions": asset_positions,
                "time": int(self.current_time.timestamp() * 1000) if self.current_time else 0
            }
            
            asset_positions_json = json.dumps(asset_positions, indent=4)
            
            # 构建返回消息（完全模拟 HyperliquidEnvironment）
            metrics_text = ""
            if metrics:
                metrics_text = dedent(f"""
                Performance Metrics:
                Total Profit: ${metrics['total_profit']:,.2f} ({metrics['profit_percentage']:+.2f}%)
                Current Drawdown: ${metrics['current_drawdown']:,.2f} ({metrics['current_drawdown_percentage']:.2f}%)
                Max Drawdown: ${metrics['max_drawdown']:,.2f} ({metrics['max_drawdown_percentage']:.2f}%)
                """)
            
            result_text = dedent(f"""
                Account Information:
                Timestamp: {account.get("time", "N/A")}
                Account Value: ${account_value:,.2f}
                Asset Positions: {asset_positions_json}
                {metrics_text}
                """)
            
            extra = {
                "account_value": account_value,
                "asset_positions": asset_positions_json,
                "account": account,
                "time": account.get("time", "N/A"),
            }
            
            if metrics:
                extra["metrics"] = metrics
            
            return {
                "success": True,
                "message": result_text,
                "extra": extra
            }
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Failed to get account information: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def get_positions(self, trade_type: Optional[str] = None) -> Dict[str, Any]:
        """获取持仓信息（完全模拟 HyperliquidEnvironment 的返回格式）"""
        from src.utils import dedent
        
        try:
            await self._update_unrealized_pnl()
            
            positions_list = []
            for symbol, pos in self.positions.items():
                # 模拟 Hyperliquid 的格式
                positions_list.append({
                    "symbol": symbol,
                    "position_amt": str(pos.size),  # 使用字符串格式，和在线API一致
                    "entry_price": str(pos.entry_price),
                    "mark_price": str(self.current_price),
                    "unrealized_profit": str(pos.unrealized_pnl),
                    "leverage": str(pos.leverage),
                    "trade_type": "perpetual",
                    "return_on_equity": str(pos.unrealized_pnl / (pos.size * pos.entry_price / pos.leverage) * 100) if pos.size != 0 else "0"
                })
            
            # 构建返回消息（完全模拟 HyperliquidEnvironment）
            if len(positions_list) == 0:
                result_text = "No open positions."
            else:
                position_lines = []
                for pos in positions_list:
                    try:
                        position_amt = float(pos.get("position_amt", 0))
                        entry_price = float(pos.get("entry_price", 0))
                        mark_price = float(pos.get("mark_price", 0))
                        unrealized_profit = float(pos.get("unrealized_profit", 0))
                        leverage = pos.get("leverage", "N/A")
                        trade_type_str = pos.get("trade_type", "N/A")
                        
                        position_lines.append(
                            f"  {pos.get('symbol', 'N/A')} ({trade_type_str}): {position_amt:+.6f} @ Entry: {entry_price}, "
                            f"Mark: {mark_price}, Leverage: {leverage}x, "
                            f"P&L: {unrealized_profit:.6f}"
                        )
                    except (ValueError, TypeError, KeyError):
                        position_lines.append(
                            f"  {pos.get('symbol', 'N/A')}: {pos.get('position_amt', 'N/A')} "
                            f"(P&L: {pos.get('unrealized_profit', 'N/A')})"
                        )
                
                result_text = dedent(f"""
                    {len(positions_list)} open position(s):
                    {chr(10).join(position_lines)}
                    """)
            
            return {
                "success": True,
                "message": result_text,
                "extra": {
                    "positions": positions_list
                }
            }
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {
                "success": False,
                "message": f"Failed to get positions information: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def create_order(
        self,
        symbol: str,
        side: str,
        qty: Optional[float] = None,
        price: Optional[float] = None,
        order_type: str = "Market",
        leverage: Optional[int] = None,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        创建订单（模拟交易执行）
        
        Args:
            symbol: 交易币种
            side: "buy" 或 "sell"
            qty: 交易数量（张数）
            price: 限价单价格（Market 订单忽略）
            order_type: "Market" 或 "Limit"
            leverage: 杠杆倍数
            stop_loss_price: 止损价格（回测环境中记录但不自动执行）
            take_profit_price: 止盈价格（回测环境中记录但不自动执行）
        
        Returns:
            订单执行结果
        """
        try:
            # 回测环境中，所有交易都成功
            # 如果 symbol 不匹配，自动使用 self.symbol
            if symbol != self.symbol:
                symbol = self.symbol
            
            # 确定执行价格（考虑滑点）
            slippage = self.slippage_bps / 10000.0
            if order_type == "Market":
                if side.lower() == "buy":
                    fill_price = self.current_price * (1 + slippage)
                else:  # sell
                    fill_price = self.current_price * (1 - slippage)
            else:  # Limit
                fill_price = price if price else self.current_price
            
            # 回测环境中，所有交易都成功（不检查失败情况）
            if qty is None or qty <= 0:
                qty = 0.01  # 默认最小数量，确保交易成功
            
            # 计算手续费（固定0.6块，开仓和平仓都收取）
            FIXED_FEE = 0.6  # 固定手续费
            
            # 计算名义价值（用于记录）
            notional = qty * fill_price
            
            # 计算当前权益（包括未实现盈亏）
            await self._update_equity()
            
            # 执行交易：更新持仓
            current_pos = self.positions.get(symbol)
            
            if side.lower() == "buy":
                # 买入：增加多头持仓或减少空头持仓
                if current_pos and current_pos.side == "short":
                    # 平空（收取平仓手续费0.6块）
                    close_fee = FIXED_FEE
                    if qty >= current_pos.size:
                        # 完全平仓，可能还要开多
                        close_qty = current_pos.size
                        remaining_qty = qty - close_qty
                        
                        # 计算平仓盈亏（扣除平仓手续费）
                        pnl = (current_pos.entry_price - fill_price) * close_qty - close_fee
                        self.base_equity += pnl
                        
                        # 移除或更新持仓
                        if remaining_qty > 0:
                            # 开多（收取开仓手续费0.6块）
                            open_fee = FIXED_FEE
                            self.base_equity -= open_fee
                            self.positions[symbol] = BacktestPosition(
                                symbol=symbol,
                                side="long",
                                size=remaining_qty,
                                entry_price=fill_price,
                                leverage=leverage or 1
                            )
                            fee = close_fee + open_fee  # 总手续费 = 平仓 + 开仓
                        else:
                            del self.positions[symbol]
                            fee = close_fee  # 只有平仓手续费
                    else:
                        # 部分平空（只收取平仓手续费0.6块）
                        fee = FIXED_FEE
                        pnl = (current_pos.entry_price - fill_price) * qty - fee
                        self.base_equity += pnl
                        current_pos.size -= qty
                else:
                    # 开多或加多（收取开仓手续费0.6块）
                    fee = FIXED_FEE
                    if current_pos and current_pos.side == "long":
                        # 加多（简单平均成本）
                        total_value = current_pos.size * current_pos.entry_price + qty * fill_price
                        current_pos.size += qty
                        current_pos.entry_price = total_value / current_pos.size
                    else:
                        # 开多
                        self.positions[symbol] = BacktestPosition(
                            symbol=symbol,
                            side="long",
                            size=qty,
                            entry_price=fill_price,
                            leverage=leverage or 1
                        )
                    
                    # 扣除手续费和保证金
                    self.base_equity -= fee
            else:  # sell
                # 卖出：减少多头持仓或增加空头持仓
                if current_pos and current_pos.side == "long":
                    # 平多（收取平仓手续费0.6块）
                    close_fee = FIXED_FEE
                    if qty >= current_pos.size:
                        # 完全平仓，可能还要开空
                        close_qty = current_pos.size
                        remaining_qty = qty - close_qty
                        
                        # 计算平仓盈亏（扣除平仓手续费）
                        pnl = (fill_price - current_pos.entry_price) * close_qty - close_fee
                        self.base_equity += pnl
                        
                        # 移除或更新持仓
                        if remaining_qty > 0:
                            # 开空（收取开仓手续费0.6块）
                            open_fee = FIXED_FEE
                            self.base_equity -= open_fee
                            self.positions[symbol] = BacktestPosition(
                                symbol=symbol,
                                side="short",
                                size=remaining_qty,
                                entry_price=fill_price,
                                leverage=leverage or 1
                            )
                            fee = close_fee + open_fee  # 总手续费 = 平仓 + 开仓
                        else:
                            del self.positions[symbol]
                            fee = close_fee  # 只有平仓手续费
                    else:
                        # 部分平多（只收取平仓手续费0.6块）
                        fee = FIXED_FEE
                        pnl = (fill_price - current_pos.entry_price) * qty - fee
                        self.base_equity += pnl
                        current_pos.size -= qty
                else:
                    # 开空或加空（收取开仓手续费0.6块）
                    fee = FIXED_FEE
                    if current_pos and current_pos.side == "short":
                        # 加空（简单平均成本）
                        total_value = current_pos.size * current_pos.entry_price + qty * fill_price
                        current_pos.size += qty
                        current_pos.entry_price = total_value / current_pos.size
                    else:
                        # 开空
                        self.positions[symbol] = BacktestPosition(
                            symbol=symbol,
                            side="short",
                            size=qty,
                            entry_price=fill_price,
                            leverage=leverage or 1
                        )
                    
                    # 扣除手续费和保证金
                    self.base_equity -= fee
            
            # 创建订单记录
            self._order_id_counter += 1
            order = BacktestOrder(
                order_id=f"backtest_{self._order_id_counter}",
                symbol=symbol,
                side=side,
                qty=qty,
                price=fill_price,
                order_type=order_type,
                status="filled",
                filled_qty=qty,
                filled_price=fill_price,
                timestamp=self.current_time,
                fee=fee
            )
            self.orders.append(order)
            
            # 记录交易历史
            self.trades_history.append({
                "time": self.current_time,
                "order_id": order.order_id,
                "symbol": symbol,
                "side": side,
                "qty": qty,
                "price": fill_price,
                "fee": fee,
                "equity_after": self.equity
            })
            
            # 更新权益
            await self._update_equity()
            
            # 构建返回消息（模拟在线环境的格式）
            result_text = f"Order {order.order_id} filled successfully"
            
            # 添加止损/止盈信息（如果提供）
            order_info = {
                "order_id": order.order_id,
                "symbol": symbol,
                "side": side,
                "quantity": str(qty),
                "order_status": "filled",
                "order_type": order_type,
            }
            
            # 在回测环境中，止损和止盈只是记录，不自动执行
            if stop_loss_price is not None:
                order_info["stop_loss_price"] = str(stop_loss_price)
                result_text += f"\nStop Loss Price: {stop_loss_price} (recorded, not auto-executed in backtest)"
            
            if take_profit_price is not None:
                order_info["take_profit_price"] = str(take_profit_price)
                result_text += f"\nTake Profit Price: {take_profit_price} (recorded, not auto-executed in backtest)"
            
            return {
                "success": True,
                "message": result_text,
                "extra": {
                    "order_info": order_info,
                    "order_id": order.order_id,
                    "symbol": symbol,
                    "side": side,
                    "filled_qty": qty,
                    "filled_price": fill_price,
                    "fee": fee,
                    "equity_after": self.equity
                }
            }
        except Exception as e:
            logger.error(f"Error creating order: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Failed to create order: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def close_order(
        self,
        symbol: str,
        side: str,
        size: float,
        order_type: str = "Market",
        price: Optional[float] = None
    ) -> Dict[str, Any]:
        """平仓订单（模拟 close_order，与在线环境一致）"""
        from src.utils import dedent
        
        try:
            # 回测环境中，所有交易都成功
            # 如果 symbol 不匹配，自动使用 self.symbol
            if symbol != self.symbol:
                symbol = self.symbol
            
            # 获取当前持仓
            current_pos = self.positions.get(symbol)
            
            # 如果没有持仓，创建一个虚拟持仓用于平仓（回测环境中总是成功）
            if not current_pos:
                # 根据 side 推断持仓方向
                if side.lower() in ["sell", "s"]:
                    # 平多，创建虚拟多头持仓
                    current_pos = BacktestPosition(
                        symbol=symbol,
                        side="long",
                        size=size if size > 0 else 0.01,
                        entry_price=self.current_price,
                        leverage=1
                    )
                else:  # buy
                    # 平空，创建虚拟空头持仓
                    current_pos = BacktestPosition(
                        symbol=symbol,
                        side="short",
                        size=size if size > 0 else 0.01,
                        entry_price=self.current_price,
                        leverage=1
                    )
                # 不添加到 positions，因为这是虚拟持仓
            
            # 确定平仓方向（自动修正）
            # LONG 持仓需要 SELL 来平仓
            # SHORT 持仓需要 BUY 来平仓
            if current_pos.side == "long" and side.lower() not in ["sell", "s"]:
                side = "sell"  # 自动修正为正确的平仓方向
            if current_pos.side == "short" and side.lower() not in ["buy", "b"]:
                side = "buy"  # 自动修正为正确的平仓方向
            
            # 确定平仓数量
            close_qty = min(size, current_pos.size) if size > 0 else current_pos.size
            
            # 确定执行价格（考虑滑点）
            slippage = self.slippage_bps / 10000.0
            if order_type == "Market":
                if side.lower() in ["sell", "s"]:
                    fill_price = self.current_price * (1 - slippage)  # 卖出时价格略低
                else:  # buy
                    fill_price = self.current_price * (1 + slippage)  # 买入时价格略高
            else:  # Limit
                fill_price = price if price else self.current_price
            
            # 计算手续费（固定0.6块，平仓时收取）
            FIXED_FEE = 0.6  # 固定手续费
            fee = FIXED_FEE
            
            # 计算平仓盈亏（扣除平仓手续费）
            if current_pos.side == "long":
                pnl = (fill_price - current_pos.entry_price) * close_qty - fee
            else:  # short
                pnl = (current_pos.entry_price - fill_price) * close_qty - fee
            
            # 更新基础权益（已实现盈亏）
            self.base_equity += pnl
            
            # 更新持仓（只有真实持仓才更新）
            if symbol in self.positions:
                if close_qty >= self.positions[symbol].size:
                    # 完全平仓
                    del self.positions[symbol]
                else:
                    # 部分平仓
                    self.positions[symbol].size -= close_qty
            
            # 创建订单记录
            self._order_id_counter += 1
            order = BacktestOrder(
                order_id=f"backtest_close_{self._order_id_counter}",
                symbol=symbol,
                side=side,
                qty=close_qty,
                price=fill_price,
                order_type=order_type,
                status="filled",
                filled_qty=close_qty,
                filled_price=fill_price,
                timestamp=self.current_time,
                fee=fee
            )
            self.orders.append(order)
            
            # 记录交易历史
            self.trades_history.append({
                "time": self.current_time,
                "order_id": order.order_id,
                "symbol": symbol,
                "side": side,
                "qty": close_qty,
                "price": fill_price,
                "fee": fee,
                "equity_after": self.equity
            })
            
            # 更新权益
            await self._update_equity()
            
            result_text = dedent(f"""
                Close order submitted successfully:
                Order ID: {order.order_id}
                Symbol: {symbol}
                Side: {side}
                Quantity: {close_qty}
                Status: {order.status}
                Order Type: {order_type}
                Trade Type: perpetual
                Price: {fill_price}
                """)
            
            return {
                "success": True,
                "message": result_text,
                "extra": {
                    "close_order": {
                        "order_id": order.order_id,
                        "symbol": symbol,
                        "side": side,
                        "quantity": str(close_qty),
                        "status": order.status,
                        "type": order_type,
                        "trade_type": "perpetual",
                        "price": str(fill_price) if fill_price else None
                    },
                    "order_id": order.order_id,
                    "symbol": symbol,
                    "side": side,
                    "filled_qty": close_qty,
                    "filled_price": fill_price,
                    "fee": fee,
                    "equity_after": self.equity
                }
            }
        except Exception as e:
            logger.error(f"Error closing order: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Failed to close order: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def get_orders(
        self,
        trade_type: Optional[str] = None,
        symbol: Optional[str] = None,
        limit: Optional[int] = None,
        order_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取订单列表（从本地维护的订单列表中获取，与在线环境一致）"""
        from src.utils import dedent
        
        try:
            # 从本地订单列表中筛选
            filtered_orders = self.orders.copy()
            
            # 按 symbol 筛选
            if symbol:
                filtered_orders = [o for o in filtered_orders if o.symbol == symbol]
            
            # 按 order_id 筛选
            if order_id:
                filtered_orders = [o for o in filtered_orders if o.order_id == order_id]
            
            # 限制数量
            if limit:
                filtered_orders = filtered_orders[-limit:]  # 取最新的 N 条
            
            # 转换为在线环境的格式
            orders_list = []
            for order in filtered_orders:
                orders_list.append({
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": str(order.qty),
                    "price": str(order.price) if order.price else None,
                    "status": order.status,
                    "type": order.order_type,
                    "trade_type": "perpetual"
                })
            
            # 构建返回消息
            if len(orders_list) == 0:
                result_text = f"No perpetual orders found."
            else:
                order_lines = []
                for order in orders_list:
                    qty_display = order.get("quantity", "N/A")
                    order_lines.append(
                        f"  {order['symbol']}: {order['side']} {qty_display} "
                        f"(Status: {order['status']}, Trade Type: {order.get('trade_type', 'N/A')})"
                    )
                
                result_text = dedent(f"""
                    {len(orders_list)} order(s) found:
                    {chr(10).join(order_lines)}
                    """)
            
            return {
                "success": True,
                "message": result_text,
                "extra": {
                    "orders": orders_list
                }
            }
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return {
                "success": False,
                "message": f"Failed to get orders: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def get_order(
        self,
        order_id: str,
        symbol: str,
        trade_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取单个订单信息（从本地维护的订单列表中获取，与在线环境一致）"""
        from src.utils import dedent
        
        try:
            # 查找订单
            order = None
            for o in self.orders:
                if o.order_id == order_id and o.symbol == symbol:
                    order = o
                    break
            
            if not order:
                return {
                    "success": False,
                    "message": f"Order {order_id} not found for {symbol}",
                    "extra": {"error": "Order not found"}
                }
            
            # 转换为在线环境的格式
            order_dict = {
                "order_id": order.order_id,
                "symbol": order.symbol,
                "side": order.side,
                "quantity": str(order.qty),
                "status": order.status,
                "type": order.order_type,
                "trade_type": "perpetual"
            }
            
            if order.price:
                order_dict["price"] = str(order.price)
            
            qty_display = order_dict.get("quantity", "N/A")
            
            result_text = dedent(f"""
                Order Information:
                Order ID: {order.order_id}
                Symbol: {order.symbol}
                Side: {order.side}
                Quantity: {qty_display}
                Status: {order.status}
                Order Type: {order.order_type}
                Trade Type: perpetual
                """)
            
            return {
                "success": True,
                "message": result_text,
                "extra": {
                    "order": order_dict
                }
            }
        except Exception as e:
            logger.error(f"Error getting order: {e}")
            return {
                "success": False,
                "message": f"Failed to get order: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def cancel_order(
        self,
        order_id: str,
        symbol: str,
        trade_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """取消订单（回测环境中订单立即成交，所以取消操作主要用于兼容性）"""
        try:
            # 查找订单
            order = None
            for o in self.orders:
                if o.order_id == order_id and o.symbol == symbol:
                    order = o
                    break
            
            if not order:
                return {
                    "success": False,
                    "message": f"Order {order_id} not found for {symbol}",
                    "extra": {"error": "Order not found"}
                }
            
            # 在回测环境中，订单通常已经成交，所以取消操作主要用于兼容性
            # 如果订单状态是 "filled"，则无法取消
            if order.status == "filled":
                return {
                    "success": False,
                    "message": f"Cannot cancel filled order {order_id}",
                    "extra": {"error": "Order already filled"}
                }
            
            # 更新订单状态
            order.status = "cancelled"
            
            return {
                "success": True,
                "message": f"Order {order_id} cancelled successfully",
                "extra": {
                    "order_id": order_id,
                    "symbol": symbol,
                    "status": "cancelled"
                }
            }
        except Exception as e:
            logger.error(f"Error cancelling order: {e}")
            return {
                "success": False,
                "message": f"Failed to cancel order: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def cancel_all_orders(
        self,
        symbol: Optional[str] = None,
        trade_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """取消所有订单（回测环境中订单立即成交，所以取消操作主要用于兼容性）"""
        try:
            # 查找待取消的订单（非成交状态）
            cancelled_count = 0
            for order in self.orders:
                if order.status != "filled":
                    if symbol is None or order.symbol == symbol:
                        order.status = "cancelled"
                        cancelled_count += 1
            
            if cancelled_count == 0:
                message = "No pending orders to cancel."
            else:
                message = f"Cancelled {cancelled_count} order(s)."
            
            return {
                "success": True,
                "message": message,
                "extra": {
                    "cancelled_count": cancelled_count,
                    "symbol": symbol
                }
            }
        except Exception as e:
            logger.error(f"Error cancelling all orders: {e}")
            return {
                "success": False,
                "message": f"Failed to cancel all orders: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def get_assets(self, status: Optional[str] = None, asset_class: Optional[str] = None) -> Dict[str, Any]:
        """获取所有资产信息（模拟 HyperliquidEnvironment 的接口）"""
        try:
            # 回测环境中只支持当前交易的币种
            symbols = [{"symbol": self.symbol}]
            
            result_text = dedent(f"""
                {len(symbols)} symbol(s) available for backtest:
                {", ".join([symbol["symbol"] for symbol in symbols])}
                """)
            
            return {
                "success": True,
                "message": result_text,
                "extra": {
                    "symbols": symbols,
                    "exchange_info": {
                        "universe": [{"name": self.symbol}]
                    }
                }
            }
        except Exception as e:
            logger.error(f"Error getting assets: {e}")
            return {
                "success": False,
                "message": f"Failed to get assets information: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    @ecp.action(name="step",
                type="Hyperliquid Trading",
                description=dedent("""
                    Step the trading environment for perpetual futures trading.
                    Example:
                        - SHORT: {"symbol": "BTC", "action": "SHORT", "qty": 0.01, "leverage": 10, "stop_loss_price": 100000, "take_profit_price": 110000}
                        - LONG: {"symbol": "BTC", "action": "LONG", "qty": 0.01, "leverage": 10, "stop_loss_price": 90000, "take_profit_price": 110000}
                        - CLOSE_LONG: {"symbol": "BTC", "action": "CLOSE_LONG"}
                        - CLOSE_SHORT: {"symbol": "BTC", "action": "CLOSE_SHORT"}
                        - HOLD: {"symbol": "BTC", "action": "HOLD"}
                    """)
                )
    async def step(
        self,
        symbol: str = "BTC",
        action: str = "HOLD",  # LONG, SHORT, CLOSE_LONG, CLOSE_SHORT, HOLD
        qty: float = 0.0,
        leverage: Optional[int] = 10,
        stop_loss_price: Optional[float] = None,
        take_profit_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Step the trading environment for perpetual futures trading (回测版本).
        
        Hyperliquid Perpetual Futures Trading Rules:
        ┌──────────────┬─────────────────────────────────────────────────────────────┐
        │ Action       │ Description                                                 │
        ├──────────────┼─────────────────────────────────────────────────────────────┤
        │ LONG         │ Open long position (stop loss & take profit recorded only) │
        │ CLOSE_LONG   │ Close long position (market order)                         │
        │ SHORT        │ Open short position (stop loss & take profit recorded only)│
        │ CLOSE_SHORT  │ Close short position (market order)                        │
        │ HOLD         │ Do nothing                                                  │
        └──────────────┴─────────────────────────────────────────────────────────────┘
        
        Note: In backtest environment, stop_loss_price and take_profit_price are recorded
        but not automatically executed (unlike online environment where exchange handles them).
        
        Args:
            symbol (str): Symbol to trade (e.g., 'BTC', 'ETH')
            action (str): Trading action for perpetual futures:
                - 'LONG': Open long position (with optional stop loss & take profit)
                - 'SHORT': Open short position (with optional stop loss & take profit)
                - 'CLOSE_LONG': Close long position (market order)
                - 'CLOSE_SHORT': Close short position (market order)
                - 'HOLD': Do nothing (default)
            qty (float): Quantity to trade.
            leverage (Optional[int]): Leverage for perpetual futures.
            stop_loss_price (Optional[float]): Stop loss trigger price (recorded only in backtest)
            take_profit_price (Optional[float]): Take profit trigger price (recorded only in backtest)
        
        Returns:
            Dictionary with success, message, and order information
        """
        action = action.upper()
        try:
            if action == "HOLD":
                result = {
                    "success": True,
                    "message": "HOLD action performed successfully. No order submitted.",
                    "extra": {}
                }
            elif action == "LONG":
                # Open long position: BUY
                result = await self.create_order(
                    symbol=symbol,
                    side="buy",
                    qty=qty,
                    order_type="Market",
                    leverage=leverage,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price
                )
            elif action == "SHORT":
                # Open short position: SELL
                result = await self.create_order(
                    symbol=symbol,
                    side="sell",
                    qty=qty,
                    order_type="Market",
                    leverage=leverage,
                    stop_loss_price=stop_loss_price,
                    take_profit_price=take_profit_price
                )
            elif action == "CLOSE_LONG":
                # Close long position: SELL to close LONG position
                result = await self.close_order(
                    symbol=symbol,
                    side="sell",  # SELL to close LONG position
                    size=qty,
                    order_type="Market"
                )
            elif action == "CLOSE_SHORT":
                # Close short position: BUY to close SHORT position
                result = await self.close_order(
                    symbol=symbol,
                    side="buy",  # BUY to close SHORT position
                    size=qty,
                    order_type="Market"
                )
            else:
                result = {
                    "success": False,
                    "message": f"Invalid action: {action}. Must be LONG, SHORT, CLOSE_LONG, CLOSE_SHORT, or HOLD",
                    "extra": {"error": f"Invalid action: {action}"}
                }
            
            # Get account information and calculate metrics (模拟在线环境)
            account_result = await self.get_account()
            if account_result.get("success"):
                result["extra"].update(account_result.get("extra", {}))
            
            return result
        
        except Exception as e:
            logger.error(f"Error in step: {e}")
            import traceback
            traceback.print_exc()
            return {
                "success": False,
                "message": f"Failed to step the trading environment: {str(e)}",
                "extra": {"error": str(e)}
            }
    
    async def cleanup(self) -> None:
        """清理回测环境"""
        logger.info("| 🧹 回测环境清理完成")
    
    def get_backtest_results(self) -> Dict[str, Any]:
        """获取回测结果统计"""
        if len(self.equity_curve) == 0:
            return {}
        
        final_equity = self.equity_curve[-1]["equity"]
        total_return = (final_equity / self.initial_equity - 1) * 100
        
        return {
            "initial_equity": self.initial_equity,
            "final_equity": final_equity,
            "total_return": total_return,
            "max_drawdown": self.max_drawdown * 100,
            "total_trades": len(self.trades_history),
            "total_orders": len(self.orders),
            "equity_curve": self.equity_curve,
            "trades_history": self.trades_history
        }


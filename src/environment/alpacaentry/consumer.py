"""Data consumer: reads data from database."""
from typing import Optional, List, Dict

from src.logger import logger
from src.environment.protocol.types import ActionResult
from src.environment.alpacaentry.bars import BarsHandler
from src.environment.alpacaentry.quotes import QuotesHandler
from src.environment.alpacaentry.trades import TradesHandler
from src.environment.alpacaentry.orderbooks import OrderbooksHandler
from src.environment.alpacaentry.news import NewsHandler
from src.environment.alpacaentry.types import DataStreamType, GetDataRequest
from src.environment.alpacaentry.exceptions import AlpacaError


class DataConsumer:
    """Consumer: reads data from database."""
    
    def __init__(
        self,
        bars_handler: BarsHandler,
        quotes_handler: QuotesHandler,
        trades_handler: TradesHandler,
        orderbooks_handler: OrderbooksHandler,
        news_handler: NewsHandler,
    ):
        """Initialize data consumer.
        
        Args:
            bars_handler: Bars data handler
            quotes_handler: Quotes data handler
            trades_handler: Trades data handler
            orderbooks_handler: Orderbooks data handler
            news_handler: News data handler
        """
        self._bars_handler = bars_handler
        self._quotes_handler = quotes_handler
        self._trades_handler = trades_handler
        self._orderbooks_handler = orderbooks_handler
        self._news_handler = news_handler
    
    async def _get_data_from_handler(
        self, 
        symbol: str, 
        data_type: DataStreamType, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None, 
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Helper method to get data from handler."""
        if data_type == DataStreamType.QUOTES:
            return await self._quotes_handler.get_data(symbol, start_date, end_date, limit)
        elif data_type == DataStreamType.TRADES:
            return await self._trades_handler.get_data(symbol, start_date, end_date, limit)
        elif data_type == DataStreamType.BARS:
            return await self._bars_handler.get_data(symbol, start_date, end_date, limit)
        elif data_type == DataStreamType.ORDERBOOKS:
            return await self._orderbooks_handler.get_data(symbol, start_date, end_date, limit)
        elif data_type == DataStreamType.NEWS:
            return await self._news_handler.get_data(symbol, start_date, end_date, limit)
        else:
            raise ValueError(f"Invalid data type: {data_type}")
    
    async def get_data(self, request: GetDataRequest) -> ActionResult:
        """Get historical data from database.
        
        Args:
            request: GetDataRequest with symbol (str or list), data_type (str or list),
                    optional start_date, end_date, and limit
            
        Returns:
            ActionResult with data organized by symbol in extra field
        """
        try:
            # Normalize symbol and data_type to lists
            symbols = request.symbol if isinstance(request.symbol, list) else [request.symbol]
            data_types = request.data_type if isinstance(request.data_type, list) else [request.data_type]
            data_types = [DataStreamType(data_type) for data_type in data_types]
            
            # Organize data by symbol
            result_data: Dict[str, Dict[str, List[Dict]]] = {}
            total_rows = 0
            
            # Get data for each symbol and data_type combination
            for symbol in symbols:
                result_data[symbol] = {}
                
                for data_type in data_types:
                    logger.info(f"| 🔍 Getting {data_type.value} data for {symbol}...")
                    data = await self._get_data_from_handler(
                        symbol=symbol,
                        data_type=data_type,
                        start_date=request.start_date,
                        end_date=request.end_date,
                        limit=request.limit
                    )
                    result_data[symbol][data_type.value] = data
                    total_rows += len(data)
                    logger.info(f"| ✅ Retrieved {len(data)} {data_type.value} records for {symbol}")
            
            # Build message
            symbol_str = ", ".join(symbols) if len(symbols) <= 10 else f"{len(symbols)} symbols"
            data_type_str = ", ".join([datatype.value for datatype in data_types]) if len(data_types) <= 10 else f"{len(data_types)} types"
            
            if request.start_date and request.end_date:
                message = f"Retrieved {total_rows} records ({data_type_str}) for {symbol_str} from {request.start_date} to {request.end_date}."
            else:
                message = f"Retrieved {total_rows} latest records ({data_type_str}) for {symbol_str}."
            
            return ActionResult(
                success=True,
                message=message,
                extra={
                    "data": result_data,
                    "symbols": symbols,
                    "data_types": data_types,
                    "start_date": request.start_date,
                    "end_date": request.end_date,
                    "row_count": total_rows
                }
            )
            
        except Exception as e:
            raise AlpacaError(f"Failed to get data: {e}.")


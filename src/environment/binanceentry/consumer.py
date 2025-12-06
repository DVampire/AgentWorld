"""Data consumer: reads data from database."""
from typing import Optional, List, Dict

from src.logger import logger
from src.environment.protocol.types import ActionResult
from src.environment.binanceentry.klines import KlinesHandler
from src.environment.binanceentry.types import DataStreamType, GetDataRequest
from src.environment.binanceentry.exceptions import BinanceError


class DataConsumer:
    """Consumer: reads data from database."""
    
    def __init__(
        self,
        klines_handler: KlinesHandler,
    ):
        """Initialize data consumer.
        
        Args:
            klines_handler: Klines data handler
        """
        self._klines_handler = klines_handler
    
    async def _get_data_from_handler(
        self, 
        symbol: str, 
        data_type: DataStreamType, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None, 
        limit: Optional[int] = None
    ) -> List[Dict]:
        """Helper method to get data from handler."""
        if data_type == DataStreamType.KLINES:
            return await self._klines_handler.get_data(symbol, start_date, end_date, limit)
        else:
            raise ValueError(f"Invalid data type: {data_type}")
    
    async def get_data(self, request: GetDataRequest) -> ActionResult:
        """Get historical data from database.
        
        Args:
            request: GetDataRequest with symbol (str or list), data_type,
                    optional start_date, end_date, and limit
            
        Returns:
            ActionResult with data organized by symbol in extra field
        """
        try:
            # Normalize symbol and data_type to lists
            symbols = request.symbol if isinstance(request.symbol, list) else [request.symbol]
            data_type = DataStreamType(request.data_type)
            
            # Organize data by symbol
            result_data: Dict[str, Dict[str, List[Dict]]] = {}
            total_rows = 0
            
            # Get data for each symbol
            for symbol in symbols:
                result_data[symbol] = {}
                
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
            
            if request.start_date and request.end_date:
                message = f"Retrieved {total_rows} records ({data_type.value}) for {symbol_str} from {request.start_date} to {request.end_date}."
            else:
                message = f"Retrieved {total_rows} latest records ({data_type.value}) for {symbol_str}."
            
            return ActionResult(
                success=True,
                message=message,
                extra={
                    "data": result_data,
                    "symbols": symbols,
                    "data_type": data_type.value,
                    "start_date": request.start_date,
                    "end_date": request.end_date,
                    "row_count": total_rows
                }
            )
            
        except Exception as e:
            raise BinanceError(f"Failed to get data: {e}.")


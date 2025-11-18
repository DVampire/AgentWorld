"""Data consumer: reads data from database."""
from typing import Optional, List, Dict

from src.logger import logger
from src.environments.protocol.types import ActionResult
from src.environments.hyperliquidentry.candle import CandleHandler
from src.environments.hyperliquidentry.types import DataStreamType, GetDataRequest
from src.environments.hyperliquidentry.exceptions import HyperliquidError


class DataConsumer:
    """Consumer: reads data from database."""
    
    def __init__(
        self,
        candle_handler: CandleHandler,
    ):
        """Initialize data consumer.
        
        Args:
            candle_handler: Candle data handler
        """
        self._candle_handler = candle_handler
    
    async def _get_data_from_handler(
        self, 
        symbol: str, 
        data_type: DataStreamType, 
        start_date: Optional[str] = None, 
        end_date: Optional[str] = None, 
        limit: Optional[int] = None
    ) -> Dict[str, List[Dict]]:
        """Helper method to get data from handler.
        
        Returns:
            For CANDLE: Dict with 'candles' and 'indicators' keys, each containing a list of records
        """
        if data_type == DataStreamType.CANDLE:
            return await self._candle_handler.get_data(symbol, start_date, end_date, limit)
        else:
            raise ValueError(f"Invalid data type: {data_type}. Only CANDLE is supported.")
    
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
                logger.info(f"| 🔍 Getting {data_type.value} data for {symbol}...")
                data = await self._get_data_from_handler(
                    symbol=symbol,
                    data_type=data_type,
                    start_date=request.start_date,
                    end_date=request.end_date,
                    limit=request.limit
                )
                
                # For CANDLE, data is a dict with 'candles' and 'indicators'
                if data_type == DataStreamType.CANDLE:
                    result_data[symbol] = data  # Already contains 'candles' and 'indicators'
                    total_rows += len(data.get('candles', [])) + len(data.get('indicators', []))
                else:
                    result_data[symbol] = {data_type.value: data}
                    total_rows += len(data) if isinstance(data, list) else 0
            
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
            raise HyperliquidError(f"Failed to get data: {e}.")


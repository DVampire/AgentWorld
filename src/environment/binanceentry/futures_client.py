"""Binance Futures REST API client implementation."""
import time
import hmac
import hashlib
import urllib.parse
from typing import Dict, Optional, Any, List
import requests
from src.logger import logger


class BinanceFuturesClient:
    """Binance Futures REST API client using direct HTTP requests."""
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False
    ):
        """Initialize Binance Futures client.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret key
            testnet: Whether to use testnet (True) or live (False)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Binance Futures REST API base URLs
        # Live: https://fapi.binance.com
        # Testnet: https://testnet.binancefuture.com
        self.base_url = "https://testnet.binancefuture.com" if testnet else "https://fapi.binance.com"
    
    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """Generate HMAC SHA256 signature for authenticated requests.
        
        Args:
            params: Request parameters dictionary
            
        Returns:
            Signature string
        """
        query_string = urllib.parse.urlencode(params)
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _get_headers(self) -> Dict[str, str]:
        """Get request headers with API key.
        
        Returns:
            Headers dictionary
        """
        return {
            'X-MBX-APIKEY': self.api_key,
            'Content-Type': 'application/json'
        }
    
    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        signed: bool = False
    ) -> Dict[str, Any]:
        """Make HTTP request to Binance Futures API.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint (e.g., '/fapi/v2/account')
            params: Request parameters
            signed: Whether this is a signed request (requires signature)
            
        Returns:
            Response data as dictionary
        """
        if params is None:
            params = {}
        
        url = f"{self.base_url}{endpoint}"
        
        # Add timestamp and signature for signed requests
        if signed:
            params['timestamp'] = int(time.time() * 1000)
            signature = self._generate_signature(params)
            params['signature'] = signature
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, params=params, headers=self._get_headers(), timeout=10)
            elif method.upper() == 'POST':
                response = requests.post(url, params=params, headers=self._get_headers(), timeout=10)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, params=params, headers=self._get_headers(), timeout=10)
            elif method.upper() == 'PUT':
                response = requests.put(url, params=params, headers=self._get_headers(), timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            if response.status_code == 401:
                raise Exception(f"(401, -2015, 'Invalid API-key, IP, or permissions for action.', {dict(response.headers)}, None)")
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def sign_request(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Sign and make a request to Binance Futures API.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint (e.g., '/fapi/v2/account')
            payload: Request payload/parameters
            
        Returns:
            Response data as dictionary
        """
        return self._request(method, endpoint, params=payload or {}, signed=True)
    
    def get_account(self) -> Dict[str, Any]:
        """Get futures account information.
        
        Returns:
            Account information dictionary
        """
        return self._request('GET', '/fapi/v2/account', signed=True)
    
    def get_position_risk(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get position risk information.
        
        Args:
            symbol: Optional trading symbol to filter by
            
        Returns:
            List of position risk dictionaries
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        
        return self._request('GET', '/fapi/v2/positionRisk', params=params, signed=True)
    
    def get_position_mode(self) -> Dict[str, Any]:
        """Get current position mode (one-way or hedge mode).
        
        Returns:
            Result dictionary with 'dualSidePosition' field (True for hedge mode, False for one-way mode)
        """
        return self._request('GET', '/fapi/v1/positionSide/dual', params={}, signed=True)
    
    def set_position_mode(self, dual_side_position: bool) -> Dict[str, Any]:
        """Set position mode.
        
        Args:
            dual_side_position: True for hedge mode (allows both LONG and SHORT positions), 
                               False for one-way mode (net position only)
            
        Returns:
            Result dictionary
        """
        params = {
            'dualSidePosition': 'true' if dual_side_position else 'false'
        }
        return self._request('POST', '/fapi/v1/positionSide/dual', params=params, signed=True)
    
    def set_leverage(self, symbol: str, leverage: int) -> Dict[str, Any]:
        """Set leverage for a symbol.
        
        Args:
            symbol: Trading symbol
            leverage: Leverage value
            
        Returns:
            Result dictionary
        """
        params = {
            'symbol': symbol,
            'leverage': leverage
        }
        return self._request('POST', '/fapi/v1/leverage', params=params, signed=True)
    
    def create_order(
        self,
        symbol: str,
        side: str,
        type: str,
        quantity: str,
        price: Optional[str] = None,
        timeInForce: Optional[str] = None,
        positionSide: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new futures order.
        
        Args:
            symbol: Trading symbol
            side: Order side ('BUY' or 'SELL')
            type: Order type ('MARKET', 'LIMIT', etc.)
            quantity: Order quantity
            price: Order price (required for LIMIT orders)
            timeInForce: Time in force (required for LIMIT orders)
            positionSide: Position side ('LONG', 'SHORT') - only used in hedge mode
            **kwargs: Additional order parameters
            
        Returns:
            Order information dictionary
            
        Note:
            - If account is in one-way mode, positionSide should NOT be specified
            - If account is in hedge mode, positionSide MUST be specified ('LONG' or 'SHORT')
        """
        params = {
            'symbol': symbol,
            'side': side,
            'type': type,
            'quantity': quantity,
            **kwargs
        }
        
        if price:
            params['price'] = price
        if timeInForce:
            params['timeInForce'] = timeInForce
        
        # Check position mode and handle positionSide accordingly
        try:
            position_mode = self.get_position_mode()
            is_hedge_mode = position_mode.get('dualSidePosition', False)
            
            if is_hedge_mode:
                # Hedge mode: positionSide is required
                if positionSide:
                    params['positionSide'] = positionSide
                else:
                    logger.warning(f"| ⚠️  Hedge mode detected but positionSide not specified. Defaulting to 'BOTH'")
                    # In hedge mode, if not specified, we might need to determine based on side
                    # But typically, positionSide should be explicitly provided
            else:
                # One-way mode: positionSide should NOT be specified
                if positionSide:
                    logger.warning(f"| ⚠️  One-way mode detected but positionSide specified ({positionSide}). Removing positionSide parameter.")
                    # Don't add positionSide in one-way mode
        except Exception as e:
            # If we can't get position mode, use the provided positionSide (if any)
            # This allows the API to return the error if it's wrong
            logger.warning(f"| ⚠️  Could not get position mode: {e}. Using provided positionSide: {positionSide}")
            if positionSide:
                params['positionSide'] = positionSide
        
        return self._request('POST', '/fapi/v1/order', params=params, signed=True)
    
    def get_order(
        self,
        symbol: str,
        orderId: Optional[str] = None,
        origClientOrderId: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get order status.
        
        Args:
            symbol: Trading symbol
            orderId: Order ID
            origClientOrderId: Original client order ID
            
        Returns:
            Order information dictionary
        """
        params = {'symbol': symbol}
        if orderId:
            params['orderId'] = orderId
        if origClientOrderId:
            params['origClientOrderId'] = origClientOrderId
        
        return self._request('GET', '/fapi/v1/order', params=params, signed=True)
    
    def get_all_orders(
        self,
        symbol: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """Get all orders for a symbol.
        
        Args:
            symbol: Trading symbol (optional)
            **kwargs: Additional query parameters
            
        Returns:
            List of order dictionaries
        """
        params = {}
        if symbol:
            params['symbol'] = symbol
        params.update(kwargs)
        
        return self._request('GET', '/fapi/v1/allOrders', params=params, signed=True)
    
    def cancel_order(
        self,
        symbol: str,
        orderId: Optional[str] = None,
        origClientOrderId: Optional[str] = None
    ) -> Dict[str, Any]:
        """Cancel an order.
        
        Args:
            symbol: Trading symbol
            orderId: Order ID
            origClientOrderId: Original client order ID
            
        Returns:
            Cancellation result dictionary
        """
        params = {'symbol': symbol}
        if orderId:
            params['orderId'] = orderId
        if origClientOrderId:
            params['origClientOrderId'] = origClientOrderId
        
        return self._request('DELETE', '/fapi/v1/order', params=params, signed=True)
    
    def cancel_all_open_orders(self, symbol: str) -> Dict[str, Any]:
        """Cancel all open orders for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Cancellation result dictionary
        """
        params = {'symbol': symbol}
        return self._request('DELETE', '/fapi/v1/allOpenOrders', params=params, signed=True)


"""Binance Spot REST API client implementation."""
import time
import hmac
import hashlib
import urllib.parse
from typing import Dict, Optional, Any, List
import requests
from src.logger import logger


class BinanceSpotClient:
    """Binance Spot REST API client using direct HTTP requests."""
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        testnet: bool = False
    ):
        """Initialize Binance Spot client.
        
        Args:
            api_key: Binance API key
            api_secret: Binance API secret key
            testnet: Whether to use testnet (True) or live (False)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        
        # Binance Spot REST API base URLs
        # Live: https://api.binance.com
        # Testnet: https://testnet.binance.vision
        self.base_url = "https://testnet.binance.vision" if testnet else "https://api.binance.com"
    
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
        """Make HTTP request to Binance API.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint (e.g., '/api/v3/account')
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
    
    def account(self) -> Dict[str, Any]:
        """Get account information.
        
        Returns:
            Account information dictionary
        """
        return self._request('GET', '/api/v3/account', signed=True)
    
    def exchange_info(self) -> Dict[str, Any]:
        """Get exchange trading rules and symbol information.
        
        Returns:
            Exchange information dictionary
        """
        return self._request('GET', '/api/v3/exchangeInfo', signed=False)
    
    def new_order(
        self,
        symbol: str,
        side: str,
        type: str,
        quantity: Optional[str] = None,
        price: Optional[str] = None,
        timeInForce: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new order.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            side: Order side ('BUY' or 'SELL')
            type: Order type ('MARKET', 'LIMIT', etc.)
            quantity: Order quantity
            price: Order price (required for LIMIT orders)
            timeInForce: Time in force (required for LIMIT orders)
            **kwargs: Additional order parameters
            
        Returns:
            Order information dictionary
        """
        params = {
            'symbol': symbol,
            'side': side,
            'type': type,
            **kwargs
        }
        
        if quantity:
            params['quantity'] = quantity
        if price:
            params['price'] = price
        if timeInForce:
            params['timeInForce'] = timeInForce
        
        return self._request('POST', '/api/v3/order', params=params, signed=True)
    
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
        
        return self._request('GET', '/api/v3/order', params=params, signed=True)
    
    def get_orders(
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
        
        return self._request('GET', '/api/v3/allOrders', params=params, signed=True)
    
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
        
        return self._request('DELETE', '/api/v3/order', params=params, signed=True)
    
    def cancel_open_orders(self, symbol: str) -> Dict[str, Any]:
        """Cancel all open orders for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Cancellation result dictionary
        """
        params = {'symbol': symbol}
        return self._request('DELETE', '/api/v3/openOrders', params=params, signed=True)
    
    def sign_request(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Sign and make a request (for compatibility with existing code).
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            payload: Request payload/parameters
            
        Returns:
            Response data as dictionary
        """
        return self._request(method, endpoint, params=payload or {}, signed=True)


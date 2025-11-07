"""Hyperliquid REST API client implementation."""
import time
import json
from typing import Dict, Optional, Any, List
import requests
from src.logger import logger

# Hyperliquid uses wallet-based authentication
# We'll need to implement signature generation using wallet private key
try:
    from eth_account import Account
    from eth_account.messages import encode_defunct
    ETH_ACCOUNT_AVAILABLE = True
except ImportError:
    ETH_ACCOUNT_AVAILABLE = False
    logger.warning("| ⚠️  eth_account not available. Install with: pip install eth-account")


class HyperliquidClient:
    """Hyperliquid REST API client using direct HTTP requests."""
    
    def __init__(
        self,
        wallet_address: str,
        private_key: Optional[str] = None,
        testnet: bool = False
    ):
        """Initialize Hyperliquid client.
        
        Args:
            wallet_address: Hyperliquid wallet address
            private_key: Private key for signing requests (optional, can be provided later)
            testnet: Whether to use testnet (True) or mainnet (False)
        """
        self.wallet_address = wallet_address
        self.private_key = private_key
        self.testnet = testnet
        
        # Hyperliquid REST API base URLs
        # Mainnet: https://api.hyperliquid.xyz
        # Testnet: https://api.hyperliquid-testnet.xyz (or similar, need to verify)
        # Note: Hyperliquid may use the same API endpoint for both, with different authentication
        if testnet:
            # Testnet URL - may need to be verified
            self.base_url = "https://api.hyperliquid-testnet.xyz"
        else:
            self.base_url = "https://api.hyperliquid.xyz"
    
    def _sign_message(self, message: str) -> Optional[str]:
        """Sign a message using wallet private key.
        
        Args:
            message: Message to sign
            
        Returns:
            Signature string or None if signing fails
        """
        if not self.private_key:
            logger.warning("| ⚠️  No private key provided for signing")
            return None
        
        if not ETH_ACCOUNT_AVAILABLE:
            logger.error("| ❌ eth_account not available. Cannot sign messages.")
            return None
        
        try:
            account = Account.from_key(self.private_key)
            message_hash = encode_defunct(text=message)
            signed_message = account.sign_message(message_hash)
            return signed_message.signature.hex()
        except Exception as e:
            logger.error(f"| ❌ Failed to sign message: {e}")
            return None
    
    def _get_headers(self, signed: bool = False) -> Dict[str, str]:
        """Get request headers.
        
        Args:
            signed: Whether this is a signed request
            
        Returns:
            Headers dictionary
        """
        headers = {
            'Content-Type': 'application/json'
        }
        
        if signed and self.wallet_address:
            headers['X-Hyperliquid-Address'] = self.wallet_address
        
        return headers
    
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        signed: bool = False
    ) -> Dict[str, Any]:
        """Make HTTP request to Hyperliquid API.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint (e.g., '/info')
            data: Request data/payload
            signed: Whether this is a signed request (requires signature)
            
        Returns:
            Response data as dictionary
        """
        url = f"{self.base_url}{endpoint}"
        
        # For signed requests, we may need to include signature in the payload
        payload = data or {}
        
        if signed and self.private_key:
            # Hyperliquid uses wallet-based authentication
            # The signature is typically included in the request body
            # This is a placeholder - actual implementation depends on Hyperliquid's API spec
            timestamp = int(time.time() * 1000)
            message = json.dumps(payload) if payload else ""
            signature = self._sign_message(message)
            
            if signature:
                payload['signature'] = signature
                payload['timestamp'] = timestamp
        
        try:
            if method.upper() == 'GET':
                response = requests.get(url, params=payload, headers=self._get_headers(signed), timeout=10)
            elif method.upper() == 'POST':
                response = requests.post(url, json=payload, headers=self._get_headers(signed), timeout=10)
            elif method.upper() == 'DELETE':
                response = requests.delete(url, json=payload, headers=self._get_headers(signed), timeout=10)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response.raise_for_status()
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            error_msg = str(e)
            if response.status_code == 401:
                raise Exception(f"(401, 'Invalid authentication', {dict(response.headers)}, None)")
            raise Exception(f"HTTP {response.status_code}: {response.text}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")
    
    def get_exchange_info(self) -> Dict[str, Any]:
        """Get exchange information including available symbols.
        
        Returns:
            Exchange information dictionary
        """
        return self._request('POST', '/info', data={'type': 'meta'}, signed=False)
    
    def get_user_state(self) -> Dict[str, Any]:
        """Get user account state.
        
        Returns:
            User state dictionary with account information
        """
        return self._request('POST', '/info', data={
            'type': 'clearinghouseState',
            'user': self.wallet_address
        }, signed=False)
    
    def get_account(self) -> Dict[str, Any]:
        """Get account information.
        
        Returns:
            Account information dictionary
        """
        return self.get_user_state()
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """Get open positions.
        
        Returns:
            List of position dictionaries
        """
        user_state = self.get_user_state()
        # Extract positions from user state
        # This depends on Hyperliquid's actual response structure
        return user_state.get('assetPositions', [])
    
    def create_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        size: float,
        price: Optional[float] = None,
        reduce_only: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Create a new order.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC')
            side: Order side ('B' for buy, 'A' for sell)
            order_type: Order type ('Market' or 'Limit')
            size: Order size
            price: Order price (required for Limit orders)
            reduce_only: Whether this is a reduce-only order
            **kwargs: Additional order parameters
            
        Returns:
            Order information dictionary
        """
        if not self.private_key:
            raise Exception("Private key required for order creation")
        
        # Build order payload
        order = {
            'a': int(size * 1e6),  # Size in base units (assuming 6 decimals)
            'b': side,  # 'B' for buy, 'A' for sell
            'p': str(price) if price else None,
            'r': reduce_only,
            's': symbol,
            't': {'limit': {'tif': 'Gtc'}} if order_type == 'Limit' else {'market': {}},
        }
        
        # Remove None values
        order = {k: v for k, v in order.items() if v is not None}
        
        # Hyperliquid requires signed requests for order creation
        # The actual signing mechanism may differ - this is a placeholder
        return self._request('POST', '/exchange', data={
            'action': {'type': 'order', 'orders': [order], 'grouping': 'na'},
            'nonce': int(time.time() * 1000),
            'vaultAddress': None,
        }, signed=True)
    
    def get_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Get order status.
        
        Args:
            order_id: Order ID
            symbol: Trading symbol
            
        Returns:
            Order information dictionary
        """
        # Get open orders and filter by order_id
        open_orders = self.get_open_orders(symbol)
        for order in open_orders:
            if str(order.get('oid')) == str(order_id):
                return order
        
        # If not in open orders, check user fills
        fills = self.get_user_fills()
        for fill in fills:
            if str(fill.get('oid')) == str(order_id):
                return fill
        
        raise Exception(f"Order {order_id} not found")
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get open orders.
        
        Args:
            symbol: Optional trading symbol to filter by
            
        Returns:
            List of open order dictionaries
        """
        user_state = self.get_user_state()
        open_orders = user_state.get('openOrders', [])
        
        if symbol:
            open_orders = [o for o in open_orders if o.get('coin') == symbol]
        
        return open_orders
    
    def get_user_fills(self) -> List[Dict[str, Any]]:
        """Get user fill history.
        
        Returns:
            List of fill dictionaries
        """
        return self._request('POST', '/info', data={
            'type': 'userFills',
            'user': self.wallet_address
        }, signed=False)
    
    def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel an order.
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol
            
        Returns:
            Cancellation result dictionary
        """
        if not self.private_key:
            raise Exception("Private key required for order cancellation")
        
        return self._request('POST', '/exchange', data={
            'action': {'type': 'cancel', 'cancels': [{'a': int(order_id), 's': symbol}]},
            'nonce': int(time.time() * 1000),
            'vaultAddress': None,
        }, signed=True)
    
    def cancel_all_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """Cancel all open orders.
        
        Args:
            symbol: Optional symbol to cancel orders for
            
        Returns:
            Cancellation result dictionary
        """
        if not self.private_key:
            raise Exception("Private key required for order cancellation")
        
        open_orders = self.get_open_orders(symbol)
        
        if not open_orders:
            return {'status': 'ok', 'message': 'No orders to cancel'}
        
        cancels = []
        for order in open_orders:
            cancels.append({
                'a': order.get('oid'),
                's': order.get('coin')
            })
        
        return self._request('POST', '/exchange', data={
            'action': {'type': 'cancel', 'cancels': cancels},
            'nonce': int(time.time() * 1000),
            'vaultAddress': None,
        }, signed=True)


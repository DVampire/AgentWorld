from __future__ import annotations

import random
import time
from typing import Iterator, Optional, List

import requests

from .constants import INTERVAL_MS


class BinanceKlinesClient:
    def __init__(self, timeout: int = 15, max_retries: int = 8):
        self.timeout = timeout
        self.max_retries = max_retries
        self.spot_base = "https://api.binance.com"
        self.um_base = "https://fapi.binance.com"

    def _sleep_backoff(self, k: int):
        t = min(20.0, 0.5 * (2 ** k))
        t *= (0.7 + 0.6 * random.random())
        time.sleep(t)

    def _get_json(self, url: str, params: dict):
        k = 0
        while True:
            r = requests.get(url, params=params, timeout=self.timeout)
            if r.status_code in (418, 429) or r.status_code >= 500:
                k += 1
                if k > self.max_retries:
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
                self._sleep_backoff(k)
                continue
            r.raise_for_status()
            return r.json()

    def _endpoint(self, venue: str) -> str:
        if venue == "binance_spot":
            return self.spot_base + "/api/v3/klines"
        if venue == "binance_um":
            return self.um_base + "/fapi/v1/klines"
        raise ValueError("venue must be binance_spot or binance_um")

    def page_klines(
        self,
        venue: str,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: Optional[int],
    ) -> Iterator[List[list]]:
        if interval not in INTERVAL_MS:
            raise ValueError(f"unsupported interval: {interval}")

        limit = 1000 if venue == "binance_spot" else 1500
        step = INTERVAL_MS[interval]
        url = self._endpoint(venue)

        cursor = int(start_ms)
        while True:
            params = {"symbol": symbol, "interval": interval, "startTime": cursor, "limit": limit}
            if end_ms is not None:
                params["endTime"] = int(end_ms)

            batch = self._get_json(url, params)
            if not batch:
                break

            yield batch

            cursor = int(batch[-1][0]) + step
            if len(batch) < limit:
                break

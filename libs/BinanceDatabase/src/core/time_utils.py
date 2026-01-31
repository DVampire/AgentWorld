import time
from datetime import datetime, timezone


def now_ms() -> int:
    return int(time.time() * 1000)


def floor_to_minute_ms(ts_ms: int) -> int:
    return (ts_ms // 60_000) * 60_000


def ms_to_utc_date(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).date().isoformat()


def utc_ms(dt_utc: datetime) -> int:
    return int(dt_utc.replace(tzinfo=timezone.utc).timestamp() * 1000)

def ms_to_utc_datetime(ts_ms: int) -> datetime:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)

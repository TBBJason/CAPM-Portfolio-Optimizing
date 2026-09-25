"""Read through cache for daily price data

Entry point is get_daily_prices, which checks if we have the data locally in Neon.
If not, then we request the data from Yahoo Finance.

Per call:
    1. Query Neon for cached prices in the requested range
    2. Determine which symbol/date ranges are missing or stale
    3. Call Yahoo only for those ranges.
    4. Upsert returned prices into Neon
    5. Record fetch coverage in ''market_data_fetches''
    6. Query Neon again.
    7. Return a wide Pandas DataFrame (index=date, columns=symbol).

We deliberately use the market_data_fetches table to track rather than relying on a 
market calendar, since the ranges of what we asked will always be logged, it shouldn't affect 
the logic of keeping the gaps.
"""

from datetime import date, datetime, timedelta, timezone
from typing import Iterable

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from database.models import DailyPrice, MarketDataFetch
from database.session import SessionLocal


# --- Freshness / retry tuning knobs -----------------------------------------
RECENT_WINDOW = timedelta(days=7)      # "latest seven days" boundary
RECENT_REFRESH_AFTER = timedelta(hours=6)   # refresh recent data after 6h
FAILURE_BACKOFF = timedelta(minutes=15)     # wait 15m before retrying a failure

DEFAULT_PROVIDER = "yahoo"
DEFAULT_INTERVAL = "1d"

ONE_DAY = timedelta(days=1)

# --- Small date helpers ------------------------------------------------------
def _to_date(value) -> date:
    """Coerce str / datetime / pandas.Timestamp / date into a plain ``date``."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    # Handles ISO strings and pandas.Timestamp uniformly.
    return pd.Timestamp(value).date()

def _now() -> datetime:
    return datetime.now(timezone.utc)

def _compute_gaps(start: date, end: date, covered: Iterable[tuple[date, date]]):

    if start > end:
        return []


    # go over the logic of this code
    clipped = sorted( 
        (max(s, start), min(e, end)) for s, e in covered if e >= start and s <= end
    )


    # Explain why we use list here and Iterable in the parameter annotation
    gaps = list[tuple[date, date]] = []
    cursor = start

    # Explain the logic of this code as well
    for s, e in clipped:
        if s > cursor:
            gaps.append((cursor, min(s - ONE_DAY, end)))
        cursor = max(cursor, e + ONE_DAY)
        if cursor > end:
            break
    if cursor <= end:
        gaps.append((cursor, end))
    return gaps
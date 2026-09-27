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

def _overlaps(fetch: MarketDataFetch, start: date, end: date) -> bool:
    return fetch.range_start <= end and fetch.range_end >= start

def _latest_overlapping(fetches, start: date, end: date):
    candidates = [f for f in fetches if _overlaps(f, start, end)]
    if not candidates:
        return None
    return max(candidates, key=lambda f: f.fetched_at)

def _in_failure_backoff(fetches, start: date, end: date, now: datetime) -> bool:
    """True if the most recent attempt on [start, end] failed within backoff."""
    latest = _latest_overlapping(fetches, start, end)
    if latest is None or latest.status == "success":
        return False
    return (now - latest.fetched_at) < FAILURE_BACKOFF


# --- Range planning -----------------------
def _plan_symbol(fetches, start:date, end:date, now: datetime):
    today = now.date()
    recent_cutoff = today - RECENT_WINDOW
    successes = [f for f in fetches if f.status == "success"] # does this just take in the database?
    needed: list[tuple[date, date]] = []

    # --- Old portion: dates strictly older than the recent window ----------
    old_end = min(end, recent_cutoff - ONE_DAY)
    if start <= old_end:
        gaps = _compute_gaps(start, old_end, [(f.range_start, f.range_end) for f in successes])
        for g_start, g_end in gaps:
            # Respect the failure backoff so we don't hammer a bad range.
            if not _in_failure_backoff(fetches, g_start, g_end, now):
                needed.append((g_start, g_end))
                
    # --- Recent portion: the latest seven days -----------------------------
    recent_start = max(start, recent_cutoff)
    if recent_start <= end:
        latest = _latest_overlapping(fetches, recent_start, end)
        if latest is None:
            needed.append((recent_start, end))
        elif latest.status == "success":
            if (now - latest.fetched_at) >= RECENT_REFRESH_AFTER:
                needed.append((recent_start, end))
            # else: fresh enough, serve from cache
        else:  # empty / error -> apply 15 minute backoff
            if (now - latest.fetched_at) >= FAILURE_BACKOFF:
                needed.append((recent_start, end))

    return _merge_ranges(needed)


def _merge_ranges(ranges: list[tuple[date, date]]):
    """Merge overlapping/adjacent inclusive date ranges."""
    if not ranges:
        return []
    ranges = sorted(ranges)
    merged = [ranges[0]]
    for s, e in ranges[1:]:
        last_s, last_e = merged[-1]
        if s <= last_e + ONE_DAY:
            merged[-1] = (last_s, max(last_e, e))
        else:
            merged.append((s, e))
    return merged

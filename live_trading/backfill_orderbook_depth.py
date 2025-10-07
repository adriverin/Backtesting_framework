from __future__ import annotations

"""Backfill Binance orderbook depth daily CSVs for a symbol.

Usage:
  python live_trading/backfill_orderbook_depth.py \
    --symbol VETUSDT \
    --mode futures \
    --start 2025-03-11 \
    --end 2025-10-06

Or rely on config for base_dir and defaults:
  python live_trading/backfill_orderbook_depth.py --symbol VETUSDT --days 210

Notes:
  - Downloads from Binance Vision daily archives (bookDepth). For futures UM:
      https://data.binance.vision/data/futures/um/daily/bookDepth/{SYMBOL}/
  - Saves CSV as {base_dir}/{SYMBOL}/{SYMBOL}-bookDepth-YYYY-MM-DD.csv
  - Skips today; Binance only publishes daily files T+1
  - Creates/updates parquet day-cache if your strategy uses it later
"""

import argparse
import io
import os
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from live_trading.config import TradingConfig  # type: ignore


def normalize_symbol(symbol: str) -> str:
    s = symbol.upper()
    # Convert XYZUSD -> XYZUSDT if missing T
    if s.endswith("USD") and not s.endswith("USDT"):
        s = s + "T"
    return s


def build_daily_url(symbol: str, date: datetime, mode: str = "futures") -> str:
    symbol = normalize_symbol(symbol)
    base = "https://data.binance.vision/data"
    if mode.lower() == "futures":
        path = f"futures/um/daily/bookDepth/{symbol}/{symbol}-bookDepth-{date.year}-{date.month:02d}-{date.day:02d}.zip"
    else:
        path = f"spot/daily/bookDepth/{symbol}/{symbol}-bookDepth-{date.year}-{date.month:02d}-{date.day:02d}.zip"
    return f"{base}/{path}"


def destination_paths(base_dir: Path, symbol: str, date: datetime) -> tuple[Path, Path]:
    symbol = normalize_symbol(symbol)
    csv_name = f"{symbol}-bookDepth-{date.year}-{date.month:02d}-{date.day:02d}.csv"
    pq_dir = base_dir / symbol / "_parquet"
    csv_path = base_dir / symbol / csv_name
    pq_path = pq_dir / csv_name.replace(".csv", ".parquet")
    return csv_path, pq_path


def csv_exists(csv_path: Path) -> bool:
    return csv_path.exists() and csv_path.stat().st_size > 0


def load_from_zip_bytes(zbytes: bytes) -> Optional[pd.DataFrame]:
    try:
        with zipfile.ZipFile(io.BytesIO(zbytes)) as zf:
            # Expect single CSV inside zip
            names = [n for n in zf.namelist() if n.endswith('.csv')]
            if not names:
                return None
            with zf.open(names[0]) as f:
                df = pd.read_csv(f)
        return df
    except Exception:
        return None


def normalize_depth_csv(df: pd.DataFrame) -> pd.DataFrame:
    # Expect columns: timestamp, percentage, depth, notional (as in repo data)
    # Some archives may have slightly different numeric dtypes; coerce.
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    for col in ("percentage", "depth", "notional"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    # Keep only expected columns, drop NaNs
    keep = [c for c in ['timestamp', 'percentage', 'depth', 'notional'] if c in df.columns]
    df = df[keep].dropna()
    return df


def write_outputs(df: pd.DataFrame, csv_path: Path, pq_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    try:
        pq_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(pq_path, index=False)
    except Exception:
        pass


def backfill(symbol: str, start: datetime, end: datetime, mode: str, base_dir: Path) -> None:
    symbol = normalize_symbol(symbol)
    today_utc = datetime.now(timezone.utc).date()
    # Only backfill complete days (T+1 availability)
    hard_end = min(end.date(), today_utc - timedelta(days=1))
    if start.date() > hard_end:
        print("[backfill] Nothing to do: start > yesterday")
        return

    current = start.date()
    while current <= hard_end:
        date_dt = datetime(year=current.year, month=current.month, day=current.day, tzinfo=timezone.utc)
        csv_path, pq_path = destination_paths(base_dir, symbol, date_dt)

        if csv_exists(csv_path):
            print(f"[backfill] Exists, skipping: {csv_path.name}")
            current += timedelta(days=1)
            continue

        url = build_daily_url(symbol, date_dt, mode)
        print(f"[backfill] Downloading: {url}")
        try:
            resp = requests.get(url, timeout=60)
            if resp.status_code != 200:
                print(f"[backfill] Not available ({resp.status_code}), skipping {current}")
                current += timedelta(days=1)
                continue
            df = load_from_zip_bytes(resp.content)
            if df is None or df.empty:
                print(f"[backfill] Empty zip for {current}")
                current += timedelta(days=1)
                continue
            df = normalize_depth_csv(df)
            if df.empty:
                print(f"[backfill] No usable rows after normalization for {current}")
                current += timedelta(days=1)
                continue
            write_outputs(df, csv_path, pq_path)
            print(f"[backfill] Wrote {csv_path.name} ({len(df)} rows)")
        except Exception as e:
            print(f"[backfill] Error for {current}: {e}")
        finally:
            current += timedelta(days=1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backfill Binance orderbook depth daily CSVs")
    p.add_argument('--symbol', required=True, help='Symbol, e.g. VETUSDT')
    p.add_argument('--mode', default='futures', choices=['futures','spot'], help='Market type')
    p.add_argument('--start', help='Start date YYYY-MM-DD (optional if --days)')
    p.add_argument('--end', help='End date YYYY-MM-DD (default: today-1)')
    p.add_argument('--days', type=int, help='Backfill last N days ending yesterday')
    p.add_argument('--base_dir', help='Override base dir; defaults to config data.orderbook_depth_dir')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TradingConfig()
    base_dir_cfg = args.base_dir or cfg.get('data.orderbook_depth_dir', 'data/orderbook_depth')
    base_dir = Path(base_dir_cfg)

    if args.days:
        end = datetime.now(timezone.utc) - timedelta(days=1)
        start = end - timedelta(days=args.days-1)
    else:
        if not args.start:
            raise SystemExit("--start YYYY-MM-DD or --days N required")
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        if args.end:
            end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        else:
            end = datetime.now(timezone.utc) - timedelta(days=1)

    backfill(symbol=args.symbol, start=start, end=end, mode=args.mode, base_dir=base_dir)


if __name__ == '__main__':
    main()



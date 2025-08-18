from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def _normalize_symbol_for_filename(symbol: str) -> str:
    return symbol.replace("/", "").replace("-", "")


def _normalize_symbol_for_binance(symbol: str) -> str:
    """Map user symbols to Binance spot symbols.

    Accepts BTCUSD, BTC-USD, BTC/USD, BTCUSDT and returns 'BTC/USDT'.
    Default quote is USDT for 'USD' inputs.
    """
    s = symbol.upper().replace(" ", "")
    base: str
    quote: str

    if "/" in s:
        base, quote = s.split("/")
    elif "-" in s:
        base, quote = s.split("-")
    else:
        if s.endswith("USDT"):
            base, quote = s[:-4], "USDT"
        elif s.endswith("USD"):
            base, quote = s[:-3], "USDT"
        else:
            base, quote = s, "USDT"

    if quote == "USD":
        quote = "USDT"
    return f"{base}/{quote}"


def _interval_to_pandas_rule(interval: str) -> Tuple[str, pd.Timedelta]:
    if interval == "1m":
        return "1T", pd.Timedelta(minutes=1)
    if interval == "5m":
        return "5T", pd.Timedelta(minutes=5)
    if interval == "15m":
        return "15T", pd.Timedelta(minutes=15)
    if interval == "1h":
        return "1H", pd.Timedelta(hours=1)
    if interval == "4h":
        return "4H", pd.Timedelta(hours=4)
    if interval == "1d":
        return "1D", pd.Timedelta(days=1)
    raise ValueError(f"Unsupported interval: {interval}")


def _interval_to_ccxt(interval: str) -> str:
    """Validate/convert interval to ccxt timeframe string."""
    supported = {"1m", "5m", "15m", "1h", "4h", "1d"}
    if interval not in supported:
        raise ValueError(f"Unsupported interval for Binance via ccxt: {interval}")
    return interval


def _download_ohlcv(symbol_binance: str, start: str, end: str, interval: str) -> pd.DataFrame:
    """Download OHLCV from Binance via ccxt over a date range.

    Returns a DataFrame with columns: open, high, low, close, volume and a UTC-naive DatetimeIndex.
    """
    try:
        import ccxt  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise ImportError(
            "ccxt is required to download data from Binance. Install with `pip install ccxt`"
        ) from exc

    timeframe = _interval_to_ccxt(interval)

    # Prepare time bounds in ms
    start_dt = pd.to_datetime(start, utc=True)
    end_dt = pd.to_datetime(end, utc=True)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    exchange = ccxt.binance({"enableRateLimit": True})
    all_rows: list[list[float]] = []

    # Compute step based on interval to advance reliably
    _, expected_delta = _interval_to_pandas_rule(interval)
    step_ms = int(expected_delta.total_seconds() * 1000)

    since = start_ms
    limit = 1000  # exchange max per call

    while since < end_ms:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol_binance, timeframe=timeframe, since=since, limit=limit)
        except Exception as e:
            # On transient errors, break to avoid infinite loops
            print(f"⚠️  Fetch error for {symbol_binance} at {since}: {e}")
            break

        if not ohlcv:
            break

        all_rows.extend(ohlcv)

        last_ts = ohlcv[-1][0]
        if last_ts is None:
            break
        # Advance 'since' to the next bar to avoid duplicates
        since = max(since + step_ms, last_ts + 1)

        # Safety: if no forward progress, break
        if len(ohlcv) < limit and last_ts + step_ms >= end_ms:
            break

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    # Convert timestamp to UTC-naive datetime index
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    # Filter to requested window and drop duplicates
    df = df[(df.index >= start_dt) & (df.index < end_dt)]
    df = df[~df.index.duplicated(keep="last")]
    # Make index tz-naive (UTC)
    df.index = df.index.tz_convert("UTC").tz_localize(None)

    # Additional derived price columns used by ML strategies
    if not df.empty:
        df["typical"] = (df["high"] + df["low"] + df["close"]) / 3.0
        df["median"] = (df["high"] + df["low"]) / 2.0
        df["vwap"] = df["typical"]  # aligned with existing usage in this project

    return df


def create_maximum_cache_for_assets(
    assets: List[str],
    interval: str = "1h",
    start: str = "2017-01-01",
    end: str = "2025-12-31",
    create_info_file: bool = True,
) -> Dict[str, dict]:
    """
    Download OHLCV for multiple assets and save under data/ with the framework's naming:
    `data/ohlcv_{SYMBOL}_{INTERVAL}.parquet` where SYMBOL is like BTCUSD (no dash).
    Also generates `data/asset_cache_info_{interval}.json` and a text summary.
    """
    import json

    print(f"🗄️  Creating OHLCV cache files for {len(assets)} assets (Binance via ccxt)")
    print(f"📅 Requested range: {start} to {end} ({interval} interval)")
    print("=" * 70)

    asset_info: Dict[str, dict] = {}
    cache_stats = {
        "creation_date": datetime.now().isoformat(),
        "requested_range": {"start": start, "end": end},
        "interval": interval,
        "cache_type": "ohlcv",
        "total_assets": len(assets),
        "successful_assets": 0,
        "failed_assets": 0,
        "assets": {},
    }

    cache_dir = Path("")
    cache_dir.mkdir(parents=True, exist_ok=True)

    rule_str, expected_delta = _interval_to_pandas_rule(interval)

    for i, asset in enumerate(assets, 1):
        print(f"\n📦 Caching {asset} ({i}/{len(assets)})...")
        symbol_binance = _normalize_symbol_for_binance(asset)
        symbol_file = _normalize_symbol_for_filename(asset)
        cache_file = cache_dir / f"ohlcv_{symbol_file}_{interval}.parquet"

        try:
            # Determine if we should append or create fresh
            df_existing: pd.DataFrame | None = None
            if cache_file.exists():
                try:
                    df_existing = pd.read_parquet(cache_file)
                except Exception:
                    df_existing = None

            if df_existing is not None and not df_existing.empty:
                # Normalize existing index to UTC-naive to avoid tz-aware/naive comparison issues
                try:
                    if not isinstance(df_existing.index, pd.DatetimeIndex):
                        df_existing.index = pd.to_datetime(df_existing.index, utc=True, errors="coerce")
                    if getattr(df_existing.index, "tz", None) is not None:
                        df_existing.index = df_existing.index.tz_convert("UTC").tz_localize(None)
                except Exception:
                    pass
                last_idx: pd.Timestamp = pd.to_datetime(df_existing.index.max())
                # Start strictly after the last saved bar to avoid refetching duplicates
                _, expected_delta = _interval_to_pandas_rule(interval)
                effective_start_dt = max(pd.to_datetime(start), last_idx + expected_delta)
                # If we already have data up to (or beyond) the requested end, skip fetching
                if effective_start_dt >= pd.to_datetime(end):
                    df = df_existing
                    if df is not None and not df.empty:
                        df.to_parquet(cache_file)

                        actual_start = df.index.min()
                        actual_end = df.index.max()
                        actual_start_str = actual_start.strftime("%Y-%m-%d")
                        actual_end_str = actual_end.strftime("%Y-%m-%d")

                        # Data coverage and gap metrics
                        rule_str, expected_delta = _interval_to_pandas_rule(interval)
                        if len(df) > 1:
                            time_diffs = df.index.to_series().diff().dropna()
                            large_gaps = int((time_diffs > expected_delta * 2).sum())
                        else:
                            large_gaps = 0

                        total_expected = 0
                        if pd.notna(actual_start) and pd.notna(actual_end) and actual_end > actual_start:
                            total_expected = int(((actual_end - actual_start) / expected_delta))
                        coverage_pct = (len(df) / total_expected * 100.0) if total_expected > 0 else 0.0

                        info = {
                            "status": "success",
                            "cache_type": "ohlcv",
                            "available_from": actual_start_str,
                            "available_to": actual_end_str,
                            "total_bars": int(len(df)),
                            "data_coverage_pct": round(float(coverage_pct), 2),
                            "large_gaps_detected": int(large_gaps),
                            "trading_days": int((actual_end - actual_start).days) if len(df) else 0,
                            "ohlcv_columns": list(df.columns),
                            "cache_file": f"ohlcv_{symbol_file}_{interval}.parquet",
                        }

                        asset_info[asset] = info
                        cache_stats["successful_assets"] += 1
                        cache_stats["assets"][asset] = info

                        print(f"✅ {asset}: up to date (no new bars to fetch)")
                        print(f"   📅 Available: {actual_start_str} to {actual_end_str}")
                        print(f"   📈 Columns: {list(df.columns)}")
                    else:
                        info = {"status": "no_data", "error": "No OHLCV data returned"}
                        asset_info[asset] = info
                        cache_stats["failed_assets"] += 1
                        cache_stats["assets"][asset] = info
                        print(f"⚠️  {asset}: No OHLCV data available")
                    continue
                effective_start = effective_start_dt.strftime("%Y-%m-%d %H:%M:%S")
            else:
                effective_start = start

            df_new = _download_ohlcv(symbol_binance, start=effective_start, end=end, interval=interval)

            if df_existing is not None and not df_existing.empty and not df_new.empty:
                df_combined = pd.concat([df_existing, df_new])
                df = (
                    df_combined[~df_combined.index.duplicated(keep="last")]
                    .sort_index()
                )
            elif df_existing is not None and not df_existing.empty and df_new.empty:
                df = df_existing
            else:
                df = df_new

            if df is not None and not df.empty:
                df.to_parquet(cache_file)

                actual_start = df.index.min()
                actual_end = df.index.max()
                actual_start_str = actual_start.strftime("%Y-%m-%d")
                actual_end_str = actual_end.strftime("%Y-%m-%d")

                # Data coverage and gap metrics
                if len(df) > 1:
                    time_diffs = df.index.to_series().diff().dropna()
                    large_gaps = int((time_diffs > expected_delta * 2).sum())
                else:
                    large_gaps = 0

                total_expected = 0
                if pd.notna(actual_start) and pd.notna(actual_end) and actual_end > actual_start:
                    total_expected = int(((actual_end - actual_start) / expected_delta))
                coverage_pct = (len(df) / total_expected * 100.0) if total_expected > 0 else 0.0

                info = {
                    "status": "success",
                    "cache_type": "ohlcv",
                    "available_from": actual_start_str,
                    "available_to": actual_end_str,
                    "total_bars": int(len(df)),
                    "data_coverage_pct": round(float(coverage_pct), 2),
                    "large_gaps_detected": int(large_gaps),
                    "trading_days": int((actual_end - actual_start).days) if len(df) else 0,
                    "ohlcv_columns": list(df.columns),
                    "cache_file": f"ohlcv_{symbol_file}_{interval}.parquet",
                }

                asset_info[asset] = info
                cache_stats["successful_assets"] += 1
                cache_stats["assets"][asset] = info

                print(f"✅ {asset}: {len(df):,} OHLCV bars cached (updated)")
                print(f"   📅 Available: {actual_start_str} to {actual_end_str}")
                print(f"   📊 Coverage: {coverage_pct:.1f}% ({large_gaps} gaps detected)")
                print(f"   📈 Columns: {list(df.columns)}")
            else:
                info = {"status": "no_data", "error": "No OHLCV data returned"}
                asset_info[asset] = info
                cache_stats["failed_assets"] += 1
                cache_stats["assets"][asset] = info
                print(f"⚠️  {asset}: No OHLCV data available")
        except Exception as exc:
            error_msg = str(exc)
            info = {"status": "error", "error": error_msg}
            asset_info[asset] = info
            cache_stats["failed_assets"] += 1
            cache_stats["assets"][asset] = info
            print(f"❌ {asset}: Failed - {error_msg}")

    if create_info_file:
        info_file = cache_dir / f"asset_cache_info_{interval}.json"
        summary_file = cache_dir / f"asset_cache_summary_{interval}.txt"

        with open(info_file, "w") as f:
            import json as _json

            _json.dump(cache_stats, f, indent=2, default=str)

        with open(summary_file, "w") as f:
            f.write("Cryptocurrency Asset OHLCV Cache Summary\n")
            f.write("=========================================\n\n")
            f.write(f"Created: {cache_stats['creation_date']}\n")
            f.write("Cache Type: OHLCV (Open, High, Low, Close, Volume)\n")
            f.write(f"Interval: {interval}\n")
            f.write(f"Requested Range: {start} to {end}\n")
            f.write(f"Total Assets: {len(assets)}\n")
            f.write(f"Successful: {cache_stats['successful_assets']}\n")
            f.write(f"Failed: {cache_stats['failed_assets']}\n\n")
            f.write("Asset Details:\n")
            f.write("-" * 50 + "\n")
            for asset, info in asset_info.items():
                f.write(f"\n{asset}:\n")
                if info.get("status") == "success":
                    f.write("  Status: ✅ Success\n")
                    f.write("  Type: OHLCV Cache\n")
                    f.write(f"  Available: {info['available_from']} to {info['available_to']}\n")
                    f.write(f"  Bars: {info['total_bars']:,}\n")
                    f.write(f"  Coverage: {info['data_coverage_pct']}%\n")
                    f.write(f"  Trading Days: {info['trading_days']:,}\n")
                    f.write(f"  Gaps: {info['large_gaps_detected']}\n")
                    f.write(f"  Columns: {', '.join(info['ohlcv_columns'])}\n")
                    f.write(f"  Cache File: {info['cache_file']}\n")
                else:
                    f.write(f"  Status: ❌ {info.get('status', 'unknown')}\n")
                    f.write(f"  Error: {info.get('error', 'Unknown error')}\n")

        print("\n📋 Asset information files created:")
        print(f"   📄 Detailed: {info_file}")
        print(f"   📄 Summary: {summary_file}")

    print("\n🎉 OHLCV cache creation complete!")
    print(
        f"📊 Results: {cache_stats['successful_assets']} successful, {cache_stats['failed_assets']} failed"
    )
    return asset_info


if __name__ == "__main__":
    # time_intervals = ["1m", "5m", "15m", "4h", "1d"]
    time_intervals = ["5m","15m"]

    for tf in time_intervals:
        print(f"Getting max cached for {tf}")
        create_maximum_cache_for_assets(
            assets=[
                "BTC-USD",
                "ETH-USD",
                "SOL-USD",
                "ADA-USD",
                "AVAX-USD",
                "BNB-USD",
                "XRP-USD",
                "LTC-USD",
                "LINK-USD",
                "XLM-USD",
                "ATOM-USD",
                "HBAR-USD",
                "BCH-USD",
                "DOT-USD",
                "UNI-USD",
                "AAVE-USD",
                "SCRT-USD",
                "ALGO-USD",
                "VET-USD",
                "XTZ-USD",
                # Meme coins (availability may vary on Yahoo Finance):
                "DOGE-USD",
                "PEPE-USD",
                "SHIB-USD",
                "BONK-USD",
                "WIF-USD",
                "FLOKI-USD",
            ],
            interval=tf,
            start="2010-01-01",
            end="2030-12-31",
            create_info_file=True,
        )
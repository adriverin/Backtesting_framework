from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Reuse helpers to match schema/filenames
from download_data import _normalize_symbol_for_binance, _normalize_symbol_for_filename, _interval_to_pandas_rule


def _ensure_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
	if isinstance(df.index, pd.DatetimeIndex):
		# Make UTC-naive per project convention
		if getattr(df.index, "tz", None) is not None:
			df.index = df.index.tz_convert("UTC").tz_localize(None)
		return df
	# Try common columns
	for col in ("timestamp", "index", "openTime"):
		if col in df.columns:
			df[col] = pd.to_datetime(df[col], errors="coerce", utc=True)
			df = df.set_index(col).sort_index()
			if getattr(df.index, "tz", None) is not None:
				df.index = df.index.tz_convert("UTC").tz_localize(None)
			return df
	# Fallback: coerce current index
	df.index = pd.to_datetime(df.index, errors="coerce", utc=True)
	if getattr(df.index, "tz", None) is not None:
		df.index = df.index.tz_convert("UTC").tz_localize(None)
	return df


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
	# Align with download_data: typical, median, vwap
	if not df.empty:
		df["typical"] = (df["high"] + df["low"] + df["close"]) / 3.0
		df["median"] = (df["high"] + df["low"]) / 2.0
		# vwap placeholder (uses typical due to project convention)
		df["vwap"] = df["typical"]
	return df


def _fetch_from_since(symbol_ccxt: str, interval: str, since_ms: int, limit: int = 1000) -> pd.DataFrame:
	try:
		import ccxt  # type: ignore
	except Exception as exc:
		raise ImportError("ccxt is required. Install with `pip install ccxt`.") from exc

	exchange = ccxt.binance({"enableRateLimit": True})
	all_rows: list[list[float]] = []
	cur = since_ms
	while True:
		raw = exchange.fetch_ohlcv(symbol_ccxt, timeframe=interval, since=cur, limit=limit)
		if not raw:
			break
		all_rows.extend(raw)
		last_ts = raw[-1][0]
		if last_ts is None:
			break
		# Advance at least one ms to avoid duplicates
		cur = int(last_ts) + 1
		# Stop if we didn't fill the page (likely caught up)
		if len(raw) < limit:
			break
	if not all_rows:
		return pd.DataFrame()
	df = pd.DataFrame(all_rows, columns=["timestamp", "open", "high", "low", "close", "volume"]).set_index("timestamp")
	df.index = pd.to_datetime(df.index, unit="ms", utc=True).tz_convert("UTC").tz_localize(None)
	return df.sort_index()


def main():
	parser = argparse.ArgumentParser(description="CCXT-based live feeder that appends OHLCV to parquet (matching download_data schema)")
	parser.add_argument("--symbol", required=True, help="Asset symbol like VET-USD, VETUSD, or VET/USDT")
	parser.add_argument("--interval", required=True, choices=["1m","5m","15m","1h","4h","1d"], help="Bar interval")
	parser.add_argument("--poll", type=int, default=30, help="Polling seconds between updates")
	parser.add_argument("--mode", type=str, default="spot", choices=["spot","futures"], help="Instrument mode: spot or futures (USDT-M)")
	args = parser.parse_args()

	sym_ccxt = _normalize_symbol_for_binance(args.symbol)  # e.g., 'VET/USDT'
	sym_file = _normalize_symbol_for_filename(args.symbol)  # e.g., 'VETUSD'
	base_dir = Path("data/futures") if args.mode.strip().lower() == "futures" else Path("data/spot")
	parquet_path = base_dir / f"ohlcv_{sym_file}_{args.interval}.parquet"
	parquet_path.parent.mkdir(parents=True, exist_ok=True)

	venue = "Binance USDT-M Futures" if args.mode.strip().lower() == "futures" else "Binance Spot"
	print(f"[feeder-ccxt] Writing to {parquet_path} | source={venue} via ccxt | symbol={sym_ccxt} interval={args.interval} poll={args.poll}s mode={args.mode}")

	# Determine expected bar delta
	_, expected_delta = _interval_to_pandas_rule(args.interval)

	while True:
		try:
			if parquet_path.exists():
				existing = pd.read_parquet(parquet_path)
				existing = _ensure_datetime_index(existing)
				last_ts: pd.Timestamp | None = None
				if not existing.empty:
					last_ts = pd.to_datetime(existing.index.max())
				# If no data, backfill a reasonable history (e.g., 1500 bars)
				if last_ts is None:
					since_ms = int((datetime.now(timezone.utc) - expected_delta * 1500).timestamp() * 1000)
				else:
					# Start strictly after the last saved bar
					last_ts_aware = last_ts.tz_localize("UTC") if last_ts.tzinfo is None else last_ts.tz_convert("UTC")
					since_ms = int((last_ts_aware + expected_delta).timestamp() * 1000)
			else:
				# No parquet: backfill initial window
				since_ms = int((datetime.now(timezone.utc) - expected_delta * 1500).timestamp() * 1000)

			# Fetch from 'since' forward (may return multiple pages)
			# For futures, use binanceusdm; for spot, use binance
			try:
				import ccxt  # type: ignore
				exchange = ccxt.binanceusdm({"enableRateLimit": True}) if args.mode.strip().lower() == "futures" else ccxt.binance({"enableRateLimit": True})
				raw_rows = []
				cur = since_ms
				while True:
					chunk = exchange.fetch_ohlcv(sym_ccxt, timeframe=args.interval, since=cur, limit=1000)
					if not chunk:
						break
					raw_rows.extend(chunk)
					last_ts = chunk[-1][0]
					if last_ts is None:
						break
					cur = int(last_ts) + 1
					if len(chunk) < 1000:
						break
				new_df = pd.DataFrame(raw_rows, columns=["timestamp","open","high","low","close","volume"]).set_index("timestamp")
				if not new_df.empty:
					new_df.index = pd.to_datetime(new_df.index, unit="ms", utc=True).tz_convert("UTC").tz_localize(None)
					new_df = new_df.sort_index()
			except Exception as _e:
				print(f"[feeder-ccxt] fetch ERROR: {_e}")
				new_df = pd.DataFrame()
			if parquet_path.exists():
				existing = pd.read_parquet(parquet_path)
				existing = _ensure_datetime_index(existing)
				merged = pd.concat([existing, new_df]) if not new_df.empty else existing
			else:
				merged = new_df

			if merged is None or merged.empty:
				print("[feeder-ccxt] No data to write this cycle")
				time.sleep(args.poll)
				continue

			merged = merged[~merged.index.duplicated(keep="last")].sort_index()
			merged = _add_derived_columns(merged)

			tmp = parquet_path.with_suffix(".parquet.tmp")
			merged.to_parquet(tmp)
			os.replace(tmp, parquet_path)
			print(f"[feeder-ccxt] Upserted through {merged.index[-1]} | total bars={len(merged)}")
		except Exception as e:
			print(f"[feeder-ccxt] ERROR: {e}")
		finally:
			time.sleep(args.poll)


if __name__ == "__main__":
	main()

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd
import requests

BINANCE_API_BASE = os.environ.get("BINANCE_API_BASE", "https://api.binance.com")


def _symbol_to_asset_filename(symbol: str) -> str:
	return symbol.replace("USDT", "USD") if symbol.endswith("USDT") else symbol


def _now_ms() -> int:
	return int(datetime.now(timezone.utc).timestamp() * 1000)


def fetch_klines(symbol: str, interval: str, limit: int = 1000) -> List[list]:
	url = f"{BINANCE_API_BASE}/api/v3/klines"
	params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
	r = requests.get(url, params=params, timeout=10)
	r.raise_for_status()
	return r.json()


def klines_to_df(kl: List[list]) -> pd.DataFrame:
	rows = [
		[ int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5]), int(k[6]) ]
		for k in kl
	]
	df = pd.DataFrame(rows, columns=["openTime","open","high","low","close","volume","closeTime"])
	# Build DatetimeIndex in UTC-naive (matching loaders)
	ts = pd.to_datetime(df["openTime"], unit="ms", utc=True).tz_convert("UTC").tz_localize(None)
	df = df[["open","high","low","close","volume","closeTime"]].copy()
	df.index = ts
	df.index.name = "timestamp"
	return df


def _coerce_existing_index(existing: pd.DataFrame) -> pd.DataFrame:
	# If parquet was written without index or with wrong dtype, try to fix
	if isinstance(existing.index, pd.DatetimeIndex):
		return existing
	# Common cases: an 'index' or 'timestamp' column exists
	for cand in ("timestamp", "index", "openTime"):
		if cand in existing.columns:
			try:
				existing[cand] = pd.to_datetime(existing[cand], errors="coerce", utc=True).dt.tz_convert("UTC").dt.tz_localize(None)
				existing = existing.set_index(cand)
				existing.index.name = "timestamp"
				return existing
			except Exception:
				pass
	# Fallback: try parsing current index values
	try:
		existing.index = pd.to_datetime(existing.index, errors="coerce", utc=True).tz_convert("UTC").tz_localize(None)
		existing.index.name = "timestamp"
	except Exception:
		raise ValueError("Existing parquet index is not datetime-like; consider deleting the file to rebuild cleanly")
	return existing


def main():
	parser = argparse.ArgumentParser(description="Live feeder that appends Binance klines to local parquet")
	parser.add_argument("--symbol", required=True, help="Trading symbol, e.g. VETUSDT")
	parser.add_argument("--interval", required=True, help="Binance interval, e.g. 1h, 4h, 1m")
	parser.add_argument("--poll", type=int, default=30, help="Polling seconds")
	parser.add_argument("--limit", type=int, default=1000, help="Max klines per request")
	args = parser.parse_args()

	symbol = args.symbol.upper()
	interval = args.interval
	poll_seconds = max(5, int(args.poll))
	limit = int(args.limit)

	asset_name = _symbol_to_asset_filename(symbol)
	parquet_path = Path(f"data/ohlcv_{asset_name}_{interval}.parquet")
	parquet_path.parent.mkdir(parents=True, exist_ok=True)

	print(f"[feeder] Writing to {parquet_path} from {BINANCE_API_BASE} | {symbol} {interval} | poll={poll_seconds}s")

	engine = None
	try:
		import pyarrow  # type: ignore
		engine = "pyarrow"
	except Exception:
		engine = None

	while True:
		try:
			kl = fetch_klines(symbol, interval, limit=limit)
			df = klines_to_df(kl)
			nowms = _now_ms()
			df = df[df["closeTime"] <= nowms].drop(columns=["closeTime"])  # keep only finalized

			if parquet_path.exists():
				existing = pd.read_parquet(parquet_path)
				existing = _coerce_existing_index(existing)
				merged = pd.concat([existing, df])
				merged = merged[~merged.index.duplicated(keep="last")].sort_index()
			else:
				merged = df.sort_index()

			# Ensure a valid DatetimeIndex
			if not isinstance(merged.index, pd.DatetimeIndex):
				raise ValueError("index is not a valid DatetimeIndex or PeriodIndex (after coercion)")
			merged.index.name = "timestamp"

			# Atomic write
			tmp_path = parquet_path.with_suffix(".parquet.tmp")
			if engine:
				merged.to_parquet(tmp_path, engine=engine)
			else:
				merged.to_parquet(tmp_path)
			os.replace(tmp_path, parquet_path)

			print(f"[feeder] Upserted {len(df)} bars | total={len(merged)} | last={merged.index[-1]}")
		except Exception as e:
			print(f"[feeder] ERROR: {e}")
			if "pyarrow" in str(e).lower():
				print("[feeder] Tip: pip install pyarrow for more robust parquet IO")

		time.sleep(poll_seconds)


if __name__ == "__main__":
	main()

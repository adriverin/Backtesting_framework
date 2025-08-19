from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

from strategies import aVAILABLE_STRATEGIES, BaseStrategy
from strategies.ml_strategy import MLStrategy
from logger_ndjson import NDJSONLogger

# Optional Binance Testnet client
try:
	from binance.spot import Spot as BinanceSpot  # type: ignore
except Exception:
	BinanceSpot = None  # type: ignore


@dataclass
class TradeConfig:
	strategy_name: str = "ml"
	symbol: str = "BTCUSDT"
	base_asset: str = "BTC"
	quote_asset: str = "USDT"
	timeframe: str = "4h"
	price_column: str = "vwap"  # use MLStrategy-compatible price column
	ml_params: Dict[str, Any] = None  # filled from is_results.ml_params
	position_sizing_mode: str = "fixed_fraction"
	position_sizing_params: Dict[str, Any] = None
	paper: bool = True
	poll_seconds: int = 5
	logs_dir: str = "logs"
	vwap_window: int = 10  # map 'vwap_10' to vwap + window
	market_quote_usdt: float = 25.0  # quote notional to spend per market trade when client is enabled
	start_equity: float = 10000.0
	max_position_notional: Optional[float] = None  # e.g., 500 USDT
	daily_loss_limit_quote: Optional[float] = None  # e.g., 200 USDT
	wf_lookback_years: float = 2.0
	wf_step_days: int = 7
	sim_loop: bool = False  # if True, loop over historical bars indefinitely
	base_qty: float = 0.001  # default simulated base amount per buy


def load_latest_ohlcv(asset: str, timeframe: str) -> pd.DataFrame:
	"""Load local OHLCV parquet prepared by backtests."""
	path = f"data/ohlcv_{asset}_{timeframe}.parquet"
	if not os.path.exists(path):
		raise FileNotFoundError(path)
	df = pd.read_parquet(path)
	if not isinstance(df.index, pd.DatetimeIndex):
		df.index = pd.to_datetime(df.index, errors="coerce")
	if getattr(df.index, "tz", None) is not None:
		df.index = df.index.tz_convert("UTC").tz_localize(None)
	return df


def timeframe_to_bars_per_day(tf: str) -> int:
	tf = tf.lower().strip()
	if tf.endswith("m"):
		bar_hours = int(tf[:-1]) / 60.0
	elif tf.endswith("h"):
		bar_hours = float(int(tf[:-1]))
	elif tf.endswith("d"):
		bar_hours = float(int(tf[:-1]) * 24)
	else:
		bar_hours = 1.0
	return max(1, int(round(24.0 / bar_hours)))


def make_client() -> Optional[object]:
	"""Create Binance Spot client for Testnet if env vars are present."""
	use_testnet = os.environ.get("BINANCE_TESTNET", "1") == "1"
	api_key = os.environ.get("BINANCE_TESTNET_API_KEY")
	api_secret = os.environ.get("BINANCE_TESTNET_API_SECRET")
	if not use_testnet or not api_key or not api_secret or BinanceSpot is None:
		return None
	try:
		client = BinanceSpot(key=api_key, secret=api_secret, base_url="https://testnet.binance.vision")
		client.ping()
		return client
	except Exception:
		return None


def main() -> None:
	# ----- Config -----
	from is_results import ml_params  # reuse your tuned params
	params = dict(ml_params)
	params.setdefault("vwap_window", 10)
	params.setdefault("hold_until_opposite", True)
	cfg = TradeConfig(
		strategy_name="ml",
		symbol=os.environ.get("SYMBOL", "VETUSDT"),
		base_asset=os.environ.get("BASE_ASSET", "VET"),
		quote_asset=os.environ.get("QUOTE_ASSET", "USDT"),
		timeframe=params.get("interval", os.environ.get("TF", "4h")),
		price_column="vwap",
		ml_params=params,
		position_sizing_mode="fixed_fraction",
		position_sizing_params={"fraction": float(os.environ.get("SIZE_FRACTION", "0.1"))},
		paper=True,
		poll_seconds=int(os.environ.get("POLL_SECONDS", "5")),
		logs_dir="logs",
		vwap_window=params.get("vwap_window", 10),
		market_quote_usdt=float(os.environ.get("TEST_ORDER_QUOTE", "1000")),
		start_equity=float(os.environ.get("START_EQUITY", "10000")),
		max_position_notional=float(os.environ.get("MAX_POSITION_NOTIONAL", "0") or 0) or None,
		daily_loss_limit_quote=float(os.environ.get("DAILY_LOSS_LIMIT", "0") or 0) or None,
		wf_lookback_years=float(os.environ.get("WF_LOOKBACK_YEARS", "2.0")),
		wf_step_days=int(os.environ.get("WF_STEP_DAYS", "7")),
		sim_loop=(os.environ.get("SIM_LOOP", "0") == "1"),
		base_qty=float(os.environ.get("BASE_QTY", "40000")),
	)

	logger = NDJSONLogger(logs_dir=cfg.logs_dir, strategy_id=f"{cfg.strategy_name}_params", is_paper=cfg.paper)
	run_id = f"run_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H%M%S')}"
	logger.log_run_event(
		event_type="RUN_START",
		run_id=run_id,
		version="dev",
		mode="PAPER" if cfg.paper else "LIVE",
		config={"symbol": cfg.symbol, "timeframe": cfg.timeframe},
	)

	asset = cfg.symbol.replace("USDT", "USD") if cfg.symbol.endswith("USDT") else cfg.symbol

	# Summary accumulators
	gross_pnl = 0.0
	fees_quote = 0.0
	trades_count = 0
	wins = 0
	max_drawdown_quote = 0.0
	peak_equity = cfg.start_equity

	# Position state
	position_qty = 0.0
	position_entry_price: Optional[float] = None

	client = make_client()

	try:
		# Prepare walk-forward parameters
		bpd = timeframe_to_bars_per_day(cfg.timeframe)
		lookback_bars = max(bpd * 365 * int(round(cfg.wf_lookback_years)), bpd * 30)  # at least 30 days
		step_bars = max(1, bpd * cfg.wf_step_days)
		retrain_due_index = None  # type: Optional[int]
		model_strategy: Optional[BaseStrategy] = None
		i = 0
		last_bar_ts_str = None

		while True:
			# Reload data periodically to pick new bars if present
			df = load_latest_ohlcv(asset, cfg.timeframe)
			# Require only a minimal dataset (>= ~30 days or 200 bars) to start
			required_min_bars = max(200, timeframe_to_bars_per_day(cfg.timeframe) * 30)
			if len(df) < required_min_bars:
				time.sleep(cfg.poll_seconds)
				continue

			# Initialize i to start at the most recent bar once we have enough history
			if i < lookback_bars:
				i = len(df) - 1

			# Progress one bar per tick until we reach the end, then either loop or idle
			if i >= len(df):
				if cfg.sim_loop:
					i = lookback_bars
					continue
				else:
					time.sleep(cfg.poll_seconds)
					continue

			window = df.iloc[: i + 1]
			bar_ts = window.index[-1].isoformat()
			price = float(window[cfg.price_column].iloc[-1]) if cfg.price_column in window.columns else float(window["close"].iloc[-1])

			# Write bar snapshot
			bar = {k: (float(window[k].iloc[-1]) if k in window.columns else None) for k in ["open","high","low","close","volume"]}
			bar["timestamp"] = bar_ts
			if last_bar_ts_str != bar_ts:
				logger.log_bar_snapshot(cfg.symbol, cfg.timeframe, bar)
				last_bar_ts_str = bar_ts

			# Risk: daily loss limit (simple equity calc)
			current_equity = cfg.start_equity + gross_pnl - fees_quote
			if cfg.daily_loss_limit_quote is not None and (current_equity - cfg.start_equity) <= -abs(cfg.daily_loss_limit_quote):
				logger.log_error(severity="WARNING", source="RISK", message="Daily loss limit reached. Pausing trades.")
				i += 1
				time.sleep(cfg.poll_seconds)
				continue

			# Walk-forward retrain schedule
			if retrain_due_index is None or i >= retrain_due_index or model_strategy is None:
				# Use an effective lookback if the data is shorter than configured lookback
				effective_lb = max(50, min(lookback_bars, len(df) - 5))
				train_start = max(0, i - effective_lb)
				train_end = i
				calc_start = max(0, train_end - effective_lb)
				train_df = df.iloc[train_start:train_end]
				calc_df = df.iloc[calc_start:i+1]
				# Fresh model each retrain
				strategy_cls: type[BaseStrategy] = aVAILABLE_STRATEGIES[cfg.strategy_name]
				model_strategy = strategy_cls(price_column=cfg.price_column, **cfg.ml_params)
				model_strategy.optimize(train_df)
				retrain_due_index = i + step_bars

			# Generate current signal using fitted model on calc_df
			signals = model_strategy.generate_signals(calc_df)
			current_signal = float(signals.iloc[-1])

			# Trade logic: long-only enter/exit
			latency_ms = None
			execution = None
			order = None

			if current_signal > 0 and position_qty == 0.0:
				base_qty = float(cfg.base_qty)
				if cfg.max_position_notional is not None:
					base_qty = min(base_qty, abs(cfg.max_position_notional) / max(price, 1e-8))
				client_order_id = f"cli-{uuid.uuid4().hex[:8]}"
				order = {
					"event_type": "ORDER_PLACED",
					"client_order_id": client_order_id,
					"side": "BUY",
					"type": "MARKET",
					"time_in_force": "IOC",
					"quantity": base_qty,
					"price": price,
					"status": "NEW",
				}
				if client is not None:
					try:
						start = time.perf_counter()
						resp = client.new_order(symbol=cfg.symbol, side="BUY", type="MARKET", quoteOrderQty=min(cfg.market_quote_usdt, (cfg.max_position_notional or cfg.market_quote_usdt)), newClientOrderId=client_order_id)
						latency_ms = (time.perf_counter() - start) * 1000.0
						order["exchange_order_id"] = str(resp.get("orderId"))
						order["status"] = resp.get("status", "FILLED")
						fills = resp.get("fills") or []
						if fills:
							cum_quote = sum(float(f.get("price", 0)) * float(f.get("qty", 0)) for f in fills)
							cum_qty = sum(float(f.get("qty", 0)) for f in fills)
							execution = {
								"last_fill_qty": float(fills[-1].get("qty", 0)),
								"last_fill_price": float(fills[-1].get("price", 0)),
								"cumulative_qty": float(cum_qty),
								"cumulative_quote_qty": float(cum_quote),
								"average_fill_price": float(cum_quote / cum_qty) if cum_qty else None,
								"slippage": None,
								"fills": [
									{
										"fill_id": f"ex-{j}",
										"timestamp": datetime.now(timezone.utc).isoformat(),
										"price": float(f.get("price", 0)),
										"qty": float(f.get("qty", 0)),
										"quote_qty": float(f.get("price", 0)) * float(f.get("qty", 0)),
										"fee_asset": f.get("commissionAsset", cfg.quote_asset),
										"fee_amount": float(f.get("commission", 0)),
									}
									for j, f in enumerate(fills)
								],
						}
						fees_quote += sum(float(f.get("commission", 0)) for f in fills)
					except Exception as ex:
						logger.log_error(severity="WARNING", source="API", message="Buy failed", context={"error": str(ex)})
				if execution is None:
					execution = {
						"last_fill_qty": base_qty,
						"last_fill_price": price,
						"cumulative_qty": base_qty,
						"cumulative_quote_qty": base_qty * price,
						"average_fill_price": price,
						"slippage": 0.0,
						"fills": [
							{
								"fill_id": f"f-{uuid.uuid4().hex[:6]}",
								"timestamp": datetime.now(timezone.utc).isoformat(),
								"price": price,
								"qty": base_qty,
								"quote_qty": base_qty * price,
								"fee_asset": cfg.quote_asset,
								"fee_amount": 0.0,
							}
						]
					}
				# Synchronize order display fields from execution
				order["status"] = "FILLED"
				order["quote_quantity"] = float(execution.get("cumulative_quote_qty", base_qty * price))
				order["quantity"] = float(execution.get("cumulative_qty", base_qty))
				order["price"] = float(execution.get("average_fill_price", price))
				position_qty = float(execution["cumulative_qty"])  # now long
				position_entry_price = float(execution["average_fill_price"]) or price
				pnl_payload = {
					"unrealized_pnl": 0.0,
					"realized_pnl": None,
					"fees_total_quote": fees_quote,
					"position_qty_after": position_qty,
					"position_side": "LONG",
				}
				logger.log_trade_event("BINANCE", cfg.symbol, cfg.base_asset, cfg.quote_asset, order, execution, pnl=pnl_payload, latency_ms=latency_ms, run_id=run_id)

			elif current_signal <= 0 and position_qty > 0.0:
				close_qty = position_qty
				client_order_id = f"cli-{uuid.uuid4().hex[:8]}"
				order = {
					"event_type": "ORDER_PLACED",
					"client_order_id": client_order_id,
					"side": "SELL",
					"type": "MARKET",
					"time_in_force": "IOC",
					"quantity": close_qty,
					"price": price,
					"status": "NEW",
				}
				if client is not None:
					try:
						start = time.perf_counter()
						resp = client.new_order(symbol=cfg.symbol, side="SELL", type="MARKET", quantity=close_qty, newClientOrderId=client_order_id)
						latency_ms = (time.perf_counter() - start) * 1000.0
						order["exchange_order_id"] = str(resp.get("orderId"))
						order["status"] = resp.get("status", "FILLED")
						fills = resp.get("fills") or []
						if fills:
							cum_quote = sum(float(f.get("price", 0)) * float(f.get("qty", 0)) for f in fills)
							cum_qty = sum(float(f.get("qty", 0)) for f in fills)
							execution = {
								"last_fill_qty": float(fills[-1].get("qty", 0)),
								"last_fill_price": float(fills[-1].get("price", 0)),
								"cumulative_qty": float(cum_qty),
								"cumulative_quote_qty": float(cum_quote),
								"average_fill_price": float(cum_quote / cum_qty) if cum_qty else None,
								"slippage": None,
								"fills": [
									{
										"fill_id": f"ex-{j}",
										"timestamp": datetime.now(timezone.utc).isoformat(),
										"price": float(f.get("price", 0)),
										"qty": float(f.get("qty", 0)),
										"quote_qty": float(f.get("price", 0)) * float(f.get("qty", 0)),
										"fee_asset": f.get("commissionAsset", cfg.quote_asset),
										"fee_amount": float(f.get("commission", 0)),
									}
									for j, f in enumerate(fills)
								],
						}
						fees_quote += sum(float(f.get("commission", 0)) for f in fills)
					except Exception as ex:
						logger.log_error(severity="WARNING", source="API", message="Sell failed", context={"error": str(ex)})
				if execution is None:
					execution = {
						"last_fill_qty": close_qty,
						"last_fill_price": price,
						"cumulative_qty": close_qty,
						"cumulative_quote_qty": close_qty * price,
						"average_fill_price": price,
						"slippage": 0.0,
						"fills": [
							{
								"fill_id": f"f-{uuid.uuid4().hex[:6]}",
								"timestamp": datetime.now(timezone.utc).isoformat(),
								"price": price,
								"qty": close_qty,
								"quote_qty": close_qty * price,
								"fee_asset": cfg.quote_asset,
								"fee_amount": 0.0,
							}
						]
					}
				# Synchronize order display fields from execution
				order["status"] = "FILLED"
				order["quote_quantity"] = float(execution.get("cumulative_quote_qty", close_qty * price))
				order["quantity"] = float(execution.get("cumulative_qty", close_qty))
				order["price"] = float(execution.get("average_fill_price", price))

				realized = (float(execution["average_fill_price"]) - float(position_entry_price or price)) * close_qty
				gross_pnl += realized
				wins += 1 if realized > 0 else 0
				trades_count += 1
				position_qty = 0.0
				position_entry_price = None
				pnl_payload = {
					"unrealized_pnl": 0.0,
					"realized_pnl": realized,
					"fees_total_quote": fees_quote,
					"position_qty_after": position_qty,
					"position_side": None,
				}
				logger.log_trade_event("BINANCE", cfg.symbol, cfg.base_asset, cfg.quote_asset, order, execution, pnl=pnl_payload, latency_ms=latency_ms, run_id=run_id)

			# Update equity / dd
			unrealized = 0.0
			if position_qty > 0.0 and position_entry_price is not None:
				unrealized = (price - position_entry_price) * position_qty
			current_equity = cfg.start_equity + gross_pnl + unrealized - fees_quote
			if current_equity > peak_equity:
				peak_equity = current_equity
			else:
				dd = current_equity - peak_equity
				if dd < max_drawdown_quote:
					max_drawdown_quote = dd

			# heartbeat: account snapshot every N bars
			if i % max(1, bpd // 6) == 0:
				account = {
					"equity_quote": current_equity,
					"cash_quote": None,
					"unrealized_pnl_quote": unrealized,
					"positions": [
						{"symbol": cfg.symbol, "qty": position_qty, "entry_price": position_entry_price or price, "mark_price": price, "unrealized_pnl_quote": unrealized, "side": ("LONG" if position_qty >= 0 else "SHORT")}
					],
					"balances": [],
				}
				if client is not None:
					try:
						acc = client.account()
						balances = acc.get("balances", [])
						account["balances"] = [
							{"asset": b.get("asset"), "free": float(b.get("free", 0)), "locked": float(b.get("locked", 0))}
							for b in balances if b.get("asset") in {cfg.quote_asset, cfg.base_asset}
						]
					except Exception as ex:
						logger.log_error(severity="WARNING", source="API", message="Account fetch failed", context={"error": str(ex)})
				logger.log_account_snapshot(exchange="BINANCE", account=account, run_id=run_id)

			# Advance and wait next poll
			i += 1
			time.sleep(cfg.poll_seconds)

	finally:
		win_rate = (wins / trades_count) if trades_count else None
		logger.log_run_event(
			event_type="RUN_END",
			run_id=run_id,
			version="dev",
			mode="PAPER" if cfg.paper else "LIVE",
			summary={
				"trades": trades_count,
				"win_rate": win_rate,
				"gross_pnl_quote": gross_pnl,
				"fees_quote": -fees_quote,
				"net_pnl_quote": gross_pnl - fees_quote,
				"max_drawdown_quote": max_drawdown_quote,
			},
		)


if __name__ == "__main__":
	main()

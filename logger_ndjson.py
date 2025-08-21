from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional
from uuid import uuid4

from jsonschema import Draft202012Validator


class NDJSONLogger:
	"""Structured NDJSON logger with JSON Schema validation and daily rotation.

	Files under logs/ by default:
	  - trades_YYYY-MM-DD.ndjson
	  - account_YYYY-MM-DD.ndjson
	  - run_YYYY-MM-DD.ndjson
	  - errors_YYYY-MM-DD.ndjson
	  - trading.log (human-readable)
	"""

	def __init__(self, logs_dir: str = "logs", strategy_id: str = "unknown", is_paper: bool = True):
		self.logs_dir = Path(logs_dir)
		self.logs_dir.mkdir(parents=True, exist_ok=True)
		self.strategy_id = strategy_id
		self.is_paper = is_paper
		self._schemas = _build_schemas()
		self._validators = {k: Draft202012Validator(v) for k, v in self._schemas.items()}

	# ----------------------------- utilities ----------------------------- #
	def _now(self) -> str:
		return datetime.now(timezone.utc).isoformat()

	def _date_str(self) -> str:
		return datetime.now(timezone.utc).strftime("%Y-%m-%d")

	def _write_line(self, filename: str, payload: Dict[str, Any]) -> None:
		path = self.logs_dir / filename
		with open(path, "a", encoding="utf-8") as f:
			f.write(json.dumps(payload, ensure_ascii=False) + "\n")

	def _append_human_log(self, message: str) -> None:
		with open(self.logs_dir / "trading.log", "a", encoding="utf-8") as f:
			ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z")
			f.write(f"[{ts}] {message}\n")

	def _validate_and_write(self, schema_key: str, payload: Dict[str, Any], filename: str) -> None:
		validator = self._validators[schema_key]
		errors = sorted(validator.iter_errors(payload), key=lambda e: e.path)
		if errors:
			err_msgs = [f"{e.message} at /{'/'.join(map(str, e.path))}" for e in errors]
			self.log_error(
				severity="ERROR",
				source="SERIALIZATION",
				message="Schema validation failed",
				context={"schema": schema_key, "errors": err_msgs},
			)
			raise ValueError("Schema validation failed: " + "; ".join(err_msgs))
		self._write_line(filename, payload)

	# ----------------------------- public API ---------------------------- #
	def log_run_event(self, event_type: str, run_id: str, version: str, mode: str, config: Optional[Dict[str, Any]] = None, summary: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		payload = {
			"event_type": event_type,
			"event_id": str(uuid4()),
			"timestamp": self._now(),
			"run_id": run_id,
			"strategy_id": self.strategy_id,
			"version": version,
			"mode": mode,
		}
		if config is not None:
			payload["config"] = config
		if summary is not None:
			payload["summary"] = summary
		self._validate_and_write("run", payload, f"run_{self._date_str()}.ndjson")
		self._append_human_log(f"{event_type} run_id={run_id} mode={mode}")
		return payload

	def log_trade_event(self, exchange: str, symbol: str, base_asset: str, quote_asset: str, order: Dict[str, Any], execution: Optional[Dict[str, Any]] = None, pnl: Optional[Dict[str, Any]] = None, latency_ms: Optional[float] = None, error: Optional[Dict[str, Any]] = None, metadata: Optional[Dict[str, Any]] = None, run_id: Optional[str] = None) -> Dict[str, Any]:
		payload: Dict[str, Any] = {
			"event_type": order.get("event_type", "ORDER_PLACED"),
			"event_id": str(uuid4()),
			"timestamp": self._now(),
			"strategy_id": self.strategy_id,
			"run_id": run_id or "",
			"exchange": exchange,
			"is_paper": bool(self.is_paper),
			"symbol": symbol,
			"base_asset": base_asset,
			"quote_asset": quote_asset,
			"order": order,
		}
		if execution is not None:
			payload["execution"] = execution
		if pnl is not None:
			payload["pnl"] = pnl
		if latency_ms is not None:
			payload["latency_ms"] = latency_ms
		if error is not None:
			payload["error"] = error
		if metadata is not None:
			payload["metadata"] = metadata
		self._validate_and_write("trade", payload, f"trades_{self._date_str()}.ndjson")
		side = order.get("side", "?")
		qty = order.get("quantity", order.get("quote_quantity", "?"))
		price = order.get("price", "-")
		status = order.get("status", "-")
		self._append_human_log(f"TRADE {symbol} {side} qty={qty} price={price} status={status}")
		return payload

	def log_account_snapshot(self, exchange: str, account: Dict[str, Any], run_id: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		payload: Dict[str, Any] = {
			"event_type": "ACCOUNT_SNAPSHOT",
			"event_id": str(uuid4()),
			"timestamp": self._now(),
			"strategy_id": self.strategy_id,
			"run_id": run_id or "",
			"exchange": exchange,
			"is_paper": bool(self.is_paper),
			"account": account,
		}
		if metadata is not None:
			payload["metadata"] = metadata
		self._validate_and_write("account", payload, f"account_{self._date_str()}.ndjson")
		self._append_human_log("ACCOUNT_SNAPSHOT written")
		return payload

	def log_error(self, severity: str, source: str, message: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
		payload: Dict[str, Any] = {
			"event_type": "ERROR",
			"event_id": str(uuid4()),
			"timestamp": self._now(),
			"severity": severity,
			"source": source,
			"message": message,
		}
		if context is not None:
			payload["context"] = context
		self._validate_and_write("error", payload, f"errors_{self._date_str()}.ndjson")
		self._append_human_log(f"ERROR {severity} {source}: {message}")
		return payload

	def log_bar_snapshot(self, symbol: str, timeframe: str, bar: Dict[str, Any]) -> Dict[str, Any]:
		"""Write an OHLCV bar snapshot to NDJSON (no schema validation)."""
		filename = f"bars_{symbol}_{timeframe}_{self._date_str()}.ndjson"
		payload = {
			"symbol": symbol,
			"timeframe": timeframe,
			"bar": bar,
		}
		self._write_line(filename, payload)
		return payload

	def log_signal_snapshot(self, symbol: str, timeframe: str, signal: Dict[str, Any]) -> Dict[str, Any]:
		"""Write a signal snapshot to NDJSON (no schema validation)."""
		filename = f"signals_{symbol}_{timeframe}_{self._date_str()}.ndjson"
		payload = {
			"symbol": symbol,
			"timeframe": timeframe,
			"signal": signal,
		}
		self._write_line(filename, payload)
		return payload


# ------------------------------- Schemas -------------------------------- #

def _build_schemas() -> Dict[str, Dict[str, Any]]:
	trade_schema: Dict[str, Any] = {
		"$schema": "https://json-schema.org/draft/2020-12/schema",
		"$id": "https://yourdomain/schemas/trade_event.json",
		"title": "TradeEvent",
		"type": "object",
		"required": [
			"event_type", "event_id", "timestamp", "strategy_id",
			"exchange", "symbol", "base_asset", "quote_asset",
			"order"
		],
		"properties": {
			"event_type": {"type": "string", "enum": ["ORDER_PLACED","ORDER_ACK","PARTIAL_FILL","FILLED","CANCELED","REJECTED"]},
			"event_id": {"type": "string"},
			"timestamp": {"type": "string"},
			"strategy_id": {"type": "string"},
			"run_id": {"type": "string"},
			"exchange": {"type": "string"},
			"is_paper": {"type": "boolean"},
			"symbol": {"type": "string"},
			"base_asset": {"type": "string"},
			"quote_asset": {"type": "string"},
			"order": {
				"type": "object",
				"required": ["client_order_id","side","type","time_in_force","quantity","status"],
				"properties": {
					"exchange_order_id": {"type": ["string","null"]},
					"client_order_id": {"type": "string"},
					"side": {"type": "string"},
					"type": {"type": "string"},
					"time_in_force": {"type": "string"},
					"quantity": {"type": "number"},
					"quote_quantity": {"type": ["number","null"]},
					"price": {"type": ["number","null"]},
					"stop_price": {"type": ["number","null"]},
					"status": {"type": "string"},
					"post_only": {"type": ["boolean","null"]},
					"reduce_only": {"type": ["boolean","null"]},
					"leverage": {"type": ["integer","null"]},
				},
				"additionalProperties": True,
			},
			"execution": {"type": ["object","null"]},
			"pnl": {"type": ["object","null"]},
			"latency_ms": {"type": ["number","null"]},
			"error": {"type": ["object","null"]},
			"metadata": {"type": ["object","null"]},
		},
		"additionalProperties": False,
	}

	account_schema: Dict[str, Any] = {
		"$schema": "https://json-schema.org/draft/2020-12/schema",
		"$id": "https://yourdomain/schemas/account_snapshot.json",
		"title": "AccountSnapshot",
		"type": "object",
		"required": ["event_type","event_id","timestamp","strategy_id","exchange","is_paper","account"],
		"properties": {
			"event_type": {"type": "string", "const": "ACCOUNT_SNAPSHOT"},
			"event_id": {"type": "string"},
			"timestamp": {"type": "string"},
			"strategy_id": {"type": "string"},
			"run_id": {"type": "string"},
			"exchange": {"type": "string"},
			"is_paper": {"type": "boolean"},
			"account": {"type": "object"},
			"risk_limits": {"type": ["object","null"]},
			"metadata": {"type": ["object","null"]},
		},
		"additionalProperties": False,
	}

	run_schema: Dict[str, Any] = {
		"$schema": "https://json-schema.org/draft/2020-12/schema",
		"$id": "https://yourdomain/schemas/run_metadata.json",
		"title": "RunMetadata",
		"type": "object",
		"required": ["event_type","event_id","timestamp","run_id","strategy_id","version","mode"],
		"properties": {
			"event_type": {"type": "string"},
			"event_id": {"type": "string"},
			"timestamp": {"type": "string"},
			"run_id": {"type": "string"},
			"strategy_id": {"type": "string"},
			"version": {"type": "string"},
			"mode": {"type": "string"},
			"config": {"type": ["object","null"]},
			"summary": {"type": ["object","null"]},
		},
		"additionalProperties": False,
	}

	error_schema: Dict[str, Any] = {
		"$schema": "https://json-schema.org/draft/2020-12/schema",
		"$id": "https://yourdomain/schemas/error_event.json",
		"title": "ErrorEvent",
		"type": "object",
		"required": ["event_type","event_id","timestamp","severity","source","message"],
		"properties": {
			"event_type": {"type": "string"},
			"event_id": {"type": "string"},
			"timestamp": {"type": "string"},
			"severity": {"type": "string"},
			"source": {"type": "string"},
			"code": {"type": ["string","number","null"]},
			"message": {"type": "string"},
			"context": {"type": ["object","null"]},
		},
		"additionalProperties": False,
	}

	return {
		"trade": trade_schema,
		"account": account_schema,
		"run": run_schema,
		"error": error_schema,
	}

"""Parameter sweep runner for is_results.plot_cumulative_returns

Runs the in-sample evaluator many times while varying one or two parameters,
collects net Profit Factor and Sharpe ratio, and plots either:
- 1D: line charts vs a single parameter
- 2D: heatmaps vs two parameters

Examples
--------
Single parameter (1D):
    python param_sweep.py \
        --start 2018-07-25 --end 2024-10-31 \
        --strategy ml --asset VETUSD --tf 4h --price_column vwap \
        --fee_bps 10 --slippage_bps 10 \
        --x_param strategy_kwargs.lr --x_values 1e-5,2e-5,5e-5,1e-4 \
        --metrics pf_net,sharpe_net \
        --save_fig reports/sweep_lr.png

Two parameters (2D):
    python param_sweep.py \
        --start 2018-07-25 --end 2024-10-31 \
        --strategy ml --asset VETUSD --tf 4h --price_column vwap \
        --fee_bps 10 --slippage_bps 10 \
        --x_param strategy_kwargs.dropout_rate --x_values 0.3,0.4,0.5,0.6 \
        --y_param strategy_kwargs.weight_decay --y_values 0.0,0.0025,0.005,0.01 \
        --metrics pf_net,sharpe_net \
        --save_fig reports/sweep_dropout_wd.png

Notes
-----
- Metrics are obtained by importing is_results and calling plot_cumulative_returns
  with show_plot=False and a temporary JSON output directory. The script parses
  that JSON to gather metrics without modifying existing code.
- You can choose a preset base config for ML strategy params via --ml_preset.
  Otherwise defaults to is_results.ml_params.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt

try:
    import seaborn as sns  # type: ignore
    _HAS_SEABORN = True
except Exception:
    _HAS_SEABORN = False

import is_results  # local module in the same project


# --------------------------- Utilities ------------------------------------ #


def _parse_values_list(raw: str) -> List[Any]:
    """Parse a comma-separated list into typed Python values.

    Tries float -> int (if integral) -> bool -> str.
    """
    def parse_one(token: str) -> Any:
        t = token.strip()
        if t.lower() in {"true", "false"}:
            return t.lower() == "true"
        # Try numeric
        try:
            val = float(t)
            if math.isfinite(val) and abs(val - int(val)) < 1e-12:
                return int(val)
            return val
        except Exception:
            return t

    return [parse_one(x) for x in raw.split(",") if x.strip() != ""]


def _set_nested_param(target: Dict[str, Any], path: str, value: Any) -> None:
    """Set a nested parameter on the call-args dict given a dot-path.

    Supports paths like:
      - fee_bps
      - strategy_kwargs.lr
      - strategy_kwargs.signal_percentiles  (replaces entire object)
    """
    parts = [p for p in path.split(".") if p]
    node: Any = target
    for key in parts[:-1]:
        if key not in node or not isinstance(node[key], dict):
            node[key] = {}
        node = node[key]
    node[parts[-1]] = value


def _get_metrics_from_json(json_dir: str) -> Dict[str, Any]:
    json_path = os.path.join(json_dir, "is.json")
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("metrics", {})


@dataclass
class BaseRunParams:
    start: str
    end: str
    strategy: str
    asset: str
    tf: str
    price_column: str
    fee_bps: float
    slippage_bps: float
    strategy_kwargs: Dict[str, Any]


def _run_once(base: BaseRunParams) -> Dict[str, Any]:
    """Execute a single evaluation and return the metrics dict."""
    with TemporaryDirectory() as tmpdir:
        is_results.plot_cumulative_returns(
            start_date=base.start,
            end_date=base.end,
            strategy_name=base.strategy,
            asset=base.asset,
            timeframe=base.tf,
            price_column=base.price_column,
            strategy_kwargs=base.strategy_kwargs,
            show_plot=False,
            fee_bps=float(base.fee_bps),
            slippage_bps=float(base.slippage_bps),
            save_json_dir=tmpdir,
        )
        metrics = _get_metrics_from_json(tmpdir)
    return metrics


# --------------------------- Plotting ------------------------------------- #


def _plot_1d(x_vals: Sequence[Any], y_dict: Dict[str, List[float]], xlabel: str, title: str, save_fig: Optional[str]) -> None:
    n_series = len(y_dict)
    plt.figure(figsize=(8, 4 + max(0, n_series - 1)))
    for name, series in y_dict.items():
        plt.plot(x_vals, series, marker="o", label=name)
    plt.xlabel(xlabel)
    plt.ylabel("Metric value")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save_fig:
        os.makedirs(os.path.dirname(save_fig), exist_ok=True)
        plt.savefig(save_fig, dpi=150)
    else:
        plt.show()


def _plot_2d(
    x_vals: Sequence[Any],
    y_vals: Sequence[Any],
    z_mat: List[List[float]],
    xlabel: str,
    ylabel: str,
    title: str,
    save_fig: Optional[str],
    annotate: bool,
):
    plt.figure(figsize=(8, 6))
    if _HAS_SEABORN:
        ax = sns.heatmap(
            z_mat,
            annot=annotate,
            fmt=".3f" if annotate else "",
            xticklabels=[str(v) for v in x_vals],
            yticklabels=[str(v) for v in y_vals],
            cmap="viridis",
        )
    else:
        ax = plt.gca()
        im = ax.imshow(z_mat, cmap="viridis", aspect="auto", origin="upper")
        plt.colorbar(im, ax=ax)
        ax.set_xticks(range(len(x_vals)))
        ax.set_xticklabels([str(v) for v in x_vals], rotation=45, ha="right")
        ax.set_yticks(range(len(y_vals)))
        ax.set_yticklabels([str(v) for v in y_vals])
        if annotate:
            for i in range(len(y_vals)):
                for j in range(len(x_vals)):
                    ax.text(j, i, f"{z_mat[i][j]:.3f}", ha="center", va="center", color="white")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    if save_fig:
        os.makedirs(os.path.dirname(save_fig), exist_ok=True)
        plt.savefig(save_fig, dpi=150)
    else:
        plt.show()


# --------------------------- Main logic ----------------------------------- #


def _choose_ml_preset(name: str) -> Dict[str, Any]:
    name = name.lower().strip()
    if name in {"default", "ml_params"}:
        return deepcopy(getattr(is_results, "ml_params", {}))
    if name in {"conservative", "ml_params_conservative"}:
        return deepcopy(getattr(is_results, "ml_params_conservative", {}))
    if name in {"mc_safe", "ml_params_mc_safe"}:
        return deepcopy(getattr(is_results, "ml_params_mc_safe", {}))
    if name in {"mc_strict", "ml_params_mc_strict"}:
        return deepcopy(getattr(is_results, "ml_params_mc_strict", {}))
    raise ValueError(f"Unknown ml preset: {name}")


def main():
    parser = argparse.ArgumentParser(description="Sweep parameters for is_results and plot metrics")
    parser.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    parser.add_argument("--end", required=True, help="End date YYYY-MM-DD")
    parser.add_argument("--strategy", required=True, help="Strategy identifier, e.g. ml or ma")
    parser.add_argument("--asset", required=True, help="Asset symbol, e.g. BTCUSD")
    parser.add_argument("--tf", required=True, help="Timeframe, e.g. 1h, 4h, 1d")
    parser.add_argument("--price_column", default="close", help="Which price column to use (e.g. close, median, typical, vwap)")
    parser.add_argument("--fee_bps", type=float, default=0.0, help="Per-side fee in basis points")
    parser.add_argument("--slippage_bps", type=float, default=0.0, help="Per-side slippage in basis points")

    # Strategy kwargs base preset
    parser.add_argument("--ml_preset", default="default", help="Which ML preset to use for strategy_kwargs (default/conservative/mc_safe/mc_strict)")

    # Sweep parameters
    parser.add_argument("--x_param", required=True, help="Dot-path of the parameter to sweep (e.g. fee_bps or strategy_kwargs.lr)")
    parser.add_argument("--x_values", required=True, help="Comma-separated list of values for x_param")
    parser.add_argument("--y_param", default=None, help="Optional dot-path for a second parameter")
    parser.add_argument("--y_values", default=None, help="Comma-separated list of values for y_param")

    # Output controls
    parser.add_argument("--metrics", default="pf_net,sharpe_net", help="Comma-separated list of metrics to plot: pf_net, sharpe_net")
    parser.add_argument("--annotate", action="store_true", help="Annotate values onto heatmap")
    parser.add_argument("--save_fig", default=None, help="Path to save the figure; if omitted, shows interactively")
    parser.add_argument("--save_csv", default=None, help="Optional path to save a CSV of results")

    args = parser.parse_args()

    x_vals = _parse_values_list(args.x_values)
    y_vals = _parse_values_list(args.y_values) if args.y_param and args.y_values else None
    metrics_to_plot = [m.strip() for m in args.metrics.split(",") if m.strip()]

    # Build base params
    base_kwargs = _choose_ml_preset(args.ml_preset) if args.strategy.lower() == "ml" else {}
    base = BaseRunParams(
        start=args.start,
        end=args.end,
        strategy=args.strategy,
        asset=args.asset,
        tf=args.tf,
        price_column=args.price_column,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        strategy_kwargs=base_kwargs,
    )

    # Prepare CSV collection
    csv_rows: List[Dict[str, Any]] = []

    if not y_vals:
        # 1D sweep
        series_data: Dict[str, List[float]] = {m: [] for m in metrics_to_plot}
        for xv in x_vals:
            params = deepcopy(base)
            call_args: Dict[str, Any] = {
                "start": params.start,
                "end": params.end,
                "strategy": params.strategy,
                "asset": params.asset,
                "tf": params.tf,
                "price_column": params.price_column,
                "fee_bps": params.fee_bps,
                "slippage_bps": params.slippage_bps,
                "strategy_kwargs": deepcopy(params.strategy_kwargs),
            }
            _set_nested_param(call_args, args.x_param, xv)
            # Convert back to BaseRunParams for execution
            run_params = BaseRunParams(
                start=call_args["start"],
                end=call_args["end"],
                strategy=call_args["strategy"],
                asset=call_args["asset"],
                tf=call_args["tf"],
                price_column=call_args["price_column"],
                fee_bps=call_args["fee_bps"],
                slippage_bps=call_args["slippage_bps"],
                strategy_kwargs=call_args["strategy_kwargs"],
            )
            metrics = _run_once(run_params)
            row = {"x_param": args.x_param, "x_value": xv, **metrics}
            csv_rows.append(row)
            for m in metrics_to_plot:
                series_data[m].append(float(metrics.get(m, float("nan"))))

        title = f"1D sweep: {args.x_param}"
        _plot_1d(x_vals, series_data, xlabel=args.x_param, title=title, save_fig=args.save_fig)

    else:
        # 2D sweep
        mats: Dict[str, List[List[float]]] = {m: [[float("nan") for _ in x_vals] for _ in y_vals] for m in metrics_to_plot}
        for yi, yv in enumerate(y_vals):
            for xi, xv in enumerate(x_vals):
                params = deepcopy(base)
                call_args: Dict[str, Any] = {
                    "start": params.start,
                    "end": params.end,
                    "strategy": params.strategy,
                    "asset": params.asset,
                    "tf": params.tf,
                    "price_column": params.price_column,
                    "fee_bps": params.fee_bps,
                    "slippage_bps": params.slippage_bps,
                    "strategy_kwargs": deepcopy(params.strategy_kwargs),
                }
                _set_nested_param(call_args, args.x_param, xv)
                _set_nested_param(call_args, args.y_param, yv)  # type: ignore[arg-type]
                run_params = BaseRunParams(
                    start=call_args["start"],
                    end=call_args["end"],
                    strategy=call_args["strategy"],
                    asset=call_args["asset"],
                    tf=call_args["tf"],
                    price_column=call_args["price_column"],
                    fee_bps=call_args["fee_bps"],
                    slippage_bps=call_args["slippage_bps"],
                    strategy_kwargs=call_args["strategy_kwargs"],
                )
                metrics = _run_once(run_params)
                row = {
                    "x_param": args.x_param,
                    "x_value": xv,
                    "y_param": args.y_param,
                    "y_value": yv,
                    **metrics,
                }
                csv_rows.append(row)
                for m in metrics_to_plot:
                    mats[m][yi][xi] = float(metrics.get(m, float("nan")))

        # Create one subplot per metric
        for idx, m in enumerate(metrics_to_plot):
            title = f"2D sweep: {args.x_param} vs {args.y_param} | metric={m}"
            save = None
            if args.save_fig:
                base, ext = os.path.splitext(args.save_fig)
                save = f"{base}_{m}{ext or '.png'}"
            _plot_2d(x_vals, y_vals, mats[m], args.x_param, args.y_param or "", title, save, args.annotate)

    # Save CSV if requested
    if args.save_csv and csv_rows:
        import csv
        os.makedirs(os.path.dirname(args.save_csv), exist_ok=True)
        fieldnames = sorted({k for row in csv_rows for k in row.keys()})
        with open(args.save_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)


if __name__ == "__main__":
    main()



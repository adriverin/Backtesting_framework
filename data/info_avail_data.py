def load_asset_cache_info(interval: str = "1h", mode: str = "spot") -> dict:
    """
    Load and display asset OHLCV cache information for spot, futures, or both.

    Args:
        interval: Time interval to check info for
        mode: 'spot', 'futures', or 'both'

    Returns:
        Dict with keys per mode requested, e.g. {'spot': {...}, 'futures': {...}}
        or a single dict if mode != 'both'.
    """
    from pathlib import Path
    import json

    def _show(info_path: Path, label: str) -> dict:
        if not info_path.exists():
            print(f"❌ No cache info file found for {label}: {info_path}")
            print(f"💡 Run create_maximum_cache_for_assets(..., mode='{label}') first to generate cache info")
            return {}
        with open(info_path, 'r') as f:
            data = json.load(f)
        cache_type = data.get('cache_type', 'unknown')
        print("")
        print("=" * 70)
        print(f"📋 [{label.upper()}] Asset {cache_type.upper()} Cache Information ({interval} interval)")
        print("=" * 70)
        print(f"Created: {data.get('creation_date', '-')}")
        print(f"Cache Type: {cache_type.upper()} (Open, High, Low, Close, Volume)")
        rr = data.get('requested_range', {})
        print(f"Requested Range: {rr.get('start','?')} to {rr.get('end','?')}")
        print(f"Success Rate: {data.get('successful_assets',0)}/{data.get('total_assets',0)} assets")
        print()

        successful_assets = []
        failed_assets = []
        for asset, info in data.get('assets', {}).items():
            if info.get('status') == 'success':
                successful_assets.append((asset, info))
                print(f"✅ {asset}:")
                print(f"   📅 Available: {info.get('available_from','?')} to {info.get('available_to','?')}")
                print(f"   📊 Quality: {info.get('data_coverage_pct',0)}% coverage, {int(info.get('total_bars',0)):,} bars")
                if 'ohlcv_columns' in info:
                    print(f"   📈 Columns: {', '.join(info['ohlcv_columns'])}")
                print(f"   🗃️  Cache: {info.get('cache_file','?')}")
                if int(info.get('large_gaps_detected', 0)) > 0:
                    print(f"   ⚠️  {info.get('large_gaps_detected',0)} gaps detected")
                print()
            else:
                failed_assets.append((asset, info))

        if failed_assets:
            print("❌ Failed Assets:")
            for asset, info in failed_assets:
                print(f"   {asset}: {info.get('error', 'Unknown error')}")
            print()

        if successful_assets:
            print("📅 Available Date Ranges Summary:")
            for asset, info in successful_assets:
                print(f"   {asset}: {info.get('available_from','?')} → {info.get('available_to','?')}")
            print()

        if cache_type == 'ohlcv':
            print("🏷️  ML Training Price Column Options:")
            print("    • 'close': Close price (default)")
            print("    • 'open': Open price")
            print("    • 'high': High price")
            print("    • 'low': Low price")
            print("    • 'vwap': Calculated VWAP (H+L+C)/3")
            print("    • 'typical': Same as VWAP (H+L+C)/3")
            print("    • 'median': Median price (H+L)/2")

        return data

    m = (mode or "spot").strip().lower()
    if m == "both":
        spot_path = Path(f"data/spot/asset_cache_info_{interval}.json")
        fut_path = Path(f"data/futures/asset_cache_info_{interval}.json")
        spot = _show(spot_path, 'spot')
        fut = _show(fut_path, 'futures')
        return {"spot": spot, "futures": fut}
    else:
        base_dir = "data/futures" if m == "futures" else "data/spot"
        info_file = Path(f"{base_dir}/asset_cache_info_{interval}.json")
        return _show(info_file, m)


if __name__ == "__main__":
    # Show both spot and futures by default when run directly
    load_asset_cache_info("5m", mode="futures")
    

def load_asset_cache_info(interval: str = "1h") -> dict:
    """
    Load and display asset OHLCV cache information.
    
    Args:
        interval: Time interval to check info for
        
    Returns:
        Dictionary with asset cache information
    """
    from pathlib import Path
    import json
    
    info_file = Path(f"data/asset_cache_info_{interval}.json")
    
    if not info_file.exists():
        print(f"❌ No cache info file found: {info_file}")
        print(f"💡 Run create_maximum_cache_for_assets() first to generate cache info")
        return {}
    
    with open(info_file, 'r') as f:
        data = json.load(f)
    
    cache_type = data.get('cache_type', 'unknown')
    print(f"📋 Asset {cache_type.upper()} Cache Information ({interval} interval)")
    print("=" * 70)
    print(f"Created: {data['creation_date']}")
    print(f"Cache Type: {cache_type.upper()} (Open, High, Low, Close, Volume)")
    print(f"Requested Range: {data['requested_range']['start']} to {data['requested_range']['end']}")
    print(f"Success Rate: {data['successful_assets']}/{data['total_assets']} assets")
    print()
    
    successful_assets = []
    failed_assets = []
    
    for asset, info in data['assets'].items():
        if info['status'] == 'success':
            successful_assets.append((asset, info))
            print(f"✅ {asset}:")
            print(f"   📅 Available: {info['available_from']} to {info['available_to']}")
            print(f"   📊 Quality: {info['data_coverage_pct']}% coverage, {info['total_bars']:,} bars")
            if 'ohlcv_columns' in info:
                print(f"   📈 Columns: {', '.join(info['ohlcv_columns'])}")
            print(f"   🗃️  Cache: {info['cache_file']}")
            if int(info['large_gaps_detected']) > 0:
                print(f"   ⚠️  {info['large_gaps_detected']} gaps detected")
            print()
        else:
            failed_assets.append((asset, info))
    
    if failed_assets:
        print("❌ Failed Assets:")
        for asset, info in failed_assets:
            print(f"   {asset}: {info.get('error', 'Unknown error')}")
        print()
    
    # Summary of date ranges
    if successful_assets:
        print("📅 Available Date Ranges Summary:")
        for asset, info in successful_assets:
            print(f"   {asset}: {info['available_from']} → {info['available_to']}")
        print()
    
    # ML Training Usage Instructions (kept for convenience)
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


if __name__ == "__main__":
    load_asset_cache_info("1h")
    
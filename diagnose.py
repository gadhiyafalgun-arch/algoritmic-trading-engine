"""
diagnose.py — Run diagnostics to find out WHY strategies are failing
"""

import yaml
import sys
from src.data_pipeline import DataPipeline
from src.indicators import TechnicalIndicators
from src.strategy import TradingStrategy
from src.diagnostics import StrategyDiagnostics, run_diagnostics_for_all
from src.utils import ensure_directories
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")


def main():
    print("=" * 70)
    print("STRATEGY DIAGNOSTIC TOOL")
    print("=" * 70)

    ensure_directories()

    logger.info("Loading data...")
    pipeline = DataPipeline()
    data = pipeline.run_pipeline()

    logger.info("Adding indicators & strategies...")
    indicators = TechnicalIndicators()
    strategy = TradingStrategy()

    processed_data = {}
    for symbol, df in data.items():
        df = indicators.add_all_indicators(df)
        df = strategy.apply_all_strategies(df)
        processed_data[symbol] = df
        logger.info(f"   {symbol}: {len(df)} rows")

    # Find signal columns — but EXCLUDE macd_signal (that's the MACD indicator line, not a trade signal)
    sample_df = list(processed_data.values())[0]
    
    # Only use ACTUAL trade signal columns
    trade_signal_columns = [
        "sma_signal",
        "rsi_signal",
        "macd_trade_signal",
        "bb_signal",
        "combined_signal",
    ]
    
    # Filter to columns that actually exist
    valid_columns = [col for col in trade_signal_columns if col in sample_df.columns]

    print(f"\nTrade signal columns found: {valid_columns}")
    print(f"Total stocks: {len(processed_data)}")

    # Run diagnostics
    print("\n" + "=" * 70)
    print("RUNNING FULL DIAGNOSTICS...")
    print("=" * 70)

    summary = run_diagnostics_for_all(processed_data, valid_columns)

    print("\nDiagnostics complete!")
    print("Copy the output above and share it.\n")


if __name__ == "__main__":
    main()
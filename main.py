"""
Algorithmic Trading Engine
==========================
Main entry point for the application.

Author: Your Name
Version: 5.0.0 — Phase 5 (Machine Learning)
"""

from src.data_pipeline import DataPipeline
from src.indicators import TechnicalIndicators
from src.strategy import TradingStrategy
from src.backtester import Backtester
from src.performance import PerformanceAnalyzer
from src.risk_manager import RiskManager
from src.portfolio_manager import PortfolioManager
from src.visualizer import Visualizer
from src.utils import ensure_directories
from models.feature_engineer import FeatureEngineer
from models.train import MLTrainer
from models.predict import MLPredictor
from loguru import logger
import sys

# Setup logging
logger.remove()
logger.add(sys.stdout, level="INFO", format="{time:HH:mm:ss} | {level} | {message}")
logger.add("logs/main.log", rotation="5 MB", level="DEBUG")


def main():
    """Main function — runs the complete trading engine."""

    logger.info("=" * 60)
    logger.info("🚀 ALGORITHMIC TRADING ENGINE v5.0.0")
    logger.info("=" * 60)

    ensure_directories()

    # ==========================================
    # PHASE 1: Data Pipeline
    # ==========================================
    logger.info("\n📊 PHASE 1: Data Pipeline")
    pipeline = DataPipeline()
    data = pipeline.run_pipeline()

    summary = pipeline.get_summary(data)
    print("\n📋 DATA SUMMARY:")
    print(summary.to_string(index=False))

    # ==========================================
    # PHASE 2: Technical Indicators & Strategy
    # ==========================================
    logger.info("\n📈 PHASE 2: Technical Indicators & Strategy")

    indicators = TechnicalIndicators()
    strategy = TradingStrategy()

    processed_data = {}
    for symbol, df in data.items():
        logger.info(f"Processing {symbol}...")
        df = indicators.add_all_indicators(df)
        df = strategy.apply_all_strategies(df)
        processed_data[symbol] = df

    # ==========================================
    # PHASE 3: Basic Backtesting
    # ==========================================
    logger.info("\n🏃 PHASE 3: Basic Backtesting")

    backtester = Backtester()
    performance = PerformanceAnalyzer()
    visualizer = Visualizer()

    first_symbol = list(processed_data.keys())[0]
    first_df = processed_data[first_symbol]

    basic_results = backtester.run_multiple_strategies(
        first_df, first_symbol,
        ["sma_signal", "combined_signal"]
    )

    basic_metrics = {}
    for strat_name, results in basic_results.items():
        metrics = performance.calculate_all_metrics(
            results["portfolio_history"],
            results["trades"],
            results["initial_capital"]
        )
        basic_metrics[strat_name] = metrics

    if basic_metrics:
        comparison = performance.generate_comparison_report(basic_metrics)
        print(comparison)

    # ==========================================
    # PHASE 4: Risk Management
    # ==========================================
    logger.info("\n🛡️ PHASE 4: Risk Management")

    risk_manager = RiskManager()
    risk_report = risk_manager.generate_risk_report(
        processed_data,
        list(basic_results.values())[0]["final_value"] if basic_results else 100000
    )
    print(risk_report)

    # ==========================================
    # PHASE 5: Machine Learning
    # ==========================================
    logger.info("\n🧠 PHASE 5: Machine Learning")
    logger.info("=" * 60)

    # 5A: Feature Engineering
    logger.info("\n📐 Step 5A: Feature Engineering")
    feature_eng = FeatureEngineer()

    ml_processed = {}
    for symbol, df in processed_data.items():
        logger.info(f"Creating features for {symbol}...")
        ml_df = feature_eng.create_all_features(df)
        ml_processed[symbol] = ml_df

    # 5B: Prepare ML Data (using first stock)
    logger.info(f"\n📊 Step 5B: Preparing ML Data for {first_symbol}")
    ml_data = feature_eng.prepare_ml_data(ml_processed[first_symbol])

    if ml_data:
        # 5C: Train Models
        logger.info("\n🏋️ Step 5C: Training Models")
        trainer = MLTrainer()
        model_results = trainer.train_all_models(ml_data)

        # 5D: Walk-Forward Validation
        logger.info("\n🔄 Step 5D: Walk-Forward Validation")
        feature_cols = ml_data["feature_columns"]
        wf_results = trainer.walk_forward_validation(
            ml_processed[first_symbol],
            feature_cols,
            target_col="target_1d",
            n_splits=5
        )

        # Print ML Report
        ml_report = trainer.generate_ml_report(model_results, wf_results)
        print(ml_report)

        # 5E: Generate ML Signals
        logger.info("\n🎯 Step 5E: Generating ML Trading Signals")
        predictor = MLPredictor()

        for symbol, df in ml_processed.items():
            ml_processed[symbol] = predictor.generate_ml_signals(df)

        # 5F: Backtest ML Strategy
        logger.info("\n🏃 Step 5F: Backtesting ML Strategy")
        first_ml_df = ml_processed[first_symbol]

        if "ml_signal" in first_ml_df.columns and first_ml_df["ml_signal"].abs().sum() > 0:
            ml_backtest = backtester.run(
                first_ml_df,
                signal_column="ml_signal",
                symbol=first_symbol
            )

            if ml_backtest:
                ml_metrics = performance.calculate_all_metrics(
                    ml_backtest["portfolio_history"],
                    ml_backtest["trades"],
                    ml_backtest["initial_capital"]
                )

                ml_performance_report = performance.generate_report(
                    ml_metrics, first_symbol, "ML Strategy"
                )
                print(ml_performance_report)

                # Compare ML vs Traditional
                print(f"\n{'='*60}")
                print(f"{'🏆 ML vs TRADITIONAL COMPARISON':^60}")
                print(f"{'='*60}")

                all_compare = {**basic_metrics}
                all_compare["ML_Strategy"] = ml_metrics

                final_comparison = performance.generate_comparison_report(all_compare)
                print(final_comparison)
        else:
            logger.warning("ML model generated no signals — skipping ML backtest")

    # ==========================================
    # VISUALIZATION
    # ==========================================
    logger.info("\n📊 Generating Charts...")

    visualizer.plot_price_with_signals(
        first_df, first_symbol, signal_column="combined_signal"
    )

    if "combined_signal" in basic_results:
        visualizer.plot_backtest_results(
            basic_results["combined_signal"]["portfolio_history"],
            basic_results["combined_signal"]["trades"],
            first_symbol, "combined_signal",
            basic_results["combined_signal"]["initial_capital"]
        )

    visualizer.plot_equity_comparison(basic_results, first_symbol)

    # ==========================================
    # SAVE
    # ==========================================
    logger.info("\n💾 Saving all data...")
    pipeline.save_data(processed_data, data_type="processed")

    # ==========================================
    # FINAL
    # ==========================================
    logger.info("\n" + "=" * 60)
    logger.info("✅ ALL PHASES (1-5) COMPLETE!")
    logger.info("=" * 60)
    logger.info("📁 Charts: docs/charts/")
    logger.info("📁 Data: data/processed/")
    logger.info("📁 Models: models/saved/")
    logger.info("🔜 Next: Phase 6 — Dashboard & Polish")


if __name__ == "__main__":
    main()
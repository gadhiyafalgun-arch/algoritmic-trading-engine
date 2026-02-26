"""
Feature Engineering Module
============================
Transforms raw market data into meaningful features for ML models.

"In quant finance, 90% of the work is feature engineering."

Features created:
- Price-based features
- Volume-based features
- Technical indicator features
- Momentum features
- Volatility features
- Time-based features
- Statistical features
"""

import pandas as pd
import numpy as np
from loguru import logger


class FeatureEngineer:
    """
    Creates features from OHLCV data for machine learning models.
    
    Important Rules:
    1. NO LOOKAHEAD BIAS — only use past data to create features
    2. All features must be available at prediction time
    3. Handle NaN values properly
    """

    def __init__(self):
        logger.info("FeatureEngineer initialized")

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ALL features from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV + indicators
            
        Returns:
            DataFrame with all features added
        """
        if df.empty:
            return df

        df = df.copy()
        logger.info(f"Creating features... Starting columns: {len(df.columns)}")

        # Create each feature group
        df = self._price_features(df)
        df = self._volume_features(df)
        df = self._momentum_features(df)
        df = self._volatility_features(df)
        df = self._time_features(df)
        df = self._statistical_features(df)
        df = self._pattern_features(df)
        df = self._target_variable(df)

        logger.info(f"✅ Feature engineering complete — {len(df.columns)} total columns")
        return df

    def _price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price-based features."""

        # Returns over different periods
        for period in [1, 2, 3, 5, 10, 20]:
            df[f"return_{period}d"] = df["close"].pct_change(period)

        # Log returns
        for period in [1, 5, 10, 20]:
            df[f"log_return_{period}d"] = np.log(
                df["close"] / df["close"].shift(period)
            )

        # Price relative to moving averages
        if "sma_10" in df.columns:
            df["price_to_sma10"] = df["close"] / df["sma_10"] - 1
        if "sma_20" in df.columns:
            df["price_to_sma20"] = df["close"] / df["sma_20"] - 1
        if "sma_50" in df.columns:
            df["price_to_sma50"] = df["close"] / df["sma_50"] - 1
        if "sma_200" in df.columns:
            df["price_to_sma200"] = df["close"] / df["sma_200"] - 1

        # Price relative to high/low
        df["price_to_high_20d"] = df["close"] / df["high"].rolling(20).max() - 1
        df["price_to_low_20d"] = df["close"] / df["low"].rolling(20).min() - 1
        df["price_to_high_50d"] = df["close"] / df["high"].rolling(50).max() - 1
        df["price_to_low_50d"] = df["close"] / df["low"].rolling(50).min() - 1

        # Candle body features
        df["candle_body"] = (df["close"] - df["open"]) / df["open"]
        df["candle_upper_shadow"] = (df["high"] - df[["open", "close"]].max(axis=1)) / df["close"]
        df["candle_lower_shadow"] = (df[["open", "close"]].min(axis=1) - df["low"]) / df["close"]
        df["candle_range"] = (df["high"] - df["low"]) / df["close"]

        # Gap (open vs previous close)
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

        logger.debug("Price features created")
        return df

    def _volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volume-based features."""

        # Volume relative to average
        for period in [5, 10, 20]:
            vol_ma = df["volume"].rolling(period).mean()
            df[f"volume_ratio_{period}d"] = df["volume"] / vol_ma

        # Volume change
        df["volume_change_1d"] = df["volume"].pct_change(1)
        df["volume_change_5d"] = df["volume"].pct_change(5)

        # Volume trend
        df["volume_trend"] = df["volume"].rolling(10).mean() / df["volume"].rolling(30).mean()

        # On-Balance Volume (OBV) change
        obv = (np.sign(df["close"].diff()) * df["volume"]).cumsum()
        df["obv_change_5d"] = obv.pct_change(5)
        df["obv_change_10d"] = obv.pct_change(10)

        # Price-Volume relationship
        df["price_volume_corr_10d"] = (
            df["close"].pct_change()
            .rolling(10)
            .corr(df["volume"].pct_change())
        )

        logger.debug("Volume features created")
        return df

    def _momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Momentum-based features."""

        # RSI features
        if "rsi_14" in df.columns:
            df["rsi_change"] = df["rsi_14"].diff()
            df["rsi_above_50"] = (df["rsi_14"] > 50).astype(int)
            df["rsi_extreme_low"] = (df["rsi_14"] < 30).astype(int)
            df["rsi_extreme_high"] = (df["rsi_14"] > 70).astype(int)

        # MACD features
        if "macd_line" in df.columns and "macd_signal" in df.columns:
            df["macd_diff"] = df["macd_line"] - df["macd_signal"]
            df["macd_diff_change"] = df["macd_diff"].diff()
            df["macd_above_signal"] = (df["macd_line"] > df["macd_signal"]).astype(int)
            if "macd_histogram" in df.columns:
                df["macd_hist_change"] = df["macd_histogram"].diff()

        # Stochastic features
        if "stoch_k" in df.columns:
            df["stoch_diff"] = df["stoch_k"] - df["stoch_d"]
            df["stoch_extreme_low"] = (df["stoch_k"] < 20).astype(int)
            df["stoch_extreme_high"] = (df["stoch_k"] > 80).astype(int)

        # Bollinger Band features
        if "bb_percent_b" in df.columns:
            df["bb_position"] = df["bb_percent_b"]
            df["bb_squeeze"] = df.get("bb_bandwidth", pd.Series(0, index=df.index))

        # SMA crossover features
        if "sma_20" in df.columns and "sma_50" in df.columns:
            df["sma_20_above_50"] = (df["sma_20"] > df["sma_50"]).astype(int)
            df["sma_20_50_diff"] = (df["sma_20"] - df["sma_50"]) / df["sma_50"]

        if "sma_50" in df.columns and "sma_200" in df.columns:
            df["sma_50_above_200"] = (df["sma_50"] > df["sma_200"]).astype(int)

        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f"roc_{period}d"] = (
                (df["close"] - df["close"].shift(period)) / df["close"].shift(period)
            )

        logger.debug("Momentum features created")
        return df

    def _volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Volatility-based features."""

        # Historical volatility over different windows
        for period in [5, 10, 20, 30]:
            df[f"volatility_{period}d"] = (
                df["close"].pct_change().rolling(period).std() * np.sqrt(252)
            )

        # Volatility change
        df["volatility_change"] = df["volatility_20d"] / df["volatility_20d"].shift(5) - 1

        # ATR features
        if "atr_14" in df.columns:
            df["atr_ratio"] = df["atr_14"] / df["close"]
            df["atr_change"] = df["atr_14"].pct_change(5)

        # Realized vs expected volatility
        if "volatility_10d" in df.columns and "volatility_20d" in df.columns:
            df["vol_ratio_10_20"] = df["volatility_10d"] / df["volatility_20d"]

        # Parkinson volatility (uses high-low range)
        df["parkinson_vol"] = np.sqrt(
            (1 / (4 * np.log(2))) * 
            (np.log(df["high"] / df["low"]) ** 2).rolling(20).mean()
        )

        logger.debug("Volatility features created")
        return df

    def _time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Time-based features."""

        df["day_of_week"] = df.index.dayofweek        # 0=Monday, 4=Friday
        df["day_of_month"] = df.index.day
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["is_month_start"] = df.index.is_month_start.astype(int)
        df["is_month_end"] = df.index.is_month_end.astype(int)
        df["is_quarter_start"] = df.index.is_quarter_start.astype(int)
        df["is_quarter_end"] = df.index.is_quarter_end.astype(int)

        # Days since year start
        df["day_of_year"] = df.index.dayofyear

        # Cyclical encoding (so model knows Dec 31 is close to Jan 1)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 5)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 5)

        logger.debug("Time features created")
        return df

    def _statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Statistical features."""

        returns = df["close"].pct_change()

        # Skewness (asymmetry of returns)
        for period in [20, 50]:
            df[f"skewness_{period}d"] = returns.rolling(period).skew()

        # Kurtosis (tail risk)
        for period in [20, 50]:
            df[f"kurtosis_{period}d"] = returns.rolling(period).kurt()

        # Z-Score of price (how many std devs from mean)
        for period in [20, 50]:
            rolling_mean = df["close"].rolling(period).mean()
            rolling_std = df["close"].rolling(period).std()
            df[f"zscore_{period}d"] = (df["close"] - rolling_mean) / rolling_std

        # Autocorrelation of returns
        df["autocorr_5d"] = returns.rolling(20).apply(
            lambda x: x.autocorr(lag=5) if len(x) > 5 else 0,
            raw=False
        )

        # Max drawdown over rolling window
        for period in [20, 50]:
            rolling_max = df["close"].rolling(period).max()
            df[f"max_drawdown_{period}d"] = (df["close"] - rolling_max) / rolling_max

        logger.debug("Statistical features created")
        return df

    def _pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Price pattern features."""

        # Consecutive up/down days
        daily_direction = (df["close"].diff() > 0).astype(int)

        # Count consecutive same-direction days
        df["consecutive_up"] = daily_direction.groupby(
            (daily_direction != daily_direction.shift()).cumsum()
        ).cumcount() + 1
        df["consecutive_up"] = df["consecutive_up"] * daily_direction

        daily_down = (df["close"].diff() < 0).astype(int)
        df["consecutive_down"] = daily_down.groupby(
            (daily_down != daily_down.shift()).cumsum()
        ).cumcount() + 1
        df["consecutive_down"] = df["consecutive_down"] * daily_down

        # Higher highs / Lower lows
        df["higher_high"] = (df["high"] > df["high"].shift(1)).astype(int)
        df["lower_low"] = (df["low"] < df["low"].shift(1)).astype(int)

        # Inside/Outside bars
        df["inside_bar"] = (
            (df["high"] < df["high"].shift(1)) & 
            (df["low"] > df["low"].shift(1))
        ).astype(int)

        df["outside_bar"] = (
            (df["high"] > df["high"].shift(1)) & 
            (df["low"] < df["low"].shift(1))
        ).astype(int)

        logger.debug("Pattern features created")
        return df

    def _target_variable(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the TARGET variable for ML model.
        
        Target: Will the stock go UP or DOWN in the next N days?
        
        1 = Price will go UP (BUY opportunity)
        0 = Price will go DOWN (SELL/HOLD)
        
        IMPORTANT: This uses FUTURE data — only for training labels!
        Never use this as a feature!
        """
        # Forward return (next day)
        df["forward_return_1d"] = df["close"].shift(-1) / df["close"] - 1

        # Forward return (next 5 days)
        df["forward_return_5d"] = df["close"].shift(-5) / df["close"] - 1

        # Binary target: 1 if next day return > 0
        df["target_1d"] = (df["forward_return_1d"] > 0).astype(int)

        # Binary target: 1 if next 5 day return > 0
        df["target_5d"] = (df["forward_return_5d"] > 0).astype(int)

        # Multi-class target for more nuanced predictions
        # 2 = Strong up (>1%), 1 = Slight up, 0 = Slight down, -1 = Strong down (<-1%)
        df["target_multi"] = 0
        df.loc[df["forward_return_1d"] > 0.01, "target_multi"] = 2
        df.loc[
            (df["forward_return_1d"] > 0) & (df["forward_return_1d"] <= 0.01),
            "target_multi"
        ] = 1
        df.loc[
            (df["forward_return_1d"] < 0) & (df["forward_return_1d"] >= -0.01),
            "target_multi"
        ] = -1
        df.loc[df["forward_return_1d"] < -0.01, "target_multi"] = -2

        logger.debug("Target variables created")
        return df

    def get_feature_columns(self, df: pd.DataFrame) -> list:
        """
        Get list of columns that should be used as FEATURES (not targets).
        
        Excludes:
        - OHLCV raw data
        - Target variables
        - Non-numeric columns
        """
        exclude_cols = [
            "open", "high", "low", "close", "volume",
            "symbol",
            "forward_return_1d", "forward_return_5d",
            "target_1d", "target_5d", "target_multi",
            "sma_signal", "rsi_signal", "macd_trade_signal",
            "bb_signal", "combined_signal",
            "sma_position", "macd_position",
            "combined_buy_score", "combined_sell_score"
        ]

        feature_cols = [
            col for col in df.columns
            if col not in exclude_cols
            and df[col].dtype in ["float64", "int64", "int32", "float32"]
        ]

        logger.info(f"📊 Feature columns: {len(feature_cols)}")
        return feature_cols

    def prepare_ml_data(self, df: pd.DataFrame, target_col: str = "target_1d",
                        test_ratio: float = 0.2) -> dict:
        """
        Prepare data for ML model training.
        
        IMPORTANT: Uses TIME-BASED split, NOT random split!
        Random split would cause lookahead bias.
        
        Args:
            df: DataFrame with all features
            target_col: Target column name
            test_ratio: Fraction of data for testing
            
        Returns:
            Dictionary with train/test splits
        """
        feature_cols = self.get_feature_columns(df)

        # Drop rows with NaN
        clean_df = df.dropna(subset=feature_cols + [target_col])

        if clean_df.empty:
            logger.error("No clean data available after dropping NaN!")
            return {}

        # TIME-BASED SPLIT (NOT random!)
        split_idx = int(len(clean_df) * (1 - test_ratio))

        train_df = clean_df.iloc[:split_idx]
        test_df = clean_df.iloc[split_idx:]

        X_train = train_df[feature_cols]
        y_train = train_df[target_col]
        X_test = test_df[feature_cols]
        y_test = test_df[target_col]

        logger.info(f"📊 ML Data Prepared:")
        logger.info(f"   Features: {len(feature_cols)}")
        logger.info(f"   Training: {len(X_train)} samples ({train_df.index.min().strftime('%Y-%m-%d')} to {train_df.index.max().strftime('%Y-%m-%d')})")
        logger.info(f"   Testing:  {len(X_test)} samples ({test_df.index.min().strftime('%Y-%m-%d')} to {test_df.index.max().strftime('%Y-%m-%d')})")
        logger.info(f"   Target distribution (train): {y_train.value_counts().to_dict()}")

        return {
            "X_train": X_train,
            "y_train": y_train,
            "X_test": X_test,
            "y_test": y_test,
            "feature_columns": feature_cols,
            "train_dates": train_df.index,
            "test_dates": test_df.index,
            "train_df": train_df,
            "test_df": test_df
        }
"""
ML Prediction Module
=====================
Uses trained ML models to generate trading signals.
"""

import pandas as pd
import numpy as np
import joblib
import json
import os
from loguru import logger


class MLPredictor:
    """
    Generates trading signals from ML model predictions.
    """

    def __init__(self, model_dir: str = "models/saved"):
        self.model_dir = model_dir
        self.model = None
        self.scaler = None
        self.feature_columns = None

        self._load_model()
        logger.info("MLPredictor initialized")

    def _load_model(self):
        """Load saved model, scaler, and feature columns."""
        try:
            model_path = os.path.join(self.model_dir, "best_model.joblib")
            scaler_path = os.path.join(self.model_dir, "scaler.joblib")
            features_path = os.path.join(self.model_dir, "feature_columns.json")

            if os.path.exists(model_path):
                self.model = joblib.load(model_path)
                logger.info("✅ Model loaded")
            else:
                logger.warning("No saved model found")

            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                logger.info("✅ Scaler loaded")

            if os.path.exists(features_path):
                with open(features_path, "r") as f:
                    self.feature_columns = json.load(f)
                logger.info(f"✅ Feature columns loaded ({len(self.feature_columns)} features)")

        except Exception as e:
            logger.error(f"Error loading model: {e}")

    def generate_ml_signals(self, df: pd.DataFrame,
                             confidence_threshold: float = 0.55) -> pd.DataFrame:
        """
        Generate trading signals from ML model.
        
        Args:
            df: DataFrame with all features
            confidence_threshold: Minimum probability to generate signal
                                  0.55 = model must be at least 55% confident
                                  
        Returns:
            DataFrame with ML signals added
        """
        df = df.copy()

        if self.model is None or self.feature_columns is None:
            logger.error("Model not loaded! Train first.")
            df["ml_signal"] = 0
            df["ml_confidence"] = 0.5
            return df

        # Check which features are available
        available_features = [f for f in self.feature_columns if f in df.columns]
        missing_features = [f for f in self.feature_columns if f not in df.columns]

        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features — using available ones")

        if not available_features:
            logger.error("No features available for prediction!")
            df["ml_signal"] = 0
            df["ml_confidence"] = 0.5
            return df

        # Prepare features
        X = df[available_features].copy()

        # Handle NaN
        X = X.fillna(method="ffill").fillna(0)

        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        # Get probability of class 1 (price goes up)
        if probabilities.shape[1] == 2:
            confidence = probabilities[:, 1]
        else:
            confidence = probabilities.max(axis=1)

        # Generate signals based on confidence threshold
        df["ml_prediction"] = predictions
        df["ml_confidence"] = confidence
        df["ml_signal"] = 0

        # BUY: Model predicts UP with high confidence
        buy_mask = (predictions == 1) & (confidence >= confidence_threshold)

        # Only signal on CHANGES (not every day)
        buy_change = buy_mask.astype(int).diff().fillna(0)
        df.loc[buy_change == 1, "ml_signal"] = 1

        # SELL: Model predicts DOWN with high confidence
        sell_mask = (predictions == 0) & (confidence >= confidence_threshold)
        sell_change = sell_mask.astype(int).diff().fillna(0)
        df.loc[sell_change == 1, "ml_signal"] = -1

        buys = (df["ml_signal"] == 1).sum()
        sells = (df["ml_signal"] == -1).sum()
        logger.info(f"📊 ML Signals: {buys} BUYs, {sells} SELLs "
                   f"(threshold: {confidence_threshold})")

        return df
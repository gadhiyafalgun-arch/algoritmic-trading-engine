"""
ML Model Training Module
==========================
Trains machine learning models to predict stock price movements.

Models:
1. Random Forest Classifier
2. XGBoost Classifier
3. Logistic Regression (baseline)

Validation:
- Walk-Forward Validation (proper time series CV)
- NO random train-test split (prevents lookahead bias)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
import xgboost as xgb
from loguru import logger
import joblib
import os
import json


class MLTrainer:
    """
    Trains and evaluates ML models for trading signal prediction.
    
    Key Principles:
    1. TIME-BASED splits only (no random)
    2. Walk-forward validation
    3. Feature scaling
    4. Prevent overfitting
    5. Track feature importance
    """

    def __init__(self, model_dir: str = "models/saved"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        logger.info("MLTrainer initialized")

    def train_all_models(self, ml_data: dict) -> dict:
        """
        Train all ML models and compare them.
        
        Args:
            ml_data: Dictionary from FeatureEngineer.prepare_ml_data()
            
        Returns:
            Dictionary with all model results
        """
        if not ml_data:
            logger.error("No ML data provided!")
            return {}

        X_train = ml_data["X_train"]
        y_train = ml_data["y_train"]
        X_test = ml_data["X_test"]
        y_test = ml_data["y_test"]

        # Scale features
        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        all_results = {}

        # Model 1: Logistic Regression (Baseline)
        logger.info("\n🔵 Training Logistic Regression (Baseline)...")
        lr_results = self._train_logistic_regression(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        all_results["logistic_regression"] = lr_results

        # Model 2: Random Forest
        logger.info("\n🟢 Training Random Forest...")
        rf_results = self._train_random_forest(
            X_train, y_train, X_test, y_test
        )
        all_results["random_forest"] = rf_results

        # Model 3: XGBoost
        logger.info("\n🟡 Training XGBoost...")
        xgb_results = self._train_xgboost(
            X_train, y_train, X_test, y_test
        )
        all_results["xgboost"] = xgb_results

        self.results = all_results

        # Print comparison
        self._print_model_comparison(all_results)

        # Save best model
        self._save_best_model(all_results, ml_data["feature_columns"])

        return all_results

    def _train_logistic_regression(self, X_train, y_train,
                                    X_test, y_test) -> dict:
        """Train Logistic Regression as baseline."""
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            C=0.1  # Regularization to prevent overfitting
        )

        model.fit(X_train, y_train)
        self.models["logistic_regression"] = model

        return self._evaluate_model(model, X_train, y_train,
                                     X_test, y_test, "Logistic Regression")

    def _train_random_forest(self, X_train, y_train,
                              X_test, y_test) -> dict:
        """Train Random Forest Classifier."""
        model = RandomForestClassifier(
            n_estimators=200,       # Number of trees
            max_depth=10,           # Prevent overfitting
            min_samples_split=20,   # Prevent overfitting
            min_samples_leaf=10,    # Prevent overfitting
            max_features="sqrt",    # Use sqrt(n) features per tree
            random_state=42,
            n_jobs=-1               # Use all CPU cores
        )

        model.fit(X_train, y_train)
        self.models["random_forest"] = model

        results = self._evaluate_model(model, X_train, y_train,
                                        X_test, y_test, "Random Forest")

        # Feature importance
        importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)

        results["feature_importance"] = importance
        results["top_20_features"] = importance.head(20)

        logger.info("\n📊 Top 10 Features (Random Forest):")
        for feat, imp in importance.head(10).items():
            bar = "█" * int(imp * 200)
            logger.info(f"   {feat:<30} {imp:.4f} {bar}")

        return results

    def _train_xgboost(self, X_train, y_train,
                        X_test, y_test) -> dict:
        """Train XGBoost Classifier."""
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,     # Slow learning = better generalization
            subsample=0.8,          # Use 80% of data per tree
            colsample_bytree=0.8,   # Use 80% of features per tree
            min_child_weight=5,
            gamma=0.1,              # Regularization
            reg_alpha=0.1,          # L1 regularization
            reg_lambda=1.0,         # L2 regularization
            random_state=42,
            use_label_encoder=False,
            eval_metric="logloss",
            verbosity=0
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            verbose=False
        )

        self.models["xgboost"] = model

        results = self._evaluate_model(model, X_train, y_train,
                                        X_test, y_test, "XGBoost")

        # Feature importance
        importance = pd.Series(
            model.feature_importances_,
            index=X_train.columns
        ).sort_values(ascending=False)

        results["feature_importance"] = importance
        results["top_20_features"] = importance.head(20)

        logger.info("\n📊 Top 10 Features (XGBoost):")
        for feat, imp in importance.head(10).items():
            bar = "█" * int(imp * 200)
            logger.info(f"   {feat:<30} {imp:.4f} {bar}")

        return results

    def _evaluate_model(self, model, X_train, y_train,
                         X_test, y_test, model_name: str) -> dict:
        """Evaluate a trained model."""

        # Predictions
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)

        # Probabilities (for confidence)
        train_proba = model.predict_proba(X_train)[:, 1]
        test_proba = model.predict_proba(X_test)[:, 1]

        # Metrics
        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)
        test_precision = precision_score(y_test, test_pred, zero_division=0)
        test_recall = recall_score(y_test, test_pred, zero_division=0)
        test_f1 = f1_score(y_test, test_pred, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_test, test_pred)

        results = {
            "model_name": model_name,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
            "confusion_matrix": cm,
            "test_predictions": test_pred,
            "test_probabilities": test_proba,
            "overfit_gap": train_accuracy - test_accuracy
        }

        # Log results
        logger.info(f"\n📊 {model_name} Results:")
        logger.info(f"   Train Accuracy: {train_accuracy:.4f}")
        logger.info(f"   Test Accuracy:  {test_accuracy:.4f}")
        logger.info(f"   Test Precision: {test_precision:.4f}")
        logger.info(f"   Test Recall:    {test_recall:.4f}")
        logger.info(f"   Test F1:        {test_f1:.4f}")
        logger.info(f"   Overfit Gap:    {results['overfit_gap']:.4f}")

        if results["overfit_gap"] > 0.1:
            logger.warning(f"   ⚠️ HIGH OVERFITTING detected! Gap: {results['overfit_gap']:.4f}")
        elif results["overfit_gap"] > 0.05:
            logger.warning(f"   ⚠️ Moderate overfitting. Gap: {results['overfit_gap']:.4f}")
        else:
            logger.info(f"   ✅ Low overfitting. Good generalization!")

        return results

    def walk_forward_validation(self, df: pd.DataFrame,
                                 feature_cols: list,
                                 target_col: str = "target_1d",
                                 n_splits: int = 5,
                                 train_ratio: float = 0.7) -> dict:
        """
        Walk-Forward Validation
        
        This is the GOLD STANDARD for time series model validation.
        
        How it works:
            Split 1: Train [====] Test [==]
            Split 2: Train [======] Test [==]
            Split 3: Train [========] Test [==]
            Split 4: Train [==========] Test [==]
            
        The training window GROWS over time, just like in real trading.
        """
        logger.info(f"\n🔄 Walk-Forward Validation ({n_splits} splits)")

        clean_df = df.dropna(subset=feature_cols + [target_col])

        if len(clean_df) < 100:
            logger.error("Not enough data for walk-forward validation!")
            return {}

        total_len = len(clean_df)
        test_size = int(total_len * (1 - train_ratio) / n_splits)

        all_predictions = []
        all_actuals = []
        split_results = []

        for i in range(n_splits):
            # Calculate split points
            test_end = total_len - (n_splits - i - 1) * test_size
            test_start = test_end - test_size
            train_end = test_start

            if train_end < 50:  # Need minimum training data
                continue

            # Split data
            train_data = clean_df.iloc[:train_end]
            test_data = clean_df.iloc[test_start:test_end]

            X_train = train_data[feature_cols]
            y_train = train_data[target_col]
            X_test = test_data[feature_cols]
            y_test = test_data[target_col]

            # Train XGBoost on this split
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0
            )

            model.fit(X_train, y_train, verbose=False)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)

            split_results.append({
                "split": i + 1,
                "train_size": len(X_train),
                "test_size": len(X_test),
                "train_period": f"{train_data.index[0].strftime('%Y-%m-%d')} to {train_data.index[-1].strftime('%Y-%m-%d')}",
                "test_period": f"{test_data.index[0].strftime('%Y-%m-%d')} to {test_data.index[-1].strftime('%Y-%m-%d')}",
                "accuracy": accuracy
            })

            all_predictions.extend(predictions)
            all_actuals.extend(y_test.values)

            logger.info(f"   Split {i+1}: Train={len(X_train)}, Test={len(X_test)}, "
                       f"Accuracy={accuracy:.4f}")

        # Overall results
        overall_accuracy = accuracy_score(all_actuals, all_predictions)
        avg_accuracy = np.mean([r["accuracy"] for r in split_results])

        wf_results = {
            "split_results": split_results,
            "overall_accuracy": overall_accuracy,
            "avg_accuracy": avg_accuracy,
            "std_accuracy": np.std([r["accuracy"] for r in split_results]),
            "all_predictions": all_predictions,
            "all_actuals": all_actuals
        }

        logger.info(f"\n📊 Walk-Forward Results:")
        logger.info(f"   Overall Accuracy: {overall_accuracy:.4f}")
        logger.info(f"   Average Accuracy: {avg_accuracy:.4f}")
        logger.info(f"   Std Deviation:    {wf_results['std_accuracy']:.4f}")

        return wf_results

    def _print_model_comparison(self, all_results: dict) -> None:
        """Print comparison of all models."""

        print(f"\n{'='*70}")
        print(f"{'🏆 MODEL COMPARISON':^70}")
        print(f"{'='*70}")
        print(f"{'Model':<25} {'Test Acc':>10} {'Precision':>10} {'F1':>10} {'Overfit':>10}")
        print(f"{'-'*70}")

        for name, results in all_results.items():
            print(f"{results['model_name']:<25} "
                  f"{results['test_accuracy']:>10.4f} "
                  f"{results['test_precision']:>10.4f} "
                  f"{results['test_f1']:>10.4f} "
                  f"{results['overfit_gap']:>10.4f}")

        print(f"{'='*70}")

        # Find best model
        best = max(all_results.items(), key=lambda x: x[1]["test_f1"])
        print(f"\n🏆 Best Model (by F1): {best[1]['model_name']}")
        print()

    def _save_best_model(self, all_results: dict, feature_columns: list) -> None:
        """Save the best performing model."""

        best_name = max(all_results.items(), key=lambda x: x[1]["test_f1"])[0]
        best_model = self.models[best_name]

        # Save model
        model_path = os.path.join(self.model_dir, "best_model.joblib")
        joblib.dump(best_model, model_path)

        # Save scaler
        scaler_path = os.path.join(self.model_dir, "scaler.joblib")
        joblib.dump(self.scaler, scaler_path)

        # Save feature columns
        features_path = os.path.join(self.model_dir, "feature_columns.json")
        with open(features_path, "w") as f:
            json.dump(feature_columns, f)

        # Save model info
        info = {
            "best_model": best_name,
            "test_accuracy": all_results[best_name]["test_accuracy"],
            "test_f1": all_results[best_name]["test_f1"],
            "n_features": len(feature_columns)
        }
        info_path = os.path.join(self.model_dir, "model_info.json")
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)

        logger.info(f"💾 Best model ({best_name}) saved to {self.model_dir}/")

    def generate_ml_report(self, all_results: dict, wf_results: dict = None) -> str:
        """Generate ML performance report."""

        report = f"""
╔══════════════════════════════════════════════════════════════╗
║              🧠 MACHINE LEARNING REPORT                     ║
╠══════════════════════════════════════════════════════════════╣
"""

        for name, results in all_results.items():
            report += f"""
║  📊 {results['model_name']:<40}               ║
║  ─────────────────────────────────                           ║
║  Train Accuracy:    {results['train_accuracy']:>10.4f}                        ║
║  Test Accuracy:     {results['test_accuracy']:>10.4f}                        ║
║  Precision:         {results['test_precision']:>10.4f}                        ║
║  Recall:            {results['test_recall']:>10.4f}                        ║
║  F1 Score:          {results['test_f1']:>10.4f}                        ║
║  Overfit Gap:       {results['overfit_gap']:>10.4f}                        ║
║                                                              ║"""

        if wf_results:
            report += f"""
║  🔄 WALK-FORWARD VALIDATION                                 ║
║  ─────────────────────────────────                           ║
║  Overall Accuracy:  {wf_results['overall_accuracy']:>10.4f}                        ║
║  Average Accuracy:  {wf_results['avg_accuracy']:>10.4f}                        ║
║  Std Deviation:     {wf_results['std_accuracy']:>10.4f}                        ║
║                                                              ║"""

        report += """
╚══════════════════════════════════════════════════════════════╝
"""
        return report
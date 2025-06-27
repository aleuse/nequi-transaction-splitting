"""
Machine Learning models for transaction splitting detection.
"""

import pickle
import numpy as np
import polars as pl
from typing import Tuple, Optional, Dict, Any
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from .config import (
    DEFAULT_IF_PARAMS,
    DEFAULT_DBSCAN_PARAMS,
    ENSEMBLE_WEIGHTS,
    ALERT_THRESHOLD_PERCENTILE,
)
from .utils import normalize_isolation_forest_scores


class FractionmentDetectionEnsemble:
    """
    Ensemble model for transaction splitting (fractionment) detection.
    Combines Isolation Forest and DBSCAN.
    """

    def __init__(
        self,
        if_params: Optional[Dict[str, Any]] = None,
        dbscan_params: Optional[Dict[str, Any]] = None,
        ensemble_weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize the ensemble model.

        Args:
            if_params: Parameters for Isolation Forest
            dbscan_params: Parameters for DBSCAN
            ensemble_weights: Weights for combining model scores
        """
        self.if_params = if_params or DEFAULT_IF_PARAMS.copy()
        self.dbscan_params = dbscan_params or DEFAULT_DBSCAN_PARAMS.copy()
        self.ensemble_weights = ensemble_weights or ENSEMBLE_WEIGHTS.copy()

        # Models
        self.isolation_forest: Optional[IsolationForest] = None
        self.dbscan: Optional[DBSCAN] = None
        self.scaler: Optional[StandardScaler] = None

        # Training history
        self.is_fitted = False
        self.training_metrics: Dict[str, float] = {}

    def fit(self, X: np.ndarray, scaler: StandardScaler) -> Dict[str, float]:
        """
        Fit the ensemble model.

        Args:
            X: Scaled feature matrix
            scaler: Fitted StandardScaler

        Returns:
            Dictionary with training metrics
        """
        print("Training ensemble model...")

        self.scaler = scaler

        # 1. Train Isolation Forest
        print("Training Isolation Forest...")
        self.isolation_forest = IsolationForest(**self.if_params)
        self.isolation_forest.fit(X)

        # 2. Train DBSCAN
        print("Training DBSCAN...")
        self.dbscan = DBSCAN(**self.dbscan_params)
        dbscan_labels = self.dbscan.fit_predict(X)

        # 3. Calculate training metrics
        metrics = self._calculate_training_metrics(X, dbscan_labels)
        self.training_metrics = metrics

        self.is_fitted = True
        print("Ensemble model training completed")

        return metrics

    def predict_scores(
        self, X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate risk scores for input data.

        Args:
            X: Scaled feature matrix

        Returns:
            Tuple of (isolation_forest_scores, dbscan_noise_flags, combined_scores)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")

        # Isolation Forest scores
        if_scores = self.isolation_forest.decision_function(X)
        if_risk_scores = normalize_isolation_forest_scores(if_scores)

        # DBSCAN predictions
        dbscan_labels = self.dbscan.fit_predict(X)  # DBSCAN needs fit_predict each time
        dbscan_noise = (dbscan_labels == -1).astype(int)

        # Combine scores
        combined_scores = (
            self.ensemble_weights["isolation_forest"] * if_risk_scores
            + self.ensemble_weights["dbscan"] * dbscan_noise
        )

        return if_risk_scores, dbscan_noise, combined_scores

    def predict_alerts(
        self, X: np.ndarray, threshold_percentile: float = ALERT_THRESHOLD_PERCENTILE
    ) -> Tuple[np.ndarray, float]:
        """
        Generate binary alerts based on risk scores.

        Args:
            X: Scaled feature matrix
            threshold_percentile: Percentile for alert threshold

        Returns:
            Tuple of (alert_flags, threshold_value)
        """
        _, _, combined_scores = self.predict_scores(X)
        threshold = np.percentile(combined_scores, threshold_percentile)
        alerts = (combined_scores >= threshold).astype(int)

        return alerts, threshold

    def _calculate_training_metrics(
        self, X: np.ndarray, dbscan_labels: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate training metrics.

        Args:
            X: Feature matrix
            dbscan_labels: DBSCAN cluster labels

        Returns:
            Dictionary with metrics
        """
        metrics = {}

        # DBSCAN metrics
        unique_labels = np.unique(dbscan_labels)
        non_noise_labels = unique_labels[unique_labels != -1]

        metrics["num_clusters"] = len(non_noise_labels)
        metrics["num_noise_points"] = np.sum(dbscan_labels == -1)
        metrics["noise_ratio"] = metrics["num_noise_points"] / len(dbscan_labels)

        # Silhouette score for DBSCAN (if applicable)
        if len(non_noise_labels) >= 2:
            non_noise_indices = dbscan_labels != -1
            if np.sum(non_noise_indices) > 1:
                try:
                    silhouette_avg = silhouette_score(
                        X[non_noise_indices], dbscan_labels[non_noise_indices]
                    )
                    metrics["silhouette_score"] = silhouette_avg
                except Exception:
                    metrics["silhouette_score"] = -1.0
            else:
                metrics["silhouette_score"] = -1.0
        else:
            metrics["silhouette_score"] = -1.0

        return metrics

    def save_models(self, if_path: str, dbscan_path: str, scaler_path: str) -> None:
        """
        Save trained models to disk.

        Args:
            if_path: Path to save Isolation Forest model
            dbscan_path: Path to save DBSCAN model
            scaler_path: Path to save scaler
        """
        if not self.is_fitted:
            raise ValueError("Models not fitted. Call fit() first.")

        with open(if_path, "wb") as f:
            pickle.dump(self.isolation_forest, f)

        with open(dbscan_path, "wb") as f:
            pickle.dump(self.dbscan, f)

        with open(scaler_path, "wb") as f:
            pickle.dump(self.scaler, f)

        print(f"Models saved to {if_path}, {dbscan_path}, {scaler_path}")

    @classmethod
    def load_models(
        cls, if_path: str, dbscan_path: str, scaler_path: str
    ) -> "FractionmentDetectionEnsemble":
        """
        Load trained models from disk.

        Args:
            if_path: Path to Isolation Forest model
            dbscan_path: Path to DBSCAN model
            scaler_path: Path to scaler

        Returns:
            Loaded ensemble model
        """
        ensemble = cls()

        with open(if_path, "rb") as f:
            ensemble.isolation_forest = pickle.load(f)

        with open(dbscan_path, "rb") as f:
            ensemble.dbscan = pickle.load(f)

        with open(scaler_path, "rb") as f:
            ensemble.scaler = pickle.load(f)

        ensemble.is_fitted = True
        print(f"Models loaded from {if_path}, {dbscan_path}, {scaler_path}")

        return ensemble


class TransactionFractionmentPredictor:
    """
    High-level predictor for transaction fractionment detection.
    """

    def __init__(self, ensemble: FractionmentDetectionEnsemble):
        """
        Initialize predictor with ensemble model.

        Args:
            ensemble: Fitted ensemble model
        """
        self.ensemble = ensemble

    def predict_fractionment(
        self, candidate_groups_df: pl.DataFrame, return_scores: bool = True
    ) -> pl.DataFrame:
        """
        Predict fractionment for candidate groups.

        Args:
            candidate_groups_df: DataFrame with candidate groups and features
            return_scores: Whether to include individual model scores

        Returns:
            DataFrame with fractionment predictions and scores
        """
        from .preprocessing import TransactionPreprocessor

        preprocessor = TransactionPreprocessor()
        preprocessor.scaler = self.ensemble.scaler

        # Prepare features
        X, _ = preprocessor.prepare_features_for_ml(candidate_groups_df)
        X_scaled = preprocessor.transform_features(X)

        # Get predictions
        if_scores, dbscan_noise, combined_scores = self.ensemble.predict_scores(
            X_scaled
        )
        alerts, threshold = self.ensemble.predict_alerts(X_scaled)

        # Add results to DataFrame
        result_columns = [
            pl.Series("combined_risk_score", combined_scores),
            pl.Series("is_fractionment_alert", alerts.astype(bool)),
            pl.Series("alert_threshold", [threshold] * len(alerts)),
        ]

        if return_scores:
            result_columns.extend(
                [
                    pl.Series("isolation_forest_score", if_scores),
                    pl.Series("dbscan_noise_flag", dbscan_noise.astype(bool)),
                ]
            )

        return candidate_groups_df.with_columns(result_columns)

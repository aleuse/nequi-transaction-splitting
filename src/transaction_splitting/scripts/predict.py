#!/usr/bin/env python3
"""
Prediction script for transaction splitting detection.
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.transaction_splitting.preprocessing import TransactionPreprocessor
from src.transaction_splitting.models import (
    FractionmentDetectionEnsemble,
    TransactionFractionmentPredictor,
)
from src.transaction_splitting.config import (
    IF_MODEL_PATH,
    DBSCAN_MODEL_PATH,
    SCALER_PATH,
    PROCESSED_DATA_DIR,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict transaction splitting using trained models"
    )

    parser.add_argument(
        "--data-file",
        type=str,
        required=True,
        help="Path to input data file (parquet format)",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        help="Path to save predictions (parquet format). If not specified, prints to console.",
    )

    parser.add_argument(
        "--model-dir", type=str, help="Directory containing trained models"
    )

    parser.add_argument(
        "--if-model-path", type=str, help="Path to Isolation Forest model"
    )

    parser.add_argument("--dbscan-model-path", type=str, help="Path to DBSCAN model")

    parser.add_argument("--scaler-path", type=str, help="Path to scaler model")

    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=95,
        help="Percentile threshold for alerts (default: 95)",
    )

    parser.add_argument(
        "--include-scores",
        action="store_true",
        help="Include individual model scores in output",
    )

    parser.add_argument(
        "--alerts-only", action="store_true", help="Return only records with alerts"
    )

    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


def get_model_paths(args):
    """Get model file paths."""
    if args.model_dir:
        model_dir = Path(args.model_dir)
        return {
            "if_path": model_dir / "if_model.pkl",
            "dbscan_path": model_dir / "dbscan_model.pkl",
            "scaler_path": model_dir / "scaler.pkl",
        }
    else:
        return {
            "if_path": args.if_model_path or IF_MODEL_PATH,
            "dbscan_path": args.dbscan_model_path or DBSCAN_MODEL_PATH,
            "scaler_path": args.scaler_path or SCALER_PATH,
        }


def validate_model_files(model_paths):
    """Validate that model files exist."""
    for name, path in model_paths.items():
        if not Path(path).exists():
            raise FileNotFoundError(f"Model file not found: {path}")


def main():
    """Main prediction function."""
    args = parse_args()

    if args.verbose:
        print("=" * 60)
        print("TRANSACTION SPLITTING DETECTION - PREDICTION")
        print("=" * 60)

    try:
        # Get model paths
        model_paths = get_model_paths(args)
        if args.verbose:
            print(f"Using models from:")
            for name, path in model_paths.items():
                print(f"  {name}: {path}")

        # Validate model files exist
        validate_model_files(model_paths)

        # Load trained models
        print("Loading trained models...")
        ensemble = FractionmentDetectionEnsemble.load_models(
            str(model_paths["if_path"]),
            str(model_paths["dbscan_path"]),
            str(model_paths["scaler_path"]),
        )

        # Initialize predictor
        predictor = TransactionFractionmentPredictor(ensemble)

        # Load and preprocess data
        print(f"Loading data from: {args.data_file}")
        preprocessor = TransactionPreprocessor()

        # Load and clean data
        df = preprocessor.load_and_clean_data(args.data_file)
        df = preprocessor.add_temporal_features(df)

        # Generate candidate groups
        candidate_groups_df = preprocessor.generate_candidate_groups(df)

        if candidate_groups_df.shape[0] == 0:
            print("No candidate groups found in the data.")
            return

        print(f"Generated {candidate_groups_df.shape[0]} candidate groups for analysis")

        # Make predictions
        print("Making predictions...")

        # Temporarily set the threshold percentile in ensemble
        original_threshold = ensemble.__class__.predict_alerts.__defaults__[0]

        # Make predictions with custom threshold
        preprocessor.scaler = ensemble.scaler
        X, _ = preprocessor.prepare_features_for_ml(candidate_groups_df)
        X_scaled = preprocessor.transform_features(X)

        # Get predictions
        if_scores, dbscan_noise, combined_scores = ensemble.predict_scores(X_scaled)
        alerts, threshold = ensemble.predict_alerts(X_scaled, args.threshold_percentile)

        # Add results to DataFrame
        import polars as pl

        result_columns = [
            pl.Series("combined_risk_score", combined_scores),
            pl.Series("is_fractionment_alert", alerts.astype(bool)),
            pl.Series("alert_threshold", [threshold] * len(alerts)),
        ]

        if args.include_scores:
            result_columns.extend(
                [
                    pl.Series("isolation_forest_score", if_scores),
                    pl.Series("dbscan_noise_flag", dbscan_noise.astype(bool)),
                ]
            )

        predictions_df = candidate_groups_df.with_columns(result_columns)

        # Filter alerts only if requested
        if args.alerts_only:
            predictions_df = predictions_df.filter(pl.col("is_fractionment_alert"))
            print(f"Found {predictions_df.shape[0]} alerts")

        # Output results
        if args.output_file:
            print(f"Saving predictions to: {args.output_file}")
            predictions_df.write_parquet(args.output_file)
            print("Predictions saved successfully!")
        else:
            print("\nPredictions:")
            print(predictions_df)

        # Print summary statistics
        num_alerts = predictions_df.filter(pl.col("is_fractionment_alert")).shape[0]
        total_groups = candidate_groups_df.shape[0]
        alert_rate = (num_alerts / total_groups) * 100 if total_groups > 0 else 0

        print(f"\nSummary:")
        print(f"  Total candidate groups: {total_groups}")
        print(f"  Alerts generated: {num_alerts}")
        print(f"  Alert rate: {alert_rate:.2f}%")
        print(f"  Alert threshold: {threshold:.4f}")

        if args.verbose and num_alerts > 0:
            avg_risk_score = predictions_df.filter(pl.col("is_fractionment_alert"))[
                "combined_risk_score"
            ].mean()
            print(f"  Average risk score of alerts: {avg_risk_score:.4f}")

    except Exception as e:
        print(f"Error during prediction: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

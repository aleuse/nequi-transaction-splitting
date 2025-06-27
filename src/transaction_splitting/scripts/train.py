#!/usr/bin/env python3
"""
Training script for transaction splitting detection models.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.transaction_splitting.preprocessing import TransactionPreprocessor
from src.transaction_splitting.models import FractionmentDetectionEnsemble
from src.transaction_splitting.config import (
    RAW_DATA_DIR,
    IF_MODEL_PATH,
    DBSCAN_MODEL_PATH,
    SCALER_PATH,
    DEFAULT_IF_PARAMS,
    DEFAULT_DBSCAN_PARAMS,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train transaction splitting detection models"
    )

    parser.add_argument(
        "--data-file", type=str, help="Path to training data file (parquet format)"
    )

    parser.add_argument(
        "--output-dir", type=str, help="Directory to save trained models"
    )

    # Isolation Forest parameters
    parser.add_argument(
        "--if-contamination",
        type=float,
        default=DEFAULT_IF_PARAMS["contamination"],
        help="Isolation Forest contamination parameter",
    )

    parser.add_argument(
        "--if-n-estimators",
        type=int,
        default=DEFAULT_IF_PARAMS["n_estimators"],
        help="Isolation Forest number of estimators",
    )

    parser.add_argument(
        "--if-random-state",
        type=int,
        default=DEFAULT_IF_PARAMS["random_state"],
        help="Isolation Forest random state",
    )

    # DBSCAN parameters
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=DEFAULT_DBSCAN_PARAMS["eps"],
        help="DBSCAN epsilon parameter",
    )

    parser.add_argument(
        "--dbscan-min-samples",
        type=int,
        default=DEFAULT_DBSCAN_PARAMS["min_samples"],
        help="DBSCAN minimum samples parameter",
    )

    # Ensemble parameters
    parser.add_argument(
        "--ensemble-if-weight",
        type=float,
        default=0.7,
        help="Weight for Isolation Forest in ensemble",
    )

    parser.add_argument(
        "--ensemble-dbscan-weight",
        type=float,
        default=0.3,
        help="Weight for DBSCAN in ensemble",
    )

    parser.add_argument(
        "--save-metrics", action="store_true", help="Save training metrics to JSON file"
    )

    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    return parser.parse_args()


def get_data_file(args):
    """Get the data file path."""
    if args.data_file:
        return args.data_file

    # Try to find file in RAW_DATA_DIR
    if RAW_DATA_DIR.exists():
        parquet_files = list(RAW_DATA_DIR.glob("*.parquet"))
        if parquet_files:
            return str(parquet_files[0])

    raise ValueError(
        "No data file specified and no parquet file found in raw data directory. "
        "Use --data-file to specify a file."
    )


def setup_output_dir(args):
    """Setup output directory for models."""
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return {
            "if_path": output_dir / "if_model.pkl",
            "dbscan_path": output_dir / "dbscan_model.pkl",
            "scaler_path": output_dir / "scaler.pkl",
        }
    else:
        return {
            "if_path": IF_MODEL_PATH,
            "dbscan_path": DBSCAN_MODEL_PATH,
            "scaler_path": SCALER_PATH,
        }


def main():
    """Main training function."""
    args = parse_args()

    if args.verbose:
        print("=" * 60)
        print("TRANSACTION SPLITTING DETECTION - TRAINING")
        print("=" * 60)

    try:
        # Get data file
        data_file = get_data_file(args)
        print(f"Using data file: {data_file}")

        # Setup output paths
        model_paths = setup_output_dir(args)
        print(f"Models will be saved to: {model_paths}")

        # Prepare model parameters
        if_params = {
            "contamination": args.if_contamination,
            "n_estimators": args.if_n_estimators,
            "random_state": args.if_random_state,
        }

        dbscan_params = {"eps": args.dbscan_eps, "min_samples": args.dbscan_min_samples}

        ensemble_weights = {
            "isolation_forest": args.ensemble_if_weight,
            "dbscan": args.ensemble_dbscan_weight,
        }

        if args.verbose:
            print(f"Isolation Forest params: {if_params}")
            print(f"DBSCAN params: {dbscan_params}")
            print(f"Ensemble weights: {ensemble_weights}")

        # Initialize preprocessor and ensemble
        preprocessor = TransactionPreprocessor()
        ensemble = FractionmentDetectionEnsemble(
            if_params=if_params,
            dbscan_params=dbscan_params,
            ensemble_weights=ensemble_weights,
        )

        # Run preprocessing pipeline
        print("\nStarting preprocessing pipeline...")
        candidate_groups_df, X, X_scaled = preprocessor.full_preprocessing_pipeline(
            data_file
        )

        # Train ensemble model
        print("\nStarting model training...")
        training_metrics = ensemble.fit(X_scaled, preprocessor.scaler)

        # Save models
        print("\nSaving models...")
        ensemble.save_models(
            str(model_paths["if_path"]),
            str(model_paths["dbscan_path"]),
            str(model_paths["scaler_path"]),
        )

        # Print training results
        print("\nTraining completed successfully!")
        print(f"Training metrics:")
        for metric, value in training_metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Save metrics if requested
        if args.save_metrics:
            metrics_path = model_paths["if_path"].parent / "training_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(training_metrics, f, indent=2)
            print(f"Metrics saved to: {metrics_path}")

        print(f"\nModels saved successfully!")
        print(f"You can now use these models for inference with:")
        print(f"  python -m src.transaction_splitting.scripts.predict --help")

    except Exception as e:
        print(f"Error during training: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

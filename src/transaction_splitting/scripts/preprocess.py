#!/usr/bin/env python3
"""
Preprocessing script for transaction splitting detection data.
"""
import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.transaction_splitting.preprocessing import TransactionPreprocessor
from src.transaction_splitting.config import RAW_DATA_DIR, PROCESSED_DATA_DIR


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Preprocess transaction data for splitting detection"
    )
    
    parser.add_argument(
        "--input-file", 
        type=str, 
        help="Path to input data file (parquet format)"
    )
    
    parser.add_argument(
        "--output-file", 
        type=str,
        help="Path to save processed data (parquet format)"
    )
    
    parser.add_argument(
        "--output-groups", 
        type=str,
        help="Path to save candidate groups (parquet format)"
    )
    
    parser.add_argument(
        "--output-features", 
        type=str,
        help="Path to save feature matrix (numpy format)"
    )
    
    parser.add_argument(
        "--save-scaler", 
        type=str,
        help="Path to save fitted scaler"
    )
    
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def get_input_file(args):
    """Get the input file path."""
    if args.input_file:
        return args.input_file
    
    # Try to find file in RAW_DATA_DIR
    if RAW_DATA_DIR.exists():
        parquet_files = list(RAW_DATA_DIR.glob("*.parquet"))
        if parquet_files:
            return str(parquet_files[0])
    
    raise ValueError(
        "No input file specified and no parquet file found in raw data directory. "
        "Use --input-file to specify a file."
    )


def main():
    """Main preprocessing function."""
    args = parse_args()
    
    if args.verbose:
        print("=" * 60)
        print("TRANSACTION SPLITTING DETECTION - PREPROCESSING")
        print("=" * 60)
    
    try:
        # Get input file
        input_file = get_input_file(args)
        print(f"Using input file: {input_file}")
        
        # Initialize preprocessor
        preprocessor = TransactionPreprocessor()
        
        # Step 1: Load and clean data
        print("\nStep 1: Loading and cleaning data...")
        df = preprocessor.load_and_clean_data(input_file)
        
        # Step 2: Add temporal features
        print("Step 2: Adding temporal features...")
        df = preprocessor.add_temporal_features(df)
        
        # Save processed data if requested
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(output_path)
            print(f"Processed data saved to: {output_path}")
        
        # Step 3: Generate candidate groups
        print("Step 3: Generating candidate groups...")
        candidate_groups_df = preprocessor.generate_candidate_groups(df)
        
        # Save candidate groups if requested
        if args.output_groups:
            groups_path = Path(args.output_groups)
            groups_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_groups_df.write_parquet(groups_path)
            print(f"Candidate groups saved to: {groups_path}")
        
        # Step 4: Prepare features for ML
        print("Step 4: Preparing features for ML...")
        X, feature_df = preprocessor.prepare_features_for_ml(candidate_groups_df)
        
        # Step 5: Fit scaler
        print("Step 5: Fitting scaler...")
        scaler = preprocessor.fit_scaler(X)
        X_scaled = preprocessor.transform_features(X)
        
        # Save features if requested
        if args.output_features:
            features_path = Path(args.output_features)
            features_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save both original and scaled features
            import numpy as np
            np.save(features_path.with_suffix('.npy'), X)
            np.save(features_path.with_name(features_path.stem + '_scaled.npy'), X_scaled)
            print(f"Features saved to: {features_path} and {features_path.with_name(features_path.stem + '_scaled.npy')}")
        
        # Save scaler if requested
        if args.save_scaler:
            scaler_path = Path(args.save_scaler)
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            
            import pickle
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Scaler saved to: {scaler_path}")
        
        # Print summary
        print("\nPreprocessing completed successfully!")
        print(f"Summary:")
        print(f"  Input transactions: {df.shape[0]:,}")
        print(f"  Candidate groups: {candidate_groups_df.shape[0]:,}")
        print(f"  Features shape: {X.shape}")
        
        if args.verbose:
            print(f"\nCandidate groups preview:")
            print(candidate_groups_df.head())
            
            print(f"\nFeature statistics:")
            print(f"  Original features range: [{X.min():.4f}, {X.max():.4f}]")
            print(f"  Scaled features range: [{X_scaled.min():.4f}, {X_scaled.max():.4f}]")
        
        # If no specific outputs requested, save to default locations
        if not any([args.output_file, args.output_groups, args.output_features, args.save_scaler]):
            print("\nNo output paths specified. Saving to default locations...")
            
            # Save to processed data directory
            PROCESSED_DATA_DIR.mkdir(exist_ok=True)
            
            # Save processed data
            processed_data_path = PROCESSED_DATA_DIR / "processed_transactions.parquet"
            df.write_parquet(processed_data_path)
            print(f"Processed data saved to: {processed_data_path}")
            
            # Save candidate groups
            groups_path = PROCESSED_DATA_DIR / "candidate_groups.parquet"
            candidate_groups_df.write_parquet(groups_path)
            print(f"Candidate groups saved to: {groups_path}")
            
            # Save features
            import numpy as np
            features_path = PROCESSED_DATA_DIR / "features.npy"
            scaled_features_path = PROCESSED_DATA_DIR / "features_scaled.npy"
            np.save(features_path, X)
            np.save(scaled_features_path, X_scaled)
            print(f"Features saved to: {features_path} and {scaled_features_path}")
            
            # Save scaler
            scaler_path = PROCESSED_DATA_DIR / "scaler.pkl"
            import pickle
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            print(f"Scaler saved to: {scaler_path}")
        
    except Exception as e:
        print(f"Error during preprocessing: {str(e)}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main() 
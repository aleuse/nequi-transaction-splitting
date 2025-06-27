"""
Data preprocessing and feature engineering for transaction splitting detection.
"""
import polars as pl
import numpy as np
from typing import Optional, List
from sklearn.preprocessing import StandardScaler

from .config import (
    WINDOW_DURATION, 
    MIN_TRANSACTIONS_IN_GROUP, 
    NUMERICAL_FEATURES
)
from .utils import (
    calculate_amount_entropy,
    calculate_avg_time_delta_minutes,
    has_round_amounts,
    safe_datetime_conversion,
    validate_required_columns
)


class TransactionPreprocessor:
    """
    Preprocessor for transaction data.
    """
    
    def __init__(self):
        self.scaler: Optional[StandardScaler] = None
        self.required_columns = [
            "user_id", "merchant_id", "transaction_date", 
            "transaction_amount", "transaction_type"
        ]
    
    def load_and_clean_data(self, file_path: str) -> pl.DataFrame:
        """
        Load and perform basic cleaning of transaction data.
        
        Args:
            file_path: Path to the parquet file
            
        Returns:
            Cleaned DataFrame
        """
        print(f"Loading data from {file_path}...")
        
        # Load data
        df = pl.read_parquet(file_path)
        
        # Validate required columns
        validate_required_columns(df, self.required_columns)
        
        # Basic preprocessing
        df = df.with_columns([
            pl.col("transaction_amount").cast(pl.Float64),
        ])
        
        # Safe datetime conversion
        df = safe_datetime_conversion(df, "transaction_date")
        
        # Remove duplicates
        df = df.unique()
        
        print(f"Data loaded. Dimensions: {df.shape}")
        return df
    
    def add_temporal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Add temporal features to the DataFrame.
        
        Args:
            df: DataFrame with transaction_date column
            
        Returns:
            DataFrame with additional temporal features
        """
        return df.with_columns([
            pl.col("transaction_date").dt.hour().alias("hour"),
            pl.col("transaction_date").dt.weekday().alias("day_of_week"),
            pl.col("transaction_date").dt.month().alias("month"),
            pl.col("transaction_date").dt.date().alias("transaction_day")
        ])
    
    def generate_candidate_groups(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Generate candidate groups using dynamic grouping and feature engineering.
        
        Args:
            df: Preprocessed transaction DataFrame
            
        Returns:
            DataFrame with candidate groups and features
        """
        print("Generating candidate groups and features...")
        
        candidate_groups = df.sort("transaction_date").group_by_dynamic(
            index_column="transaction_date",
            every=WINDOW_DURATION,
            group_by=["user_id", "merchant_id", "transaction_type"]
        ).agg([
            pl.len().alias("transaction_count"),
            pl.col("transaction_amount").sum().alias("total_group_amount"),
            pl.col("transaction_amount").std().alias("amount_stdev"),
            pl.col("transaction_amount").median().alias("amount_median"),
            pl.col("transaction_amount").min().alias("amount_min"),
            pl.col("transaction_amount").max().alias("amount_max"),
            (pl.col("transaction_date").max() - pl.col("transaction_date").min()).dt.total_minutes().alias("time_span_minutes"),
            pl.col("transaction_amount").implode().alias("all_amounts_in_group"),
            pl.col("transaction_date").implode().alias("all_dates_in_group")
        ]).filter(
            pl.col("transaction_count") >= MIN_TRANSACTIONS_IN_GROUP
        ).with_columns([
            # Coefficient of variation
            (pl.col("amount_stdev") / pl.col("total_group_amount") * pl.col("transaction_count")).alias("amount_coeff_of_variation").fill_nan(0.0),
            # Amount entropy
            pl.col("all_amounts_in_group").map_elements(
                calculate_amount_entropy, 
                return_dtype=pl.Float64
            ).alias("amount_entropy"),
            # Average time delta
            pl.col("all_dates_in_group").map_elements(
                calculate_avg_time_delta_minutes,
                return_dtype=pl.Float64
            ).alias("avg_time_delta_minutes"),
            # Round amounts flag
            pl.col("all_amounts_in_group").map_elements(
                has_round_amounts,
                return_dtype=pl.Boolean
            ).alias("has_round_amounts")
        ]).drop(["all_amounts_in_group", "all_dates_in_group"])
        
        print(f"Generated {candidate_groups.shape[0]} candidate groups")
        return candidate_groups
    
    def prepare_features_for_ml(self, df: pl.DataFrame) -> tuple[np.ndarray, pl.DataFrame]:
        """
        Prepare numerical features for machine learning models.
        
        Args:
            df: DataFrame with candidate groups
            
        Returns:
            Tuple of (feature matrix, feature DataFrame)
        """
        print("Preparing features for ML models...")
        
        # Check which numerical features exist
        existing_features = [col for col in NUMERICAL_FEATURES if col in df.columns]
        if len(existing_features) != len(NUMERICAL_FEATURES):
            missing = [col for col in NUMERICAL_FEATURES if col not in existing_features]
            print(f"Warning: Missing features {missing}. Using: {existing_features}")
        
        # Select numerical features
        feature_df = df.select(existing_features)
        feature_matrix = feature_df.to_numpy()
        
        print(f"Feature matrix shape: {feature_matrix.shape}")
        return feature_matrix, feature_df
    
    def fit_scaler(self, X: np.ndarray) -> StandardScaler:
        """
        Fit StandardScaler on the feature matrix.
        
        Args:
            X: Feature matrix
            
        Returns:
            Fitted scaler
        """
        self.scaler = StandardScaler()
        self.scaler.fit(X)
        print("Scaler fitted")
        return self.scaler
    
    def transform_features(self, X: np.ndarray) -> np.ndarray:
        """
        Transform features using fitted scaler.
        
        Args:
            X: Feature matrix
            
        Returns:
            Scaled feature matrix
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted. Call fit_scaler first.")
        
        return self.scaler.transform(X)
    
    def full_preprocessing_pipeline(self, file_path: str) -> tuple[pl.DataFrame, np.ndarray, np.ndarray]:
        """
        Complete preprocessing pipeline from file to ML-ready features.
        
        Args:
            file_path: Path to data file
            
        Returns:
            Tuple of (candidate_groups_df, original_features, scaled_features)
        """
        # Load and clean data
        df = self.load_and_clean_data(file_path)
        
        # Add temporal features
        df = self.add_temporal_features(df)
        
        # Generate candidate groups
        candidate_groups = self.generate_candidate_groups(df)
        
        # Prepare features for ML
        X, feature_df = self.prepare_features_for_ml(candidate_groups)
        
        # Fit and transform features
        self.fit_scaler(X)
        X_scaled = self.transform_features(X)
        
        return candidate_groups, X, X_scaled 
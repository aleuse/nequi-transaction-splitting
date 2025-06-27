"""
Service layer for API business logic.
"""
import io
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Tuple
import polars as pl
import pandas as pd
from datetime import datetime

from ..preprocessing import TransactionPreprocessor
from ..models import FractionmentDetectionEnsemble, TransactionFractionmentPredictor
from ..config import IF_MODEL_PATH, DBSCAN_MODEL_PATH, SCALER_PATH
from .models import (
    TransactionRecord, 
    TransactionBatch, 
    PredictionResult,
    PredictionConfig
)


class ModelService:
    """Service for managing ML models."""
    
    def __init__(self, model_dir: str = None):
        """Initialize model service."""
        self.model_dir = Path(model_dir) if model_dir else None
        self.ensemble = None
        self.predictor = None
        self._model_loaded = False
        
    def load_models(self) -> None:
        """Load trained models."""
        try:
            if self.model_dir:
                if_path = self.model_dir / "if_model.pkl"
                dbscan_path = self.model_dir / "dbscan_model.pkl"
                scaler_path = self.model_dir / "scaler.pkl"
            else:
                if_path = IF_MODEL_PATH
                dbscan_path = DBSCAN_MODEL_PATH
                scaler_path = SCALER_PATH
            
            # Check if all model files exist
            for path in [if_path, dbscan_path, scaler_path]:
                if not Path(path).exists():
                    raise FileNotFoundError(f"Model file not found: {path}")
            
            # Load ensemble
            self.ensemble = FractionmentDetectionEnsemble.load_models(
                str(if_path), str(dbscan_path), str(scaler_path)
            )
            
            # Initialize predictor
            self.predictor = TransactionFractionmentPredictor(self.ensemble)
            self._model_loaded = True
            
        except Exception as e:
            raise Exception(f"Failed to load models: {str(e)}")
    
    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._model_loaded
    
    def ensure_models_loaded(self) -> None:
        """Ensure models are loaded, load if not."""
        if not self.is_loaded:
            self.load_models()


class FileProcessingService:
    """Service for processing different file formats."""
    
    @staticmethod
    def read_file(file_content: bytes, filename: str) -> pl.DataFrame:
        """
        Read file content and return Polars DataFrame.
        
        Args:
            file_content: File content as bytes
            filename: Original filename for format detection
            
        Returns:
            Polars DataFrame with transaction data
        """
        file_extension = Path(filename).suffix.lower()
        
        try:
            if file_extension == '.parquet':
                return pl.read_parquet(io.BytesIO(file_content))
            
            elif file_extension == '.csv':
                return pl.read_csv(io.BytesIO(file_content))
            
            elif file_extension in ['.xlsx', '.xls']:
                # Read with pandas first, then convert to polars
                df_pandas = pd.read_excel(io.BytesIO(file_content))
                return pl.from_pandas(df_pandas)
            
            else:
                raise ValueError(f"Unsupported file format: {file_extension}")
                
        except Exception as e:
            raise ValueError(f"Failed to read file {filename}: {str(e)}")
    
    @staticmethod
    def validate_transaction_columns(df: pl.DataFrame) -> None:
        """
        Validate that DataFrame has required columns.
        
        Args:
            df: DataFrame to validate
            
        Raises:
            ValueError: If required columns are missing
        """
        required_columns = [
            "user_id", "merchant_id", "transaction_date", 
            "transaction_amount", "transaction_type"
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")


class TransactionConversionService:
    """Service for converting API models to DataFrame."""
    
    @staticmethod
    def records_to_dataframe(records: List[TransactionRecord]) -> pl.DataFrame:
        """
        Convert list of TransactionRecord to Polars DataFrame.
        
        Args:
            records: List of transaction records
            
        Returns:
            Polars DataFrame
        """
        data = []
        for record in records:
            # Convert datetime to string if needed
            transaction_date = record.transaction_date
            if isinstance(transaction_date, datetime):
                transaction_date = transaction_date.strftime('%Y-%m-%d %H:%M:%S')
            
            data.append({
                "user_id": record.user_id,
                "merchant_id": record.merchant_id,
                "transaction_date": transaction_date,
                "transaction_amount": record.transaction_amount,
                "transaction_type": record.transaction_type
            })
        
        return pl.DataFrame(data)
    
    @staticmethod
    def batch_to_dataframe(batch: TransactionBatch) -> pl.DataFrame:
        """
        Convert TransactionBatch to Polars DataFrame.
        
        Args:
            batch: Transaction batch
            
        Returns:
            Polars DataFrame
        """
        return TransactionConversionService.records_to_dataframe(batch.transactions)


class PredictionService:
    """Service for making predictions."""
    
    def __init__(self, model_service: ModelService):
        """Initialize prediction service."""
        self.model_service = model_service
        self.preprocessor = APITransactionPreprocessor()
    
    def predict_from_dataframe(
        self, 
        df: pl.DataFrame, 
        config: PredictionConfig
    ) -> Tuple[List[PredictionResult], Dict[str, Any]]:
        """
        Make predictions from DataFrame.
        
        Args:
            df: Input DataFrame
            config: Prediction configuration
            
        Returns:
            Tuple of (predictions, metadata)
        """
        # Ensure models are loaded
        self.model_service.ensure_models_loaded()
        
        # Validate columns
        FileProcessingService.validate_transaction_columns(df)
        
        # Preprocess data
        df = self.preprocessor.load_and_clean_data_from_df(df)
        df = self.preprocessor.add_temporal_features(df)
        
        # Generate candidate groups
        candidate_groups_df = self.preprocessor.generate_candidate_groups(df)
        
        if candidate_groups_df.shape[0] == 0:
            return [], {
                "total_candidate_groups": 0,
                "alerts_count": 0,
                "alert_rate": 0.0
            }
        
        # Make predictions
        self.preprocessor.scaler = self.model_service.ensemble.scaler
        X, _ = self.preprocessor.prepare_features_for_ml(candidate_groups_df)
        X_scaled = self.preprocessor.transform_features(X)
        
        # Get predictions
        if_scores, dbscan_noise, combined_scores = self.model_service.ensemble.predict_scores(X_scaled)
        alerts, threshold = self.model_service.ensemble.predict_alerts(X_scaled, config.threshold_percentile)
        
        # Convert to response format
        predictions = []
        for i in range(len(candidate_groups_df)):
            row = candidate_groups_df.row(i, named=True)
            
            prediction = PredictionResult(
                user_id=row["user_id"],
                merchant_id=row["merchant_id"],
                transaction_type=row["transaction_type"],
                transaction_count=row["transaction_count"],
                total_group_amount=row["total_group_amount"],
                combined_risk_score=float(combined_scores[i]),
                is_fractionment_alert=bool(alerts[i]),
                alert_threshold=float(threshold)
            )
            
            # Add detailed scores if requested
            if config.include_scores:
                prediction.isolation_forest_score = float(if_scores[i])
                prediction.dbscan_noise_flag = bool(dbscan_noise[i])
            
            # Filter alerts only if requested
            if config.alerts_only and not prediction.is_fractionment_alert:
                continue
                
            predictions.append(prediction)
        
        # Calculate metadata
        alerts_count = sum(alerts)
        total_groups = len(candidate_groups_df)
        alert_rate = (alerts_count / total_groups * 100) if total_groups > 0 else 0.0
        
        metadata = {
            "total_candidate_groups": total_groups,
            "alerts_count": alerts_count,
            "alert_rate": alert_rate
        }
        
        return predictions, metadata
    
    def predict_from_file(
        self, 
        file_content: bytes, 
        filename: str,
        config: PredictionConfig
    ) -> Tuple[List[PredictionResult], Dict[str, Any]]:
        """
        Make predictions from uploaded file.
        
        Args:
            file_content: File content
            filename: Original filename
            config: Prediction configuration
            
        Returns:
            Tuple of (predictions, metadata)
        """
        # Read file
        df = FileProcessingService.read_file(file_content, filename)
        
        # Make predictions
        return self.predict_from_dataframe(df, config)
    
    def predict_from_records(
        self, 
        records: List[TransactionRecord],
        config: PredictionConfig
    ) -> Tuple[List[PredictionResult], Dict[str, Any]]:
        """
        Make predictions from transaction records.
        
        Args:
            records: List of transaction records
            config: Prediction configuration
            
        Returns:
            Tuple of (predictions, metadata)
        """
        # Convert to DataFrame
        df = TransactionConversionService.records_to_dataframe(records)
        
        # Add total transactions to metadata
        predictions, metadata = self.predict_from_dataframe(df, config)
        metadata["total_transactions_processed"] = len(records)
        
        return predictions, metadata


# Extension to TransactionPreprocessor for API usage  
class APITransactionPreprocessor(TransactionPreprocessor):
    """Extended preprocessor for API usage."""
    
    def load_and_clean_data_from_df(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Load and clean data from existing DataFrame.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Cleaned DataFrame
        """
        from ..utils import safe_datetime_conversion, validate_required_columns
        
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
        
        return df 
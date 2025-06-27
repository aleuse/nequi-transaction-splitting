"""
Pydantic models for API requests and responses.
"""
from datetime import datetime
from typing import List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum


class TransactionType(str, Enum):
    """Transaction type enumeration."""
    CREDIT = "credit"
    DEBIT = "debit"


class TransactionRecord(BaseModel):
    """Single transaction record."""
    user_id: int = Field(..., description="User ID")
    merchant_id: int = Field(..., description="Merchant ID") 
    transaction_date: Union[datetime, str] = Field(..., description="Transaction date and time")
    transaction_amount: float = Field(..., gt=0, description="Transaction amount (must be positive)")
    transaction_type: TransactionType = Field(..., description="Transaction type")
    
    @validator('transaction_date', pre=True)
    def parse_transaction_date(cls, v):
        """Parse transaction date from string if needed."""
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v.replace('Z', '+00:00'))
            except ValueError:
                try:
                    return datetime.strptime(v, '%Y-%m-%d %H:%M:%S')
                except ValueError:
                    raise ValueError(f"Invalid date format: {v}. Use ISO format or 'YYYY-MM-DD HH:MM:SS'")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": 123456,
                "merchant_id": 789012,
                "transaction_date": "2023-06-15 14:30:00",
                "transaction_amount": 150000.0,
                "transaction_type": "debit"
            }
        }


class TransactionBatch(BaseModel):
    """Batch of transaction records."""
    transactions: List[TransactionRecord] = Field(..., min_items=1, description="List of transactions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transactions": [
                    {
                        "user_id": 123456,
                        "merchant_id": 789012,
                        "transaction_date": "2023-06-15 14:30:00",
                        "transaction_amount": 150000.0,
                        "transaction_type": "debit"
                    },
                    {
                        "user_id": 123456,
                        "merchant_id": 789012,
                        "transaction_date": "2023-06-15 14:35:00",
                        "transaction_amount": 75000.0,
                        "transaction_type": "debit"
                    }
                ]
            }
        }


class CandidateGroup(BaseModel):
    """Candidate group with features."""
    user_id: int
    merchant_id: int
    transaction_type: str
    transaction_count: int
    total_group_amount: float
    amount_stdev: float
    amount_median: float
    amount_min: float
    amount_max: float
    time_span_minutes: float
    amount_coeff_of_variation: float
    amount_entropy: float
    avg_time_delta_minutes: float
    has_round_amounts: bool


class PredictionResult(BaseModel):
    """Prediction result for a candidate group."""
    # Original group information
    user_id: int
    merchant_id: int
    transaction_type: str
    transaction_count: int
    total_group_amount: float
    
    # Risk scores
    combined_risk_score: float = Field(..., description="Combined risk score (0-1)")
    is_fractionment_alert: bool = Field(..., description="Whether this is flagged as fractionment")
    alert_threshold: float = Field(..., description="Threshold used for alerts")
    
    # Optional detailed scores
    isolation_forest_score: Optional[float] = Field(None, description="Isolation Forest anomaly score")
    dbscan_noise_flag: Optional[bool] = Field(None, description="DBSCAN noise flag")


class FileUploadResponse(BaseModel):
    """Response for file upload predictions."""
    message: str
    total_candidate_groups: int
    alerts_count: int
    alert_rate: float
    predictions: List[PredictionResult]


class SinglePredictionResponse(BaseModel):
    """Response for single transaction prediction."""
    message: str
    candidate_groups_found: int
    predictions: List[PredictionResult]


class BatchPredictionResponse(BaseModel):
    """Response for batch transaction prediction."""
    message: str
    total_transactions_processed: int
    candidate_groups_found: int
    alerts_count: int
    alert_rate: float
    predictions: List[PredictionResult]


class PredictionConfig(BaseModel):
    """Configuration for prediction parameters."""
    threshold_percentile: float = Field(default=95, ge=50, le=99.9, description="Alert threshold percentile")
    include_scores: bool = Field(default=True, description="Include individual model scores")
    alerts_only: bool = Field(default=False, description="Return only alerts")
    
    class Config:
        json_schema_extra = {
            "example": {
                "threshold_percentile": 95,
                "include_scores": True,
                "alerts_only": False
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    version: str
    timestamp: datetime


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None
    timestamp: datetime 
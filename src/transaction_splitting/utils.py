"""
Utility functions for transaction splitting detection.
"""
import numpy as np
from scipy.stats import entropy
import polars as pl
from typing import List, Union


def calculate_amount_entropy(amounts: Union[List[float], pl.Series]) -> float:
    """
    Calculate the entropy of a list or Series of amounts.
    
    Args:
        amounts: List of amounts or Polars Series
        
    Returns:
        Entropy value as float
    """
    # Convert Polars Series to list if necessary
    if hasattr(amounts, 'to_list'):
        amounts = amounts.to_list()
    
    if len(amounts) == 0:
        return 0.0
    
    # Calculate value_counts and then entropy
    unique_amounts, counts = np.unique(amounts, return_counts=True)
    probabilities = counts / counts.sum()
    return entropy(probabilities)


def calculate_coeff_of_variation(amounts: List[float]) -> float:
    """
    Calculate coefficient of variation for a list of amounts.
    
    Args:
        amounts: List of amounts
        
    Returns:
        Coefficient of variation as float
    """
    if not amounts:
        return 0.0
    
    amounts_arr = np.array(amounts)
    std_val = np.std(amounts_arr)
    mean_val = np.mean(amounts_arr)
    return std_val / mean_val if mean_val != 0 else 0.0


def calculate_avg_time_delta_minutes(dates: pl.Series) -> float:
    """
    Calculate average time delta between dates in minutes.
    
    Args:
        dates: Polars Series of dates
        
    Returns:
        Average time delta in minutes
    """
    dates_list = dates.to_list()
    if len(dates_list) <= 1:
        return 0.0
    
    timestamps = [d.timestamp() for d in dates_list]
    return np.mean(np.diff(np.sort(timestamps))) / 60


def has_round_amounts(amounts: pl.Series, multiple: int = 100000) -> bool:
    """
    Check if any amount is a multiple of the specified value.
    
    Args:
        amounts: Polars Series of amounts
        multiple: Multiple to check for (default: 100000)
        
    Returns:
        True if any amount is a multiple of the specified value
    """
    amounts_list = amounts.to_list()
    return any(a % multiple == 0 for a in amounts_list)


def safe_datetime_conversion(df: pl.DataFrame, date_column: str) -> pl.DataFrame:
    """
    Safely convert string dates to datetime with multiple format attempts.
    
    Args:
        df: Polars DataFrame
        date_column: Name of the date column
        
    Returns:
        DataFrame with converted date column
    """
    try:
        # First attempt: parse with specific format
        return df.with_columns([
            pl.col(date_column).str.to_datetime("%Y-%m-%d %H:%M:%S")
        ])
    except Exception:
        try:
            # Second attempt: direct conversion
            return df.with_columns([
                pl.col(date_column).cast(pl.Datetime)
            ])
        except Exception:
            # Last attempt: with strict=False
            return df.with_columns([
                pl.col(date_column).str.to_datetime(strict=False)
            ])


def validate_required_columns(df: pl.DataFrame, required_columns: List[str]) -> None:
    """
    Validate that all required columns exist in the DataFrame.
    
    Args:
        df: Polars DataFrame
        required_columns: List of required column names
        
    Raises:
        ValueError: If any required column is missing
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def normalize_isolation_forest_scores(scores: np.ndarray) -> np.ndarray:
    """
    Normalize Isolation Forest scores to [0, 1] range.
    Lower IF scores (more anomalous) become higher risk scores.
    
    Args:
        scores: Array of Isolation Forest decision function scores
        
    Returns:
        Normalized risk scores (0 = normal, 1 = anomalous)
    """
    min_score = scores.min()
    max_score = scores.max()
    
    if max_score == min_score:
        return np.zeros_like(scores)
    
    return 1 - (scores - min_score) / (max_score - min_score) 
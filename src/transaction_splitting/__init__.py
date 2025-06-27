"""
Transaction Splitting Detection Package

A modular system for detecting transaction splitting/fractionment using
ensemble machine learning methods.
"""

__version__ = "1.0.0"
__author__ = "Transaction Analytics Team"

# Main components
from .preprocessing import TransactionPreprocessor
from .models import FractionmentDetectionEnsemble, TransactionFractionmentPredictor
from .utils import (
    calculate_amount_entropy,
    calculate_coeff_of_variation,
    normalize_isolation_forest_scores,
)

# Configuration
from . import config

__all__ = [
    "TransactionPreprocessor",
    "FractionmentDetectionEnsemble",
    "TransactionFractionmentPredictor",
    "calculate_amount_entropy",
    "calculate_coeff_of_variation",
    "normalize_isolation_forest_scores",
    "config",
]

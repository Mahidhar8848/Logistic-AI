"""
Feature Engineering Module for AI Logistics Pipeline.
Defines leakage-safe temporal, vehicle, route, and market demand feature transformations.
"""
from typing import Dict, Any, List
import pandas as pd
import numpy as np

def extract_temporal_features(dispatch_date: pd.Timestamp) -> Dict[str, Any]:
    """Extracts month, day of week, quarter, and weekend indicator from timestamp."""
    dt = pd.to_datetime(dispatch_date)
    dow = dt.dayofweek
    return {
        "dispatch_month": dt.month,
        "dispatch_dayofweek": dow,
        "dispatch_quarter": dt.quarter,
        "is_weekend": 1 if dow in [5, 6] else 0
    }

def calculate_load_utilization(weight_lbs: float, capacity_lbs: float = 45000.0) -> float:
    """Calculates weight utilization percentage against maximum vehicle payload capacity."""
    if capacity_lbs <= 0:
        return 0.0
    return min(1.0, float(weight_lbs) / float(capacity_lbs))

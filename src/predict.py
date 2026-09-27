import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
import joblib
from typing import Dict, Any, Union

try:
    from src.config import PROJECT_ROOT
except ImportError:
    from config import PROJECT_ROOT

class EmptyReturnPredictor:
    """
    Inference engine for predicting vehicle empty-return risk
    prior to delivery completion.
    """
    def __init__(self, models_dir: Union[str, Path] = None):
        if models_dir is None:
            models_dir = PROJECT_ROOT / "models"
        else:
            models_dir = Path(models_dir)

        self.model_path = models_dir / "model.pkl"
        self.preprocessor_path = models_dir / "preprocessing.pkl"
        self.feature_list_path = models_dir / "feature_list.json"

        if not self.model_path.exists() or not self.preprocessor_path.exists():
            raise FileNotFoundError(f"Model artifacts not found in {models_dir}. Please run src/train.py first.")

        print(f"[Predictor] Loading model artifacts from {models_dir}...")
        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)

        with open(self.feature_list_path, "r") as f:
            self.feature_meta = json.load(f)

        self.raw_cat_features = self.feature_meta["raw_categorical_features"]
        self.raw_num_features = self.feature_meta["raw_numerical_features"]
        self.all_raw_features = self.raw_cat_features + self.raw_num_features

    def _prepare_input_df(self, trip_data: Dict[str, Any]) -> pd.DataFrame:
        """Converts raw trip input dictionary into structured DataFrame with defaults."""
        data = {}
        for col in self.all_raw_features:
            val = trip_data.get(col, None)
            if val is None:
                # Provide sensible defaults for missing fields
                if col in self.raw_cat_features:
                    val = "Unknown"
                elif "hist" in col or "rate" in col:
                    val = 0.35
                elif "capacity" in col or "weight" in col:
                    val = 25000.0
                elif "distance" in col:
                    val = 500.0
                elif "month" in col or "quarter" in col:
                    val = 6
                else:
                    val = 0.0
            data[col] = [val]

        return pd.DataFrame(data)

    def predict(self, trip_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Accepts trip dictionary and returns empty-return probability and risk assessment.
        """
        input_df = self._prepare_input_df(trip_data)
        
        # Transform inputs using fitted preprocessor
        X_trans = self.preprocessor.transform(input_df)
        
        # Calculate raw probability from trained ML model
        if hasattr(self.model, "predict_proba"):
            prob = float(self.model.predict_proba(X_trans)[0, 1])
        else:
            prob = float(self.model.predict(X_trans)[0])

        prob_percentage = round(prob * 100, 1)

        # Risk Classification Thresholds
        if prob_percentage >= 50.0:
            risk_level = "HIGH"
        elif prob_percentage >= 35.0:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        # Determine key risk drivers
        risk_drivers = []
        if trip_data.get("booking_type") == "Spot":
            risk_drivers.append("Spot booking type (higher backhaul uncertainty)")
        if trip_data.get("current_load_utilization", 0.6) < 0.4:
            risk_drivers.append("Low current trip load weight")
        if trip_data.get("dest_outbound_market_volume", 1000) < 2000:
            risk_drivers.append("Destination city has low outbound freight market volume")
        if trip_data.get("is_heading_home", 0) == 0:
            risk_drivers.append("Destination is far from truck home terminal")
        
        if not risk_drivers:
            risk_drivers.append("Baseline historical lane empty-return variance")

        return {
            "empty_return_probability": f"{prob_percentage}%",
            "probability_value": prob,
            "risk_level": risk_level,
            "empty_return_predicted": int(prob >= 0.50),
            "key_risk_drivers": risk_drivers,
            "origin": trip_data.get("origin_city", "Unknown"),
            "destination": trip_data.get("destination_city", "Unknown"),
            "load_type": trip_data.get("current_load_type", "Unknown")
        }

def predict_empty_return(trip_data: Dict[str, Any]) -> Dict[str, Any]:
    """Convenience wrapper for single trip prediction."""
    predictor = EmptyReturnPredictor()
    return predictor.predict(trip_data)

if __name__ == "__main__":
    # Test sample trip
    sample_trip = {
        "trip_id": "TRIP00099999",
        "origin_city": "Chicago",
        "origin_state": "IL",
        "destination_city": "Las Vegas",
        "destination_state": "NV",
        "truck_make": "Peterbilt",
        "truck_home_terminal": "Omaha",
        "trailer_type": "Dry Van",
        "booking_type": "Spot",
        "current_load_type": "Dry Van",
        "dispatch_month": 10,
        "dispatch_dayofweek": 4,
        "dispatch_quarter": 4,
        "is_weekend": 0,
        "truck_model_year": 2018.0,
        "truck_tank_capacity": 200.0,
        "trailer_length": 53.0,
        "driver_years_exp": 5.0,
        "current_weight_lbs": 14500.0,
        "current_pieces": 12.0,
        "current_revenue": 3200.0,
        "current_load_utilization": 0.32,
        "route_distance": 1750.0,
        "route_base_rate": 2.10,
        "route_fuel_surcharge_rate": 0.25,
        "route_transit_days": 3.0,
        "dest_outbound_market_volume": 1450.0,
        "is_heading_home": 0,
        "route_hist_empty_rate": 0.38,
        "truck_hist_empty_rate": 0.40
    }

    print("Running sample empty-return prediction:")
    result = predict_empty_return(sample_trip)
    print(json.dumps(result, indent=2))

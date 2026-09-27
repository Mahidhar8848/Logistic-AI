import unittest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from predict import predict_empty_return, EmptyReturnPredictor

class TestPrediction(unittest.TestCase):
    def test_predictor_initialization(self):
        predictor = EmptyReturnPredictor()
        self.assertIsNotNone(predictor.model)
        self.assertIsNotNone(predictor.preprocessor)

    def test_sample_prediction(self):
        sample = {
            "origin_city": "Chicago",
            "destination_city": "Las Vegas",
            "booking_type": "Spot",
            "current_load_type": "Dry Van",
            "current_weight_lbs": 15000.0
        }
        res = predict_empty_return(sample)
        self.assertIn("empty_return_probability", res)
        self.assertIn("risk_level", res)
        self.assertIn(res["risk_level"], ["LOW", "MEDIUM", "HIGH"])

if __name__ == "__main__":
    unittest.main()

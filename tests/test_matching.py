import unittest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))

from matching import ReturnLoadMatcher, GeoDistanceCalculator, recommend_return_load_for_trip
from impact import ImpactCalculator

class TestReturnLoadMatching(unittest.TestCase):
    def setUp(self):
        self.matcher = ReturnLoadMatcher()
        self.geo = GeoDistanceCalculator()
        self.impact_calc = ImpactCalculator()

    def test_geo_distance(self):
        """Test geographic distance calculations between known cities."""
        dist = self.geo.get_distance("Chicago", "Indianapolis")
        self.assertGreater(dist, 100.0)
        self.assertLess(dist, 300.0)
        
        same_dist = self.geo.get_distance("Chicago", "Chicago")
        self.assertEqual(same_dist, 0.0)

    def test_capacity_constraint_filter(self):
        """Test that candidate loads exceeding vehicle remaining capacity are rejected."""
        candidates = self.matcher.filter_and_rank_candidates(
            current_dest_city="Las Vegas",
            target_return_city="Chicago",
            vehicle_trailer_type="Dry Van",
            remaining_capacity_lbs=10000.0  # Low capacity limit
        )
        for cand in candidates:
            self.assertLessEqual(cand['shipment_weight_lbs'], 10000.0)

    def test_equipment_type_constraint_filter(self):
        """Test that Refrigerated loads are rejected for Dry Van trailers."""
        candidates = self.matcher.filter_and_rank_candidates(
            current_dest_city="Las Vegas",
            target_return_city="Chicago",
            vehicle_trailer_type="Dry Van"
        )
        for cand in candidates:
            self.assertNotEqual(cand['load_type'], "Refrigerated")

    def test_detour_distance_constraint_filter(self):
        """Test that candidate loads causing excessive detour are filtered out."""
        max_detour = 50.0
        candidates = self.matcher.filter_and_rank_candidates(
            current_dest_city="Las Vegas",
            target_return_city="Chicago",
            vehicle_trailer_type="Dry Van",
            max_detour_miles=max_detour
        )
        for cand in candidates:
            self.assertLessEqual(cand['detour_miles'], max_detour)

    def test_transparent_matching_score(self):
        """Test that candidate loads are strictly ranked by matching score in descending order."""
        candidates = self.matcher.filter_and_rank_candidates(
            current_dest_city="Chicago",
            target_return_city="Las Vegas",
            vehicle_trailer_type="Dry Van",
            top_n=5
        )
        if len(candidates) > 1:
            for i in range(len(candidates) - 1):
                self.assertGreaterEqual(candidates[i]['matching_score'], candidates[i+1]['matching_score'])

    def test_fleet_optimization_solver(self):
        """Test MIP / Greedy optimization solver on multiple high-risk vehicles."""
        vehicles = [
            {"truck_id": "TRK00001", "destination_city": "Las Vegas", "home_terminal": "Chicago", "trailer_type": "Dry Van"},
            {"truck_id": "TRK00002", "destination_city": "Dallas", "home_terminal": "Houston", "trailer_type": "Refrigerated"}
        ]
        opt_res = self.matcher.optimize_fleet_assignments(vehicles)
        self.assertIn("assignments", opt_res)
        self.assertIn("solver_status", opt_res)

    def test_impact_calculations(self):
        """Test sustainability and financial impact calculation logic."""
        sample_assignment = {
            "recommended_load": {
                "shipment_distance_miles": 500.0,
                "detour_miles": 50.0,
                "shipment_weight_lbs": 30000.0,
                "revenue": 1200.0
            }
        }
        imp = self.impact_calc.calculate_assignment_impact(sample_assignment)
        self.assertEqual(imp['empty_miles_avoided'], 450.0)
        self.assertGreater(imp['fuel_gallons_saved'], 0.0)
        self.assertGreater(imp['co2_reduction_kg'], 0.0)

    def test_edge_cases_unknown_location(self):
        """Test edge case handling for unknown origin or destination cities."""
        candidates = self.matcher.filter_and_rank_candidates(
            current_dest_city="UnknownCityName123",
            target_return_city="UnknownCityName456",
            vehicle_trailer_type="Dry Van"
        )
        self.assertIsInstance(candidates, list)

if __name__ == "__main__":
    unittest.main()

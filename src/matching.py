import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Optional, Tuple

try:
    from ortools.linear_solver import pywraplp
    HAS_OR_TOOLS = True
except ImportError:
    HAS_OR_TOOLS = False

try:
    from src.config import DATASET_DIR, PROJECT_ROOT
    from src.data_loader import DataLoader
    from src.predict import predict_empty_return
except ImportError:
    from config import DATASET_DIR, PROJECT_ROOT
    from data_loader import DataLoader
    from predict import predict_empty_return

# Standard Maximum Payload Capacity (53ft Trailer)
MAX_CAPACITY_LBS = 45000.0

def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculates Haversine distance in miles between two latitude/longitude pairs."""
    R = 3958.8  # Earth radius in miles
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2.0)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0)**2
    return float(R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a)))

class GeoDistanceCalculator:
    """Calculates distance between cities using facilities dataset lat/lon coordinates."""
    def __init__(self, data_dir: Optional[Path] = None):
        loader = DataLoader(data_dir or DATASET_DIR)
        facilities = loader.load_dataset("facilities.csv")
        routes = loader.load_dataset("routes.csv")
        
        self.city_coords = {}
        for _, row in facilities.iterrows():
            city = str(row['city']).strip()
            self.city_coords[city] = (float(row['latitude']), float(row['longitude']))
            
        self.route_distances = {}
        for _, row in routes.iterrows():
            key = (str(row['origin_city']).strip(), str(row['destination_city']).strip())
            self.route_distances[key] = float(row['typical_distance_miles'])

    def get_distance(self, origin: str, destination: str) -> float:
        """Returns distance in miles between origin and destination cities."""
        origin, destination = origin.strip(), destination.strip()
        if origin == destination:
            return 0.0

        if (origin, destination) in self.route_distances:
            return self.route_distances[(origin, destination)]
        if (destination, origin) in self.route_distances:
            return self.route_distances[(destination, origin)]

        if origin in self.city_coords and destination in self.city_coords:
            c1 = self.city_coords[origin]
            c2 = self.city_coords[destination]
            h_dist = haversine_miles(c1[0], c1[1], c2[0], c2[1])
            return round(h_dist * 1.22, 1)  # 1.22 factor for road network distance

        return 500.0  # Fallback default distance

class ReturnLoadMatcher:
    """
    Candidate discovery, feasibility filtering, transparent multi-attribute scoring,
    and OR-Tools MIP optimization engine for matching empty-return vehicles with return loads.
    """
    def __init__(self, data_dir: Optional[Path] = None):
        self.loader = DataLoader(data_dir or DATASET_DIR)
        self.geo = GeoDistanceCalculator(data_dir or DATASET_DIR)
        self.loads_df = self.loader.load_dataset("loads.csv")
        self.routes_df = self.loader.load_dataset("routes.csv")
        
        # Merge routes info onto loads
        self.available_loads = self.loads_df.merge(self.routes_df, on="route_id", how="left")

    def filter_and_rank_candidates(
        self,
        current_dest_city: str,
        target_return_city: str,
        vehicle_trailer_type: str,
        remaining_capacity_lbs: float = 45000.0,
        max_detour_miles: float = 200.0,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Filters and scores return shipment candidates for a single vehicle.
        Vectorized for high-performance sub-second search across 85,000+ loads.
        """
        direct_return_dist = self.geo.get_distance(current_dest_city, target_return_city)

        # 1. Fast Pre-filter: Capacity and Equipment Type
        df = self.available_loads[self.available_loads['weight_lbs'] <= remaining_capacity_lbs].copy()
        if vehicle_trailer_type != "Refrigerated":
            df = df[df['load_type'] != "Refrigerated"]

        if df.empty:
            return []

        # 2. Pre-filter by Origin City matching or nearby origin cities
        # Focus search on loads originating within 200 miles of current destination
        pickup_cities = df['origin_city'].unique()
        nearby_pickups = {c for c in pickup_cities if self.geo.get_distance(current_dest_city, c) <= max_detour_miles}
        
        df = df[df['origin_city'].isin(nearby_pickups)].copy()
        if df.empty:
            return []

        # Sample up to 1,000 top matching loads for detailed scoring
        df = df.head(1000)

        candidates = []
        for _, load in df.iterrows():
            pickup_city = str(load['origin_city'])
            drop_city = str(load['destination_city'])
            weight_lbs = float(load['weight_lbs'])
            load_type = str(load['load_type'])
            revenue = float(load['revenue'])
            load_id = str(load['load_id'])

            deadhead_pickup = self.geo.get_distance(current_dest_city, pickup_city)
            transit_dist = self.geo.get_distance(pickup_city, drop_city)
            deadhead_drop = self.geo.get_distance(drop_city, target_return_city)

            total_dist_with_load = deadhead_pickup + transit_dist + deadhead_drop
            detour_miles = max(0.0, total_dist_with_load - direct_return_dist)

            if detour_miles > max_detour_miles:
                continue

            capacity_score = min(100.0, (weight_lbs / MAX_CAPACITY_LBS) * 100.0)
            route_compat_pct = max(0.0, 100.0 * (1.0 - (detour_miles / (direct_return_dist + 1.0))))
            detour_score = max(0.0, 100.0 * (1.0 - (detour_miles / max_detour_miles)))
            rev_per_mile = revenue / max(1.0, transit_dist)
            rev_score = min(100.0, (rev_per_mile / 3.0) * 100.0)

            matching_score = round(
                0.35 * capacity_score +
                0.30 * route_compat_pct +
                0.20 * detour_score +
                0.15 * rev_score,
                2
            )

            candidates.append({
                "load_id": load_id,
                "pickup_city": pickup_city,
                "drop_city": drop_city,
                "shipment_weight_lbs": weight_lbs,
                "load_type": load_type,
                "revenue": revenue,
                "booking_type": str(load.get('booking_type', 'Spot')),
                "deadhead_pickup_miles": round(deadhead_pickup, 1),
                "shipment_distance_miles": round(transit_dist, 1),
                "deadhead_drop_miles": round(deadhead_drop, 1),
                "detour_miles": round(detour_miles, 1),
                "total_return_distance": round(total_dist_with_load, 1),
                "route_compatibility_pct": round(route_compat_pct, 1),
                "expected_utilization_pct": round((weight_lbs / MAX_CAPACITY_LBS) * 100.0, 1),
                "matching_score": matching_score
            })

        candidates.sort(key=lambda x: x['matching_score'], reverse=True)
        return candidates[:top_n]

    def optimize_fleet_assignments(
        self,
        high_risk_vehicles: List[Dict[str, Any]],
        candidate_pool_size: int = 50
    ) -> Dict[str, Any]:
        """
        Solves multi-vehicle return-load assignment optimization problem using OR-Tools MIP solver.
        Maximizes total matched payload and route score while minimizing empty detour miles.
        """
        if not HAS_OR_TOOLS:
            print("[Warning] OR-Tools not available. Using greedy ranking assignment.")
            return self._greedy_optimize(high_risk_vehicles)

        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            solver = pywraplp.Solver.CreateSolver('CBC')

        if not solver:
            print("[Warning] MIP solver creation failed. Falling back to greedy solver.")
            return self._greedy_optimize(high_risk_vehicles)

        # Collect candidate options per vehicle
        vehicle_candidates = {}
        all_candidate_loads = set()

        for veh in high_risk_vehicles:
            v_id = veh['truck_id']
            cands = self.filter_and_rank_candidates(
                current_dest_city=veh['destination_city'],
                target_return_city=veh.get('home_terminal', veh.get('origin_city', 'Chicago')),
                vehicle_trailer_type=veh.get('trailer_type', 'Dry Van'),
                remaining_capacity_lbs=veh.get('remaining_capacity_lbs', 45000.0),
                top_n=10
            )
            vehicle_candidates[v_id] = cands
            for c in cands:
                all_candidate_loads.add(c['load_id'])

        # Create Decision Variables: x[v, s] = 1 if vehicle v assigned load s
        x = {}
        for veh in high_risk_vehicles:
            v_id = veh['truck_id']
            for c in vehicle_candidates[v_id]:
                s_id = c['load_id']
                x[v_id, s_id] = solver.BoolVar(f"x_{v_id}_{s_id}")

        # Constraint 1: Single Vehicle per Load
        for s_id in all_candidate_loads:
            solver.Add(solver.Sum([x[v_id, s_id] for v_id in vehicle_candidates if (v_id, s_id) in x]) <= 1)

        # Constraint 2: Single Load per Vehicle
        for veh in high_risk_vehicles:
            v_id = veh['truck_id']
            solver.Add(solver.Sum([x[v_id, s_id] for (v, s_id) in x if v == v_id]) <= 1)

        # Objective Function: Maximize (Matching Score * Weight - 0.5 * Detour)
        objective = solver.Objective()
        for (v_id, s_id), var in x.items():
            # Find candidate detail
            cand = next(c for c in vehicle_candidates[v_id] if c['load_id'] == s_id)
            coeff = (cand['matching_score'] * 10.0) + (cand['shipment_weight_lbs'] / 1000.0) - (cand['detour_miles'] * 0.5)
            objective.SetCoefficient(var, float(coeff))
        
        objective.SetMaximization()

        status = solver.Solve()

        assignments = []
        total_empty_miles_saved = 0.0
        total_payload_added = 0.0

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            for (v_id, s_id), var in x.items():
                if var.solution_value() > 0.5:
                    veh = next(v for v in high_risk_vehicles if v['truck_id'] == v_id)
                    cand = next(c for c in vehicle_candidates[v_id] if c['load_id'] == s_id)

                    assignments.append({
                        "vehicle_id": v_id,
                        "current_trip": veh,
                        "recommended_load": cand
                    })
                    total_empty_miles_saved += cand['shipment_distance_miles']
                    total_payload_added += cand['shipment_weight_lbs']

        return {
            "solver_status": "OPTIMAL" if status == pywraplp.Solver.OPTIMAL else "FEASIBLE",
            "matched_assignments_count": len(assignments),
            "assignments": assignments,
            "total_empty_miles_saved": round(total_empty_miles_saved, 1),
            "total_payload_added_lbs": round(total_payload_added, 1)
        }

    def _greedy_optimize(self, high_risk_vehicles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Greedy fallback matching when MIP solver is unavailable."""
        used_loads = set()
        assignments = []
        total_empty_miles_saved = 0.0
        total_payload_added = 0.0

        for veh in high_risk_vehicles:
            v_id = veh['truck_id']
            cands = self.filter_and_rank_candidates(
                current_dest_city=veh['destination_city'],
                target_return_city=veh.get('home_terminal', veh.get('origin_city', 'Chicago')),
                vehicle_trailer_type=veh.get('trailer_type', 'Dry Van'),
                top_n=5
            )
            for c in cands:
                if c['load_id'] not in used_loads:
                    used_loads.add(c['load_id'])
                    assignments.append({
                        "vehicle_id": v_id,
                        "current_trip": veh,
                        "recommended_load": c
                    })
                    total_empty_miles_saved += c['shipment_distance_miles']
                    total_payload_added += c['shipment_weight_lbs']
                    break

        return {
            "solver_status": "GREEDY_FALLBACK",
            "matched_assignments_count": len(assignments),
            "assignments": assignments,
            "total_empty_miles_saved": round(total_empty_miles_saved, 1),
            "total_payload_added_lbs": round(total_payload_added, 1)
        }

def recommend_return_load_for_trip(trip_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for generating return load recommendations for a high-risk trip.
    """
    # 1. Run ML Prediction
    ml_res = predict_empty_return(trip_data)
    prob_str = ml_res['empty_return_probability']
    prob_val = ml_res['probability_value']
    risk_level = ml_res['risk_level']

    dest_city = trip_data.get("destination_city", "Las Vegas")
    target_city = trip_data.get("truck_home_terminal", trip_data.get("origin_city", "Chicago"))
    trailer_type = trip_data.get("trailer_type", "Dry Van")

    matcher = ReturnLoadMatcher()
    candidates = matcher.filter_and_rank_candidates(
        current_dest_city=dest_city,
        target_return_city=target_city,
        vehicle_trailer_type=trailer_type,
        top_n=3
    )

    return {
        "vehicle": {
            "truck_id": trip_data.get("truck_id", "TRK00001"),
            "trailer_type": trailer_type,
            "home_terminal": target_city,
            "capacity_lbs": MAX_CAPACITY_LBS
        },
        "current_trip": {
            "origin": trip_data.get("origin_city", "Chicago"),
            "destination": dest_city,
            "current_load_weight_lbs": trip_data.get("current_weight_lbs", 18000.0),
            "remaining_capacity_lbs": MAX_CAPACITY_LBS - float(trip_data.get("current_weight_lbs", 18000.0)),
            "empty_return_probability": prob_str,
            "risk_level": risk_level
        },
        "recommended_return_loads": candidates
    }

if __name__ == "__main__":
    sample_high_risk_trip = {
        "truck_id": "TRK00035",
        "origin_city": "Chicago",
        "destination_city": "Las Vegas",
        "truck_home_terminal": "Omaha",
        "trailer_type": "Dry Van",
        "booking_type": "Spot",
        "current_load_type": "Dry Van",
        "current_weight_lbs": 15118.0,
        "route_distance": 1749.0
    }

    print("Running Phase 3 Return Load Recommendation System:")
    rec = recommend_return_load_for_trip(sample_high_risk_trip)
    print(json.dumps(rec, indent=2))

import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List

try:
    from src.config import DATASET_DIR, PROJECT_ROOT
    from src.data_loader import DataLoader
except ImportError:
    from config import DATASET_DIR, PROJECT_ROOT
    from data_loader import DataLoader

# Environmental and Fuel Constants (Industry Standards & Fleet Averages)
AVERAGE_FLEET_MPG = 6.8         # Average miles per gallon for heavy-duty Class 8 trucks
DIESEL_PRICE_PER_GALLON = 4.15  # Average diesel fuel price ($ per gallon)
CO2_KG_PER_GALLON_DIESEL = 10.18 # EPA standard: 10.18 kg CO2 emitted per gallon of diesel burned

class ImpactCalculator:
    """
    Calculates operational, financial, and environmental sustainability impacts
    achieved by return-load matching.
    """
    def __init__(self, mpg: float = AVERAGE_FLEET_MPG, fuel_price: float = DIESEL_PRICE_PER_GALLON):
        self.mpg = mpg
        self.fuel_price = fuel_price

    def calculate_assignment_impact(self, assignment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates impact metrics for a single matched return assignment.
        """
        rec_load = assignment.get("recommended_load", {})
        curr_trip = assignment.get("current_trip", {})

        shipment_dist = float(rec_load.get("shipment_distance_miles", 0.0))
        detour_miles = float(rec_load.get("detour_miles", 0.0))
        shipment_weight = float(rec_load.get("shipment_weight_lbs", 0.0))
        revenue = float(rec_load.get("revenue", 0.0))

        # Net useful miles saved from running completely empty deadhead
        net_useful_miles_saved = max(0.0, shipment_dist - detour_miles)
        
        # Fuel savings from avoiding empty deadhead miles
        fuel_gallons_saved = round(net_useful_miles_saved / self.mpg, 2)
        
        # Direct fuel cost savings
        fuel_cost_savings = round(fuel_gallons_saved * self.fuel_price, 2)
        
        # Total financial benefit = Revenue earned from return load + fuel savings
        total_economic_benefit = round(revenue + fuel_cost_savings, 2)
        
        # CO2 Emissions reduction (kg)
        co2_reduction_kg = round(fuel_gallons_saved * CO2_KG_PER_GALLON_DIESEL, 2)
        
        # Payload utilization gained percentage
        utilization_gained_pct = round((shipment_weight / 45000.0) * 100.0, 1)

        return {
            "empty_miles_avoided": round(net_useful_miles_saved, 1),
            "payload_weight_hauled_lbs": shipment_weight,
            "utilization_gained_pct": utilization_gained_pct,
            "fuel_gallons_saved": fuel_gallons_saved,
            "fuel_cost_savings_usd": fuel_cost_savings,
            "freight_revenue_earned_usd": revenue,
            "total_economic_benefit_usd": total_economic_benefit,
            "co2_reduction_kg": co2_reduction_kg
        }

    def calculate_fleet_impact(self, assignments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregates impact metrics across an entire fleet of matched return dispatches.
        """
        total_empty_miles = 0.0
        total_payload_lbs = 0.0
        total_fuel_gallons = 0.0
        total_fuel_savings_usd = 0.0
        total_revenue_usd = 0.0
        total_economic_usd = 0.0
        total_co2_kg = 0.0

        for assign in assignments:
            imp = self.calculate_assignment_impact(assign)
            total_empty_miles += imp["empty_miles_avoided"]
            total_payload_lbs += imp["payload_weight_hauled_lbs"]
            total_fuel_gallons += imp["fuel_gallons_saved"]
            total_fuel_savings_usd += imp["fuel_cost_savings_usd"]
            total_revenue_usd += imp["freight_revenue_earned_usd"]
            total_economic_usd += imp["total_economic_benefit_usd"]
            total_co2_kg += imp["co2_reduction_kg"]

        avg_utilization_gain = round((total_payload_lbs / (len(assignments) * 45000.0)) * 100.0, 1) if assignments else 0.0

        return {
            "matched_trips_count": len(assignments),
            "total_empty_miles_avoided": round(total_empty_miles, 1),
            "total_payload_hauled_lbs": round(total_payload_lbs, 1),
            "average_return_utilization_gained_pct": avg_utilization_gain,
            "total_fuel_gallons_saved": round(total_fuel_gallons, 1),
            "total_fuel_cost_savings_usd": round(total_fuel_savings_usd, 2),
            "total_freight_revenue_earned_usd": round(total_revenue_usd, 2),
            "total_economic_impact_usd": round(total_economic_usd, 2),
            "total_co2_emissions_reduction_kg": round(total_co2_kg, 1),
            "total_co2_emissions_reduction_metric_tons": round(total_co2_kg / 1000.0, 2),
            "calculation_assumptions": {
                "fleet_average_mpg": self.mpg,
                "diesel_price_per_gallon_usd": self.fuel_price,
                "epa_co2_emission_factor_kg_per_gal": CO2_KG_PER_GALLON_DIESEL
            }
        }

if __name__ == "__main__":
    sample_assignment = {
        "recommended_load": {
            "shipment_distance_miles": 712.0,
            "detour_miles": 83.1,
            "shipment_weight_lbs": 44179.0,
            "revenue": 1576.46
        },
        "current_trip": {
            "origin": "Chicago",
            "destination": "Las Vegas"
        }
    }

    calc = ImpactCalculator()
    res = calc.calculate_assignment_impact(sample_assignment)
    print("Single Assignment Impact Calculation:")
    print(json.dumps(res, indent=2))

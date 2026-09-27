import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import DataLoader

def inspect_matching_sources():
    loader = DataLoader()
    
    print("=== INSPECTING SHIPMENT & MATCHING DATA SOURCES ===")
    facilities = loader.load_dataset("facilities.csv")
    routes = loader.load_dataset("routes.csv")
    distance = loader.load_dataset("distance.csv")
    loads = loader.load_dataset("loads.csv")
    shipment = loader.load_dataset("shipment.csv")
    order_large = loader.load_dataset("order_large.csv")

    print("\nFacilities sample (city, lat/lon):")
    print(facilities[['facility_id', 'facility_name', 'city', 'state', 'latitude', 'longitude']].head(5).to_string())

    print("\nRoutes sample (origin, destination, miles):")
    print(routes[['route_id', 'origin_city', 'origin_state', 'destination_city', 'destination_state', 'typical_distance_miles']].head(5).to_string())

    print("\nDistance.csv sample:")
    print(distance.head(5).to_string())

    print("\nLoads sample:")
    print(loads.head(3).to_string())

    print("\nOrder Large sample:")
    print(order_large.head(3).to_string())

if __name__ == "__main__":
    inspect_matching_sources()

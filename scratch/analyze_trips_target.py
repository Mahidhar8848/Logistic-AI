import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import DataLoader

def analyze_target_feasibility():
    loader = DataLoader()
    
    trips = loader.load_dataset("trips.csv")
    loads = loader.load_dataset("loads.csv")
    routes = loader.load_dataset("routes.csv")
    trucks = loader.load_dataset("trucks.csv")
    trailers = loader.load_dataset("trailers.csv")
    delivery = loader.load_dataset("delivery_events.csv")
    utilization = loader.load_dataset("truck_utilization_metrics.csv")
    dynamic_sc = loader.load_dataset("dynamic_supply_chain_logistics_dataset.csv")

    print("=== TRIPS & LOADS MERGE ANALYSES ===")
    print(f"Trips shape: {trips.shape}")
    print(f"Loads shape: {loads.shape}")
    
    merged_trips = trips.merge(loads, on="load_id", how="left").merge(routes, on="route_id", how="left")
    print("Merged trips columns:", list(merged_trips.columns))
    print("\nLoad Types distribution:")
    print(loads['load_type'].value_counts(dropna=False))
    
    print("\nBooking Types distribution:")
    print(loads['booking_type'].value_counts(dropna=False))

    print("\nLoad Status distribution:")
    print(loads['load_status'].value_counts(dropna=False))

    print("\nTrip Status distribution:")
    print(trips['trip_status'].value_counts(dropna=False))

    print("\nWeight statistics:")
    print(loads['weight_lbs'].describe())
    
    # Check consecutive trips for trucks
    merged_trips['dispatch_date'] = pd.to_datetime(merged_trips['dispatch_date'])
    merged_trips = merged_trips.sort_values(by=['truck_id', 'dispatch_date'])
    
    # Look at sample sequence for a single truck
    sample_truck = merged_trips['truck_id'].iloc[0]
    truck_seq = merged_trips[merged_trips['truck_id'] == sample_truck][
        ['trip_id', 'truck_id', 'dispatch_date', 'origin_city', 'origin_state', 'destination_city', 'destination_state', 'load_type', 'weight_lbs', 'booking_type']
    ].head(10)
    print(f"\nSample Trip Sequence for Truck {sample_truck}:")
    print(truck_seq.to_string())

    # Check origin vs destination chaining across consecutive trips
    merged_trips['next_origin_city'] = merged_trips.groupby('truck_id')['origin_city'].shift(-1)
    merged_trips['next_origin_state'] = merged_trips.groupby('truck_id')['origin_state'].shift(-1)
    merged_trips['next_load_type'] = merged_trips.groupby('truck_id')['load_type'].shift(-1)
    merged_trips['next_weight_lbs'] = merged_trips.groupby('truck_id')['weight_lbs'].shift(-1)
    merged_trips['next_booking_type'] = merged_trips.groupby('truck_id')['booking_type'].shift(-1)

    print("\nNext trip analysis (same location return vs empty return vs deadhead):")
    # Same origin as current destination means next trip starts where current ended!
    same_location_next = (merged_trips['destination_city'] == merged_trips['next_origin_city'])
    print(f"Total consecutive trips where Next Origin == Current Destination: {same_location_next.sum()} / {len(merged_trips.dropna(subset=['next_origin_city']))}")
    
    # Let's inspect dynamic_supply_chain_logistics_dataset.csv as well
    print("\n=== DYNAMIC SUPPLY CHAIN DATASET COLUMNS & HEAD ===")
    print(dynamic_sc.columns.tolist())
    print(dynamic_sc.head(2).to_dict(orient="records"))

if __name__ == "__main__":
    analyze_target_feasibility()

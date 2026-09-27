import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import DataLoader

def analyze_target_formulations():
    loader = DataLoader()
    trips = loader.load_dataset("trips.csv")
    loads = loader.load_dataset("loads.csv")
    routes = loader.load_dataset("routes.csv")
    trucks = loader.load_dataset("trucks.csv")
    trailers = loader.load_dataset("trailers.csv")

    df = trips.merge(loads, on="load_id", how="left").merge(routes, on="route_id", how="left")
    df['dispatch_date'] = pd.to_datetime(df['dispatch_date'])
    
    # Sort chronologically by truck and dispatch date
    df = df.sort_values(by=['truck_id', 'dispatch_date', 'trip_id']).reset_index(drop=True)

    # Calculate max capacity (45,000 lbs standard for trailer)
    CAPACITY_LBS = 45000.0
    df['utilization'] = df['weight_lbs'] / CAPACITY_LBS

    # Look ahead to the NEXT trip for the same truck
    df['next_trip_id'] = df.groupby('truck_id')['trip_id'].shift(-1)
    df['next_dispatch_date'] = df.groupby('truck_id')['dispatch_date'].shift(-1)
    df['next_origin_city'] = df.groupby('truck_id')['origin_city'].shift(-1)
    df['next_origin_state'] = df.groupby('truck_id')['origin_state'].shift(-1)
    df['next_destination_city'] = df.groupby('truck_id')['destination_city'].shift(-1)
    df['next_destination_state'] = df.groupby('truck_id')['destination_state'].shift(-1)
    df['next_weight_lbs'] = df.groupby('truck_id')['weight_lbs'].shift(-1)
    df['next_booking_type'] = df.groupby('truck_id')['booking_type'].shift(-1)
    df['next_revenue'] = df.groupby('truck_id')['revenue'].shift(-1)
    df['days_to_next_dispatch'] = (df['next_dispatch_date'] - df['dispatch_date']).dt.total_seconds() / (24*3600)

    # Define return journey relationships:
    # Scenario A: Next trip originates directly from current destination city (Local Return/Backhaul)
    same_location_next = (df['destination_city'] == df['next_origin_city'])
    
    # Scenario B: Next trip returns directly back to current origin city (Round-trip)
    round_trip_next = (df['destination_city'] == df['next_origin_city']) & (df['next_destination_city'] == df['origin_city'])

    # Deadhead repositioning needed if next_origin != current_destination
    deadhead_reposition = (df['destination_city'] != df['next_origin_city']) & df['next_origin_city'].notnull()

    print("=== TRIP CHAINING STATISTICS ===")
    print(f"Total valid trips analyzed: {len(df):,}")
    print(f"Trips with a subsequent trip for same truck: {df['next_trip_id'].notnull().sum():,}")
    print(f"Subsequent trip originates from same destination city: {same_location_next.sum():,} ({same_location_next.mean()*100:.2f}%)")
    print(f"Subsequent trip is a direct round-trip back to origin: {round_trip_next.sum():,} ({round_trip_next.mean()*100:.2f}%)")
    print(f"Subsequent trip requires deadhead repositioning to another city: {deadhead_reposition.sum():,} ({deadhead_reposition.mean()*100:.2f}%)")

    print("\n=== NEXT TRIP LOAD WEIGHT DISTRIBUTION ===")
    print(df['next_weight_lbs'].describe())

    print("\n=== NEXT TRIP UTILIZATION DISTRIBUTION ===")
    next_util = df['next_weight_lbs'] / CAPACITY_LBS
    print(next_util.describe())

    # Target Definition Threshold Analysis
    print("\n=== TARGET CANDIDATE DEFINITIONS ===")
    # Candidate Target 1: Underutilized Return (< 50% capacity, i.e., < 22,500 lbs)
    underutilized_50 = (next_util < 0.50)
    print(f"Target 1: Next Trip Weight < 50% Capacity (22,500 lbs): {underutilized_50.sum():,} ({underutilized_50.mean()*100:.2f}%)")

    # Candidate Target 2: Highly Underutilized Return (< 35% capacity, i.e., < 15,750 lbs)
    underutilized_35 = (next_util < 0.35)
    print(f"Target 2: Next Trip Weight < 35% Capacity (15,750 lbs): {underutilized_35.sum():,} ({underutilized_35.mean()*100:.2f}%)")

    # Candidate Target 3: Deadhead OR Underutilized (< 50% capacity OR repositioning needed)
    deadhead_or_low_util = deadhead_reposition | (next_util < 0.50)
    print(f"Target 3: Deadhead Repositioning OR Next Weight < 50%: {deadhead_or_low_util.sum():,} ({deadhead_or_low_util.mean()*100:.2f}%)")

    # Save summary to file for documentation
    output_df = df[['trip_id', 'truck_id', 'dispatch_date', 'origin_city', 'destination_city', 'weight_lbs', 
                    'next_trip_id', 'next_origin_city', 'next_destination_city', 'next_weight_lbs', 'days_to_next_dispatch']]
    output_df.head(100).to_csv("scratch/target_analysis_sample.csv", index=False)
    print("\nSaved sample to scratch/target_analysis_sample.csv")

if __name__ == "__main__":
    analyze_target_formulations()

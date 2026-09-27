import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import DataLoader

def build_feature_pipeline():
    loader = DataLoader()
    
    print("Loading datasets for feature engineering...")
    trips = loader.load_dataset("trips.csv")
    loads = loader.load_dataset("loads.csv")
    routes = loader.load_dataset("routes.csv")
    trucks = loader.load_dataset("trucks.csv")
    trailers = loader.load_dataset("trailers.csv")
    drivers = loader.load_dataset("drivers.csv")
    driver_metrics = loader.load_dataset("driver_monthly_metrics.csv")
    truck_metrics = loader.load_dataset("truck_utilization_metrics.csv")
    telemetry = loader.load_dataset("dynamic_supply_chain_logistics_dataset.csv")

    print(f"Initial Trips: {len(trips):,} rows")
    
    # 1. Merge Trips + Loads + Routes + Trucks + Trailers + Drivers
    df = trips.merge(loads, on="load_id", how="inner")
    df = df.merge(routes, on="route_id", how="left")
    df = df.merge(trucks, on="truck_id", how="left", suffixes=('', '_truck'))
    df = df.merge(trailers, on="trailer_id", how="left", suffixes=('', '_trailer'))
    df = df.merge(drivers, on="driver_id", how="left", suffixes=('', '_driver'))

    # Convert dispatch_date to datetime
    df['dispatch_date'] = pd.to_datetime(df['dispatch_date'])
    
    # Sort chronologically by truck and dispatch date
    df = df.sort_values(by=['truck_id', 'dispatch_date', 'trip_id']).reset_index(drop=True)

    # 2. Derive Target Variable
    # Capacity max payload = 45,000 lbs
    CAPACITY_LBS = 45000.0
    
    df['next_weight_lbs'] = df.groupby('truck_id')['weight_lbs'].shift(-1)
    df['next_origin_city'] = df.groupby('truck_id')['origin_city'].shift(-1)
    df['next_dispatch_date'] = df.groupby('truck_id')['dispatch_date'].shift(-1)
    
    # Filter out trips where truck has no subsequent trip (e.g. final trip in log)
    valid_mask = df['next_weight_lbs'].notnull()
    df_valid = df[valid_mask].copy()
    
    df_valid['next_return_utilization'] = df_valid['next_weight_lbs'] / CAPACITY_LBS
    
    # Target definition validated in Phase 1:
    # 1 if return utilization < 0.50 (weight < 22,500 lbs), 0 otherwise
    df_valid['empty_return_target'] = (df_valid['next_return_utilization'] < 0.50).astype(int)
    
    print(f"Valid dataset size after target derivation: {len(df_valid):,} rows")
    print("Target class distribution:")
    print(df_valid['empty_return_target'].value_counts(normalize=True))

    # 3. Leakage-Safe Feature Engineering
    # A. Temporal features
    df_valid['dispatch_month'] = df_valid['dispatch_date'].dt.month
    df_valid['dispatch_dayofweek'] = df_valid['dispatch_date'].dt.dayofweek
    df_valid['dispatch_quarter'] = df_valid['dispatch_date'].dt.quarter
    df_valid['is_weekend'] = df_valid['dispatch_dayofweek'].isin([5, 6]).astype(int)

    # B. Vehicle & Trailer Features
    df_valid['truck_make'] = df_valid['make'].fillna('Unknown')
    df_valid['truck_model_year'] = df_valid['model_year'].fillna(df_valid['model_year'].median())
    df_valid['truck_tank_capacity'] = df_valid['tank_capacity_gallons'].fillna(200)
    df_valid['truck_home_terminal'] = df_valid['home_terminal'].fillna('Unknown')
    df_valid['trailer_type'] = df_valid['trailer_type'].fillna('Unknown')
    df_valid['trailer_length'] = df_valid['length_feet'].fillna(53)

    # C. Driver Baseline Features
    df_valid['driver_years_exp'] = df_valid['years_experience'].fillna(df_valid['years_experience'].median())
    df_valid['driver_cdl_class'] = df_valid['cdl_class'].fillna('Class A')

    # D. Current Trip & Route Features
    df_valid['current_load_type'] = df_valid['load_type'].fillna('Unknown')
    df_valid['current_weight_lbs'] = df_valid['weight_lbs']
    df_valid['current_pieces'] = df_valid['pieces']
    df_valid['current_revenue'] = df_valid['revenue']
    df_valid['booking_type'] = df_valid['booking_type'].fillna('Spot')
    df_valid['current_load_utilization'] = df_valid['weight_lbs'] / CAPACITY_LBS
    
    df_valid['route_distance'] = df_valid['typical_distance_miles']
    df_valid['route_base_rate'] = df_valid['base_rate_per_mile']
    df_valid['route_fuel_surcharge_rate'] = df_valid['fuel_surcharge_rate']
    df_valid['route_transit_days'] = df_valid['typical_transit_days']
    
    # E. Regional Historical Return Load Availability (Destination Demand Proxy)
    # Estimate volume of loads originating from current destination city in preceding historical window
    origin_counts = df_valid['origin_city'].value_counts().to_dict()
    df_valid['dest_outbound_market_volume'] = df_valid['destination_city'].map(origin_counts).fillna(0)
    
    # F. Deadhead indicator preview (Is destination equal to home terminal?)
    df_valid['is_heading_home'] = (df_valid['destination_city'] == df_valid['truck_home_terminal']).astype(int)

    print("\nFeature Summary Sample:")
    feature_cols = [
        'dispatch_month', 'dispatch_dayofweek', 'dispatch_quarter', 'is_weekend',
        'truck_make', 'truck_model_year', 'truck_tank_capacity', 'truck_home_terminal',
        'trailer_type', 'trailer_length', 'driver_years_exp',
        'current_load_type', 'current_weight_lbs', 'current_pieces', 'current_revenue',
        'booking_type', 'current_load_utilization', 'route_distance', 'route_base_rate',
        'route_fuel_surcharge_rate', 'route_transit_days', 'dest_outbound_market_volume',
        'is_heading_home'
    ]
    
    print(df_valid[feature_cols].head(3).to_string())
    print("\nCheck nulls in selected features:")
    print(df_valid[feature_cols].isnull().sum())

if __name__ == "__main__":
    build_feature_pipeline()

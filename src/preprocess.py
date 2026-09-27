import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
from typing import Tuple

try:
    from src.config import DATASET_DIR, PROJECT_ROOT
    from src.data_loader import DataLoader
except ImportError:
    from config import DATASET_DIR, PROJECT_ROOT
    from data_loader import DataLoader

def run_preprocessing() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes full data preprocessing, leakage-safe feature engineering,
    and chronological train/val/test splitting for Phase 2.
    """
    loader = DataLoader(DATASET_DIR)
    
    print("Loading raw operational datasets...")
    trips = loader.load_dataset("trips.csv")
    loads = loader.load_dataset("loads.csv")
    routes = loader.load_dataset("routes.csv")
    trucks = loader.load_dataset("trucks.csv")
    trailers = loader.load_dataset("trailers.csv")
    drivers = loader.load_dataset("drivers.csv")

    # 1. Merge Core Tables
    df = trips.merge(loads, on="load_id", how="inner")
    df = df.merge(routes, on="route_id", how="left")
    df = df.merge(trucks, on="truck_id", how="left", suffixes=('', '_truck'))
    df = df.merge(trailers, on="trailer_id", how="left", suffixes=('', '_trailer'))
    df = df.merge(drivers, on="driver_id", how="left", suffixes=('', '_driver'))

    # Datatype correction & Timestamp conversion
    df['dispatch_date'] = pd.to_datetime(df['dispatch_date'])
    
    # Sort chronologically by truck and dispatch date
    df = df.sort_values(by=['dispatch_date', 'trip_id']).reset_index(drop=True)

    # 2. Target Creation (Phase 1 Validated Target Rule)
    # Maximum Payload Capacity = 45,000 lbs (Standard 53ft trailer)
    CAPACITY_LBS = 45000.0
    
    # Identify subsequent return load weight for the same vehicle
    df['next_weight_lbs'] = df.groupby('truck_id')['weight_lbs'].shift(-1)
    
    # Filter out trips where truck has no subsequent trip (e.g. final logged trip per truck)
    df_valid = df[df['next_weight_lbs'].notnull()].copy()
    
    # Compute return utilization rate
    df_valid['next_return_utilization'] = df_valid['next_weight_lbs'] / CAPACITY_LBS
    
    # Binary Target: 1 if Return Utilization < 50% (< 22,500 lbs), 0 otherwise
    df_valid['empty_return_target'] = (df_valid['next_return_utilization'] < 0.50).astype(int)

    print(f"Total Dispatches Processed: {len(df_valid):,}")
    print(f"Target Distribution: {df_valid['empty_return_target'].value_counts(normalize=True).to_dict()}")

    # 3. Leakage-Safe Feature Engineering
    # A. Temporal Features
    df_valid['dispatch_month'] = df_valid['dispatch_date'].dt.month
    df_valid['dispatch_dayofweek'] = df_valid['dispatch_date'].dt.dayofweek
    df_valid['dispatch_quarter'] = df_valid['dispatch_date'].dt.quarter
    df_valid['is_weekend'] = df_valid['dispatch_dayofweek'].isin([5, 6]).astype(int)

    # B. Vehicle & Trailer Attributes
    df_valid['truck_make'] = df_valid['make'].fillna('Unknown')
    df_valid['truck_model_year'] = df_valid['model_year'].fillna(df_valid['model_year'].median())
    df_valid['truck_tank_capacity'] = df_valid['tank_capacity_gallons'].fillna(200.0)
    df_valid['truck_home_terminal'] = df_valid['home_terminal'].fillna('Unknown')
    df_valid['trailer_type'] = df_valid['trailer_type'].fillna('Unknown')
    df_valid['trailer_length'] = df_valid['length_feet'].fillna(53.0)

    # C. Driver Baseline Attributes
    df_valid['driver_years_exp'] = df_valid['years_experience'].fillna(df_valid['years_experience'].median())
    df_valid['driver_cdl_class'] = df_valid['cdl_class'].fillna('Class A')

    # D. Current Trip & Route Features
    df_valid['current_load_type'] = df_valid['load_type'].fillna('Unknown')
    df_valid['current_weight_lbs'] = df_valid['weight_lbs'].fillna(df_valid['weight_lbs'].median())
    df_valid['current_pieces'] = df_valid['pieces'].fillna(df_valid['pieces'].median())
    df_valid['current_revenue'] = df_valid['revenue'].fillna(df_valid['revenue'].median())
    df_valid['booking_type'] = df_valid['booking_type'].fillna('Spot')
    df_valid['current_load_utilization'] = df_valid['current_weight_lbs'] / CAPACITY_LBS
    
    df_valid['route_distance'] = df_valid['typical_distance_miles'].fillna(df_valid['typical_distance_miles'].median())
    df_valid['route_base_rate'] = df_valid['base_rate_per_mile'].fillna(df_valid['base_rate_per_mile'].median())
    df_valid['route_fuel_surcharge_rate'] = df_valid['fuel_surcharge_rate'].fillna(0.20)
    df_valid['route_transit_days'] = df_valid['typical_transit_days'].fillna(1.0)
    
    df_valid['origin_city'] = df_valid['origin_city'].fillna('Unknown')
    df_valid['origin_state'] = df_valid['origin_state'].fillna('Unknown')
    df_valid['destination_city'] = df_valid['destination_city'].fillna('Unknown')
    df_valid['destination_state'] = df_valid['destination_state'].fillna('Unknown')

    # E. Market Demand Proxy & Geographic Repositioning
    dest_demand_map = df_valid['origin_city'].value_counts().to_dict()
    df_valid['dest_outbound_market_volume'] = df_valid['destination_city'].map(dest_demand_map).fillna(0)
    df_valid['is_heading_home'] = (df_valid['destination_city'] == df_valid['truck_home_terminal']).astype(int)

    # F. Cumulative Historical Empty Rate per Route & Vehicle (Expanding Window, Zero Leakage)
    df_valid['route_cum_empty'] = df_valid.groupby('route_id')['empty_return_target'].cumsum() - df_valid['empty_return_target']
    df_valid['route_cum_count'] = df_valid.groupby('route_id').cumcount()
    df_valid['route_hist_empty_rate'] = np.where(df_valid['route_cum_count'] > 0, df_valid['route_cum_empty'] / df_valid['route_cum_count'], 0.35)

    df_valid['truck_cum_empty'] = df_valid.groupby('truck_id')['empty_return_target'].cumsum() - df_valid['empty_return_target']
    df_valid['truck_cum_count'] = df_valid.groupby('truck_id').cumcount()
    df_valid['truck_hist_empty_rate'] = np.where(df_valid['truck_cum_count'] > 0, df_valid['truck_cum_empty'] / df_valid['truck_cum_count'], 0.35)

    # Feature List Selection
    feature_cols = [
        # Categorical
        'origin_city', 'origin_state', 'destination_city', 'destination_state',
        'truck_make', 'truck_home_terminal', 'trailer_type', 'booking_type', 'current_load_type',
        # Numerical / Temporal
        'dispatch_month', 'dispatch_dayofweek', 'dispatch_quarter', 'is_weekend',
        'truck_model_year', 'truck_tank_capacity', 'trailer_length', 'driver_years_exp',
        'current_weight_lbs', 'current_pieces', 'current_revenue', 'current_load_utilization',
        'route_distance', 'route_base_rate', 'route_fuel_surcharge_rate', 'route_transit_days',
        'dest_outbound_market_volume', 'is_heading_home', 'route_hist_empty_rate', 'truck_hist_empty_rate'
    ]

    target_col = 'empty_return_target'
    metadata_cols = ['trip_id', 'truck_id', 'dispatch_date', 'next_weight_lbs', 'next_return_utilization']

    all_cols = metadata_cols + feature_cols + [target_col]
    clean_df = df_valid[all_cols].copy()

    # 4. Chronological Train / Validation / Test Split (70% Train, 15% Val, 15% Test)
    clean_df = clean_df.sort_values(by='dispatch_date').reset_index(drop=True)
    n = len(clean_df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    train_df = clean_df.iloc[:train_end].copy()
    val_df = clean_df.iloc[train_end:val_end].copy()
    test_df = clean_df.iloc[val_end:].copy()

    print(f"\nChronological Split Results:")
    print(f" - Train set: {len(train_df):,} rows ({train_df['dispatch_date'].min().date()} to {train_df['dispatch_date'].max().date()})")
    print(f" - Val set:   {len(val_df):,} rows ({val_df['dispatch_date'].min().date()} to {val_df['dispatch_date'].max().date()})")
    print(f" - Test set:  {len(test_df):,} rows ({test_df['dispatch_date'].min().date()} to {test_df['dispatch_date'].max().date()})")

    # Save to data/processed
    processed_dir = PROJECT_ROOT / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(processed_dir / "train_processed.csv", index=False)
    val_df.to_csv(processed_dir / "val_processed.csv", index=False)
    test_df.to_csv(processed_dir / "test_processed.csv", index=False)
    clean_df.to_csv(processed_dir / "full_processed.csv", index=False)

    # Save feature metadata
    feature_meta = {
        "categorical_features": [
            'origin_city', 'origin_state', 'destination_city', 'destination_state',
            'truck_make', 'truck_home_terminal', 'trailer_type', 'booking_type', 'current_load_type'
        ],
        "numerical_features": [
            'dispatch_month', 'dispatch_dayofweek', 'dispatch_quarter', 'is_weekend',
            'truck_model_year', 'truck_tank_capacity', 'trailer_length', 'driver_years_exp',
            'current_weight_lbs', 'current_pieces', 'current_revenue', 'current_load_utilization',
            'route_distance', 'route_base_rate', 'route_fuel_surcharge_rate', 'route_transit_days',
            'dest_outbound_market_volume', 'is_heading_home', 'route_hist_empty_rate', 'truck_hist_empty_rate'
        ],
        "target_col": target_col,
        "metadata_cols": metadata_cols
    }

    with open(processed_dir / "feature_metadata.json", "w") as f:
        json.dump(feature_meta, f, indent=2)

    print(f"\nSuccessfully saved processed datasets to {processed_dir}")
    return train_df, val_df, test_df

if __name__ == "__main__":
    run_preprocessing()

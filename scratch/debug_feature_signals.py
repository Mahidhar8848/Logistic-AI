import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import DataLoader

def analyze_feature_signals():
    loader = DataLoader()
    trips = loader.load_dataset("trips.csv")
    loads = loader.load_dataset("loads.csv")
    routes = loader.load_dataset("routes.csv")
    trucks = loader.load_dataset("trucks.csv")

    df = trips.merge(loads, on="load_id", how="inner").merge(routes, on="route_id", how="left")
    df = df.merge(trucks, on="truck_id", how="left")
    df['dispatch_date'] = pd.to_datetime(df['dispatch_date'])
    
    # Sort strictly by truck_id and dispatch_date
    df = df.sort_values(by=['truck_id', 'dispatch_date']).reset_index(drop=True)
    
    # Target derivation
    CAPACITY_LBS = 45000.0
    df['next_weight_lbs'] = df.groupby('truck_id')['weight_lbs'].shift(-1)
    df_valid = df[df['next_weight_lbs'].notnull()].copy()
    df_valid['next_return_utilization'] = df_valid['next_weight_lbs'] / CAPACITY_LBS
    df_valid['empty_return_target'] = (df_valid['next_return_utilization'] < 0.50).astype(int)

    print("=== LANE INBALANCE & MARKET FEATURES ===")
    
    # 1. Outbound vs Inbound city volume (Headhaul / Backhaul Index)
    city_outbound = df_valid['origin_city'].value_counts()
    city_inbound = df_valid['destination_city'].value_counts()
    
    city_imbalance = (city_outbound / city_inbound).fillna(1.0).to_dict()
    df_valid['dest_imbalance_ratio'] = df_valid['destination_city'].map(city_imbalance).fillna(1.0)
    
    print("\nEmpty return rate by Destination Imbalance Ratio quantile:")
    df_valid['imbalance_quantile'] = pd.qcut(df_valid['dest_imbalance_ratio'], q=4, duplicates='drop')
    print(df_valid.groupby('imbalance_quantile')['empty_return_target'].agg(['count', 'mean']))

    # 2. Target rate by Booking Type
    print("\nEmpty return rate by Booking Type:")
    print(df_valid.groupby('booking_type')['empty_return_target'].agg(['count', 'mean']))

    # 3. Target rate by Load Type
    print("\nEmpty return rate by Load Type:")
    print(df_valid.groupby('load_type')['empty_return_target'].agg(['count', 'mean']))

    # 4. Target rate by Destination City (Top 10 highest vs lowest)
    dest_rates = df_valid.groupby('destination_city')['empty_return_target'].agg(['count', 'mean']).sort_values(by='mean', ascending=False)
    print("\nTop 5 Destination Cities with highest empty return rate:")
    print(dest_rates.head(5))
    print("\nTop 5 Destination Cities with lowest empty return rate:")
    print(dest_rates.tail(5))

    # 5. Target rate by Route ID
    route_rates = df_valid.groupby('route_id')['empty_return_target'].agg(['count', 'mean']).sort_values(by='mean', ascending=False)
    print("\nTop 5 Routes with highest empty return rate:")
    print(route_rates.head(5))

    # 6. Distance to Home Terminal
    df_valid['dist_to_home'] = (df_valid['destination_city'] != df_valid['home_terminal']).astype(int)
    print("\nEmpty return rate by Heading away from Home Terminal vs at Home Terminal:")
    print(df_valid.groupby('dist_to_home')['empty_return_target'].agg(['count', 'mean']))

if __name__ == "__main__":
    analyze_feature_signals()

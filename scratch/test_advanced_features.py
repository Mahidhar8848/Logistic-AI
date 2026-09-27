import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import DataLoader

def test_advanced_features():
    loader = DataLoader()
    
    trips = loader.load_dataset("trips.csv")
    loads = loader.load_dataset("loads.csv")
    routes = loader.load_dataset("routes.csv")
    trucks = loader.load_dataset("trucks.csv")
    trailers = loader.load_dataset("trailers.csv")
    drivers = loader.load_dataset("drivers.csv")

    df = trips.merge(loads, on="load_id", how="inner")
    df = df.merge(routes, on="route_id", how="left")
    df = df.merge(trucks, on="truck_id", how="left", suffixes=('', '_truck'))
    df = df.merge(trailers, on="trailer_id", how="left", suffixes=('', '_trailer'))
    df = df.merge(drivers, on="driver_id", how="left", suffixes=('', '_driver'))

    df['dispatch_date'] = pd.to_datetime(df['dispatch_date'])
    df = df.sort_values(by=['dispatch_date', 'trip_id']).reset_index(drop=True)

    # Derive Target
    CAPACITY_LBS = 45000.0
    df['next_weight_lbs'] = df.groupby('truck_id')['weight_lbs'].shift(-1)
    df_valid = df[df['next_weight_lbs'].notnull()].copy()
    df_valid['next_return_utilization'] = df_valid['next_weight_lbs'] / CAPACITY_LBS
    df_valid['empty_return_target'] = (df_valid['next_return_utilization'] < 0.50).astype(int)

    # 1. Cumulative Expanding Historical Empty Rate per Route (Leakage Safe)
    df_valid['route_cum_empty'] = df_valid.groupby('route_id')['empty_return_target'].cumsum() - df_valid['empty_return_target']
    df_valid['route_cum_count'] = df_valid.groupby('route_id').cumcount()
    df_valid['route_hist_empty_rate'] = np.where(df_valid['route_cum_count'] > 0, df_valid['route_cum_empty'] / df_valid['route_cum_count'], 0.35)

    # 2. Cumulative Expanding Historical Empty Rate per Truck (Leakage Safe)
    df_valid['truck_cum_empty'] = df_valid.groupby('truck_id')['empty_return_target'].cumsum() - df_valid['empty_return_target']
    df_valid['truck_cum_count'] = df_valid.groupby('truck_id').cumcount()
    df_valid['truck_hist_empty_rate'] = np.where(df_valid['truck_cum_count'] > 0, df_valid['truck_cum_empty'] / df_valid['truck_cum_count'], 0.35)

    # 3. Destination City Outbound Load Demand Proxy
    dest_demand_map = df_valid['origin_city'].value_counts().to_dict()
    df_valid['dest_outbound_load_demand'] = df_valid['destination_city'].map(dest_demand_map).fillna(0)

    # 4. Same Route Round-trip frequency
    route_demand_map = df_valid['route_id'].value_counts().to_dict()
    df_valid['route_load_volume'] = df_valid['route_id'].map(route_demand_map).fillna(0)

    print("Advanced features test results:")
    print(df_valid[['route_id', 'route_hist_empty_rate', 'truck_id', 'truck_hist_empty_rate', 'dest_outbound_load_demand', 'empty_return_target']].head(10).to_string())
    
    print("\nCorrelation with empty_return_target:")
    corrs = df_valid[['route_hist_empty_rate', 'truck_hist_empty_rate', 'dest_outbound_load_demand', 
                      'typical_distance_miles', 'weight_lbs', 'empty_return_target']].corr()['empty_return_target']
    print(corrs)

if __name__ == "__main__":
    test_advanced_features()

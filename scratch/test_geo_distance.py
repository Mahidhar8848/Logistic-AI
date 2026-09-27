import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import DataLoader

def test_geo():
    loader = DataLoader()
    facilities = loader.load_dataset("facilities.csv")
    routes = loader.load_dataset("routes.csv")
    
    # Map city -> (lat, lon)
    city_coords = {}
    for _, row in facilities.iterrows():
        city_coords[row['city']] = (row['latitude'], row['longitude'])
        
    print(f"Mapped {len(city_coords)} cities from facilities:")
    for city, (lat, lon) in list(city_coords.items())[:5]:
        print(f" - {city}: ({lat}, {lon})")

    # Function to calculate Haversine distance in miles
    def haversine_miles(lat1, lon1, lat2, lon2):
        R = 3958.8 # Earth radius in miles
        phi1, phi2 = np.radians(lat1), np.radians(lat2)
        dphi = np.radians(lat2 - lat1)
        dlambda = np.radians(lon2 - lon1)
        a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
        return R * 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))

    print("\nSample route distance check vs Haversine x 1.25:")
    for _, row in routes.head(5).iterrows():
        o, d, miles = row['origin_city'], row['destination_city'], row['typical_distance_miles']
        if o in city_coords and d in city_coords:
            c1, c2 = city_coords[o], city_coords[d]
            h_miles = haversine_miles(c1[0], c1[1], c2[0], c2[1])
            est_road = h_miles * 1.22
            print(f" {o} -> {d}: Actual Route={miles} mi | Haversine={h_miles:.1f} mi | Est Road={est_road:.1f} mi")

if __name__ == "__main__":
    test_geo()

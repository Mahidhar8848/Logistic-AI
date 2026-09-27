import sys
from pathlib import Path
import pandas as pd
import json

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import DataLoader

def generate_report_tables():
    loader = DataLoader()
    files = loader.discover_datasets()
    
    rows = []
    for f in files:
        df = loader.load_dataset(f.name)
        file_type = f.suffix.upper().lstrip(".")
        size_kb = round(f.stat().st_size / 1024, 1)
        r_count = len(df)
        c_count = len(df.columns)
        missing_count = int(df.isnull().sum().sum())
        dup_count = int(df.duplicated().sum())
        cols_str = ", ".join(list(df.columns)[:5]) + ("..." if c_count > 5 else "")
        
        # Categorize
        if f.name in ["trips.csv", "loads.csv", "routes.csv", "trucks.csv", "trailers.csv", "drivers.csv"]:
            domain = "Core Logistics Operations"
        elif f.name in ["delivery_events.csv", "facilities.csv", "fuel_purchases.csv", "maintenance_records.csv", "safety_incidents.csv", "truck_utilization_metrics.csv", "driver_monthly_metrics.csv"]:
            domain = "Fleet & Telemetry Metrics"
        elif f.name in ["dynamic_supply_chain_logistics_dataset.csv"]:
            domain = "Real-time Telemetry & Risk IoT"
        elif f.name in ["customer.csv", "customers.csv", "supplier.csv", "product.csv", "inventory_snapshot.csv", "production_run.csv", "purchase_order.csv", "purchase_order_line.csv", "shipment.csv", "shipment_line.csv", "warehouse.csv"]:
            domain = "ERP & Supply Chain Orders"
        else:
            domain = "Optimization & Benchmarking"
            
        rows.append({
            "Dataset": f.name,
            "Domain": domain,
            "File Type": file_type,
            "Size (KB)": size_kb,
            "Rows": f"{r_count:,}",
            "Columns": c_count,
            "Missing": f"{missing_count:,}",
            "Duplicates": f"{dup_count:,}",
            "Sample Columns": cols_str
        })
        
    df_res = pd.DataFrame(rows)
    headers = list(df_res.columns)
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, r in df_res.iterrows():
        print("| " + " | ".join(str(r[h]) for h in headers) + " |")

if __name__ == "__main__":
    generate_report_tables()

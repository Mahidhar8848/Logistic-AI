import sys
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import DataLoader

def inspect_all_details():
    loader = DataLoader()
    dataset_files = loader.discover_datasets()
    
    print(f"=== FULL AUDIT OF ALL {len(dataset_files)} DATASETS ===\n")
    
    for f in dataset_files:
        df = loader.load_dataset(f.name)
        print("="*80)
        print(f"FILE: {f.name} | ROWS: {len(df):,} | COLS: {len(df.columns)}")
        print("Columns:", list(df.columns))
        print("Null values per column:")
        nulls = df.isnull().sum()
        null_cols = nulls[nulls > 0]
        if len(null_cols) == 0:
            print("  No missing values.")
        else:
            for col, count in null_cols.items():
                pct = (count / len(df)) * 100
                print(f"  - {col}: {count:,} missing ({pct:.2f}%)")
        print(f"Duplicates: {df.duplicated().sum():,}")
        print("Sample head:")
        print(df.head(1).to_string())
        print("\n")

if __name__ == "__main__":
    inspect_all_details()

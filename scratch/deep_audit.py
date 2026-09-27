import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json

# Add src to path
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
from data_loader import DataLoader

def run_deep_audit():
    loader = DataLoader()
    dataset_files = loader.discover_datasets()
    print(f"Total dataset files found: {len(dataset_files)}\n")

    summary_list = []
    columns_map = {}
    sample_rows = {}
    
    for f in dataset_files:
        filename = f.name
        try:
            df = loader.load_dataset(filename)
            missing = int(df.isnull().sum().sum())
            dups = int(df.duplicated().sum())
            cols = list(df.columns)
            columns_map[filename] = cols
            sample_rows[filename] = df.head(2).to_dict(orient="records")
            
            summary_list.append({
                "Filename": filename,
                "FileType": f.suffix.upper().lstrip("."),
                "Rows": len(df),
                "Columns": len(cols),
                "MissingValuesTotal": missing,
                "DuplicateRows": dups,
                "ColumnNames": cols
            })
            print(f"Loaded {filename}: {len(df)} rows, {len(cols)} cols.")
        except Exception as e:
            print(f"Error loading {filename}: {e}")

    # Save summary json for inspection
    with open("scratch/audit_raw_summary.json", "w") as out:
        json.dump(summary_list, out, indent=2)

    print("\n--- SUMMARY OF COLUMNS PER DATASET ---")
    for fname, cols in columns_map.items():
        print(f"\n>> {fname}:")
        print("   " + ", ".join(cols))

if __name__ == "__main__":
    Path("scratch").mkdir(exist_ok=True)
    run_deep_audit()

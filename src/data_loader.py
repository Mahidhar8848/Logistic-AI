import os
from pathlib import Path
import pandas as pd
import json
from typing import Dict, Any, List, Optional
try:
    from src.config import DATASET_DIR, SUPPORTED_EXTENSIONS, get_dataset_path
except ImportError:
    from config import DATASET_DIR, SUPPORTED_EXTENSIONS, get_dataset_path

class DataLoader:
    """
    Automated Data Loader and Auditor for AI Logistics Datasets.
    Supports CSV, XLSX, XLS, JSON, Parquet files.
    """
    def __init__(self, dataset_dir: Optional[Path] = None):
        self.dataset_dir = Path(dataset_dir) if dataset_dir else DATASET_DIR

    def discover_datasets(self) -> List[Path]:
        """Scans the dataset directory for supported dataset files."""
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")
        
        files = []
        for file in self.dataset_dir.iterdir():
            if file.is_file() and file.suffix.lower() in SUPPORTED_EXTENSIONS:
                files.append(file)
        return sorted(files, key=lambda x: x.name)

    def load_dataset(self, filename: str, **kwargs) -> pd.DataFrame:
        """Loads a dataset into a pandas DataFrame based on file extension."""
        filepath = self.dataset_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"File {filename} not found in {self.dataset_dir}")
        
        ext = filepath.suffix.lower()
        if ext == ".csv":
            return pd.read_csv(filepath, **kwargs)
        elif ext in [".xlsx", ".xls"]:
            return pd.read_excel(filepath, **kwargs)
        elif ext == ".json":
            return pd.read_json(filepath, **kwargs)
        elif ext == ".parquet":
            return pd.read_parquet(filepath, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {ext}")

    def inspect_dataset(self, filename: str) -> Dict[str, Any]:
        """Provides detailed metadata and statistics for a single dataset file."""
        filepath = self.dataset_dir / filename
        file_size_bytes = filepath.stat().st_size
        file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
        ext = filepath.suffix.lower()

        df = self.load_dataset(filename)
        
        missing_per_col = df.isnull().sum().to_dict()
        total_missing = int(df.isnull().sum().sum())
        duplicate_rows = int(df.duplicated().sum())
        
        col_dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        sample_records = df.head(3).to_dict(orient="records")

        return {
            "filename": filename,
            "filepath": str(filepath),
            "file_type": ext.lstrip(".").upper(),
            "file_size_mb": file_size_mb,
            "rows": len(df),
            "columns_count": len(df.columns),
            "columns": list(df.columns),
            "dtypes": col_dtypes,
            "missing_values_total": total_missing,
            "missing_values_per_col": missing_per_col,
            "duplicate_rows": duplicate_rows,
            "sample_records": sample_records
        }

    def audit_all_datasets(self) -> List[Dict[str, Any]]:
        """Scans and audits all supported dataset files in the dataset directory."""
        files = self.discover_datasets()
        audit_results = []
        for file in files:
            print(f"Auditing dataset: {file.name}...")
            info = self.inspect_dataset(file.name)
            audit_results.append(info)
        return audit_results

if __name__ == "__main__":
    loader = DataLoader()
    datasets = loader.discover_datasets()
    print(f"Found {len(datasets)} dataset files in {loader.dataset_dir}:")
    for d in datasets:
        print(f" - {d.name} ({round(d.stat().st_size / 1024, 1)} KB)")

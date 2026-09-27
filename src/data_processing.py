"""
Data Processing Module for AI Logistics Pipeline.
Handles raw data cleaning, imputation, timestamp conversions, and dataset merging.
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from preprocess import run_preprocessing

if __name__ == "__main__":
    run_preprocessing()

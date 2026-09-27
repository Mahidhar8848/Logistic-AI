"""
Model Training Module for AI Logistics Pipeline.
Trains and evaluates ML models for empty-return risk classification.
"""
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent))
from train import train_and_evaluate

if __name__ == "__main__":
    train_and_evaluate()

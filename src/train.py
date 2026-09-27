import os
from pathlib import Path
import pandas as pd
import numpy as np
import json
import joblib
from typing import Dict, Any, List, Tuple

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)

try:
    from src.config import PROJECT_ROOT
except ImportError:
    from config import PROJECT_ROOT

def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    """Calculates comprehensive classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc = 0.5
        
    p_precision, p_recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(p_recall, p_precision)
    
    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(float(roc), 4),
        "pr_auc": round(float(pr_auc), 4),
        "confusion_matrix": cm
    }

def train_and_evaluate():
    processed_dir = PROJECT_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    print("Loading processed datasets...")
    train_df = pd.read_csv(processed_dir / "train_processed.csv")
    val_df = pd.read_csv(processed_dir / "val_processed.csv")
    test_df = pd.read_csv(processed_dir / "test_processed.csv")

    with open(processed_dir / "feature_metadata.json", "r") as f:
        meta = json.load(f)

    cat_cols = meta["categorical_features"]
    num_cols = meta["numerical_features"]
    target_col = meta["target_col"]

    X_train, y_train = train_df[cat_cols + num_cols], train_df[target_col].values
    X_val, y_val = val_df[cat_cols + num_cols], val_df[target_col].values
    X_test, y_test = test_df[cat_cols + num_cols], test_df[target_col].values

    print(f"Dataset shapes -> Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Build Preprocessing Pipeline
    num_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

    print("\nFitting preprocessing pipeline on training data...")
    preprocessor.fit(X_train)

    X_train_trans = preprocessor.transform(X_train)
    X_val_trans = preprocessor.transform(X_val)
    X_test_trans = preprocessor.transform(X_test)

    # Extract feature names after one-hot encoding
    ohe_categories = preprocessor.named_transformers_['cat'].named_steps['encoder'].get_feature_names_out(cat_cols)
    all_feature_names = list(num_cols) + list(ohe_categories)
    print(f"Total engineered features after encoding: {len(all_feature_names)}")

    # Candidate Models Dictionary
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=20, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=14, class_weight="balanced", n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_depth=6, class_weight="balanced", random_state=42)
    }

    results = {}
    best_model_name = None
    best_score = -1.0
    best_model_obj = None

    print("\nTraining and evaluating candidate models:")
    for name, model in models.items():
        print(f" - Training {name}...")
        model.fit(X_train_trans, y_train)

        # Validation set evaluation
        val_pred = model.predict(X_val_trans)
        val_prob = model.predict_proba(X_val_trans)[:, 1] if hasattr(model, "predict_proba") else val_pred
        val_metrics = evaluate_predictions(y_val, val_pred, val_prob)

        # Test set evaluation
        test_pred = model.predict(X_test_trans)
        test_prob = model.predict_proba(X_test_trans)[:, 1] if hasattr(model, "predict_proba") else test_pred
        test_metrics = evaluate_predictions(y_test, test_pred, test_prob)

        results[name] = {
            "validation": val_metrics,
            "test": test_metrics
        }

        print(f"   [{name}] Val ROC-AUC: {val_metrics['roc_auc']}, F1: {val_metrics['f1_score']} | Test ROC-AUC: {test_metrics['roc_auc']}, F1: {test_metrics['f1_score']}")

        # Track best model based on validation ROC-AUC
        if val_metrics["roc_auc"] > best_score:
            best_score = val_metrics["roc_auc"]
            best_model_name = name
            best_model_obj = model

    print(f"\nWinner Model: {best_model_name} (Val ROC-AUC: {best_score})")

    # Feature Importance / Coefficient Analysis for explainability
    feature_importances = {}
    if hasattr(best_model_obj, "feature_importances_"):
        imps = best_model_obj.feature_importances_
        for fn, imp in zip(all_feature_names, imps):
            feature_importances[fn] = round(float(imp), 6)
    elif hasattr(best_model_obj, "coef_"):
        coefs = best_model_obj.coef_[0]
        for fn, coef in zip(all_feature_names, coefs):
            feature_importances[fn] = round(float(coef), 6)

    # Sort feature importances
    sorted_importances = dict(sorted(feature_importances.items(), key=lambda item: abs(item[1]), reverse=True))

    # Save Serialization Artifacts
    print("\nSaving model artifacts to models/ directory...")
    joblib.dump(best_model_obj, models_dir / "model.pkl")
    joblib.dump(preprocessor, models_dir / "preprocessing.pkl")

    with open(models_dir / "feature_list.json", "w") as f:
        json.dump({
            "raw_categorical_features": cat_cols,
            "raw_numerical_features": num_cols,
            "all_transformed_features": all_feature_names
        }, f, indent=2)

    summary_metrics = {
        "best_model": best_model_name,
        "best_validation_roc_auc": best_score,
        "all_model_results": results,
        "top_15_feature_importances": dict(list(sorted_importances.items())[:15])
    }

    with open(models_dir / "metrics.json", "w") as f:
        json.dump(summary_metrics, f, indent=2)

    print(f"Artifacts successfully saved:")
    print(f" - Model: {models_dir / 'model.pkl'}")
    print(f" - Preprocessor: {models_dir / 'preprocessing.pkl'}")
    print(f" - Feature List: {models_dir / 'feature_list.json'}")
    print(f" - Metrics: {models_dir / 'metrics.json'}")

if __name__ == "__main__":
    train_and_evaluate()

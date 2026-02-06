#!/usr/bin/env python
"""Training Script for Genomic Variant Classifier
Author: Monzia Moodie
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import pandas as pd
from src.features.engineering import FeatureEngineeringPipeline
from src.models.ensemble import EnsembleClassifier
from sklearn.model_selection import train_test_split

def main():
    print("=" * 50)
    print("Genomic Variant Classifier Training")
    print("=" * 50)
    # Create synthetic data
    np.random.seed(42)
    n_samples = 1000
    data = pd.DataFrame({
        "gnomad_af": np.random.exponential(0.001, n_samples),
        "cadd_score": np.random.normal(20, 8, n_samples),
        "revel_score": np.random.beta(2, 5, n_samples),
    })
    prob = 0.3*(data["cadd_score"]>25) + 0.3*(data["revel_score"]>0.5)
    data["label"] = (prob + 0.2*np.random.random(n_samples) > 0.5).astype(int)
    # Feature engineering
    pipeline = FeatureEngineeringPipeline()
    data = pipeline.fit_transform(data)
    # Prepare data
    feature_cols = [c for c in data.columns if c != "label"]
    X = data[feature_cols]
    y = data["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    # Train
    ensemble = EnsembleClassifier()
    ensemble.fit(X_train, y_train)
    # Evaluate
    results = ensemble.evaluate(X_test, y_test)
    print(f"\nRESULTS:")
    print(f"AUROC: {results['auroc']:.4f}")
    print(f"F1: {results['f1']:.4f}")

if __name__ == "__main__":
    main()

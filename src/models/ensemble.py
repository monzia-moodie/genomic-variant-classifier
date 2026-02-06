"""
Ensemble Model Module for Genomic Variant Classification
Author: Monzia Moodie
"""
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
import joblib

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    name: str
    model_type: str
    weight: float = 1.0
    calibrate: bool = True
    params: Dict = field(default_factory=dict)

@dataclass
class EnsembleConfig:
    models: List[ModelConfig] = field(default_factory=list)
    aggregation: str = "weighted_average"
    calibration_method: str = "isotonic"
    random_state: int = 42

    @classmethod
    def default(cls):
        return cls(models=[
            ModelConfig("xgboost", "xgboost", 0.4,
                        params={"n_estimators": 200, "max_depth": 6}),
            ModelConfig("random_forest", "random_forest", 0.35,
                        params={"n_estimators": 200, "max_depth": 10}),
        ])

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """Ensemble classifier with confidence-weighted aggregation."""
    def __init__(self, config=None):
        self.config = config or EnsembleConfig.default()
        self.models = {}
        self.calibrated_models = {}
        self.weights = {}
        self.fitted = False
        self.feature_names_ = None

    def _create_model(self, model_config):
        params = model_config.params.copy()
        params["random_state"] = self.config.random_state
        if model_config.model_type == "xgboost" and HAS_XGBOOST:
            return xgb.XGBClassifier(**params)
        elif model_config.model_type == "random_forest":
            params["class_weight"] = "balanced"
            return RandomForestClassifier(**params)
        else:
            return GradientBoostingClassifier(**params)

    def fit(self, X, y, feature_names=None):
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        logger.info(f"Training ensemble with {len(self.config.models)} models")
        for model_config in self.config.models:
            logger.info(f"Training {model_config.name}...")
            try:
                model = self._create_model(model_config)
                model.fit(X, y)
                self.models[model_config.name] = model
                self.weights[model_config.name] = model_config.weight
                if model_config.calibrate:
                    calibrated = CalibratedClassifierCV(model,
                                                      method=self.config.calibration_method, cv=3)
                    calibrated.fit(X, y)
                    self.calibrated_models[model_config.name] = calibrated
            except Exception as e:
                logger.error(f"Failed to train {model_config.name}: {e}")
        total = sum(self.weights.values())
        self.weights = {k: v/total for k, v in self.weights.items()}
        self.fitted = True
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X, return_individual=False):
        if not self.fitted:
            raise ValueError("Not fitted")
        if isinstance(X, pd.DataFrame):
            X = X.values
        individual = {}
        for name, model in self.models.items():
            if name in self.calibrated_models:
                proba = self.calibrated_models[name].predict_proba(X)
            else:
                proba = model.predict_proba(X)
            individual[name] = proba
        ensemble_proba = np.zeros((X.shape[0], 2))
        for name, proba in individual.items():
            ensemble_proba += self.weights[name] * proba
        if return_individual:
            return ensemble_proba, individual
        return ensemble_proba

    def evaluate(self, X, y, threshold=0.5):
        if isinstance(y, pd.Series):
            y = y.values
        proba, individual = self.predict_proba(X, return_individual=True)
        pred = (proba[:, 1] >= threshold).astype(int)
        results = {
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
            "auroc": roc_auc_score(y, proba[:, 1]),
            "auprc": average_precision_score(y, proba[:, 1]),
            "individual_models": {},
        }
        for name, model_proba in individual.items():
            model_pred = (model_proba[:, 1] >= threshold).astype(int)
            results["individual_models"][name] = {
                "auroc": roc_auc_score(y, model_proba[:, 1]),
                "f1": f1_score(y, model_pred, zero_division=0),
            }
        return results

    def get_feature_importance(self):
        importance = {}
        for name, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                importance[name] = model.feature_importances_
        if not importance:
            return pd.DataFrame()
        df = pd.DataFrame(importance)
        if self.feature_names_:
            df.index = self.feature_names_
        df["ensemble"] = sum(df[n] * self.weights.get(n, 0) for n in df.columns)
        return df.sort_values("ensemble", ascending=False)

    def find_optimal_threshold(self, X, y, metric="f1"):
        proba = self.predict_proba(X)[:, 1]
        if isinstance(y, pd.Series):
            y = y.values
        best_threshold, best_score = 0.5, 0.0
        for threshold in np.arange(0.1, 0.9, 0.05):
            pred = (proba >= threshold).astype(int)
            score = f1_score(y, pred, zero_division=0)
            if score > best_score:
                best_score = score
                best_threshold = threshold
        pred = (proba >= best_threshold).astype(int)
        return best_threshold, {
            "threshold": best_threshold,
            "f1": f1_score(y, pred, zero_division=0),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
        }

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for name, model in self.models.items():
            joblib.dump(model, path / f"{name}.joblib")
        with open(path / "config.json", "w") as f:
            json.dump({"weights": self.weights}, f)

if __name__ == "__main__":
    np.random.seed(42)
    X = np.random.randn(500, 10)
    y = (X[:, 0] + X[:, 1] + np.random.randn(500) * 0.5 > 0).astype(int)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    ensemble = EnsembleClassifier()
    ensemble.fit(X_train, y_train)
    results = ensemble.evaluate(X_test, y_test)
    print(f"AUROC: {results['auroc']:.4f}, F1: {results['f1']:.4f}")

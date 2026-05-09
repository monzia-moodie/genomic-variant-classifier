"""
Feature Engineering Module - ACMG-aligned feature extraction
Author: Monzia Moodie
"""
import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

@dataclass
class ACMGEvidence:
    """Container for ACMG evidence criteria."""
    pvs1: bool = False # Null variant in LOF gene
    pm2: bool = False # Absent from population databases
    pp3: bool = False # Computational evidence deleterious
    ba1: bool = False # AF > 5%
    bp4: bool = False # Computational evidence benign

    def get_pathogenic_score(self):
        score = 0.0
        if self.pvs1: score += 8.0
        if self.pm2: score += 2.0
        if self.pp3: score += 1.0
        return score

    def get_benign_score(self):
        score = 0.0
        if self.ba1: score += 8.0
        if self.bp4: score += 1.0
        return score

class PopulationFrequencyFeatures:
    """Extract features from population allele frequencies."""
    BA1_THRESHOLD = 0.05

    def extract(self, df):
        result = df.copy()
        if "gnomad_af" in df.columns:
            af = df["gnomad_af"].fillna(0)
            result["af_log10"] = np.log10(af + 1e-10)
            result["af_ba1"] = (af > self.BA1_THRESHOLD).astype(int)
            result["af_pm2"] = (af < 1e-6).astype(int)
        return result

class ComputationalPredictorFeatures:
    """Extract features from computational predictors."""
    def extract(self, df):
        result = df.copy()
        if "cadd_score" in df.columns:
            cadd = df["cadd_score"].fillna(df["cadd_score"].median())
            result["cadd_deleterious"] = (cadd >= 25).astype(int)
            result["cadd_benign"] = (cadd < 15).astype(int)
        if "revel_score" in df.columns:
            revel = df["revel_score"].fillna(0.5)
            result["revel_deleterious"] = (revel >= 0.7).astype(int)
            result["revel_benign"] = (revel < 0.15).astype(int)
        return result

class FeatureEngineeringPipeline:
    """Complete feature engineering pipeline."""
    def __init__(self, disease_type="rare"):
        self.disease_type = disease_type
        self.pop_features = PopulationFrequencyFeatures()
        self.pred_features = ComputationalPredictorFeatures()
        self.fitted = False

    def fit_transform(self, df):
        result = df.copy()
        result = self.pop_features.extract(result)
        result = self.pred_features.extract(result)
        self.fitted = True
        logger.info(f"Extracted features. Shape: {result.shape}")
        return result

    def transform(self, df):
        if not self.fitted:
            raise ValueError("Pipeline not fitted")
        result = self.pop_features.extract(df.copy())
        result = self.pred_features.extract(result)
        return result

if __name__ == "__main__":
    data = pd.DataFrame({
        "gnomad_af": np.random.exponential(0.001, 100),
        "cadd_score": np.random.normal(20, 8, 100),
    })
    pipeline = FeatureEngineeringPipeline()
    result = pipeline.fit_transform(data)
    print(f"Original: {len(data.columns)}, Engineered: {len(result.columns)}")

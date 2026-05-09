"""
Data Pipeline Module for Genomic Variant Classification
Author: Monzia Moodie
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.utils.helpers import load_config


logger = logging.getLogger(__name__)

@dataclass
class VariantRecord:
    """Data class representing a single genetic variant."""
    chrom: str
    pos: int
    ref: str
    alt: str
    gene: Optional[str] = None
    gnomad_af: Optional[float] = None

    @property
    def variant_id(self) -> str:
        return f"{self.chrom}-{self.pos}-{self.ref}-{self.alt}"

class ClinVarLoader:
    """Loader for ClinVar variant data."""
    PATHOGENIC_LABELS = ["Pathogenic", "Likely pathogenic"]
    BENIGN_LABELS = ["Benign", "Likely benign"]

    def __init__(self, filepath=None):
        self.filepath = filepath
        self.data = None

    def load(self, filepath=None):
        filepath = filepath or self.filepath
        logger.info(f"Loading ClinVar data from {filepath}")
        df = pd.read_csv(filepath, sep="\t", low_memory=False)
        if "Assembly" in df.columns:
            df = df[df["Assembly"] == "GRCh38"].copy()
        df["label"] = df["ClinicalSignificance"].apply(self._map_significance)
        self.data = df
        return df

    def _map_significance(self, sig):
        if pd.isna(sig):
            return None
        if sig in self.PATHOGENIC_LABELS:
            return 1
        elif sig in self.BENIGN_LABELS:
            return 0
        return None

    def filter_high_quality(self, min_quality=2):
        if self.data is None:
            raise ValueError("Data not loaded")
        return self.data[self.data["label"].notna()]

class GnomADLoader:
    """Loader for gnomAD population frequency data."""
    def __init__(self, filepath=None):
        self.filepath = filepath
        self.data = None

    def load(self, filepath=None):
        filepath = filepath or self.filepath
        logger.info(f"Loading gnomAD data from {filepath}")
        df = pd.read_csv(filepath, sep="\t", low_memory=False)
        self.data = df
        return df

class VariantDataPipeline:
    """Main data pipeline for integrating variant data."""
    def __init__(self, config_path=None):
        self.config = load_config() if config_path is None else yaml.safe_load(open(config_path))
        self.clinvar_loader = ClinVarLoader()
        self.gnomad_loader = GnomADLoader()
        self.merged_data = None
        self.scaler = StandardScaler()

    def load_all_data(self, clinvar_path=None, gnomad_path=None):
        if clinvar_path:
            self.clinvar_loader.load(clinvar_path)
        self.merged_data = self.clinvar_loader.filter_high_quality()
        return self.merged_data

    def preprocess(self, df=None):
        df = df.copy() if df is not None else self.merged_data.copy()
        if "gnomad_af" in df.columns:
            df["gnomad_af"] = df["gnomad_af"].fillna(0.0)
            df["gnomad_af_log"] = np.log10(df["gnomad_af"] + 1e-10)
        return df

    def prepare_data(self, test_size=0.2, random_state=42):
        if self.merged_data is None:
            raise ValueError("Data not loaded")
        df = self.preprocess()
        df = df[df["label"].notna()].copy()
        feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in feature_cols if c != "label"]
        X = df[feature_cols].fillna(0)
        y = df["label"].astype(int)
        return train_test_split(X, y, test_size=test_size,
                                random_state=random_state, stratify=y)

if __name__ == "__main__":
    pipeline = VariantDataPipeline()
    sample = pd.DataFrame({
        "gnomad_af": np.random.exponential(0.001, 100),
        "cadd_score": np.random.normal(20, 8, 100),
        "label": np.random.choice([0, 1], 100),
    })
    pipeline.merged_data = sample
    X_train, X_test, y_train, y_test = pipeline.prepare_data()
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")
    

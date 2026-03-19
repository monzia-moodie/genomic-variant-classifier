"""
Ensemble Model Framework for Genomic Variant Classification
============================================================
Implements 8 base classifiers + 1 stacking meta-learner.

Base classifiers:
  1. Random Forest         (sklearn)
  2. XGBoost               (xgboost)
  3. LightGBM              (lightgbm)
  4. SVM (RBF kernel)      (sklearn)
  5. Logistic Regression   (sklearn)
  6. Gradient Boosting     (sklearn)
  7. 1D-CNN                (TensorFlow/Keras)  — sequence-based
  8. Feedforward NN        (TensorFlow/Keras)  — tabular features

Meta-learner:
  Logistic Regression stacker trained on OOF predictions

CHANGES FROM PHASE 1:
  - Consolidated src/models/ensemble.py + src/models/variant_ensemble.py
    into this single file (Issue A).
  - Removed unused `k` parameter from encode_sequence; docstring fixed (Bug 7).
  - codon_position removed from TABULAR_FEATURES — it was always 0 (Issue P).
  - base_estimators cleared after fitting to avoid double-memory (Issue H).
  - from __future__ import annotations added (Issue N).
  - Module-level logging.basicConfig removed (Issue L).
  - VariantEnsemble.fit() / evaluate() now robust when CNN/NN are excluded.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import xgboost as xgb
import lightgbm as lgb

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Feature definitions
# ---------------------------------------------------------------------------
TABULAR_FEATURES = [
    # Allele / variant properties
    "allele_freq",              # population AF from gnomAD
    "ref_len",                  # length of reference allele
    "alt_len",                  # length of alternate allele
    "is_snv",                   # 1 if single nucleotide variant
    "is_indel",                 # 1 if insertion or deletion
    # Functional scores
    "cadd_phred",               # CADD PHRED (higher = more deleterious)
    "sift_score",               # SIFT (lower = more deleterious)
    "polyphen2_score",          # PolyPhen-2 HDIV score
    "revel_score",              # REVEL ensemble score
    "phylop_score",             # phyloP conservation score
    # Coding context
    "in_coding_region",         # 1 if in coding region
    "in_splice_site",           # 1 if within 2 bp of exon boundary
    # NOTE: codon_position removed — was always 0 (placeholder never filled).
    #       Will be added in Phase 2 after VEP annotation. (Issue P)
    "is_missense",              # 1 if missense variant
    "is_nonsense",              # 1 if stop-gain / nonsense
    # Gene-level
    "gene_constraint_oe",       # gnomAD pLoF observed/expected ratio
    "num_pathogenic_in_gene",   # ClinVar count of pathogenic variants in gene
    # Protein features (from UniProt)
    "in_active_site",           # 1 if overlaps active site annotation
    "in_domain",                # 1 if overlaps annotated protein domain
]

# Features planned for Phase 2 (require VEP annotation or external tools)
PHASE_2_FEATURES = [
    "codon_position",           # 1, 2, or 3 — requires VEP
    "splice_ai_score",          # SpliceAI delta score
    "alphamissense_score",       # AlphaMissense pathogenicity score
]

SEQUENCE_FEATURES = ["fasta_seq"]   # 101 bp context window, one-hot encoded


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass
class EnsembleConfig:
    n_folds:        int   = 5
    random_state:   int   = 42
    calibrate:      bool  = True         # Platt scaling on base models
    class_weight:   str   = "balanced"   # handles class imbalance
    n_jobs:         int   = -1
    model_dir:      Path  = Path("models/ensemble")

    def __post_init__(self) -> None:
        self.model_dir = Path(self.model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive tabular features from a canonical variant DataFrame.

    Input columns used:
        allele_freq, ref, alt, consequence, cadd_phred, sift_score,
        polyphen2_score, revel_score, phylop_score, gene_constraint_oe,
        num_pathogenic_in_gene, in_active_site, in_domain

    All missing columns are filled with population-median defaults so the
    function is safe to call on partially-populated DataFrames.
    """
    feats = pd.DataFrame(index=df.index)

    # Allele frequency
    feats["allele_freq"] = df.get("allele_freq", 0.0).fillna(0.0)

    # Allele length
    ref_col = df.get("ref", pd.Series(["A"] * len(df), index=df.index)).fillna("A")
    alt_col = df.get("alt", pd.Series(["A"] * len(df), index=df.index)).fillna("A")
    feats["ref_len"] = ref_col.str.len().fillna(1).astype(int)
    feats["alt_len"] = alt_col.str.len().fillna(1).astype(int)
    feats["is_snv"]   = ((feats["ref_len"] == 1) & (feats["alt_len"] == 1)).astype(int)
    feats["is_indel"] = (feats["is_snv"] == 0).astype(int)

    # Consequence-based booleans
    consequence = df.get("consequence", pd.Series([""] * len(df), index=df.index)).fillna("")
    feats["is_missense"]     = consequence.str.contains("missense",                case=False).astype(int)
    feats["is_nonsense"]     = consequence.str.contains("stop_gained|nonsense",    case=False).astype(int)
    feats["in_splice_site"]  = consequence.str.contains("splice",                  case=False).astype(int)
    feats["in_coding_region"] = consequence.str.contains(
        "missense|synonymous|stop|frameshift|inframe", case=False
    ).astype(int)

    # Precomputed scores — fill with population median if missing
    score_defaults = {
        "cadd_phred":     15.0,
        "sift_score":      0.5,
        "polyphen2_score": 0.5,
        "revel_score":     0.5,
        "phylop_score":    0.0,
    }
    for col, default in score_defaults.items():
        feats[col] = df.get(col, pd.Series([default] * len(df), index=df.index)).fillna(default).astype(float)

    # Gene constraint
    feats["gene_constraint_oe"]    = df.get("gene_constraint_oe",    pd.Series([1.0] * len(df), index=df.index)).fillna(1.0)
    feats["num_pathogenic_in_gene"] = df.get("num_pathogenic_in_gene", pd.Series([0]   * len(df), index=df.index)).fillna(0)

    # Protein annotations
    feats["in_active_site"] = df.get("in_active_site", pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)
    feats["in_domain"]      = df.get("in_domain",      pd.Series([0] * len(df), index=df.index)).fillna(0).astype(int)

    # Validate
    feats = feats[TABULAR_FEATURES]
    feats = feats[TABULAR_FEATURES]
    assert list(feats.columns) == TABULAR_FEATURES, (
        f"Feature column mismatch. Expected {TABULAR_FEATURES}, got {list(feats.columns)}"
    )
    return feats


def encode_sequence(seq: str, window: int = 101) -> np.ndarray:
    """
    One-hot encode a nucleotide sequence for CNN input.

    Args:
        seq:    Raw nucleotide string (any length; will be padded/trimmed to `window`).
        window: Context window size in base pairs. Default 101 bp.

    Returns:
        np.ndarray of shape (window, 4), dtype float32.
        Axis 1 encodes [A, C, G, T]. Ambiguous bases (N) are encoded as all zeros.

    CHANGE: Removed unused `k` k-mer parameter. The function only performs
    one-hot encoding. k-mer frequency encoding will be added in Phase 2
    as a separate `kmer_encode_sequence()` function. (Bug 7)
    """
    BASES = "ACGT"
    base_map = {b: i for i, b in enumerate(BASES)}

    # Normalise, pad/trim to window
    seq = seq.upper()[:window].ljust(window, "A")

    one_hot = np.zeros((window, len(BASES)), dtype=np.float32)
    for i, nuc in enumerate(seq):
        if nuc in base_map:
            one_hot[i, base_map[nuc]] = 1.0
    return one_hot


# ---------------------------------------------------------------------------
# Sklearn-compatible 1D-CNN wrapper (sequence input)
# ---------------------------------------------------------------------------
class CNN1DClassifier(BaseEstimator, ClassifierMixin):
    """
    1D Convolutional neural network for variant sequence classification.
    Wraps Keras for sklearn cross_val_predict compatibility.
    Input: pd.Series of nucleotide strings (length ≥ window bp).
    """

    def __init__(
        self,
        window: int = 101,
        filters: int = 64,
        kernel_size: int = 7,
        dropout: float = 0.3,
        epochs: int = 30,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        self.window        = window
        self.filters       = filters
        self.kernel_size   = kernel_size
        self.dropout       = dropout
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.random_state  = random_state
        self.model_: Optional[object] = None
        self.classes_      = np.array([0, 1])

    def _build_model(self):
        import tensorflow as tf
        from tensorflow.keras import layers, models as km

        tf.random.set_seed(self.random_state)
        inp = layers.Input(shape=(self.window, 4))
        x   = layers.Conv1D(self.filters,     self.kernel_size, activation="relu", padding="same")(inp)
        x   = layers.MaxPooling1D(2)(x)
        x   = layers.Conv1D(self.filters * 2, self.kernel_size, activation="relu", padding="same")(x)
        x   = layers.GlobalMaxPooling1D()(x)
        x   = layers.Dense(128, activation="relu")(x)
        x   = layers.Dropout(self.dropout)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        model = km.Model(inp, out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="binary_crossentropy",
            metrics=["AUC"],
        )
        return model

    def _encode_X(self, X) -> np.ndarray:
        if isinstance(X, pd.Series):
            seqs = X.fillna("A" * self.window)
        elif isinstance(X, pd.DataFrame) and "fasta_seq" in X.columns:
            seqs = X["fasta_seq"].fillna("A" * self.window)
        else:
            seqs = pd.Series(X).fillna("A" * self.window)
        return np.stack([encode_sequence(s, window=self.window) for s in seqs])

    def fit(self, X, y):
        import tensorflow as tf
        self.model_ = self._build_model()
        self.model_.fit(
            self._encode_X(X), y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
            ],
            verbose=0,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        proba = self.model_.predict(self._encode_X(X), verbose=0).flatten()
        return np.column_stack([1 - proba, proba])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# ---------------------------------------------------------------------------
# Sklearn-compatible feedforward NN (tabular input)
# ---------------------------------------------------------------------------
class TabularNNClassifier(BaseEstimator, ClassifierMixin):
    """Feedforward neural network for tabular variant features."""

    def __init__(
        self,
        hidden_dims: tuple = (256, 128, 64),
        dropout: float = 0.3,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        random_state: int = 42,
    ) -> None:
        self.hidden_dims   = hidden_dims
        self.dropout       = dropout
        self.epochs        = epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.random_state  = random_state
        self.model_: Optional[object]  = None
        self.scaler_       = StandardScaler()
        self.classes_      = np.array([0, 1])

    def _build_model(self, input_dim: int):
        import tensorflow as tf
        from tensorflow.keras import layers, models as km, regularizers

        tf.random.set_seed(self.random_state)
        inp = layers.Input(shape=(input_dim,))
        x   = inp
        for dim in self.hidden_dims:
            x = layers.Dense(dim, activation="relu",
                             kernel_regularizer=regularizers.l2(1e-4))(x)
            x = layers.BatchNormalization()(x)
            x = layers.Dropout(self.dropout)(x)
        out = layers.Dense(1, activation="sigmoid")(x)
        model = km.Model(inp, out)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(self.learning_rate),
            loss="binary_crossentropy",
            metrics=["AUC"],
        )
        return model

    def fit(self, X, y):
        import tensorflow as tf
        X_scaled = self.scaler_.fit_transform(X)
        self.model_ = self._build_model(X_scaled.shape[1])
        self.model_.fit(
            X_scaled, y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)
            ],
            verbose=0,
        )
        return self

    def predict_proba(self, X) -> np.ndarray:
        proba = self.model_.predict(self.scaler_.transform(X), verbose=0).flatten()
        return np.column_stack([1 - proba, proba])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)


# ---------------------------------------------------------------------------
# Ensemble orchestrator
# ---------------------------------------------------------------------------
class VariantEnsemble:
    """
    Orchestrates training and evaluation of all base classifiers
    plus a stacking meta-learner (Logistic Regression on OOF predictions).

    Usage:
        ensemble = VariantEnsemble()
        ensemble.fit(X_tab_train, X_seq_train, y_train)
        results  = ensemble.evaluate(X_tab_test, X_seq_test, y_test)
        ensemble.save(Path("models/v1"))
    """

    def __init__(self, config: Optional[EnsembleConfig] = None) -> None:
        self.config = config or EnsembleConfig()
        self._build_estimators()

    def _build_estimators(self) -> None:
        cfg = self.config
        self.base_estimators: dict = {
            "random_forest": RandomForestClassifier(
                n_estimators=500, max_features="sqrt",
                class_weight=cfg.class_weight,
                n_jobs=cfg.n_jobs, random_state=cfg.random_state,
            ),
            "xgboost": xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                scale_pos_weight=10,
                eval_metric="auc",
                n_jobs=cfg.n_jobs, random_state=cfg.random_state, verbosity=0,
            ),
            "lightgbm": lgb.LGBMClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                class_weight=cfg.class_weight,
                n_jobs=cfg.n_jobs, random_state=cfg.random_state, verbose=-1,
            ),
            "svm": CalibratedClassifierCV(
                SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
                cv=3,
            ),
            "logistic_regression": LogisticRegression(
                C=0.1, max_iter=1000, class_weight=cfg.class_weight,
                n_jobs=cfg.n_jobs, random_state=cfg.random_state,
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, random_state=cfg.random_state,
            ),
            "tabular_nn": TabularNNClassifier(random_state=cfg.random_state),
            "cnn_1d":     CNN1DClassifier(random_state=cfg.random_state),
        }
        self.meta_learner = LogisticRegression(
            C=0.1, max_iter=1000, random_state=cfg.random_state
        )
        # Populated during fit(); kept separate from base_estimators
        self.trained_models_: dict = {}

    def fit(
        self,
        X_tab: pd.DataFrame,
        X_seq: pd.Series,
        y: pd.Series,
    ) -> "VariantEnsemble":
        """
        Two-phase training:
          Phase 1 — cross-val all base models → OOF predictions
          Phase 2 — train meta-learner on OOF predictions

        CHANGE: base_estimators is cleared after fitting; trained_models_
        is the single source of truth for fitted models. (Issue H)
        """
        y_arr = np.asarray(y)
        logger.info(
            "Training ensemble: %d samples, %d pathogenic.",
            len(y_arr), int(y_arr.sum()),
        )
        cv = StratifiedKFold(
            n_splits=self.config.n_folds, shuffle=True,
            random_state=self.config.random_state,
        )

        oof_preds = np.zeros((len(y_arr), len(self.base_estimators)))

        for model_idx, (name, model) in enumerate(self.base_estimators.items()):
            logger.info("  Training %s ...", name)
            X_input = X_seq if name == "cnn_1d" else X_tab.values

            try:
                oof = cross_val_predict(
                    model, X_input, y_arr,
                    cv=cv, method="predict_proba", n_jobs=1,
                )[:, 1]
            except Exception as exc:
                logger.error("  %s OOF failed: %s — skipping.", name, exc)
                oof_preds[:, model_idx] = 0.5
                continue

            oof_preds[:, model_idx] = oof
            model.fit(X_input, y_arr)
            self.trained_models_[name] = model
            logger.info("  %s OOF AUROC: %.4f", name, roc_auc_score(y_arr, oof))

        # Remove columns for skipped models
        valid_cols = [
            i for i, name in enumerate(self.base_estimators)
            if name in self.trained_models_
        ]
        oof_preds = oof_preds[:, valid_cols]

        logger.info("Training meta-learner on %d base-model OOF columns ...", len(valid_cols))
        self.meta_learner.fit(oof_preds, y_arr)
        self.feature_names_ = list(self.trained_models_.keys())

        # Free memory from unfitted duplicates (Issue H)
        self.base_estimators.clear()
        return self

    def predict_proba(
        self,
        X_tab: pd.DataFrame,
        X_seq: pd.Series,
    ) -> np.ndarray:
        if not self.trained_models_:
            raise RuntimeError("Call fit() before predict_proba().")
        base_preds = np.zeros((len(X_tab), len(self.trained_models_)))
        for i, (name, model) in enumerate(self.trained_models_.items()):
            X_input = X_seq if name == "cnn_1d" else X_tab.values
            base_preds[:, i] = model.predict_proba(X_input)[:, 1]
        return self.meta_learner.predict_proba(base_preds)

    def predict(self, X_tab: pd.DataFrame, X_seq: pd.Series) -> np.ndarray:
        return (self.predict_proba(X_tab, X_seq)[:, 1] > 0.5).astype(int)

    def evaluate(
        self,
        X_tab: pd.DataFrame,
        X_seq: pd.Series,
        y: pd.Series,
    ) -> pd.DataFrame:
        """
        Full per-model + ensemble evaluation.
        Returns a DataFrame sorted by AUROC (descending).
        """
        y_arr = np.asarray(y)
        results: dict[str, dict] = {}

        for name, model in self.trained_models_.items():
            X_input = X_seq if name == "cnn_1d" else X_tab.values
            proba   = model.predict_proba(X_input)[:, 1]
            preds   = (proba > 0.5).astype(int)
            results[name] = {
                "auroc":       roc_auc_score(y_arr, proba),
                "auprc":       average_precision_score(y_arr, proba),
                "f1_macro":    f1_score(y_arr, preds, average="macro",    zero_division=0),
                "f1_weighted": f1_score(y_arr, preds, average="weighted", zero_division=0),
                "mcc":         matthews_corrcoef(y_arr, preds),
                "brier":       brier_score_loss(y_arr, proba),
            }

        ens_proba = self.predict_proba(X_tab, X_seq)[:, 1]
        ens_preds = (ens_proba > 0.5).astype(int)
        results["ENSEMBLE_STACKER"] = {
            "auroc":       roc_auc_score(y_arr, ens_proba),
            "auprc":       average_precision_score(y_arr, ens_proba),
            "f1_macro":    f1_score(y_arr, ens_preds, average="macro",    zero_division=0),
            "f1_weighted": f1_score(y_arr, ens_preds, average="weighted", zero_division=0),
            "mcc":         matthews_corrcoef(y_arr, ens_preds),
            "brier":       brier_score_loss(y_arr, ens_proba),
        }

        df = pd.DataFrame(results).T.round(4)
        df = df.sort_values("auroc", ascending=False)
        logger.info("\n%s", df.to_string())
        return df

    def save(self, path: Optional[Path] = None) -> None:
        import joblib
        path = Path(path or self.config.model_dir / "ensemble.joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info("Ensemble saved to %s", path)

    @classmethod
    def load(cls, path: Path) -> "VariantEnsemble":
        import joblib
        return joblib.load(path)

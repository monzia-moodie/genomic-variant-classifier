"""
agents/training_lifecycle_agent.py
===================================
Training Lifecycle Agent — manages the continual learning loop for both
the ResNet-50 CNN branch and the XGBoost / LightGBM ensemble.

Decision logic
--------------
The agent reads the current drift_score from SharedState (written by
DataFreshnessAgent after each corpus update) and picks a strategy:

    drift < EWC_DRIFT_LOW                       → no update, log and exit
    EWC_DRIFT_LOW  ≤ drift < EWC_DRIFT_HIGH     → EWC fine-tune (ResNet-50)
                                                   + replay-buffer update
                                                     (ensemble)
    drift ≥ EWC_DRIFT_HIGH                      → queue full retrain for
                                                   human review
    reclassification_rate ≥ threshold           → force update regardless
                                                   of drift score

Checkpointing
-------------
Each successful update produces a versioned checkpoint directory:
    <CHECKPOINT_DIR>/<timestamp>_v<N>/
        resnet50/
            model.pt          — state_dict
            fisher.pt         — Fisher diagonal
            training_history.json
        ensemble/
            xgb_model.ubj     — XGBoost booster
            lgb_model.txt     — LightGBM booster
        metadata.json         — version, val_acc, drift_score, …

After a successful run the agent updates SharedState with:
    model_checkpoint_ref   → path to the new checkpoint directory
    model_last_trained     → ISO timestamp
    ewc_fisher_ref         → path to fisher.pt
    ewc_lambda             → current lambda value
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_AL = Path(__file__).resolve().parent.parent
for _p in (str(_AL), str(_AL / "agents")):
    if _p not in sys.path: sys.path.insert(0, _p)

from base_agent import AgentResult, BaseAgent
from config import (
    CHECKPOINT_DIR,
    ENSEMBLE_BOOST_ROUNDS,
    ENSEMBLE_SUBDIR,
    EWC_BATCH_SIZE,
    EWC_DRIFT_HIGH,
    EWC_DRIFT_LOW,
    EWC_EPOCHS,
    EWC_FISHER_SAMPLES,
    EWC_LAMBDA,
    EWC_LR,
    GCS_CHECKPOINT_PREFIX,
    GDRIVE_CHECKPOINT_DIR,
    PROCESSED_DATA_DIR,
    RECLASSIFICATION_RATE_THRESHOLD,
    REPLAY_BUFFER_PARQUET,
    REPLAY_BUFFER_SIZE,
    REPLAY_SAMPLE_FRAC,
    REQUIRE_HUMAN_APPROVAL,
    RESNET_NUM_CLASSES,
    RESNET_SUBDIR,
    TRAIN_PARQUET,
    VAL_PARQUET,
)
from shared_state import SharedState

log = logging.getLogger("TrainingLifecycleAgent")


# ---------------------------------------------------------------------------
# Strategy constants
# ---------------------------------------------------------------------------

STRATEGY_SKIP         = "skip"
STRATEGY_EWC_UPDATE   = "ewc_update"
STRATEGY_FULL_RETRAIN = "full_retrain_queued"


class TrainingLifecycleAgent(BaseAgent):
    """
    Manages EWC continual learning for the ResNet-50 branch and
    memory-replay continual learning for the XGBoost / LightGBM ensemble.
    """

    def __init__(self, state: SharedState, dry_run: bool = False):
        super().__init__(state, dry_run)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    @property
    def name(self) -> str:
        return "TrainingLifecycleAgent"

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> AgentResult:
        results: dict[str, Any] = {
            "strategy":         None,
            "drift_score":      None,
            "resnet_updated":   False,
            "ensemble_updated": False,
            "checkpoint_path":  None,
            "resnet_val_acc":   None,
            "ensemble_metrics": None,
            "errors":           [],
        }

        # ---- 1. Assess current drift ------------------------------------
        drift_score   = self.state.get("drift_score") or 0.0
        reclass_rate  = self.state.get("reclassification_rate") or 0.0
        results["drift_score"] = drift_score

        strategy = self._decide_strategy(drift_score, reclass_rate)
        results["strategy"] = strategy
        self.log.info(
            "drift=%.4f  reclass_rate=%.4f  → strategy: %s",
            drift_score, reclass_rate, strategy,
        )

        if strategy == STRATEGY_SKIP:
            return AgentResult(success=True, action=STRATEGY_SKIP, details=results)

        if strategy == STRATEGY_FULL_RETRAIN:
            self._queue_full_retrain(drift_score, reclass_rate)
            results["queued_for_review"] = True
            return AgentResult(
                success=True, action=STRATEGY_FULL_RETRAIN, details=results
            )

        # ---- 2. Human gate ----------------------------------------------
        if REQUIRE_HUMAN_APPROVAL:
            approved = self.require_human_approval(
                f"Run EWC update? drift={drift_score:.4f}, "
                f"reclass_rate={reclass_rate:.4f}"
            )
            if not approved:
                self.state.add_pending_review({
                    "reason": "EWC update blocked pending human approval",
                    "drift":  drift_score,
                    "agent":  self.name,
                })
                return AgentResult(
                    success=True, action="blocked_pending_approval", details=results
                )

        # ---- 3. Versioned checkpoint directory --------------------------
        checkpoint_dir = self._new_checkpoint_dir()
        results["checkpoint_path"] = str(checkpoint_dir)

        # ---- 4. ResNet-50 EWC update ------------------------------------
        try:
            resnet_result = self._run_ewc_update_resnet(checkpoint_dir)
            results["resnet_updated"] = resnet_result["updated"]
            results["resnet_val_acc"] = resnet_result.get("val_acc")
            if resnet_result.get("error"):
                results["errors"].append(f"ResNet EWC: {resnet_result['error']}")
        except Exception as exc:
            msg = f"ResNet EWC update failed: {exc}"
            self.log.error("%s\n%s", msg, traceback.format_exc())
            results["errors"].append(msg)

        # ---- 5. Ensemble replay-buffer update ---------------------------
        try:
            ens_result = self._run_ensemble_replay_update(checkpoint_dir)
            results["ensemble_updated"] = ens_result["updated"]
            results["ensemble_metrics"] = ens_result.get("metrics")
            if ens_result.get("error"):
                results["errors"].append(f"Ensemble: {ens_result['error']}")
        except Exception as exc:
            msg = f"Ensemble update failed: {exc}"
            self.log.error("%s\n%s", msg, traceback.format_exc())
            results["errors"].append(msg)

        # ---- 6. Persist and update state --------------------------------
        anything_updated = results["resnet_updated"] or results["ensemble_updated"]
        if anything_updated:
            self._write_checkpoint_metadata(checkpoint_dir, results)
            self._push_checkpoint(checkpoint_dir)
            self._update_state(checkpoint_dir, results)
        else:
            shutil.rmtree(checkpoint_dir, ignore_errors=True)

        success = anything_updated and len(results["errors"]) == 0
        return AgentResult(
            success=success,
            action=strategy,
            details=results,
            errors=results["errors"],
        )

    # ------------------------------------------------------------------
    # Strategy decision
    # ------------------------------------------------------------------

    def _decide_strategy(self, drift: float, reclass_rate: float) -> str:
        if reclass_rate >= RECLASSIFICATION_RATE_THRESHOLD:
            self.log.info(
                "Reclassification rate %.4f >= threshold %.4f → force EWC update.",
                reclass_rate, RECLASSIFICATION_RATE_THRESHOLD,
            )
            return STRATEGY_EWC_UPDATE
        if drift < EWC_DRIFT_LOW:
            return STRATEGY_SKIP
        if drift < EWC_DRIFT_HIGH:
            return STRATEGY_EWC_UPDATE
        return STRATEGY_FULL_RETRAIN

    def _queue_full_retrain(self, drift: float, reclass_rate: float) -> None:
        self.log.warning(
            "Drift %.4f >= EWC_DRIFT_HIGH %.4f — queuing full retrain.",
            drift, EWC_DRIFT_HIGH,
        )
        self.state.add_pending_review({
            "reason":       "Drift exceeds EWC safe range — full retrain required",
            "drift_score":  drift,
            "reclass_rate": reclass_rate,
            "agent":        self.name,
            "action_required": (
                "EWC cannot safely bridge this level of distribution shift. "
                "Schedule a full retrain from the updated corpus. "
                "Review reclassified variants before proceeding."
            ),
        })

    # ------------------------------------------------------------------
    # ResNet-50 EWC update
    # ------------------------------------------------------------------

    def _run_ewc_update_resnet(self, checkpoint_dir: Path) -> dict:
        """
        1. Load current ResNet-50 checkpoint + Fisher diagonal
        2. Compute updated Fisher on the validation set
        3. Fine-tune with EWC penalty on new training data
        4. Recompute Fisher on updated model
        5. Save model.pt + fisher.pt to checkpoint_dir/resnet50/
        """
        result: dict[str, Any] = {"updated": False, "val_acc": None, "error": None}

        try:
            import torch
            from ewc_utils import (
                EWCPenalty,
                build_resnet50,
                compute_fisher_diagonal,
                ewc_fine_tune,
                load_fisher,
                save_fisher,
                snapshot_params,
            )
        except ImportError as exc:
            result["error"] = f"PyTorch / torchvision not available: {exc}"
            self.log.warning(result["error"])
            return result

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log.info("ResNet-50 EWC update — device: %s", device)

        # Locate previous checkpoint artifacts
        checkpoint_ref = self.state.get("model_checkpoint_ref")
        prev_resnet_dir = (
            Path(checkpoint_ref) / RESNET_SUBDIR
            if checkpoint_ref else None
        )
        prev_weights = (
            prev_resnet_dir / "model.pt"
            if prev_resnet_dir and (prev_resnet_dir / "model.pt").exists()
            else None
        )
        prev_fisher_path = (
            prev_resnet_dir / "fisher.pt"
            if prev_resnet_dir and (prev_resnet_dir / "fisher.pt").exists()
            else None
        )

        if self.dry_run:
            self._dry_run_log(
                f"Would EWC fine-tune ResNet-50 from {prev_weights} "
                f"for {EWC_EPOCHS} epochs (lambda={EWC_LAMBDA})."
            )
            return {"updated": False, "dry_run": True}

        model = build_resnet50(RESNET_NUM_CLASSES, prev_weights, device)
        train_loader, val_loader = self._build_image_loaders(device)

        if train_loader is None:
            result["error"] = "No image training data found — skipping ResNet update."
            self.log.warning(result["error"])
            return result

        # Load or compute Fisher diagonal
        if prev_fisher_path:
            fisher = load_fisher(prev_fisher_path, device)
        else:
            self.log.info("No prior Fisher matrix — computing from val set.")
            fisher = compute_fisher_diagonal(
                model, val_loader, EWC_FISHER_SAMPLES, device
            )

        old_params  = snapshot_params(model, device)
        ewc_penalty = EWCPenalty(model, fisher, old_params, EWC_LAMBDA, device)

        history = ewc_fine_tune(
            model, ewc_penalty, train_loader, val_loader,
            lr=EWC_LR, epochs=EWC_EPOCHS, device=device,
        )

        best_val_acc = max(history["val_acc"]) if history["val_acc"] else None
        result["val_acc"] = best_val_acc
        self.log.info("ResNet-50 fine-tune complete.  best_val_acc=%.4f", best_val_acc or 0)

        # Recompute Fisher on updated model for next run
        new_fisher = compute_fisher_diagonal(
            model, val_loader, EWC_FISHER_SAMPLES, device
        )

        out_dir = checkpoint_dir / RESNET_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_dir / "model.pt")
        save_fisher(new_fisher, out_dir / "fisher.pt")
        with open(out_dir / "training_history.json", "w") as fh:
            json.dump(history, fh, indent=2)

        result["updated"] = True
        return result

    def _build_image_loaders(self, device):
        """
        Build DataLoaders for TCGA histopathology images.
        Expects PROCESSED_DATA_DIR/images/{train,val}/ as torchvision ImageFolder.
        Returns (None, None) if dirs don't exist.
        """
        train_dir = PROCESSED_DATA_DIR / "images" / "train"
        val_dir   = PROCESSED_DATA_DIR / "images" / "val"
        if not train_dir.exists():
            return None, None

        try:
            import torch
            from torchvision import datasets, transforms  # type: ignore

            tfm = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std =[0.229, 0.224, 0.225],
                ),
            ])
            train_ds = datasets.ImageFolder(str(train_dir), transform=tfm)
            val_ds   = datasets.ImageFolder(str(val_dir),   transform=tfm)
            pin      = (device.type == "cuda")
            return (
                torch.utils.data.DataLoader(
                    train_ds, batch_size=EWC_BATCH_SIZE, shuffle=True,
                    num_workers=2, pin_memory=pin,
                ),
                torch.utils.data.DataLoader(
                    val_ds, batch_size=EWC_BATCH_SIZE, shuffle=False,
                    num_workers=2, pin_memory=pin,
                ),
            )
        except Exception as exc:
            self.log.error("Failed to build image loaders: %s", exc)
            return None, None

    # ------------------------------------------------------------------
    # Ensemble replay-buffer update
    # ------------------------------------------------------------------

    def _run_ensemble_replay_update(self, checkpoint_dir: Path) -> dict:
        """
        Continual learning for XGBoost / LightGBM via memory replay:
        1. Load replay buffer → stratified-sample REPLAY_SAMPLE_FRAC
        2. Concatenate with new training data
        3. Continue training existing boosters for ENSEMBLE_BOOST_ROUNDS
        4. Save updated boosters; update replay buffer (reservoir sampling)
        """
        result: dict[str, Any] = {"updated": False, "metrics": {}, "error": None}

        try:
            import pandas as pd
        except ImportError as exc:
            result["error"] = f"pandas not available: {exc}"
            return result

        if not TRAIN_PARQUET.exists():
            result["error"] = "train.parquet not found — skipping ensemble update."
            self.log.warning(result["error"])
            return result

        try:
            train_df = pd.read_parquet(TRAIN_PARQUET)
            val_df   = pd.read_parquet(VAL_PARQUET)
        except Exception as exc:
            result["error"] = f"Could not load parquet data: {exc}"
            return result

        if self.dry_run:
            self._dry_run_log(
                f"Would update ensemble on {len(train_df)} new + replay samples "
                f"for {ENSEMBLE_BOOST_ROUNDS} rounds."
            )
            return {"updated": False, "dry_run": True}

        # Mix in replay buffer
        if REPLAY_BUFFER_PARQUET.exists():
            try:
                replay_df = pd.read_parquet(REPLAY_BUFFER_PARQUET)
                n_replay  = max(1, int(len(replay_df) * REPLAY_SAMPLE_FRAC))
                sample    = replay_df.sample(n=min(n_replay, len(replay_df)),
                                              random_state=42)
                train_df  = pd.concat([train_df, sample], ignore_index=True)
                self.log.info("Replay buffer: sampled %d / %d records.",
                              len(sample), len(replay_df))
            except Exception as exc:
                self.log.warning("Replay buffer load failed (%s) — proceeding without.", exc)

        label_col = "pathogenicity_class"
        if label_col not in train_df.columns:
            result["error"] = f"Label column '{label_col}' missing."
            return result

        feature_cols = [c for c in train_df.columns if c != label_col]
        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df[label_col].values
        X_val   = val_df[feature_cols].values.astype(np.float32)
        y_val   = val_df[label_col].values

        out_dir = checkpoint_dir / ENSEMBLE_SUBDIR
        out_dir.mkdir(parents=True, exist_ok=True)

        metrics: dict[str, Any] = {}
        metrics["xgboost"]  = self._update_xgboost(X_train, y_train, X_val, y_val, out_dir)
        metrics["lightgbm"] = self._update_lightgbm(X_train, y_train, X_val, y_val, out_dir)

        self._update_replay_buffer(train_df, label_col)

        result["updated"] = (
            metrics["xgboost"].get("saved", False) or
            metrics["lightgbm"].get("saved", False)
        )
        result["metrics"] = metrics
        return result

    def _update_xgboost(self, X_train, y_train, X_val, y_val, out_dir: Path) -> dict:
        result: dict[str, Any] = {"saved": False}
        try:
            import xgboost as xgb  # type: ignore
        except ImportError:
            return {"skipped": "xgboost not installed"}

        dtrain = xgb.DMatrix(X_train, label=y_train)
        dval   = xgb.DMatrix(X_val,   label=y_val)

        checkpoint_ref = self.state.get("model_checkpoint_ref")
        prev_path = (
            Path(checkpoint_ref) / ENSEMBLE_SUBDIR / "xgb_model.ubj"
            if checkpoint_ref else None
        )
        init_model = str(prev_path) if prev_path and prev_path.exists() else None

        params = {
            "objective":        "multi:softprob",
            "num_class":        5,
            "eval_metric":      ["mlogloss", "merror"],
            "max_depth":        6,
            "learning_rate":    0.05,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "tree_method":      "hist",
            "device":           "cuda" if _cuda_available() else "cpu",
        }

        evals_result: dict = {}
        booster = xgb.train(
            params, dtrain,
            num_boost_round=ENSEMBLE_BOOST_ROUNDS,
            evals=[(dtrain, "train"), (dval, "val")],
            evals_result=evals_result,
            xgb_model=init_model,
            verbose_eval=False,
        )
        booster.save_model(str(out_dir / "xgb_model.ubj"))
        result["saved"]        = True
        result["val_mlogloss"] = evals_result.get("val", {}).get("mlogloss", [None])[-1]
        result["val_merror"]   = evals_result.get("val", {}).get("merror",   [None])[-1]
        self.log.info("XGBoost: val_mlogloss=%.4f  val_merror=%.4f",
                      result["val_mlogloss"] or 0, result["val_merror"] or 0)
        return result

    def _update_lightgbm(self, X_train, y_train, X_val, y_val, out_dir: Path) -> dict:
        result: dict[str, Any] = {"saved": False}
        try:
            import lightgbm as lgb  # type: ignore
        except ImportError:
            return {"skipped": "lightgbm not installed"}

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set   = lgb.Dataset(X_val,   label=y_val, reference=train_set)

        checkpoint_ref = self.state.get("model_checkpoint_ref")
        prev_path = (
            Path(checkpoint_ref) / ENSEMBLE_SUBDIR / "lgb_model.txt"
            if checkpoint_ref else None
        )
        init_model = str(prev_path) if prev_path and prev_path.exists() else None

        params = {
            "objective":        "multiclass",
            "num_class":        5,
            "metric":           ["multi_logloss", "multi_error"],
            "num_leaves":       63,
            "learning_rate":    0.05,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq":     5,
            "verbose":          -1,
            "device_type":      "gpu" if _cuda_available() else "cpu",
        }

        evals_result: dict = {}
        booster = lgb.train(
            params, train_set,
            num_boost_round=ENSEMBLE_BOOST_ROUNDS,
            valid_sets=[val_set],
            valid_names=["val"],
            init_model=init_model,
            callbacks=[lgb.record_evaluation(evals_result),
                       lgb.log_evaluation(period=10)],
        )
        booster.save_model(str(out_dir / "lgb_model.txt"))
        result["saved"]      = True
        result["val_logloss"] = booster.best_score.get("val", {}).get("multi_logloss")
        result["val_error"]   = booster.best_score.get("val", {}).get("multi_error")
        self.log.info("LightGBM: val_logloss=%.4f  val_error=%.4f",
                      result["val_logloss"] or 0, result["val_error"] or 0)
        return result

    # ------------------------------------------------------------------
    # Replay buffer — reservoir sampling
    # ------------------------------------------------------------------

    def _update_replay_buffer(self, new_data: "pd.DataFrame", label_col: str) -> None:
        """
        Maintain a stratified replay buffer via reservoir sampling.
        New samples have an equal probability of displacing old ones,
        preserving the running distribution without loading full history.
        """
        try:
            import pandas as pd
        except ImportError:
            return

        buffer = (
            pd.read_parquet(REPLAY_BUFFER_PARQUET)
            if REPLAY_BUFFER_PARQUET.exists()
            else pd.DataFrame()
        )

        combined = pd.concat([buffer, new_data], ignore_index=True)

        if label_col in combined.columns:
            classes   = combined[label_col].unique()
            per_class = REPLAY_BUFFER_SIZE // len(classes)
            parts = [
                combined[combined[label_col] == cls].sample(
                    n=min(per_class, (combined[label_col] == cls).sum()),
                    random_state=42,
                )
                for cls in classes
            ]
            new_buffer = pd.concat(parts, ignore_index=True)
        else:
            new_buffer = combined.sample(
                n=min(REPLAY_BUFFER_SIZE, len(combined)), random_state=42
            )

        new_buffer.to_parquet(REPLAY_BUFFER_PARQUET, index=False)
        self.log.info(
            "Replay buffer updated: %d records.", len(new_buffer)
        )

    # ------------------------------------------------------------------
    # Checkpointing helpers
    # ------------------------------------------------------------------

    def _new_checkpoint_dir(self) -> Path:
        ts      = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        version = len(list(CHECKPOINT_DIR.glob("*_v*"))) + 1
        path    = CHECKPOINT_DIR / f"{ts}_v{version}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_checkpoint_metadata(self, checkpoint_dir: Path, results: dict) -> None:
        meta = {
            "version":          checkpoint_dir.name,
            "created_at":       datetime.now(timezone.utc).isoformat(),
            "strategy":         results.get("strategy"),
            "drift_score":      results.get("drift_score"),
            "resnet_val_acc":   results.get("resnet_val_acc"),
            "ensemble_metrics": results.get("ensemble_metrics"),
            "ewc_lambda":       EWC_LAMBDA,
            "resnet_updated":   results.get("resnet_updated"),
            "ensemble_updated": results.get("ensemble_updated"),
        }
        with open(checkpoint_dir / "metadata.json", "w") as fh:
            json.dump(meta, fh, indent=2)

    def _push_checkpoint(self, checkpoint_dir: Path) -> None:
        # Google Drive (Colab)
        gdrive = GDRIVE_CHECKPOINT_DIR / checkpoint_dir.name
        try:
            if GDRIVE_CHECKPOINT_DIR.parent.exists():
                shutil.copytree(str(checkpoint_dir), str(gdrive))
                self.log.info("Checkpoint mirrored to GDrive → %s", gdrive)
        except Exception as exc:
            self.log.warning("GDrive push skipped: %s", exc)

        # GCS
        if GCS_CHECKPOINT_PREFIX.startswith("gs://"):
            try:
                from google.cloud import storage  # type: ignore
                client      = storage.Client()
                bucket_name, prefix = GCS_CHECKPOINT_PREFIX[5:].split("/", 1)
                bucket      = client.bucket(bucket_name)
                gcs_prefix  = f"{prefix}/{checkpoint_dir.name}"
                for f in checkpoint_dir.rglob("*"):
                    if f.is_file():
                        bucket.blob(
                            f"{gcs_prefix}/{f.relative_to(checkpoint_dir)}"
                        ).upload_from_filename(str(f))
                self.log.info("Checkpoint pushed to GCS: gs://%s/%s",
                              bucket_name, gcs_prefix)
            except ImportError:
                self.log.warning("google-cloud-storage not installed — GCS skipped.")
            except Exception as exc:
                self.log.warning("GCS push failed: %s", exc)

    def _update_state(self, checkpoint_dir: Path, results: dict) -> None:
        now = datetime.now(timezone.utc).isoformat()
        self.state.set("model_checkpoint_ref", str(checkpoint_dir))
        self.state.set("model_last_trained",   now)
        fisher_path = checkpoint_dir / RESNET_SUBDIR / "fisher.pt"
        if fisher_path.exists():
            self.state.set("ewc_fisher_ref", str(fisher_path))
        self.state.set("ewc_lambda", EWC_LAMBDA)
        self.log.info("State updated: checkpoint=%s", checkpoint_dir.name)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False

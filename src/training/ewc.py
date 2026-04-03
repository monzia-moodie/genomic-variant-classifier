"""
src/training/ewc.py
====================
Elastic Weight Consolidation (EWC) for continual learning.

EWC prevents catastrophic forgetting when retraining on new ClinVar releases.
It anchors the weights that were most important for the old data (measured by
the Fisher Information Matrix) while allowing unimportant weights to adapt
freely to new data.

Mathematical foundation:
    L_EWC(θ) = L_B(θ) + (λ/2) Σ_i F_i (θ_i - θ_A*_i)²

where:
    L_B(θ)   = loss on new data (task B)
    F_i      = diagonal Fisher Information at weight i (importance measure)
    θ_A*     = optimal parameters after training on old data (task A)
    λ        = regularisation strength (tune via BWT metric)

Genomic context:
    Task A = model trained on ClinVar v1 + gnomAD v3
    Task B = updated model with ClinVar v2 reclassifications + gnomAD v4
    Important weights = those encoding stable biology:
      conservation scores, splice site disruption, loss-of-function flags
    Free weights = those encoding priors that change:
      allele frequency distributions, population-specific features

Implementations:
    EWC           — full neural network EWC (PyTorch)
    OnlineEWC     — running Fisher estimate across multiple tasks (avoids
                    growing penalty term after many ClinVar releases)
    TreeEWCProxy  — Fisher-inspired sample weighting for XGBoost/LightGBM
                    (since tree models don't have differentiable parameters)

Reference: Kirkpatrick et al. (2017), "Overcoming catastrophic forgetting in
neural networks." PNAS 114(13): 3521–3526.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Neural network backbone for tabular genomic features
# ---------------------------------------------------------------------------

class GenomicVariantMLP(nn.Module):
    """
    Feedforward network for tabular genomic variant features.

    Can replace or augment the tabular branch of the stacking ensemble.
    Designed for EWC compatibility: all parameters are named and accessible
    via named_parameters().

    Architecture:
        Input → [Linear → LayerNorm → GELU → Dropout] × n_layers → Output(2)

    LayerNorm preferred over BatchNorm for stability at small batch sizes and
    for continual learning (BN statistics can shift between tasks).
    """

    def __init__(
        self,
        input_dim:    int,
        hidden_dims:  list[int] = (512, 256, 128),
        output_dim:   int = 2,
        dropout:      float = 0.3,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        self.input_dim    = input_dim

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for i, h in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.LayerNorm(h))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        self.hidden = nn.Sequential(*layers)
        self.head   = nn.Linear(prev_dim, output_dim)

        # Residual projection (if first hidden dim != input_dim)
        self.residual_proj: Optional[nn.Module] = None
        if use_residual and input_dim != hidden_dims[0]:
            self.residual_proj = nn.Linear(input_dim, hidden_dims[0], bias=False)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden(x)
        return self.head(h)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def predict_proba_numpy(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X)
            return self.predict_proba(t).numpy()


# ---------------------------------------------------------------------------
# EWC (single task-pair)
# ---------------------------------------------------------------------------

class EWC:
    """
    Elastic Weight Consolidation for a single task-pair (A → B).

    Parameters
    ----------
    model      : fitted GenomicVariantMLP (or any nn.Module) after task A
    dataloader : DataLoader over task A data (for Fisher computation)
    device     : "cpu" or "cuda"
    """

    def __init__(
        self,
        model:      nn.Module,
        dataloader: DataLoader,
        device:     str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.model  = model.to(self.device)

        # θ_A* — snapshot of parameters after task A
        self.theta_star: dict[str, torch.Tensor] = {
            name: param.clone().detach().to(self.device)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # F_i — diagonal Fisher Information
        self.fisher: dict[str, torch.Tensor] = self._compute_fisher(dataloader)

        # Log the top-10 most important parameter groups
        fisher_norms = {k: float(v.sum()) for k, v in self.fisher.items()}
        top10 = sorted(fisher_norms.items(), key=lambda x: -x[1])[:10]
        logger.info("Fisher computed. Top-10 parameter importance:")
        for name, norm in top10:
            logger.info("  %-40s  %.4f", name, norm)

    def _compute_fisher(self, dataloader: DataLoader) -> dict[str, torch.Tensor]:
        """
        Empirical diagonal Fisher: F̂_i = (1/N) Σ (∂ log p(y|x,θ) / ∂θ_i)²
        Evaluated at θ = θ_A*.
        """
        fisher: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param, device=self.device)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        self.model.eval()
        n_samples = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)
            n_samples += inputs.size(0)

            self.model.zero_grad()
            log_probs = F.log_softmax(self.model(inputs), dim=-1)
            loss = F.nll_loss(log_probs, labels)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name].add_(param.grad.detach() ** 2 * inputs.size(0))

        for name in fisher:
            fisher[name] /= max(n_samples, 1)

        return fisher

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        EWC regularisation term: Σ_i F_i (θ_i - θ_A*_i)²

        Add to the task-B loss: total = ce_loss + (λ/2) * ewc.penalty(model)
        """
        pen = torch.tensor(0.0, device=self.device)
        for name, param in model.named_parameters():
            if name in self.fisher:
                pen = pen + (
                    self.fisher[name] * (param - self.theta_star[name]) ** 2
                ).sum()
        return pen


# ---------------------------------------------------------------------------
# Online EWC — accumulates Fisher across N tasks without growing penalty
# ---------------------------------------------------------------------------

class OnlineEWC:
    """
    Online EWC (Schwarz et al. 2018): maintains a single running Fisher
    estimate rather than accumulating one penalty term per ClinVar release.

    After N ClinVar releases, regular EWC has N-1 penalty terms in the loss.
    OnlineEWC collapses them into a single term using a running exponential
    average of the Fisher matrix.

    Usage:
        ewc = OnlineEWC(gamma=0.9)          # gamma: Fisher decay factor
        ewc.update(model, task_a_loader)    # after each training run
        # then in the next training loop:
        loss = ce_loss + (lambda/2) * ewc.penalty(model)
    """

    def __init__(self, gamma: float = 0.9) -> None:
        self.gamma       = gamma
        self.fisher:     Optional[dict[str, torch.Tensor]] = None
        self.theta_star: Optional[dict[str, torch.Tensor]] = None
        self.n_updates   = 0

    def update(
        self,
        model:      nn.Module,
        dataloader: DataLoader,
        device:     str = "cpu",
    ) -> None:
        """
        Update the running Fisher estimate after training on a new ClinVar release.
        Call this immediately after model training converges on the new data.
        """
        device_t = torch.device(device)
        new_ewc  = EWC(model, dataloader, device=device)

        if self.fisher is None:
            self.fisher     = {k: v.clone() for k, v in new_ewc.fisher.items()}
            self.theta_star = {k: v.clone() for k, v in new_ewc.theta_star.items()}
        else:
            # Exponential moving average: F̄ ← γ·F̄ + (1-γ)·F_new
            for name in self.fisher:
                self.fisher[name] = (
                    self.gamma * self.fisher[name]
                    + (1 - self.gamma) * new_ewc.fisher[name]
                )
            # Anchor shifts to the most recent optimal parameters
            self.theta_star = {k: v.clone() for k, v in new_ewc.theta_star.items()}

        self.n_updates += 1
        logger.info("OnlineEWC updated (%d total releases tracked).", self.n_updates)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        if self.fisher is None:
            return torch.tensor(0.0)
        device = next(model.parameters()).device
        pen = torch.tensor(0.0, device=device)
        for name, param in model.named_parameters():
            if name in self.fisher:
                pen = pen + (
                    self.fisher[name].to(device)
                    * (param - self.theta_star[name].to(device)) ** 2
                ).sum()
        return pen


# ---------------------------------------------------------------------------
# Training loop with EWC
# ---------------------------------------------------------------------------

def train_with_ewc(
    model:          nn.Module,
    dataloader:     DataLoader,
    ewc:            Optional[EWC | OnlineEWC],
    lambda_ewc:     float = 1000.0,
    epochs:         int   = 50,
    lr:             float = 1e-3,
    weight_decay:   float = 1e-4,
    device:         str   = "cpu",
    log_every:      int   = 10,
) -> dict[str, list[float]]:
    """
    Train on new ClinVar data with optional EWC regularisation.

    lambda_ewc controls the plasticity-stability trade-off:
      - Too high (>10 000): model can't adapt, AUROC on new data stagnates
      - Too low  (<100):    catastrophic forgetting, AUROC on old data drops
      - Sweet spot (500-5000): both tasks retained; tune via BWT metric

    Returns
    -------
    history dict with keys: total_loss, ce_loss, ewc_penalty (per epoch)
    """
    device_t = torch.device(device)
    model    = model.to(device_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: dict[str, list[float]] = {"total_loss": [], "ce_loss": [], "ewc_penalty": []}

    for epoch in range(1, epochs + 1):
        model.train()
        ep_ce   = 0.0
        ep_ewc  = 0.0
        n_batches = 0

        for inputs, labels in dataloader:
            inputs = inputs.to(device_t)
            labels = labels.to(device_t)
            optimizer.zero_grad()

            logits    = model(inputs)
            ce_loss   = F.cross_entropy(logits, labels)
            ewc_pen   = ewc.penalty(model) if ewc is not None else torch.tensor(0.0)
            total     = ce_loss + (lambda_ewc / 2) * ewc_pen

            total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            ep_ce  += ce_loss.item()
            ep_ewc += ewc_pen.item()
            n_batches += 1

        scheduler.step()
        history["ce_loss"].append(ep_ce / n_batches)
        history["ewc_penalty"].append(ep_ewc / n_batches)
        history["total_loss"].append(
            history["ce_loss"][-1] + (lambda_ewc / 2) * history["ewc_penalty"][-1]
        )

        if epoch % log_every == 0:
            logger.info(
                "Epoch %3d | CE: %.4f | EWC: %.6f | Total: %.4f",
                epoch, history["ce_loss"][-1],
                history["ewc_penalty"][-1], history["total_loss"][-1],
            )

    return history


def evaluate_continual_learning(
    model:      nn.Module,
    loader_a:   DataLoader,   # old task data
    loader_b:   DataLoader,   # new task data
    device:     str = "cpu",
) -> dict[str, float]:
    """
    Measure Backward Transfer (BWT) and task-B accuracy.

    BWT = acc_task_A_after_B - acc_task_A_before_B
      BWT ≈ 0  : no forgetting (ideal)
      BWT < 0  : catastrophic forgetting
      BWT > 0  : positive transfer (rare but possible)
    """
    device_t = torch.device(device)
    model.eval()

    def accuracy(loader: DataLoader) -> float:
        correct = total = 0
        with torch.no_grad():
            for x, y in loader:
                preds = model(x.to(device_t)).argmax(dim=-1)
                correct += (preds == y.to(device_t)).sum().item()
                total   += y.size(0)
        return correct / max(total, 1)

    return {
        "task_a_retention": accuracy(loader_a),
        "task_b_accuracy":  accuracy(loader_b),
    }


# ---------------------------------------------------------------------------
# Tree-model EWC proxy (for XGBoost / LightGBM)
# ---------------------------------------------------------------------------

class TreeEWCProxy:
    """
    EWC-inspired sample weighting for XGBoost and LightGBM.

    Tree models are not differentiable, so we cannot compute a Fisher matrix.
    Instead, we approximate the EWC principle via sample weighting:

      - Variants where the OLD model is confident and correct → stable biology
        → downweight in retraining (they are well-modelled, don't over-pull)
      - Variants where the OLD model is wrong or uncertain → novel/drifted
        → upweight in retraining (model needs to correct these)
      - ClinVar reclassifications → always upweight (ground truth changed)

    This approximation implements the key insight of EWC — "protect what you
    know, adapt to what you don't" — without requiring gradient computation.

    Usage:
        proxy = TreeEWCProxy(lambda_decay=0.5)
        weights = proxy.compute_weights(
            old_model, X_train_new, y_train_new,
            reclassified_ids=set(reclassification_manifest["variant_id"]),
        )
        new_model.fit(X_train_new, y_train_new, sample_weight=weights)
    """

    def __init__(
        self,
        lambda_decay:          float = 0.5,
        confidence_threshold:  float = 0.7,
        reclassified_boost:    float = 2.0,
        temporal_decay_lambda: float = 0.0,    # 0 = no temporal decay
    ) -> None:
        self.lambda_decay         = lambda_decay
        self.confidence_threshold = confidence_threshold
        self.reclassified_boost   = reclassified_boost
        self.temporal_decay_lambda = temporal_decay_lambda

    def compute_weights(
        self,
        old_model,
        X_new:           np.ndarray,
        y_new:           np.ndarray,
        reclassified_ids: Optional[set[str]]     = None,
        variant_ids:      Optional[np.ndarray]   = None,
        submission_dates: Optional[np.ndarray]   = None,  # datetime array
        current_date:     Optional[str]          = None,
    ) -> np.ndarray:
        """
        Compute per-sample training weights.

        Parameters
        ----------
        old_model : fitted XGBClassifier or LGBMClassifier
        X_new     : feature matrix for new training run
        y_new     : labels for new training run
        reclassified_ids : set of variant_id strings that ClinVar reclassified
        variant_ids : array of variant_id strings (same length as X_new)
        submission_dates : array of pd.Timestamp (for temporal decay)
        current_date : ISO date string for temporal decay reference

        Returns
        -------
        weights : array of shape (n_samples,), values in [0.1, reclassified_boost]
        """
        n = len(X_new)
        weights = np.ones(n, dtype=np.float64)

        # --- Confidence-based stability weighting ---
        try:
            proba_old = old_model.predict_proba(X_new)
            if proba_old.ndim == 2:
                proba_old = proba_old[:, 1]
            predicted_class = (proba_old >= 0.5).astype(int)
            correct   = (predicted_class == y_new).astype(float)
            confidence = np.where(proba_old >= 0.5, proba_old, 1 - proba_old)
            # stability_score ∈ [0, 1]: 1 = old model confident and correct
            stability_score = confidence * correct
            # Downweight stable examples, upweight uncertain / wrong ones
            weights *= (1.0 - self.lambda_decay * stability_score)
        except Exception as e:
            logger.warning("TreeEWCProxy: could not compute stability weights: %s", e)

        # --- Reclassification boost ---
        if reclassified_ids and variant_ids is not None:
            for i, vid in enumerate(variant_ids):
                if str(vid) in reclassified_ids:
                    weights[i] *= self.reclassified_boost
                    logger.debug("Reclassified variant upweighted: %s", vid)

        # --- Temporal decay (recent data gets higher weight) ---
        if (
            self.temporal_decay_lambda > 0
            and submission_dates is not None
            and current_date is not None
        ):
            import pandas as pd
            ref_date = pd.Timestamp(current_date)
            for i, sub_date in enumerate(submission_dates):
                if pd.isnull(sub_date):
                    continue
                days_old = (ref_date - pd.Timestamp(sub_date)).days
                decay = np.exp(-self.temporal_decay_lambda * days_old / 365)
                weights[i] *= decay

        # Clip to [floor, ceiling]
        weights = np.clip(weights, 0.1, float(self.reclassified_boost))
        logger.info(
            "TreeEWCProxy: weights computed. mean=%.3f, std=%.3f, "
            "upweighted(>1.2)=%d, downweighted(<0.5)=%d",
            weights.mean(), weights.std(),
            int((weights > 1.2).sum()), int((weights < 0.5).sum()),
        )
        return weights
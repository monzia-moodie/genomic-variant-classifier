"""
agents/ewc_utils.py
===================
Elastic Weight Consolidation (EWC) utilities for the ResNet-50 branch.

Theory
------
EWC (Kirkpatrick et al., 2017) prevents catastrophic forgetting by adding a
quadratic penalty to the loss that resists parameter movement away from the
previous task's optimum, weighted by parameter importance (Fisher diagonal):

    L_EWC(θ) = L_new(θ)  +  (λ/2) · Σ_i  F_i · (θ_i - θ*_i)²

where:
    θ*    = parameters after training on the previous task
    F_i   = i-th diagonal of the empirical Fisher information matrix
    λ     = regularisation strength (EWC_LAMBDA in config)

The Fisher diagonal is approximated with sampled gradients:
    F_i ≈ (1/N) · Σ_{x,y} (∂ log p(y|x,θ) / ∂θ_i)²

Note on applicability
---------------------
EWC is defined for neural networks with differentiable parameters.
The XGBoost / LightGBM ensemble uses a different continual-learning strategy
(memory replay buffer) implemented directly in TrainingLifecycleAgent.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Fisher diagonal computation
# ---------------------------------------------------------------------------

def compute_fisher_diagonal(
    model: nn.Module,
    data_loader: DataLoader,
    n_samples: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """
    Compute the empirical Fisher information diagonal for all named parameters
    that require gradients.

    Parameters
    ----------
    model       : the network (in eval mode)
    data_loader : yields (images, labels) batches from the *reference* dataset
                  (a stratified sample of the previous training distribution)
    n_samples   : maximum number of samples to use (for speed)
    device      : torch.device

    Returns
    -------
    dict mapping param_name → 1-D tensor of Fisher values (same shape as param)
    """
    model.eval()
    fisher: dict[str, torch.Tensor] = {
        name: torch.zeros_like(param, device=device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    total = 0
    for images, labels in data_loader:
        if total >= n_samples:
            break

        images = images.to(device)
        labels = labels.to(device)
        batch_size = images.size(0)

        model.zero_grad()
        logits   = model(images)
        log_probs = F.log_softmax(logits, dim=1)

        # Use the *predicted* label (not ground-truth) for the Fisher estimate
        # — this is the standard "empirical Fisher" used in the original paper
        predicted = log_probs.max(1)[1]
        loss = F.nll_loss(log_probs, predicted)
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher[name] += param.grad.detach().pow(2) * batch_size

        total += batch_size

    # Normalise by total samples seen
    if total > 0:
        for name in fisher:
            fisher[name] /= total

    log.info("Fisher diagonal computed over %d samples.", total)
    return fisher


def save_fisher(fisher: dict[str, torch.Tensor], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.cpu() for k, v in fisher.items()}, path)
    log.info("Fisher diagonal saved → %s", path)


def load_fisher(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    data = torch.load(path, map_location=device)
    log.info("Fisher diagonal loaded ← %s", path)
    return data


# ---------------------------------------------------------------------------
# EWC penalty
# ---------------------------------------------------------------------------

class EWCPenalty:
    """
    Computes the EWC quadratic penalty term during fine-tuning.

    Usage
    -----
    ewc = EWCPenalty(model, fisher, old_params, lam=400.0)

    # inside training loop:
    loss = criterion(logits, labels) + ewc.penalty(model)
    loss.backward()
    """

    def __init__(
        self,
        model:      nn.Module,
        fisher:     dict[str, torch.Tensor],
        old_params: dict[str, torch.Tensor],
        lam:        float,
        device:     torch.device,
    ):
        self.fisher     = {k: v.to(device) for k, v in fisher.items()}
        self.old_params = {k: v.to(device) for k, v in old_params.items()}
        self.lam        = lam

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """
        Returns scalar EWC penalty: (λ/2) · Σ F_i · (θ_i − θ*_i)²
        """
        total = torch.tensor(0.0, device=next(model.parameters()).device)
        for name, param in model.named_parameters():
            if name in self.fisher and name in self.old_params:
                diff    = param - self.old_params[name]
                penalty = (self.fisher[name] * diff.pow(2)).sum()
                total   = total + penalty
        return (self.lam / 2.0) * total


def snapshot_params(model: nn.Module, device: torch.device) -> dict[str, torch.Tensor]:
    """Return a detached copy of all trainable parameters — the θ* anchor."""
    return {
        name: param.detach().clone().to(device)
        for name, param in model.named_parameters()
        if param.requires_grad
    }


# ---------------------------------------------------------------------------
# EWC fine-tune loop
# ---------------------------------------------------------------------------

def ewc_fine_tune(
    model:       nn.Module,
    ewc_penalty: EWCPenalty,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    lr:          float,
    epochs:      int,
    device:      torch.device,
) -> dict[str, list[float]]:
    """
    Fine-tune *model* on new data with the EWC regularisation term.

    Returns a history dict with keys 'train_loss', 'val_loss', 'val_acc'.
    """
    model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )
    criterion = nn.CrossEntropyLoss()
    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [], "val_acc": []
    }

    for epoch in range(1, epochs + 1):
        # ---- Training ----
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits      = model(images)
            task_loss   = criterion(logits, labels)
            total_loss  = task_loss + ewc_penalty.penalty(model)
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item() * images.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ---- Validation ----
        model.eval()
        val_loss_sum, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits    = model(images)
                val_loss_sum += criterion(logits, labels).item() * images.size(0)
                preds     = logits.argmax(dim=1)
                correct  += (preds == labels).sum().item()
                total    += images.size(0)

        val_loss = val_loss_sum / total if total else 0.0
        val_acc  = correct / total if total else 0.0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        log.info(
            "  Epoch %d/%d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
            epoch, epochs, train_loss, val_loss, val_acc,
        )

    return history


# ---------------------------------------------------------------------------
# ResNet-50 builder (lightweight wrapper around torchvision)
# ---------------------------------------------------------------------------

def build_resnet50(num_classes: int, pretrained_path: Path | None = None,
                   device: torch.device | None = None) -> nn.Module:
    """
    Build (or reload) the ResNet-50 with a custom classification head.

    Parameters
    ----------
    num_classes     : number of output classes
    pretrained_path : if provided, loads state_dict from this .pt file
    device          : target device
    """
    try:
        from torchvision import models  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "torchvision is required for the ResNet-50 branch. "
            "pip install torchvision --trusted-host pypi.org"
        ) from exc

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    if pretrained_path and Path(pretrained_path).exists():
        state = torch.load(pretrained_path, map_location="cpu")
        # Tolerate checkpoints saved as {"model_state_dict": ...}
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        model.load_state_dict(state)
        log.info("ResNet-50 weights loaded ← %s", pretrained_path)
    else:
        log.warning("No pretrained weights found at %s — using random init.", pretrained_path)

    if device:
        model = model.to(device)
    return model

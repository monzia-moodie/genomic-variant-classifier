"""
Evaluation Module for Genomic Variant Classification
Author: Monzia Moodie
"""
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix
)
from sklearn.calibration import calibration_curve

def compute_classification_metrics(y_true, y_pred, y_proba):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auroc": roc_auc_score(y_true, y_proba),
        "auprc": average_precision_score(y_true, y_proba),
        "brier_score": brier_score_loss(y_true, y_proba),
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }

class ModelEvaluator:
    def __init__(self, y_true, y_proba, threshold=0.5):
        self.y_true = np.array(y_true)
        self.y_proba = np.array(y_proba)
        self.threshold = threshold
        self.y_pred = (self.y_proba >= threshold).astype(int)

    def get_all_metrics(self):
        return {
            "classification": compute_classification_metrics(
                self.y_true, self.y_pred, self.y_proba
            ),
        }

    def generate_report(self):
        metrics = self.get_all_metrics()
        clf = metrics["classification"]
        lines = [
            "=" * 50,
            "MODEL EVALUATION REPORT",
            "=" * 50,
            f"Samples: {len(self.y_true)} ({self.y_true.sum()} positive)",
            f"AUROC: {clf['auroc']:.4f}",
            f"AUPRC: {clf['auprc']:.4f}",
            f"F1: {clf['f1']:.4f}",
            f"Precision: {clf['precision']:.4f}",
            f"Recall: {clf['recall']:.4f}",
            "=" * 50,
        ]
        return "\n".join(lines)

if __name__ == "__main__":
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.3, 200)
    y_proba = 0.6 * y_true + 0.4 * np.random.beta(2, 5, 200)
    y_proba = np.clip(y_proba, 0, 1)
    evaluator = ModelEvaluator(y_true, y_proba)
    print(evaluator.generate_report())

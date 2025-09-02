"""
Evaluation utilities for classification models.

Computes ROC AUC, PR AUC, and classification report 
given true labels and predicted probabilities.
"""
# -- Imports --
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
import numpy as np
from typing import Dict, Any


# -- Function Definitions --
def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> Dict[str, Any]:
    """
    Evaluates a model for key metrics.

    Inputs:
        y_true (np.ndarray): True labels.
        proba (np.ndarray): Predicted probabilities.
        threshold (float, optional): Probability cutoff for classification.
    
    Returns:
        dict: {
            "roc_auc": float,
            "pr_auc": float,
            "report": str (classification report)
        }
    """
    p = np.asarray(proba)
    roc = roc_auc_score(y_true, p)
    pr = average_precision_score(y_true, p)
    pred = (p >= threshold).astype(int)
    report = classification_report(y_true, pred, digits=3)
    return {"roc_auc": roc, "pr_auc": pr, "report": report}

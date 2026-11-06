"""
Utility functions for model metric evaluations and image handling.
"""
import numpy as np


def compute_classification_metrics(confusion_mat):
    """
    Computes macro-averaged Precision, Recall, and F1-Score
    from an (N, N) confusion matrix.
    """
    matrix = np.array(confusion_mat, dtype=float)
    n_classes = matrix.shape[0]
    if n_classes == 0:
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    precisions = []
    recalls = []

    for i in range(n_classes):
        tp = matrix[i, i]
        fp = np.sum(matrix[:, i]) - tp
        fn = np.sum(matrix[i, :]) - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)

    avg_precision = float(np.mean(precisions))
    avg_recall = float(np.mean(recalls))
    f1 = (2 * avg_precision * avg_recall / (avg_precision + avg_recall)) if (avg_precision + avg_recall) > 0 else 0.0

    return {
        'precision': round(avg_precision, 4),
        'recall': round(avg_recall, 4),
        'f1': round(f1, 4),
    }

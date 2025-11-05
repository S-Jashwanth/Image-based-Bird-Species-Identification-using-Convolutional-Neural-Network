# classification_app/utils.py
import numpy as np
from sklearn.metrics import confusion_matrix

def precision(matrix):
    avg_precise = 0
    for i in range(16):
        tp = matrix[i][i]
        fp = np.sum(matrix[:, i])
        if fp != 0:
            avg_precise += tp / fp
    return avg_precise / 16

def recall(matrix):
    avg_recall = 0
    for i in range(16):
        tp = matrix[i][i]
        fn = np.sum(matrix[i, :])
        avg_recall += tp / fn
    return avg_recall / 16

def f1_score(precision, recall):
    return (2 * precision * recall) / (precision + recall)

from typing import List, Dict
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_classification(y_true: List[int], y_pred: List[int], class_names=None) -> Dict:
    if not y_true:
        return {"message":"No evaluation samples supplied."}
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classes": class_names or []
    }

def uncertainty_level(std: float) -> str:
    if std < 0.05:
        return "Low"
    if std < 0.12:
        return "Moderate"
    return "High"

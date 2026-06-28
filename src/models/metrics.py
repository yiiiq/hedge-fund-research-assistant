"""Small dependency-free classification metrics."""

from __future__ import annotations

from collections import Counter


def classification_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    label_set = sorted(set(y_true) | set(y_pred))
    total = len(y_true)
    accuracy = sum(true == pred for true, pred in zip(y_true, y_pred)) / total if total else 0.0

    per_class = {}
    weighted_f1 = 0.0
    macro_f1 = 0.0
    supports = Counter(y_true)
    for label in label_set:
        tp = sum(true == label and pred == label for true, pred in zip(y_true, y_pred))
        fp = sum(true != label and pred == label for true, pred in zip(y_true, y_pred))
        fn = sum(true == label and pred != label for true, pred in zip(y_true, y_pred))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        support = supports[label]
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
        macro_f1 += f1
        weighted_f1 += f1 * support

    if label_set:
        macro_f1 /= len(label_set)
    if total:
        weighted_f1 /= total

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }


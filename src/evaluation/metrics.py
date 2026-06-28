"""Classification metrics used across model families."""

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


def confusion_matrix_counts(
    y_true: list[str],
    y_pred: list[str],
    label_order: list[str] | None = None,
) -> list[dict[str, int | str]]:
    label_order = label_order or sorted(set(y_true) | set(y_pred))
    rows = []
    for true_label in label_order:
        row: dict[str, int | str] = {"true_label": true_label}
        for pred_label in label_order:
            row[pred_label] = sum(
                true == true_label and pred == pred_label
                for true, pred in zip(y_true, y_pred)
            )
        rows.append(row)
    return rows


def classification_report_rows(metrics: dict) -> list[dict[str, float | int | str]]:
    rows = []
    for label, values in metrics["per_class"].items():
        rows.append(
            {
                "label": label,
                "precision": values["precision"],
                "recall": values["recall"],
                "f1": values["f1"],
                "support": values["support"],
            }
        )
    rows.append(
        {
            "label": "macro_avg",
            "precision": "",
            "recall": "",
            "f1": metrics["macro_f1"],
            "support": sum(values["support"] for values in metrics["per_class"].values()),
        }
    )
    rows.append(
        {
            "label": "weighted_avg",
            "precision": "",
            "recall": "",
            "f1": metrics["weighted_f1"],
            "support": sum(values["support"] for values in metrics["per_class"].values()),
        }
    )
    return rows

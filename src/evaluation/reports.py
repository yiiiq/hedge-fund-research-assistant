"""Write standard evaluation artifacts for a model."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.error_analysis import write_error_analysis
from src.evaluation.metrics import (
    classification_metrics,
    classification_report_rows,
    confusion_matrix_counts,
)
from src.evaluation.predictions import build_prediction_rows, write_csv
from src.models.data import labels
from src.models.io import write_json


def evaluate_split(rows: list[dict[str, str]], predictions: list[str]) -> dict:
    return classification_metrics(labels(rows), predictions)


def write_split_artifacts(
    output_dir: Path,
    split_name: str,
    rows: list[dict[str, str]],
    predictions: list[str],
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    y_true = labels(rows)
    metrics = classification_metrics(y_true, predictions)
    prediction_rows = build_prediction_rows(rows, predictions)

    write_csv(
        output_dir / f"{split_name}_predictions.csv",
        prediction_rows,
    )
    write_csv(
        output_dir / f"{split_name}_classification_report.csv",
        classification_report_rows(metrics),
        ["label", "precision", "recall", "f1", "support"],
    )
    label_order = sorted(set(y_true) | set(predictions))
    write_csv(
        output_dir / f"{split_name}_confusion_matrix.csv",
        confusion_matrix_counts(y_true, predictions, label_order=label_order),
        ["true_label", *label_order],
    )
    write_error_analysis(output_dir, split_name, prediction_rows)
    return metrics


def write_evaluation_artifacts(
    output_dir: Path,
    split_rows: dict[str, list[dict[str, str]]],
    split_predictions: dict[str, list[str]],
) -> dict:
    metrics = {
        split_name: write_split_artifacts(output_dir, split_name, rows, split_predictions[split_name])
        for split_name, rows in split_rows.items()
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics

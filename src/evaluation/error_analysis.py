"""Error-analysis summaries for model predictions."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.evaluation.predictions import write_csv


ERROR_COLUMNS = [
    "chunk_id",
    "ticker",
    "form",
    "section",
    "true_label",
    "pred_label",
    "secondary_label",
    "text",
]


def error_rows(prediction_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [
        {
            "chunk_id": row["chunk_id"],
            "ticker": row["ticker"],
            "form": row["form"],
            "section": row["section"],
            "true_label": row["true_label"],
            "pred_label": row["pred_label"],
            "secondary_label": row["secondary_label"],
            "text": row["text"],
        }
        for row in prediction_rows
        if row["is_correct"] == "False"
    ]


def error_summary_rows(prediction_rows: list[dict[str, str]]) -> list[dict[str, int | str]]:
    counts = Counter(
        (row["true_label"], row["pred_label"])
        for row in prediction_rows
        if row["is_correct"] == "False"
    )
    return [
        {
            "true_label": true_label,
            "pred_label": pred_label,
            "count": count,
        }
        for (true_label, pred_label), count in counts.most_common()
    ]


def write_error_analysis(output_dir: Path, split_name: str, prediction_rows: list[dict[str, str]]) -> None:
    write_csv(
        output_dir / f"{split_name}_errors.csv",
        error_rows(prediction_rows),
        ERROR_COLUMNS,
    )
    write_csv(
        output_dir / f"{split_name}_error_summary.csv",
        error_summary_rows(prediction_rows),
        ["true_label", "pred_label", "count"],
    )

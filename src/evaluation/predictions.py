"""Prediction table helpers."""

from __future__ import annotations

import csv
from pathlib import Path


PREDICTION_COLUMNS = [
    "chunk_id",
    "ticker",
    "form",
    "filing_date",
    "section",
    "true_label",
    "pred_label",
    "is_correct",
    "secondary_label",
    "source_path",
    "text",
]


def build_prediction_rows(rows: list[dict[str, str]], predictions: list[str]) -> list[dict[str, str]]:
    output = []
    for row, pred_label in zip(rows, predictions):
        true_label = row["label"]
        output.append(
            {
                "chunk_id": row.get("chunk_id", ""),
                "ticker": row.get("ticker", ""),
                "form": row.get("form", ""),
                "filing_date": row.get("filing_date", ""),
                "section": row.get("section", ""),
                "true_label": true_label,
                "pred_label": pred_label,
                "is_correct": str(true_label == pred_label),
                "secondary_label": row.get("secondary_label", ""),
                "source_path": row.get("source_path", ""),
                "text": row.get("text", ""),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_prediction_csv(path: Path, rows: list[dict[str, str]], predictions: list[str]) -> None:
    write_csv(path, build_prediction_rows(rows, predictions), PREDICTION_COLUMNS)

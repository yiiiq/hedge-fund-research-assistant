"""Shared CSV loading helpers for model training."""

from __future__ import annotations

import csv
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def read_split(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_splits(data_dir: Path | None = None) -> dict[str, list[dict[str, str]]]:
    data_dir = data_dir or project_root() / "data" / "splits"
    return {
        "train": read_split(data_dir / "train.csv"),
        "validation": read_split(data_dir / "validation.csv"),
        "test": read_split(data_dir / "test.csv"),
    }


def labels(rows: list[dict[str, str]], label_col: str = "label") -> list[str]:
    return [row[label_col] for row in rows]


def texts(rows: list[dict[str, str]], text_col: str = "text") -> list[str]:
    return [row[text_col] for row in rows]


"""Sampling helpers for model training."""

from __future__ import annotations

from collections import defaultdict
import random


def oversample_minority_classes(
    rows: list[dict[str, str]],
    min_count: int | None,
    label_col: str = "label",
    random_state: int = 42,
) -> tuple[list[dict[str, str]], dict[str, dict[str, int]]]:
    """Duplicate minority-class rows until each class has at least min_count rows."""
    if min_count is None:
        return list(rows), class_count_summary(rows, label_col=label_col)
    if min_count < 1:
        raise ValueError("min_count must be a positive integer.")

    rng = random.Random(random_state)
    rows_by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_label[row[label_col]].append(row)

    sampled_rows = list(rows)
    summary = {}
    for label, label_rows in sorted(rows_by_label.items()):
        original_count = len(label_rows)
        target_count = max(original_count, min_count)
        added_count = target_count - original_count
        if added_count:
            sampled_rows.extend(rng.choices(label_rows, k=added_count))
        summary[label] = {
            "original": original_count,
            "added": added_count,
            "final": target_count,
        }

    rng.shuffle(sampled_rows)
    return sampled_rows, summary


def class_count_summary(
    rows: list[dict[str, str]],
    label_col: str = "label",
) -> dict[str, dict[str, int]]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[row[label_col]] += 1
    return {
        label: {"original": count, "added": 0, "final": count}
        for label, count in sorted(counts.items())
    }

"""Train and evaluate a majority-class baseline."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.evaluation.reports import write_evaluation_artifacts
from src.models.data import labels, load_splits, project_root
from src.models.io import write_json


def train_majority_baseline(train_rows: list[dict[str, str]]) -> dict:
    counts = Counter(labels(train_rows))
    majority_label, majority_count = counts.most_common(1)[0]
    return {
        "model_type": "majority_baseline",
        "majority_label": majority_label,
        "majority_count": majority_count,
        "label_counts": dict(counts),
    }


def predict(model: dict, rows: list[dict[str, str]]) -> list[str]:
    return [model["majority_label"]] * len(rows)


def run(output_dir: Path | None = None) -> dict:
    splits = load_splits()
    model = train_majority_baseline(splits["train"])
    split_predictions = {split: predict(model, rows) for split, rows in splits.items()}
    output_dir = output_dir or project_root() / "backend" / "model_artifacts" / "majority_baseline"
    metrics = write_evaluation_artifacts(output_dir, splits, split_predictions)
    payload = {"model": model, "metrics": metrics}
    write_json(output_dir / "model.json", model)
    return payload


def main() -> None:
    payload = run()
    print("Majority label:", payload["model"]["majority_label"])
    print("Validation macro F1:", round(payload["metrics"]["validation"]["macro_f1"], 4))
    print("Test macro F1:", round(payload["metrics"]["test"]["macro_f1"], 4))


if __name__ == "__main__":
    main()

"""Train and evaluate the investment-theme keyword baseline."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from src.labeling.taxonomy import WEAK_LABEL_KEYWORDS
from src.models.data import labels, load_splits, project_root
from src.models.io import write_json
from src.models.metrics import classification_metrics


def train_keyword_baseline(train_rows: list[dict[str, str]]) -> dict:
    label_counts = Counter(labels(train_rows))
    return {
        "model_type": "keyword_baseline",
        "keyword_map": WEAK_LABEL_KEYWORDS,
        "label_priors": dict(label_counts),
        "default_label": label_counts.most_common(1)[0][0],
    }


def keyword_scores(text: str, keyword_map: dict[str, list[str]]) -> dict[str, int]:
    text_l = f" {(text or '').lower()} "
    return {
        label: sum(keyword.lower() in text_l for keyword in keywords)
        for label, keywords in keyword_map.items()
    }


def predict_one(text: str, model: dict) -> str:
    scores = keyword_scores(text, model["keyword_map"])
    best_score = max(scores.values()) if scores else 0
    if best_score == 0:
        return model["default_label"]

    tied = [label for label, score in scores.items() if score == best_score]
    priors = model["label_priors"]
    return max(tied, key=lambda label: priors.get(label, 0))


def predict(model: dict, rows: list[dict[str, str]]) -> list[str]:
    return [predict_one(row["text"], model) for row in rows]


def run(output_dir: Path | None = None) -> dict:
    splits = load_splits()
    model = train_keyword_baseline(splits["train"])
    metrics = {
        split: classification_metrics(labels(rows), predict(model, rows))
        for split, rows in splits.items()
    }
    payload = {"model": model, "metrics": metrics}
    output_dir = output_dir or project_root() / "backend" / "model_artifacts" / "keyword_baseline"
    write_json(output_dir / "model.json", model)
    write_json(output_dir / "metrics.json", metrics)
    return payload


def main() -> None:
    payload = run()
    print("Default label:", payload["model"]["default_label"])
    print("Validation macro F1:", round(payload["metrics"]["validation"]["macro_f1"], 4))
    print("Test macro F1:", round(payload["metrics"]["test"]["macro_f1"], 4))


if __name__ == "__main__":
    main()


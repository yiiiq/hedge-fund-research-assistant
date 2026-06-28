"""Train and evaluate a TF-IDF logistic regression classifier."""

from __future__ import annotations

from pathlib import Path

from src.evaluation.reports import write_evaluation_artifacts
from src.models.data import labels, load_splits, project_root, texts


DEFAULT_TFIDF_PARAMS = {
    "lowercase": True,
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "max_features": None,
    "sublinear_tf": True,
    "strip_accents": "unicode",
}

DEFAULT_LOGREG_PARAMS = {
    "C": 3.0,
    "class_weight": "balanced",
    "max_iter": 1000,
    "n_jobs": None,
    "random_state": 42,
}


def build_tfidf_logreg_pipeline(
    tfidf_params: dict | None = None,
    logreg_params: dict | None = None,
):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "TF-IDF logistic regression requires scikit-learn. "
            "Install project dependencies before running this trainer."
        ) from exc

    vectorizer_params = DEFAULT_TFIDF_PARAMS | (tfidf_params or {})
    classifier_params = DEFAULT_LOGREG_PARAMS | (logreg_params or {})
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(**vectorizer_params)),
            ("classifier", LogisticRegression(**classifier_params)),
        ]
    )


def train_tfidf_logreg(
    train_rows: list[dict[str, str]],
    tfidf_params: dict | None = None,
    logreg_params: dict | None = None,
):
    pipeline = build_tfidf_logreg_pipeline(
        tfidf_params=tfidf_params,
        logreg_params=logreg_params,
    )
    pipeline.fit(texts(train_rows), labels(train_rows))
    return pipeline


def run(output_dir: Path | None = None) -> dict:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError(
            "TF-IDF logistic regression artifact writing requires joblib. "
            "Install project dependencies before running this trainer."
        ) from exc

    splits = load_splits()
    model = train_tfidf_logreg(splits["train"])
    output_dir = output_dir or project_root() / "backend" / "model_artifacts" / "tfidf_logreg"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    split_predictions = {
        split: list(model.predict(texts(rows)))
        for split, rows in splits.items()
    }
    metrics = write_evaluation_artifacts(output_dir, splits, split_predictions)
    return {"model_path": str(output_dir / "model.joblib"), "metrics": metrics}


def main() -> None:
    payload = run()
    print("Model path:", payload["model_path"])
    print("Validation macro F1:", round(payload["metrics"]["validation"]["macro_f1"], 4))
    print("Test macro F1:", round(payload["metrics"]["test"]["macro_f1"], 4))


if __name__ == "__main__":
    main()

"""Tune TF-IDF logistic regression hyperparameters on the validation split."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from src.evaluation.metrics import classification_metrics
from src.evaluation.reports import write_evaluation_artifacts
from src.models.data import labels, load_splits, project_root, texts
from src.models.io import write_json
from src.models.sampling import oversample_minority_classes
from src.models.tfidf_logreg import train_tfidf_logreg


MIN_DF_PERCENTAGES = [0.001, 0.0025, 0.005, 0.01]

PARAM_GRID = [
    {
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "tfidf__min_df": MIN_DF_PERCENTAGES,
        "tfidf__max_df": [0.7, 0.85, 0.95],
        "tfidf__max_features": [None, 10000, 20000],
        "tfidf__sublinear_tf": [True, False],
        "classifier__C": [0.1, 0.3, 1.0, 3.0, 10.0],
        "classifier__class_weight": [None, "balanced"],
    }
]

QUICK_PARAM_GRID = [
    {
        "tfidf__ngram_range": [(1, 1), (1, 2), (1, 3)],
        "tfidf__min_df": [0.001, 0.005],
        "tfidf__max_df": [0.85, 0.95],
        "tfidf__max_features": [None, 10000],
        "tfidf__sublinear_tf": [True],
        "classifier__C": [0.3, 1.0, 3.0],
        "classifier__class_weight": [None, "balanced"],
    }
]


def parameter_product(grid: list[dict]) -> list[dict]:
    try:
        from sklearn.model_selection import ParameterGrid
    except ImportError as exc:
        raise RuntimeError(
            "Hyperparameter tuning requires scikit-learn. "
            "Install project dependencies before running this tuner."
        ) from exc

    return list(ParameterGrid(grid))


def split_pipeline_params(params: dict) -> tuple[dict, dict]:
    tfidf_params = {}
    logreg_params = {}
    for key, value in params.items():
        if key.startswith("tfidf__"):
            tfidf_params[key.removeprefix("tfidf__")] = value
        elif key.startswith("classifier__"):
            logreg_params[key.removeprefix("classifier__")] = value
        else:
            raise ValueError(f"Unexpected parameter name: {key}")
    return tfidf_params, logreg_params


def serializable_params(params: dict) -> dict:
    serializable = {}
    for key, value in params.items():
        if isinstance(value, tuple):
            serializable[key] = list(value)
        else:
            serializable[key] = value
    return serializable


def score_params(train_rows: list[dict[str, str]], validation_rows: list[dict[str, str]], params: dict) -> dict:
    tfidf_params, logreg_params = split_pipeline_params(params)
    model = train_tfidf_logreg(
        train_rows,
        tfidf_params=tfidf_params,
        logreg_params=logreg_params,
    )
    validation_predictions = list(model.predict(texts(validation_rows)))
    validation_metrics = classification_metrics(labels(validation_rows), validation_predictions)
    return {
        "params": params,
        "validation_accuracy": validation_metrics["accuracy"],
        "validation_macro_f1": validation_metrics["macro_f1"],
        "validation_weighted_f1": validation_metrics["weighted_f1"],
    }


def write_tuning_results(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "validation_macro_f1",
        "validation_accuracy",
        "validation_weighted_f1",
        "tfidf__ngram_range",
        "tfidf__min_df",
        "tfidf__max_df",
        "tfidf__max_features",
        "tfidf__sublinear_tf",
        "classifier__C",
        "classifier__class_weight",
        "oversample_min_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            params = serializable_params(result["params"])
            writer.writerow(
                {
                    "rank": rank,
                    "validation_macro_f1": result["validation_macro_f1"],
                    "validation_accuracy": result["validation_accuracy"],
                    "validation_weighted_f1": result["validation_weighted_f1"],
                    "oversample_min_count": result["oversample_min_count"],
                    **params,
                }
            )


def tune(
    grid: list[dict] | None = None,
    output_dir: Path | None = None,
    refit_on_train_validation: bool = True,
    oversample_min_count: int | None = None,
    random_state: int = 42,
) -> dict:
    try:
        import joblib
    except ImportError as exc:
        raise RuntimeError(
            "Hyperparameter tuning artifact writing requires joblib. "
            "Install project dependencies before running this tuner."
        ) from exc

    splits = load_splits()
    selection_train_rows, selection_sampling = oversample_minority_classes(
        splits["train"],
        min_count=oversample_min_count,
        random_state=random_state,
    )
    candidates = parameter_product(grid or QUICK_PARAM_GRID)
    results = []
    for index, params in enumerate(candidates, start=1):
        result = score_params(selection_train_rows, splits["validation"], params)
        result["oversample_min_count"] = oversample_min_count
        results.append(result)
        print(
            f"[{index}/{len(candidates)}] "
            f"validation_macro_f1={result['validation_macro_f1']:.4f} "
            f"params={serializable_params(params)}"
        )

    results = sorted(results, key=lambda item: item["validation_macro_f1"], reverse=True)
    best = results[0]
    best_tfidf_params, best_logreg_params = split_pipeline_params(best["params"])

    natural_final_train_rows = (
        splits["train"] + splits["validation"] if refit_on_train_validation else splits["train"]
    )
    final_train_rows, final_sampling = oversample_minority_classes(
        natural_final_train_rows,
        min_count=oversample_min_count,
        random_state=random_state,
    )
    model = train_tfidf_logreg(
        final_train_rows,
        tfidf_params=best_tfidf_params,
        logreg_params=best_logreg_params,
    )
    output_dir = output_dir or project_root() / "backend" / "model_artifacts" / "tfidf_logreg_tuned"
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "model.joblib")
    evaluation_splits = (
        {"train_validation": natural_final_train_rows, "test": splits["test"]}
        if refit_on_train_validation
        else splits
    )
    split_predictions = {
        split: list(model.predict(texts(rows)))
        for split, rows in evaluation_splits.items()
    }
    metrics = write_evaluation_artifacts(output_dir, evaluation_splits, split_predictions)
    write_tuning_results(output_dir / "tuning_results.csv", results)
    write_json(
        output_dir / "best_params.json",
        {
            "selection_metric": "validation_macro_f1",
            "selected_validation_macro_f1": best["validation_macro_f1"],
            "refit_on_train_validation": refit_on_train_validation,
            "oversample_min_count": oversample_min_count,
            "random_state": random_state,
            "selection_train_sampling": selection_sampling,
            "final_train_sampling": final_sampling,
            "best_params": serializable_params(best["params"]),
        },
    )
    write_json(output_dir / "metrics.json", metrics)
    return {
        "model_path": str(output_dir / "model.joblib"),
        "best_params": serializable_params(best["params"]),
        "selected_validation_macro_f1": best["validation_macro_f1"],
        "metrics": metrics,
        "tuning_results_path": str(output_dir / "tuning_results.csv"),
        "oversample_min_count": oversample_min_count,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Run the larger grid instead of the quick default grid.",
    )
    parser.add_argument(
        "--no-refit-train-validation",
        action="store_true",
        help="Fit the final model on train only after choosing hyperparameters.",
    )
    parser.add_argument(
        "--oversample-min-count",
        type=int,
        default=None,
        help=(
            "Duplicate only training rows so each class has at least this many examples. "
            "Validation and test rows are never oversampled."
        ),
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for reproducible oversampling.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = tune(
        grid=PARAM_GRID if args.full_grid else QUICK_PARAM_GRID,
        refit_on_train_validation=not args.no_refit_train_validation,
        oversample_min_count=args.oversample_min_count,
        random_state=args.random_state,
    )
    print("Best params:", payload["best_params"])
    print("Tuning results:", payload["tuning_results_path"])
    print("Model path:", payload["model_path"])
    print("Selected validation macro F1:", round(payload["selected_validation_macro_f1"], 4))
    print("Final test macro F1:", round(payload["metrics"]["test"]["macro_f1"], 4))


if __name__ == "__main__":
    main()

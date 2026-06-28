"""Fine-tune a BERT/FinBERT classifier on investment-theme labels."""

from __future__ import annotations

import argparse
import inspect
from dataclasses import asdict, dataclass
from pathlib import Path

from src.evaluation.metrics import classification_metrics
from src.evaluation.reports import write_evaluation_artifacts
from src.models.data import labels, load_splits, project_root, texts
from src.models.io import write_json


DEFAULT_MODEL_NAME = "ProsusAI/finbert"


@dataclass
class BertTrainingConfig:
    model_name: str = DEFAULT_MODEL_NAME
    max_length: int = 256
    num_train_epochs: float = 3.0
    train_batch_size: int = 8
    eval_batch_size: int = 16
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    random_seed: int = 42
    limit_rows: int | None = None


class TextClassificationDataset:
    def __init__(
        self,
        rows: list[dict[str, str]],
        tokenizer,
        label_to_id: dict[str, int],
        max_length: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.label_to_id = label_to_id
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        encoded = self.tokenizer(
            row["text"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        item = {key: value.squeeze(0) for key, value in encoded.items()}
        item["labels"] = self.label_to_id[row["label"]]
        return item


def require_transformers():
    try:
        import numpy as np
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "BERT training requires torch, transformers, and accelerate. "
            "Install project dependencies with `.conda/bin/pip install -r requirements.txt`."
        ) from exc

    return {
        "np": np,
        "torch": torch,
        "AutoModelForSequenceClassification": AutoModelForSequenceClassification,
        "AutoTokenizer": AutoTokenizer,
        "Trainer": Trainer,
        "TrainingArguments": TrainingArguments,
        "set_seed": set_seed,
    }


def build_label_maps(split_rows: dict[str, list[dict[str, str]]]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    label_names = sorted(
        {
            label
            for rows in split_rows.values()
            for label in labels(rows)
        }
    )
    label_to_id = {label: index for index, label in enumerate(label_names)}
    id_to_label = {index: label for label, index in label_to_id.items()}
    return label_names, label_to_id, id_to_label


def maybe_limit_rows(
    splits: dict[str, list[dict[str, str]]],
    limit_rows: int | None,
) -> dict[str, list[dict[str, str]]]:
    if not limit_rows:
        return splits
    return {split: rows[:limit_rows] for split, rows in splits.items()}


def build_training_arguments(training_args_cls, output_dir: Path, config: BertTrainingConfig):
    kwargs = {
        "output_dir": str(output_dir / "trainer"),
        "num_train_epochs": config.num_train_epochs,
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "seed": config.random_seed,
        "logging_strategy": "epoch",
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "report_to": "none",
    }
    try:
        return training_args_cls(evaluation_strategy="epoch", **kwargs)
    except TypeError:
        return training_args_cls(eval_strategy="epoch", **kwargs)


def build_trainer(trainer_cls, tokenizer, **kwargs):
    trainer_signature = inspect.signature(trainer_cls.__init__)
    if "processing_class" in trainer_signature.parameters:
        return trainer_cls(processing_class=tokenizer, **kwargs)
    return trainer_cls(tokenizer=tokenizer, **kwargs)


def train_bert_classifier(
    config: BertTrainingConfig | None = None,
    output_dir: Path | None = None,
) -> dict:
    config = config or BertTrainingConfig()
    deps = require_transformers()
    np = deps["np"]
    auto_model_cls = deps["AutoModelForSequenceClassification"]
    auto_tokenizer_cls = deps["AutoTokenizer"]
    trainer_cls = deps["Trainer"]
    training_args_cls = deps["TrainingArguments"]
    deps["set_seed"](config.random_seed)

    output_dir = output_dir or project_root() / "backend" / "model_artifacts" / "bert_classifier"
    output_dir.mkdir(parents=True, exist_ok=True)

    splits = maybe_limit_rows(load_splits(), config.limit_rows)
    label_names, label_to_id, id_to_label = build_label_maps(splits)
    tokenizer = auto_tokenizer_cls.from_pretrained(config.model_name)
    model = auto_model_cls.from_pretrained(
        config.model_name,
        num_labels=len(label_names),
        id2label={index: label for index, label in id_to_label.items()},
        label2id=label_to_id,
        ignore_mismatched_sizes=True,
    )

    train_dataset = TextClassificationDataset(
        splits["train"],
        tokenizer,
        label_to_id,
        config.max_length,
    )
    validation_dataset = TextClassificationDataset(
        splits["validation"],
        tokenizer,
        label_to_id,
        config.max_length,
    )

    def compute_metrics(eval_prediction) -> dict[str, float]:
        logits, label_ids = eval_prediction
        pred_ids = np.argmax(logits, axis=-1)
        y_true = [id_to_label[int(label_id)] for label_id in label_ids]
        y_pred = [id_to_label[int(pred_id)] for pred_id in pred_ids]
        metrics = classification_metrics(y_true, y_pred)
        return {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
        }

    trainer = build_trainer(
        trainer_cls,
        tokenizer,
        model=model,
        args=build_training_arguments(training_args_cls, output_dir, config),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_metrics=compute_metrics,
    )
    trainer.train()

    split_predictions = {}
    for split_name, rows in splits.items():
        dataset = TextClassificationDataset(rows, tokenizer, label_to_id, config.max_length)
        prediction_output = trainer.predict(dataset)
        pred_ids = np.argmax(prediction_output.predictions, axis=-1)
        split_predictions[split_name] = [id_to_label[int(pred_id)] for pred_id in pred_ids]

    metrics = write_evaluation_artifacts(output_dir, splits, split_predictions)
    trainer.save_model(output_dir / "model")
    tokenizer.save_pretrained(output_dir / "model")
    write_json(
        output_dir / "label_mapping.json",
        {
            "labels": label_names,
            "label_to_id": label_to_id,
            "id_to_label": {str(index): label for index, label in id_to_label.items()},
        },
    )
    write_json(output_dir / "training_config.json", asdict(config))
    return {
        "model_dir": str(output_dir / "model"),
        "metrics": metrics,
        "label_mapping": label_to_id,
        "training_config": asdict(config),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--limit-rows",
        type=int,
        default=None,
        help="Use only the first N rows per split for a quick smoke test.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = train_bert_classifier(
        BertTrainingConfig(
            model_name=args.model_name,
            max_length=args.max_length,
            num_train_epochs=args.epochs,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            learning_rate=args.learning_rate,
            limit_rows=args.limit_rows,
        )
    )
    print("Model dir:", payload["model_dir"])
    print("Validation macro F1:", round(payload["metrics"]["validation"]["macro_f1"], 4))
    print("Test macro F1:", round(payload["metrics"]["test"]["macro_f1"], 4))


if __name__ == "__main__":
    main()

"""Create train/validation/test splits from manually reviewed labels."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


MODEL_COLUMNS = [
    "chunk_id",
    "ticker",
    "form",
    "filing_date",
    "section",
    "text",
    "manual_label",
    "secondary_label",
    "source_path",
]


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    model_df = df[df["review_status"] == "reviewed"].copy()
    model_df = model_df[model_df["manual_label"].notna()]
    model_df = model_df[model_df["text"].notna()]
    model_df = model_df[MODEL_COLUMNS].rename(columns={"manual_label": "label"})
    return model_df


def create_splits(
    df: pd.DataFrame,
    test_size: float = 0.30,
    validation_fraction_of_temp: float = 0.50,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model_df = prepare_model_frame(df)
    train_df, temp_df = train_test_split(
        model_df,
        test_size=test_size,
        stratify=model_df["label"],
        random_state=random_state,
    )
    validation_df, test_df = train_test_split(
        temp_df,
        test_size=validation_fraction_of_temp,
        stratify=temp_df["label"],
        random_state=random_state,
    )
    return train_df, validation_df, test_df


def write_splits(
    input_path: Path,
    output_dir: Path,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(input_path)
    train_df, validation_df, test_df = create_splits(df, random_state=random_state)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train.csv", index=False)
    validation_df.to_csv(output_dir / "validation.csv", index=False)
    test_df.to_csv(output_dir / "test.csv", index=False)
    return train_df, validation_df, test_df


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    train_df, validation_df, test_df = write_splits(
        root / "data" / "labeled" / "manual_review_taxonomy_final.csv",
        root / "data" / "splits",
    )
    print("Train:", train_df.shape)
    print("Validation:", validation_df.shape)
    print("Test:", test_df.shape)


if __name__ == "__main__":
    main()


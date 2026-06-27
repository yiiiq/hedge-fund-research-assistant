"""Keyword-based weak labeling for SEC text chunks."""

from __future__ import annotations

import pandas as pd

from src.labeling.taxonomy import WEAK_LABEL_KEYWORDS


def weak_label(text: str, keyword_map: dict[str, list[str]] | None = None) -> tuple[str, int]:
    keyword_map = keyword_map or WEAK_LABEL_KEYWORDS
    text_l = f" {(text or '').lower()} "
    scores = {
        label: sum(keyword.lower() in text_l for keyword in keywords)
        for label, keywords in keyword_map.items()
    }
    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]
    if best_score == 0:
        return "Neutral / Other", 0
    return best_label, best_score


def add_weak_labels(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    labeled = df.copy()
    labeled[["weak_label", "weak_label_score"]] = labeled[text_col].apply(
        lambda text: pd.Series(weak_label(text))
    )
    return labeled


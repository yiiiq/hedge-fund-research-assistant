"""Run all baseline model trainers."""

from __future__ import annotations

from src.models import keyword_baseline, majority_baseline, tfidf_logreg


def main() -> None:
    """Run all non-deep-learning baseline trainers."""
    majority_baseline.main()
    keyword_baseline.main()
    tfidf_logreg.main()


if __name__ == "__main__":
    main()

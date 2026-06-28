# Hedge Fund Research Assistant

This project develops a hedge fund research assistant that uses NLP to classify financial text passages into analyst-relevant investment themes and combines those predictions with vector-based retrieval to support evidence-based bull/bear thesis generation.

The system is evaluated using:

- A naive baseline
- A classical TF-IDF logistic regression model
- A deep learning FinBERT model

The final application provides a public, interactive analyst dashboard where users can search company documents, inspect classified evidence, and generate cited investment research summaries.

## Project Structure

- `data/`: Raw, processed, labeled, and split datasets
- `notebooks/`: Exploration, modeling, training, and error analysis notebooks
- `src/`: Core research pipeline modules
- `backend/`: API routes, services, and model artifacts
- `frontend/`: Analyst dashboard source, components, and pages
- `experiments/`: Experiment configuration and results
- `report/`: Final report, figures, and pitch slides
- `deployment/`: Container and deployment files

## Baseline Training

Install dependencies into the local environment:

```bash
.conda/bin/pip install -r requirements.txt
```

Run the baselines:

```bash
.conda/bin/python -m src.models.majority_baseline
.conda/bin/python -m src.models.keyword_baseline
.conda/bin/python -m src.models.tfidf_logreg
```

Tune TF-IDF logistic regression hyperparameters:

```bash
.conda/bin/python -m src.models.tune_tfidf_logreg
```

Fine-tune FinBERT/BERT:

```bash
.conda/bin/pip install -r requirements.txt
.conda/bin/python -m src.models.bert_classifier
```

For a quick smoke test before a full run:

```bash
.conda/bin/python -m src.models.bert_classifier --limit-rows 16 --epochs 1
```

Artifacts are written under `backend/model_artifacts/`.

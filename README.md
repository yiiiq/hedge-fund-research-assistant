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

## Interactive Filing Highlighter

The deployed proof-of-concept app is **SECtion Finder**, a Streamlit interface designed for Hugging Face Spaces.
It runs inference only with the trained TF-IDF logistic regression artifact at
`backend/model_artifacts/tfidf_logreg/model.joblib`.

Run locally:

```bash
streamlit run app.py
```

The app supports:

- Searching SEC filings by ticker
- Selecting recent 10-K, 10-Q, and 8-K filings
- Downloading SEC filing HTML and classifying extracted visible text
- Viewing highlighted SEC passages with confidence and source-link details
- Uploading `.pdf` SEC filings
- Running whole-document PDF analysis by default
- Viewing original PDF pages with topic-colored highlights
- Filtering color-coded highlights by investment theme

For Hugging Face Spaces, create a Streamlit Space and include this repository with
`app.py`, `requirements.txt`, `SECtionFinderLogo.png`, `src/`, and the model
artifact. Spaces will install dependencies and launch the Streamlit app
automatically.

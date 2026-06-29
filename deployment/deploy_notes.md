# Deployment Notes

The proof-of-concept app is designed for Hugging Face Spaces with Streamlit.

## Local Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads `backend/model_artifacts/tfidf_logreg/model.joblib` at startup and
performs inference only. The first tab searches SEC filings by ticker through
the helpers in `src/labeling/sec_exploration.py`; the second tab keeps the PDF
upload workflow.

## Hugging Face Spaces

1. Create a new Space with the Streamlit SDK.
2. Push this repository or upload the required files.
3. Confirm these files are present in the Space:
   - `app.py`
   - `requirements.txt`
   - `SECtionFinderLogo.png`
   - `backend/model_artifacts/tfidf_logreg/model.joblib`
   - `src/`
4. Spaces will install dependencies and run `app.py` automatically.

The deployed app should remain live through grading and should be checked after
each push to confirm the model artifact loads successfully.

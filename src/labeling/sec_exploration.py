"""SEC filing collection, section extraction, and chunking helpers."""

from __future__ import annotations

import re
import time
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

DEFAULT_TICKERS = ["MU", "MRVL", "NBIS"]
DEFAULT_HEADERS = {
    "User-Agent": "yiqian academic project yiqian1999@outlook.com",
}

TARGET_FORMS = {
    "MU": ["10-K", "10-Q", "8-K"],
    "MRVL": ["10-K", "10-Q", "8-K"],
    "NBIS": ["20-F", "6-K"],
}

SECTION_RULES = {
    "risk_factors": {
        "start": [r"item\s+1a\.?\s+risk\s+factors", r"item\s+3d\.?\s+risk\s+factors"],
        "end": [r"item\s+1b\.?", r"item\s+2\.?", r"item\s+4\.?"],
    },
    "mda": {
        "start": [r"item\s+7\.?\s+management", r"item\s+2\.?\s+management", r"item\s+5\.?\s+operating"],
        "end": [r"item\s+7a\.?", r"item\s+3\.?", r"item\s+6\.?"],
    },
    "market_risk": {
        "start": [r"item\s+7a\.?\s+quantitative", r"item\s+3\.?\s+quantitative"],
        "end": [r"item\s+8\.?", r"item\s+4\.?"],
    },
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_ticker_cik_map(headers: dict[str, str] | None = None) -> pd.DataFrame:
    url = "https://www.sec.gov/files/company_tickers.json"
    response = requests.get(url, headers=headers or DEFAULT_HEADERS, timeout=30)
    response.raise_for_status()
    data = response.json()
    rows = [
        {
            "ticker": item["ticker"].upper(),
            "title": item["title"],
            "cik": str(item["cik_str"]).zfill(10),
        }
        for item in data.values()
    ]
    return pd.DataFrame(rows)


def get_submissions(cik: str, headers: dict[str, str] | None = None) -> dict:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    response = requests.get(url, headers=headers or DEFAULT_HEADERS, timeout=30)
    response.raise_for_status()
    return response.json()


def recent_filings_for_ticker(
    ticker: str,
    ticker_map: pd.DataFrame,
    headers: dict[str, str] | None = None,
) -> pd.DataFrame:
    cik = ticker_map.loc[ticker_map["ticker"] == ticker, "cik"].iloc[0]
    submissions = get_submissions(cik, headers=headers)
    recent = pd.DataFrame(submissions["filings"]["recent"])
    recent["ticker"] = ticker
    recent["cik"] = cik
    return recent


def collect_recent_filings(
    tickers: list[str] | None = None,
    target_forms: dict[str, list[str]] | None = None,
    headers: dict[str, str] | None = None,
) -> pd.DataFrame:
    tickers = tickers or DEFAULT_TICKERS
    target_forms = target_forms or TARGET_FORMS
    ticker_map = load_ticker_cik_map(headers=headers)
    filings = pd.concat(
        [recent_filings_for_ticker(ticker, ticker_map, headers=headers) for ticker in tickers],
        ignore_index=True,
    )
    return pd.concat(
        [
            filings[(filings["ticker"] == ticker) & (filings["form"].isin(forms))]
            for ticker, forms in target_forms.items()
        ],
        ignore_index=True,
    )


def filing_url(row: pd.Series) -> str:
    cik_int = str(int(row["cik"]))
    accession = row["accessionNumber"].replace("-", "")
    doc = row["primaryDocument"]
    return f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession}/{doc}"


def download_filing(
    row: pd.Series,
    raw_dir: Path | None = None,
    headers: dict[str, str] | None = None,
    sleep_seconds: float = 0.2,
) -> Path:
    raw_dir = raw_dir or project_root() / "data" / "raw"
    out_dir = raw_dir / "sec" / row["ticker"]
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / f"{row['filingDate']}_{row['form']}_{row['accessionNumber']}.html"
    if path.exists():
        return path

    response = requests.get(filing_url(row), headers=headers or DEFAULT_HEADERS, timeout=60)
    response.raise_for_status()
    path.write_text(response.text, encoding="utf-8")
    time.sleep(sleep_seconds)
    return path


def html_to_text(path: str | Path) -> str:
    html = Path(path).read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "table"]):
        tag.decompose()
    text = soup.get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def extract_between(text: str, start_patterns: list[str], end_patterns: list[str]) -> str | None:
    lower = text.lower()
    starts = [match.start() for pattern in start_patterns if (match := re.search(pattern, lower, flags=re.I))]
    if not starts:
        return None

    start = min(starts)
    tail = lower[start:]
    ends = [
        start + match.start()
        for pattern in end_patterns
        if (match := re.search(pattern, tail, flags=re.I)) and match.start() > 500
    ]
    end = min(ends) if ends else min(len(text), start + 120_000)
    return text[start:end].strip()


def extract_sections(text: str, section_rules: dict | None = None) -> dict[str, str | None]:
    section_rules = section_rules or SECTION_RULES
    return {
        name: extract_between(text, rules["start"], rules["end"])
        for name, rules in section_rules.items()
    }


def chunk_words(text: str | None, chunk_size: int = 220, overlap: int = 40, min_words: int = 80) -> list[str]:
    if not text:
        return []
    words = text.split()
    step = chunk_size - overlap
    return [
        " ".join(words[i : i + chunk_size])
        for i in range(0, len(words), step)
        if len(words[i : i + chunk_size]) >= min_words
    ]


def build_sec_chunks(filings: pd.DataFrame, output_path: Path | None = None) -> pd.DataFrame:
    records = []
    for _, row in filings.iterrows():
        text = row.get("text") or html_to_text(row["local_path"])
        for section_name, section_text in extract_sections(text).items():
            for index, chunk in enumerate(chunk_words(section_text)):
                records.append(
                    {
                        "chunk_id": f"{row['ticker']}_{row['form']}_{row['filingDate']}_{section_name}_{index:04d}",
                        "ticker": row["ticker"],
                        "form": row["form"],
                        "filing_date": row["filingDate"],
                        "section": section_name,
                        "source_path": str(row["local_path"]),
                        "text": chunk,
                    }
                )
    chunks = pd.DataFrame(records)
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        chunks.to_csv(output_path, index=False)
    return chunks


"""Streamlit app for classifying uploaded SEC filing PDFs by investment theme."""

from __future__ import annotations

import base64
import html
import io
import re
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "backend" / "model_artifacts" / "tfidf_logreg" / "model.joblib"
LOGO_PATH = PROJECT_ROOT / "SECtionFinderLogo.png"
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_CHUNKS = 450
MIN_CHUNK_CHARS = 350
MAX_CHUNK_CHARS = 1800
PDF_RENDER_ZOOM = 1.5
LOW_CONFIDENCE_THRESHOLD = 0.45

LABELS = [
    "Regulation / Legal",
    "Capital Allocation / CAPEX",
    "Macro Risk",
    "AI / Product Strategy",
    "Margins / Profitability",
    "Competition",
    "Demand Growth",
    "Neutral / Other",
]

LABEL_COLORS = {
    "Regulation / Legal": "#0b60e7",
    "Capital Allocation / CAPEX": "#19c4b4",
    "Macro Risk": "#dc2626",
    "AI / Product Strategy": "#5a6cf4",
    "Margins / Profitability": "#16a34a",
    "Competition": "#f97316",
    "Demand Growth": "#7c3aed",
    "Neutral / Other": "#64748b",
}


@st.cache_resource(show_spinner=False)
def load_model() -> Any:
    """Load the trained classifier artifact once."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_PATH}. Run python3 -m src.models.tfidf_logreg first."
        )
    return joblib.load(MODEL_PATH)


def clean_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph breaks."""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def validate_pdf(path: Path) -> None:
    """Validate an uploaded PDF before processing."""
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Uploaded file is larger than the 25 MB limit for this demo.")
    if path.suffix.lower() != ".pdf":
        raise ValueError("Unsupported file type. Upload a PDF filing.")


def extract_pdf_blocks(path: Path) -> list[dict[str, Any]]:
    """Extract text blocks and PDF coordinates from each page."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PDF highlighting requires pymupdf. Install dependencies from requirements.txt.") from exc

    blocks = []
    with fitz.open(path) as document:
        for page_index, page in enumerate(document):
            for block in page.get_text("blocks"):
                x0, y0, x1, y1, text = block[:5]
                text = clean_text(str(text))
                if len(text) < 20:
                    continue
                blocks.append(
                    {
                        "page": page_index,
                        "bbox": (float(x0), float(y0), float(x1), float(y1)),
                        "text": text,
                    }
                )
    return blocks


def chunk_pdf_blocks(blocks: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Group PDF text blocks into model-sized chunks while keeping coordinates."""
    chunks = []
    current_blocks = []
    current_text = ""
    messages = []

    for block in blocks:
        block_text = block["text"]
        candidate = f"{current_text}\n\n{block_text}".strip()
        if len(candidate) <= MAX_CHUNK_CHARS:
            current_blocks.append(block)
            current_text = candidate
            if len(current_text) >= MIN_CHUNK_CHARS:
                chunks.append({"text": current_text, "blocks": current_blocks})
                current_blocks = []
                current_text = ""
            continue

        if current_blocks:
            chunks.append({"text": current_text, "blocks": current_blocks})
        chunks.append({"text": block_text[:MAX_CHUNK_CHARS], "blocks": [block]})
        current_blocks = []
        current_text = ""

    if current_blocks:
        chunks.append({"text": current_text, "blocks": current_blocks})

    if len(chunks) > MAX_CHUNKS:
        messages.append(f"Document produced {len(chunks)} chunks; showing the first {MAX_CHUNKS} for demo speed.")
        chunks = chunks[:MAX_CHUNKS]
    return chunks, messages


def predict_chunks(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Classify chunks and attach confidence scores when available."""
    model = load_model()
    texts = [chunk["text"] for chunk in chunks]
    labels = list(model.predict(texts))
    probabilities = model.predict_proba(texts) if hasattr(model, "predict_proba") else None
    classes = list(getattr(model, "classes_", []))

    predictions = []
    for index, (chunk, label) in enumerate(zip(chunks, labels, strict=True)):
        confidence = None
        if probabilities is not None and label in classes:
            confidence = float(probabilities[index][classes.index(label)])
        predictions.append(
            {
                "index": index + 1,
                "text": chunk["text"],
                "blocks": chunk["blocks"],
                "label": label,
                "confidence": confidence,
            }
        )
    return predictions


def image_to_data_uri(image: Image.Image) -> str:
    """Encode a page image for inline HTML display."""
    output = io.BytesIO()
    image.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@st.cache_data(show_spinner=False)
def logo_data_uri() -> str:
    """Crop logo whitespace and encode it for the custom header."""
    if not LOGO_PATH.exists():
        return ""
    image = Image.open(LOGO_PATH).convert("RGBA")
    background = Image.new("RGBA", image.size, (255, 255, 255, 255))
    diff = Image.composite(image, background, image).convert("RGB")
    mask = diff.point(lambda value: 0 if value > 246 else 255).convert("L")
    bbox = mask.getbbox()
    if bbox:
        pad = 24
        left = max(bbox[0] - pad, 0)
        top = max(bbox[1] - pad, 0)
        right = min(bbox[2] + pad, image.width)
        bottom = min(bbox[3] + pad, image.height)
        image = image.crop((left, top, right, bottom))
    return image_to_data_uri(image)


@st.cache_data(show_spinner=False)
def render_pdf_page_images(pdf_path: str) -> list[dict[str, Any]]:
    """Render PDF pages once so topic filter toggles do not re-rasterize the document."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError("PDF highlighting requires pymupdf. Install dependencies from requirements.txt.") from exc

    rendered_pages = []
    matrix = fitz.Matrix(PDF_RENDER_ZOOM, PDF_RENDER_ZOOM)
    with fitz.open(pdf_path) as document:
        for page_index, page in enumerate(document):
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            image = Image.open(io.BytesIO(pixmap.tobytes("png"))).convert("RGB")
            rendered_pages.append(
                {
                    "page_index": page_index,
                    "width": image.width,
                    "height": image.height,
                    "data_uri": image_to_data_uri(image),
                }
            )
    return rendered_pages


def overlay_opacity(confidence: float | None) -> float:
    """Choose highlight intensity from confidence."""
    if confidence is not None and confidence < LOW_CONFIDENCE_THRESHOLD:
        return 0.16
    return 0.30


def render_review_component(
    pdf_path: str,
    predictions: list[dict[str, Any]],
    active_labels: list[str],
) -> str:
    """Render PDF pages and clickable detail drawer as one HTML component."""
    active = set(active_labels)
    overlays_by_page: dict[int, list[dict[str, Any]]] = {}
    for prediction in predictions:
        if prediction["label"] not in active:
            continue
        for block in prediction["blocks"]:
            overlays_by_page.setdefault(block["page"], []).append(
                {
                    "bbox": block["bbox"],
                    "index": prediction["index"],
                    "label": prediction["label"],
                    "confidence": prediction["confidence"],
                }
            )

    if not overlays_by_page:
        return "<div class='empty-state'>No PDF regions match the selected topic filters.</div>"

    detail_cards = []
    for prediction in predictions:
        if prediction["label"] not in active:
            continue
        color = LABEL_COLORS.get(prediction["label"], "#64748b")
        confidence = prediction["confidence"]
        confidence_text = "n/a" if confidence is None else f"{confidence:.0%}"
        detail_cards.append(
            f"""
            <article class="detail-card" id="detail-{prediction["index"]}">
                <div class="detail-topic" style="background:{color};">{html.escape(prediction["label"])}</div>
                <dl class="detail-metadata">
                    <div><dt>Confidence</dt><dd>{confidence_text}</dd></div>
                    <div><dt>Chunk</dt><dd>{prediction["index"]}</dd></div>
                </dl>
                <h3>Extracted text</h3>
                <p>{html.escape(prediction["text"])}</p>
            </article>
            """
        )

    pages = []
    for page_image in render_pdf_page_images(pdf_path):
        page_index = page_image["page_index"]
        overlays = []
        for item in overlays_by_page.get(page_index, []):
            x0, y0, x1, y1 = item["bbox"]
            color = LABEL_COLORS.get(item["label"], "#64748b")
            confidence = item["confidence"]
            confidence_text = "n/a" if confidence is None else f"{confidence:.0%}"
            left_pct = (x0 * PDF_RENDER_ZOOM / page_image["width"]) * 100
            top_pct = (y0 * PDF_RENDER_ZOOM / page_image["height"]) * 100
            width_pct = max(((x1 - x0) * PDF_RENDER_ZOOM / page_image["width"]) * 100, 0.5)
            height_pct = max(((y1 - y0) * PDF_RENDER_ZOOM / page_image["height"]) * 100, 0.5)
            overlays.append(
                f"""
                <button
                    type="button"
                    class="pdf-highlight"
                    style="left:{left_pct:.4f}%; top:{top_pct:.4f}%; width:{width_pct:.4f}%; height:{height_pct:.4f}%; --topic-color:{color}; --topic-alpha:{overlay_opacity(confidence):.2f};"
                    data-target="detail-{item["index"]}"
                    title="{html.escape(item["label"], quote=True)} | Confidence {html.escape(confidence_text, quote=True)}"
                    aria-label="View {html.escape(item["label"], quote=True)} details"></button>
                """
            )
        pages.append(
            f"""
            <section class="pdf-page">
                <div class="page-label">Page {page_index + 1}</div>
                <div class="pdf-page-canvas">
                    <img src="{page_image["data_uri"]}" alt="PDF page {page_index + 1}">
                    {''.join(overlays)}
                </div>
            </section>
            """
        )

    return f"""
    <style>
    {COMPONENT_CSS}
    </style>
    <div class="review-note">Click a section to view details.</div>
    <div class="document-review">
        <div class="pdf-view">{''.join(pages)}</div>
        <aside class="detail-drawer" id="detail-drawer">
            <div class="detail-empty">Click a highlighted section to inspect the model prediction.</div>
            {''.join(detail_cards)}
        </aside>
    </div>
    <script>
    const root = document.currentScript.parentElement;
    const drawer = root.querySelector("#detail-drawer");
    root.querySelectorAll(".pdf-highlight").forEach((button) => {{
        button.addEventListener("click", () => {{
            root.querySelectorAll(".pdf-highlight.is-selected").forEach((node) => node.classList.remove("is-selected"));
            root.querySelectorAll(".detail-card").forEach((card) => card.classList.remove("is-visible"));
            const target = root.querySelector("#" + button.dataset.target);
            if (target) {{
                target.classList.add("is-visible");
                drawer.querySelector(".detail-empty").style.display = "none";
                button.classList.add("is-selected");
            }}
        }});
    }});
    </script>
    """


def analyze_pdf(uploaded_file: Any) -> tuple[str, list[dict[str, Any]], list[str]]:
    """Persist an uploaded PDF temporarily, extract text, and classify it."""
    suffix = Path(uploaded_file.name).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
        handle.write(uploaded_file.getbuffer())
        pdf_path = Path(handle.name)

    validate_pdf(pdf_path)
    blocks = extract_pdf_blocks(pdf_path)
    text = "\n\n".join(block["text"] for block in blocks)
    if len(text) < 100:
        raise ValueError("The selected document text is too short to classify.")
    chunks, messages = chunk_pdf_blocks(blocks)
    if not chunks:
        raise ValueError("No classifiable text chunks were found.")
    predictions = predict_chunks(chunks)
    return str(pdf_path), predictions, messages


def render_summary_grid(predictions: list[dict[str, Any]]) -> None:
    """Render topic counts with native Streamlit layout primitives."""
    counts = Counter(prediction["label"] for prediction in predictions)
    with st.container(border=True):
        header_topic, header_count = st.columns([5, 1])
        header_topic.markdown("<div class='summary-native-header'>Topic</div>", unsafe_allow_html=True)
        header_count.markdown("<div class='summary-native-header count'>Passages</div>", unsafe_allow_html=True)
        for label in LABELS:
            topic_col, count_col = st.columns([5, 1])
            color = LABEL_COLORS[label]
            topic_col.markdown(
                f"""
                <div class="summary-native-topic">
                    <span class="summary-dot" style="background:{color};"></span>
                    <span>{html.escape(label)}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            count_col.markdown(
                f"<div class='summary-native-count'>{counts.get(label, 0)}</div>",
                unsafe_allow_html=True,
            )


def render_topic_filters() -> list[str]:
    """Render checkbox topic filters and return active labels."""
    st.markdown("<div class='filter-title'>Visible topic highlights</div>", unsafe_allow_html=True)
    selected = []
    columns = st.columns(2)
    for index, label in enumerate(LABELS):
        filter_key = f"topic_filter_{label}"
        if filter_key not in st.session_state:
            st.session_state[filter_key] = True
        color = LABEL_COLORS[label]
        with columns[index % 2]:
            dot_col, checkbox_col = st.columns([0.12, 0.88])
            dot_col.markdown(
                f"<span class='filter-dot' style='background:{color};'></span>",
                unsafe_allow_html=True,
            )
            with checkbox_col:
                if st.checkbox(label, key=filter_key, label_visibility="visible"):
                    selected.append(label)
    return selected


def inject_styles() -> None:
    """Apply fintech product styling."""
    st.markdown(
        f"""
        <style>
        {APP_CSS}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Render app header."""
    logo_uri = logo_data_uri()
    logo_html = f'<img src="{logo_uri}" alt="SECtion Finder logo">' if logo_uri else ""
    st.markdown(
        f"""
        <div class="app-title">
            <div class="brand-lockup">
                <div class="brand-logo">{logo_html}</div>
                <div class="brand-copy">
                    <div class="eyebrow">10-K / 10-Q topic intelligence</div>
                    <h1>SECtion Finder</h1>
                    <p>Upload a filing PDF and review model-tagged investment themes directly on the source document.</p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_loading() -> None:
    """Render a custom loading indicator while Streamlit runs analysis."""
    st.markdown(
        """
        <div class="loading-card">
            <div class="ticker-loader">
                <div class="ticker-line"></div>
                <div class="ticker-dot"></div>
            </div>
            <h3>Analyzing filing</h3>
            <p>Parsing the PDF, classifying passages, and rendering topic highlights.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="SECtion Finder",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_styles()
    render_header()

    uploaded_file = st.file_uploader("Upload filing PDF", type=["pdf"], label_visibility="visible")
    analyze = st.button("Analyze Filing", type="primary", disabled=uploaded_file is None)

    if uploaded_file is None and "predictions" not in st.session_state:
        st.markdown(
            "<div class='empty-state'>Upload a filing PDF to begin whole-document theme classification.</div>",
            unsafe_allow_html=True,
        )
        return

    if analyze and uploaded_file is not None:
        loading_slot = st.empty()
        with loading_slot:
            render_loading()
        try:
            pdf_path, predictions, messages = analyze_pdf(uploaded_file)
        except Exception as exc:
            loading_slot.empty()
            st.error(str(exc))
            return
        loading_slot.empty()
        st.session_state.pdf_path = pdf_path
        st.session_state.predictions = predictions
        st.session_state.messages = messages
        for label in LABELS:
            st.session_state[f"topic_filter_{label}"] = True

    if "predictions" not in st.session_state:
        return

    predictions = st.session_state.predictions
    messages = st.session_state.get("messages", [])
    st.markdown("### Classification Summary")
    st.caption(f"{len(predictions)} passages classified with the TF-IDF logistic regression model.")
    if messages:
        for message in messages:
            st.warning(message)

    render_summary_grid(predictions)

    active_labels = render_topic_filters()

    review_html = render_review_component(st.session_state.pdf_path, predictions, active_labels)
    components.html(review_html, height=900, scrolling=True)


APP_CSS = """
.stApp {
    background: #f7fbff;
    color: #082b63;
    font-family: "Aptos", "Inter", "IBM Plex Sans", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.block-container {
    max-width: 1380px;
    padding-top: 1.4rem;
}
header[data-testid="stHeader"] {
    background: #f7fbff !important;
    border-bottom: 1px solid #dbe8f8 !important;
    box-shadow: none !important;
}
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"] {
    background: #f7fbff !important;
    color: #082b63 !important;
}
.app-title {
    background:
        linear-gradient(135deg, rgba(11, 96, 231, 0.10), rgba(25, 196, 180, 0.08)),
        #ffffff;
    border: 1px solid #dbe8f8;
    border-radius: 8px;
    box-shadow: 0 14px 34px rgba(8, 43, 99, 0.10);
    margin-bottom: 1.2rem;
    overflow: hidden;
    padding: 1.35rem 1.55rem;
}
.brand-lockup {
    align-items: center;
    display: flex;
    gap: 1.25rem;
}
.brand-logo {
    align-items: center;
    background: #ffffff;
    border: 1px solid #d6e5fb;
    border-radius: 8px;
    box-shadow: 0 10px 24px rgba(8, 43, 99, 0.10);
    display: flex;
    flex: 0 0 auto;
    height: 94px;
    justify-content: center;
    padding: 0.55rem 0.75rem;
    width: 210px;
}
.brand-logo img {
    display: block;
    height: auto;
    max-height: 82px;
    max-width: 100%;
}
.brand-copy {
    min-width: 0;
}
.eyebrow {
    color: #19c4b4;
    font-size: 0.78rem;
    font-weight: 800;
    letter-spacing: 0.08em;
    margin-bottom: 0.22rem;
    text-transform: uppercase;
}
.app-title h1 {
    color: #082b63;
    font-size: 2.05rem;
    font-weight: 820;
    letter-spacing: 0;
    line-height: 1.05;
    margin: 0 0 0.32rem;
}
.app-title p {
    color: #315f92;
    font-size: 0.96rem;
    margin: 0;
}
[data-testid="stFileUploader"] {
    background: #ffffff;
    border: 1px solid #dbe8f8;
    border-radius: 8px;
    box-shadow: 0 10px 28px rgba(8, 43, 99, 0.08);
    padding: 1rem;
}
[data-testid="stFileUploaderDropzone"] {
    background: #f0f7ff !important;
    border: 1px dashed #9ebff1 !important;
    color: #082b63 !important;
}
[data-testid="stFileUploaderDropzone"] * {
    color: #082b63 !important;
}
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stBaseButton-secondary"] {
    background: #e9f2ff !important;
    border: 1px solid #9ebff1 !important;
    color: #082b63 !important;
}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploader"] section div,
[data-testid="stFileUploader"] ul,
[data-testid="stFileUploader"] li {
    background: #f0f7ff !important;
    color: #082b63 !important;
}
[data-testid="stFileUploader"] span,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] p,
[data-testid="stFileUploader"] label {
    color: #082b63 !important;
}
[data-testid="stFileUploader"] svg {
    color: #0b60e7 !important;
    fill: #0b60e7 !important;
}
[data-testid="stFileUploader"] button[kind="secondary"],
[data-testid="stFileUploader"] button[title],
[data-testid="stFileUploader"] button[aria-label] {
    background: #e9f2ff !important;
    border-color: #9ebff1 !important;
    color: #0b60e7 !important;
}
.filter-title {
    color: #082b63;
    font-size: 0.9rem;
    font-weight: 760;
    margin: 1rem 0 0.35rem;
}
.filter-dot {
    border-radius: 999px;
    display: inline-block;
    height: 0.72rem;
    margin-top: 0.58rem;
    width: 0.72rem;
}
.stButton > button {
    background: #0b60e7;
    border: 1px solid #0b60e7;
    border-radius: 6px;
    color: #ffffff;
    font-weight: 750;
}
.stButton > button:hover {
    background: #082b63;
    border-color: #082b63;
    color: #ffffff;
}
.summary-native-header {
    color: #0b60e7 !important;
    font-size: 0.76rem;
    font-weight: 760;
    padding-bottom: 0.35rem;
    text-transform: uppercase;
}
.summary-native-header.count {
    text-align: right;
}
.summary-native-topic {
    align-items: center;
    color: #082b63 !important;
    display: flex;
    min-width: 0;
    padding: 0.16rem 0;
}
.summary-native-count {
    color: #082b63 !important;
    font-variant-numeric: tabular-nums;
    font-weight: 760;
    padding: 0.16rem 0;
    text-align: right;
}
.summary-dot {
    border-radius: 999px;
    display: inline-block;
    height: 0.7rem;
    margin-right: 0.55rem;
    vertical-align: -0.05rem;
    width: 0.7rem;
}
.empty-state {
    background: #f0f7ff;
    border-radius: 8px;
    color: #315f92;
    font-size: 0.95rem;
    margin-top: 1rem;
    padding: 1rem;
}
.loading-card {
    background: #ffffff;
    border: 1px solid #dbe8f8;
    border-radius: 8px;
    box-shadow: 0 18px 45px rgba(8, 43, 99, 0.14);
    color: #082b63;
    margin: 1rem 0;
    max-width: 420px;
    padding: 1.5rem;
    text-align: center;
}
.ticker-loader {
    height: 78px;
    margin: 0 auto 12px;
    position: relative;
    width: 220px;
}
.ticker-loader::before {
    background: linear-gradient(90deg, transparent, rgba(11, 96, 231, 0.20), transparent);
    content: "";
    height: 1px;
    left: 0;
    position: absolute;
    right: 0;
    top: 38px;
}
.ticker-dot {
    animation: tickerPulse 1.4s ease-in-out infinite;
    background: #19c4b4;
    border-radius: 999px;
    box-shadow: 0 0 22px rgba(25, 196, 180, 0.70);
    height: 12px;
    left: 0;
    position: absolute;
    top: 32px;
    width: 12px;
}
.ticker-line {
    animation: tickerTrace 1.4s ease-in-out infinite;
    border-bottom: 3px solid #5a6cf4;
    border-right: 3px solid #5a6cf4;
    height: 42px;
    left: 20px;
    position: absolute;
    top: 18px;
    transform: skewX(-32deg);
    width: 160px;
}
.loading-card h3 {
    font-size: 1.05rem;
    margin: 0 0 0.35rem;
}
.loading-card p {
    color: #315f92;
    font-size: 0.9rem;
    line-height: 1.45;
    margin: 0;
}
@keyframes tickerPulse {
    0% { left: 0; top: 46px; }
    35% { left: 72px; top: 18px; }
    68% { left: 142px; top: 42px; }
    100% { left: 208px; top: 14px; }
}
@keyframes tickerTrace {
    0% { clip-path: inset(0 100% 0 0); opacity: 0.25; }
    35% { clip-path: inset(0 58% 0 0); opacity: 1; }
    68% { clip-path: inset(0 22% 0 0); opacity: 1; }
    100% { clip-path: inset(0 0 0 0); opacity: 0.55; }
}
@media (max-width: 760px) {
    .brand-lockup {
        align-items: flex-start;
        flex-direction: column;
    }
    .brand-logo {
        height: 82px;
        width: 190px;
    }
    .app-title h1 {
        font-size: 1.75rem;
    }
}
"""


COMPONENT_CSS = """
body {
    background: #f7fbff;
    color: #082b63;
    font-family: "Aptos", "Inter", "IBM Plex Sans", ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    margin: 0;
}
.review-note {
    color: #082b63;
    font-size: 14px;
    font-weight: 700;
    margin: 8px 0 12px;
}
.document-review {
    align-items: flex-start;
    display: grid;
    gap: 16px;
    grid-template-columns: minmax(0, 1fr) 340px;
}
.pdf-view {
    display: flex;
    flex-direction: column;
    gap: 18px;
    max-height: 840px;
    overflow: auto;
    padding-right: 8px;
}
.pdf-page {
    align-items: center;
    background: #f0f7ff;
    border: 1px solid #dbe8f8;
    border-radius: 8px;
    display: flex;
    flex-direction: column;
    padding: 12px;
}
.pdf-page-canvas {
    max-width: 100%;
    overflow: auto;
    position: relative;
    width: fit-content;
}
.pdf-page img {
    background: #fbfdff;
    border: 1px solid #c8d9f3;
    box-shadow: 0 12px 32px rgba(8, 43, 99, 0.14);
    display: block;
    height: auto;
    max-width: 100%;
    width: 100%;
}
.page-label {
    color: #0b60e7;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 8px;
}
.pdf-highlight {
    background: color-mix(in srgb, var(--topic-color) calc(var(--topic-alpha) * 100%), transparent);
    border: 2px solid var(--topic-color);
    cursor: pointer;
    display: block;
    opacity: 0.88;
    padding: 0;
    position: absolute;
}
.pdf-highlight:hover,
.pdf-highlight.is-selected {
    box-shadow: 0 0 0 3px color-mix(in srgb, var(--topic-color) 30%, transparent);
    opacity: 1;
}
.detail-drawer {
    background: #ffffff;
    border: 1px solid #dbe8f8;
    border-radius: 8px;
    box-shadow: 0 10px 28px rgba(8, 43, 99, 0.08);
    max-height: 840px;
    overflow: auto;
    padding: 14px;
    position: sticky;
    top: 0;
}
.detail-empty {
    color: #315f92;
    font-size: 14px;
}
.detail-card {
    display: none;
}
.detail-card.is-visible {
    display: block;
}
.detail-topic {
    border-radius: 999px;
    color: #ffffff;
    display: inline-flex;
    font-size: 13px;
    font-weight: 760;
    padding: 7px 10px;
}
.detail-metadata {
    margin: 14px 0;
}
.detail-metadata div {
    align-items: baseline;
    display: flex;
    justify-content: space-between;
}
.detail-metadata dt {
    color: #315f92;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
}
.detail-metadata dd {
    color: #082b63;
    font-size: 14px;
    font-weight: 700;
    margin: 0;
}
.detail-drawer h3 {
    font-size: 14px;
    margin: 12px 0 8px;
}
.detail-drawer p {
    color: #082b63;
    font-size: 14px;
    line-height: 1.55;
    white-space: pre-wrap;
}
.empty-state {
    background: #f0f7ff;
    border-radius: 8px;
    color: #315f92;
    font-size: 14px;
    padding: 14px;
}
@media (max-width: 980px) {
    .document-review {
        grid-template-columns: 1fr;
    }
    .detail-drawer {
        max-height: 420px;
        position: static;
    }
}
"""


if __name__ == "__main__":
    main()

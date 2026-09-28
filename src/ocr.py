"""
ocr.py — Google Cloud Vision OCR for image-based vendor quotes.

Uses DOCUMENT_TEXT_DETECTION (not LABEL_DETECTION or the basic TEXT_DETECTION):
  - Preserves spatial layout: words grouped into lines, lines into blocks
  - Handles table reconstruction via bounding-polygon positions
  - Applies automatic skew correction and glare compensation server-side
  - Returns per-word confidence scores

Why Vision over Document AI Form Parser:
  Vendor quotes are heterogeneous (each vendor has their own column layout).
  Form Parser is tuned for standardised forms (invoices, receipts).
  Vision's raw block output gives us more layout signal to pass as context
  to the LLM, which then does the actual field extraction.
"""
from __future__ import annotations
import os
import io
from typing import Optional

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".heic", ".heif"}


def _retry_vision_call(fn, max_attempts: int = 3, delay_s: float = 8.0):
    """
    Retry a Vision API call for transient billing-propagation errors.
    GCP says 'wait a few minutes' after enabling billing; retrying with backoff
    handles the window when some backend nodes haven't picked up the new billing state yet.
    """
    import time
    last_exc = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except OCRBillingError as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                wait = delay_s * (attempt + 1)
                print(f"  Vision billing propagating — retry {attempt+1}/{max_attempts-1} in {wait:.0f}s...")
                time.sleep(wait)
    raise last_exc  # type: ignore[misc]


def is_image_file(path: str) -> bool:
    ext = os.path.splitext(path)[1].lower()
    return ext in SUPPORTED_IMAGE_EXTENSIONS


def _read_image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


class OCRBillingError(RuntimeError):
    """Raised when Cloud Vision is unavailable due to billing not being enabled."""


def ocr_image(image_path: str) -> tuple[str, float]:
    """
    Run DOCUMENT_TEXT_DETECTION on an image file.

    Returns:
        (text, avg_confidence)
        text            — spatially-aware OCR text, newlines preserved per layout
        avg_confidence  — mean word-level confidence (0.0–1.0); 0.0 if unavailable
    """
    import warnings
    # Suppress Python 3.9 EOL warnings from google-cloud libs
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from google.cloud import vision as gv

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path or not os.path.exists(creds_path):
        raise RuntimeError(
            f"GOOGLE_APPLICATION_CREDENTIALS not set or file missing: {creds_path!r}"
        )

    try:
        gv_client = gv.ImageAnnotatorClient()
        image_bytes = _read_image_bytes(image_path)
        image_obj = gv.Image(content=image_bytes)
        response = gv_client.document_text_detection(image=image_obj)
    except Exception as exc:
        if "BILLING_DISABLED" in str(exc) or "billing" in str(exc).lower():
            raise OCRBillingError(
                "Cloud Vision requires billing to be enabled on the GCP project. "
                "Visit: https://console.developers.google.com/billing/enable"
            ) from exc
        raise

    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    annotation = response.full_text_annotation
    if not annotation or not annotation.text:
        return "", 0.0

    # Collect per-word confidences for a quality signal
    confidences: list[float] = []
    for page in annotation.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                for word in para.words:
                    if word.confidence > 0:
                        confidences.append(word.confidence)

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    # full_text_annotation.text already has layout-aware newlines inserted
    # by the Vision API based on bounding-polygon positions — use it directly.
    return annotation.text, round(avg_conf, 3)


def ocr_image_to_table_text(image_path: str) -> tuple[str, float]:
    """
    Same as ocr_image but additionally attempts a simple table reconstruction:
    groups words by approximate row position (Y-coordinate band) and joins
    columns with tabs, so the LLM sees something closer to a table structure
    than bare newline-delimited text.

    Falls back gracefully to plain text if layout data is unavailable.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        from google.cloud import vision as gv

    creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not creds_path or not os.path.exists(creds_path):
        raise RuntimeError(
            f"GOOGLE_APPLICATION_CREDENTIALS not set or file missing: {creds_path!r}"
        )

    try:
        gv_client = gv.ImageAnnotatorClient()
        img_bytes = _read_image_bytes(image_path)
        img_obj = gv.Image(content=img_bytes)
        response = gv_client.document_text_detection(image=img_obj)
    except Exception as exc:
        if "BILLING_DISABLED" in str(exc) or "billing" in str(exc).lower():
            raise OCRBillingError(
                "Cloud Vision requires billing to be enabled on the GCP project. "
                "Visit: https://console.developers.google.com/billing/enable"
            ) from exc
        raise

    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    annotation = response.full_text_annotation
    if not annotation or not annotation.text:
        return "", 0.0

    confidences: list[float] = []

    # Collect all words with their midpoint Y coordinate
    word_rows: list[tuple[float, float, str]] = []  # (y_mid, x_mid, text)
    for page in annotation.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                for word in para.words:
                    if word.confidence > 0:
                        confidences.append(word.confidence)
                    verts = word.bounding_box.vertices
                    if not verts:
                        continue
                    ys = [v.y for v in verts]
                    xs = [v.x for v in verts]
                    y_mid = sum(ys) / len(ys)
                    x_mid = sum(xs) / len(xs)
                    word_text = "".join(
                        s.text for sym_list in [p.symbols for p in para.words if p == word]
                        for s in sym_list
                    ) if False else "".join(  # simpler path
                        s.text for s in word.symbols
                    )
                    word_rows.append((y_mid, x_mid, word_text))

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    if not word_rows:
        return annotation.text, avg_conf

    # Sort by Y first, then X
    word_rows.sort(key=lambda w: (w[0], w[1]))

    # Group into rows using a Y-band threshold (≈line height)
    # Estimate line height from the page height / number of distinct Y clusters
    y_values = sorted(set(round(w[0] / 10) * 10 for w in word_rows))
    band_size = 12  # pixels — generous to handle slight skew

    rows: list[list[tuple[float, str]]] = []  # each row: list of (x, text)
    current_row: list[tuple[float, str]] = []
    current_y: Optional[float] = None

    for y_mid, x_mid, text in word_rows:
        if current_y is None or abs(y_mid - current_y) > band_size:
            if current_row:
                rows.append(sorted(current_row, key=lambda c: c[0]))
            current_row = [(x_mid, text)]
            current_y = y_mid
        else:
            current_row.append((x_mid, text))
    if current_row:
        rows.append(sorted(current_row, key=lambda c: c[0]))

    # Join each row with tab separators, rows with newlines
    table_lines = ["\t".join(cell for _, cell in row) for row in rows]
    table_text = "\n".join(table_lines)

    return table_text, avg_conf


def pdf_page_to_image(pdf_path: str, page_number: int = 0, dpi: int = 200) -> str:
    """
    Convert one PDF page to a PNG in /tmp and return the path.
    Used to create a test image from an existing vendor PDF without a phone.
    page_number is 0-indexed.
    """
    from pdf2image import convert_from_path
    import tempfile

    pages = convert_from_path(pdf_path, dpi=dpi, first_page=page_number + 1,
                              last_page=page_number + 1)
    if not pages:
        raise ValueError(f"Could not convert page {page_number} of {pdf_path}")

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    pages[0].save(tmp.name, "PNG")
    return tmp.name


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ocr.py <image_or_pdf_path> [page_number]")
        sys.exit(1)

    path = sys.argv[1]
    page = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    if path.lower().endswith(".pdf"):
        print(f"Converting PDF page {page} to PNG...")
        path = pdf_page_to_image(path, page_number=page)
        print(f"  → {path}")

    print(f"Running OCR on {path}...")
    text, conf = ocr_image_to_table_text(path)
    print(f"\nOCR confidence: {conf:.3f}")
    print(f"\n--- OCR TEXT (first 2000 chars) ---\n{text[:2000]}")

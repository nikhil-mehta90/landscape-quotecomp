"""
extract.py — Extract vendor quotes and questionnaire responses into vendor_raw/*.json.
One JSON file per vendor document. Phoenix v1 and v2 are separate files with versioning.

Image input path (phone photos / scanned quotes):
  - OCR is run first via ocr.py (Google Cloud Vision DOCUMENT_TEXT_DETECTION)
  - The raw image AND the OCR text are both sent to GPT-4o:
      * Image as a vision content block (preserves table structure, handles layout)
      * OCR text as explicit context (helps with skewed/glared characters)
  - Low extraction confidence is expected and normal for messy photos;
    rows flow through as low-confidence without any retry loop.
"""
from __future__ import annotations
import base64
import json
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Optional
import pdfplumber
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from ocr import is_image_file, ocr_image_to_table_text, OCRBillingError, _retry_vision_call

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Model constants — pinned per call type, not env-overridable.
# Vision models must support image_url content blocks.
MODEL_EXTRACTION = "gpt-4o"   # PDF/XLSX text extraction: heterogeneous layouts, spec-grade inference, flag generation
MODEL_VISION     = "gpt-4o"   # Image/photo extraction: spatial table reconstruction across multiple pages
MODEL_QUESTIONNAIRE = "gpt-4o"  # Questionnaire: 4-state (PASS/FAIL/AMBIGUOUS/NO_RESPONSE) nuance requires full model

# NOTE — future ingestion formats (Word/DOCX, plain-text para, email body with inline line items):
# These formats blur the boundary between structured and unstructured extraction.
# Revisit model choice AND prompt strategy before wiring them in:
#   - Word/DOCX with tables → likely MODEL_EXTRACTION is fine; test paragraph BOQ edge case
#   - Plain-text / email body → higher ambiguity; may need a pre-parse step (segment into
#     line items first, then extract) rather than one-shot extraction; consider mini for
#     the segmentation pass and full model for the extraction pass.
#   - Email with inline pricing ("₹42/kg, rest same as last year") → requires reference
#     to prior quote for delta resolution; extraction prompt needs to change materially.

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
KUKAS_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_data", "kukas")
KUKAS_INBOUND = os.path.join(
    os.path.dirname(__file__), "..", "rfx_workspace",
    "RFX-2026-LANDSCAPE-001 — Skydome Kukas Landscaping", "inbound"
)
KUKAS_OUTBOUND = os.path.join(
    os.path.dirname(__file__), "..", "rfx_workspace",
    "RFX-2026-LANDSCAPE-001 — Skydome Kukas Landscaping", "outbound"
)
PROJECT_DIR = os.path.join(os.path.dirname(__file__), "..")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "comparison.db")

LANDSCAPE_RFX_ID = "RFX-2026-LANDSCAPE-001"


# ── Text extraction helpers ───────────────────────────────────────────────────

def extract_pdf_text(path: str) -> str:
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    for row in table:
                        if row:
                            pages.append(" | ".join(str(c or "").strip() for c in row))
            else:
                pages.append(f"[PAGE {i+1}]\n{text}")
    return "\n".join(pages)


def extract_xlsx_text(path: str) -> str:
    """Generic xlsx → text for non-BOQ sheets (fallback)."""
    xl = pd.ExcelFile(path)
    out = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, header=None)
        out.append(f"[SHEET: {sheet}]")
        out.append(df.to_string(index=False, na_rep=""))
    return "\n".join(out)


def extract_image_doc(
    image_path: str,
    vendor_id: str,
    source_file: str,
    document_version: str,
    superseded_by: Optional[str],
    rfx_lines: list,
) -> dict:
    """
    Extract a vendor quote from a phone photo or scanned image.

    Strategy: send BOTH the image (GPT-4o vision block) AND the OCR text
    (as an explicit text block) in the same LLM call.
    - Vision block: GPT-4o reads table structure, column alignment, spatial layout
    - OCR text: compensates for skew/glare where pixel-level reading struggles
    The LLM synthesises both signals; it does the actual field extraction.
    """
    print(f"  Running Cloud Vision OCR on {os.path.basename(image_path)}...")
    ocr_text = ""
    ocr_conf = 0.0
    ocr_available = True
    try:
        ocr_text, ocr_conf = _retry_vision_call(lambda: ocr_image_to_table_text(image_path))
        print(f"  OCR confidence: {ocr_conf:.3f}  ({len(ocr_text)} chars)")
    except OCRBillingError as e:
        print(f"  WARNING: Cloud Vision billing not enabled — falling back to image-only mode.")
        ocr_available = False
    except Exception as e:
        print(f"  WARNING: OCR failed ({type(e).__name__}: {e}) — falling back to image-only mode.")
        ocr_available = False

    # Encode image as base64 for GPT-4o vision
    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    ext = os.path.splitext(image_path)[1].lower().lstrip(".")
    mime = {
        "jpg": "image/jpeg", "jpeg": "image/jpeg",
        "png": "image/png", "gif": "image/gif",
        "webp": "image/webp", "tiff": "image/tiff", "tif": "image/tiff",
        "heic": "image/heic", "heif": "image/heif",
        "bmp": "image/bmp",
    }.get(ext, "image/jpeg")

    rfx_ctx = rfx_context_summary(rfx_lines)
    if ocr_available and ocr_text:
        ocr_block = (
            f"OCR text (Cloud Vision layout-aware output, confidence={ocr_conf:.2f} — "
            f"use as supplementary context):\n---\n{ocr_text[:20000]}\n---\n\n"
            f"The image is attached below. Cross-reference the image and OCR text to extract "
            f"all line items. Where they conflict, trust the image layout over OCR text."
        )
    else:
        ocr_block = (
            "OCR text: unavailable (Cloud Vision billing not enabled). "
            "Extract all line items from the image alone."
        )

    preamble = (
        f"RFx canonical line items for matching:\n{rfx_ctx}\n\n"
        f"vendor_id: {vendor_id}\n"
        f"source_file: {source_file}\n"
        f"document_version: {document_version}\n"
        f"superseded_by: {superseded_by}\n\n"
        f"NOTE: This is a photographed/scanned document. "
        f"Low extraction_confidence is expected — do not force certainty.\n"
        f"Set source_location to 'photo_ocr:line_N' for rows read from the OCR text "
        f"and 'photo_image:approx_row_N' for rows read primarily from the image.\n\n"
        f"{ocr_block}"
    )

    print(f"  LLM vision extraction for {source_file} (image + {len(ocr_text)} OCR chars)...")
    resp = client.chat.completions.create(
        model=MODEL_VISION,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": preamble},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{img_b64}", "detail": "high"},
                    },
                ],
            },
        ],
        temperature=0,
    )
    result = json.loads(resp.choices[0].message.content)
    result["extracted_at"] = datetime.now(timezone.utc).isoformat()
    result["ocr_confidence"] = ocr_conf if ocr_available else None
    result["ocr_mode"] = "vision+image" if (ocr_available and ocr_text) else "image_only"
    return result


def extract_boq_xlsx_rows(path: str) -> list[dict]:
    """
    Parse a landscaping BOQ xlsx into clean per-row records.
    Extracts S.NO, description, height/spread/caliper specs, qty, unit, rate, amount.
    Works for BOQs with the standard KUKAS/SKYDOME column layout.
    Returns a list of dicts — one per numeric line item row.
    """
    xl = pd.ExcelFile(path)
    # Use the main BOQ sheet (first sheet or one containing 'Boq'/'BOQ')
    sheet_name = xl.sheet_names[0]
    for s in xl.sheet_names:
        if "boq" in s.lower() or "hor" in s.lower():
            sheet_name = s
            break

    df = xl.parse(sheet_name, header=None)
    records = []

    # Detect column positions from header rows (rows 4-5 in standard layout)
    # Standard layout: 0=S.NO, 1=DESCRIPTION, 2=FORM/QUALITY,
    # 3=HEIGHT, 4=SPREAD, 5=CALIPER, 6=ON NGL, 7=ON SLAB, 8=QTY, 9=UNIT, 10=RATE, 11=AMOUNT

    col_sno, col_desc, col_ht, col_spread, col_caliper = 0, 1, 3, 4, 5
    col_qty, col_unit, col_rate, col_amount = 8, 9, 10, 11

    current_section = "unknown"
    section_keywords = {
        "soil": "soil_prep", "planting soil": "soil_prep", "anti termite": "soil_prep",
        "tree": "trees", "palm": "trees", "bamboo": "trees", "specimen": "trees",
        "shrub": "shrubs", "climber": "shrubs",
        "ground cover": "ground_covers", "ground_cover": "ground_covers",
        "grass": "ground_covers", "seasonal": "ground_covers",
        "lawn": "lawn",
        "stak": "staking",
    }

    for _, row in df.iterrows():
        sno = str(row.iloc[col_sno]).strip() if col_sno < len(row) else ""
        desc = str(row.iloc[col_desc]).strip() if col_desc < len(row) else ""

        # Detect section headers (rows where col0 is a Roman numeral or section label)
        if sno and sno not in ["nan", "S.NO.", "NaN", ""] and not any(c.isdigit() for c in sno):
            desc_lower = desc.lower()
            for kw, sec in section_keywords.items():
                if kw in desc_lower:
                    current_section = sec
                    break
            continue  # section header row, not a data row

        # Skip rows without numeric S.NO
        if not sno or sno in ["nan", "NaN", "", "S.NO."]:
            continue
        # S.NO must be numeric (possibly like "3.1", "13.2")
        try:
            float(sno)
        except ValueError:
            continue

        # Infer section from description if still unknown
        if current_section == "unknown":
            desc_lower = desc.lower()
            for kw, sec in section_keywords.items():
                if kw in desc_lower:
                    current_section = sec
                    break

        def safe_float(val):
            try:
                v = str(val).replace(",", "").strip()
                return float(v) if v not in ["nan", "NaN", ""] else None
            except (ValueError, TypeError):
                return None

        def safe_str(val):
            s = str(val).strip()
            return s if s not in ["nan", "NaN", ""] else None

        qty = safe_float(row.iloc[col_qty]) if col_qty < len(row) else None
        unit = safe_str(row.iloc[col_unit]) if col_unit < len(row) else None
        rate = safe_float(row.iloc[col_rate]) if col_rate < len(row) else None
        amount = safe_float(row.iloc[col_amount]) if col_amount < len(row) else None
        height = safe_str(row.iloc[col_ht]) if col_ht < len(row) else None
        spread = safe_str(row.iloc[col_spread]) if col_spread < len(row) else None
        caliper = safe_str(row.iloc[col_caliper]) if col_caliper < len(row) else None

        # Skip rows with no useful data
        if qty is None and rate is None:
            continue

        spec_parts = [p for p in [
            f"{height}m ht" if height else None,
            f"{spread}m spread" if spread else None,
            f"{caliper}mm caliper" if caliper else None,
        ] if p]

        records.append({
            "sno": sno,
            "section": current_section,
            "description": desc[:300],  # cap at 300 chars
            "spec_notes": ", ".join(spec_parts) if spec_parts else None,
            "unit": unit,
            "quantity": qty,
            "rate": rate,
            "amount": amount,
        })

    return records


def boq_rows_to_extraction_prompt(records: list[dict], vendor_id: str,
                                   source_file: str, rfx_lines: list) -> str:
    """Convert clean structured rows into a compact LLM prompt for semantic matching."""
    rows_csv = "sno|section|description|spec_notes|unit|qty|rate|amount\n"
    for r in records:
        rows_csv += (
            f"{r['sno']}|{r['section']}|{r['description'][:100]}|"
            f"{r.get('spec_notes','')}|{r['unit']}|{r['quantity']}|{r['rate']}|{r['amount']}\n"
        )
    rfx_ctx = rfx_context_summary(rfx_lines)
    return (
        f"vendor_id: {vendor_id}\nsource_file: {source_file}\n"
        f"document_version: v1\nsuperseded_by: null\n\n"
        f"RFx canonical lines:\n{rfx_ctx}\n\n"
        f"Vendor structured rows (pre-parsed from xlsx — quantities/rates are correct, "
        f"just needs rfx line_id matching and confidence scoring):\n{rows_csv}"
    )


# ── RFx context loader ────────────────────────────────────────────────────────

def load_rfx_lines() -> list:
    path = os.path.join(DATA_DIR, "rfx_lines.json")
    with open(path) as f:
        return json.load(f)


def load_rfx_lines_from_db(rfx_id: str) -> list:
    """Load rfx_lines from comparison.db filtered by rfx_id. Returns [] if DB absent."""
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM rfx_lines WHERE rfx_id = ?", (rfx_id,)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def load_rfx_lines_for(rfx_id: str) -> list:
    """
    Load rfx_lines for the given rfx_id.
    Tries DB first; falls back to rfx_lines.json for the landscape project.
    """
    rows = load_rfx_lines_from_db(rfx_id)
    if rows:
        return rows
    if rfx_id == LANDSCAPE_RFX_ID:
        return load_rfx_lines()
    raise ValueError(f"No rfx_lines found in DB for rfx_id={rfx_id!r}")


def rfx_context_summary(rfx_lines: list) -> str:
    lines = ["line_id | section | description | unit | quantity | boq_unit_rate | spec_notes"]
    for l in rfx_lines:
        lines.append(
            f"{l['line_id']} | {l['section']} | {l['description']} | "
            f"{l['unit']} | {l.get('quantity','')} | {l.get('boq_unit_rate','')} | "
            f"{l.get('spec_notes','')}"
        )
    return "\n".join(lines)


# ── Extraction system prompt ──────────────────────────────────────────────────

EXTRACTION_SYSTEM = """
You are extracting a vendor's landscaping quotation into a structured JSON object.
The vendor document will be provided along with the canonical RFx line items for matching.
Return a single JSON object (not an array) with the exact schema below.

KNOWN UGLY EDGES — handle these explicitly, do not silently resolve:
1. Mamta.pdf: the RATE column sometimes contains "Xsqm" or "Xcum" entries (bag size merged into rate).
   When you see this, use Amount ÷ Quantity to derive the actual unit rate; set normalization_note
   to explain; set extraction_confidence ≤ 0.75 and add flag "rate_column_garbled".
2. Jai Balaji staking (item ~57): unit may be listed as "Sqm" while the BOQ requires "Each".
   Flag as "unit_mismatch_sqm_vs_each"; do NOT normalize silently; set value_source "inferred"
   and mark spec_grade_match "unknown".
3. Phoenix PDFs: item 15 name may read "Ficus benghalensis variegated" — BOQ requires
   "Ficus benjamina variegated". These are different species. Set flag "species_name_mismatch"
   and match_confidence ≤ 0.60; do NOT silently match.
4. Any vendor quoting plants at a different height/girth than the BOQ spec_notes: set
   spec_grade_match to "undergrade" or "overgrade"; add flag "spec_grade_mismatch".
5. If a line item is in the vendor document but cannot be confidently matched to any RFx line:
   set matched=false, match_confidence ≤ 0.40, value_source "unknown".
6. If a price value cannot be read reliably: set extraction_confidence ≤ 0.40,
   value_source "unknown", raw_unit_price null.

Return a JSON object with this exact schema:
{
  "rfx_id": "<rfx_id as provided>",
  "vendor_id": "<as given>",
  "vendor_name": "<extract from document>",
  "source_file": "<filename>",
  "document_version": "<v1 or v2>",
  "superseded_by": "<filename or null>",
  "extracted_at": "<ISO timestamp>",
  "extractions": [
    {
      "line_id": "<matched rfx line_id or null>",
      "matched": true/false,
      "match_confidence": 0.0-1.0,
      "raw_unit_price": <number or null>,
      "raw_unit": "<unit string from vendor doc>",
      "raw_currency": "INR",
      "normalized_unit_price": <number or null>,
      "normalization_note": "<how/why converted, or null>",
      "quantity_quoted": <number or null>,
      "freight_included": "yes"/"no"/"unknown",
      "labor_included": "yes"/"no"/"unknown",
      "spec_grade_quoted": "<vendor's stated ht/size/grade, or null>",
      "spec_grade_match": "match"/"undergrade"/"overgrade"/"unknown",
      "source_snippet": "<verbatim text from document for this line>",
      "source_location": "<e.g. 'row 12' or 'page 2, item 5'>",
      "extraction_confidence": 0.0-1.0,
      "value_source": "extracted"/"inferred"/"unknown",
      "flags": ["flag1", "flag2"]
    }
  ],
  "questionnaire_responses": []
}

IMPORTANT:
- Include ALL line items from the vendor document, even if unmatched to RFx lines.
- Prefer extraction_confidence ≤ 0.70 over false certainty.
- freight_included and labor_included: default "unknown" unless document explicitly states.
- Return ONLY the JSON object, no prose.
"""

QUESTIONNAIRE_SYSTEM = """
You are extracting a vendor's questionnaire responses from their written reply.
You are given:
1. The vendor's response text
2. The questionnaire definitions (q_id, question, pass_criteria)
3. The vendor_id

Return a JSON object with key "responses" containing an array. Each element:
{
  "q_id": "<q_id from definitions>",
  "question": "<question text>",
  "answer": "<verbatim or close paraphrase of vendor's answer>",
  "answer_type": "text",
  "passes": 1/0/null,
  "confidence": 0.0-1.0,
  "source_location": "<which question number or paragraph in the response>",
  "value_source": "extracted"/"unknown"
}

RULES FOR passes:
- 1 (pass): vendor clearly meets the criterion
- 0 (fail): vendor clearly fails the criterion
- null: cannot determine — ambiguous, non-committal, or question not answered at all

SPECIFIC CASES — handle exactly as described:
1. If Mamta doesn't answer the site-access question: passes=null, value_source="unknown",
   answer="(no response)"
2. If Phoenix's mortality answer says "management's discretion, case-by-case, no fixed percentage":
   passes=null (cannot evaluate against ≥80% criterion), add a note in answer about why it can't
   be evaluated; do NOT default to pass or fail
3. If Green Thumbs proposes May start: passes=0 (clearly fails monsoon window criterion);
   capture their stated reason (crew availability) in the answer field

Return {"responses": [...]} only.
"""


SECTION_QUOTE_SYSTEM = """
You are extracting a vendor's landscaping quotation that is stated at SECTION level (not per-line-item).
The document may be an email, letter, or written quote where the vendor gives one total per category.

Return a JSON object with this exact schema:
{
  "vendor_name": "<extract from document>",
  "revision_type": "quote",
  "granularity": "section_total",
  "section_quotes": [
    {
      "section": "<section key — one of: trees, shrubs, ground_covers, lawn, soil_prep, staking, package_non_tree, trees_balance>",
      "total_amount": <INR number or null if unknown>,
      "estimate_low": null,
      "estimate_high": null,
      "currency": "INR",
      "value_source": "vendor_quote",
      "package_description": "<what this section total covers, if different from the standard section>",
      "source_snippet": "<verbatim text from document for this section>",
      "extraction_confidence": 0.0-1.0,
      "flags": []
    }
  ],
  "extractions": [],
  "questionnaire_responses": []
}

SECTION KEY RULES:
- Use "package_non_tree" when a vendor lumps multiple sections (e.g. shrubs+lawn+soil_prep) into one price.
- Use "trees_balance" for "all other trees at average rate" that doesn't break down by species.
- Include soil_prep = null if vendor explicitly defers site prep pricing.
- Set extraction_confidence high (0.90+) for clearly stated amounts; lower for approximations.
- Capture any quantity discrepancies vs. BOQ in flags (e.g. "vendor_qty_3215_vs_boq_3846").

Return ONLY the JSON object, no prose.
"""

VERBAL_ESTIMATE_SYSTEM = """
You are extracting per-unit rate ranges from a VERBAL call transcript for a landscaping project.
The transcript contains rough verbal estimates, not a formal quote. Return rates per category.

Return a JSON object:
{
  "vendor_name": "<from transcript>",
  "verbal_rates": {
    "trees": {"low": <INR per Nos or null>, "high": <INR per Nos or null>, "note": "<verbatim>"},
    "bamboo": {"low": <INR per Nos or null>, "high": <INR per Nos or null>, "note": "<verbatim>"},
    "shrubs": {"low": <INR per Nos or null>, "high": <INR per Nos or null>, "note": "<verbatim>"},
    "ground_covers": {"low": <INR per Sqm or null>, "high": <INR per Sqm or null>, "note": "<verbatim>"},
    "lawn": {"low": <INR per Sqm or null>, "high": <INR per Sqm or null>, "note": "<verbatim>"},
    "soil_prep": null
  },
  "soil_prep_deferred": <true if vendor explicitly deferred soil_prep to written quote>,
  "questionnaire_responses": []
}

RULES:
- soil_prep MUST be null if the vendor says they'll give a lump sum later (deferred).
- If bamboo is quoted separately from other trees, capture it separately under "bamboo".
- "high" == "low" for single-value rates (no range).
- Return ONLY the JSON object, no prose.
"""

QUESTIONNAIRE_UPDATE_SYSTEM = """
You are extracting questionnaire answer UPDATES from a follow-up call transcript.
This call clarifies existing answers; there is NO pricing in this call.
Match each clarified answer to the closest q_id from the definitions provided.

Return:
{
  "revision_type": "questionnaire_update",
  "questionnaire_responses": [
    {
      "q_id": "<q_id from definitions>",
      "question": "<question text>",
      "answer": "<updated answer from the call — paraphrase clearly>",
      "answer_type": "text" or "conditional",
      "passes": 1/0/null,
      "confidence": 0.0-1.0,
      "source_location": "<call transcript reference>",
      "value_source": "extracted"
    }
  ]
}

RULES:
- Only include q_ids that were actually discussed and clarified in this call.
- "conditional" answer_type when the answer depends on a condition (e.g. watering compliance).
- passes=null when the answer is conditional or cannot be evaluated against pass_criteria alone.
- Return ONLY the JSON object, no prose.
"""

MIXED_QUOTE_SYSTEM = """
You are extracting a vendor's landscaping quotation with MIXED granularity:
some tree species are priced individually (line_item), everything else as section/package totals.

Return a JSON object with this schema:
{
  "vendor_name": "<from document>",
  "revision_type": "quote",
  "granularity": "mixed",
  "extractions": [
    {
      "line_id": "<matched rfx line_id or null>",
      "matched": true/false,
      "match_confidence": 0.0-1.0,
      "raw_unit_price": <number or null>,
      "raw_unit": "<unit>",
      "raw_currency": "INR",
      "normalized_unit_price": <number or null>,
      "normalization_note": null,
      "quantity_quoted": <number or null>,
      "freight_included": "yes"/"no"/"unknown",
      "labor_included": "yes"/"no"/"unknown",
      "spec_grade_quoted": null,
      "spec_grade_match": "unknown",
      "source_snippet": "<verbatim>",
      "source_location": "<location>",
      "extraction_confidence": 0.0-1.0,
      "value_source": "extracted",
      "flags": [],
      "granularity": "line_item"
    }
  ],
  "section_quotes": [
    {
      "section": "<trees_balance or package_non_tree or other section key>",
      "total_amount": <number or null>,
      "estimate_low": null,
      "estimate_high": null,
      "currency": "INR",
      "value_source": "vendor_quote",
      "package_description": "<what is covered>",
      "source_snippet": "<verbatim>",
      "extraction_confidence": 0.0-1.0,
      "flags": []
    }
  ],
  "questionnaire_responses": []
}

SECTION KEYS: trees, shrubs, ground_covers, lawn, soil_prep, staking, trees_balance, package_non_tree
- Use "trees_balance" for "all other trees at average rate X per plant".
- Use "package_non_tree" for a lump sum covering multiple non-tree sections combined.
- For species where vendor names differ from BOQ (e.g. Ceiba pentandra vs Ceiba speciosa): flag "species_name_mismatch" and set match_confidence ≤ 0.65.
- For staking: if vendor qty differs from BOQ qty, flag "quantity_discrepancy_vendor_N_vs_boq_M".
- Return ONLY the JSON object, no prose.
"""

PLANT_LIST_SYSTEM = """
You are parsing a buyer-side landscaping plant list, CAD export, or BOQ into structured line items.
This is NOT a vendor quote — do not extract prices, vendor names, or match confidence.
Return a single JSON object with the exact schema below.

Each line item is a plantable / billable scope item. Parse every row you can identify.
Flag ambiguities — do NOT silently drop partial rows.

Sections to use (match from document headings or species type):
  trees       — trees, palms, bamboo, specimen plants (pit-planted, counted in Nos)
  shrubs      — shrubs, climbers (pit-planted, counted in Nos)
  ground_covers — ground covers, grasses, seasonals (bed-planted, measured in Sqm)
  lawn        — lawn / turf (bed-planted, measured in Sqm)
  staking     — staking items (Nos, one per tree — usually derived, not explicitly listed)
  soil_prep   — site prep, anti-termite, pit digging, soil amendments (usually derived)
  other       — anything else (note in parse_notes)

Spec fields to capture in spec_notes (concatenate all that apply):
  height: "X.X-Y.Ym ht"   spread: "X.X-Y.Ym spread"   caliper: "NNmm caliper"
  pot size: "NNNmm dia pot"  clump spread: "X.X-Y.Ym clump spread"

Return:
{
  "line_items": [
    {
      "section": "<one of the sections above>",
      "description": "<item description as it appears in the document>",
      "species_name": "<botanical/scientific name if a plant, else null>",
      "unit": "<Nos|Sqm|Cum|Kg|Rmt|Lump|Each>",
      "quantity": <number or null>,
      "spec_notes": "<height/spread/caliper/pot-size specs, null if none>",
      "boq_unit_rate": <INR rate if stated, else null>,
      "parse_confidence": <0.0-1.0 — confidence you read this row correctly>,
      "parse_notes": "<ambiguities, missing values, assumptions made, or null>"
    }
  ],
  "parse_summary": "<1-2 sentences: what was found, any structural issues or gaps>"
}

IMPORTANT:
- Include ALL line items, even partial ones (set parse_confidence ≤ 0.5 and note the issue).
- Do NOT infer soil/staking/site-prep quantities — those are derived by a separate tool.
  Only include soil_prep/staking lines if they are explicitly stated in the document with a quantity.
- Return ONLY the JSON object, no prose.
"""


def extract_plant_list_doc(doc_text: str) -> dict:
    """
    Parse a buyer-side plant list / CAD export / BOQ (no prices, no vendor matching).
    Returns {line_items: [...], parse_summary: "..."} with rfx_lines-compatible fields.
    Used by the RFQ co-pilot when a buyer uploads a plant list.
    """
    resp = client.chat.completions.create(
        model=MODEL_EXTRACTION,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": PLANT_LIST_SYSTEM},
            {"role": "user", "content": f"Document text:\n\n{doc_text[:75000]}"},
        ],
        temperature=0,
    )
    result = json.loads(resp.choices[0].message.content)
    if "line_items" not in result:
        result["line_items"] = []
    if "parse_summary" not in result:
        result["parse_summary"] = ""
    return result


# ── Core extraction function ──────────────────────────────────────────────────

def extract_vendor_doc(
    doc_text: str,
    vendor_id: str,
    source_file: str,
    document_version: str,
    superseded_by: Optional[str],
    rfx_lines: list
) -> dict:
    rfx_ctx = rfx_context_summary(rfx_lines)
    user_msg = (
        f"RFx canonical line items for matching:\n{rfx_ctx}\n\n"
        f"vendor_id: {vendor_id}\n"
        f"source_file: {source_file}\n"
        f"document_version: {document_version}\n"
        f"superseded_by: {superseded_by}\n\n"
        f"Vendor document text:\n{doc_text[:75000]}"
    )
    print(f"  LLM extraction for {source_file} ({len(doc_text)} chars)...")
    resp = client.chat.completions.create(
        model=MODEL_EXTRACTION,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": user_msg}
        ],
        temperature=0
    )
    result = json.loads(resp.choices[0].message.content)
    result["extracted_at"] = datetime.now(timezone.utc).isoformat()
    return result


def extract_questionnaire_responses(
    vendor_id: str,
    vendor_section_text: str,
    q_definitions: list,
    source_file: str = "synthetic-questionnaire.md"
) -> list:
    user_msg = (
        f"vendor_id: {vendor_id}\n"
        f"source_file: {source_file}\n\n"
        f"Questionnaire definitions:\n{json.dumps(q_definitions, indent=2)}\n\n"
        f"Vendor response text:\n{vendor_section_text}"
    )
    print(f"  LLM questionnaire extraction for {vendor_id}...")
    resp = client.chat.completions.create(
        model=MODEL_QUESTIONNAIRE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": QUESTIONNAIRE_SYSTEM},
            {"role": "user", "content": user_msg}
        ],
        temperature=0
    )
    raw = json.loads(resp.choices[0].message.content)
    if "responses" in raw:
        return raw["responses"]
    if isinstance(raw, list):
        return raw
    for v in raw.values():
        if isinstance(v, list):
            return v
    return []


# ── Questionnaire file parser ─────────────────────────────────────────────────

def parse_synthetic_questionnaire(md_text: str) -> tuple[list, dict]:
    """
    Returns:
      q_definitions — list of {q_id, question, pass_criteria}
      vendor_sections — dict of vendor_name -> section text
    """
    # Extract definitions table
    q_defs = []
    table_pattern = re.compile(
        r'\|\s*(Q_\w+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|'
    )
    for m in table_pattern.finditer(md_text):
        q_id, question, pass_criteria = m.group(1), m.group(2).strip(), m.group(3).strip()
        if q_id != "q_id":  # skip header row
            q_defs.append({"q_id": q_id, "question": question, "pass_criteria": pass_criteria})

    # Split vendor sections
    vendor_sections = {}
    vendor_pattern = re.compile(r'^## Vendor:\s*(.+)$', re.MULTILINE)
    matches = list(vendor_pattern.finditer(md_text))
    for i, m in enumerate(matches):
        vendor_name = m.group(1).strip()
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(md_text)
        vendor_sections[vendor_name] = md_text[start:end].strip()

    return q_defs, vendor_sections


# ── TXT document extraction functions ────────────────────────────────────────

def extract_section_quote_txt(
    doc_text: str,
    vendor_id: str,
    source_file: str,
    document_version: str,
    superseded_by: Optional[str],
    rfx_id: str,
    rfx_lines: list,
) -> dict:
    """Extract a section-total TXT quote (email format with one price per category)."""
    rfx_ctx = rfx_context_summary(rfx_lines)
    user_msg = (
        f"rfx_id: {rfx_id}\nvendor_id: {vendor_id}\nsource_file: {source_file}\n"
        f"document_version: {document_version}\nsuperseded_by: {superseded_by or 'null'}\n\n"
        f"RFx sections and quantities for reference:\n{rfx_ctx}\n\n"
        f"Vendor document (email/letter):\n{doc_text}"
    )
    print(f"  LLM section-quote extraction for {vendor_id} {document_version}...")
    resp = client.chat.completions.create(
        model=MODEL_EXTRACTION,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SECTION_QUOTE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )
    result = json.loads(resp.choices[0].message.content)
    result.update({
        "rfx_id": rfx_id,
        "vendor_id": vendor_id,
        "source_file": source_file,
        "document_version": document_version,
        "superseded_by": superseded_by,
        "revision_type": "quote",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    })
    result.setdefault("extractions", [])
    result.setdefault("questionnaire_responses", [])
    return result


def _compute_verbal_section_totals(verbal_rates: dict, rfx_lines: list) -> list:
    """
    From LLM-extracted per-unit verbal rates, compute estimated section totals
    using rfx_lines quantities. Returns section_quotes list.
    """
    section_qty: dict[str, float] = {}
    for l in rfx_lines:
        s = l["section"]
        section_qty[s] = section_qty.get(s, 0) + (l.get("quantity") or 0)

    bamboo_qty = sum(
        (l.get("quantity") or 0) for l in rfx_lines
        if l["section"] == "trees" and "bamboo" in (l.get("description") or "").lower()
    )
    non_bamboo_tree_qty = section_qty.get("trees", 0) - bamboo_qty

    section_quotes = []

    def _sq(section: str, total: Optional[float], low: Optional[float],
            high: Optional[float], value_source: str, snippet: str,
            confidence: float, flags: list) -> dict:
        return {
            "section": section,
            "total_amount": round(total) if total is not None else None,
            "estimate_low": round(low) if low is not None else None,
            "estimate_high": round(high) if high is not None else None,
            "currency": "INR",
            "value_source": value_source,
            "source_snippet": snippet,
            "extraction_confidence": confidence,
            "flags": flags,
        }

    rates = verbal_rates.get("verbal_rates", {}) if "verbal_rates" in verbal_rates else verbal_rates

    # Trees (non-bamboo + bamboo, quoted separately in the call)
    tree_r = rates.get("trees") or {}
    bam_r = rates.get("bamboo") or {}
    if tree_r.get("low") is not None or bam_r.get("low") is not None:
        tl = (tree_r.get("low", 0) or 0) * non_bamboo_tree_qty + (bam_r.get("low", 0) or 0) * bamboo_qty
        th = (tree_r.get("high", 0) or 0) * non_bamboo_tree_qty + (bam_r.get("high", 0) or 0) * bamboo_qty
        avg = (tl + th) / 2
        snippet = (tree_r.get("note", "") or "") + " | " + (bam_r.get("note", "") or "")
        section_quotes.append(_sq("trees", avg, tl, th, "verbal_estimate",
            f"Derived: non-bamboo {non_bamboo_tree_qty:.0f} nos × avg rate + bamboo {bamboo_qty:.0f} nos × avg rate. Sources: {snippet.strip(' |')}",
            0.70, ["verbal_estimate_range", "section_total_computed_from_avg_rate"]))

    # Other per-unit sections
    for section, qty_key, rate_key in [
        ("shrubs", "shrubs", "shrubs"),
        ("ground_covers", "ground_covers", "ground_covers"),
        ("lawn", "lawn", "lawn"),
    ]:
        r = rates.get(rate_key)
        if not r or r.get("low") is None:
            continue
        qty = section_qty.get(section, 0)
        low = r["low"] * qty
        high = (r.get("high") or r["low"]) * qty
        avg = (low + high) / 2
        flags = ["verbal_estimate_range"] if r.get("high") != r.get("low") else ["verbal_estimate_range"]
        flags_extra = ["section_total_computed_from_avg_rate"] if r.get("high") != r.get("low") else []
        section_quotes.append(_sq(section, avg, low, high, "verbal_estimate",
            r.get("note", f"Derived: {qty:.0f} units × rate. {r.get('note','')}"),
            0.70 if r.get("high") != r.get("low") else 0.75,
            ["verbal_estimate_range"] + flags_extra))

    # Soil prep: always unknown if deferred
    if verbal_rates.get("soil_prep_deferred"):
        section_quotes.append(_sq("soil_prep", None, None, None, "unknown",
            "Vendor explicitly deferred site prep to written quote (pending soil test).",
            0.0, ["verbal_deferred_to_written_quote"]))

    return section_quotes


def extract_verbal_transcript_txt(
    doc_text: str,
    vendor_id: str,
    source_file: str,
    document_version: str,
    superseded_by: Optional[str],
    rfx_id: str,
    rfx_lines: list,
) -> dict:
    """Extract a verbal call transcript: LLM gets per-unit rates, we compute section totals."""
    user_msg = (
        f"vendor_id: {vendor_id}\nCall transcript:\n\n{doc_text}"
    )
    print(f"  LLM verbal-rate extraction for {vendor_id} {document_version}...")
    resp = client.chat.completions.create(
        model=MODEL_EXTRACTION,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": VERBAL_ESTIMATE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )
    verbal_rates = json.loads(resp.choices[0].message.content)
    vendor_name = verbal_rates.pop("vendor_name", None)

    section_quotes = _compute_verbal_section_totals(verbal_rates, rfx_lines)

    return {
        "rfx_id": rfx_id,
        "vendor_id": vendor_id,
        "vendor_name": vendor_name,
        "source_file": source_file,
        "document_version": document_version,
        "superseded_by": superseded_by,
        "revision_type": "quote",
        "granularity": "section_total",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "extractions": [],
        "section_quotes": section_quotes,
        "questionnaire_responses": [],
    }


def extract_questionnaire_update_txt(
    doc_text: str,
    vendor_id: str,
    source_file: str,
    document_version: str,
    rfx_id: str,
    q_definitions: list,
) -> dict:
    """Extract questionnaire answer updates from a follow-up call — no pricing."""
    user_msg = (
        f"vendor_id: {vendor_id}\n"
        f"Questionnaire definitions:\n{json.dumps(q_definitions, indent=2)}\n\n"
        f"Call transcript:\n{doc_text}"
    )
    print(f"  LLM questionnaire-update extraction for {vendor_id} {document_version}...")
    resp = client.chat.completions.create(
        model=MODEL_QUESTIONNAIRE,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": QUESTIONNAIRE_UPDATE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )
    raw = json.loads(resp.choices[0].message.content)
    responses = raw.get("questionnaire_responses", [])

    return {
        "rfx_id": rfx_id,
        "vendor_id": vendor_id,
        "vendor_name": None,
        "source_file": source_file,
        "document_version": document_version,
        "superseded_by": None,
        "revision_type": "questionnaire_update",
        "granularity": "questionnaire_only",
        "extracted_at": datetime.now(timezone.utc).isoformat(),
        "extractions": [],
        "section_quotes": [],
        "questionnaire_responses": responses,
    }


def extract_mixed_quote_txt(
    doc_text: str,
    vendor_id: str,
    source_file: str,
    document_version: str,
    superseded_by: Optional[str],
    rfx_id: str,
    rfx_lines: list,
) -> dict:
    """Extract a mixed-granularity TXT quote: some line items, some section totals."""
    rfx_ctx = rfx_context_summary(rfx_lines)
    user_msg = (
        f"rfx_id: {rfx_id}\nvendor_id: {vendor_id}\nsource_file: {source_file}\n"
        f"document_version: {document_version}\n\n"
        f"RFx canonical lines:\n{rfx_ctx}\n\n"
        f"Vendor document:\n{doc_text}"
    )
    print(f"  LLM mixed-quote extraction for {vendor_id} {document_version}...")
    resp = client.chat.completions.create(
        model=MODEL_EXTRACTION,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": MIXED_QUOTE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )
    result = json.loads(resp.choices[0].message.content)
    result.update({
        "rfx_id": rfx_id,
        "vendor_id": vendor_id,
        "source_file": source_file,
        "document_version": document_version,
        "superseded_by": superseded_by,
        "extracted_at": datetime.now(timezone.utc).isoformat(),
    })
    result.setdefault("revision_type", "quote")
    result.setdefault("extractions", [])
    result.setdefault("section_quotes", [])
    result.setdefault("questionnaire_responses", [])
    return result


# ── Vendor registry ───────────────────────────────────────────────────────────

IMAGE_FOLDER_VENDORS = [
    {
        "vendor_id": "V_MAMTA2",
        "vendor_name": "Mamta Plant Nursery (photo quote)",
        "folder": "mamta/mamta_quote_photos",
        "document_version": "v1",
        "superseded_by": None,
        "output_name": "mamta2",
        "questionnaire_name": None,  # no questionnaire for this vendor yet
    },
]

VENDORS = [
    {
        "vendor_id": "V_JAI",
        "source_file": "jai_balaji/jai_balaji.pdf",
        "document_version": "v1",
        "superseded_by": None,
        "questionnaire_name": "Jai Balaji",
    },
    {
        "vendor_id": "V_MAMTA",
        "source_file": "mamta/mamta.pdf",
        "document_version": "v1",
        "superseded_by": None,
        "questionnaire_name": "Mamta",
    },
    {
        "vendor_id": "V_PHOENIX",
        "source_file": "phoenix/phoenix.pdf",
        "document_version": "v1",
        "superseded_by": "phoenix/phoenix_r1.pdf",
        "questionnaire_name": "Phoenix",
    },
    {
        "vendor_id": "V_PHOENIX",
        "source_file": "phoenix/phoenix_r1.pdf",
        "document_version": "v2",
        "superseded_by": None,
        "questionnaire_name": "Phoenix",
    },
    {
        "vendor_id": "V_GREEN",
        "source_file": "green_thumbs/green_thumbs.xlsx",
        "document_version": "v1",
        "superseded_by": None,
        "questionnaire_name": "Green Thumbs",
    },
]

OUTPUT_NAMES = {
    "jai_balaji/jai_balaji.pdf": "jai_balaji",
    "mamta/mamta.pdf": "mamta",
    "phoenix/phoenix.pdf": "phoenix_v1",
    "phoenix/phoenix_r1.pdf": "phoenix_v2",
    "green_thumbs/green_thumbs.xlsx": "green_thumbs",
}

# TXT vendor documents: call transcripts, email quotes, follow-up clarifications.
# Each entry specifies the extraction strategy ('verbal', 'section_quote', 'mixed', 'questionnaire_update').
TXT_VENDORS = [
    {
        "vendor_id": "V_DESERT_BLOOM",
        "source_file": "desert_bloom/desert_bloom_call_transcript.txt",
        "document_version": "v1",
        "superseded_by": "desert_bloom/desert_bloom_quote_email.txt",
        "extraction_strategy": "verbal",
        "output_name": "desert_bloom_v1",
    },
    {
        "vendor_id": "V_DESERT_BLOOM",
        "source_file": "desert_bloom/desert_bloom_quote_email.txt",
        "document_version": "v2",
        "superseded_by": None,
        "extraction_strategy": "section_quote",
        "output_name": "desert_bloom_v2",
    },
    {
        "vendor_id": "V_GREEN_HORIZON",
        "source_file": "green_horizon/green_horizon_quote_email.txt",
        "document_version": "v1",
        "superseded_by": None,
        "extraction_strategy": "mixed",
        "output_name": "green_horizon_v1",
    },
    {
        "vendor_id": "V_PHOENIX",
        "source_file": "phoenix/phoenix_followup_call_transcript.txt",
        "document_version": "v3",
        "superseded_by": None,
        "extraction_strategy": "questionnaire_update",
        "output_name": "phoenix_v3",
    },
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main(rfx_id: str = LANDSCAPE_RFX_ID):
    os.makedirs(os.path.join(DATA_DIR, "vendor_raw", "kukas"), exist_ok=True)

    print(f"Loading rfx_lines for {rfx_id}...")
    rfx_lines = load_rfx_lines_for(rfx_id)

    # Load and parse synthetic questionnaire
    q_md_path = os.path.join(KUKAS_OUTBOUND, "synthetic-questionnaire.md")
    print(f"Parsing {q_md_path}...")
    with open(q_md_path) as f:
        q_md = f.read()
    q_definitions, vendor_q_sections = parse_synthetic_questionnaire(q_md)
    print(f"  {len(q_definitions)} questionnaire definitions, {len(vendor_q_sections)} vendor sections")

    # Save questionnaire_definitions for build_db
    q_def_path = os.path.join(DATA_DIR, "questionnaire_definitions.json")
    with open(q_def_path, "w") as f:
        json.dump(q_definitions, f, indent=2)
    print(f"  Written: {q_def_path}")

    # Questionnaire responses: one LLM call per vendor (not per document version)
    q_responses_by_vendor: dict[str, list] = {}
    for vendor_name_key, section_text in vendor_q_sections.items():
        # Map questionnaire name to vendor_id
        vendor_id = next(
            (v["vendor_id"] for v in VENDORS if v["questionnaire_name"] == vendor_name_key),
            None
        )
        if not vendor_id:
            print(f"  WARNING: No vendor_id for questionnaire section '{vendor_name_key}'")
            continue
        responses = extract_questionnaire_responses(
            vendor_id, section_text, q_definitions
        )
        q_responses_by_vendor[vendor_id] = responses

    # Process each vendor document
    for v in VENDORS:
        src = os.path.join(KUKAS_INBOUND, v["source_file"])
        if not os.path.exists(src):
            print(f"  SKIP: {src} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing: {v['source_file']} ({v['document_version']})")

        # Extract document text / structured rows
        if is_image_file(src):
            # Phone photo / scan: dual image+OCR extraction
            result = extract_image_doc(
                image_path=src,
                vendor_id=v["vendor_id"],
                source_file=v["source_file"],
                document_version=v["document_version"],
                superseded_by=v["superseded_by"],
                rfx_lines=rfx_lines,
            )
        elif v["source_file"].endswith(".xlsx"):
            # Use structured row extraction for BOQ xlsx (avoids the dataframe dump problem)
            rows = extract_boq_xlsx_rows(src)
            if rows:
                print(f"  Parsed {len(rows)} structured rows from xlsx")
                doc_text = boq_rows_to_extraction_prompt(
                    rows, v["vendor_id"], v["source_file"], rfx_lines
                )
            else:
                print(f"  WARNING: structured extraction got 0 rows, falling back to text dump")
                doc_text = extract_xlsx_text(src)
            result = extract_vendor_doc(
                doc_text=doc_text,
                vendor_id=v["vendor_id"],
                source_file=v["source_file"],
                document_version=v["document_version"],
                superseded_by=v["superseded_by"],
                rfx_lines=rfx_lines,
            )
        else:
            doc_text = extract_pdf_text(src)
            result = extract_vendor_doc(
                doc_text=doc_text,
                vendor_id=v["vendor_id"],
                source_file=v["source_file"],
                document_version=v["document_version"],
                superseded_by=v["superseded_by"],
                rfx_lines=rfx_lines,
            )

        # Force-set rfx_id regardless of what the LLM echoed back
        result["rfx_id"] = rfx_id

        # Attach questionnaire responses to current version only (avoid duplication for Phoenix)
        # For Phoenix v1 (superseded), no questionnaire responses; attach to v2 only
        if v["superseded_by"] is None:
            result["questionnaire_responses"] = q_responses_by_vendor.get(v["vendor_id"], [])
        else:
            result["questionnaire_responses"] = []

        # Write output
        out_name = OUTPUT_NAMES[v["source_file"]]
        out_path = os.path.join(DATA_DIR, "vendor_raw", "kukas", f"{out_name}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        n_extractions = len(result.get("extractions", []))
        n_q = len(result.get("questionnaire_responses", []))
        print(f"  Written: {out_path} ({n_extractions} extractions, {n_q} Q responses)")

    # Process image-folder vendors (multi-page phone photos)
    for iv in IMAGE_FOLDER_VENDORS:
        folder = os.path.join(KUKAS_INBOUND, iv["folder"])
        if not os.path.isdir(folder):
            print(f"  SKIP image folder: {folder} not found")
            continue
        out_name = iv["output_name"]
        out_path = os.path.join(DATA_DIR, "vendor_raw", "kukas", f"{out_name}.json")
        print(f"\n{'='*60}")
        print(f"Processing image folder: {iv['folder']} ({iv['vendor_id']})")
        extract_image_folder(
            folder_path=folder,
            vendor_id=iv["vendor_id"],
            vendor_name=iv["vendor_name"],
            document_version=iv["document_version"],
            superseded_by=iv["superseded_by"],
            output_name=out_name,
            rfx_id=rfx_id,
        )

    # Process TXT vendors (call transcripts, email quotes)
    for tv in TXT_VENDORS:
        src = os.path.join(KUKAS_INBOUND, tv["source_file"])
        if not os.path.exists(src):
            print(f"  SKIP: {src} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Processing TXT: {tv['source_file']} ({tv['document_version']}, strategy={tv['extraction_strategy']})")

        with open(src, encoding="utf-8") as f:
            doc_text = f.read()

        strategy = tv["extraction_strategy"]
        if strategy == "verbal":
            result = extract_verbal_transcript_txt(
                doc_text=doc_text,
                vendor_id=tv["vendor_id"],
                source_file=tv["source_file"],
                document_version=tv["document_version"],
                superseded_by=tv["superseded_by"],
                rfx_id=rfx_id,
                rfx_lines=rfx_lines,
            )
        elif strategy == "section_quote":
            result = extract_section_quote_txt(
                doc_text=doc_text,
                vendor_id=tv["vendor_id"],
                source_file=tv["source_file"],
                document_version=tv["document_version"],
                superseded_by=tv["superseded_by"],
                rfx_id=rfx_id,
                rfx_lines=rfx_lines,
            )
        elif strategy == "mixed":
            result = extract_mixed_quote_txt(
                doc_text=doc_text,
                vendor_id=tv["vendor_id"],
                source_file=tv["source_file"],
                document_version=tv["document_version"],
                superseded_by=tv["superseded_by"],
                rfx_id=rfx_id,
                rfx_lines=rfx_lines,
            )
        elif strategy == "questionnaire_update":
            result = extract_questionnaire_update_txt(
                doc_text=doc_text,
                vendor_id=tv["vendor_id"],
                source_file=tv["source_file"],
                document_version=tv["document_version"],
                rfx_id=rfx_id,
                q_definitions=q_definitions,
            )
        else:
            print(f"  SKIP: unknown strategy {strategy!r}")
            continue

        result["rfx_id"] = rfx_id

        out_path = os.path.join(DATA_DIR, "vendor_raw", "kukas", f"{tv['output_name']}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        n_ex = len(result.get("extractions", []))
        n_sq = len(result.get("section_quotes", []))
        n_q = len(result.get("questionnaire_responses", []))
        print(f"  Written: {out_path} ({n_ex} extractions, {n_sq} section_quotes, {n_q} Q responses)")

    print("\nextract.py done.")


def extract_image_folder(
    folder_path: str,
    vendor_id: str,
    vendor_name: str,
    document_version: str = "v1",
    superseded_by: Optional[str] = None,
    output_name: Optional[str] = None,
    rfx_id: str = LANDSCAPE_RFX_ID,
) -> str:
    """
    Extract a multi-page vendor quote photographed as separate images (one per page).

    Sends ALL page images + concatenated OCR text in a single GPT-4o call so the
    LLM sees the full document context and can resolve cross-page line numbering.
    Images are sorted by filename (timestamp filenames sort chronologically).

    Returns the path to the written JSON file.
    """
    import glob

    image_exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tiff", ".tif", ".heic", ".heif", ".bmp"}
    all_files = sorted(
        p for p in glob.glob(os.path.join(folder_path, "*"))
        if os.path.splitext(p)[1].lower() in image_exts
    )
    if not all_files:
        raise ValueError(f"No image files found in {folder_path}")

    print(f"\nMulti-page image extraction: {len(all_files)} pages from {folder_path}")
    rfx_lines = load_rfx_lines_for(rfx_id)
    os.makedirs(os.path.join(DATA_DIR, "vendor_raw"), exist_ok=True)

    # OCR each page with retry
    page_texts: list[str] = []
    page_confs: list[float] = []
    for i, img_path in enumerate(all_files):
        print(f"  Page {i+1}/{len(all_files)}: {os.path.basename(img_path)}")
        try:
            text, conf = _retry_vision_call(lambda p=img_path: ocr_image_to_table_text(p))
            page_texts.append(f"[PAGE {i+1} OCR]\n{text}")
            page_confs.append(conf)
            print(f"    OCR conf={conf:.3f}, chars={len(text)}")
        except OCRBillingError:
            print(f"    WARNING: OCR unavailable — this page will be image-only")
            page_texts.append(f"[PAGE {i+1} OCR — unavailable]")
            page_confs.append(0.0)
        except Exception as e:
            print(f"    WARNING: OCR failed ({e}) — this page will be image-only")
            page_texts.append(f"[PAGE {i+1} OCR — failed: {e}]")
            page_confs.append(0.0)

    combined_ocr = "\n\n".join(page_texts)
    avg_ocr_conf = sum(page_confs) / len(page_confs) if page_confs else 0.0
    ocr_available = any(c > 0 for c in page_confs)

    # Build multi-image GPT-4o message
    rfx_ctx = rfx_context_summary(rfx_lines)
    folder_name = os.path.basename(folder_path.rstrip("/"))

    if ocr_available:
        ocr_block = (
            f"OCR text — all {len(all_files)} pages (Cloud Vision, avg conf={avg_ocr_conf:.2f}):\n"
            f"---\n{combined_ocr[:25000]}\n---\n\n"
            f"All page images follow. Cross-reference images and OCR text. "
            f"Where they conflict, trust image layout over OCR text."
        )
    else:
        ocr_block = (
            f"OCR text: unavailable. Extract all line items from the images alone."
        )

    preamble = (
        f"RFx canonical line items for matching:\n{rfx_ctx}\n\n"
        f"vendor_id: {vendor_id}\n"
        f"source_file: {folder_name} ({len(all_files)} page images)\n"
        f"document_version: {document_version}\n"
        f"superseded_by: {superseded_by}\n\n"
        f"NOTE: These are phone photographs of a vendor quote. Expect skew, glare, "
        f"partial occlusion. Low extraction_confidence is expected — do not force certainty.\n"
        f"Set source_location to 'page_N:row_M' (page number, approximate row).\n\n"
        f"{ocr_block}"
    )

    # Build content blocks: text preamble + one image_url per page
    content: list[dict] = [{"type": "text", "text": preamble}]
    for i, img_path in enumerate(all_files):
        with open(img_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        ext = os.path.splitext(img_path)[1].lower().lstrip(".")
        mime = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "gif": "image/gif", "webp": "image/webp",
            "tiff": "image/tiff", "tif": "image/tiff",
            "heic": "image/heic", "heif": "image/heif", "bmp": "image/bmp",
        }.get(ext, "image/jpeg")
        content.append({"type": "text", "text": f"[Page {i+1}: {os.path.basename(img_path)}]"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}", "detail": "high"},
        })

    print(f"  LLM vision extraction ({len(all_files)} images + {len(combined_ocr)} OCR chars)...")
    resp = client.chat.completions.create(
        model=MODEL_VISION,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM},
            {"role": "user", "content": content},
        ],
        temperature=0,
    )
    result = json.loads(resp.choices[0].message.content)
    result["extracted_at"] = datetime.now(timezone.utc).isoformat()
    result["ocr_confidence"] = round(avg_ocr_conf, 3) if ocr_available else None
    result["ocr_mode"] = "vision+image" if ocr_available else "image_only"
    result["rfx_id"] = rfx_id
    result["vendor_id"] = vendor_id
    result["vendor_name"] = vendor_name

    safe_name = output_name or re.sub(r"[^a-z0-9_]", "_", vendor_id.lower())
    out_path = os.path.join(DATA_DIR, "vendor_raw", f"{safe_name}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    n = len(result.get("extractions", []))
    ocr_str = f"{avg_ocr_conf:.3f}" if ocr_available else "n/a"
    print(f"  Written: {out_path} ({n} extractions, ocr_confidence={ocr_str})")
    return out_path


def extract_image_file(
    image_path: str,
    vendor_id: str,
    vendor_name: str,
    document_version: str = "v1",
    superseded_by: Optional[str] = None,
    output_name: Optional[str] = None,
    rfx_id: str = LANDSCAPE_RFX_ID,
) -> str:
    """
    Ad-hoc entry-point: extract a single image file (phone photo) into vendor_raw/.

    Returns the path to the written JSON file.

    Usage:
        python -c "
        import sys; sys.path.insert(0, 'src')
        from extract import extract_image_file
        extract_image_file('path/to/photo.jpg', 'V_NEW', 'New Vendor')
        "
    Or via CLI:
        python src/extract.py --image path/to/photo.jpg --vendor-id V_NEW --vendor-name "New Vendor"
    """
    os.makedirs(os.path.join(DATA_DIR, "vendor_raw"), exist_ok=True)
    rfx_lines = load_rfx_lines_for(rfx_id)
    source_file = os.path.basename(image_path)

    result = extract_image_doc(
        image_path=image_path,
        vendor_id=vendor_id,
        source_file=source_file,
        document_version=document_version,
        superseded_by=superseded_by,
        rfx_lines=rfx_lines,
    )
    result["rfx_id"] = rfx_id
    result["vendor_id"] = vendor_id
    result["vendor_name"] = vendor_name

    safe_name = output_name or re.sub(r"[^a-z0-9_]", "_", vendor_id.lower())
    out_path = os.path.join(DATA_DIR, "vendor_raw", f"{safe_name}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    n = len(result.get("extractions", []))
    ocr_conf = result.get("ocr_confidence", 0.0)
    ocr_conf_str = f"{ocr_conf:.3f}" if ocr_conf is not None else "n/a (billing)"
    print(f"  Written: {out_path} ({n} extractions, ocr_confidence={ocr_conf_str})")
    return out_path


def extract_single_doc(
    source_file: str,
    vendor_id: str,
    vendor_name: str,
    rfx_id: str,
    document_version: str = "v1",
    superseded_by: Optional[str] = None,
    output_name: Optional[str] = None,
    questionnaire_responses: Optional[list] = None,
) -> str:
    """
    Generic entry point: extract one vendor document for a given rfx_id.
    Loads rfx_lines from DB (or falls back to JSON for the landscape RFx).
    Writes vendor_raw/<output_name>.json and returns its path.

    Called by run_pipeline.py for single-file ingestion; can also be imported
    directly when wiring new intake flows.
    """
    rfx_lines = load_rfx_lines_for(rfx_id)
    os.makedirs(os.path.join(DATA_DIR, "vendor_raw"), exist_ok=True)
    fname = os.path.basename(source_file)

    if is_image_file(source_file):
        result = extract_image_doc(
            image_path=source_file,
            vendor_id=vendor_id,
            source_file=fname,
            document_version=document_version,
            superseded_by=superseded_by,
            rfx_lines=rfx_lines,
        )
    elif source_file.lower().endswith((".xlsx", ".xls")):
        rows = extract_boq_xlsx_rows(source_file)
        if rows:
            print(f"  Parsed {len(rows)} structured rows from xlsx")
            doc_text = boq_rows_to_extraction_prompt(rows, vendor_id, fname, rfx_lines)
        else:
            print(f"  WARNING: structured extraction got 0 rows, falling back to text dump")
            doc_text = extract_xlsx_text(source_file)
        result = extract_vendor_doc(
            doc_text=doc_text,
            vendor_id=vendor_id,
            source_file=fname,
            document_version=document_version,
            superseded_by=superseded_by,
            rfx_lines=rfx_lines,
        )
    else:
        doc_text = extract_pdf_text(source_file)
        result = extract_vendor_doc(
            doc_text=doc_text,
            vendor_id=vendor_id,
            source_file=fname,
            document_version=document_version,
            superseded_by=superseded_by,
            rfx_lines=rfx_lines,
        )

    result["rfx_id"] = rfx_id
    result["vendor_id"] = vendor_id
    result["vendor_name"] = vendor_name
    if questionnaire_responses is not None:
        result["questionnaire_responses"] = questionnaire_responses

    safe_name = output_name or re.sub(r"[^a-z0-9_]", "_", f"{rfx_id}_{vendor_id}_{document_version}".lower())
    out_path = os.path.join(DATA_DIR, "vendor_raw", f"{safe_name}.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    n = len(result.get("extractions", []))
    print(f"  Written: {out_path} ({n} extractions)")
    return out_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Path to a vendor document (PDF, XLSX, or image)")
    parser.add_argument("--image", help="(deprecated) Use --file instead")
    parser.add_argument("--vendor-id", help="Vendor ID (e.g. V_NEW)")
    parser.add_argument("--vendor-name", help="Vendor display name")
    parser.add_argument("--rfx-id", default=LANDSCAPE_RFX_ID, help="RFx project ID")
    parser.add_argument("--version", default="v1", help="Document version (default: v1)")
    parser.add_argument("--output-name", help="Output JSON basename (default: rfx_id_vendor_id)")
    args = parser.parse_args()

    target = args.file or args.image
    if target:
        if not args.vendor_id or not args.vendor_name:
            parser.error("--file requires --vendor-id and --vendor-name")
        extract_single_doc(
            source_file=target,
            vendor_id=args.vendor_id,
            vendor_name=args.vendor_name,
            rfx_id=args.rfx_id,
            document_version=args.version,
            output_name=args.output_name,
        )
    else:
        main()

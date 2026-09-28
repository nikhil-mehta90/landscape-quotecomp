"""
rfx_builder.py — Reconcile client BOQ.pdf + gpt BOQ.xlsx into rfx_lines.json.
Discrepancies are logged, never silently resolved.
"""
import json
import os
import re
import sys
import pdfplumber
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
KUKAS_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_data", "kukas")
RFX_001_OUTBOUND = os.path.join(
    os.path.dirname(__file__), "..", "rfx_workspace",
    "RFX-2026-LANDSCAPE-001 — Skydome Kukas Landscaping", "outbound"
)


# ── PDF text extraction ───────────────────────────────────────────────────────

def extract_pdf_text(path: str) -> str:
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            # Also try table extraction for structured PDFs
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    for row in table:
                        if row:
                            pages.append(" | ".join(str(c or "").strip() for c in row))
            else:
                pages.append(f"[PAGE {i+1}]\n{text}")
    return "\n".join(pages)


# ── Excel extraction ──────────────────────────────────────────────────────────

def extract_xlsx_sheets(path: str) -> str:
    xl = pd.ExcelFile(path)
    out = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet, header=None)
        out.append(f"[SHEET: {sheet}]")
        out.append(df.to_string(index=False, na_rep=""))
    return "\n".join(out)


# ── LLM parsing ───────────────────────────────────────────────────────────────

CLIENT_BOQ_SYSTEM = """
You are parsing a landscaping Bill of Quantities (BOQ) document for project SKYDOME KUKAS, Jaipur.
Extract every line item. Return a JSON object with key "line_items" containing an array.
Each element must have:
  line_id        — generate sequentially: L_SP_01 for soil/prep, L_TR_01 for trees,
                   L_SH_01 for shrubs, L_GC_01 for ground covers/grasses, L_LW_01 for lawn,
                   L_ST_01 for staking
  section        — one of: soil_prep | trees | shrubs | ground_covers | lawn | staking
  description    — verbatim item name from document
  species_name   — botanical/scientific name if a plant item, else null
  unit           — exactly as stated (Sqm / Nos / Cum / Kg / Each / Rmt / Lump)
  quantity       — numeric value only (no commas), null if not stated
  boq_unit_rate  — numeric unit rate in INR (no commas), null if not stated
  boq_total      — numeric line total in INR (no commas), null if not stated
  spec_notes     — any height/caliper/grade/quality specs (e.g. "2.0-2.5m ht, 50-60mm caliper"), null if none

If a value is genuinely unclear, use null rather than guessing.
Return {"line_items": [...]} only.
"""

GPT_BOQ_SYSTEM = """
You are parsing an AI-drafted landscaping BOQ (Excel). This is a noisy buyer-side draft, not authoritative.
Extract every line item. Return a JSON object with key "line_items" containing an array.
Each element: line_id, section, description, species_name, unit, quantity, boq_unit_rate, boq_total, spec_notes
(same schema as the client BOQ).
Return {"line_items": [...]} only.
"""


def parse_boq_with_llm(text: str, system_prompt: str, label: str) -> list:
    print(f"  Calling LLM to parse {label} ({len(text)} chars)...")
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Document text:\n\n{text[:80000]}"}
        ],
        temperature=0
    )
    raw = resp.choices[0].message.content
    data = json.loads(raw)
    # Try known key first, then any list value
    if "line_items" in data:
        return data["line_items"]
    if isinstance(data, list):
        return data
    for v in data.values():
        if isinstance(v, list):
            return v
    print(f"  WARNING: unexpected response structure, keys={list(data.keys())}")
    return []


# ── Reconciliation ────────────────────────────────────────────────────────────

RECONCILE_SYSTEM = """
You are reconciling two lists of landscaping BOQ line items:
1. CLIENT_BOQ — authoritative source of truth
2. GPT_DRAFT — AI-generated draft, potentially noisy

For each CLIENT_BOQ line, find the best matching GPT_DRAFT line and check for discrepancies.
Return a JSON object with key "line_items" containing an array.
Each element is the final canonical line item (use CLIENT_BOQ values as authoritative) plus a
"gpt_discrepancy" field (string or null):

gpt_discrepancy should describe the discrepancy if any of these differ meaningfully:
- unit (different unit of measure)
- quantity (>5% difference)
- description (significantly different name — species variant, merged/split items)
- item missing from GPT draft entirely

If GPT draft matches cleanly: gpt_discrepancy = null
If discrepancy exists: write a concise one-line description of what differs, e.g.:
  "GPT uses 'per plant' vs BOQ 'Sqm'; GPT qty=800 vs BOQ qty=464"
  "Item missing from GPT draft"
  "GPT merges Ficus benjamina and F. benjamina variegated into one line"

Return {"line_items": [...]} only.
"""


def reconcile(client_lines: list, gpt_lines: list) -> list:
    print("  Reconciling client BOQ vs GPT draft...")
    payload = {
        "CLIENT_BOQ": client_lines,
        "GPT_DRAFT": gpt_lines
    }
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": RECONCILE_SYSTEM},
            {"role": "user", "content": json.dumps(payload)[:80000]}
        ],
        temperature=0
    )
    raw = resp.choices[0].message.content
    data = json.loads(raw)
    if "line_items" in data:
        return data["line_items"]
    if isinstance(data, list):
        return data
    for v in data.values():
        if isinstance(v, list):
            return v
    print("  WARNING: reconcile returned unexpected structure; using client BOQ as-is")
    return client_lines  # fallback: return client lines untouched


# ── GPT BOQ summary for audit ─────────────────────────────────────────────────

def build_gpt_boq_json(gpt_lines: list, discrepancies: list) -> dict:
    """Store GPT BOQ as a top-level data artifact (not vendor_raw)."""
    return {
        "role": "buyer_draft",
        "source_file": "gpt_BOQ.xlsx",
        "note": "AI-generated draft BOQ. Treat as low-confidence input; reconcile against client BOQ.",
        "line_count": len(gpt_lines),
        "lines": gpt_lines,
        "discrepancies_with_client_boq": [
            d for d in discrepancies if d.get("gpt_discrepancy")
        ]
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    # 1. Extract text from client BOQ PDF
    client_boq_path = os.path.join(RFX_001_OUTBOUND, "client_BOQ.pdf")
    print(f"Extracting text from {client_boq_path}...")
    client_text = extract_pdf_text(client_boq_path)

    # 2. Extract text from GPT BOQ xlsx
    gpt_boq_path = os.path.join(KUKAS_DIR, "gpt_BOQ.xlsx")
    print(f"Extracting text from {gpt_boq_path}...")
    gpt_text = extract_xlsx_sheets(gpt_boq_path)

    # 3. Parse both with LLM
    client_lines = parse_boq_with_llm(client_text, CLIENT_BOQ_SYSTEM, "client BOQ.pdf")
    print(f"  Parsed {len(client_lines)} lines from client BOQ")

    gpt_lines = parse_boq_with_llm(gpt_text, GPT_BOQ_SYSTEM, "gpt BOQ.xlsx")
    print(f"  Parsed {len(gpt_lines)} lines from GPT BOQ")

    # 4. Reconcile
    canonical_lines = reconcile(client_lines, gpt_lines)
    print(f"  Canonical line count: {len(canonical_lines)}")

    # 5. Write rfx_lines.json
    rfx_path = os.path.join(DATA_DIR, "rfx_lines.json")
    with open(rfx_path, "w") as f:
        json.dump(canonical_lines, f, indent=2, ensure_ascii=False)
    print(f"  Written: {rfx_path}")

    # 6. Write gpt_boq.json (top-level data, not vendor_raw)
    gpt_boq_out = build_gpt_boq_json(gpt_lines, canonical_lines)
    gpt_path = os.path.join(DATA_DIR, "gpt_boq.json")
    with open(gpt_path, "w") as f:
        json.dump(gpt_boq_out, f, indent=2, ensure_ascii=False)
    print(f"  Written: {gpt_path}")

    # 7. Print discrepancy summary
    discrepant = [l for l in canonical_lines if l.get("gpt_discrepancy")]
    print(f"\n=== GPT BOQ discrepancies ({len(discrepant)} lines) ===")
    for l in discrepant:
        print(f"  {l['line_id']} [{l['description'][:50]}]: {l['gpt_discrepancy']}")

    print(f"\nrfx_builder done. {len(canonical_lines)} canonical lines.")


if __name__ == "__main__":
    main()

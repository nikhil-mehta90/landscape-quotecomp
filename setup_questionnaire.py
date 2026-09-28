"""
setup_questionnaire.py — One-shot questionnaire bootstrap for RFX-2026-LANDSCAPE-001.

1. Update rfx_projects with real scope + 12-month maintenance_duration.
2. Set vendor_metadata types for all 7 vendor IDs.
3. Generate dynamic questionnaires for two vendor types:
   - Landscape General Contractor (6 vendors)
   - Wholesale Plant Nursery (V_MAMTA2)
4. Extract questionnaire answers for every vendor.
"""

import json
import os
import re
import sqlite3
from datetime import datetime, timezone

from openai import OpenAI

DB_PATH = os.path.join(os.path.dirname(__file__), "data", "comparison.db")
MODEL = "gpt-4o"
RFX_ID = "RFX-2026-LANDSCAPE-001"
VENDOR_TYPE_CONTRACTOR = "Landscape General Contractor"
VENDOR_TYPE_NURSERY = "Wholesale Plant Nursery"

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Step 1: Update rfx_projects ───────────────────────────────────────────────

def update_rfx_project():
    terms = json.dumps({
        "maintenance_duration": "12 months",
        "maintenance_note": (
            "Grounded in BOQ: lawn line (L_LW_01) specifies 12-month maintenance, "
            "and gpt_boq.json contains a '12-month establishment maintenance' line."
        ),
        "site_area_sqm": 4250,
        "total_trees": 588,
        "site_location": "Kukas, Jaipur, Rajasthan",
        "water_source": "borewell/tanker — STP water possible, not confirmed",
    })
    description = (
        "Horticulture and landscaping works for a ~3-acre residential/commercial site at Kukas, "
        "Jaipur. Scope: 588 trees/palms/bamboo, ~3,846 shrubs/climbers, ~732 sqm ground covers, "
        "~2,460 sqm lawn, full soil preparation (4,250 sqm), anti-termite, staking (473 nos). "
        "12-month lawn establishment maintenance stated in BOQ. "
        "6 vendors evaluated, 73 BOQ lines."
    )
    conn = db()
    conn.execute(
        "UPDATE rfx_projects SET description=?, terms=? WHERE rfx_id=?",
        (description, terms, RFX_ID),
    )
    conn.commit()
    conn.close()
    print("[1] rfx_projects updated.")


# ── Step 2: Set vendor types ───────────────────────────────────────────────────

VENDOR_TYPES = {
    "V_JAI":          (VENDOR_TYPE_CONTRACTOR, "Jai Balaji"),
    "V_MAMTA":        (VENDOR_TYPE_CONTRACTOR, "Mamta Horticultural Services"),
    "V_PHOENIX":      (VENDOR_TYPE_CONTRACTOR, "Phoenix Landscaping"),
    "V_GREEN":        (VENDOR_TYPE_CONTRACTOR, "Green Thumbs Landscaping"),
    "V_DESERT_BLOOM": (VENDOR_TYPE_CONTRACTOR, "Desert Bloom Landscapes"),
    "V_GREEN_HORIZON":(VENDOR_TYPE_CONTRACTOR, "Green Horizon Nursery & Landscapes"),
    "V_MAMTA2":       (VENDOR_TYPE_NURSERY,    "Mamta Plant Nursery (photo quote)"),
}


def set_vendor_types():
    now = datetime.now(timezone.utc).isoformat()
    conn = db()
    for vid, (vtype, display) in VENDOR_TYPES.items():
        conn.execute("""
            INSERT INTO vendor_metadata (vendor_id, vendor_type, display_name, updated_at)
            VALUES (?,?,?,?)
            ON CONFLICT(vendor_id) DO UPDATE SET
                vendor_type=excluded.vendor_type,
                display_name=excluded.display_name,
                updated_at=excluded.updated_at
        """, (vid, vtype, display, now))
    conn.commit()
    conn.close()
    print(f"[2] vendor_metadata set for {len(VENDOR_TYPES)} vendors.")


# ── Questionnaire generation system prompt ────────────────────────────────────

_GEN_SYSTEM = """\
You are a procurement intelligence agent generating a project-specific vendor questionnaire.

Given: project context, a BOQ (bill of quantities line items), and the vendor type being assessed.
Output: a JSON object {"questions": [...]} with 3–5 questions total (3–4 Stage 1, at most 1 Stage 2).

BREVITY RULE — apply first, before selecting any dimension:
  A buyer attaches this to the initial RFQ and expects a vendor to answer it in the same reply as
  their quote, without extra effort. Real procurement questionnaires at this stage are short —
  3 to 5 questions. Before writing anything, mentally rank ALL candidate dimensions by how much
  the answer would change a shortlisting decision for THIS specific project. Return only the top
  3–4. Reject any dimension where the answer wouldn't meaningfully separate one vendor from
  another on this scope — generic questions about payment terms, general experience, or phasing
  that every vendor answers the same way are cut. When in doubt, cut.

STAGE RULES — apply before writing any question:
  stage_1_screening: Default for ALL questions. These are sent alongside the initial RFx and must
    be answerable from what a vendor already knows / has already committed to. They ask for CLAIMS,
    not proof. A vendor can answer from their own experience and existing commitments without
    gathering additional evidence.
    Examples: "Which region do you typically source [species] from?", "What is your estimated
    lead time for assembling 588 trees?", "Who would lead site execution — name the site manager
    and their experience?", "What payment terms do you require?", "Any BOQ items you'd flag or
    substitute?"
  stage_2_evaluation: Use ONLY when a project condition makes early evidence genuinely critical —
    e.g. a large accent tree species that is known to be scarce and whose unavailability would
    block the project. Still generate, but tagged stage_2. Ask for PROOF: "Provide photographs
    of the specific Ficus benjamina specimens you propose", "Submit a lab report for soil amendments."
    The same topic at claim-depth = Stage 1; at proof-depth = Stage 2.
  DEFAULT: If in doubt, Stage 1. Never generate a Stage 2 question just to sound thorough.
  RATIO: 3–4 Stage 1, at most 1 Stage 2 per questionnaire. Total must not exceed 5.

DESIGN RULES (apply in order):
1. Only generate dimensions materially relevant to this project + vendor type.
   Omit generic questions whose answers cannot distinguish vendors on this specific scope.
2. Ask at Stage 1 depth unless Stage 2 is explicitly justified:
   – Stage 1: "Which nurseries or regions do you source [species] from?" (claim)
   – Stage 2: "Provide photographs of proposed [species] stock" (proof — only if scarcity risk)
3. Reason from the actual BOQ:
   – Large accent trees (qty > 5, ht > 4m in spec_notes) → sourcing claim (S1); photos only if S2 justified.
   – Any species > 100 units → nursery network, aggregation region, rough lead time (S1).
   – Maintenance duration > 6 months → ask about staffing model and what maintenance includes (S1).
   – Arid/semi-arid location → ask which heat-tolerant amendments they'd propose, water-source approach (S1).
   – Soil amendment lines → ask sourcing region and whether they've worked with borewell/STP water (S1).
4. Apply vendor-type-specific focus (in combination with BOQ signals):
   – Landscape General Contractor: experience (comparable project claim, S1), site manager identity (S1),
     sourcing regions for key species (S1), rough procurement timeline (S1), BOQ concerns/substitutions (S1),
     phasing capability (S1), maintenance staffing model (S1), payment terms (S1).
   – Wholesale Plant Nursery: which species are in stock now (S1), source regions (S1), rough lead times (S1),
     hold policy until site is ready (S1), minimum order quantities (S1), replacement/rejection policy (S1).

Return this exact JSON structure:
{"questions": [
  {
    "q_id": "Q01",
    "dimension": "<experience|availability|sourcing|procurement_timeline|technical_suitability|site_compatibility|substitution|execution|logistics|phasing|quality_evidence|production_capability|commercial_assumptions|programme_commitment>",
    "stage": "stage_1_screening",
    "question": "<claim-level question, answerable from vendor's existing knowledge, specific to this project>",
    "pass_criteria": "<what constitutes a satisfactory Stage 1 answer>"
  }
]}

stage must be "stage_1_screening" or "stage_2_evaluation".
q_id values must be Q01, Q02, ... in sequence. Return the JSON object only, no prose.
"""


def generate_questionnaire(vendor_type: str) -> list[dict]:
    conn = db()
    boq_rows = conn.execute(
        "SELECT section, description, species_name, unit, quantity, boq_unit_rate, spec_notes "
        "FROM rfx_lines WHERE rfx_id=? ORDER BY section, line_id",
        (RFX_ID,),
    ).fetchall()
    proj_row = conn.execute(
        "SELECT name, description, terms FROM rfx_projects WHERE rfx_id=?", (RFX_ID,)
    ).fetchone()
    conn.close()

    boq_summary = "section | description | species | unit | qty | rate | spec_notes\n"
    for r in boq_rows:
        boq_summary += (
            f"{r['section']} | {r['description'][:60]} | {r['species_name'] or ''} | "
            f"{r['unit']} | {r['quantity'] or ''} | {r['boq_unit_rate'] or ''} | "
            f"{r['spec_notes'] or ''}\n"
        )

    terms = {}
    if proj_row and proj_row["terms"]:
        try:
            terms = json.loads(proj_row["terms"])
        except Exception:
            pass

    user_msg = (
        f"Project: {proj_row['name']}\n"
        f"Description: {proj_row['description']}\n"
        f"Site: {terms.get('site_location', 'Kukas, Jaipur, Rajasthan')}\n"
        f"Maintenance duration in BOQ: {terms.get('maintenance_duration', '12 months')}\n"
        f"Water source: {terms.get('water_source', 'borewell/tanker')}\n"
        f"Vendor type being assessed: {vendor_type}\n\n"
        f"BOQ ({len(boq_rows)} lines):\n{boq_summary}"
    )

    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _GEN_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )
    raw = json.loads(resp.choices[0].message.content)
    questions = raw.get("questions", [])
    if not isinstance(questions, list):
        questions = next((v for v in raw.values() if isinstance(v, list)), [])

    now = datetime.now(timezone.utc).isoformat()
    conn = db()
    conn.execute(
        "DELETE FROM rfx_questionnaire_definitions WHERE rfx_id=? AND vendor_type=?",
        (RFX_ID, vendor_type),
    )
    for q in questions:
        conn.execute("""
            INSERT OR REPLACE INTO rfx_questionnaire_definitions
            (rfx_id, vendor_type, q_id, dimension, stage, question, pass_criteria, generated_at)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            RFX_ID, vendor_type,
            q.get("q_id", ""), q.get("dimension", ""),
            q.get("stage", "stage_1_screening"),
            q.get("question", ""), q.get("pass_criteria", ""), now,
        ))
    conn.commit()
    conn.close()
    print(f"  Generated {len(questions)} questions for [{vendor_type}].")
    return questions


# ── Questionnaire extraction system prompt ────────────────────────────────────

_EXTRACT_SYSTEM = """\
You are extracting a vendor's questionnaire responses from a vendor document.
Use semantic/fuzzy matching — the vendor may not have answered in Q&A format;
look for relevant statements throughout the entire document text.

CONFIDENCE BAR (apply strictly):
A question is answered ONLY if the vendor document contains a statement that directly and
specifically addresses the question's subject matter — such that a procurement analyst reading
the document WITHOUT seeing the question would independently reach the same topic.
Connection requiring more than one inferential step = NOT answered.
Do NOT infer answers from adjacent context (e.g. "they work in Jaipur" ≠ "they understand
monsoon planting window").

Given:
1. Vendor document text
2. Questionnaire definitions (q_id, question, pass_criteria)
3. vendor_id

Return {"responses": [...]} where each element is:
{
  "q_id": "<from definitions>",
  "question": "<question text>",
  "answer": "<verbatim/close paraphrase from vendor doc, or '(no response)' if not addressed>",
  "answer_type": "text",
  "passes": 1 or 0 or null,
  "confidence": 0.0-1.0,
  "source_location": "<where in the document, or 'not found'>",
  "value_source": "extracted" or "unknown"
}

passes: 1=clearly meets criterion, 0=clearly fails, null=ambiguous/not addressed.
For questions not addressed: passes=null, value_source="unknown", answer="(no response)", confidence=0.
Return a JSON object {"responses": [...]} only.
"""


def extract_answers(vendor_id: str, vendor_type: str, doc_text: str, source_file: str) -> int:
    conn = db()
    q_defs = [dict(r) for r in conn.execute(
        "SELECT q_id, dimension, question, pass_criteria "
        "FROM rfx_questionnaire_definitions WHERE rfx_id=? AND vendor_type=? ORDER BY q_id",
        (RFX_ID, vendor_type),
    ).fetchall()]
    conn.close()

    if not q_defs:
        print(f"  [{vendor_id}] No questionnaire found for {vendor_type} — skipping.")
        return 0

    user_msg = (
        f"vendor_id: {vendor_id}\n"
        f"source_file: {source_file}\n\n"
        f"Questionnaire definitions:\n{json.dumps(q_defs, indent=2)}\n\n"
        f"Vendor document text:\n{doc_text[:60000]}"
    )

    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _EXTRACT_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )
    raw = json.loads(resp.choices[0].message.content)
    responses = raw.get("responses", [])
    if not isinstance(responses, list):
        responses = next((v for v in raw.values() if isinstance(v, list)), [])

    conn = db()
    conn.execute(
        "DELETE FROM questionnaire_responses WHERE vendor_id=? AND rfx_id=? AND vendor_type=?",
        (vendor_id, RFX_ID, vendor_type),
    )
    for r in responses:
        conn.execute("""
            INSERT INTO questionnaire_responses
            (vendor_id, source_file, q_id, question, answer, answer_type,
             passes, confidence, source_location, value_source, rfx_id, vendor_type)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            vendor_id, source_file,
            r.get("q_id"), r.get("question"),
            r.get("answer", "(no response)"),
            r.get("answer_type", "text"),
            r.get("passes"), r.get("confidence"),
            r.get("source_location"), r.get("value_source", "extracted"),
            RFX_ID, vendor_type,
        ))
    conn.commit()
    conn.close()
    answered = sum(1 for r in responses if r.get("value_source") == "extracted")
    print(f"  [{vendor_id}] {answered}/{len(responses)} questions answered.")
    return len(responses)


# ── Parse synthetic-questionnaire.md per vendor ───────────────────────────────

def parse_synthetic_questionnaire() -> dict[str, str]:
    path = os.path.join(os.path.dirname(__file__), "sample_data", "kukas", "synthetic-questionnaire.md")
    with open(path) as f:
        text = f.read()

    vendor_map = {
        "Jai Balaji": "V_JAI",
        "Mamta":      "V_MAMTA",
        "Phoenix":    "V_PHOENIX",
        "Green Thumbs": "V_GREEN",
    }

    # Split on "## Vendor: <name>" headers
    sections = re.split(r"^## Vendor: (.+)$", text, flags=re.MULTILINE)
    # sections = [pre-text, name1, body1, name2, body2, ...]
    result = {}
    for i in range(1, len(sections), 2):
        name = sections[i].strip()
        body = sections[i + 1].strip() if i + 1 < len(sections) else ""
        vid = vendor_map.get(name)
        if vid:
            result[vid] = body
    return result


# ── Load vendor document text ─────────────────────────────────────────────────

def load_vendor_docs() -> dict[str, tuple[str, str]]:
    """Returns {vendor_id: (doc_text, source_file)}."""
    base = os.path.dirname(__file__)
    vendors_dir = os.path.join(base, "landscape project")

    docs = {}

    # 4 original vendors — from synthetic-questionnaire.md
    synthetic = parse_synthetic_questionnaire()
    source = "sample_data/kukas/synthetic-questionnaire.md"
    for vid, text in synthetic.items():
        docs[vid] = (text, source)

    # Desert Bloom — call transcript + email (both have questionnaire signal)
    desert_parts = []
    for fname in ["desert_bloom_call_transcript.txt", "desert_bloom_quote_email.txt"]:
        fpath = os.path.join(vendors_dir, fname)
        with open(fpath) as f:
            desert_parts.append(f.read())
    docs["V_DESERT_BLOOM"] = (
        "\n\n--- CALL TRANSCRIPT ---\n\n" + desert_parts[0] +
        "\n\n--- WRITTEN QUOTE EMAIL (supersedes call for pricing) ---\n\n" + desert_parts[1],
        "landscape project/desert_bloom_call_transcript.txt + desert_bloom_quote_email.txt",
    )

    # Green Horizon — email only
    gh_path = os.path.join(vendors_dir, "green_horizon_quote_email.txt")
    with open(gh_path) as f:
        docs["V_GREEN_HORIZON"] = (f.read(), "landscape project/green_horizon_quote_email.txt")

    # V_MAMTA2 — photo quote: use source_snippets from extraction JSON as doc text
    mamta2_json = os.path.join(base, "data", "vendor_raw", "kukas", "rfx_2026_landscape_001_v_mamta2_v1.json")
    with open(mamta2_json) as f:
        m2 = json.load(f)
    snippets = [e.get("source_snippet", "") for e in m2.get("extractions", []) if e.get("source_snippet")]
    docs["V_MAMTA2"] = (
        "Source: photo quote (4 page images, OCR extracted). "
        "Line-item descriptions from quote:\n" + "\n".join(f"- {s}" for s in snippets[:30]),
        "Quote image_mamta2 (4 page images)",
    )

    return docs


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=== Questionnaire bootstrap for", RFX_ID, "===\n")

    print("[1] Updating rfx_projects...")
    update_rfx_project()

    print("[2] Setting vendor types...")
    set_vendor_types()

    print("[3] Generating questionnaires...")
    contractor_qs = generate_questionnaire(VENDOR_TYPE_CONTRACTOR)
    nursery_qs = generate_questionnaire(VENDOR_TYPE_NURSERY)

    print("\n[4] Loading vendor documents...")
    docs = load_vendor_docs()
    print(f"  Loaded docs for: {', '.join(docs.keys())}")

    print("\n[5] Extracting questionnaire answers...")
    for vid, vtype, _ in [
        ("V_JAI",          VENDOR_TYPE_CONTRACTOR, None),
        ("V_MAMTA",        VENDOR_TYPE_CONTRACTOR, None),
        ("V_PHOENIX",      VENDOR_TYPE_CONTRACTOR, None),
        ("V_GREEN",        VENDOR_TYPE_CONTRACTOR, None),
        ("V_DESERT_BLOOM", VENDOR_TYPE_CONTRACTOR, None),
        ("V_GREEN_HORIZON",VENDOR_TYPE_CONTRACTOR, None),
        ("V_MAMTA2",       VENDOR_TYPE_NURSERY,    None),
    ]:
        if vid not in docs:
            print(f"  [{vid}] No document loaded — skipping.")
            continue
        doc_text, source_file = docs[vid]
        extract_answers(vid, vtype, doc_text, source_file)

    print("\n=== Done ===")

    # Print summary
    print("\n--- Generated Questions: Landscape General Contractor ---")
    for q in contractor_qs:
        stage = q.get('stage', 'stage_1_screening')
        tag = "[S1]" if stage == "stage_1_screening" else "[S2]"
        print(f"  {q['q_id']} {tag} [{q['dimension']}] {q['question'][:80]}")

    print("\n--- Generated Questions: Wholesale Plant Nursery ---")
    for q in nursery_qs:
        stage = q.get('stage', 'stage_1_screening')
        tag = "[S1]" if stage == "stage_1_screening" else "[S2]"
        print(f"  {q['q_id']} {tag} [{q['dimension']}] {q['question'][:80]}")

    print("\n--- Coverage Summary (Stage 1 questions only) ---")
    conn = db()
    rows = conn.execute("""
        SELECT qr.vendor_id, vm.vendor_type,
               SUM(CASE WHEN qr.value_source='extracted'
                         AND qd.stage='stage_1_screening' THEN 1 ELSE 0 END) AS s1_answered,
               SUM(CASE WHEN qd.stage='stage_1_screening' THEN 1 ELSE 0 END) AS s1_total,
               SUM(CASE WHEN qr.value_source='extracted' THEN 1 ELSE 0 END) AS total_answered,
               COUNT(*) AS total
        FROM questionnaire_responses qr
        LEFT JOIN vendor_metadata vm ON qr.vendor_id = vm.vendor_id
        LEFT JOIN rfx_questionnaire_definitions qd
          ON qd.rfx_id=qr.rfx_id AND qd.vendor_type=qr.vendor_type AND qd.q_id=qr.q_id
        WHERE qr.rfx_id=? AND qr.vendor_type IS NOT NULL
        GROUP BY qr.vendor_id
        ORDER BY vm.vendor_type, qr.vendor_id
    """, (RFX_ID,)).fetchall()
    for r in rows:
        print(f"  {r['vendor_id']:20} [{r['vendor_type'][:30]}] "
              f"S1: {r['s1_answered']}/{r['s1_total']}  total: {r['total_answered']}/{r['total']}")
    conn.close()


if __name__ == "__main__":
    main()

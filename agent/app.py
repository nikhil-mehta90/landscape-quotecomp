"""
app.py — Streamlit analyst chat interface.

Architecture:
  - Two tools: run_query(code) for DB queries, lookup_benchmark(species, unit)
    for Serper web searches when an external price sanity-check is needed.
  - Tool-calling loop: ask GPT-4o with tools, execute any tool calls, feed results back,
    repeat until the model produces a final text response with no tool calls.
  - Every answer must cite the data it came from (source_ref shown in UI).
"""
from __future__ import annotations
import io
import os
import re
import json
import traceback
import textwrap
import sqlite3
from datetime import datetime, timezone
from typing import Optional
import requests
import streamlit as st
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"), override=False)

# ── Config ────────────────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "comparison.db")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.70"))
MAX_TOOL_ROUNDS = 8  # prevent infinite loops
_DATA_URI_RE = re.compile(r'!\[([^\]]*)\]\(data:image/[^;]+;base64,([A-Za-z0-9+/=\r\n\s]+)\)', re.DOTALL)

# Model routing — analyst agent uses two tiers.
# Heavy: questionnaire 4-state logic, coverage disclosure (Rule E), multi-step aggregation,
#        error recovery across retries, flagging/confidence analysis.
# Light: simple single-table lookups, list/show queries, benchmark result interpretation.
MODEL_HEAVY = "gpt-4o"
MODEL_LIGHT = "gpt-4o-mini"

_HEAVY_SIGNALS = frozenset({
    "questionnaire", "q_mortality", "q_water", "q_planting", "q_sourcing",
    "q_site", "q_maintenance", "passes", "pass ", "fail", "ambiguous",
    "coverage", "covered", "compare", "comparison", "total cost", "grand total",
    "flag", "flagged", "confidence", "anomal", "variance", "outlier", "mismatch",
    "missing", "estimate", "impute", "common subset", "all vendor",
    "step", "section by section",
    # distribution / ranking queries
    "percentile", "quartile", "distribution", "p20", "p50", "p80", "median",
    "percentil", "spread", "caliper", "trunk", "girth", "highest", "lowest",
    "best ", "most ", "rank", "ranking", "top ", "bottom ", "cheapest", "expensive",
})

_VENDORS = ["V_JAI", "V_MAMTA", "V_MAMTA2", "V_PHOENIX", "V_GREEN"]
_SECTIONS_ORDER = ["soil_prep", "trees", "shrubs", "ground_covers", "lawn", "staking"]

def _select_model(user_message: str) -> str:
    lower = user_message.lower()
    if any(sig in lower for sig in _HEAVY_SIGNALS):
        return MODEL_HEAVY
    return MODEL_LIGHT

def _openai_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        try:
            import streamlit as _st
            key = _st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            pass
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set — add it to Streamlit secrets or .env")
    return key

openai_client = OpenAI(api_key=_openai_api_key())

TRACE_LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "agent_traces.jsonl")

# ── Query helpers (inline subset — full helpers in db/query_helpers.py) ───────

def _db_conn():
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def _ensure_drafts_table():
    """Create rfx_drafts table if it doesn't exist yet (self-initializing, safe to call every run)."""
    conn = _db_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rfx_drafts (
                draft_id        TEXT PRIMARY KEY,
                rfx_name        TEXT NOT NULL,
                step            INTEGER NOT NULL DEFAULT 1,
                project_context TEXT,
                vision_text     TEXT,
                vision_summary  TEXT,
                staged_lines    TEXT,
                chat_history    TEXT,
                updated_at      TEXT NOT NULL
            )
        """)
        conn.commit()
    finally:
        conn.close()


def _ensure_app_tables():
    """
    Self-initializing schema additions that live in the app layer (not build_db.py).
    Safe to call every run — all statements are idempotent.
    """
    conn = _db_conn()
    try:
        # Add 'terms' and 'vision_summary' columns to rfx_projects
        cols = {r[1] for r in conn.execute("PRAGMA table_info(rfx_projects)").fetchall()}
        if "terms" not in cols:
            conn.execute("ALTER TABLE rfx_projects ADD COLUMN terms TEXT")
        if "vision_summary" not in cols:
            conn.execute("ALTER TABLE rfx_projects ADD COLUMN vision_summary TEXT")

        # Vendor metadata table: stores vendor_type tag used for questionnaire generation
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vendor_metadata (
                vendor_id       TEXT PRIMARY KEY,
                vendor_type     TEXT NOT NULL DEFAULT 'other',
                display_name    TEXT,
                updated_at      TEXT NOT NULL,
                rfx_id          TEXT,
                email           TEXT,
                website         TEXT,
                poc_name        TEXT,
                poc_phone       TEXT,
                inbound_folder  TEXT
            )
        """)
        # Backfill columns for older installs that created the table without them
        vm_cols = {r[1] for r in conn.execute("PRAGMA table_info(vendor_metadata)").fetchall()}
        for _col, _def in [
            ("rfx_id", "TEXT"), ("email", "TEXT"), ("website", "TEXT"),
            ("poc_name", "TEXT"), ("poc_phone", "TEXT"), ("inbound_folder", "TEXT"),
        ]:
            if _col not in vm_cols:
                conn.execute(f"ALTER TABLE vendor_metadata ADD COLUMN {_col} {_def}")

        # Dynamic questionnaire definitions (per rfx_id + vendor_type)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS rfx_questionnaire_definitions (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                rfx_id       TEXT NOT NULL,
                vendor_type  TEXT NOT NULL,
                q_id         TEXT NOT NULL,
                dimension    TEXT,
                question     TEXT NOT NULL,
                pass_criteria TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                UNIQUE(rfx_id, vendor_type, q_id)
            )
        """)

        # Add rfx_id + vendor_type to questionnaire_responses for joining with dynamic definitions
        qr_cols = {r[1] for r in conn.execute("PRAGMA table_info(questionnaire_responses)").fetchall()}
        if "rfx_id" not in qr_cols:
            conn.execute("ALTER TABLE questionnaire_responses ADD COLUMN rfx_id TEXT")
        if "vendor_type" not in qr_cols:
            conn.execute("ALTER TABLE questionnaire_responses ADD COLUMN vendor_type TEXT")

        # Add granularity column to vendor_extractions (for section-total / mixed vendors)
        ve_cols = {r[1] for r in conn.execute("PRAGMA table_info(vendor_extractions)").fetchall()}
        if "granularity" not in ve_cols:
            conn.execute("ALTER TABLE vendor_extractions ADD COLUMN granularity TEXT DEFAULT 'line_item'")

        # vendor_section_quotes: stores section-level totals for vendors who quote at package level
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vendor_section_quotes (
                id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                rfx_id              TEXT NOT NULL,
                vendor_id           TEXT NOT NULL,
                vendor_name         TEXT,
                source_file         TEXT,
                document_version    TEXT,
                superseded_by       TEXT,
                revision_type       TEXT DEFAULT 'quote',
                section             TEXT NOT NULL,
                total_amount        REAL,
                estimate_low        REAL,
                estimate_high       REAL,
                currency            TEXT DEFAULT 'INR',
                value_source        TEXT DEFAULT 'vendor_quote',
                package_description TEXT,
                source_snippet      TEXT,
                extraction_confidence REAL,
                flags               TEXT DEFAULT '[]',
                extracted_at        TEXT,
                UNIQUE(rfx_id, vendor_id, source_file, section)
            )
        """)

        conn.commit()
    finally:
        conn.close()


def _save_draft():
    """Upsert current wizard state to rfx_drafts. No-op if copilot_rfx_id not set."""
    draft_id = st.session_state.get("copilot_rfx_id")
    if not draft_id:
        return
    conn = _db_conn()
    try:
        conn.execute("""
            INSERT OR REPLACE INTO rfx_drafts
            (draft_id, rfx_name, step, project_context, vision_text,
             vision_summary, staged_lines, chat_history, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            draft_id,
            st.session_state.get("copilot_rfx_name_draft", ""),
            st.session_state.get("copilot_step", 1),
            json.dumps(st.session_state.get("copilot_project_context", {})),
            st.session_state.get("copilot_vision_text", ""),
            st.session_state.get("copilot_vision_summary", ""),
            json.dumps(st.session_state.get("copilot_staged_lines", [])),
            json.dumps(st.session_state.get("copilot_history", [])),
            datetime.now(timezone.utc).isoformat(),
        ))
        conn.commit()
    finally:
        conn.close()


def _delete_draft(draft_id: str):
    """Remove a draft from rfx_drafts."""
    if not draft_id:
        return
    conn = _db_conn()
    try:
        conn.execute("DELETE FROM rfx_drafts WHERE draft_id = ?", (draft_id,))
        conn.commit()
    finally:
        conn.close()


def _load_all_drafts() -> list:
    """Return all draft rows ordered by most recently updated."""
    conn = _db_conn()
    try:
        rows = conn.execute(
            "SELECT draft_id, rfx_name, step, updated_at FROM rfx_drafts ORDER BY updated_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def _history_to_display_messages(history: list) -> list:
    """Convert OpenAI-format history to Streamlit display messages, filtering tool entries."""
    out = []
    for m in history:
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", None)
        content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content})
    return out


def _load_draft(draft_id: str):
    """Restore wizard session state from a saved draft and rerun."""
    conn = _db_conn()
    try:
        row = conn.execute("SELECT * FROM rfx_drafts WHERE draft_id = ?", (draft_id,)).fetchone()
    finally:
        conn.close()
    if not row:
        return
    st.session_state["copilot_rfx_id"] = draft_id
    st.session_state["copilot_rfx_name_draft"] = row["rfx_name"]
    st.session_state["copilot_step"] = row["step"]
    st.session_state["copilot_project_context"] = json.loads(row["project_context"] or "{}")
    st.session_state["copilot_vision_text"] = row["vision_text"] or ""
    st.session_state["copilot_vision_summary"] = row["vision_summary"] or ""
    st.session_state["copilot_staged_lines"] = json.loads(row["staged_lines"] or "[]")
    st.session_state["copilot_history"] = json.loads(row["chat_history"] or "[]")
    st.session_state["copilot_messages"] = _history_to_display_messages(
        st.session_state["copilot_history"]
    )
    st.session_state["copilot_finalized_rfx_id"] = None
    st.session_state["copilot_review_mode"] = False
    st.rerun()


def _schema_summary() -> str:
    """Return a compact schema description for the system prompt."""
    return textwrap.dedent("""
    SQLite database: data/comparison.db

    Tables:
    - rfx_lines(line_id PK, section, description, species_name, unit, quantity,
                boq_unit_rate, boq_total, spec_notes, gpt_discrepancy)
      Sections: soil_prep | trees | shrubs | ground_covers | lawn | staking

    - vendor_extractions(id, rfx_id, vendor_id, vendor_name, source_file,
                         document_version, superseded_by,
                         line_id FK→rfx_lines, matched, match_confidence,
                         raw_unit_price, raw_unit, raw_currency,
                         normalized_unit_price, normalization_note,
                         quantity_quoted, freight_included, labor_included,
                         spec_grade_quoted, spec_grade_match,
                         source_snippet, source_location,
                         extraction_confidence, value_source, flags, extracted_at,
                         granularity)
      vendor_id values: V_JAI, V_MAMTA, V_MAMTA2, V_PHOENIX, V_GREEN, V_GREEN_HORIZON, V_DESERT_BLOOM
      V_MAMTA2 is Mamta Plant Nursery extracted from phone photos
      V_DESERT_BLOOM: section-total vendor (call transcript v1 verbal, email v2 firm). NO per-line rows.
      V_GREEN_HORIZON: mixed granularity — some tree species line-item, rest as section packages.
      document_version: v1/v2/v3. Phoenix v1 superseded_by='phoenix_r1.pdf'. Phoenix v3 = questionnaire update only (no pricing).
      value_source: 'extracted'|'inferred'|'verbal_estimate'|'unknown'. verbal_estimate = from call transcript.
      granularity: 'line_item' (default) | 'section_total' | 'mixed' | 'questionnaire_only'
      flags: JSON array string — e.g. '["spec_grade_mismatch","verbal_estimate_range"]'
      spec_grade_match: 'match'|'undergrade'|'overgrade'|'unknown'

    - vendor_section_quotes(id, rfx_id, vendor_id, vendor_name, source_file,
                            document_version, superseded_by, revision_type,
                            section, total_amount, estimate_low, estimate_high,
                            currency, value_source, package_description,
                            source_snippet, extraction_confidence, flags, extracted_at)
      section keys: soil_prep | trees | shrubs | ground_covers | lawn | staking |
                    trees_balance (aggregate avg-rate rows) | package_non_tree (multi-section package)
      value_source: 'vendor_quote' (firm) | 'verbal_estimate' (from call, may be superseded)
      estimate_low/high: non-null for verbal_estimate rows (range from call)
      revision_type: 'quote' | 'questionnaire_update'
      V_DESERT_BLOOM v1 (verbal) is superseded by v2 (firm email) — both rows present for audit.
      V_GREEN_HORIZON has section='trees_balance' (all-other-trees avg) and section='package_non_tree' (680k lump).
      Always filter WHERE (superseded_by IS NULL OR superseded_by = '') for current pricing.

    - questionnaire_definitions(q_id PK, question, pass_criteria)
      Legacy Kukas questions only (Q_MORTALITY_GUARANTEE, Q_WATER_RESPONSIBILITY,
      Q_PLANTING_WINDOW, Q_SOURCING, Q_SITE_ACCESS, Q_MAINTENANCE_HANDOVER).
      New RFxes use rfx_questionnaire_definitions instead.

    - rfx_questionnaire_definitions(id, rfx_id, vendor_type, q_id, dimension,
                                    question, pass_criteria, generated_at)
      Dynamic questionnaire per rfx_id + vendor_type. UNIQUE(rfx_id, vendor_type, q_id).
      dimension values: experience | availability | sourcing | procurement_timeline |
      technical_suitability | site_compatibility | substitution | execution | logistics |
      phasing | quality_evidence | production_capability | commercial_assumptions | programme_commitment

    - vendor_metadata(vendor_id PK, vendor_type, display_name, updated_at)
      Stores the buyer-assigned vendor type for each vendor.

    - questionnaire_responses(id, vendor_id, source_file, document_version, q_id,
                              question, answer, answer_type, passes, confidence,
                              source_location, value_source, rfx_id, vendor_type)
      passes: 1=pass, 0=fail, NULL=ambiguous/unknown. value_source='unknown' means no response.
      rfx_id + vendor_type are NULL for legacy Kukas rows; set for dynamic-questionnaire responses.
      To query dynamic responses: WHERE rfx_id=<id> AND vendor_type=<type>
      To query legacy responses:  WHERE rfx_id IS NULL OR rfx_id = ''

    Key rules when writing queries:
    - For CURRENT prices / totals: filter WHERE (superseded_by IS NULL OR superseded_by = '')
    - For REVISION DIFF (what changed in Phoenix v1 vs v2): do NOT apply the superseded_by filter.
      Instead join on vendor_id + line_id with explicit document_version filters:
        JOIN vendor_extractions ve2 ON ve1.line_id=ve2.line_id AND ve1.vendor_id=ve2.vendor_id
        WHERE ve1.vendor_id='V_PHOENIX' AND ve1.document_version='v1' AND ve2.document_version='v2'
      (no superseded_by filter — you need both rows)
      ALSO filter WHERE ve1.normalized_unit_price != ve2.normalized_unit_price to show only changed
      lines — then report the count of unchanged lines separately.
    - Never sum/average rows where value_source='unknown'
    - Never compare prices where spec_grade_match='undergrade' or 'overgrade' without noting it
    - 'cheapest' means lowest normalized_unit_price among matched=1 rows with ec >= 0.7

    RULE A — Totals always use BOQ quantity:
      For any total or cross-vendor cost comparison, multiply normalized_unit_price by
      rl.quantity (from rfx_lines — the CLIENT's required quantity). NEVER use
      ve.quantity_quoted (the vendor's own quoted quantity, which varies per vendor and
      makes cross-vendor totals incomparable).
      Correct:  SUM(ve.normalized_unit_price * rl.quantity)
      Wrong:    SUM(ve.normalized_unit_price * ve.quantity_quoted)

    RULE B — Questionnaire: always show 4 states, never a single pass-count:
      For ANY questionnaire summary, compute all four states per vendor per question:
        PASS        (passes = 1)
        FAIL        (passes = 0)
        AMBIGUOUS   (passes IS NULL AND value_source != 'unknown')
        NO_RESPONSE (value_source = 'unknown')
      Never say "passed N of 6" without also stating how many failed, were ambiguous, or had
      no response. Example pivot query:
        SELECT vendor_id,
          SUM(CASE WHEN passes=1 THEN 1 ELSE 0 END) as pass_count,
          SUM(CASE WHEN passes=0 THEN 1 ELSE 0 END) as fail_count,
          SUM(CASE WHEN passes IS NULL AND value_source!='unknown' THEN 1 ELSE 0 END) as ambiguous,
          SUM(CASE WHEN value_source='unknown' THEN 1 ELSE 0 END) as no_response
        FROM questionnaire_responses GROUP BY vendor_id

    RULE C — Report uncovered lines when filtering vendors:
      After any query that filters to a subset of vendors (e.g. questionnaire-passing only),
      run a SECOND query to count rfx_lines that have zero qualifying vendor options:
        SELECT COUNT(*) as uncovered_lines FROM rfx_lines rl
        WHERE NOT EXISTS (
          SELECT 1 FROM vendor_extractions ve
          WHERE ve.line_id=rl.line_id AND ve.vendor_id IN (<qualifying_vendors>)
            AND ve.matched=1 AND ve.extraction_confidence>=0.7
            AND (ve.superseded_by IS NULL OR ve.superseded_by='')
        )
      Report this count prominently — "X of 69 lines have no qualifying vendor."

    RULE D — Flagged/low-confidence lines: show all vendors first:
      When asked about a flagged or low-confidence line, FIRST run a query showing ALL
      vendors for that line (no confidence filter), with their confidence, flags, and price.
      Then state explicitly which vendors are excluded from comparison and why. Do NOT
      silently apply ec>=0.7 and report only survivors.

    RULE F — Percentile / distribution analysis: always use pandas, never SQL aggregates:
      SQLite has no PERCENTILE_CONT or MEDIAN aggregate. Any time a question asks for
      percentiles, quartiles, deciles, or a distribution (p20/p50/p80, median, IQR, histogram):
      1. Pull the raw rows with pd.read_sql_query.
      2. Compute all stats in Python using pandas .quantile(), .describe(), or numpy.
      3. Always report n (sample size) alongside any distribution stat — a percentile without
         a count is uninterpretable.
      4. When computing per-group distributions (e.g. per-vendor price spread), always check
         whether the groups have comparable coverage/mix before comparing percentiles.
         If group sizes or composition differ materially, flag it explicitly:
         e.g. "V_MAMTA2 p50 reflects only 23 lines (trees only); other vendors cover 55–56
         lines including cheaper ground-cover lines — distributions are not directly comparable."
      Example — per-vendor unit price percentiles:
        df = pd.read_sql_query("SELECT vendor_id, normalized_unit_price FROM ...", conn)
        result = df.groupby('vendor_id')['normalized_unit_price'].describe(
            percentiles=[.2, .5, .8]).round(0)

    RULE G — Quantity / volume aggregation across mixed units: count lines, don't sum qty:
      Line items across sections (soil prep, plants, staking, materials) carry quantities
      in different units (Nos, Sqm, Kg, Rmt, etc.). Summing rl.quantity across sections
      produces a dimensionless number with no physical meaning.
      Rules:
      1. NEVER sum quantities across lines with different units.
      2. When asked "which vendor covers the most qty/volume of X", first check whether
         all matched lines share a single unit. If yes, sum is valid for that unit.
         If units are mixed, report COUNT of lines covered instead, with a note:
         e.g. "Coverage: 9/12 lines — summing quantities not meaningful (mixed units:
         Sqm, Kg, Nos)."
      3. For comparing vendor reach on a multi-unit scope, report: lines covered / total
         lines in scope, broken down by unit type if useful.

    RULE H — Ambiguous superlatives ("best", "cheapest", "most"): state interpretation, cover both angles:
      When a question uses "best", "most", "cheapest", "highest" without specifying the
      dimension, do not silently pick one interpretation. Instead:
      1. State the interpretation(s) you will evaluate: coverage (lines quoted) and/or
         price (unit cost) and/or spec quality (from spec_parsed).
      2. Answer both where they differ — a vendor that covers the most lines may not
         have the best price, and vice versa.
      3. For spec-based questions ("highest caliper", "widest spread"), clarify whether
         the question is about the BOQ spec (fixed per line, same for all vendors) or
         about which vendor quoted lines with the highest spec. These are different queries.

    RULE I — Quality filters must be declared and quantified: never apply silently:
      Whenever you apply ANY quality or confidence filter to price data — including
      extraction_confidence >= 0.7, matched=1, value_source != 'unknown', or any
      custom threshold — you MUST, before reporting price results:
      1. State the filter(s) applied: e.g. "filtering to matched=1 AND ec >= 0.7".
      2. Report how many rows (vendor-line pairs) were EXCLUDED by that filter vs the
         total available: e.g. "excludes 4 of 23 rows (17%) for this vendor".
      3. If the excluded rows are concentrated in one vendor (shifting that vendor's
         apparent price distribution), flag it explicitly:
         e.g. "V_JAI has 8 low-ec rows excluded — their reported p50 reflects only
         high-confidence lines and may not represent their full quote."
      Implementation pattern — always run the unfiltered count first, then compare:
        df_all = pd.read_sql_query("SELECT vendor_id, COUNT(*) as total ...", conn)
        df_filtered = pd.read_sql_query("SELECT vendor_id, COUNT(*) as kept ... WHERE ec>=0.7 ...", conn)
        # Merge and report excluded = total - kept per vendor before showing price stats.
      This rule applies to ALL price queries — totals, percentiles, averages, rankings.
      CRITICAL: if a filtered query returns 0 rows, you MUST run the unfiltered version
      and report what's there before saying "no quotes available". Zero results after
      filtering ≠ zero quotes in the database — never conflate them.

    RULE E — Coverage disclosure for any vendor total or cross-vendor comparison:
      Before computing any vendor total or cross-vendor cost comparison:
      1. Always report each vendor's LINE COVERAGE — how many lines that vendor actually
         quoted (value_source != 'unknown') vs total rfx_lines (69 total).
         Example: "V_JAI: 69/69 lines (100%), V_MAMTA2: 34/69 lines (49%)"
      2. DEFAULT to comparing on the COMMON SUBSET — only lines where ALL vendors being
         compared have a non-unknown quote — unless the user explicitly asks for an
         estimated all-in total including imputed values.
         Common-subset query (replace placeholders):
           SELECT line_id FROM vendor_extractions
           WHERE vendor_id IN (<vendor_list>) AND value_source != 'unknown'
             AND (superseded_by IS NULL OR superseded_by = '')
           GROUP BY line_id HAVING COUNT(DISTINCT vendor_id) = <n_vendors>
      3. NEVER present a partial total as if it were a full quote. Always label totals
         with the coverage scope, e.g. "Total over 34 common lines (out of 69 RFx lines)".

    RULE J — Entity resolution: ALWAYS call resolve_lines before SQL involving any name:
      For EVERY query that mentions a plant, species, item, section category, or vendor
      by any name — even "trees", "ground cover", "staking", "Mamta", "palms", etc. —
      call resolve_lines() before writing any SQL. This is non-negotiable; there are NO
      exceptions for "obvious" names you think you already know.

      NEVER write LIKE '%anything%' SQL for species, items, sections, or vendors directly.
      ALL name-to-ID mapping must go through resolve_lines.

      Line/item/section queries: resolve_lines(query, entity_type="line")  → line_ids
      Vendor queries:            resolve_lines(query, entity_type="vendor") → vendor_ids
      Use returned IDs in run_query: WHERE line_id IN (...) / WHERE vendor_id IN (...)

      Workflow (no step may be skipped):
        1. Call resolve_lines(query, entity_type=...)
        2. Read the "strategy" and "needs_clarification" fields from the result
        3. If needs_clarification=true OR total_found > 3: list the options to the user
           and ASK which one(s) they mean — do NOT run SQL yet
        4. If matches=[]: show catalog_by_section / all_vendors, ask the user to clarify
        5. Only if exactly 1 match, or user has confirmed their choice: run SQL with exact IDs

      INTERPRETATION DISCLOSURE (mandatory when strategy ≠ "exact"):
        The resolve_lines result always contains an "interpretation_disclosure" field.
        When that field is not null, you MUST copy it verbatim as the first line of your
        answer, before any data or table. Do not paraphrase, summarize, or omit it.
        Never silently substitute and answer as if the user's exact wording matched.

    RULE K — Line context block: always show spec + qty + BOQ rate before vendor prices:
      Whenever a query is about a specific resolved line item — price comparison, cheapest
      quote, anomaly explanation, flag investigation, or any single-line focus — you MUST
      open the answer with a compact context block before the vendor table. This is required
      even when the user did not explicitly ask "what's the spec."

      The context block must include (omit fields that are null or zero):
        • Description and species_name (if different from description)
        • spec_notes (the raw spec string from the RFx)
        • Relevant parsed spec fields from spec_parsed JSON (use json_extract or parse in Python):
            - Plants/trees/shrubs/bamboo: height_low/height_high, spread_low/spread_high,
              caliper_low_mm/caliper_high_mm, trunk_dia_low_mm/trunk_dia_high_mm,
              stem_dia_low_mm/stem_dia_high_mm, culm_dia_low_mm/culm_dia_high_mm
            - Ground covers: height_low/height_high, pot_dia_mm
            - Soil prep: pit_dia_m, pit_depth_m, depth_mm
            - Materials: thickness_mm, length_m, area_sqm
        • quantity and unit (from rfx_lines)
        • boq_unit_rate (client's own reference rate for this line)

      Format example (adapt fields to what's non-null for the line):
        **Line: L_SH_08 — Bauhinia tomentosa**
        Spec: 2.0–2.5 m ht, spread 0.75 m | Qty: 92 Nos. | BOQ ref: ₹1,250/unit

      Rationale: flags like spec_grade_mismatch are uninterpretable without knowing what
      grade the RFx actually calls for. The analyst needs to see spec + qty + reference
      rate alongside vendor numbers to judge whether a cheap quote is genuine or a
      downgraded substitution.

      Query pattern for the context block (include in the same run_query call as vendor prices):
        SELECT rl.line_id, rl.description, rl.species_name, rl.spec_notes, rl.spec_parsed,
               rl.quantity, rl.unit, rl.boq_unit_rate
        FROM rfx_lines rl WHERE rl.line_id = '<resolved_id>'

    IMPORTANT — text/keyword search across rfx_lines:
      Always use the 'search_text' column for any text/keyword search on rfx_lines.
      It is a pre-built concatenation of description + species_name + spec_notes for
      every line. Never search spec_notes or description directly for filtering.
        WHERE rl.search_text LIKE '%<term>%'
      vendor_extractions has NO species or plant name column — never search ve.*
      for a plant name. Always query rfx_lines.search_text first, then join vendor_extractions
      on the matched line_ids.

      FUZZY MATCH PROTOCOL — always apply this sequence before declaring "not found":
      1. Try LOWER(rl.search_text) LIKE LOWER('%<full user term>%')
      2. If 0 rows: split into words, try the longest distinctive word alone
         (e.g. "Bahunia tomentosa" → try '%tomentosa%', then '%bauhinia%')
      3. If still 0 rows: try each word individually
      4. Only after all attempts return 0 rows, report "no such item in the RFx"

      SPELLING NOTE: Users often misspell species names (e.g. "Bahunia" for "Bauhinia",
      "Ficus bengalesis" for "Ficus benghalensis"). Never reject on exact-match failure —
      always try partial/fuzzy LIKE on the most distinctive substring.

    IMPORTANT — numeric spec filtering (height, span, caliper, pit size, etc.):
      Every rfx_line has a 'spec_parsed' column containing a JSON blob of pre-extracted
      numeric specs. Use json_extract() for all numeric spec filters — never LIKE on spec_notes.

      Available fields (only fields that apply are populated; rest are null):
        height_low, height_high          — plant height range in metres
        spread_low, spread_high          — canopy/clump spread range in metres
        caliper_low_mm, caliper_high_mm  — trunk caliper range (mm), trees
        trunk_dia_low_mm, trunk_dia_high_mm — palm trunk diameter range (mm)
        stem_dia_low_mm, stem_dia_high_mm   — stem diameter range (mm), some trees
        culm_dia_low_mm, culm_dia_high_mm   — bamboo culm diameter range (mm)
        clump_spread_low, clump_spread_high — bamboo clump spread range (m)
        pot_dia_mm                       — pot diameter (mm), ground covers

      CRITICAL: all diameter/caliper fields come in _low/_high pairs. There is NO
      caliper_mm, trunk_dia_mm or culm_dia_mm singular field — those do not exist.
      Always use caliper_low_mm / caliper_high_mm etc.

      Example — trees/shrubs with spread > 2m:
        result = pd.read_sql_query('''
          SELECT rl.line_id, rl.description,
                 json_extract(rl.spec_parsed,'$.spread_low') AS spread_low,
                 json_extract(rl.spec_parsed,'$.spread_high') AS spread_high
          FROM rfx_lines rl
          WHERE json_extract(rl.spec_parsed,'$.spread_low') >= 2.0
          ORDER BY spread_low DESC
        ''', conn)

      Example — trees taller than 4m:
        WHERE json_extract(rl.spec_parsed,'$.height_low') >= 4.0

      Example — trees with caliper spec ≥ 75mm:
        WHERE json_extract(rl.spec_parsed,'$.caliper_low_mm') >= 75

      Example — caliper min/max for a specific line:
        SELECT json_extract(rl.spec_parsed,'$.caliper_low_mm') AS caliper_min_mm,
               json_extract(rl.spec_parsed,'$.caliper_high_mm') AS caliper_max_mm
        FROM rfx_lines rl WHERE rl.line_id = 'L_TR_03'

    Example query patterns (always use pd.read_sql_query, never bare SQL):
      # Vendor totals using BOQ quantity (RULE A):
      result = pd.read_sql_query('''
        SELECT ve.vendor_id, ve.vendor_name,
               SUM(ve.normalized_unit_price * rl.quantity) as total_cost
        FROM vendor_extractions ve JOIN rfx_lines rl ON ve.line_id=rl.line_id
        WHERE ve.matched=1 AND ve.extraction_confidence>=0.7
          AND (ve.superseded_by IS NULL OR ve.superseded_by='')
          AND ve.value_source!='unknown'
        GROUP BY ve.vendor_id, ve.vendor_name ORDER BY total_cost
      ''', conn)

      # Questionnaire 4-state summary (RULE B):
      result = pd.read_sql_query('''
        SELECT vendor_id,
          SUM(CASE WHEN passes=1 THEN 1 ELSE 0 END) as pass_count,
          SUM(CASE WHEN passes=0 THEN 1 ELSE 0 END) as fail_count,
          SUM(CASE WHEN passes IS NULL AND value_source!='unknown' THEN 1 ELSE 0 END) as ambiguous,
          SUM(CASE WHEN value_source='unknown' THEN 1 ELSE 0 END) as no_response
        FROM questionnaire_responses GROUP BY vendor_id ORDER BY vendor_id
      ''', conn)

      # All vendors for a flagged line — no confidence filter (RULE D):
      result = pd.read_sql_query('''
        SELECT ve.vendor_id, ve.normalized_unit_price, ve.extraction_confidence,
               ve.flags, ve.value_source, ve.source_snippet
        FROM vendor_extractions ve
        WHERE ve.line_id='L_ST_01' AND (ve.superseded_by IS NULL OR ve.superseded_by='')
        ORDER BY ve.normalized_unit_price
      ''', conn)
    """).strip()


# ── Tool definitions ──────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "resolve_lines",
            "description": (
                "Resolve any natural language reference — plant species, item description, "
                "vendor name/alias — to exact database IDs before running SQL. "
                "ALWAYS call this tool FIRST before any run_query that references a specific "
                "plant, item, or vendor by name or description. "
                "Uses a 3-strategy cascade (exact → fuzzy word-level → LLM semantic) that "
                "handles typos, misspellings, abbreviations, partial names, common-name vs. "
                "scientific-name mismatches, and short vendor aliases. "
                "Set entity_type='line' (default) for plant/item lookups — returns line_ids. "
                "Set entity_type='vendor' for vendor name lookups — returns vendor_ids. "
                "Use the returned IDs in subsequent run_query with WHERE id IN (...). "
                "If needs_clarification=true, show the options to the user and ask which one. "
                "If matches is empty, show the catalog/vendor list and ask for clarification."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The user's input verbatim — typos and misspellings are "
                            "intentional, do not correct before passing. "
                            "Examples: 'Bahunia tomentosa', 'bamboo staking', 'Jay Balaji', "
                            "'Grean Thumbs', 'ground cover', 'soil preparation pits 0.9m'."
                        )
                    },
                    "entity_type": {
                        "type": "string",
                        "enum": ["line", "vendor"],
                        "description": (
                            "'line' (default) — resolve to rfx_line IDs for plant/item queries. "
                            "'vendor' — resolve to vendor_ids when the user names a vendor."
                        )
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_query",
            "description": (
                "Execute Python code that queries comparison.db using sqlite3 or pandas. "
                "The code runs in a sandboxed exec() with 'conn' (sqlite3.Connection) and "
                "'pd' (pandas) available. The code MUST assign its result to a variable "
                "named 'result' — a string, DataFrame, dict, or list. "
                "CRITICAL: 'code' must be valid Python — NEVER pass raw SQL. "
                "Always wrap SQL in pd.read_sql_query('''...''', conn) and assign to result. "
                "Always filter out superseded rows (superseded_by IS NULL OR superseded_by = ''). "
                "Never include value_source='unknown' rows in totals or price comparisons. "
                "Always use rl.quantity (from rfx_lines) for totals, not ve.quantity_quoted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Valid Python code that sets a variable named 'result'. "
                            "Example: result = pd.read_sql_query('SELECT ...', conn). "
                            "NEVER send raw SQL — it will fail with SyntaxError."
                        )
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One sentence: what this query answers."
                    }
                },
                "required": ["code", "rationale"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_benchmark",
            "description": (
                "Search the web for market price benchmarks for a plant species or landscape "
                "construction item. Use this when you need to sanity-check whether a vendor "
                "price is plausible, when all DB prices for a line are low-confidence, or when "
                "the user asks 'should I trust this price?' Returns web search snippets with "
                "price mentions from nurseries and landscaping suppliers in India."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "species": {
                        "type": "string",
                        "description": (
                            "Plant species name or item description "
                            "(e.g. 'Agave attenuata', 'bamboo tree staking guying')"
                        )
                    },
                    "unit": {
                        "type": "string",
                        "description": "Unit of measurement (e.g. 'Nos', 'Sqm', 'Kg', 'Rmt')"
                    },
                    "context": {
                        "type": "string",
                        "description": (
                            "Optional extra context (e.g. 'size 2-2.5m', 'Jaipur nursery 2025'). "
                            "Include size/spec from boq_spec_notes when relevant."
                        )
                    }
                },
                "required": ["species", "unit"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_reference",
            "description": (
                "Free-text web search for spec conventions, pricing norms, or industry "
                "standards — anything that isn't in past RFx data and the model shouldn't "
                "guess. Use when the user asks 'what's typical for X?' or references a "
                "category/spec the DB doesn't contain. Returns web snippets + source links."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Free-text question or search phrase. "
                            "E.g. 'tree guard specs for large trees India landscaping', "
                            "'standard planting pit size shrubs', "
                            "'bamboo culm diameter staking Rajasthan'."
                        )
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "render_chart",
            "description": (
                "Draw a bar chart for visual vendor or category comparisons. "
                "ONLY use when the user explicitly asks to 'show a chart', 'plot', 'visualise', "
                "or 'compare visually'. Do NOT use for text or table answers — those go via run_query. "
                "Workflow: call run_query first to get the data, then call render_chart with the "
                "labels and values extracted from the result. "
                "Aggregation rule: when the data has 10+ line items (e.g. all BOQ lines), "
                "aggregate to section level first (GROUP BY section) before charting — "
                "never plot raw line-level data with 10+ bars. "
                "The chart automatically adds a median reference line (dashed red) per group "
                "so above/below-pack vendors are immediately visible."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Chart title, e.g. 'Total Quote by Vendor (INR)'"
                    },
                    "series": {
                        "type": "array",
                        "description": (
                            "One or more data series. "
                            "Simple bar (vendor totals): one item, labels=vendor names, values=totals. "
                            "Grouped bars (category × vendor): one item per vendor, labels=category names. "
                            "Values must be numbers — exclude null/unknown rows before passing."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string", "description": "Series label (e.g. vendor name or metric)"},
                                "labels": {"type": "array", "items": {"type": "string"}, "description": "X-axis bar labels"},
                                "values": {"type": "array", "items": {"type": "number"}, "description": "Bar heights"}
                            },
                            "required": ["name", "labels", "values"]
                        }
                    },
                    "y_label": {
                        "type": "string",
                        "description": "Y-axis label, e.g. 'Total (INR)' or 'Cost per unit'"
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One sentence: what this chart shows."
                    }
                },
                "required": ["title", "series"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "render_section_collage",
            "description": (
                "Render a multi-panel chart collage: one subplot per BOQ section, each showing "
                "vendor quote bars with a BOQ benchmark line (amber dashed) and peer median line "
                "(red dashed). Use this as the DEFAULT chart when the user asks to compare vendors, "
                "'show a chart', 'visualise', or 'compare visually' — it gives a section-by-section "
                "view in one glance. Only fall back to render_chart for a specific custom metric or "
                "single-section drill-down the user explicitly requests."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rfx_id": {
                        "type": "string",
                        "description": "The active RFx ID (e.g. 'RFX-2026-LANDSCAPE-001')."
                    }
                },
                "required": ["rfx_id"]
            }
        }
    }
]


# ── Tool execution ────────────────────────────────────────────────────────────

# SQL keywords that must be preceded by whitespace — fixes bare-newline-stripping
# that produces "vendor_extractions veJOIN rfx_lines" when lines are concatenated.
_SQL_KEYWORD_RE = re.compile(
    r'(?<!\s)(FROM|JOIN|INNER\s+JOIN|LEFT\s+JOIN|RIGHT\s+JOIN|WHERE|GROUP\s+BY'
    r'|HAVING|ORDER\s+BY|LIMIT|UNION|ON\s|AND\s|OR\s|SET\s|INTO\s)',
    re.IGNORECASE,
)


def _normalize_code(code: str) -> str:
    """
    Normalize model-generated code before exec:
    - Auto-wrap bare SQL (no 'result =' assignment) into pd.read_sql_query so the
      model's occasional raw-SQL submissions don't hard-fail with SyntaxError.
    - Replace bare newlines inside SQL string literals with a space so that
      keyword tokens on the next line don't merge with the previous token
      (avoids "vendor_extractions veJOIN").
    - Replace Unicode comparison operators with ASCII equivalents so SQLite
      doesn't reject them (≥ → >=, ≤ → <=, ≠ → !=).
    """
    # Auto-wrap bare SQL: if the code starts with a SQL keyword and has no 'result'
    # assignment, wrap it so exec() doesn't get a SyntaxError.
    # Use triple-double-quotes so SQL values like section='trees' don't produce ''''
    # (triple-close + trailing single-quote) which Python parses as a SyntaxError.
    stripped = code.strip()
    if re.match(r'^(SELECT|WITH|PRAGMA)\b', stripped, re.IGNORECASE) and 'result' not in code:
        code = f'result = pd.read_sql_query("""{stripped}""", conn)'

    # Convert model-generated pd.read_sql_query('''...''', conn) to use triple-double-quotes
    # for the same reason: SQL ending in a single-quote creates '''' which is a SyntaxError.
    # SQL never uses double-quotes for string literals (SQLite uses single-quotes), so """ is safe.
    code = re.sub(
        r"pd\.read_sql_query\('''(.*?)''',\s*conn\)",
        lambda m: 'pd.read_sql_query("""' + m.group(1) + '""", conn)',
        code,
        flags=re.DOTALL,
    )

    # Unicode operator substitution (safe across the whole code string)
    code = code.replace("≥", ">=").replace("≤", "<=").replace("≠", "!=")

    # Ensure SQL keywords inside string literals are preceded by whitespace.
    # We look for a newline (possibly with leading spaces) immediately before
    # an SQL keyword and inject a space when it would otherwise be absent.
    # Pattern: newline + optional spaces + keyword token, replace with
    # single-space + keyword so the surrounding SQL stays valid.
    code = re.sub(
        r'\n[ \t]*(FROM|JOIN|INNER JOIN|LEFT JOIN|RIGHT JOIN|WHERE|GROUP BY'
        r'|HAVING|ORDER BY|LIMIT|UNION|ON |AND |OR )',
        lambda m: ' ' + m.group(1),
        code,
        flags=re.IGNORECASE,
    )
    return code


def _resolve_vendor(query: str, rfx_id: str | None = None) -> tuple:
    """
    Same 3-strategy cascade for vendor names/aliases → vendor_ids.
    Vendors: V_GREEN/Green Thumbs, V_JAI/Jai Balaji, V_MAMTA/Mamta Horticultural Works,
             V_MAMTA2/Mamta Plant Nursery, V_PHOENIX/Phoenix Landscaping.
    """
    conn = _db_conn()
    try:
        if rfx_id:
            vendors = conn.execute(
                "SELECT DISTINCT vendor_id, vendor_name FROM vendor_extractions "
                "WHERE rfx_id=? AND (superseded_by IS NULL OR superseded_by='') ORDER BY vendor_id",
                (rfx_id,),
            ).fetchall()
        else:
            vendors = conn.execute(
                "SELECT DISTINCT vendor_id, vendor_name FROM vendor_extractions "
                "WHERE superseded_by IS NULL OR superseded_by='' ORDER BY vendor_id"
            ).fetchall()

        def _vmatch(r, method, confidence, extra=None):
            d = {"vendor_id": r["vendor_id"], "vendor_name": r["vendor_name"],
                 "method": method, "confidence": confidence}
            if extra:
                d.update(extra)
            return d

        q = query.strip().lower()

        # Strategy 1 — exact substring match on vendor_id or vendor_name
        exact = [r for r in vendors if q in r["vendor_id"].lower() or q in r["vendor_name"].lower()]
        def _vdisc(strategy, matches):
            if strategy == "exact" or not matches:
                return None
            if len(matches) == 1:
                m = matches[0]
                return f"Interpreting '{query}' as '{m['vendor_name']}' ({m['vendor_id']})."
            items = ", ".join(f"'{m['vendor_name']}' ({m['vendor_id']})" for m in matches)
            return f"'{query}' matched multiple vendors: {items}."

        if exact:
            matches = [_vmatch(r, "exact", "high") for r in exact]
            return json.dumps({"matches": matches, "needs_clarification": len(matches) > 1,
                               "total_found": len(matches), "strategy": "exact",
                               "interpretation_disclosure": _vdisc("exact", matches)}), False

        # Strategy 2 — word-level fuzzy
        tokens = [t for t in re.split(r'\W+', query) if len(t) >= 3]
        hit_count: dict[str, int] = {}
        for tok in tokens:
            for r in vendors:
                if tok.lower() in r["vendor_id"].lower() or tok.lower() in r["vendor_name"].lower():
                    hit_count[r["vendor_id"]] = hit_count.get(r["vendor_id"], 0) + 1
        if hit_count:
            ranked = sorted(vendors, key=lambda r: hit_count.get(r["vendor_id"], 0), reverse=True)
            ranked = [r for r in ranked if hit_count.get(r["vendor_id"], 0) > 0]
            matches = [_vmatch(r, "fuzzy_word",
                               "high" if hit_count[r["vendor_id"]] >= 2 else "medium",
                               {"matched_tokens": hit_count[r["vendor_id"]]}) for r in ranked]
            return json.dumps({"matches": matches, "needs_clarification": len(matches) > 1,
                               "total_found": len(matches), "strategy": "fuzzy_word",
                               "interpretation_disclosure": _vdisc("fuzzy_word", matches)}), False

        # Strategy 3 — LLM semantic
        vendor_list = "\n".join(f"{r['vendor_id']}: {r['vendor_name']}" for r in vendors)
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content":
                 "Match the user's vendor reference to the vendor list. "
                 "The user may have misspelled or partially named a vendor. "
                 "Return JSON: {\"matches\": [{\"vendor_id\": str, \"reason\": str, "
                 "\"confidence\": \"high|medium|low\"}]}."},
                {"role": "user", "content":
                 f"User query: {query}\n\nVendor list:\n{vendor_list}"}
            ],
            response_format={"type": "json_object"},
            temperature=0, max_tokens=300,
        )
        llm_out = json.loads(resp.choices[0].message.content)
        llm_matches = llm_out.get("matches", [])
        vmap = {r["vendor_id"]: r for r in vendors}
        if llm_matches:
            matches = [
                _vmatch(vmap[m["vendor_id"]], "llm_semantic", m.get("confidence", "medium"),
                        {"reason": m.get("reason", "")})
                for m in llm_matches if m["vendor_id"] in vmap
            ]
            return json.dumps({"matches": matches, "needs_clarification": len(matches) > 1,
                               "total_found": len(matches), "strategy": "llm_semantic",
                               "interpretation_disclosure": _vdisc("llm_semantic", matches)}), False

        all_vendors = [{"vendor_id": r["vendor_id"], "vendor_name": r["vendor_name"]} for r in vendors]
        return json.dumps({"matches": [], "needs_clarification": False, "total_found": 0,
                           "strategy": "exhausted", "all_vendors": all_vendors,
                           "guidance": "No vendor matched. Show the user the full vendor list."}), False
    except Exception as exc:
        return "", f"resolve_vendor error: {exc}"
    finally:
        conn.close()


def execute_resolve_lines(query: str, entity_type: str = "line", rfx_id: str | None = None) -> tuple:
    """
    Three-strategy cascade to resolve a natural-language plant/item or vendor name
    to exact DB IDs. entity_type='line' → line_ids; entity_type='vendor' → vendor_ids.

    Strategy 1 — Exact: LOWER(field) LIKE LOWER('%full_query%')
    Strategy 2 — Word-level fuzzy: each token ≥4 chars individually, ranked by hits
    Strategy 3 — LLM semantic: GPT-4o-mini against the full catalog/vendor list
    """
    if entity_type == "vendor":
        return _resolve_vendor(query, rfx_id=rfx_id)
    conn = _db_conn()
    try:
        if rfx_id:
            catalog = conn.execute(
                "SELECT line_id, section, description, species_name, search_text "
                "FROM rfx_lines WHERE rfx_id=?", (rfx_id,)
            ).fetchall()
        else:
            catalog = conn.execute(
                "SELECT line_id, section, description, species_name, search_text FROM rfx_lines"
            ).fetchall()

        def _row_to_dict(r, method, confidence, extra=None):
            d = {"line_id": r["line_id"], "section": r["section"],
                 "description": r["description"], "species_name": r["species_name"] or "",
                 "method": method, "confidence": confidence}
            if extra:
                d.update(extra)
            return d

        def _needs_clarif(matches: list, strategy: str) -> bool:
            """Clarification needed when a specific item is ambiguous across sections.
            NOT needed when all matches fall in one section (broad category query)."""
            if len(matches) <= 1:
                return False
            if strategy == "exact":
                # exact match on multiple items in same section = category query, OK
                sections = {m["section"] for m in matches}
                return len(sections) > 1
            # fuzzy/semantic: if all in same section → category query, no clarification
            sections = {m["section"] for m in matches}
            return len(sections) > 1

        def _disclosure(query: str, matches: list, strategy: str) -> str | None:
            """Pre-build the interpretation disclosure string for non-exact strategies."""
            if strategy == "exact" or not matches:
                return None
            if len(matches) == 1:
                m = matches[0]
                label = m["description"] or m.get("species_name", "")
                return f"Interpreting '{query}' as '{label}' ({m['line_id']})."
            # multiple: list them
            items = ", ".join(f"'{m['description']}' ({m['line_id']})" for m in matches[:5])
            return f"'{query}' matched {len(matches)} lines: {items}{'...' if len(matches)>5 else ''}."

        # Strategy 1 — exact LIKE on full query
        _s1_sql = ("SELECT line_id, section, description, species_name FROM rfx_lines "
                   "WHERE LOWER(search_text) LIKE LOWER(?)"
                   + (" AND rfx_id=?" if rfx_id else ""))
        _s1_params = [f"%{query.strip()}%"] + ([rfx_id] if rfx_id else [])
        rows = conn.execute(_s1_sql, _s1_params).fetchall()
        if rows:
            matches = [_row_to_dict(r, "exact", "high") for r in rows]
            nc = _needs_clarif(matches, "exact")
            return json.dumps({
                "matches": matches,
                "needs_clarification": nc,
                "total_found": len(matches),
                "strategy": "exact",
                "interpretation_disclosure": _disclosure(query, matches, "exact"),
            }), False

        # Strategy 2 — word-level fuzzy (each word ≥4 chars)
        tokens = [w for w in re.split(r'\W+', query) if len(w) >= 4]
        hit_count: dict[str, int] = {}
        hit_rows: dict[str, dict] = {}
        _s2_sql = ("SELECT line_id, section, description, species_name FROM rfx_lines "
                   "WHERE LOWER(search_text) LIKE LOWER(?)"
                   + (" AND rfx_id=?" if rfx_id else ""))
        for tok in tokens:
            _s2_params = [f"%{tok}%"] + ([rfx_id] if rfx_id else [])
            rows = conn.execute(_s2_sql, _s2_params).fetchall()
            for r in rows:
                lid = r["line_id"]
                hit_count[lid] = hit_count.get(lid, 0) + 1
                if lid not in hit_rows:
                    hit_rows[lid] = r
        if hit_rows:
            ranked = sorted(hit_rows.items(), key=lambda x: hit_count[x[0]], reverse=True)
            matches = [_row_to_dict(r, "fuzzy_word", "high" if hit_count[lid] >= 2 else "medium",
                                    {"matched_tokens": hit_count[lid]})
                       for lid, r in ranked]
            nc = _needs_clarif(matches, "fuzzy_word")
            disc = _disclosure(query, matches, "fuzzy_word")
            return json.dumps({
                "matches": matches[:10],
                "needs_clarification": nc,
                "total_found": len(matches),
                "strategy": "fuzzy_word",
                "interpretation_disclosure": disc,
            }), False

        # Strategy 3 — LLM semantic match
        catalog_lines = "\n".join(
            f"{r['line_id']} | {r['section']} | {r['description']} | {r['species_name'] or ''}"
            for r in catalog
        )
        resp = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content":
                 "You are matching a user's query to RFx landscaping line items. "
                 "The user may have misspelled a species name or used a common name. "
                 "Return JSON: {\"matches\": [{\"line_id\": \"L_XX_NN\", \"reason\": \"short reason\", "
                 "\"confidence\": \"high|medium|low\"}]}. "
                 "Include only genuinely matching lines. If nothing matches, return {\"matches\": []}."},
                {"role": "user", "content":
                 f"User query: {query}\n\nCatalog (line_id | section | description | species_name):\n{catalog_lines}"}
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=600,
        )
        llm_out = json.loads(resp.choices[0].message.content)
        llm_matches = llm_out.get("matches", [])

        if llm_matches:
            id_set = [m["line_id"] for m in llm_matches]
            placeholders = ",".join("?" * len(id_set))
            rows = conn.execute(
                f"SELECT line_id, section, description, species_name FROM rfx_lines "
                f"WHERE line_id IN ({placeholders})", id_set
            ).fetchall()
            row_map = {r["line_id"]: r for r in rows}
            reason_map = {m["line_id"]: m.get("reason", "") for m in llm_matches}
            conf_map   = {m["line_id"]: m.get("confidence", "medium") for m in llm_matches}
            matches = [
                _row_to_dict(row_map[m["line_id"]], "llm_semantic", conf_map[m["line_id"]],
                             {"reason": reason_map[m["line_id"]]})
                for m in llm_matches if m["line_id"] in row_map
            ]
            nc = _needs_clarif(matches, "llm_semantic")
            disc = _disclosure(query, matches, "llm_semantic")
            return json.dumps({
                "matches": matches,
                "needs_clarification": nc,
                "total_found": len(matches),
                "strategy": "llm_semantic",
                "interpretation_disclosure": disc,
            }), False

        # Nothing found — return catalog summary so agent can suggest alternatives
        sections_summary = {}
        for r in catalog:
            s = r["section"]
            sections_summary.setdefault(s, []).append(
                f"{r['line_id']}: {r['description']}" + (f" ({r['species_name']})" if r["species_name"] else "")
            )
        return json.dumps({
            "matches": [],
            "needs_clarification": False,
            "total_found": 0,
            "strategy": "exhausted",
            "catalog_by_section": sections_summary,
            "guidance": "No match found. Show the user what's available in the relevant section and ask them to clarify.",
        }), False

    except Exception as exc:
        return "", f"resolve_lines error: {exc}"
    finally:
        conn.close()


def execute_run_query(code: str) -> tuple:
    """
    Execute analyst-written code in a restricted namespace.
    Returns (result_str, error_str|None).

    Errors are NEVER raised — they are returned as the error string so the
    agent loop can feed them back to the model as the tool result, letting
    the model inspect and correct its own code within MAX_TOOL_ROUNDS.
    """
    if re.search(r'\bimport\s+matplotlib\b|\bfrom\s+matplotlib\b|\bplt\b|\bFigure\b|\bsubplots\b', code):
        return "", (
            "FORBIDDEN: matplotlib/chart code is not allowed in run_query. "
            "run_query is for SQL and data retrieval only. "
            "To draw a chart: (1) call run_query to get the data as a DataFrame, "
            "(2) extract labels and values from the result, "
            "(3) call render_chart with those arrays. Do NOT write plotting code here."
        )
    code = _normalize_code(code)
    namespace = {
        "conn": _db_conn(),
        "pd": pd,
        "json": json,
    }
    try:
        exec(code, namespace)  # noqa: S102
        raw = namespace.get("result", "(no result assigned)")
        if isinstance(raw, pd.DataFrame):
            if raw.empty:
                return "(empty DataFrame — no rows matched the query)", None
            return _render_table(raw, label="rows"), None
        if isinstance(raw, (dict, list)):
            return json.dumps(raw, indent=2, default=str), None
        return str(raw), None
    except Exception as exc:
        # Return a compact error (type + message + last traceback frame) so
        # the model gets enough context to fix its code without drowning in
        # a full stack trace.
        tb_lines = traceback.format_exc().strip().splitlines()
        short_tb = "\n".join(tb_lines[-6:])  # last 6 lines capture the cause
        return "", f"{type(exc).__name__}: {exc}\n---\n{short_tb}"
    finally:
        try:
            namespace["conn"].close()
        except Exception:
            pass


def execute_search_reference(query: str) -> tuple:
    """
    Free-text Serper web search. Returns (result_str, error_str|None).
    Shared by both the analyst agent (search_reference tool) and execute_lookup_benchmark.
    """
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        try:
            import streamlit as _st
            api_key = _st.secrets.get("SERPER_API_KEY", "")
        except Exception:
            pass
    if not api_key:
        return "", "SERPER_API_KEY not set — web search unavailable."
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": 5, "gl": "in", "hl": "en"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()

        lines = [f"Search: «{query}»\n"]
        if kg := data.get("knowledgeGraph"):
            if desc := kg.get("description"):
                lines.append(f"Knowledge graph: {desc}")
        organic = data.get("organic", [])
        if not organic:
            return "\n".join(lines) + "\nNo web results found.", None
        lines.append("Results:")
        for r in organic[:5]:
            lines.append(f"• {r.get('title','')}\n  {r.get('snippet','')}\n  {r.get('link','')}")
        lines.append(
            "\nNote: Web results only — not verified quotes. "
            "Extract any INR price ranges or spec conventions mentioned."
        )
        return "\n".join(lines), None
    except requests.RequestException as exc:
        return "", f"Serper request failed: {exc}"
    except Exception as exc:
        return "", f"search_reference error: {exc}"


def execute_render_chart(
    title: str,
    series: list[dict],
    y_label: str = "Value",
    rationale: str = "",
) -> tuple[str, str | None]:
    """
    Render a bar chart from label/value series. Returns (base64_png, error).
    series: list of {name, labels, values}
    Single-series → simple bar chart with a median reference line.
    Multi-series → grouped bars with per-group median step function.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np

        n_series = len(series)
        if not n_series or not series[0].get("labels"):
            return "", "render_chart: series must have at least one item with labels and values."

        labels = [str(l) for l in series[0]["labels"]]
        n_groups = len(labels)

        fig, ax = plt.subplots(figsize=(max(8, n_groups * 1.1 + 1.5), 5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#f8f8f8")

        bar_width = 0.7 / max(n_series, 1)
        offsets = np.linspace(-(n_series - 1) / 2, (n_series - 1) / 2, n_series) * bar_width
        x = np.arange(n_groups)

        palette = ["#1976d2", "#388e3c", "#f57c00", "#7b1fa2", "#c62828", "#00796b"]

        for idx, s in enumerate(series):
            vals = [float(v) if v is not None else 0.0 for v in s.get("values", [])]
            # Pad or trim to n_groups
            if len(vals) < n_groups:
                vals += [0.0] * (n_groups - len(vals))
            vals = vals[:n_groups]
            color = palette[idx % len(palette)]
            bars = ax.bar(
                x + offsets[idx], vals, width=bar_width * 0.9,
                color=color, label=s.get("name", f"Series {idx + 1}"),
                alpha=0.88, edgecolor="white", linewidth=0.5,
            )
            # Value labels on bars
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.annotate(
                        f"{h:,.0f}",
                        xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color="#333333",
                    )

        # Median reference line(s)
        if n_series == 1:
            vals = [float(v) if v is not None else 0.0 for v in series[0].get("values", [])]
            vals = vals[:n_groups]
            med = float(np.median(vals))
            ax.axhline(
                med, color="#e53935", linestyle="--", linewidth=1.3,
                label=f"Median: {med:,.0f}", zorder=3,
            )
        else:
            # Per-group median: step function at the median across all series for each label
            group_vals = []
            for gi in range(n_groups):
                gv = []
                for s in series:
                    vlist = s.get("values", [])
                    if gi < len(vlist) and vlist[gi] is not None:
                        gv.append(float(vlist[gi]))
                group_vals.append(float(np.median(gv)) if gv else 0.0)
            # Draw as a step line connecting group midpoints
            ax.step(
                x, group_vals, where="mid",
                color="#e53935", linestyle="--", linewidth=1.4,
                label="Median per group", zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20 if n_groups > 5 else 0, ha="right" if n_groups > 5 else "center", fontsize=9)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=10, color="#111111")
        ax.set_ylabel(y_label, fontsize=9, color="#444444")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
        ax.grid(axis="y", linestyle=":", alpha=0.5, color="#cccccc")
        ax.spines[["top", "right"]].set_visible(False)
        if n_series > 1:
            ax.legend(fontsize=8, loc="upper right", framealpha=0.85)
        else:
            ax.legend(fontsize=8, loc="upper right", framealpha=0.85)

        plt.tight_layout(pad=1.2)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        import base64
        b64 = base64.b64encode(buf.read()).decode("ascii")
        return b64, None
    except Exception as exc:
        return "", f"render_chart error: {traceback.format_exc()}"


def execute_render_section_collage(rfx_id: str) -> tuple[str, str | None]:
    """
    Render a 2×N grid of subcharts — one per BOQ section — showing vendor section totals
    as bars with BOQ benchmark (amber dashed) and peer median (red dashed) lines.
    Returns (base64_png, error).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        import numpy as np

        conn = _db_conn()

        # Vendor section totals from extracted line items
        ve_totals = pd.read_sql_query(
            """
            SELECT
                rl.section,
                ve.vendor_id,
                SUM(ve.normalized_unit_price * COALESCE(rl.quantity, 0)) AS section_total
            FROM vendor_extractions ve
            JOIN rfx_lines rl ON ve.line_id = rl.line_id AND ve.rfx_id = rl.rfx_id
            WHERE ve.rfx_id = ?
              AND ve.normalized_unit_price IS NOT NULL
              AND ve.value_source = 'extracted'
            GROUP BY rl.section, ve.vendor_id
            """,
            conn,
            params=(rfx_id,),
        )

        # Also pull vendor_section_quotes for section-level vendors (Desert Bloom, Green Horizon)
        vsq = pd.read_sql_query(
            """
            SELECT section, vendor_id, total_inr AS section_total
            FROM vendor_section_quotes
            WHERE rfx_id = ?
            """,
            conn,
            params=(rfx_id,),
        )

        # BOQ section benchmarks
        boq_totals = pd.read_sql_query(
            """
            SELECT section, SUM(boq_unit_rate * COALESCE(quantity, 0)) AS boq_total
            FROM rfx_lines
            WHERE rfx_id = ?
            GROUP BY section
            """,
            conn,
            params=(rfx_id,),
        )

        # Vendor display names
        vnames_df = pd.read_sql_query(
            "SELECT vendor_id, vendor_name FROM rfx_vendors WHERE rfx_id = ?",
            conn,
            params=(rfx_id,),
        )
        conn.close()

        vname_map = dict(zip(vnames_df["vendor_id"], vnames_df["vendor_name"])) if not vnames_df.empty else {}

        # Merge line-level totals with section-quote totals (prefer extracted when both exist)
        if not vsq.empty:
            combined = pd.concat([ve_totals, vsq], ignore_index=True)
            combined = combined.groupby(["section", "vendor_id"], as_index=False)["section_total"].max()
        else:
            combined = ve_totals.copy()

        if combined.empty:
            return "", "render_section_collage: no vendor data found for this RFx."

        sections = sorted(combined["section"].unique())
        n_sections = len(sections)
        if n_sections == 0:
            return "", "render_section_collage: no sections found."

        # Always single-column vertical stack: full width per panel, avoids bar crowding
        n_cols = 1
        n_rows = n_sections

        palette = ["#2563eb", "#16a34a", "#ea580c", "#7c3aed", "#0891b2", "#b45309"]
        all_vendors = sorted(combined["vendor_id"].unique())
        vendor_color = {v: palette[i % len(palette)] for i, v in enumerate(all_vendors)}

        boq_map = dict(zip(boq_totals["section"], boq_totals["boq_total"])) if not boq_totals.empty else {}

        n_vendors_max = max(
            len(combined[combined["section"] == s]["vendor_id"].unique()) for s in sections
        )
        panel_h = max(2.8, n_vendors_max * 0.38)
        fig, axes = plt.subplots(n_rows, 1, figsize=(9, panel_h * n_rows))
        fig.patch.set_facecolor("#ffffff")
        axes_flat = np.array(axes).flatten() if n_sections > 1 else [axes]

        for idx, sec in enumerate(sections):
            ax = axes_flat[idx]
            ax.set_facecolor("#f9fafb")

            sec_data = combined[combined["section"] == sec].sort_values("section_total")
            vendors = sec_data["vendor_id"].tolist()
            totals = sec_data["section_total"].tolist()
            colors = [vendor_color.get(v, "#888") for v in vendors]
            # Use first 12 chars of vendor name to keep labels readable
            short_names = [vname_map.get(v, v)[:12] for v in vendors]

            n_v = len(vendors)
            # Narrow bars so all vendors fit comfortably even with 8+ vendors
            bar_w = min(0.55, max(0.25, 5.0 / max(n_v, 1)))
            x = np.arange(n_v)
            bars = ax.bar(x, totals, width=bar_w, color=colors, alpha=0.85,
                          edgecolor="white", linewidth=0.5, zorder=2)

            # Value labels
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    lakh = h / 1e5
                    lbl = f"{lakh:.1f}L" if lakh >= 1 else f"{h/1e3:.0f}k"
                    ax.annotate(lbl, xy=(bar.get_x() + bar.get_width() / 2, h),
                                xytext=(0, 2), textcoords="offset points",
                                ha="center", va="bottom", fontsize=7, color="#111")

            # BOQ benchmark line
            boq_val = boq_map.get(sec, 0)
            if boq_val > 0:
                ax.axhline(boq_val, color="#d97706", linestyle="--", linewidth=1.6,
                           label="BOQ", zorder=3)

            # Peer median line
            if totals:
                med = float(np.median(totals))
                ax.axhline(med, color="#dc2626", linestyle=":", linewidth=1.4,
                           label="Median", zorder=3)

            ax.set_xlim(-0.5, max(n_v - 0.5, 0.5))
            ax.set_xticks(x)
            ax.set_xticklabels(short_names, fontsize=8, rotation=25, ha="right")
            ax.set_title(sec.replace("_", " ").title(), fontsize=10, fontweight="bold",
                         pad=5, color="#111", loc="left")
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(
                lambda v, _: f"{v/1e5:.0f}L" if v >= 1e5 else f"{v/1e3:.0f}k"
            ))
            ax.tick_params(axis="y", labelsize=7)
            ax.grid(axis="y", linestyle=":", alpha=0.4, color="#ccc", zorder=0)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(fontsize=7, loc="upper right", framealpha=0.8)

        # Legend for vendor colors at figure level
        import matplotlib.patches as mpatches
        legend_patches = [
            mpatches.Patch(color=vendor_color.get(v, "#888"),
                           label=vname_map.get(v, v)[:16])
            for v in all_vendors
        ]
        fig.legend(handles=legend_patches, loc="lower center",
                   ncol=min(len(all_vendors), 4), fontsize=7.5,
                   framealpha=0.9, bbox_to_anchor=(0.5, 0.0))

        fig.suptitle(f"Vendor Quote vs BOQ — by Section · {rfx_id}",
                     fontsize=11, fontweight="bold", y=1.002, color="#111")

        plt.tight_layout(rect=[0, 0.05, 1, 1], h_pad=1.5)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        import base64
        b64 = base64.b64encode(buf.read()).decode("ascii")
        return b64, None
    except Exception:
        return "", f"render_section_collage error: {traceback.format_exc()}"


def execute_lookup_benchmark(species: str, unit: str, context: str = "") -> tuple:
    """Builds a landscaping-specific query and delegates to execute_search_reference."""
    parts = [species, "price per", unit, "nursery India landscaping"]
    if context:
        parts.insert(1, context)
    return execute_search_reference(" ".join(parts))


# ── Query Understanding Pipeline ─────────────────────────────────────────────

NEGOTIATION_THRESHOLD = 0.30  # flag vendor priced >30% above peer median

_OFF_TOPIC_FRUSTRATION = frozenset({
    "stupid", "useless", "doesn't work", "not working", "broken",
    "terrible", "horrible", "waste", "wrong", "garbage", "trash",
})
_OFF_TOPIC_CHIT_CHAT = frozenset({
    "how are you", "hello", "hi ", "good morning", "good afternoon",
    "good evening", "thank you", "thanks", "bye", "goodbye",
})


def classify_query(query: str, rfx_id: str | None = None) -> dict:
    """Stage 1: Classify intent + extract entities. Returns a classification dict."""
    q_lower = query.lower().strip()

    # Fast-path off-topic (no LLM cost)
    if len(query.strip()) < 4:
        return {"intent": "off_topic", "off_topic_sub_type": "chit_chat", "entities": {}, "needs_confirmation": False}
    if any(s in q_lower for s in _OFF_TOPIC_FRUSTRATION):
        return {"intent": "off_topic", "off_topic_sub_type": "frustration", "entities": {}, "needs_confirmation": False}
    if any(s in q_lower for s in _OFF_TOPIC_CHIT_CHAT) and len(query.split()) < 8:
        return {"intent": "off_topic", "off_topic_sub_type": "chit_chat", "entities": {}, "needs_confirmation": False}

    spec_fields = [
        "height_low", "height_high", "spread_low", "spread_high",
        "caliper_low_mm", "caliper_high_mm", "trunk_dia_low_mm", "trunk_dia_high_mm",
        "stem_dia_low_mm", "stem_dia_high_mm", "culm_dia_low_mm", "culm_dia_high_mm",
        "pot_dia_mm", "clump_spread_low", "clump_spread_high",
    ]
    sections = ["soil_prep", "trees", "shrubs", "ground_covers", "lawn", "staking"]

    try:
        resp = openai_client.chat.completions.create(
            model=MODEL_LIGHT,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=350,
            messages=[{
                "role": "system",
                "content": (
                    "Classify a procurement analyst query. Return exactly this JSON schema:\n"
                    "{\n"
                    '  "intent": "spec_filter|name_lookup|rank_aggregate|visualization|negotiation_analysis'
                    '|knowledge_botanical|knowledge_availability|off_topic|hybrid",\n'
                    '  "off_topic_sub_type": "chit_chat|frustration|meta_question|out_of_scope|null",\n'
                    '  "entities": {\n'
                    f'    "section": one of {sections} or null,\n'
                    f'    "spec_field": one of {spec_fields} or null,\n'
                    '    "spec_operator": ">|>=|<|<=|between|null",\n'
                    '    "spec_value": numeric or null,\n'
                    '    "spec_value2": upper bound for between or null,\n'
                    '    "vendor": vendor name if mentioned or null,\n'
                    '    "ranking": "cheapest|most_expensive|highest|lowest|null"\n'
                    "  },\n"
                    '  "knowledge_topic": "brief description of external knowledge needed or null",\n'
                    '  "step_count": integer 1-5\n'
                    "}\n\n"
                    "Intent definitions:\n"
                    "spec_filter: numeric/attribute constraint on plant specs (height, caliper, spread, diameter)\n"
                    "name_lookup: find specific plant species or vendor by name\n"
                    "rank_aggregate: cheapest/priciest/ranking across vendors or lines\n"
                    "visualization: user asks for a chart, plot, visual, graph, or 'show me visually' — "
                    "any request whose primary output is a chart or diagram, even if it also involves comparison\n"
                    "negotiation_analysis: overcharging, negotiation room, push back on vendor\n"
                    "knowledge_botanical: botanical traits — drought tolerance, growth rate, maintenance\n"
                    "knowledge_availability: regional supply — Jaipur/Rajasthan nurseries\n"
                    "off_topic: not related to procurement analysis\n"
                    "hybrid: combines two or more of the above"
                )
            }, {
                "role": "user",
                "content": f"Query: {query}"
            }]
        )
        result = json.loads(resp.choices[0].message.content)
        intent = result.get("intent", "unknown")
        needs_confirm = intent in (
            "negotiation_analysis", "knowledge_botanical", "knowledge_availability"
        ) or (intent == "hybrid" and result.get("step_count", 1) > 1)
        result["needs_confirmation"] = needs_confirm
        return result
    except Exception:
        return {"intent": "unknown", "entities": {}, "needs_confirmation": False}


def generate_plan_text(classification: dict, rfx_id: str | None = None) -> str:
    """
    Stage 2: Plain-English restatement of what the agent understood.
    Written for the analyst to read and correct if wrong — not a technical execution plan.
    """
    intent = classification.get("intent")
    entities = classification.get("entities", {})
    section_raw = entities.get("section")
    section = (section_raw or "all sections").replace("_", " ")
    vendor = entities.get("vendor")

    if intent == "spec_filter":
        field = (entities.get("spec_field") or "spec field").replace("_", " ")
        op = entities.get("spec_operator") or ">="
        val = entities.get("spec_value", "?")
        val2 = entities.get("spec_value2")
        range_str = f"between {val} and {val2}" if val2 else f"{op} {val}"
        return (
            f"You want {section} items where {field} is {range_str} — "
            "I'll filter the spec data and pull vendor prices for those lines."
        )
    elif intent == "negotiation_analysis":
        return (
            f"You want to see which vendors are overcharging on {section}. "
            "I'll compare each vendor's prices against what the others quoted for the same lines."
        )
    elif intent == "knowledge_botanical":
        topic = classification.get("knowledge_topic") or "your criterion"
        return (
            f"You want {section} species that match '{topic}'. "
            "I'll use botanical knowledge to identify matches in your BOQ, "
            "then show vendor prices for those plants. "
            "⚠️ Results are based on general horticultural knowledge, not a verified source."
        )
    elif intent == "knowledge_availability":
        return (
            f"You want to know which plants in your {section} BOQ can be sourced from "
            "Rajasthan nurseries. I'll do a quick web check and map the results to your vendor quotes."
        )
    elif intent == "rank_aggregate":
        ranking = entities.get("ranking") or "lowest price"
        vendor_str = f" for {vendor}" if vendor else ""
        return (
            f"You want {section} vendors ranked by {ranking}{vendor_str}. "
            "I'll aggregate their quotes and sort them."
        )
    elif intent == "hybrid":
        steps = classification.get("step_count", 2)
        topic = classification.get("knowledge_topic") or "your query"
        return (
            f"This looks like a {steps}-part question about {topic}. "
            "I'll handle each part in order and combine the results."
        )
    return "I'll look this up in the vendor comparison database."


def _median(values: list) -> float | None:
    filtered = sorted(v for v in values if v is not None)
    if not filtered:
        return None
    n = len(filtered)
    mid = n // 2
    return filtered[mid] if n % 2 else (filtered[mid - 1] + filtered[mid]) / 2


TABLE_ROW_LIMIT = 20


def _render_table(df: pd.DataFrame, label: str = "rows") -> str:
    """Return markdown table if ≤TABLE_ROW_LIMIT rows, else an English summary."""
    if df.empty:
        return f"*No {label} found.*"
    if len(df) <= TABLE_ROW_LIMIT:
        try:
            return df.to_markdown(index=False)
        except Exception:
            return df.to_string(index=False)
    # Summarise instead of pasting the whole table
    parts = [f"*{len(df)} {label} — summary (full table exceeds {TABLE_ROW_LIMIT} rows):*\n"]
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            parts.append(
                f"- **{col}**: min {s.min():,.2f} · median {s.median():,.2f} · "
                f"max {s.max():,.2f} · total {s.sum():,.2f}"
            )
        else:
            top = s.dropna().value_counts().head(5)
            top_str = ", ".join(f"{v} ({c})" for v, c in top.items())
            parts.append(f"- **{col}**: {s.nunique()} unique — top: {top_str}")
    return "\n".join(parts)


def execute_negotiation_analysis(entities: dict, rfx_id: str) -> tuple:
    """
    Peer-median negotiation analysis. Returns (markdown_str, tool_log).
    Flags vendors priced >30% above peer median on like-for-like lines.
    BOQ is a secondary sanity check only — not the primary benchmark.
    """
    section = entities.get("section")
    tool_log = []
    section_label = section or "all sections"

    conn = _db_conn()
    try:
        section_filter_ve = "AND rl.section = ?" if section else ""
        section_filter_rl = "AND section = ?" if section else ""
        p_rl = [rfx_id] + ([section] if section else [])
        p_ve = [rfx_id] + ([section] if section else []) + [rfx_id]

        df_ve = pd.read_sql_query(f"""
            SELECT ve.vendor_id, ve.vendor_name, ve.line_id,
                   ve.normalized_unit_price, ve.extraction_confidence
            FROM vendor_extractions ve
            JOIN rfx_lines rl ON rl.line_id = ve.line_id
            WHERE rl.rfx_id = ?
              {section_filter_ve}
              AND ve.matched = 1
              AND (ve.superseded_by IS NULL OR ve.superseded_by = '')
              AND ve.normalized_unit_price IS NOT NULL
              AND ve.extraction_confidence >= 0.6
              AND ve.rfx_id = ?
        """, conn, params=p_ve)

        df_lines = pd.read_sql_query(f"""
            SELECT line_id, description, boq_unit_rate, quantity, section
            FROM rfx_lines WHERE rfx_id = ? {section_filter_rl}
        """, conn, params=p_rl)

        tool_log.append({
            "tool": "run_query", "model": MODEL_HEAVY,
            "rationale": f"load vendor prices for peer-median negotiation — {section_label}",
            "code": "SELECT normalized_unit_price + rfx_lines for peer stats",
            "result": f"{len(df_ve)} price rows, {len(df_lines)} lines",
            "error": None,
        })

        if df_ve.empty:
            return f"No vendor price data found for {section_label}.", tool_log

        total_vendors = df_ve["vendor_id"].nunique()
        min_vendors = max(2, -(-total_vendors * 3 // 5))  # ceil(total * 0.6)

        vendor_counts = df_ve.groupby("line_id")["vendor_id"].nunique()
        eligible_line_ids = vendor_counts[vendor_counts >= min_vendors].index.tolist()

        if not eligible_line_ids:
            return (
                f"No {section_label} lines have ≥{min_vendors} vendors quoting (need 60% of {total_vendors}). "
                "Cannot compute a reliable peer benchmark.", tool_log
            )

        df_elig = df_ve[df_ve["line_id"].isin(eligible_line_ids)].copy()
        df_lines_idx = df_lines[df_lines["line_id"].isin(eligible_line_ids)].set_index("line_id")

        # Per-line peer statistics
        line_stats: dict = {}
        for lid, grp in df_elig.groupby("line_id"):
            prices = grp["normalized_unit_price"].dropna().tolist()
            line_stats[lid] = {
                "peer_median": _median(prices),
                "peer_min": min(prices),
                "vendor_count": len(prices),
            }

        # Per-vendor flag accumulation
        vendor_data: dict = {}
        for _, row in df_elig.iterrows():
            vid = row["vendor_id"]
            vname = row.get("vendor_name") or vid
            lid = row["line_id"]
            price = row["normalized_unit_price"]
            median = line_stats[lid]["peer_median"]
            row_rl = df_lines_idx.loc[lid] if lid in df_lines_idx.index else None
            boq_rate = float(row_rl["boq_unit_rate"]) if row_rl is not None and row_rl["boq_unit_rate"] else None
            qty = float(row_rl["quantity"]) if row_rl is not None and row_rl["quantity"] else 0
            desc = str(row_rl["description"])[:50] if row_rl is not None else lid

            vendor_data.setdefault(vid, {"name": vname, "flags": [], "opportunity_inr": 0.0})

            if median and price > median * (1 + NEGOTIATION_THRESHOLD):
                pct = (price / median - 1) * 100
                # Downgrade red→amber if price is still below BOQ (BOQ was generous)
                if pct > 80 and (boq_rate is None or price > boq_rate):
                    sev = "🔴"
                else:
                    sev = "🟡"
                opp = (price - median) * qty
                vendor_data[vid]["flags"].append({
                    "line_id": lid, "description": desc,
                    "price": price, "peer_median": median,
                    "overcharge_pct": pct, "opportunity_inr": opp,
                    "severity": sev, "boq_rate": boq_rate,
                })
                vendor_data[vid]["opportunity_inr"] += opp

        # BOQ under-spec check: all vendors > 2× BOQ on a line
        boq_anomaly_lines = []
        for lid, stats in line_stats.items():
            row_rl = df_lines_idx.loc[lid] if lid in df_lines_idx.index else None
            boq = float(row_rl["boq_unit_rate"]) if row_rl is not None and row_rl["boq_unit_rate"] else None
            if boq and stats["peer_min"] > boq * 2:
                desc = str(row_rl["description"])[:50] if row_rl is not None else lid
                boq_anomaly_lines.append((lid, desc, boq, stats["peer_min"]))

        # Build output
        out = [f"### Negotiation Analysis — {section_label}\n"]
        out.append(
            f"*Like-for-like basis: {len(eligible_line_ids)} lines quoted by "
            f"≥{min_vendors} of {total_vendors} vendors. Threshold: >{NEGOTIATION_THRESHOLD*100:.0f}% above peer median.*\n"
        )

        ranked = sorted(vendor_data.items(), key=lambda x: x[1]["opportunity_inr"], reverse=True)
        has_flags = any(v["flags"] for _, v in ranked)

        if not has_flags:
            out.append("✅ No vendor exceeds the threshold on any eligible line — pricing is competitive.")
        else:
            out.append("| Vendor | Flagged Lines | Opportunity (₹) | Worst Overcharge |")
            out.append("|---|---|---|---|")
            for vid, vd in ranked:
                flags = vd["flags"]
                if not flags:
                    out.append(f"| {vd['name']} | ✅ None | — | — |")
                    continue
                worst = max(flags, key=lambda f: f["overcharge_pct"])
                opp_str = f"₹{vd['opportunity_inr']:,.0f}" if vd["opportunity_inr"] > 0 else "—"
                out.append(
                    f"| {vd['name']} | {worst['severity']} {len(flags)} | {opp_str} | "
                    f"{worst['description']} ({worst['overcharge_pct']:.0f}% above median) |"
                )
            # Drill-down for worst vendor
            worst_vid, worst_vd = ranked[0]
            if worst_vd["flags"]:
                out.append(f"\n**{worst_vd['name']} — detail:**\n")
                drill_rows = sorted(worst_vd["flags"], key=lambda x: x["overcharge_pct"], reverse=True)
                drill_df = pd.DataFrame([{
                    "Line": f["line_id"],
                    "Description": f["description"],
                    "Their Price (₹)": f["price"],
                    "Peer Median (₹)": f["peer_median"],
                    "Over%": f"{f['severity']}{f['overcharge_pct']:.0f}%",
                    "BOQ Rate (₹)": f["boq_rate"] if f["boq_rate"] else None,
                    "Opportunity (₹)": f["opportunity_inr"],
                } for f in drill_rows])
                out.append(_render_table(drill_df, label="flagged lines"))

        if boq_anomaly_lines:
            out.append(
                f"\n⚠️ **BOQ may be under-specified on {len(boq_anomaly_lines)} line(s)** "
                "(all vendors quote >2× BOQ — the BOQ was likely set too low, not vendor overcharging):"
            )
            for lid, desc, boq, peer_min in boq_anomaly_lines:
                out.append(f"- {lid} {desc}: BOQ ₹{boq:,.0f} · lowest vendor ₹{peer_min:,.0f}")

        return "\n".join(out), tool_log
    finally:
        conn.close()


def execute_knowledge_botanical(query: str, entities: dict, rfx_id: str) -> tuple:
    """
    Botanical knowledge path: LLM identifies matching species → DB join for vendor prices.
    Returns (markdown_str, tool_log).
    Every response carries a mandatory LLM-assumption disclaimer.
    """
    section = entities.get("section")
    tool_log = []

    conn = _db_conn()
    try:
        section_filter = "AND section = ?" if section else ""
        rows = conn.execute(
            f"SELECT line_id, description, species_name FROM rfx_lines "
            f"WHERE rfx_id = ? {section_filter} AND species_name IS NOT NULL",
            [rfx_id] + ([section] if section else []),
        ).fetchall()

        if not rows:
            return "No species data found in the BOQ for this section.", tool_log

        species_list = "\n".join(
            f"- {r['line_id']}: {r['species_name']} ({str(r['description'])[:40]})"
            for r in rows
        )

        resp = openai_client.chat.completions.create(
            model=MODEL_HEAVY,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=700,
            messages=[{
                "role": "system",
                "content": (
                    "You are a horticultural expert. Given a list of plant species and a query, "
                    "identify which species match the criterion. Be conservative — only include "
                    "species you are confident about. "
                    'Return JSON: {"matches": [{"line_id": "L_XX_NN", "species": "name", '
                    '"rationale": "one sentence"}], "disclaimer": "one sentence caveat"}.'
                )
            }, {
                "role": "user",
                "content": f"Query: {query}\n\nSpecies in BOQ:\n{species_list}"
            }]
        )

        llm_out = json.loads(resp.choices[0].message.content)
        matches = llm_out.get("matches", [])
        disclaimer = llm_out.get("disclaimer", "Based on general horticultural knowledge.")

        tool_log.append({
            "tool": "llm_knowledge", "model": MODEL_HEAVY,
            "rationale": f"botanical knowledge: {query[:60]}",
            "code": f"LLM identified {len(matches)} matching species",
            "result": f"matches={[m['line_id'] for m in matches]}",
            "error": None,
        })

        if not matches:
            return (
                f"No species in the BOQ matched '{query}' based on botanical knowledge.\n\n"
                f"⚠️ *{disclaimer}*"
            ), tool_log

        matched_ids = [m["line_id"] for m in matches]
        placeholders = ",".join("?" * len(matched_ids))

        df = pd.read_sql_query(f"""
            SELECT rl.line_id, rl.description, rl.species_name,
                   ve.vendor_id, ve.vendor_name, ve.normalized_unit_price, rl.boq_unit_rate
            FROM rfx_lines rl
            LEFT JOIN vendor_extractions ve ON ve.line_id = rl.line_id
              AND ve.matched = 1 AND (ve.superseded_by IS NULL OR ve.superseded_by = '')
              AND ve.normalized_unit_price IS NOT NULL
            WHERE rl.line_id IN ({placeholders})
            ORDER BY rl.line_id, ve.normalized_unit_price
        """, conn, params=matched_ids)

        tool_log.append({
            "tool": "run_query", "model": MODEL_HEAVY,
            "rationale": "fetch vendor prices for botanically-matched species",
            "code": f"SELECT prices for {matched_ids}",
            "result": f"{len(df)} rows",
            "error": None,
        })

        out = [f"### Botanical Match — {query}\n"]
        out.append(f"⚠️ *{disclaimer}*\n")
        out.append(f"**{len(matched_ids)} matching species:**\n")
        for m in matches:
            out.append(f"- **{m['line_id']}** {m['species']}: {m['rationale']}")

        if not df.empty:
            out.append("\n**Vendor prices for matching lines:**\n")
            pivot = df.pivot_table(
                index=["line_id", "description", "species_name"],
                columns="vendor_name",
                values="normalized_unit_price",
                aggfunc="min",
            ).reset_index()
            out.append(_render_table(pivot, label="species-vendor rows"))
        else:
            out.append("\n*No vendor prices found for matching lines.*")

        return "\n".join(out), tool_log
    finally:
        conn.close()


def execute_knowledge_availability(query: str, entities: dict, rfx_id: str) -> tuple:
    """
    Availability path: ≤2 web searches → LLM interprets → DB join.
    Returns (markdown_str, tool_log).
    """
    section = entities.get("section")
    tool_log = []

    conn = _db_conn()
    try:
        section_filter = "AND section = ?" if section else ""
        rows = conn.execute(
            f"SELECT line_id, description, species_name FROM rfx_lines "
            f"WHERE rfx_id = ? {section_filter} AND species_name IS NOT NULL",
            [rfx_id] + ([section] if section else []),
        ).fetchall()

        if not rows:
            return "No species data found in the BOQ.", tool_log

        # Extract location from query, default to Jaipur
        location = "Jaipur Rajasthan"
        for loc_hint in ("jaipur", "rajasthan", "delhi", "mumbai", "bangalore", "pune"):
            if loc_hint in query.lower():
                location = loc_hint.title()
                break

        species_names = [r["species_name"] for r in rows if r["species_name"]]
        search_q = (
            f"landscaping nursery {location} plant availability: "
            + ", ".join(species_names[:10])
        )

        search_result, search_err = execute_search_reference(search_q)
        st.session_state.web_search_session_count = (
            st.session_state.get("web_search_session_count", 0) + 1
        )

        tool_log.append({
            "tool": "web_search", "model": None,
            "rationale": f"availability check: {location}",
            "code": f"search({search_q[:80]}...)",
            "result": search_result[:400] if search_result else "",
            "error": search_err,
        })

        if search_err:
            return f"Web search unavailable: {search_err}. Cannot verify plant availability.", tool_log

        # LLM interprets search results
        species_list_str = "\n".join(f"- {r['line_id']}: {r['species_name']}" for r in rows)
        resp = openai_client.chat.completions.create(
            model=MODEL_LIGHT,
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=500,
            messages=[{
                "role": "system",
                "content": (
                    f"Based on the web search results, identify which plants from the BOQ list are "
                    f"likely available from nurseries in {location}. "
                    'Return JSON: {"available": ["line_id", ...], "unavailable": ["line_id", ...], '
                    '"unknown": ["line_id", ...], "confidence_note": "one sentence"}.'
                )
            }, {
                "role": "user",
                "content": f"Search results:\n{search_result}\n\nBOQ species:\n{species_list_str}"
            }]
        )

        avail_out = json.loads(resp.choices[0].message.content)
        available_ids = avail_out.get("available", [])
        unknown_ids = avail_out.get("unknown", [])
        confidence_note = avail_out.get("confidence_note", "")

        tool_log.append({
            "tool": "llm_knowledge", "model": MODEL_LIGHT,
            "rationale": f"interpret search results for species availability in {location}",
            "code": f"LLM confirmed {len(available_ids)} available, {len(unknown_ids)} unknown",
            "result": f"available={available_ids}",
            "error": None,
        })

        if not available_ids:
            return (
                f"Could not confirm availability for any species in {location} from web search. "
                f"{confidence_note}\n"
                f"⚠️ *Web-sourced data — verify directly with local nurseries.*"
            ), tool_log

        placeholders = ",".join("?" * len(available_ids))
        df = pd.read_sql_query(f"""
            SELECT rl.line_id, rl.description, rl.species_name,
                   ve.vendor_id, ve.vendor_name, ve.normalized_unit_price, rl.boq_unit_rate
            FROM rfx_lines rl
            LEFT JOIN vendor_extractions ve ON ve.line_id = rl.line_id
              AND ve.matched = 1 AND (ve.superseded_by IS NULL OR ve.superseded_by = '')
              AND ve.normalized_unit_price IS NOT NULL
            WHERE rl.line_id IN ({placeholders})
            ORDER BY rl.line_id, ve.normalized_unit_price
        """, conn, params=available_ids)

        out = [f"### Plants Available in {location}\n"]
        out.append(f"⚠️ *{confidence_note} Web-sourced — verify directly with local nurseries.*\n")
        lid_to_species = {r["line_id"]: r["species_name"] for r in rows}
        out.append(f"**{len(available_ids)} species confirmed available** (of {len(rows)} in BOQ):")
        for lid in available_ids:
            out.append(f"- {lid}: {lid_to_species.get(lid, lid)}")

        if unknown_ids:
            out.append(f"\n*Availability unknown for {len(unknown_ids)} species — excluded from results.*")

        if not df.empty:
            out.append("\n**Vendor prices for confirmed-available lines:**\n")
            out.append(_render_table(df.drop(columns=["boq_unit_rate"], errors="ignore"), label="vendor-line rows"))
        else:
            out.append("\n*No vendor prices found for confirmed-available lines.*")

        return "\n".join(out), tool_log
    finally:
        conn.close()


def handle_off_topic(classification: dict, query: str, rfx_id: str | None = None) -> str:
    """Returns inline response for off-topic inputs. No DB query, no tool loop."""
    sub = classification.get("off_topic_sub_type") or "chit_chat"

    if sub == "frustration":
        return (
            "I hear you — let me try a different approach. "
            "Could you tell me what you were looking for? "
            "I'll do my best to find the right data."
        )
    elif sub == "chit_chat":
        return "Focused on vendor comparison data here — happy to help with any procurement question."
    elif sub == "meta_question":
        try:
            conn = _db_conn()
            n_lines = conn.execute(
                "SELECT COUNT(*) FROM rfx_lines WHERE rfx_id=?", (rfx_id,)
            ).fetchone()[0] if rfx_id else "?"
            n_vendors = conn.execute(
                "SELECT COUNT(DISTINCT vendor_id) FROM vendor_extractions WHERE rfx_id=?", (rfx_id,)
            ).fetchone()[0] if rfx_id else "?"
            conn.close()
        except Exception:
            n_lines, n_vendors = "?", "?"
        return (
            f"I have access to **{n_lines} RFx line items** and **{n_vendors} vendor quotes** "
            "for this project. I can query prices, plant specs, flags, questionnaire responses, "
            "and run negotiation analysis. Ask me anything about the vendor data."
        )
    else:
        return "That's outside my scope — I'm focused on procurement data for this RFx."


def _log_trace(trace: dict):
    """Append a structured trace record to agent_traces.jsonl."""
    try:
        trace["ts"] = datetime.now(timezone.utc).isoformat()
        os.makedirs(os.path.dirname(TRACE_LOG_PATH), exist_ok=True)
        with open(TRACE_LOG_PATH, "a") as fh:
            fh.write(json.dumps(trace, default=str) + "\n")
    except Exception:
        pass  # trace failure must never block the analyst


# ── Agent loop ────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = f"""
You are a procurement analyst assistant for a real landscaping project (SKYDOME KUKAS, Jaipur).
You have access to a structured comparison database of vendor quotes and questionnaire responses.
Use the run_query tool to answer questions by generating and executing real code — do NOT make up
numbers or answer from memory. Every factual claim must come from a query result.

CRITICAL — MIXED GRANULARITY VENDORS:
Some vendors (e.g. Desert Bloom, Green Horizon) quoted at SECTION level, not per line item.
Their data lives in vendor_section_quotes, NOT vendor_extractions.
Rules when comparing prices:
1. Before answering "who is cheapest for [category]", ALWAYS run:
   SELECT vendor_id, section, total_amount, value_source, superseded_by
   FROM vendor_section_quotes WHERE rfx_id=? AND (superseded_by IS NULL OR superseded_by='')
   to find any vendors with section-level data for that category.
2. If a vendor has section-level data only (granularity='section_total'):
   - State explicitly that per-line comparison is NOT available for them.
   - Show their total_amount from vendor_section_quotes: "Desert Bloom quoted the entire Trees
     section at ₹310,000 (firm written quote) — not available per species."
   - Do NOT fabricate a per-line price or silently exclude them from your answer.
3. For MIXED granularity vendors (e.g. V_GREEN_HORIZON, granularity='mixed'):
   - They have SOME line items in vendor_extractions AND section totals in vendor_section_quotes.
   - To get their true section total, you MUST add: sum(line_item extractions for that section)
     + total_amount from vendor_section_quotes where section='trees_balance' (or similar).
   - Do NOT present the line-item sum alone as their total — it will be incomplete.
   - Flag this in your answer: "Green Horizon quoted 6 species individually + a balance of 265
     trees as a package (₹424,000). Their total trees cost is sum + ₹424,000."
4. For fair comparison: always state which vendors are per-line vs section-level, and flag
   that direct total comparisons across granularities are approximate.
5. verbal_estimate rows (value_source='verbal_estimate'): label as "verbal estimate — superseded
   by written quote" if a later v2 exists for that vendor (check superseded_by column).
6. section_total vendors have granularity='section_total' in vendor_extractions OR rows only
   in vendor_section_quotes (no extractions). Check both tables always.

{_schema_summary()}

When writing code for run_query:
- Use ONLY ASCII comparison operators: >=, <=, !=, =, >, <. Never Unicode (≥ ≤ ≠).
- In multi-line SQL strings, put a trailing space before every newline inside the SQL
  so tokens don't merge when the string is processed (write "FROM vendor_extractions ve \n"
  not "FROM vendor_extractions ve\n"). Or write SQL on a single line.
- Always wrap SQL in pd.read_sql_query(..., conn) — never pass raw SQL to exec() directly.
- If a query returns an error, read the error message, fix the exact issue, and retry.
- Follow RULES A–D from the schema section above for every relevant query.

When to use lookup_benchmark:
- When asked "should I trust this price?" or "is this reasonable?" for a line item.
- When all DB entries for a line are low-confidence or excluded (RULE D surfaces this).
- When reporting the cheapest option for a flagged line — benchmark confirms or contradicts it.
- Do NOT use lookup_benchmark for questions answerable from DB data alone.

When to use render_section_collage (DEFAULT chart):
- Use this whenever the user asks to "compare vendors", "show a chart", "plot", "visualise", or
  "show me visually" — unless they ask for a specific single metric or single-section drill-down.
- Call it directly with the active rfx_id — no prior run_query needed. It queries the DB itself.
- It renders one subplot per BOQ section with vendor bars + BOQ benchmark (amber) + peer median (red).
- After calling it, describe what the chart shows: which sections have the widest spread, which vendor
  is consistently cheapest/dearest, any sections where most vendors exceed BOQ.
- Do NOT include any image markdown (![...]) or base64 data in your text response.

When to use render_chart (for specific/custom charts only):
- ONLY when the user asks for a specific metric, single-section drill-down, or custom aggregation that
  render_section_collage does not cover (e.g. "chart only the trees section", "chart unit rates for X").
- Workflow: call run_query first to get the data → extract labels/values → call render_chart.
- NEVER write matplotlib or chart code inside run_query. render_chart is the only way to produce a visual.
- When data would produce 10+ bars, aggregate to section level first (GROUP BY section).
- The chart adds a median reference line automatically.
- Do NOT include image markdown or base64 in text responses.

When answering:
1. Run the query, interpret the result, answer in plain English.
2. Always cite where numbers came from (vendor_id + source_file + source_snippet if relevant).
3. If a comparison is affected by flags (spec_grade_mismatch, species_name_mismatch, etc.),
   call that out explicitly — do not present flagged rows as clean comparisons.
4. If you cannot answer reliably (missing data, all rows low-confidence), say so clearly.
5. Keep responses concise but complete. Use markdown tables where helpful.
""".strip()


def run_agent(
    user_message: str,
    history: list,
    rfx_id: str | None = None,
    intent_hint: str | None = None,
) -> tuple:
    """
    Run the tool-calling loop for one user turn.
    Returns (final_response, updated_history, tool_log).
    intent_hint injects routing guidance when the pre-classifier has already fired.
    """
    model = _select_model(user_message)
    rfx_scope = (
        f"\nACTIVE RFx SCOPE: rfx_id = '{rfx_id}'. "
        "All queries on rfx_lines MUST include WHERE rfx_id='{rfx_id}' (or AND rfx_id='{rfx_id}'). "
        "All queries on vendor_extractions MUST include AND rfx_id='{rfx_id}'. "
        "Never query across all RFx projects — always filter to the active rfx_id.\n"
    ).format(rfx_id=rfx_id) if rfx_id else ""
    # Intent-specific routing hints (suppress wrong tool paths)
    _intent_guidance = {
        "spec_filter": (
            "\nROUTING: spec_filter intent — use json_extract(spec_parsed,'$.field_name') "
            "directly in run_query. Do NOT call resolve_lines. "
            "Field names are range pairs: height_low/height_high, caliper_low_mm/caliper_high_mm, etc."
        ),
        "rank_aggregate": (
            "\nROUTING: rank_aggregate intent — use aggregate SQL (GROUP BY, ORDER BY) directly. "
            "Do NOT call resolve_lines unless a specific plant name was mentioned."
        ),
        "visualization": (
            "\nROUTING: visualization intent — the user wants a chart. "
            "Call render_section_collage immediately with the active rfx_id. "
            "Do NOT call run_query first. Do NOT use render_chart unless the user asked for a "
            "specific single-metric or single-section drill-down. "
            "After the chart renders, give a 2-3 sentence summary of what it shows."
        ),
    }
    intent_guidance = _intent_guidance.get(intent_hint, "")
    system_content = SYSTEM_PROMPT + rfx_scope + intent_guidance
    messages = [{"role": "system", "content": system_content}] + history + [
        {"role": "user", "content": user_message}
    ]
    tool_log = []

    for round_n in range(MAX_TOOL_ROUNDS):
        # Escalate to heavy model after the first round if errors were encountered.
        if round_n > 0 and model == MODEL_LIGHT:
            last_tool = tool_log[-1] if tool_log else {}
            if last_tool.get("error"):
                model = MODEL_HEAVY

        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        msg = response.choices[0].message

        # No tool calls → final answer
        if not msg.tool_calls:
            final = msg.content or ""
            updated_history = history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": final}
            ]
            return final, updated_history, tool_log

        # Execute each tool call
        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            fn = tc.function.name

            if fn == "resolve_lines":
                query = args.get("query", "")
                entity_type = args.get("entity_type", "line")
                rationale = f"resolve[{entity_type}]: {query}"
                result_str, error = execute_resolve_lines(query, entity_type, rfx_id=rfx_id)
                log_entry = {"tool": "resolve_lines", "model": model, "rationale": rationale,
                             "code": f"resolve_lines({query!r}, entity_type={entity_type!r})",
                             "result": "", "error": error}
            elif fn == "run_query":
                code = args.get("code", "")
                rationale = args.get("rationale", "")
                result_str, error = execute_run_query(code)
                log_entry = {"tool": "run_query", "model": model, "rationale": rationale,
                             "code": code, "result": "", "error": error}
            elif fn == "lookup_benchmark":
                species = args.get("species", "")
                unit = args.get("unit", "")
                context = args.get("context", "")
                rationale = f"benchmark: {species} per {unit}"
                result_str, error = execute_lookup_benchmark(species, unit, context)
                log_entry = {"tool": "lookup_benchmark", "model": model, "rationale": rationale,
                             "code": f"lookup_benchmark({species!r}, {unit!r}, {context!r})",
                             "result": "", "error": error}
            elif fn == "search_reference":
                query = args.get("query", "")
                rationale = f"web search: {query[:60]}"
                result_str, error = execute_search_reference(query)
                log_entry = {"tool": "search_reference", "model": model, "rationale": rationale,
                             "code": f"search_reference({query!r})",
                             "result": "", "error": error}
            elif fn == "render_chart":
                _title = args.get("title", "Chart")
                _series = args.get("series", [])
                _y_label = args.get("y_label", "Value")
                _rationale = args.get("rationale", "render chart")
                rationale = _rationale
                chart_b64, error = execute_render_chart(_title, _series, _y_label, _rationale)
                result_str = f"Chart rendered: {_title}" if not error else ""
                log_entry = {"tool": "render_chart", "model": model, "rationale": rationale,
                             "code": f"render_chart(title={_title!r}, series=[{len(_series)} series])",
                             "result": result_str, "error": error,
                             "chart_b64": chart_b64 if not error else ""}
            elif fn == "render_section_collage":
                _rfx_id = args.get("rfx_id", rfx_id or "")
                rationale = f"section collage for {_rfx_id}"
                chart_b64, error = execute_render_section_collage(_rfx_id)
                result_str = f"Section collage rendered for {_rfx_id}" if not error else ""
                log_entry = {"tool": "render_section_collage", "model": model, "rationale": rationale,
                             "code": f"render_section_collage(rfx_id={_rfx_id!r})",
                             "result": result_str, "error": error,
                             "chart_b64": chart_b64 if not error else ""}
            else:
                result_str, error = "", f"Unknown tool: {fn}"
                log_entry = {"tool": fn, "model": model, "rationale": "", "code": "", "result": "", "error": error}

            content = f"ERROR:\n{error}" if error else result_str
            log_entry["result"] = content[:2000]

            tool_log.append(log_entry)

            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": content[:8000]
            })

    # Hit max rounds — find the last substantive assistant text rather than
    # leaking a raw tool-result (which may be a traceback) as the answer.
    last_assistant_text = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            last_assistant_text = m["content"]
            break
    final = last_assistant_text or (
        "I exhausted the maximum number of query attempts without producing a "
        "clean result. The errors from each attempt are visible in the tool-call "
        "log below. Please try rephrasing or narrowing the question."
    )
    return final, history + [{"role": "user", "content": user_message}], tool_log


# ── RFQ Co-pilot — helpers, tools, agent loop ────────────────────────────────

# Section prefix mapping for auto-generating line_ids at finalize time
_SECTION_PREFIX = {
    "soil_prep": "SP", "trees": "TR", "shrubs": "SH",
    "ground_covers": "GC", "lawn": "LW", "staking": "ST",
}



def _gen_rfx_id(name_hint: str = "") -> str:
    """Generate a unique rfx_id from a project-name hint + sequential count."""
    year = datetime.now(timezone.utc).year
    slug = re.sub(r"[^A-Z0-9]+", "-", name_hint.upper()[:20]).strip("-") or "NEW"
    conn = _db_conn()
    n = conn.execute("SELECT COUNT(*) FROM rfx_projects").fetchone()[0] + 1
    conn.close()
    return f"RFX-{year}-{slug}-{n:03d}"


def _extract_file_text(uploaded_file) -> str:
    """Extract readable text from an uploaded Streamlit file (PDF, xlsx, txt/csv/md)."""
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    try:
        if name.endswith(".pdf"):
            import pdfplumber
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                pages = []
                for i, page in enumerate(pdf.pages):
                    text = page.extract_text() or ""
                    tables = page.extract_tables()
                    if tables:
                        for tbl in tables:
                            for row in tbl:
                                if row:
                                    pages.append(" | ".join(str(c or "").strip() for c in row))
                    else:
                        pages.append(f"[PAGE {i+1}]\n{text}")
            return "\n".join(pages)
        elif name.endswith((".xlsx", ".xls")):
            xl = pd.ExcelFile(io.BytesIO(raw))
            parts = []
            for sheet in xl.sheet_names:
                df = xl.parse(sheet, header=None)
                parts.append(f"[SHEET: {sheet}]\n{df.to_string(index=False, na_rep='')}")
            return "\n".join(parts)
        else:  # txt / csv / md — assume utf-8
            return raw.decode("utf-8", errors="replace")
    except Exception as exc:
        return f"(Could not extract file text: {exc})"


def _status_tag(line: dict) -> str:
    """Derive a review-status tag from a staged line's value_source."""
    vs = (line.get("value_source") or "").lower()
    if vs == "calculated":
        return "calculated"
    if vs in ("user_provided", "drawing_derived", "extracted"):
        return "confirmed"
    if vs in ("estimated", "inferred", "assumption", "vendor_proposed"):
        return "assumption"
    return "needs_input"


def execute_propose_line_item(
    description: str,
    unit: str,
    quantity: float | None,
    section: str,
    spec_notes: str = "",
    species_name: str = "",
    boq_unit_rate: float | None = None,
    value_source: str = "user_provided",
) -> tuple:
    """Stage one line item in session state. Called by co-pilot during conversation."""
    if "copilot_staged_lines" not in st.session_state:
        st.session_state.copilot_staged_lines = []
    line = {
        "description": description,
        "species_name": species_name or None,
        "unit": unit,
        "quantity": quantity,
        "section": section,
        "spec_notes": spec_notes or None,
        "boq_unit_rate": boq_unit_rate,
        "value_source": value_source,
        "formula": None,
        "remarks": "",
        "accepted": False,
    }
    st.session_state.copilot_staged_lines.append(line)
    n = len(st.session_state.copilot_staged_lines)
    _save_draft()
    qty_str = f"{quantity} {unit}" if quantity is not None else f"? {unit}"
    rate_str = f" | BOQ ref: ₹{boq_unit_rate}/unit" if boq_unit_rate else ""
    return (
        f"✓ Staged line #{n}: [{section}] {description} — {qty_str}{rate_str}. "
        f"Total staged: {n} line(s).",
        None,
    )


def _build_boq_xlsx(rfx_name: str, rfx_id: str, staged: list) -> bytes:
    """
    Generate a BOQ xlsx matching the Kukas Hor BOQ tab structure:
    Section | Item | Description | Specification | Qty | Unit | Unit Rate (blank) | Amount (blank)
    Returns raw bytes suitable for st.download_button.
    """
    import io
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Horticulture BOQ"

    # ── colour palette ────────────────────────────────────────────────────────
    HDR_FILL  = PatternFill("solid", fgColor="1F4E79")
    CAT_FILL  = PatternFill("solid", fgColor="2E75B6")
    HDR_FONT  = Font(bold=True, color="FFFFFF", size=11)
    CAT_FONT  = Font(bold=True, color="FFFFFF", size=10)
    BODY_FONT = Font(size=10)
    THIN      = Side(style="thin", color="BFBFBF")
    BORDER    = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
    WRAP      = Alignment(wrap_text=True, vertical="top")
    CENTER    = Alignment(horizontal="center", vertical="center")

    # ── title rows ────────────────────────────────────────────────────────────
    ws.merge_cells("A1:H1")
    ws["A1"] = f"HORTICULTURE BILL OF QUANTITIES — {rfx_name.upper()}"
    ws["A1"].font = Font(bold=True, size=13)
    ws["A1"].alignment = Alignment(horizontal="center")
    ws["A2"] = f"RFx ID: {rfx_id}    |    Issued: {datetime.now().strftime('%d %b %Y')}"
    ws["A2"].font = Font(italic=True, size=10, color="595959")
    ws.merge_cells("A2:H2")
    ws["A2"].alignment = Alignment(horizontal="center")

    # ── column headers (row 4) ────────────────────────────────────────────────
    headers = ["Section", "Item No.", "Description", "Specification", "Qty", "Unit",
               "Unit Rate (₹)", "Amount (₹)"]
    col_widths = [18, 9, 36, 42, 9, 8, 14, 14]
    ws.append([])  # row 3 blank
    ws.append(headers)
    for col_idx, (hdr, width) in enumerate(zip(headers, col_widths), start=1):
        cell = ws.cell(row=4, column=col_idx)
        cell.fill = HDR_FILL
        cell.font = HDR_FONT
        cell.alignment = CENTER
        cell.border = BORDER
        ws.column_dimensions[get_column_letter(col_idx)].width = width

    # ── category ordering (mirrors review table) ──────────────────────────────
    _SECTION_ORDER = ["trees", "shrubs", "ground_covers", "lawn", "soil_prep", "staking", "other"]
    _SECTION_LABEL = {
        "trees":        "I — Trees / Palms / Bamboo",
        "shrubs":       "II — Shrubs / Climbers",
        "ground_covers":"III — Ground Covers",
        "lawn":         "IV — Lawn",
        "soil_prep":    "V — Site Prep / Amendments",
        "staking":      "VI — Staking",
        "other":        "VII — Other",
    }

    from collections import defaultdict as _dd2
    by_section = _dd2(list)
    for line in staged:
        by_section[line.get("section", "other")].append(line)

    row_num = 5
    for sec in _SECTION_ORDER:
        lines_in_sec = by_section.get(sec, [])
        if not lines_in_sec:
            continue

        # Section header row
        ws.merge_cells(f"A{row_num}:H{row_num}")
        ws.cell(row=row_num, column=1).value = _SECTION_LABEL.get(sec, sec.upper())
        for c in range(1, 9):
            cell = ws.cell(row=row_num, column=c)
            cell.fill = CAT_FILL
            cell.font = CAT_FONT
            cell.border = BORDER
        ws.cell(row=row_num, column=1).alignment = Alignment(horizontal="left", vertical="center")
        ws.row_dimensions[row_num].height = 18
        row_num += 1

        for item_n, line in enumerate(lines_in_sec, start=1):
            formula_note = line.get("formula") or ""
            spec = line.get("spec_notes") or ""
            full_spec = (spec + ("  [" + formula_note + "]" if formula_note else "")).strip()
            vals = [
                _SECTION_LABEL.get(sec, sec),
                f"{sec[:2].upper()}-{item_n:02d}",
                line.get("description", ""),
                full_spec,
                line.get("quantity"),
                line.get("unit", ""),
                "",   # Unit Rate — to be filled by vendor
                "",   # Amount — formula can be added manually
            ]
            ws.append(vals)
            for c_idx, val in enumerate(vals, start=1):
                cell = ws.cell(row=row_num, column=c_idx)
                cell.font = BODY_FONT
                cell.border = BORDER
                cell.alignment = WRAP
                if c_idx in (5, 6, 7, 8):
                    cell.alignment = Alignment(horizontal="center", vertical="top")
            ws.row_dimensions[row_num].height = 32
            row_num += 1

        # Blank separator
        row_num += 1

    # ── Notes row ─────────────────────────────────────────────────────────────
    ws.cell(row=row_num, column=1).value = (
        "NOTE: Unit rates to be filled by vendor. "
        "All quantities are buyer-derived per BOQ grounding rules. "
        "Maintenance clause (12 months post-completion) to be priced separately."
    )
    ws.cell(row=row_num, column=1).font = Font(italic=True, size=9, color="595959")
    ws.merge_cells(f"A{row_num}:H{row_num}")

    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def execute_finalize_rfx(rfx_name: str, rfx_description: str = "") -> tuple:
    """
    Commit all staged lines to the DB under a new rfx_id and generate a
    downloadable BOQ xlsx. No email/SMTP — analyst downloads and distributes.
    """
    staged = st.session_state.get("copilot_staged_lines", [])
    if not staged:
        return "", "No lines staged yet — use propose_line_item() to build up the line list first."

    # Prevent concurrent double-submit (rapid clicks / Streamlit reruns)
    if st.session_state.get("_finalize_in_progress"):
        return "", "Finalization already in progress — please wait."
    st.session_state["_finalize_in_progress"] = True

    # Reuse rfx_id across retries within the same session so that the
    # DELETE-before-INSERT on rfx_lines always targets the correct id.
    # A new session (page refresh) gets a fresh id via _gen_rfx_id.
    if "copilot_rfx_id" not in st.session_state:
        st.session_state["copilot_rfx_id"] = _gen_rfx_id(rfx_name)
    rfx_id = st.session_state["copilot_rfx_id"]
    conn = _db_conn()
    now = datetime.now(timezone.utc).isoformat()
    inserted: list[str] = []

    try:
        # Insert project record (with commercial terms assembled from project_context)
        ctx = st.session_state.get("copilot_project_context", {})
        terms_json = json.dumps({
            "payment_schedule": ctx.get("payment_schedule", ""),
            "quote_validity": ctx.get("quote_validity", ""),
            "maintenance_duration": ctx.get("maintenance_duration", ""),
        })
        vision_summary = st.session_state.get("copilot_vision_summary", "")
        conn.execute(
            "INSERT OR IGNORE INTO rfx_projects (rfx_id, name, description, created_at, terms, vision_summary) VALUES (?,?,?,?,?,?)",
            (rfx_id, rfx_name, rfx_description or "", now, terms_json, vision_summary),
        )
        # Update terms + vision_summary if the row already existed (retry path)
        conn.execute(
            "UPDATE rfx_projects SET terms = ? WHERE rfx_id = ? AND (terms IS NULL OR terms = '')",
            (terms_json, rfx_id),
        )
        if vision_summary:
            conn.execute(
                "UPDATE rfx_projects SET vision_summary = ? WHERE rfx_id = ? AND (vision_summary IS NULL OR vision_summary = '')",
                (vision_summary, rfx_id),
            )

        # Clear any partial rows from a previous failed attempt for this rfx_id
        conn.execute("DELETE FROM rfx_lines WHERE rfx_id = ?", (rfx_id,))

        # Generate line_ids and insert rfx_lines
        section_counters: dict[str, int] = {}
        for line in staged:
            section = line.get("section", "other")
            prefix = _SECTION_PREFIX.get(section, "OT")
            section_counters[section] = section_counters.get(section, 0) + 1
            rfx_slug = rfx_id.replace("-", "_").upper()
            line_id = f"{rfx_slug}_L_{prefix}_{section_counters[section]:02d}"

            desc = line.get("description", "")
            sp = line.get("species_name") or ""
            snotes = line.get("spec_notes") or ""
            search_text = " | ".join(p for p in [desc, sp, snotes] if p)

            conn.execute(
                """INSERT OR REPLACE INTO rfx_lines
                   (rfx_id, line_id, section, description, species_name,
                    unit, quantity, boq_unit_rate, boq_total, spec_notes,
                    gpt_discrepancy, search_text, spec_parsed)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rfx_id, line_id, section, desc,
                 sp or None,
                 line.get("unit"), line.get("quantity"),
                 line.get("boq_unit_rate"), None, snotes or None,
                 None, search_text, None),
            )
            inserted.append(f"{line_id}: {desc}")

        conn.commit()
    finally:
        conn.close()
        st.session_state.pop("_finalize_in_progress", None)

    # On success delete the draft and clear the cached rfx_id
    _delete_draft(rfx_id)
    st.session_state.pop("copilot_rfx_id", None)

    # Generate BOQ xlsx and store bytes in session state for download
    xlsx_bytes = _build_boq_xlsx(rfx_name, rfx_id, staged)
    st.session_state.copilot_boq_xlsx = xlsx_bytes
    st.session_state.copilot_boq_filename = f"{rfx_id}_BOQ.xlsx"

    # Clear staged lines and record the new rfx_id
    st.session_state.copilot_staged_lines = []
    st.session_state.copilot_finalized_rfx_id = rfx_id

    return json.dumps({
        "rfx_id": rfx_id,
        "lines_created": len(inserted),
        "boq_xlsx_ready": True,
        "lines": inserted,
    }), None


def execute_lookup_similar_rfx(query: str) -> tuple:
    """
    Search ALL past rfx_lines (across all rfx_ids) for items similar to `query`.
    Uses the same 3-strategy cascade as resolve_lines, but without rfx_id filtering,
    and enriches results with unit, qty, boq_unit_rate, and rfx_id for context.
    """
    conn = _db_conn()
    try:
        # Strategy 1 — exact
        rows = conn.execute(
            "SELECT rl.line_id, rl.rfx_id, rl.section, rl.description, rl.species_name, "
            "rl.unit, rl.quantity, rl.boq_unit_rate, rl.search_text "
            "FROM rfx_lines rl "
            "WHERE LOWER(search_text) LIKE LOWER(?)",
            (f"%{query.strip()}%",),
        ).fetchall()

        strategy = "exact"
        if not rows:
            # Strategy 2 — word-level fuzzy (tokens ≥4 chars)
            tokens = [w for w in re.split(r"\W+", query) if len(w) >= 4]
            hit_count: dict[str, int] = {}
            hit_rows: dict[str, dict] = {}
            _sql = ("SELECT line_id, rfx_id, section, description, species_name, "
                    "unit, quantity, boq_unit_rate FROM rfx_lines "
                    "WHERE LOWER(search_text) LIKE LOWER(?)")
            for tok in tokens:
                for r in conn.execute(_sql, (f"%{tok}%",)).fetchall():
                    lid = f"{r['rfx_id']}::{r['line_id']}"
                    hit_count[lid] = hit_count.get(lid, 0) + 1
                    if lid not in hit_rows:
                        hit_rows[lid] = dict(r)
            if hit_rows:
                rows = [
                    type("R", (), dict(r))()  # duck-type as Row
                    for r in sorted(hit_rows.values(),
                                    key=lambda x: hit_count[f"{x['rfx_id']}::{x['line_id']}"],
                                    reverse=True)
                ]
                strategy = "fuzzy_word"

        if not rows:
            return json.dumps({"matches": [], "strategy": "exhausted",
                               "guidance": "No similar items found in past RFx data."}), None

        matches = []
        for r in rows[:10]:
            try:
                m = {
                    "rfx_id": r["rfx_id"] if hasattr(r, "__getitem__") else getattr(r, "rfx_id", ""),
                    "line_id": r["line_id"] if hasattr(r, "__getitem__") else getattr(r, "line_id", ""),
                    "section": r["section"] if hasattr(r, "__getitem__") else getattr(r, "section", ""),
                    "description": r["description"] if hasattr(r, "__getitem__") else getattr(r, "description", ""),
                    "unit": r["unit"] if hasattr(r, "__getitem__") else getattr(r, "unit", ""),
                    "quantity": r["quantity"] if hasattr(r, "__getitem__") else getattr(r, "quantity", None),
                    "boq_unit_rate": r["boq_unit_rate"] if hasattr(r, "__getitem__") else getattr(r, "boq_unit_rate", None),
                }
                matches.append(m)
            except Exception:
                pass

        return json.dumps({"matches": matches, "total_found": len(matches),
                           "strategy": strategy}), None
    except Exception as exc:
        return "", f"lookup_similar_rfx error: {exc}"
    finally:
        conn.close()


import math as _math

def execute_derive_site_prep_and_amendments(plant_list: list) -> tuple:
    """
    Deterministic derivation of site-prep and soil-amendment quantities from a staged plant list.
    Formulas from landscaping-boq-grounding-rules.md §3 (Kukas/SKYDOME precedent).

    plant_list: list of dicts with keys: section, unit, quantity
      sections recognised: trees, shrubs, ground_covers, lawn
      units: trees/shrubs → Nos; ground_covers/lawn → Sqm

    Returns JSON string with derived line items, each tagged value_source="calculated"
    and showing the formula used.
    """
    # ── 1. Aggregate category totals ──────────────────────────────────────────
    tree_count = 0.0
    shrub_count = 0.0
    gc_area = 0.0
    lawn_area = 0.0

    for line in plant_list:
        sec = (line.get("section") or "").lower()
        qty = line.get("quantity") or 0.0
        try:
            qty = float(qty)
        except (TypeError, ValueError):
            qty = 0.0
        if sec == "trees":
            tree_count += qty
        elif sec == "shrubs":
            shrub_count += qty
        elif sec == "ground_covers":
            gc_area += qty
        elif sec == "lawn":
            lawn_area += qty

    # ── 2. Volume per category (cum) — cylindrical pits, rectangular beds ────
    # Trees: π/4 × 0.9² × 0.9 per tree
    tree_vol_each = _math.pi / 4 * 0.9**2 * 0.9          # ≈ 0.5726 cum
    # Shrubs: π/4 × 0.45² × 0.45 per shrub
    shrub_vol_each = _math.pi / 4 * 0.45**2 * 0.45        # ≈ 0.07162 cum
    # Ground covers / lawn: area × 0.3m depth
    BED_DEPTH = 0.3

    tree_vol   = tree_count  * tree_vol_each
    shrub_vol  = shrub_count * shrub_vol_each
    gc_vol     = gc_area     * BED_DEPTH
    lawn_vol   = lawn_area   * BED_DEPTH
    total_vol  = tree_vol + shrub_vol + gc_vol + lawn_vol

    # ── 3. Area per category (sqm) — circular pit footprint or bed area ──────
    tree_area  = tree_count  * (_math.pi / 4 * 0.9**2)    # ≈ 0.6362 sqm/tree
    shrub_area = shrub_count * (_math.pi / 4 * 0.45**2)   # ≈ 0.1590 sqm/shrub
    total_area = tree_area + shrub_area + gc_area + lawn_area

    def _r(v, dp=2):
        return round(v, dp)

    derived = []

    # ── 4. Site prep (keyed to total planting area) ───────────────────────────
    if total_area > 0:
        derived += [
            {
                "section": "soil_prep",
                "description": "Trenching / uprooting of existing vegetation",
                "unit": "Sqm",
                "quantity": _r(total_area),
                "value_source": "calculated",
                "formula": f"{_r(total_area)} sqm = total planting area (trees {_r(tree_area)} + shrubs {_r(shrub_area)} + GC {_r(gc_area)} + lawn {_r(lawn_area)})",
            },
            {
                "section": "soil_prep",
                "description": "Anti-termite treatment",
                "unit": "Sqm",
                "quantity": _r(total_area),
                "value_source": "calculated",
                "formula": f"{_r(total_area)} sqm = total planting area (same as trenching)",
            },
        ]

    # Pit/bed digging
    if tree_count > 0:
        derived.append({
            "section": "soil_prep",
            "description": "Pit digging — trees/palms/bamboo (0.9m dia × 0.9m deep)",
            "unit": "Nos",
            "quantity": int(tree_count),
            "value_source": "calculated",
            "formula": f"{int(tree_count)} Nos = tree count (one pit per tree)",
        })
    if shrub_count > 0:
        derived.append({
            "section": "soil_prep",
            "description": "Pit digging — shrubs/climbers (0.45m dia × 0.45m deep)",
            "unit": "Nos",
            "quantity": int(shrub_count),
            "value_source": "calculated",
            "formula": f"{int(shrub_count)} Nos = shrub count (one pit per shrub)",
        })
    if gc_area > 0:
        derived.append({
            "section": "soil_prep",
            "description": "Bed preparation — ground covers/grasses (0.3m deep)",
            "unit": "Sqm",
            "quantity": _r(gc_area),
            "value_source": "calculated",
            "formula": f"{_r(gc_area)} sqm = ground cover area",
        })
    if lawn_area > 0:
        derived.append({
            "section": "soil_prep",
            "description": "Bed preparation — lawn (0.3m deep)",
            "unit": "Sqm",
            "quantity": _r(lawn_area),
            "value_source": "calculated",
            "formula": f"{_r(lawn_area)} sqm = lawn area",
        })

    # ── 5. Soil amendments (keyed to total planting volume) ───────────────────
    if total_vol > 0:
        tv = _r(total_vol)
        derived += [
            {
                "section": "soil_prep",
                "description": "Good earth (filling medium — 60% of total planting volume)",
                "unit": "Cum",
                "quantity": _r(total_vol * 0.60),
                "value_source": "calculated",
                "formula": f"60% × {tv} cum total volume = {_r(total_vol * 0.60)} cum",
            },
            {
                "section": "soil_prep",
                "description": "Cattle manure / FYM (30% of total planting volume)",
                "unit": "Cum",
                "quantity": _r(total_vol * 0.30),
                "value_source": "calculated",
                "formula": f"30% × {tv} cum = {_r(total_vol * 0.30)} cum",
            },
            {
                "section": "soil_prep",
                "description": "Coarse sand (10% of total planting volume)",
                "unit": "Cum",
                "quantity": _r(total_vol * 0.10),
                "value_source": "calculated",
                "formula": f"10% × {tv} cum = {_r(total_vol * 0.10)} cum",
            },
            {
                "section": "soil_prep",
                "description": "Neem cake (0.5 kg per cum of total planting volume)",
                "unit": "Kg",
                "quantity": _r(total_vol * 0.5),
                "value_source": "calculated",
                "formula": f"0.5 kg × {tv} cum = {_r(total_vol * 0.5)} kg",
            },
            {
                "section": "soil_prep",
                "description": "Bone meal (0.5 kg per cum of total planting volume)",
                "unit": "Kg",
                "quantity": _r(total_vol * 0.5),
                "value_source": "calculated",
                "formula": f"0.5 kg × {tv} cum = {_r(total_vol * 0.5)} kg",
            },
            {
                "section": "soil_prep",
                "description": "Vermicompost (10 kg per cum of total planting volume)",
                "unit": "Kg",
                "quantity": _r(total_vol * 10.0),
                "value_source": "calculated",
                "formula": f"10 kg × {tv} cum = {_r(total_vol * 10.0)} kg",
            },
            {
                "section": "soil_prep",
                "description": "Cocopeat (10 kg per cum of total planting volume)",
                "unit": "Kg",
                "quantity": _r(total_vol * 10.0),
                "value_source": "calculated",
                "formula": f"10 kg × {tv} cum = {_r(total_vol * 10.0)} kg",
            },
        ]

    # ── 6. Inorganic fertilizer (keyed to total planting area) ────────────────
    if total_area > 0:
        ta = _r(total_area)
        derived += [
            {
                "section": "soil_prep",
                "description": "Urea / Ammonium Sulphate (30 gm per sqm of total planting area)",
                "unit": "Kg",
                "quantity": _r(total_area * 30 / 1000),
                "value_source": "calculated",
                "formula": f"30 gm × {ta} sqm ÷ 1000 = {_r(total_area * 30 / 1000)} kg",
            },
            {
                "section": "soil_prep",
                "description": "Potassium Sulphate / MOP (20 gm per sqm of total planting area)",
                "unit": "Kg",
                "quantity": _r(total_area * 20 / 1000),
                "value_source": "calculated",
                "formula": f"20 gm × {ta} sqm ÷ 1000 = {_r(total_area * 20 / 1000)} kg",
            },
        ]

    # ── 7. Staking (1 stake per tree) ─────────────────────────────────────────
    if tree_count > 0:
        derived.append({
            "section": "staking",
            "description": "Tree staking (1 stake per tree / palm / bamboo clump)",
            "unit": "Nos",
            "quantity": int(tree_count),
            "value_source": "calculated",
            "formula": f"{int(tree_count)} Nos = tree count",
        })

    # ── 8. Auto-stage derived lines into session state ────────────────────────
    if "copilot_staged_lines" not in st.session_state:
        st.session_state.copilot_staged_lines = []
    # Remove any previously derived lines (avoid duplicates on re-run)
    st.session_state.copilot_staged_lines = [
        l for l in st.session_state.copilot_staged_lines
        if l.get("value_source") != "calculated"
    ]
    for d in derived:
        staged = {
            "description": d["description"],
            "species_name": None,
            "unit": d["unit"],
            "quantity": d["quantity"],
            "section": d["section"],
            "spec_notes": None,
            "boq_unit_rate": None,
            "value_source": "calculated",
            "formula": d.get("formula"),
            "remarks": "",
            "accepted": False,
        }
        st.session_state.copilot_staged_lines.append(staged)

    # ── 9. Return lean summary to LLM ─────────────────────────────────────────
    summary = {
        "derived_lines_count": len(derived),
        "auto_staged": True,
        "totals": {
            "total_planting_area_sqm": _r(total_area),
            "total_planting_volume_cum": _r(total_vol),
        },
        "key_quantities": {
            "good_earth_cum": _r(total_vol * 0.60),
            "neem_cake_kg": _r(total_vol * 0.5),
            "staking_nos": int(tree_count),
        },
        "note": (
            f"All {len(derived)} site-prep/amendment lines have been auto-staged with "
            "value_source='calculated' and formula shown. Soil ratios (60/30/10, "
            "neem cake 0.5 kg/cum) are Kukas defaults — flag if project differs."
        ),
    }
    return json.dumps(summary, indent=2), None


def execute_trigger_review() -> tuple:
    """Switch the co-pilot UI from open chat into structured review mode."""
    n = len(st.session_state.get("copilot_staged_lines", []))
    needs = sum(
        1 for l in st.session_state.get("copilot_staged_lines", [])
        if _status_tag(l) == "needs_input" and not l.get("accepted")
    )
    st.session_state.copilot_review_mode = True
    return (
        json.dumps({
            "review_mode_activated": True,
            "total_staged": n,
            "needs_input_count": needs,
            "message": (
                f"Review table is now shown to the analyst. "
                f"{n} lines staged; {needs} marked 'needs_input'. "
                "The analyst will type remarks on specific lines and submit them. "
                "Respond only to lines with remarks — revise, clarify, or explain each one."
            ),
        }),
        None,
    )


COPILOT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "propose_line_item",
            "description": (
                "Stage one confirmed line item as the conversation progresses. "
                "Call this as soon as you have enough information for a line — "
                "description, unit, section, and ideally quantity. "
                "Do NOT wait until all lines are known before calling this. "
                "Calling this does not finalize the RFx — the buyer can still add or change lines."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string", "description": "Item description as it will appear in the RFx"},
                    "unit": {"type": "string", "description": "Unit of measure (Nos, Sqm, Cum, Kg, Rmt, Lump, Each)"},
                    "quantity": {"type": ["number", "null"], "description": "Numeric quantity; null if unknown/TBD"},
                    "section": {
                        "type": "string",
                        "enum": ["soil_prep", "trees", "shrubs", "ground_covers", "lawn", "staking", "other"],
                        "description": "BOQ section this line belongs to",
                    },
                    "spec_notes": {"type": "string", "description": "Height, caliper, grade or other spec (e.g. '3.0-3.5m ht, 75mm caliper')"},
                    "species_name": {"type": "string", "description": "Botanical species name if a plant item, else omit"},
                    "boq_unit_rate": {"type": ["number", "null"], "description": "Client's own reference unit rate in INR; null if not stated"},
                },
                "required": ["description", "unit", "section"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finalize_rfx",
            "description": (
                "Commit all staged line items to the database under a new rfx_id and generate "
                "a downloadable BOQ xlsx. Call ONLY when: (1) all sections are covered, "
                "(2) no open clarifications remain, and (3) the buyer has explicitly confirmed "
                "the line list is complete. This is irreversible — do not call speculatively."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "rfx_name": {"type": "string", "description": "Short descriptive project name (e.g. 'Tower B Landscaping Phase 2')"},
                    "rfx_description": {"type": "string", "description": "Optional one-line project description"},
                },
                "required": ["rfx_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_similar_rfx",
            "description": (
                "Search all past RFx projects in the database for line items similar to `query`. "
                "Use this when the buyer is unsure how to describe a line, wants to know how "
                "similar items were specified before, or asks about typical quantities/rates. "
                "Returns matching lines with their section, unit, quantity, and BOQ reference rate."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Item or category to look up (e.g. 'bamboo staking', 'soil preparation pit', 'ficus tree')"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_reference",
            "description": (
                "Free-text web search for spec conventions, pricing norms, or industry standards. "
                "Use when the buyer references a category the DB doesn't contain, or asks "
                "'what's typical for X?' Returns web snippets + source links."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Free-text search question or phrase"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "derive_site_prep_and_amendments",
            "description": (
                "Deterministically calculate all site-prep and soil-amendment quantities "
                "from the currently staged plant list. Uses the Kukas/SKYDOME precedent formulas: "
                "cylindrical pit volumes (0.9m dia × 0.9m deep for trees, 0.45m × 0.45m for shrubs), "
                "60/30/10 earth/manure/sand split, 0.5 kg/cum neem cake & bone meal, "
                "10 kg/cum vermicompost & cocopeat, 30 gm/sqm urea, 20 gm/sqm MOP, "
                "1 stake per tree. "
                "Call this ONCE after the plant list is staged (all trees, shrubs, ground covers, "
                "lawn areas confirmed) — do NOT manually estimate these quantities. "
                "Returns each derived line with its formula for inspection."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plant_list": {
                        "type": "array",
                        "description": (
                            "The staged plant lines — pass st.session_state.copilot_staged_lines "
                            "or an equivalent list. Each item needs: section, unit, quantity."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "section": {"type": "string"},
                                "unit": {"type": "string"},
                                "quantity": {"type": ["number", "null"]},
                            },
                        },
                    },
                },
                "required": ["plant_list"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "trigger_review",
            "description": (
                "Switch the UI from open chat to the structured review table. "
                "Call this ONCE after all plant lines are staged AND "
                "derive_site_prep_and_amendments() has been called. "
                "Do NOT call finalize_rfx() directly — the analyst reviews first. "
                "After triggering review, your role changes: respond only to "
                "specific line remarks submitted by the analyst, not general chat."
            ),
            "parameters": {"type": "object", "properties": {}, "required": []},
        },
    },
]

def _scope_interpretation_call(project_context: dict, vision_text: str) -> str:
    """
    One-shot LLM call: interpret the analyst's vision text in light of the
    Step 1 structured facts. Returns a brief "Here's what I understood" summary
    covering: implied plant categories, style/character, constraints, and open gaps.
    """
    ctx = project_context
    scope_list = ", ".join(ctx.get("scope_categories", [])) or "not specified"
    system = (
        "You are an expert landscaping RFQ analyst. "
        "Given structured project facts and a free-text design vision, produce a concise "
        "bullet-point summary (5–8 bullets) of what you understood. "
        "Cover: implied plant categories (trees/shrubs/ground cover/lawn), design character, "
        "any climate or site constraints implied by the location, and genuine open questions "
        "the co-pilot will need to resolve next. "
        "Do NOT ask for facts already stated (location, area, project type, maintenance). "
        "Keep it under 150 words."
    )
    user = (
        f"Project type: {ctx.get('project_type','')}\n"
        f"Location: {ctx.get('location','')}\n"
        f"Approximate area: {ctx.get('area','')}\n"
        f"Scope categories: {scope_list}\n"
        f"Plant selection approach: {ctx.get('plant_selection','')}\n"
        f"Maintenance: {ctx.get('maintenance_duration','')}\n\n"
        f"Design vision:\n{vision_text}"
    )
    resp = openai_client.chat.completions.create(
        model=MODEL_LIGHT,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0,
        max_tokens=300,
    )
    return resp.choices[0].message.content or ""


COPILOT_SYSTEM_PROMPT = """\
You are an RFQ co-pilot helping a professional buyer create a new Request for Quotation (RFx).
Your job is to work conversationally to produce a complete, coherent set of line items that can
be issued to vendors for pricing.

TOOLS YOU HAVE:
- propose_line_item()                  — stage a confirmed plant line as the conversation progresses (call per-line, not in bulk)
- derive_site_prep_and_amendments()    — compute ALL soil/amendment/staking quantities deterministically once the plant list is staged
- trigger_review()                     — switch to the structured review table once the draft is complete (call after derive_site_prep_and_amendments)
- finalize_rfx()                       — commit staged lines to the DB (only available from the review UI, not from chat)
- lookup_similar_rfx()                 — find similar items in past RFx data for spec/rate reference
- search_reference()                   — free-text web search for spec conventions the DB doesn't cover

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GROUNDING RULES — READ BEFORE PROPOSING ANY QUANTITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

IF the category is site preparation, soil amendments, fertilizer, or staking
  (i.e. anything covered by §3 of the landscaping BOQ grounding rules —
  good earth, cattle manure, sand, neem cake, bone meal, vermicompost,
  cocopeat, urea, potassium sulphate/MOP, anti-termite, trenching/uprooting,
  pit/bed digging, staking):
  → ALWAYS call derive_site_prep_and_amendments(). Never propose these
    quantities manually or from general knowledge. These quantities are
    deterministic functions of the plant list; any number you supply from
    "common knowledge" will contradict the actual formula and cannot be cited.

ELSE IF the category is NOT covered by the grounding rules (hardscape,
  irrigation, lighting, or a "suggest a typical palette" request):
  → Step 1: Call lookup_similar_rfx() to check past RFx precedent.
  → Step 2: If lookup_similar_rfx() returns nothing relevant, call
    search_reference() for published spec conventions.
  → Step 3: Cite which tier the answer came from ("past RFx precedent" /
    "reference search" / "general estimate — unverified").
  → ALWAYS stage such lines with value_source: "estimated". These lines
    will appear as status "needs_input" in the review table and must be
    explicitly accepted by the analyst before finalisation.
  → NEVER answer these from unguarded general knowledge without first
    calling at least lookup_similar_rfx().

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

BEHAVIOR:
1. Start by understanding project scope: category, location, approximate scale.
2. If a BOQ file is uploaded, the plant lines are pre-parsed and injected as structured JSON.
   Call propose_line_item() for each extracted line — flag ambiguities or missing values.
3. If two documents are provided (client BOQ + AI draft), reconcile them:
   - Use the client/authoritative BOQ as the source of truth.
   - Flag discrepancies to the buyer (unit differences, quantity gaps, missing items) — NEVER
     silently pick one version over the other.
4. Ask clarifying questions when quantity/unit is missing, spec is too vague, or a line
   could mean multiple things. Keep missing-info lines — stage them with value_source
   "pending_confirmation" rather than dropping them.
5. Once all plant lines (trees, shrubs, ground covers, lawn) are staged and quantities
   confirmed, call derive_site_prep_and_amendments() with the full staged list.
   It auto-stages the derived lines. Briefly summarise: total area, total volume,
   key quantities, and the default ratio assumptions.
6. After derive_site_prep_and_amendments() returns, call trigger_review() IMMEDIATELY.
   Do not ask "shall I show the review?" — just call it. The analyst reviews in the table.
7. In REVIEW MODE (after trigger_review): respond ONLY to the specific line remarks the
   analyst submits. For each remarked line: revise the quantity/spec, ask one clarifying
   question, or explain why it is derived that way. Do not summarise the whole list again.
8. Do NOT call finalize_rfx() from chat — finalisation is controlled by the review UI.

CONSTRAINTS:
- Keep the buyer informed: after each propose_line_item() call, mention how many lines
  are staged so far.
- Sections to use: soil_prep | trees | shrubs | ground_covers | lawn | staking | other.
- NEVER manually estimate soil amendment, pit-digging, or staking quantities — always
  derive them with derive_site_prep_and_amendments().
- NEVER call trigger_review() more than once per session.
"""


def run_copilot_agent(user_message: str, history: list) -> tuple:
    """
    RFQ co-pilot agent loop.
    Returns (final_response, updated_history, tool_log).
    """
    messages = [{"role": "system", "content": COPILOT_SYSTEM_PROMPT}] + history + [
        {"role": "user", "content": user_message}
    ]
    tool_log: list[dict] = []

    for _ in range(MAX_TOOL_ROUNDS):
        response = openai_client.chat.completions.create(
            model=MODEL_HEAVY,  # co-pilot always uses heavy model — it's doing complex reasoning
            messages=messages,
            tools=COPILOT_TOOLS,
            tool_choice="auto",
            temperature=0,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            final = msg.content or ""
            updated = history + [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": final},
            ]
            return final, updated, tool_log

        messages.append(msg)
        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            fn = tc.function.name

            if fn == "propose_line_item":
                result_str, error = execute_propose_line_item(
                    description=args.get("description", ""),
                    unit=args.get("unit", ""),
                    quantity=args.get("quantity"),
                    section=args.get("section", "other"),
                    spec_notes=args.get("spec_notes", ""),
                    species_name=args.get("species_name", ""),
                    boq_unit_rate=args.get("boq_unit_rate"),
                )
                log_entry = {"tool": "propose_line_item", "code": json.dumps(args),
                             "result": "", "error": error}
            elif fn == "finalize_rfx":
                result_str, error = execute_finalize_rfx(
                    rfx_name=args.get("rfx_name", "Unnamed RFx"),
                    rfx_description=args.get("rfx_description", ""),
                )
                log_entry = {"tool": "finalize_rfx", "code": json.dumps(args),
                             "result": "", "error": error}
            elif fn == "lookup_similar_rfx":
                result_str, error = execute_lookup_similar_rfx(args.get("query", ""))
                log_entry = {"tool": "lookup_similar_rfx", "code": json.dumps(args),
                             "result": "", "error": error}
            elif fn == "search_reference":
                result_str, error = execute_search_reference(args.get("query", ""))
                log_entry = {"tool": "search_reference", "code": json.dumps(args),
                             "result": "", "error": error}
            elif fn == "derive_site_prep_and_amendments":
                plant_list = args.get("plant_list")
                if plant_list is None:
                    plant_list = st.session_state.get("copilot_staged_lines", [])
                result_str, error = execute_derive_site_prep_and_amendments(plant_list)
                log_entry = {"tool": "derive_site_prep_and_amendments",
                             "code": json.dumps({"plant_list_len": len(plant_list)}),
                             "result": "", "error": error}
            elif fn == "trigger_review":
                result_str, error = execute_trigger_review()
                log_entry = {"tool": "trigger_review", "code": "{}", "result": "", "error": error}
            else:
                result_str, error = "", f"Unknown tool: {fn}"
                log_entry = {"tool": fn, "code": "", "result": "", "error": error}

            content = f"ERROR:\n{error}" if error else result_str
            log_entry["result"] = content[:2000]
            tool_log.append(log_entry)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": content[:8000]})

    # Max rounds exhausted
    def _mget(m, key):
        return m.get(key) if isinstance(m, dict) else getattr(m, key, None)

    last = next(
        (_mget(m, "content") for m in reversed(messages)
         if _mget(m, "role") == "assistant" and _mget(m, "content")), ""
    )
    return (last or "Reached max tool rounds without a final answer."), history, tool_log


# ── Vendor type registry ─────────────────────────────────────────────────────

_VENDOR_TYPE_SUGGESTIONS = [
    "Landscape General Contractor",
    "Wholesale Plant Nursery",
    "Quarry & Stone Supplier",
    "Soil & Bulk Material Vendor",
    "Irrigation & Water Management Supplier",
    "Outdoor Lighting & Electrical Vendor",
    "Site Furnishings & Amenities Manufacturer",
    "Water Feature & Pool Specialist",
    "Precast Concrete & Hardscape Manufacturer",
    "Landscape Architectural Firm",
    "Civil & Structural Engineering Consultant",
    "Site MEP Contractor",
    "Land Surveying & Geospatial Firm",
    "Landscape Maintenance & Management Firm",
    "Other",
]


def _get_all_vendors(rfx_id: str | None = None) -> list[dict]:
    """
    Return vendors with type tags and contact info.
    Includes vendors that have extraction rows AND vendors added via the UI
    that have not yet had files extracted (metadata-only rows).
    """
    conn = _db_conn()
    rfx_clause = "AND ve.rfx_id=?" if rfx_id else ""
    params = (rfx_id,) if rfx_id else ()

    extracted_rows = conn.execute(
        f"SELECT ve.vendor_id, MAX(ve.vendor_name) AS vendor_name, "
        f"COALESCE(MAX(vm.vendor_type), 'untagged') AS vendor_type, "
        f"COALESCE(MAX(vm.display_name), '') AS display_name, "
        f"COALESCE(MAX(vm.email), '') AS email, "
        f"COALESCE(MAX(vm.website), '') AS website, "
        f"COALESCE(MAX(vm.poc_name), '') AS poc_name, "
        f"COALESCE(MAX(vm.poc_phone), '') AS poc_phone, "
        f"COALESCE(MAX(vm.inbound_folder), '') AS inbound_folder, "
        f"1 AS has_extractions "
        f"FROM vendor_extractions ve "
        f"LEFT JOIN vendor_metadata vm ON ve.vendor_id = vm.vendor_id "
        f"WHERE (ve.superseded_by IS NULL OR ve.superseded_by='') {rfx_clause} "
        f"GROUP BY ve.vendor_id "
        f"ORDER BY ve.vendor_id",
        params,
    ).fetchall()
    result = [dict(r) for r in extracted_rows]
    extracted_ids = {r["vendor_id"] for r in result}

    if rfx_id:
        meta_only = conn.execute(
            "SELECT vendor_id, COALESCE(display_name,'') AS vendor_name, "
            "vendor_type, COALESCE(display_name,'') AS display_name, "
            "COALESCE(email,'') AS email, COALESCE(website,'') AS website, "
            "COALESCE(poc_name,'') AS poc_name, COALESCE(poc_phone,'') AS poc_phone, "
            "COALESCE(inbound_folder,'') AS inbound_folder, 0 AS has_extractions "
            "FROM vendor_metadata WHERE rfx_id=? ORDER BY vendor_id",
            (rfx_id,),
        ).fetchall()
        for r in meta_only:
            if r["vendor_id"] not in extracted_ids:
                result.append(dict(r))

    conn.close()
    return result


def _set_vendor_type(vendor_id: str, vendor_type: str, display_name: str = "") -> None:
    """Upsert vendor_type + display_name into vendor_metadata."""
    conn = _db_conn()
    try:
        conn.execute("""
            INSERT INTO vendor_metadata (vendor_id, vendor_type, display_name, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(vendor_id) DO UPDATE SET
                vendor_type  = excluded.vendor_type,
                display_name = CASE WHEN excluded.display_name != '' THEN excluded.display_name
                               ELSE vendor_metadata.display_name END,
                updated_at   = excluded.updated_at
        """, (vendor_id, vendor_type, display_name, datetime.now(timezone.utc).isoformat()))
        conn.commit()
    finally:
        conn.close()


def _upsert_vendor_contact(
    vendor_id: str, rfx_id: str, display_name: str,
    vendor_type: str = "untagged",
    email: str = "", website: str = "",
    poc_name: str = "", poc_phone: str = "",
    inbound_folder: str = "",
) -> None:
    """Create or update full vendor metadata row including contact fields."""
    conn = _db_conn()
    try:
        conn.execute("""
            INSERT INTO vendor_metadata
                (vendor_id, rfx_id, display_name, vendor_type, email, website,
                 poc_name, poc_phone, inbound_folder, updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(vendor_id) DO UPDATE SET
                rfx_id         = COALESCE(excluded.rfx_id, vendor_metadata.rfx_id),
                display_name   = CASE WHEN excluded.display_name != '' THEN excluded.display_name
                                 ELSE vendor_metadata.display_name END,
                vendor_type    = excluded.vendor_type,
                email          = excluded.email,
                website        = excluded.website,
                poc_name       = excluded.poc_name,
                poc_phone      = excluded.poc_phone,
                inbound_folder = CASE WHEN excluded.inbound_folder != '' THEN excluded.inbound_folder
                                 ELSE vendor_metadata.inbound_folder END,
                updated_at     = excluded.updated_at
        """, (
            vendor_id, rfx_id, display_name, vendor_type,
            email, website, poc_name, poc_phone, inbound_folder,
            datetime.now(timezone.utc).isoformat(),
        ))
        conn.commit()
    finally:
        conn.close()


def _rfx_workspace_dir(rfx_id: str) -> str:
    """Return the workspace root for rfx_id, creating it if it doesn't exist."""
    ws_root = os.path.join(os.path.dirname(__file__), "..", "rfx_workspace")
    os.makedirs(ws_root, exist_ok=True)
    for d in os.listdir(ws_root):
        if d.startswith(rfx_id) and os.path.isdir(os.path.join(ws_root, d)):
            return os.path.join(ws_root, d)
    conn = _db_conn()
    proj = conn.execute("SELECT name FROM rfx_projects WHERE rfx_id=?", (rfx_id,)).fetchone()
    conn.close()
    proj_name = proj["name"] if proj else rfx_id
    safe_name = re.sub(r'[<>:"/\\|?*]', "", proj_name)
    folder = os.path.join(ws_root, f"{rfx_id} — {safe_name}")
    os.makedirs(folder, exist_ok=True)
    return folder


def _create_vendor_folders(rfx_id: str, vendor_slug: str) -> tuple[str, str]:
    """Create outbound/ and inbound/<vendor_slug>/ folders. Returns (outbound, inbound)."""
    ws = _rfx_workspace_dir(rfx_id)
    outbound = os.path.join(ws, "outbound")
    inbound  = os.path.join(ws, "inbound", vendor_slug)
    os.makedirs(outbound, exist_ok=True)
    os.makedirs(inbound, exist_ok=True)
    return outbound, inbound


def _gen_vendor_id(display_name: str, rfx_id: str) -> str:
    """Generate a unique V_<SLUG> vendor_id for a new vendor."""
    slug = re.sub(r"[^a-z0-9]+", "_", display_name.lower().strip())[:20].strip("_")
    base = f"V_{slug.upper()}"
    conn = _db_conn()
    existing = {r[0] for r in conn.execute(
        "SELECT vendor_id FROM vendor_metadata UNION SELECT DISTINCT vendor_id FROM vendor_extractions"
    ).fetchall()}
    conn.close()
    if base not in existing:
        return base
    for i in range(2, 100):
        candidate = f"{base}_{i}"
        if candidate not in existing:
            return candidate
    return f"{base}_{int(datetime.now().timestamp())}"


def _run_inbound_extraction(
    rfx_id: str, vendor_id: str, vendor_name: str, inbound_folder: str
) -> tuple[int, list[str]]:
    """
    Scan inbound_folder for unprocessed vendor quote files, run LLM extraction,
    write JSON to data/vendor_raw/, and load rows into vendor_extractions.
    Returns (n_extracted, error_messages).
    """
    import glob as _glob
    import sys as _sys

    _src = os.path.join(os.path.dirname(__file__), "..", "src")
    if _src not in _sys.path:
        _sys.path.insert(0, _src)

    SUPPORTED = {".pdf", ".xlsx", ".xls", ".txt", ".csv", ".md",
                 ".jpg", ".jpeg", ".png", ".webp", ".heic"}
    all_files = sorted(
        p for p in _glob.glob(os.path.join(inbound_folder, "*"))
        if os.path.splitext(p)[1].lower() in SUPPORTED and os.path.isfile(p)
    )
    if not all_files:
        return 0, ["No supported files in inbound folder."]

    conn = _db_conn()
    already_extracted = {
        r[0] for r in conn.execute(
            "SELECT source_file FROM vendor_extractions WHERE vendor_id=? AND rfx_id=?",
            (vendor_id, rfx_id),
        ).fetchall()
    }
    conn.close()

    try:
        from extract import (
            extract_pdf_text, extract_xlsx_text, extract_vendor_doc,
            load_rfx_lines_for,
        )
        from ocr import is_image_file
    except ImportError as e:
        return 0, [f"Cannot import extract.py / ocr.py: {e}"]

    try:
        rfx_lines = load_rfx_lines_for(rfx_id)
    except Exception as e:
        return 0, [f"Cannot load rfx_lines for {rfx_id}: {e}"]

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    out_dir = os.path.join(data_dir, "vendor_raw", re.sub(r"[^a-z0-9_]", "_", rfx_id.lower()))
    os.makedirs(out_dir, exist_ok=True)

    n_extracted = 0
    errors: list[str] = []

    for fpath in all_files:
        fname = os.path.basename(fpath)
        if fname in already_extracted:
            continue
        ext = os.path.splitext(fpath)[1].lower()

        # Determine document_version from existing extraction count
        conn = _db_conn()
        existing_versions = conn.execute(
            "SELECT COUNT(DISTINCT document_version) FROM vendor_extractions "
            "WHERE vendor_id=? AND rfx_id=?", (vendor_id, rfx_id)
        ).fetchone()[0]
        conn.close()
        doc_version = f"v{existing_versions + 1}"

        try:
            if is_image_file(fpath):
                from extract import extract_image_file
                out_path = extract_image_file(
                    fpath, vendor_id=vendor_id,
                    vendor_name=vendor_name, document_version=doc_version,
                    superseded_by=None, output_name=None, rfx_id=rfx_id,
                )
            else:
                if ext == ".pdf":
                    doc_text = extract_pdf_text(fpath)
                elif ext in (".xlsx", ".xls"):
                    doc_text = extract_xlsx_text(fpath)
                else:
                    with open(fpath, encoding="utf-8", errors="replace") as _f:
                        doc_text = _f.read()

                result = extract_vendor_doc(
                    doc_text, vendor_id=vendor_id, source_file=fname,
                    document_version=doc_version, superseded_by=None,
                    rfx_lines=rfx_lines,
                )
                result["rfx_id"] = rfx_id
                result["vendor_id"] = vendor_id
                result["vendor_name"] = vendor_name

                safe = re.sub(r"[^a-z0-9_]", "_", vendor_id.lower())
                out_path = os.path.join(out_dir, f"{safe}_{doc_version}.json")
                with open(out_path, "w") as _f:
                    json.dump(result, _f, indent=2, ensure_ascii=False)

            # Load the JSON into vendor_extractions
            with open(out_path) as _f:
                vendor_json = json.load(_f)

            conn = _db_conn()
            try:
                src = vendor_json.get("source_file", fname)
                conn.execute(
                    "DELETE FROM vendor_extractions WHERE vendor_id=? AND source_file=? AND rfx_id=?",
                    (vendor_id, src, rfx_id),
                )
                for ex in vendor_json.get("extractions", []):
                    flags = ex.get("flags", [])
                    conn.execute("""
                        INSERT INTO vendor_extractions
                        (rfx_id, vendor_id, vendor_name, source_file, document_version,
                         superseded_by, line_id, matched, match_confidence,
                         raw_unit_price, raw_unit, raw_currency, normalized_unit_price,
                         normalization_note, quantity_quoted, freight_included,
                         labor_included, spec_grade_quoted, spec_grade_match,
                         source_snippet, source_location, extraction_confidence,
                         value_source, flags, extracted_at, granularity)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, (
                        rfx_id, vendor_id, vendor_name,
                        src, vendor_json.get("document_version", doc_version),
                        vendor_json.get("superseded_by"),
                        ex.get("line_id"), int(ex.get("matched", False)),
                        ex.get("match_confidence"), ex.get("raw_unit_price"),
                        ex.get("raw_unit"), ex.get("raw_currency", "INR"),
                        ex.get("normalized_unit_price"), ex.get("normalization_note"),
                        ex.get("quantity_quoted"), ex.get("freight_included", "unknown"),
                        ex.get("labor_included", "unknown"), ex.get("spec_grade_quoted"),
                        ex.get("spec_grade_match", "unknown"), ex.get("source_snippet"),
                        ex.get("source_location"), ex.get("extraction_confidence"),
                        ex.get("value_source", "extracted"),
                        json.dumps(flags) if isinstance(flags, list) else flags,
                        vendor_json.get("extracted_at"),
                        ex.get("granularity", "line_item"),
                    ))
                conn.commit()
                n_extracted += 1
            finally:
                conn.close()

        except Exception as e:
            errors.append(f"{fname}: {e}")

    return n_extracted, errors


# ── Questionnaire generation ──────────────────────────────────────────────────

_QUESTIONNAIRE_GEN_SYSTEM = """\
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
    not proof. A vendor can answer from their own knowledge without gathering additional evidence.
    Examples: "Which region do you typically source [species] from?", "What is your estimated
    lead time for assembling 588 trees?", "Who would lead site execution — name the site manager
    and their experience?", "What payment terms do you require?", "Any BOQ items you'd flag or
    substitute?"
  stage_2_evaluation: Use ONLY when a project condition makes early evidence genuinely critical —
    e.g. a scarce accent tree species whose unavailability would block the project. Still generate,
    but tagged stage_2_evaluation. Ask for PROOF: photos, lab reports, named references.
    The same topic at claim-depth = Stage 1; at proof-depth = Stage 2.
  DEFAULT: If in doubt, Stage 1. RATIO: 3–4 Stage 1, at most 1 Stage 2. Total must not exceed 5.

DESIGN RULES (apply in order):
1. Only generate dimensions materially relevant to this project + vendor type.
   Omit generic questions whose answers cannot distinguish vendors on this specific scope.
2. Ask at Stage 1 (claim) depth by default:
   – "Which nurseries or regions do you source [species] from?" (S1, claim)
   – "Provide photographs of proposed [species] stock" (S2, proof — only if scarcity risk)
3. Reason from the actual BOQ:
   – Large accent trees → sourcing claim (S1); photos only if S2 warranted.
   – Any species > 100 units → nursery network, aggregation region, rough lead time (S1).
   – Maintenance > 6 months → staffing model and what's included/excluded (S1).
   – Arid/semi-arid location → which heat-tolerant amendments they'd use, water-source approach (S1).
4. Apply vendor-type-specific focus:
   – Landscape General Contractor: comparable project claim (S1), site manager identity (S1),
     sourcing regions for key species (S1), rough procurement timeline (S1), BOQ concerns/substitutions (S1),
     phasing capability (S1), maintenance staffing model (S1), payment terms (S1).
   – Wholesale Plant Nursery: which species in stock (S1), source regions (S1), rough lead times (S1),
     hold policy (S1), minimum order quantities (S1), replacement/rejection policy (S1).
   – Quarry & Stone Supplier, Soil/Bulk, Irrigation: apply most relevant claim-level dimensions (S1 default).

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


def _generate_rfx_questionnaire(rfx_id: str, vendor_type: str) -> list[dict]:
    """
    LLM-generate a project + vendor-type specific questionnaire.
    Replaces existing questions for this rfx+vendor_type in rfx_questionnaire_definitions.
    Returns the generated question list.
    """
    conn = _db_conn()
    boq_rows = conn.execute(
        "SELECT section, description, species_name, unit, quantity, boq_unit_rate, spec_notes "
        "FROM rfx_lines WHERE rfx_id=? ORDER BY section, line_id",
        (rfx_id,),
    ).fetchall()
    proj_row = conn.execute(
        "SELECT name, description, terms, vision_summary FROM rfx_projects WHERE rfx_id=?", (rfx_id,)
    ).fetchone()
    conn.close()

    boq_summary = "section | description | species | unit | qty | spec_notes\n"
    for r in boq_rows:
        boq_summary += (
            f"{r['section']} | {r['description']} | {r['species_name'] or ''} | "
            f"{r['unit']} | {r['quantity'] or ''} | {r['spec_notes'] or ''}\n"
        )
    proj_name = proj_row["name"] if proj_row else rfx_id
    proj_desc = proj_row["description"] if proj_row else ""
    proj_vision = (proj_row["vision_summary"] or "") if proj_row else ""
    terms = {}
    if proj_row and proj_row["terms"]:
        try:
            terms = json.loads(proj_row["terms"])
        except Exception:
            pass

    vision_line = f"Design vision: {proj_vision}\n" if proj_vision else ""
    user_msg = (
        f"Project: {proj_name}\n"
        f"Description: {proj_desc}\n"
        f"{vision_line}"
        f"Maintenance duration: {terms.get('maintenance_duration', 'not specified')}\n"
        f"Vendor type being assessed: {vendor_type}\n\n"
        f"BOQ ({len(boq_rows)} lines):\n{boq_summary}"
    )

    resp = openai_client.chat.completions.create(
        model=MODEL_HEAVY,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _QUESTIONNAIRE_GEN_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0,
    )
    raw = json.loads(resp.choices[0].message.content)
    questions = raw.get("questions", [])
    if not isinstance(questions, list):
        for v in raw.values():
            if isinstance(v, list):
                questions = v
                break

    now = datetime.now(timezone.utc).isoformat()
    conn = _db_conn()
    try:
        conn.execute(
            "DELETE FROM rfx_questionnaire_definitions WHERE rfx_id=? AND vendor_type=?",
            (rfx_id, vendor_type),
        )
        for q in questions:
            conn.execute("""
                INSERT OR REPLACE INTO rfx_questionnaire_definitions
                (rfx_id, vendor_type, q_id, dimension, stage, question, pass_criteria, generated_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                rfx_id, vendor_type,
                q.get("q_id", ""), q.get("dimension", ""),
                q.get("stage", "stage_1_screening"),
                q.get("question", ""), q.get("pass_criteria", ""), now,
            ))
        conn.commit()
    finally:
        conn.close()
    return questions


def _augment_rfx_questionnaire(rfx_id: str, vendor_type: str, user_prompt: str) -> list[dict]:
    """Add questions to an existing questionnaire via user prompt. Returns new questions only."""
    existing = _get_rfx_questionnaire(rfx_id, vendor_type)
    if not existing:
        return []
    next_n = len(existing) + 1
    existing_summary = json.dumps(
        [{"q_id": q["q_id"], "question": q["question"]} for q in existing], indent=2
    )
    system = (
        "You are a procurement analyst adding questions to an existing questionnaire. "
        f"Return ONLY new questions (do not repeat existing ones) as a JSON object "
        f'{{\"questions\": [...]}}. Number starting Q{next_n:02d}. '
        "Schema: [{q_id, dimension, question, pass_criteria}]. "
        "Use evidence-request phrasing. JSON object only."
    )
    resp = openai_client.chat.completions.create(
        model=MODEL_HEAVY,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": (
                f"Existing questions:\n{existing_summary}\n\n"
                f"Buyer request: {user_prompt}"
            )},
        ],
        temperature=0,
    )
    raw = json.loads(resp.choices[0].message.content)
    new_qs = raw.get("questions", [])
    if not isinstance(new_qs, list):
        new_qs = next((v for v in raw.values() if isinstance(v, list)), [])

    now = datetime.now(timezone.utc).isoformat()
    conn = _db_conn()
    try:
        for q in new_qs:
            conn.execute("""
                INSERT OR IGNORE INTO rfx_questionnaire_definitions
                (rfx_id, vendor_type, q_id, dimension, stage, question, pass_criteria, generated_at)
                VALUES (?,?,?,?,?,?,?,?)
            """, (rfx_id, vendor_type, q.get("q_id",""), q.get("dimension",""),
                  q.get("stage", "stage_1_screening"),
                  q.get("question",""), q.get("pass_criteria",""), now))
        conn.commit()
    finally:
        conn.close()
    return new_qs


def _parse_questionnaire_upload(
    uploaded_file, rfx_id: str, vendor_type: str, replace: bool = False
) -> tuple[int, str]:
    """
    Parse uploaded xlsx/csv with questionnaire questions.
    Expected columns: Question, Pass Criteria (+ optionally Dimension, Q ID).
    Returns (n_inserted, error_message).
    """
    try:
        name = uploaded_file.name.lower()
        raw = uploaded_file.read()
        df = pd.read_csv(io.BytesIO(raw)) if name.endswith(".csv") else pd.read_excel(io.BytesIO(raw))
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        q_col   = next((c for c in df.columns if "question" in c), None)
        pc_col  = next((c for c in df.columns if "pass" in c or "criteria" in c or "criterion" in c), None)
        dim_col = next((c for c in df.columns if "dimension" in c), None)
        qid_col = next((c for c in df.columns if c in ("q_id", "qid", "id")), None)

        if not q_col:
            return 0, "Could not find a 'Question' column in the uploaded file."

        now = datetime.now(timezone.utc).isoformat()
        conn = _db_conn()
        try:
            if replace:
                conn.execute(
                    "DELETE FROM rfx_questionnaire_definitions WHERE rfx_id=? AND vendor_type=?",
                    (rfx_id, vendor_type),
                )
            base_n = conn.execute(
                "SELECT COUNT(*) FROM rfx_questionnaire_definitions WHERE rfx_id=? AND vendor_type=?",
                (rfx_id, vendor_type),
            ).fetchone()[0]

            inserted = 0
            for _, row in df.iterrows():
                q_text = str(row.get(q_col, "")).strip()
                if not q_text or q_text.lower() in ("nan", "none", ""):
                    continue
                q_id = (
                    str(row[qid_col]).strip()
                    if qid_col and pd.notna(row.get(qid_col))
                    else f"Q{base_n + inserted + 1:02d}"
                )
                stage_col = next((c for c in df.columns if "stage" in c), None)
                conn.execute("""
                    INSERT OR REPLACE INTO rfx_questionnaire_definitions
                    (rfx_id, vendor_type, q_id, dimension, stage, question, pass_criteria, generated_at)
                    VALUES (?,?,?,?,?,?,?,?)
                """, (
                    rfx_id, vendor_type, q_id,
                    str(row[dim_col]).strip() if dim_col and pd.notna(row.get(dim_col)) else "user_provided",
                    str(row[stage_col]).strip() if stage_col and pd.notna(row.get(stage_col)) else "stage_1_screening",
                    q_text,
                    str(row[pc_col]).strip() if pc_col and pd.notna(row.get(pc_col)) else "",
                    now,
                ))
                inserted += 1
            conn.commit()
        finally:
            conn.close()
        return inserted, ""
    except Exception as exc:
        return 0, str(exc)


def _get_rfx_questionnaire(rfx_id: str, vendor_type: str) -> list[dict]:
    """Fetch questions from rfx_questionnaire_definitions."""
    conn = _db_conn()
    rows = conn.execute(
        "SELECT q_id, dimension, COALESCE(stage,'stage_1_screening') AS stage, question, pass_criteria, generated_at "
        "FROM rfx_questionnaire_definitions WHERE rfx_id=? AND vendor_type=? ORDER BY q_id",
        (rfx_id, vendor_type),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _get_questionnaire_vendor_types(rfx_id: str) -> list[str]:
    """Distinct vendor_types that have questionnaires generated for this rfx_id."""
    conn = _db_conn()
    rows = conn.execute(
        "SELECT DISTINCT vendor_type FROM rfx_questionnaire_definitions WHERE rfx_id=? ORDER BY vendor_type",
        (rfx_id,),
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


# ── Questionnaire answer extraction ──────────────────────────────────────────

_QUESTIONNAIRE_EXTRACTION_SYSTEM = """\
You are extracting a vendor's questionnaire responses from a vendor document.
Use semantic/fuzzy matching — the vendor may not have answered in Q&A format;
look for relevant statements throughout the entire document text.

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
For questions not addressed: passes=null, value_source="unknown", answer="(no response)".
Return {"responses": [...]} only.
"""


def _extract_vendor_questionnaire_answers(
    vendor_id: str,
    rfx_id: str,
    vendor_type: str,
    doc_text: str,
    source_file: str = "",
) -> tuple[int, str]:
    """
    Extract questionnaire answers from vendor document text using dynamic q_defs.
    Overwrites existing answers for vendor+rfx+vendor_type in questionnaire_responses.
    Returns (n_inserted, error_message).
    """
    q_defs = _get_rfx_questionnaire(rfx_id, vendor_type)
    if not q_defs:
        return 0, "No questionnaire generated for this RFx + vendor type yet."

    user_msg = (
        f"vendor_id: {vendor_id}\n"
        f"source_file: {source_file or 'uploaded document'}\n\n"
        f"Questionnaire definitions:\n{json.dumps(q_defs, indent=2)}\n\n"
        f"Vendor document text:\n{doc_text[:60000]}"
    )
    try:
        resp = openai_client.chat.completions.create(
            model=MODEL_HEAVY,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _QUESTIONNAIRE_EXTRACTION_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0,
        )
        raw = json.loads(resp.choices[0].message.content)
        responses = raw.get("responses", [])
        if not isinstance(responses, list):
            responses = next((v for v in raw.values() if isinstance(v, list)), [])
    except Exception as exc:
        return 0, str(exc)

    conn = _db_conn()
    try:
        conn.execute(
            "DELETE FROM questionnaire_responses WHERE vendor_id=? AND rfx_id=? AND vendor_type=?",
            (vendor_id, rfx_id, vendor_type),
        )
        for r in responses:
            conn.execute("""
                INSERT INTO questionnaire_responses
                (vendor_id, source_file, q_id, question, answer, answer_type,
                 passes, confidence, source_location, value_source, rfx_id, vendor_type)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                vendor_id, source_file or "",
                r.get("q_id"), r.get("question"),
                r.get("answer", "(no response)"),
                r.get("answer_type", "text"),
                r.get("passes"), r.get("confidence"),
                r.get("source_location"), r.get("value_source", "extracted"),
                rfx_id, vendor_type,
            ))
        conn.commit()
    finally:
        conn.close()
    return len(responses), ""


def _get_questionnaire_answers_df(rfx_id: str, vendor_type: str):
    """
    Returns (questions_df, responses_df) for the Questionnaire view.
    questions_df: q_id, dimension, stage, question, pass_criteria
    responses_df: vendor_id, vendor_name, q_id, answer, passes, confidence, value_source
    """
    conn = _db_conn()
    qs = pd.read_sql_query(
        "SELECT q_id, dimension, COALESCE(stage,'stage_1_screening') AS stage, question, pass_criteria "
        "FROM rfx_questionnaire_definitions WHERE rfx_id=? AND vendor_type=? ORDER BY q_id",
        conn, params=(rfx_id, vendor_type),
    )
    resp = pd.read_sql_query(
        "SELECT qr.vendor_id, COALESCE(vm.display_name, ve.vendor_name, qr.vendor_id) AS vendor_name, "
        "qr.q_id, qr.answer, qr.passes, qr.confidence, qr.value_source "
        "FROM questionnaire_responses qr "
        "LEFT JOIN vendor_metadata vm ON qr.vendor_id = vm.vendor_id "
        "LEFT JOIN (SELECT DISTINCT vendor_id, vendor_name FROM vendor_extractions) ve "
        "  ON qr.vendor_id = ve.vendor_id "
        "WHERE qr.rfx_id=? AND qr.vendor_type=?",
        conn, params=(rfx_id, vendor_type),
    )
    conn.close()
    return qs, resp


# ── Data Quality helper ───────────────────────────────────────────────────────

def _data_quality_stats(rfx_id: str | None = None):
    """Returns (df, total_rfx_lines) for the Data Quality tab."""
    conn = _db_conn()
    if rfx_id:
        total_lines = conn.execute(
            "SELECT COUNT(*) FROM rfx_lines WHERE rfx_id=?", (rfx_id,)
        ).fetchone()[0]
        rfx_clause = "AND ve.rfx_id=?"
        params: tuple = (rfx_id,)
    else:
        total_lines = conn.execute("SELECT COUNT(*) FROM rfx_lines").fetchone()[0]
        rfx_clause = ""
        params = ()
    sql = f"""
    SELECT
        ve.vendor_id,
        ve.vendor_name,
        SUM(CASE WHEN ve.value_source != 'unknown' THEN 1 ELSE 0 END) AS lines_quoted,
        SUM(CASE WHEN ve.value_source != 'unknown' AND ve.extraction_confidence >= 0.7
                 THEN 1 ELSE 0 END) AS ec_high_count,
        SUM(CASE WHEN ve.value_source != 'unknown' AND ve.extraction_confidence >= 0.4
                      AND ve.extraction_confidence < 0.7 THEN 1 ELSE 0 END) AS ec_med_count,
        SUM(CASE WHEN ve.value_source != 'unknown' AND ve.extraction_confidence < 0.4
                 THEN 1 ELSE 0 END) AS ec_low_count,
        SUM(CASE WHEN ve.value_source = 'unknown' THEN 1 ELSE 0 END) AS not_quoted_count,
        SUM(CASE WHEN ve.value_source != 'unknown' AND ve.extraction_confidence >= 0.7
                 THEN ve.normalized_unit_price * rl.quantity ELSE 0 END) AS val_high,
        SUM(CASE WHEN ve.value_source != 'unknown' AND ve.extraction_confidence >= 0.4
                      AND ve.extraction_confidence < 0.7
                 THEN ve.normalized_unit_price * rl.quantity ELSE 0 END) AS val_med,
        SUM(CASE WHEN ve.value_source != 'unknown' AND ve.extraction_confidence < 0.4
                 THEN ve.normalized_unit_price * rl.quantity ELSE 0 END) AS val_low,
        SUM(CASE WHEN ve.value_source != 'unknown'
                 THEN ve.normalized_unit_price * rl.quantity ELSE 0 END) AS val_total
    FROM vendor_extractions ve
    JOIN rfx_lines rl ON ve.line_id = rl.line_id
    WHERE (ve.superseded_by IS NULL OR ve.superseded_by = '')
      {rfx_clause}
    GROUP BY ve.vendor_id, ve.vendor_name
    ORDER BY ve.vendor_id
    """
    df = pd.read_sql_query(sql, conn, params=params if rfx_id else None)
    conn.close()
    # Collapse multiple name variants per vendor_id (keep longest name)
    if not df.empty and df['vendor_id'].duplicated().any():
        agg_cols = [c for c in df.columns if c not in ('vendor_id', 'vendor_name')]
        best_names = (
            df.groupby('vendor_id')['vendor_name']
            .apply(lambda s: max(s.fillna(''), key=len))
            .reset_index()
        )
        df = df.groupby('vendor_id')[agg_cols].sum().reset_index()
        df = df.merge(best_names, on='vendor_id', how='left')
    df["total_rfx_lines"] = total_lines
    df["coverage_pct"] = (df["lines_quoted"] / total_lines * 100).round(1)
    safe_total = df["val_total"].clip(lower=1)
    df["val_high_pct"] = (df["val_high"] / safe_total * 100).round(1)
    df["val_med_pct"]  = (df["val_med"]  / safe_total * 100).round(1)
    df["val_low_pct"]  = (df["val_low"]  / safe_total * 100).round(1)
    return df, total_lines


# ── Quote Comparison helpers ──────────────────────────────────────────────────

def _has_anomaly_flag(flags_str) -> bool:
    try:
        flags = json.loads(flags_str) if flags_str and flags_str not in ('[]', '', None) else []
        return any(str(f).startswith('anomaly_') for f in flags)
    except Exception:
        return False


def _parse_flags_text(flags_str) -> str:
    try:
        flags = json.loads(flags_str) if flags_str and flags_str not in ('[]', '', None) else []
        return ', '.join(flags) if flags else ''
    except Exception:
        return str(flags_str) if flags_str else ''


# ── Matrix view: position-only coloring (anomaly shown via cell suffix, not color overlay)
# Red is reserved exclusively for Flagged Items anomaly cells — nothing else uses red here.
_CSS_MIN_CLEAN   = 'background-color: #2e7d32; color: #ffffff;'           # cheapest
_CSS_MIN_ANOMALY = 'background-color: #2e7d32; color: #ffffff;'           # cheapest + flagged (color unchanged; ⚠/×? suffix marks the flag)
_CSS_MAX_CLEAN   = 'background-color: #4e342e; color: #ffffff;'           # priciest (dark warm brown — expensive, not alarming)
_CSS_MAX_ANOMALY = 'background-color: #4e342e; color: #ffffff;'           # priciest + flagged (color unchanged; suffix marks the flag)
_CSS_MID_ANOMALY = 'background-color: #37474f; color: #cfd8dc;'           # mid-vendor anomaly (dark slate — noted, not alarming)
# ── Shared across all views ───────────────────────────────────────────────────
_CSS_ANOMALY     = 'background-color: #b71c1c; color: #ffffff;'           # ONLY in Flagged Items view
_CSS_LOW_EC      = 'background-color: #424242; color: #cccccc; font-style: italic;'
_CSS_INFERRED    = 'background-color: #3a3a3a; color: #bbbbbb; font-style: italic;'
_CSS_NO_QUOTE    = 'color: #757575;'
_CSS_Q_PASS      = 'background-color: #2e7d32; color: #ffffff;'
_CSS_Q_FAIL      = 'background-color: #b71c1c; color: #ffffff;'
_CSS_Q_AMBIG     = 'background-color: #e65100; color: #ffffff;'
_CSS_Q_NORESP    = 'background-color: #424242; color: #ffffff;'


def _matrix_cell_css(
    price, value_source, ec,
    has_iqr: bool, has_decimal: bool,
    is_min: bool, is_max: bool,
    has_extreme_low: bool = False,
) -> str:
    """
    Three-layer CSS for the matrix view:
      Layer 1 — extraction quality (overrides everything if data is untrustworthy)
      Layer 2 — within-line position (cheapest / priciest) — the primary signal
      Layer 3 — anomaly overlay (changes the shade when position + flag coincide)
    extreme_low is treated as a strong anomaly signal regardless of position.
    """
    if value_source in (None, 'unknown') or pd.isna(price):
        return _CSS_NO_QUOTE
    if value_source in ('inferred', 'gpt_estimate'):
        return _CSS_INFERRED
    if ec is not None and ec < 0.7:
        return _CSS_LOW_EC              # don't trust the number → gray, no position color
    anomaly = has_iqr or has_decimal or has_extreme_low
    if is_min:
        return _CSS_MIN_ANOMALY if anomaly else _CSS_MIN_CLEAN
    if is_max:
        return _CSS_MAX_ANOMALY if anomaly else _CSS_MAX_CLEAN
    if anomaly:
        return _CSS_MID_ANOMALY
    return ''


def _cell_css(ec, value_source, has_anomaly: bool) -> str:
    """CSS for the Flagged Items view — anomaly-centric, no position logic."""
    if value_source in (None, 'unknown'):
        return _CSS_NO_QUOTE
    if value_source in ('inferred', 'gpt_estimate'):
        return _CSS_INFERRED
    if has_anomaly:
        return _CSS_ANOMALY
    if ec is not None and ec < 0.7:
        return _CSS_LOW_EC
    return ''


def _matrix_data(rfx_id: str | None = None):
    """
    Returns (display_df, style_df, raw_long_df, vendors_present, vname_map).

    Color scheme (three layers, applied in order of precedence):
      1. Extraction quality  — low EC / inferred override everything (gray)
      2. Within-line position — cheapest=green, priciest=red (primary signal)
      3. Anomaly overlay      — shifts shade when position + flag coincide
    High-value lines (top 75% of RFx ₹ value) are bold in the Item column.
    Decimal-mismatch cells show a '×?' suffix; IQR-outlier middle cells show '⚠'.
    """
    conn = _db_conn()
    rfx_clause_rl = "WHERE rfx_id=?" if rfx_id else ""
    rfx_clause_ve = "AND rfx_id=?" if rfx_id else ""
    rl_params = (rfx_id,) if rfx_id else None
    ve_params = (rfx_id,) if rfx_id else None
    lines_df = pd.read_sql_query(
        f"SELECT line_id, section, description, species_name, unit, quantity, boq_unit_rate "
        f"FROM rfx_lines {rfx_clause_rl} ORDER BY section, line_id",
        conn, params=rl_params,
    )
    ve_df = pd.read_sql_query(
        f"SELECT vendor_id, line_id, normalized_unit_price, "
        f"extraction_confidence, value_source, flags, source_snippet, source_location "
        f"FROM vendor_extractions "
        f"WHERE (superseded_by IS NULL OR superseded_by = '') {rfx_clause_ve} "
        f"ORDER BY vendor_id, line_id",
        conn, params=ve_params,
    )
    vendor_names = pd.read_sql_query(
        f"SELECT DISTINCT vendor_id, vendor_name FROM vendor_extractions "
        f"WHERE (superseded_by IS NULL OR superseded_by='') {rfx_clause_ve} ORDER BY vendor_id",
        conn, params=ve_params,
    )
    conn.close()

    def _has_flag_substr(flags_str, substr):
        try:
            flags = json.loads(flags_str) if flags_str and flags_str not in ('[]', '', None) else []
            return any(substr in str(f) for f in flags)
        except Exception:
            return False

    ve_df['has_iqr']         = ve_df['flags'].apply(lambda f: _has_flag_substr(f, 'iqr_outlier'))
    ve_df['has_decimal']     = ve_df['flags'].apply(lambda f: _has_flag_substr(f, 'decimal_mismatch'))
    ve_df['has_extreme_low'] = ve_df['flags'].apply(lambda f: _has_flag_substr(f, 'extreme_low'))

    # Per-line min / max across extracted quotes only (to set position benchmarks)
    extracted = ve_df[
        (ve_df['value_source'] == 'extracted') & ve_df['normalized_unit_price'].notna()
    ]
    line_stats = extracted.groupby('line_id').agg(
        line_min    = ('normalized_unit_price', 'min'),
        line_max    = ('normalized_unit_price', 'max'),
        n_extracted = ('normalized_unit_price', 'count'),
    ).reset_index()
    lines_df = lines_df.merge(line_stats, on='line_id', how='left')

    # ₹ impact = BOQ rate × BOQ quantity; high-value = top 75% of total RFx value
    lines_df['impact'] = lines_df['boq_unit_rate'].fillna(0) * lines_df['quantity'].fillna(0)
    total_val = lines_df['impact'].sum()
    cutoff    = total_val * 0.75
    cumsum    = 0.0
    hv_ids: set[str] = set()
    for _, r in lines_df.sort_values('impact', ascending=False).iterrows():
        hv_ids.add(r['line_id'])
        cumsum += r['impact']
        if cumsum >= cutoff:
            break

    sec_ord = {s: i for i, s in enumerate(_SECTIONS_ORDER)}
    lines_df['_ord'] = lines_df['section'].map(sec_ord).fillna(99)
    lines_df = lines_df.sort_values(['_ord', 'impact', 'line_id'],
                                    ascending=[True, False, True]).reset_index(drop=True)

    vendors_present = sorted(ve_df['vendor_id'].unique().tolist())
    vname_map = dict(zip(vendor_names['vendor_id'], vendor_names['vendor_name']))
    ve_idx = ve_df.set_index(['line_id', 'vendor_id'])

    display_rows, style_rows = [], []

    for _, line in lines_df.iterrows():
        lid        = line['line_id']
        boq        = line['boq_unit_rate']
        impact     = line['impact']
        is_hv      = lid in hv_ids
        line_min   = line.get('line_min')
        line_max   = line.get('line_max')
        n_ext      = int(line['n_extracted']) if pd.notna(line.get('n_extracted')) else 0
        has_spread = (
            n_ext >= 2 and
            pd.notna(line_min) and pd.notna(line_max) and
            line_max > line_min
        )

        d = {
            'Section':     line['section'],
            'Line':        lid,
            'Item':        line['description'] or '',
            'Species':     line['species_name'] or '',
            'Unit':        line['unit'] or '',
            'Qty':         (int(line['quantity']) if pd.notna(line['quantity']) and line['quantity'] == int(line['quantity']) else round(line['quantity'], 1)) if pd.notna(line['quantity']) else '—',
            'BOQ ₹/unit': f"₹{boq:,.0f}" if pd.notna(boq) else '—',
            '₹ Impact':   f"₹{impact:,.0f}" if impact > 0 else '—',
        }
        s = {col: '' for col in d}
        if is_hv:
            s['Item']      = 'font-weight: 900; font-size: 1.05em;'
            s['₹ Impact']  = 'font-weight: 900; font-size: 1.05em;'

        for vid in vendors_present:
            try:
                row = ve_idx.loc[(lid, vid)]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]

                vs           = row['value_source']
                ec           = row['extraction_confidence']
                has_iqr      = bool(row['has_iqr'])
                has_dec      = bool(row['has_decimal'])
                has_ext_low  = bool(row['has_extreme_low'])
                price        = row['normalized_unit_price']

                if vs == 'unknown' or pd.isna(price):
                    d[vid] = '—'
                    s[vid] = _CSS_NO_QUOTE
                else:
                    is_min = has_spread and round(price, 2) == round(line_min, 2)
                    is_max = has_spread and round(price, 2) == round(line_max, 2)

                    label = f"₹{price:,.0f}"
                    if has_ext_low:
                        label += " ↓!"     # extreme low — likely wrong grade/spec
                    elif has_dec:
                        label += " ×?"     # possible decimal/unit extraction error
                    elif has_iqr and not (is_min or is_max):
                        label += " ⚠"      # IQR outlier on a middle-position vendor

                    d[vid] = label
                    s[vid] = _matrix_cell_css(price, vs, ec, has_iqr, has_dec, is_min, is_max, has_ext_low)

            except KeyError:
                d[vid] = '—'
                s[vid] = _CSS_NO_QUOTE

        display_rows.append(d)
        style_rows.append(s)

    display_df = pd.DataFrame(display_rows)
    style_df   = pd.DataFrame(style_rows)

    raw_long = lines_df.drop(columns=['_ord'], errors='ignore').merge(
        ve_df.assign(flags_text=ve_df['flags'].apply(_parse_flags_text)),
        on='line_id', how='left',
    )
    return display_df, style_df, raw_long, vendors_present, vname_map


def _scorecard_data(rfx_id: str | None = None):
    """Returns (scorecard_df, n_common_lines)."""
    conn = _db_conn()
    rfx_clause_ve = "AND ve.rfx_id=?" if rfx_id else ""
    rfx_clause    = "AND rfx_id=?" if rfx_id else ""
    rfx_p         = (rfx_id,) if rfx_id else ()

    # Dynamic vendor list for this RFx
    active_vendors = [r[0] for r in conn.execute(
        f"SELECT DISTINCT vendor_id FROM vendor_extractions "
        f"WHERE (superseded_by IS NULL OR superseded_by='') {rfx_clause}",
        rfx_p,
    ).fetchall()]
    n_total_lines = conn.execute(
        f"SELECT COUNT(*) FROM rfx_lines {'WHERE rfx_id=?' if rfx_id else ''}",
        rfx_p if rfx_id else (),
    ).fetchone()[0]

    coverage = pd.read_sql_query(
        f"SELECT ve.vendor_id, ve.vendor_name, "
        f"SUM(CASE WHEN ve.value_source != 'unknown' THEN 1 ELSE 0 END) AS lines_quoted "
        f"FROM vendor_extractions ve "
        f"WHERE (ve.superseded_by IS NULL OR ve.superseded_by = '') {rfx_clause_ve} "
        f"GROUP BY ve.vendor_id, ve.vendor_name",
        conn, params=rfx_p if rfx_id else None,
    )

    # Common subset across ALL present vendors for this RFx
    if active_vendors:
        vlist_sql = "','".join(active_vendors)
        common = pd.read_sql_query(
            f"SELECT line_id FROM vendor_extractions "
            f"WHERE vendor_id IN ('{vlist_sql}') "
            f"  AND value_source != 'unknown' "
            f"  AND (superseded_by IS NULL OR superseded_by = '') {rfx_clause} "
            f"GROUP BY line_id "
            f"HAVING COUNT(DISTINCT vendor_id) = {len(active_vendors)}",
            conn, params=rfx_p if rfx_id else None,
        )
        n_common = len(common)

        if n_common > 0:
            line_params = common['line_id'].tolist()
            placeholders = ','.join('?' * n_common)
            totals = pd.read_sql_query(
                f"SELECT ve.vendor_id, "
                f"SUM(ve.normalized_unit_price * rl.quantity) AS common_total "
                f"FROM vendor_extractions ve "
                f"JOIN rfx_lines rl ON ve.line_id = rl.line_id "
                f"WHERE ve.vendor_id IN ('{vlist_sql}') "
                f"  AND ve.line_id IN ({placeholders}) "
                f"  AND ve.value_source != 'unknown' "
                f"  AND (ve.superseded_by IS NULL OR ve.superseded_by = '') {rfx_clause_ve} "
                f"GROUP BY ve.vendor_id",
                conn, params=line_params + list(rfx_p),
            )
        else:
            totals = pd.DataFrame({'vendor_id': active_vendors,
                                   'common_total': [None] * len(active_vendors)})
    else:
        common = pd.DataFrame({'line_id': []})
        n_common = 0
        totals = pd.DataFrame({'vendor_id': [], 'common_total': []})

    anomaly = pd.read_sql_query(
        f"SELECT vendor_id, COUNT(*) AS anomaly_count "
        f"FROM vendor_extractions "
        f"WHERE (superseded_by IS NULL OR superseded_by = '') {rfx_clause} "
        f"  AND flags LIKE '%anomaly_%' "
        f"GROUP BY vendor_id",
        conn, params=rfx_p if rfx_id else None,
    )

    # Use dynamic questionnaire responses for RFxes that have rfx_questionnaire_definitions rows
    has_dynamic_q = False
    if rfx_id:
        has_dynamic_q = conn.execute(
            "SELECT COUNT(*) FROM rfx_questionnaire_definitions WHERE rfx_id=?", (rfx_id,)
        ).fetchone()[0] > 0

    if has_dynamic_q and rfx_id:
        q = pd.read_sql_query("""
            SELECT vendor_id,
              SUM(CASE WHEN passes=1 THEN 1 ELSE 0 END)                                    AS q_pass,
              SUM(CASE WHEN passes=0 THEN 1 ELSE 0 END)                                    AS q_fail,
              SUM(CASE WHEN passes IS NULL AND value_source!='unknown' THEN 1 ELSE 0 END)  AS q_ambig,
              SUM(CASE WHEN value_source='unknown' THEN 1 ELSE 0 END)                      AS q_noresp
            FROM questionnaire_responses
            WHERE rfx_id=?
            GROUP BY vendor_id
        """, conn, params=(rfx_id,))
    else:
        q = pd.read_sql_query("""
            SELECT vendor_id,
              SUM(CASE WHEN passes=1 THEN 1 ELSE 0 END)                                    AS q_pass,
              SUM(CASE WHEN passes=0 THEN 1 ELSE 0 END)                                    AS q_fail,
              SUM(CASE WHEN passes IS NULL AND value_source!='unknown' THEN 1 ELSE 0 END)  AS q_ambig,
              SUM(CASE WHEN value_source='unknown' THEN 1 ELSE 0 END)                      AS q_noresp
            FROM questionnaire_responses
            WHERE rfx_id IS NULL OR rfx_id = ''
            GROUP BY vendor_id
        """, conn)
    conn.close()

    df = coverage.merge(totals, on='vendor_id', how='left')
    df = df.merge(anomaly, on='vendor_id', how='left')
    df = df.merge(q, on='vendor_id', how='left')
    df['anomaly_count'] = df['anomaly_count'].fillna(0).astype(int)
    df['coverage_pct']  = (df['lines_quoted'] / max(n_total_lines, 1) * 100).round(1)
    df['common_total_fmt'] = df['common_total'].apply(
        lambda x: f"₹{x:,.0f}" if pd.notna(x) else '—'
    )
    for col in ('q_pass', 'q_fail', 'q_ambig', 'q_noresp'):
        df[col] = df[col].fillna(0).astype(int)

    return df, n_common


def _flagged_data(rfx_id: str | None = None):
    """Returns flagged vendor-line rows (inferred, low-ec, or anomaly-flagged)."""
    conn = _db_conn()
    rfx_clause = "AND ve.rfx_id=?" if rfx_id else ""
    df = pd.read_sql_query(
        f"SELECT ve.vendor_id, ve.line_id, "
        f"       rl.section, rl.description, rl.species_name, rl.unit, "
        f"       ve.normalized_unit_price, ve.extraction_confidence, "
        f"       ve.value_source, ve.flags, "
        f"       ve.source_snippet, ve.source_location "
        f"FROM vendor_extractions ve "
        f"JOIN rfx_lines rl ON ve.line_id = rl.line_id "
        f"WHERE (ve.superseded_by IS NULL OR ve.superseded_by = '') {rfx_clause} "
        f"  AND ve.value_source != 'unknown' "
        f"  AND ( "
        f"    ve.value_source != 'extracted' "
        f"    OR ve.extraction_confidence < 0.7 "
        f"    OR ve.flags LIKE '%anomaly_%' "
        f"  ) "
        f"ORDER BY rl.section, rl.line_id, ve.vendor_id",
        conn, params=(rfx_id,) if rfx_id else None,
    )
    conn.close()

    df['flags_text'] = df['flags'].apply(_parse_flags_text)
    df['has_anomaly'] = df['flags_text'].str.contains('anomaly_', na=False)
    return df


# ── Streamlit UI ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Landscape RFx Analyst — SKYDOME KUKAS",
    page_icon="🌿",
    layout="wide"
)

# Pad content so the last chat message isn't hidden behind the sticky input bar
st.markdown("""
<style>
section.main > div.block-container {
    padding-bottom: 90px !important;
}
/* Wrap long RFx name in the sidebar selectbox */
[data-testid="stSidebar"] [data-baseweb="select"] [data-testid="stMarkdownContainer"],
[data-testid="stSidebar"] [data-baseweb="select"] div[class*="singleValue"] {
    white-space: normal !important;
    overflow: visible !important;
    text-overflow: unset !important;
}
/* Accent color: muted green for primary CTAs — red is reserved for anomaly flags only */
button[data-testid="baseButton-primary"],
[data-testid="stButton"] > button[kind="primary"] {
    background-color: #2e7d32 !important;
    border-color: #2e7d32 !important;
    color: #ffffff !important;
}
button[data-testid="baseButton-primary"]:hover,
[data-testid="stButton"] > button[kind="primary"]:hover {
    background-color: #1b5e20 !important;
    border-color: #1b5e20 !important;
}
/* Multiselect selected tags — green accent, not red */
[data-baseweb="tag"] {
    background-color: #2e7d32 !important;
}
[data-baseweb="tag"] span {
    color: #ffffff !important;
}
/* Step progress indicator — more legible */
.rfx-step-progress {
    font-size: 0.92rem;
    color: #a0a0a0;
    margin-bottom: 10px;
    letter-spacing: 0.01em;
}
.rfx-step-progress .active-step {
    color: #e0e0e0;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.title("🌿 Landscape RFx Analyst")

# Session state — initialise rfx_id before sidebar renders
if "rfx_id" not in st.session_state:
    conn = _db_conn()
    first = conn.execute(
        "SELECT rfx_id FROM rfx_projects ORDER BY created_at LIMIT 1"
    ).fetchone()
    conn.close()
    st.session_state.rfx_id = first["rfx_id"] if first else None

# Sidebar — RFx selector + data quality overview
with st.sidebar:
    # RFx selector
    conn = _db_conn()
    rfx_rows = conn.execute(
        "SELECT rfx_id, name FROM rfx_projects ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    rfx_options = {r["rfx_id"]: f"{r['rfx_id']} — {r['name']}" for r in rfx_rows}
    selected = st.selectbox(
        "Active RFx",
        options=list(rfx_options.keys()),
        format_func=lambda k: rfx_options[k],
        index=list(rfx_options.keys()).index(st.session_state.rfx_id)
              if st.session_state.rfx_id in rfx_options else 0,
        key="rfx_selector",
    )
    if selected != st.session_state.rfx_id:
        st.session_state.rfx_id = selected
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

    rfx_id = st.session_state.rfx_id

    st.divider()
    st.markdown("**Data quality**")
    try:
        conn = _db_conn()
        n_lines = conn.execute(
            "SELECT COUNT(*) FROM rfx_lines WHERE rfx_id=?", (rfx_id,)
        ).fetchone()[0]
        n_vendors = conn.execute(
            "SELECT COUNT(DISTINCT vendor_id) FROM vendor_extractions "
            "WHERE rfx_id=? AND (superseded_by IS NULL OR superseded_by='')",
            (rfx_id,),
        ).fetchone()[0]
        avg_coverage = conn.execute(
            """
            SELECT AVG(coverage_pct) FROM (
                SELECT vendor_id, COUNT(DISTINCT line_id) * 100.0 / ? AS coverage_pct
                FROM vendor_extractions
                WHERE rfx_id=? AND (superseded_by IS NULL OR superseded_by='') AND matched=1
                GROUP BY vendor_id
            )
            """,
            (n_lines if n_lines else 1, rfx_id),
        ).fetchone()[0]
        avg_conf = conn.execute(
            "SELECT AVG(extraction_confidence) FROM vendor_extractions "
            "WHERE rfx_id=? AND (superseded_by IS NULL OR superseded_by='') "
            "AND extraction_confidence IS NOT NULL",
            (rfx_id,),
        ).fetchone()[0]
        conn.close()

        cov_str = f"{avg_coverage:.0f}%" if avg_coverage is not None else "—"
        conf_str = f"{(avg_conf or 0) * 100:.0f}%" if avg_conf is not None else "—"
        st.markdown(f"""
<style>
.dq-grid {{display:grid;grid-template-columns:1fr 1fr;gap:4px 8px;margin:4px 0 0 0}}
.dq-cell {{background:rgba(255,255,255,0.05);border-radius:6px;padding:6px 8px;line-height:1.2}}
.dq-val {{font-size:1.3rem;font-weight:700;color:inherit}}
.dq-lbl {{font-size:0.68rem;color:#888;margin-top:1px}}
</style>
<div class="dq-grid">
  <div class="dq-cell"><div class="dq-val">{n_vendors}</div><div class="dq-lbl">Vendors</div></div>
  <div class="dq-cell"><div class="dq-val">{n_lines}</div><div class="dq-lbl">RFx lines</div></div>
  <div class="dq-cell"><div class="dq-val">{cov_str}</div><div class="dq-lbl">Line coverage (avg)</div></div>
  <div class="dq-cell"><div class="dq-val">{conf_str}</div><div class="dq-lbl">Confidence (avg)</div></div>
</div>
""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"DB error: {e}")

    st.divider()
    st.caption("Vendors")
    conn = _db_conn()
    # Pull all name variants per vendor_id; deduplicate in Python (pick longest name)
    _ve_rows = conn.execute(
        "SELECT DISTINCT vendor_id, vendor_name FROM vendor_extractions "
        "WHERE rfx_id=? AND (superseded_by IS NULL OR superseded_by='') ORDER BY vendor_id",
        (rfx_id,),
    ).fetchall()
    # Also pick up section-level-only vendors (e.g. Desert Bloom, Green Horizon)
    _vsq_rows = conn.execute(
        "SELECT DISTINCT vendor_id, vendor_name FROM vendor_section_quotes "
        "WHERE rfx_id=? AND (superseded_by IS NULL OR superseded_by='')",
        (rfx_id,),
    ).fetchall()
    conn.close()
    _vmap: dict[str, str] = {}
    for vid, vname in list(_ve_rows) + list(_vsq_rows):
        if vid not in _vmap or len(vname or "") > len(_vmap[vid]):
            _vmap[vid] = vname or vid
    for vid in sorted(_vmap):
        st.text(f"  {vid}: {_vmap[vid]}")

    st.divider()
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

    st.divider()
    with st.expander("🗑️ Manage / Delete", expanded=False):
        st.caption("Delete vendor data")
        _del_vendor_ids = sorted(_vmap.keys())
        if _del_vendor_ids:
            _del_vendor = st.selectbox(
                "Select vendor to delete",
                options=_del_vendor_ids,
                format_func=lambda v: f"{v}: {_vmap.get(v, v)}",
                key="del_vendor_select",
            )
            _del_v_confirm = st.checkbox(
                f"Confirm delete {_del_vendor} from {rfx_id}",
                key="del_vendor_confirm",
            )
            if st.button("Delete vendor", key="btn_del_vendor", disabled=not _del_v_confirm):
                _dc = _db_conn()
                _dc.execute(
                    "DELETE FROM vendor_extractions WHERE rfx_id=? AND vendor_id=?",
                    (rfx_id, _del_vendor),
                )
                _dc.execute(
                    "DELETE FROM vendor_section_quotes WHERE rfx_id=? AND vendor_id=?",
                    (rfx_id, _del_vendor),
                )
                _dc.execute(
                    "DELETE FROM questionnaire_responses WHERE rfx_id=? AND vendor_id=?",
                    (rfx_id, _del_vendor),
                )
                _dc.commit()
                _dc.close()
                st.success(f"Deleted {_del_vendor} from {rfx_id}")
                st.session_state.messages = []
                st.session_state.history = []
                st.rerun()
        else:
            st.caption("No vendors in this RFx.")

        st.markdown("---")
        st.caption("Delete entire RFx")
        _n_other_rfx = len(rfx_options) - 1
        if _n_other_rfx < 1:
            st.warning("Cannot delete — only one RFx exists.")
        else:
            _del_rfx_confirm = st.checkbox(
                f"Confirm delete entire RFx: {rfx_id}",
                key="del_rfx_confirm",
            )
            if st.button(
                "Delete RFx", key="btn_del_rfx",
                disabled=not _del_rfx_confirm,
                type="primary",
            ):
                _dc = _db_conn()
                for _tbl, _col in [
                    ("vendor_extractions", "rfx_id"),
                    ("vendor_section_quotes", "rfx_id"),
                    ("questionnaire_responses", "rfx_id"),
                    ("rfx_questionnaire_definitions", "rfx_id"),
                    ("rfx_lines", "rfx_id"),
                    ("rfx_projects", "rfx_id"),
                    ("rfx_drafts", "draft_id"),
                ]:
                    try:
                        _dc.execute(f"DELETE FROM {_tbl} WHERE {_col}=?", (rfx_id,))
                    except Exception:
                        pass
                _dc.commit()
                _dc.close()
                # Switch to first remaining RFx
                _remaining = [k for k in rfx_options if k != rfx_id]
                st.session_state.rfx_id = _remaining[0]
                st.session_state.messages = []
                st.session_state.history = []
                st.rerun()

STARTERS = [
    "What changed in Phoenix's revised quote compared to their original?",
    "Which vendor is cheapest for trees, excluding spec-grade-mismatched lines?",
    "Which vendors pass all questionnaire criteria?",
    "Where did the GPT BOQ draft diverge most from the client BOQ?",
    "Show me all lines where Jai Balaji was flagged low-confidence.",
]

# Ensure all app-managed tables exist (idempotent, runs once per server boot)
_ensure_drafts_table()
_ensure_app_tables()

# Session state — analyst
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []
if "pending_plan" not in st.session_state:
    st.session_state.pending_plan = None
if "web_search_session_count" not in st.session_state:
    st.session_state.web_search_session_count = 0

# Session state — co-pilot
if "copilot_messages" not in st.session_state:
    st.session_state.copilot_messages = []
if "copilot_history" not in st.session_state:
    st.session_state.copilot_history = []
if "copilot_staged_lines" not in st.session_state:
    st.session_state.copilot_staged_lines = []
if "copilot_finalized_rfx_id" not in st.session_state:
    st.session_state.copilot_finalized_rfx_id = None
if "copilot_file_name" not in st.session_state:
    st.session_state.copilot_file_name = None
if "copilot_file_injected" not in st.session_state:
    st.session_state.copilot_file_injected = False
if "copilot_review_mode" not in st.session_state:
    st.session_state.copilot_review_mode = False
if "copilot_boq_xlsx" not in st.session_state:
    st.session_state.copilot_boq_xlsx = None
if "copilot_boq_filename" not in st.session_state:
    st.session_state.copilot_boq_filename = None
# Wizard step state
if "copilot_step" not in st.session_state:
    st.session_state.copilot_step = 1
if "copilot_project_context" not in st.session_state:
    st.session_state.copilot_project_context = {}
if "copilot_vision_text" not in st.session_state:
    st.session_state.copilot_vision_text = ""
if "copilot_vision_summary" not in st.session_state:
    st.session_state.copilot_vision_summary = ""
if "copilot_rfx_name_draft" not in st.session_state:
    st.session_state.copilot_rfx_name_draft = ""

tab_chat, tab_cmp, tab_dq, tab_new = st.tabs(
    ["💬 Analyst Chat", "📋 Quote Comparison", "📊 Data Quality", "✏️ Create New RFx"]
)

# ── Tab 1: Analyst Chat ───────────────────────────────────────────────────────
with tab_chat:
    # Starter chips (only when chat is empty)
    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        cols = st.columns(len(STARTERS))
        for col, q in zip(cols, STARTERS):
            if col.button(q, key=f"starter_{q[:20]}"):
                st.session_state._pending_question = q

    # ── Render chat history ───────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            _content = msg["content"]
            _embedded = _DATA_URI_RE.findall(_content)
            if _embedded:
                # Strip base64 data: URIs from the text (they can't render inline)
                _clean = _DATA_URI_RE.sub("*(chart — see below)*", _content)
                st.markdown(_clean)
                import base64 as _b64lib
                for _m in _DATA_URI_RE.finditer(_content):
                    _b64 = _m.group(2).replace("\n", "").replace("\r", "").strip()
                    st.image(_b64lib.b64decode(_b64), use_container_width=True)
            else:
                st.markdown(_content)
            # Render any charts from tool log inline, above the expander
            if msg.get("tool_log"):
                for t in msg["tool_log"]:
                    if t.get("tool") in ("render_chart", "render_section_collage") and t.get("chart_b64") and not t.get("error"):
                        import base64
                        st.image(base64.b64decode(t["chart_b64"]), use_container_width=True)
            if msg.get("tool_log"):
                with st.expander(f"🔧 {len(msg['tool_log'])} tool call(s)", expanded=False):
                    for i, t in enumerate(msg["tool_log"], 1):
                        tool_label = t.get("tool", "run_query")
                        mdl = t.get("model", "")
                        icon = "📊" if tool_label in ("render_chart", "render_section_collage") else ("🔍" if tool_label in ("lookup_benchmark", "web_search") else "⚙️")
                        st.caption(f"**Call {i} [{tool_label}]** {icon} `{mdl}` — {t['rationale']}")
                        st.code(t["code"], language="python")
                        if t["error"]:
                            st.error(t["result"])
                        else:
                            st.text(t["result"][:800])

    # ── Plan confirmation — inline at the bottom of the conversation ─────────
    if st.session_state.pending_plan:
        _plan_data = st.session_state.pending_plan
        with st.container(border=True):
            _pc1, _pc2, _pc3 = st.columns([1, 4, 1])
            with _pc1:
                if st.button("✅ Confirm", key="plan_confirm_btn", use_container_width=True):
                    st.session_state._execute_plan = _plan_data
                    st.session_state.pending_plan = None
                    st.rerun()
            with _pc2:
                # Pre-fill with the agent's interpretation so the analyst
                # corrects the understanding, not re-types the raw query
                _mod_query = st.text_input(
                    "Correct my understanding:", value=_plan_data["plan_text"],
                    key="plan_modify_input", label_visibility="collapsed",
                )
            with _pc3:
                if st.button("📤 Send", key="plan_resubmit_btn", use_container_width=True):
                    st.session_state.pending_plan = None
                    st.session_state._pending_question = _mod_query
                    st.rerun()

    pending = st.session_state.pop("_pending_question", None)
    execute_plan = st.session_state.pop("_execute_plan", None)
    user_input = st.chat_input("Ask anything about the vendor comparison...") or pending

    if user_input or execute_plan:
        # Resolve inputs: executing a confirmed plan vs. fresh query
        if execute_plan:
            user_input = execute_plan["query"]
            pre_classification = execute_plan["classification"]
        else:
            pre_classification = None

        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        rfx_id = st.session_state.rfx_id

        # ── Stage 1: Classify ─────────────────────────────────────────────────
        if pre_classification is None:
            with st.spinner("Understanding query..."):
                classification = classify_query(user_input, rfx_id)
        else:
            classification = pre_classification

        intent = classification.get("intent", "unknown")
        entities = classification.get("entities", {})

        # ── Stage 2: Handle off_topic immediately (no tool loop) ──────────────
        if intent == "off_topic":
            answer = handle_off_topic(classification, user_input, rfx_id)
            _log_trace({
                "raw_query": user_input,
                "classified_intent": "off_topic",
                "sub_type": classification.get("off_topic_sub_type"),
                "entities": entities,
                "plan_shown": False,
                "analyst_confirmed": False,
                "tool_calls": [],
                "zero_result_triggered": False,
                "web_searches_used": 0,
            })
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer, "tool_log": []})
            # Off-topic: do not extend history (keeps context clean)

        else:
            # ── Stage 3: Plan confirmation ────────────────────────────────────
            needs_confirm = (
                pre_classification is None  # already confirmed if execute_plan was set
                and classification.get("needs_confirmation", False)
            )

            if needs_confirm:
                plan_text = generate_plan_text(classification, rfx_id)
                st.session_state.pending_plan = {
                    "classification": classification,
                    "plan_text": plan_text,
                    "query": user_input,
                }
                plan_msg = f"📋 **Here's what I understood** — confirm below or correct me:\n\n{plan_text}"
                with st.chat_message("assistant"):
                    st.markdown(plan_msg)
                st.session_state.messages.append({
                    "role": "assistant", "content": plan_msg, "tool_log": []
                })
                st.rerun()

            # ── Stage 4: Route + Execute ──────────────────────────────────────
            plan_shown = execute_plan is not None

            with st.chat_message("assistant"):
                with st.spinner("Querying..."):
                    if intent == "negotiation_analysis":
                        answer, tool_log = execute_negotiation_analysis(entities, rfx_id)
                        new_history = st.session_state.history + [
                            {"role": "user", "content": user_input},
                            {"role": "assistant", "content": answer},
                        ]

                    elif intent == "knowledge_botanical":
                        answer, tool_log = execute_knowledge_botanical(
                            user_input, entities, rfx_id
                        )
                        new_history = st.session_state.history + [
                            {"role": "user", "content": user_input},
                            {"role": "assistant", "content": answer},
                        ]

                    elif intent == "knowledge_availability":
                        answer, tool_log = execute_knowledge_availability(
                            user_input, entities, rfx_id
                        )
                        new_history = st.session_state.history + [
                            {"role": "user", "content": user_input},
                            {"role": "assistant", "content": answer},
                        ]

                    elif intent == "spec_filter":
                        answer, new_history, tool_log = run_agent(
                            user_input, st.session_state.history,
                            rfx_id=rfx_id, intent_hint="spec_filter",
                        )

                    elif intent == "rank_aggregate":
                        answer, new_history, tool_log = run_agent(
                            user_input, st.session_state.history,
                            rfx_id=rfx_id, intent_hint="rank_aggregate",
                        )

                    elif intent == "visualization":
                        answer, new_history, tool_log = run_agent(
                            user_input, st.session_state.history,
                            rfx_id=rfx_id, intent_hint="visualization",
                        )

                    else:
                        # name_lookup, hybrid, unknown → full tool-loop autonomy
                        answer, new_history, tool_log = run_agent(
                            user_input, st.session_state.history, rfx_id=rfx_id
                        )

                _ans_embedded = _DATA_URI_RE.findall(answer)
                if _ans_embedded:
                    _ans_clean = _DATA_URI_RE.sub("*(chart — see below)*", answer)
                    st.markdown(_ans_clean)
                    import base64 as _b64lib2
                    for _m in _DATA_URI_RE.finditer(answer):
                        _b64 = _m.group(2).replace("\n", "").replace("\r", "").strip()
                        st.image(_b64lib2.b64decode(_b64), use_container_width=True)
                else:
                    st.markdown(answer)
                # Render charts inline before the tool expander
                if tool_log:
                    for t in tool_log:
                        if t.get("tool") in ("render_chart", "render_section_collage") and t.get("chart_b64") and not t.get("error"):
                            import base64
                            st.image(base64.b64decode(t["chart_b64"]), use_container_width=True)
                if tool_log:
                    with st.expander(f"🔧 {len(tool_log)} tool call(s)", expanded=False):
                        for i, t in enumerate(tool_log, 1):
                            tool_label = t.get("tool", "run_query")
                            mdl = t.get("model", "")
                            icon = "📊" if tool_label in ("render_chart", "render_section_collage") else ("🔍" if tool_label in ("lookup_benchmark", "web_search", "llm_knowledge") else "⚙️")
                            st.caption(
                                f"**Call {i} [{tool_label}]** {icon} `{mdl}` — {t['rationale']}"
                            )
                            st.code(t["code"], language="python")
                            if t["error"]:
                                st.error(t["result"])
                            else:
                                st.text(t["result"][:800])

            _log_trace({
                "raw_query": user_input,
                "classified_intent": intent,
                "entities": entities,
                "plan_shown": plan_shown,
                "analyst_confirmed": plan_shown,
                "tool_calls": [{"tool": t["tool"]} for t in tool_log],
                "zero_result_triggered": False,
                "web_searches_used": st.session_state.web_search_session_count,
            })

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "tool_log": tool_log,
            })
            st.session_state.history = new_history

# ── Tab 2: Quote Comparison ───────────────────────────────────────────────────
with tab_cmp:
    _LEGEND_MATRIX = (
        "<span style='background:#2e7d32;color:#fff;padding:2px 6px;border-radius:3px'>↓ cheapest</span>&nbsp;"
        "<span style='background:#4e342e;color:#fff;padding:2px 6px;border-radius:3px'>↑ priciest</span>&nbsp;"
        "<span style='background:#37474f;color:#cfd8dc;padding:2px 6px;border-radius:3px'>⚠ mid anomaly</span>&nbsp;"
        "<span style='background:#424242;color:#ccc;font-style:italic;padding:2px 6px;border-radius:3px'>Low confidence</span>&nbsp;"
        "<span style='background:#3a3a3a;color:#bbb;font-style:italic;padding:2px 6px;border-radius:3px'>Inferred</span>&nbsp;"
        "<span style='color:#757575'>— not quoted</span>&nbsp;"
        "&nbsp;&nbsp;<b>bold</b> = top 75% RFx value &nbsp;·&nbsp; ↓! = extreme low (likely wrong grade) &nbsp;·&nbsp; ×? = decimal error &nbsp;·&nbsp; ⚠ = IQR outlier"
    )
    _LEGEND_FLAGS = (
        "<span style='background:#b71c1c;color:#fff;padding:2px 6px;border-radius:3px'>🚩 anomaly</span>&nbsp;"
        "<span style='background:#424242;color:#ccc;font-style:italic;padding:2px 6px;border-radius:3px'>Low confidence</span>&nbsp;"
        "<span style='background:#3a3a3a;color:#bbb;font-style:italic;padding:2px 6px;border-radius:3px'>Inferred</span>&nbsp;"
        "<span style='color:#757575'>— not quoted</span>"
    )

    # ── RFx Overview (vision summary) ─────────────────────────────────────────
    _ov_rfx_id = st.session_state.get("rfx_id")
    if _ov_rfx_id:
        _ov_conn = _db_conn()
        _ov_row = _ov_conn.execute(
            "SELECT name, vision_summary, terms FROM rfx_projects WHERE rfx_id=?",
            (_ov_rfx_id,),
        ).fetchone()
        _ov_conn.close()
        if _ov_row and _ov_row["vision_summary"]:
            with st.expander("📋 RFx Overview — buyer vision & scope", expanded=False):
                st.markdown(_ov_row["vision_summary"])
                if _ov_row["terms"]:
                    try:
                        _ov_terms = json.loads(_ov_row["terms"])
                        _cols = st.columns(3)
                        _cols[0].metric("Maintenance", _ov_terms.get("maintenance_duration", "—"))
                        _cols[1].metric("Payment", _ov_terms.get("payment_schedule", "—")[:30] + "…"
                                         if len(_ov_terms.get("payment_schedule", "")) > 30
                                         else _ov_terms.get("payment_schedule", "—"))
                        _cols[2].metric("Quote validity", _ov_terms.get("quote_validity", "—"))
                    except Exception:
                        pass

    # ── Vendor Setup panel ────────────────────────────────────────────────────
    with st.expander("⚙️ Vendor Setup — add & manage vendors", expanded=False):
        _vs_rfx_id = st.session_state.get("rfx_id")
        _vtab_exist, _vtab_add = st.tabs(["Existing vendors", "Add vendor"])

        with _vtab_exist:
            st.caption(
                "Tag vendor type (used by the questionnaire engine). "
                "Contact fields are stored for reference. "
                "Use **Extract quotes** to process files dropped into a vendor's inbound folder."
            )
            _all_vendors = _get_all_vendors(_vs_rfx_id)
            if not _all_vendors:
                st.info("No vendors yet — add one in the 'Add vendor' tab.")
            else:
                for _v in _all_vendors:
                    _vtype_options = _VENDOR_TYPE_SUGGESTIONS.copy()
                    _current = _v["vendor_type"]
                    if _current not in ("untagged", "") and _current not in _vtype_options:
                        _vtype_options.insert(0, _current)
                    _status_badge = "" if _v.get("has_extractions") else " ⬜ *pending extraction*"

                    with st.expander(
                        f"**{_v['vendor_name'] or _v['vendor_id']}** "
                        f"`{_v['vendor_id']}`{_status_badge}",
                        expanded=False,
                    ):
                        _vc1, _vc2 = st.columns([3, 1])
                        with _vc1:
                            _new_type = st.selectbox(
                                "Vendor type",
                                options=_vtype_options,
                                index=_vtype_options.index(_current)
                                      if _current in _vtype_options else len(_vtype_options) - 1,
                                key=f"vtype_{_v['vendor_id']}",
                            )
                            _new_email   = st.text_input("Email",   value=_v.get("email", ""),
                                                         key=f"vemail_{_v['vendor_id']}")
                            _new_poc     = st.text_input("Point of contact", value=_v.get("poc_name", ""),
                                                         key=f"vpoc_{_v['vendor_id']}")
                            _new_phone   = st.text_input("Phone",   value=_v.get("poc_phone", ""),
                                                         key=f"vphone_{_v['vendor_id']}")
                            _new_website = st.text_input("Website", value=_v.get("website", ""),
                                                         key=f"vweb_{_v['vendor_id']}")
                        with _vc2:
                            if st.button("Save", key=f"vsave_{_v['vendor_id']}"):
                                _upsert_vendor_contact(
                                    _v["vendor_id"],
                                    rfx_id=_vs_rfx_id or "",
                                    display_name=_v["vendor_name"] or "",
                                    vendor_type=_new_type,
                                    email=_new_email,
                                    website=_new_website,
                                    poc_name=_new_poc,
                                    poc_phone=_new_phone,
                                    inbound_folder=_v.get("inbound_folder", ""),
                                )
                                st.success("Saved")
                                st.rerun()

                            _inbound = _v.get("inbound_folder", "")
                            if _inbound and os.path.isdir(_inbound):
                                n_files = len([
                                    f for f in os.listdir(_inbound) if os.path.isfile(os.path.join(_inbound, f))
                                ])
                                st.caption(f"📁 {n_files} file(s) in inbound folder")
                                if st.button("Extract quotes", key=f"vextract_{_v['vendor_id']}",
                                             type="primary"):
                                    with st.spinner(f"Extracting {_v['vendor_name']}…"):
                                        _n, _errs = _run_inbound_extraction(
                                            _vs_rfx_id or "", _v["vendor_id"],
                                            _v["vendor_name"] or _v["vendor_id"], _inbound,
                                        )
                                    if _errs:
                                        st.error("\n".join(_errs[:3]))
                                    else:
                                        st.success(f"Extracted {_n} file(s). Refresh to see in matrix.")
                                        st.rerun()
                            elif _vs_rfx_id and not _v.get("has_extractions"):
                                st.caption("No inbound folder — save metadata first.")

        with _vtab_add:
            if not _vs_rfx_id:
                st.info("Select an RFx in the sidebar first.")
            else:
                st.caption(
                    "Creates the vendor's inbound folder under "
                    f"`rfx_workspace/{_vs_rfx_id}/inbound/<name>/`. "
                    "Drop quote files there, then use **Extract quotes** above."
                )
                with st.form("add_vendor_form", clear_on_submit=True):
                    _av_name     = st.text_input("Vendor name *", placeholder="e.g. Green Valley Nurseries")
                    _av_type     = st.selectbox("Vendor type", _VENDOR_TYPE_SUGGESTIONS)
                    _av_email    = st.text_input("Email", placeholder="vendor@example.com")
                    _av_poc      = st.text_input("Point of contact", placeholder="Name of key contact")
                    _av_phone    = st.text_input("Phone", placeholder="+91 98765 43210")
                    _av_website  = st.text_input("Website", placeholder="https://…")
                    _av_submit   = st.form_submit_button("Add vendor & create folders", type="primary")

                if _av_submit:
                    if not _av_name.strip():
                        st.error("Vendor name is required.")
                    else:
                        _av_vendor_id = _gen_vendor_id(_av_name.strip(), _vs_rfx_id)
                        _av_slug = re.sub(r"[^a-z0-9_]", "_", _av_name.lower().strip())[:40]
                        try:
                            _, _av_inbound = _create_vendor_folders(_vs_rfx_id, _av_slug)
                            _upsert_vendor_contact(
                                _av_vendor_id,
                                rfx_id=_vs_rfx_id,
                                display_name=_av_name.strip(),
                                vendor_type=_av_type,
                                email=_av_email.strip(),
                                website=_av_website.strip(),
                                poc_name=_av_poc.strip(),
                                poc_phone=_av_phone.strip(),
                                inbound_folder=_av_inbound,
                            )
                            st.success(
                                f"✅ Vendor **{_av_name.strip()}** added as `{_av_vendor_id}`  \n"
                                f"Inbound folder: `{_av_inbound}`"
                            )
                            st.rerun()
                        except Exception as _e:
                            st.error(f"Failed to create vendor: {_e}")

    view = st.radio(
        "view_selector",
        ["📊 Matrix", "⚠️ Flagged Items", "📝 Questionnaire"],
        horizontal=True,
        label_visibility="collapsed",
        key="cmp_view",
    )
    st.divider()

    # ── Matrix view ───────────────────────────────────────────────────────────
    if view == "📊 Matrix":
        st.markdown(_LEGEND_MATRIX, unsafe_allow_html=True)
        st.caption(
            "Color = position within that line (cheapest → green, priciest → red). "
            "Shade shifts when an anomaly flag also applies. "
            "**Bold rows** contribute to the top 75% of total RFx value — prioritise those. "
            "BOQ ₹/unit is the client reference rate; ₹ Impact = BOQ rate × BOQ qty."
        )
        try:
            display_df, style_df, raw_long, vendors_present, vname_map = _matrix_data(st.session_state.rfx_id)

            # Section filter (used for CSV export; section expanders below replace the flat view)
            sections = display_df['Section'].unique().tolist()
            with st.expander("Filter sections", expanded=True):
                sel_sections = st.multiselect(
                    "Show sections", sections, default=sections, key="matrix_section_filter",
                    label_visibility="collapsed",
                )
            if not sel_sections:
                sel_sections = sections
            mask = display_df['Section'].isin(sel_sections)
            display_filt = display_df[mask].reset_index(drop=True)
            style_filt   = style_df[mask].reset_index(drop=True)

            # ── Pre-compute vendor section totals from raw_long ────────────────
            # Used to populate the mini-table inside each section expander.
            # Only include lines where vendor price >= 70% of BOQ rate (exclude implausible anomalies).
            _BOQ_FLOOR = 0.70
            _raw_ext = raw_long[
                (raw_long['value_source'] == 'extracted') &
                raw_long['normalized_unit_price'].notna()
            ].copy()
            _raw_ext['_line_total'] = (
                _raw_ext['normalized_unit_price'] * _raw_ext['quantity'].fillna(0)
            )
            # Tag lines as valid for section-total (price >= 70% of BOQ rate, or BOQ rate missing)
            _raw_ext['_valid_for_total'] = (
                _raw_ext['boq_unit_rate'].isna() |
                (_raw_ext['boq_unit_rate'] <= 0) |
                (_raw_ext['normalized_unit_price'] >= _BOQ_FLOOR * _raw_ext['boq_unit_rate'])
            )
            # Count lines quoted per vendor-section (denominator)
            _quoted_counts = (
                _raw_ext.groupby(['section', 'vendor_id'])['line_id']
                .nunique()
                .reset_index()
                .rename(columns={'line_id': '_lines_quoted'})
            )
            # Sum only valid lines (numerator for total and line count)
            _raw_valid = _raw_ext[_raw_ext['_valid_for_total']]
            _valid_counts = (
                _raw_valid.groupby(['section', 'vendor_id'])['line_id']
                .nunique()
                .reset_index()
                .rename(columns={'line_id': '_lines_valid'})
            )
            _valid_totals = (
                _raw_valid.groupby(['section', 'vendor_id'])['_line_total']
                .sum()
                .reset_index()
                .rename(columns={'_line_total': 'section_total'})
            )
            _all_vendor_sec_totals = (
                _valid_totals
                .merge(_quoted_counts, on=['section', 'vendor_id'], how='left')
                .merge(_valid_counts,  on=['section', 'vendor_id'], how='left')
                .fillna({'_lines_quoted': 0, '_lines_valid': 0})
            )
            # BOQ section totals (one row per section, deduplicated on line_id)
            _boq_sec_totals = (
                raw_long.drop_duplicates(subset=['line_id'])
                .assign(_boq_line=lambda df: df['boq_unit_rate'].fillna(0) * df['quantity'].fillna(0))
                .groupby('section')['_boq_line']
                .sum()
                .reset_index()
                .rename(columns={'_boq_line': 'boq_total'})
            )
            _boq_sec_map = dict(zip(_boq_sec_totals['section'], _boq_sec_totals['boq_total']))

            # ── Section-collapsed view ─────────────────────────────────────────
            # Each section expander: header shows line count + flag count + quote range.
            # Inside: mini vendor-total table, then full line-item matrix.
            def _make_sec_style(sty):
                def _fn(df):
                    return sty
                return _fn

            def _fmt_lakh(v):
                if v is None or v != v:
                    return "—"
                if v >= 1e5:
                    return f"₹{v/1e5:.1f}L".rstrip('0').rstrip('.')
                return f"₹{v:,.0f}"

            for _sec in sel_sections:
                _sec_mask   = display_filt['Section'] == _sec
                _sec_disp   = display_filt[_sec_mask].reset_index(drop=True)
                _sec_sty    = style_filt[_sec_mask].reset_index(drop=True)
                _n          = len(_sec_disp)
                _sec_label  = _sec.replace('_', ' ').title()

                # Vendor quote range for this section (filtered totals only)
                _sec_v_totals = _all_vendor_sec_totals[_all_vendor_sec_totals['section'] == _sec]
                _boq_sec_val  = _boq_sec_map.get(_sec, 0)
                if not _sec_v_totals.empty and _sec_v_totals['section_total'].sum() > 0:
                    _range_min = _sec_v_totals['section_total'].min()
                    _range_max = _sec_v_totals['section_total'].max()
                    _range_hint = f" · {_fmt_lakh(_range_min)}–{_fmt_lakh(_range_max)}"
                else:
                    _range_hint = ""

                with st.expander(
                    f"**{_sec_label}** — {_n} lines{_range_hint}",
                    expanded=False,
                ):
                    # ── Mini vendor section-total table ────────────────────────
                    if not _sec_v_totals.empty:
                        _mini_rows = []
                        for _, _vr in _sec_v_totals.sort_values('section_total').iterrows():
                            _vname = vname_map.get(_vr['vendor_id'], _vr['vendor_id'])
                            _vtotal = _vr['section_total']
                            _valid  = int(_vr.get('_lines_valid', 0))
                            _quoted = int(_vr.get('_lines_quoted', 0))
                            _lines_str = f"{_valid} / {_quoted}" if _quoted else "—"
                            _mini_rows.append({
                                "Vendor":        _vname,
                                "Section Quote": f"₹{_vtotal:,.0f}",
                                "Lines":         _lines_str,
                            })
                        _mini_df = pd.DataFrame(_mini_rows)
                        st.dataframe(_mini_df, use_container_width=True, hide_index=True)
                        st.divider()

                    # ── Line items ─────────────────────────────────────────────
                    _styled_sec = _sec_disp.style.apply(_make_sec_style(_sec_sty), axis=None)
                    st.dataframe(_styled_sec, use_container_width=True, hide_index=True)

            # CSV export: long-form with all metadata
            csv_raw = raw_long[raw_long['section'].isin(sel_sections)].copy()
            csv_raw = csv_raw.drop(columns=['flags'], errors='ignore')
            csv_bytes = csv_raw.to_csv(index=False).encode('utf-8')
            st.download_button(
                "⬇ Download Matrix CSV (long-form, all metadata)",
                csv_bytes,
                file_name="quote_matrix.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Matrix load failed: {e}")

        # ── Section rollup (section-total vendors alongside summed line-item vendors)
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "db"))
            from query_helpers import get_section_totals as _get_section_totals
            sec_df = _get_section_totals(st.session_state.rfx_id)
            if not sec_df.empty:
                st.divider()
                st.markdown("**Section-level comparison** — includes vendors who quoted section totals (not itemized)")
                st.caption(
                    "Vendors who quoted at section level are shown with their stated totals. "
                    "Line-item vendors are summed to the same section grouping. "
                    "Direct = vendor's own section total; Summed = computed from line-item quotes."
                )
                pivot_sec = sec_df.pivot_table(
                    index="section",
                    columns="vendor_id",
                    values="total_amount",
                    aggfunc="sum",
                ).reset_index()
                source_info = sec_df.groupby(["vendor_id", "section"])["source"].first().reset_index()
                vs_info = sec_df.groupby(["vendor_id", "section"])["value_source"].first().reset_index()

                def _fmt_section_amt(val):
                    if val is None or (hasattr(val, '__class__') and val.__class__.__name__ == 'float' and val != val):
                        return "—"
                    try:
                        return f"₹{val:,.0f}"
                    except (TypeError, ValueError):
                        return "—"

                # Apply formatting
                for col in pivot_sec.columns:
                    if col != "section":
                        pivot_sec[col] = pivot_sec[col].apply(_fmt_section_amt)

                # Add a note column for section-total vendors
                section_total_vendors = set(
                    sec_df[sec_df["source"] == "direct"]["vendor_id"].unique()
                )
                if section_total_vendors:
                    st.info(
                        f"**Section-total vendors** (not itemized per line): "
                        f"{', '.join(sorted(section_total_vendors))}. "
                        f"Per-line comparison is not available for these vendors. "
                        f"The matrix above shows only line-item vendors."
                    )
                package_rows = sec_df[sec_df["section"].str.startswith("package_") |
                                       sec_df["section"].str.endswith("_balance")]
                if not package_rows.empty:
                    st.warning(
                        "Some vendors quoted multi-section packages (not split by section). "
                        "These appear as 'package_non_tree' or 'trees_balance' in the table below."
                    )
                st.dataframe(pivot_sec, use_container_width=True, hide_index=True)
        except Exception as _e:
            pass  # section rollup is informational; don't block the main matrix

    # ── Flagged Items ─────────────────────────────────────────────────────────
    elif view == "⚠️ Flagged Items":
        st.markdown(_LEGEND_FLAGS, unsafe_allow_html=True)
        try:
            fl = _flagged_data(st.session_state.rfx_id)
            st.caption(
                f"**{len(fl)} vendor-line rows** where value_source ≠ 'extracted', "
                "extraction_confidence < 0.7, or anomaly flag is set. "
                "Includes source_snippet and source_location for manual verification."
            )

            # Filters
            col1, col2 = st.columns(2)
            with col1:
                fl_vendors = st.multiselect(
                    "Vendor", fl['vendor_id'].unique().tolist(),
                    default=fl['vendor_id'].unique().tolist(), key="fl_vendor"
                )
            with col2:
                fl_reasons = st.multiselect(
                    "Flag type",
                    ["inferred/estimated", "low confidence (ec<0.7)", "anomaly"],
                    default=["inferred/estimated", "low confidence (ec<0.7)", "anomaly"],
                    key="fl_reason",
                )

            mask = fl['vendor_id'].isin(fl_vendors)
            if "inferred/estimated" not in fl_reasons:
                mask &= fl['value_source'] == 'extracted'
            if "low confidence (ec<0.7)" not in fl_reasons:
                mask &= ~((fl['value_source'] == 'extracted') & (fl['extraction_confidence'] < 0.7))
            if "anomaly" not in fl_reasons:
                mask &= ~fl['has_anomaly']

            fl_filt = fl[mask].reset_index(drop=True)

            display_fl = fl_filt[[
                'vendor_id', 'line_id', 'section', 'description', 'species_name',
                'unit', 'normalized_unit_price', 'extraction_confidence',
                'value_source', 'flags_text', 'source_snippet', 'source_location',
            ]].copy()
            display_fl.columns = [
                'Vendor', 'Line', 'Section', 'Item', 'Species',
                'Unit', 'Price (₹)', 'EC', 'Value Source', 'Flags',
                'Source Snippet', 'Source Location',
            ]
            display_fl['Price (₹)'] = display_fl['Price (₹)'].apply(
                lambda x: f"₹{x:,.0f}" if pd.notna(x) else '—'
            )
            display_fl['EC'] = display_fl['EC'].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else '—'
            )

            def _flag_style(df):
                styles = pd.DataFrame('', index=df.index, columns=df.columns)
                for i, row in fl_filt.iterrows():
                    css = _cell_css(
                        row['extraction_confidence'],
                        row['value_source'],
                        bool(row['has_anomaly']),
                    )
                    if css:
                        styles.loc[i, :] = css
                return styles

            styled_fl = display_fl.style.apply(_flag_style, axis=None)
            st.dataframe(styled_fl, use_container_width=True, hide_index=True)

            csv_fl = fl_filt.drop(columns=['flags', 'has_anomaly'], errors='ignore')
            st.download_button(
                "⬇ Download Flagged Items CSV",
                csv_fl.to_csv(index=False).encode('utf-8'),
                file_name="flagged_items.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Flagged items load failed: {e}")

    # ── Questionnaire view ────────────────────────────────────────────────────
    elif view == "📝 Questionnaire":
        _rfx_id = st.session_state.get("rfx_id")

        # Vendor type selector
        _tagged_vendors = _get_all_vendors(_rfx_id)
        _type_options_present = sorted({
            v["vendor_type"] for v in _tagged_vendors
            if v["vendor_type"] not in ("untagged", "Other", "")
        })
        _existing_q_types = _get_questionnaire_vendor_types(_rfx_id) if _rfx_id else []
        _all_q_types = sorted(set(_type_options_present + _existing_q_types))

        if not _all_q_types:
            st.info(
                "No vendor types tagged yet. Use **Vendor Setup** above to tag your vendors, "
                "then come back here to generate their questionnaire."
            )
        else:
            _sel_vtype = st.selectbox(
                "Vendor type to configure questionnaire for:",
                options=_all_q_types,
                key="q_vendor_type_sel",
            )

            _existing_qs = _get_rfx_questionnaire(_rfx_id, _sel_vtype) if _rfx_id else []
            _n_existing = len(_existing_qs)

            # ── Generate / Upload / Augment — one consistent decision block ────
            st.markdown("#### Build questionnaire")
            st.markdown(
                "<div style='border:1px solid rgba(255,255,255,0.1);border-radius:8px;"
                "padding:16px 20px 12px;margin-bottom:12px'>",
                unsafe_allow_html=True,
            )
            _gcol1, _gcol_div, _gcol2 = st.columns([5, 1, 5])

            with _gcol1:
                st.markdown("**Option A — Generate**")
                st.caption("Auto-build from BOQ + project context")
                if _rfx_id:
                    _gen_label = (
                        f"♻ Regenerate ({_n_existing} existing questions will be replaced)"
                        if _n_existing else
                        f"✨ Generate questionnaire for *{_sel_vtype}*"
                    )
                    if st.button(_gen_label, key="q_gen_btn", type="primary"):
                        with st.spinner(f"Generating questionnaire for {_sel_vtype}…"):
                            try:
                                _new_qs = _generate_rfx_questionnaire(_rfx_id, _sel_vtype)
                                st.success(f"Generated {len(_new_qs)} questions.")
                                st.rerun()
                            except Exception as _e:
                                st.error(f"Generation failed: {_e}")
                else:
                    st.caption("Select an RFx in the sidebar first.")

            with _gcol_div:
                st.markdown(
                    "<div style='border-left:1px solid rgba(255,255,255,0.12);"
                    "height:100%;margin:0 auto'></div>",
                    unsafe_allow_html=True,
                )

            with _gcol2:
                st.markdown("**Option B — Upload**")
                st.caption("Bring your own questionnaire (xlsx / csv)")
                _upload_replace = st.checkbox(
                    "Replace existing questions (uncheck to append)",
                    value=True, key="q_upload_replace",
                )
                _q_upload = st.file_uploader(
                    "Upload xlsx or csv (columns: Question, Pass Criteria, optionally Dimension)",
                    type=["xlsx", "csv"],
                    key="q_upload_file",
                    label_visibility="collapsed",
                )
                if _q_upload and _rfx_id and st.button("Upload questions", key="q_upload_btn"):
                    _n_ins, _err = _parse_questionnaire_upload(
                        _q_upload, _rfx_id, _sel_vtype, replace=_upload_replace
                    )
                    if _err:
                        st.error(f"Upload failed: {_err}")
                    else:
                        st.success(f"Inserted {_n_ins} questions.")
                        st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

            # ── Augment — visually connected below the A/B block ─────────────
            if _n_existing and _rfx_id:
                with st.expander("➕ Augment — add more questions via prompt", expanded=False):
                    _aug_prompt = st.text_area(
                        "What else should the questionnaire cover?",
                        placeholder="e.g. add questions about payment bond and retention requirements",
                        key="q_aug_prompt",
                        height=80,
                    )
                    if st.button("Add questions", key="q_aug_btn", disabled=not _aug_prompt.strip()):
                        with st.spinner("Generating additional questions…"):
                            try:
                                _added = _augment_rfx_questionnaire(_rfx_id, _sel_vtype, _aug_prompt.strip())
                                st.success(f"Added {len(_added)} question(s).")
                                st.rerun()
                            except Exception as _e:
                                st.error(f"Augmentation failed: {_e}")

            # ── Question review table ─────────────────────────────────────────
            if _existing_qs:
                _n_s1 = sum(1 for q in _existing_qs if q.get("stage", "stage_1_screening") == "stage_1_screening")
                _n_s2 = _n_existing - _n_s1
                _stage_label = f"{_n_s1} screening"
                if _n_s2:
                    _stage_label += f" + {_n_s2} evaluation (request if shortlisted)"
                st.markdown(f"#### Generated questionnaire — {_n_existing} questions")
                st.caption(f"Vendor type: **{_sel_vtype}** · {_stage_label}")
                _q_df = pd.DataFrame(_existing_qs)[["q_id", "dimension", "stage", "question", "pass_criteria"]]
                _q_df["stage"] = _q_df["stage"].map({
                    "stage_1_screening": "S1 screening",
                    "stage_2_evaluation": "S2 evaluation ⚑",
                }).fillna("S1 screening")
                _q_df.columns = ["Q ID", "Dimension", "Stage", "Question", "Pass Criteria"]
                st.dataframe(_q_df, use_container_width=True, hide_index=True)

                # Download as xlsx
                _q_xlsx_buf = io.BytesIO()
                _q_df.to_excel(_q_xlsx_buf, index=False, engine="openpyxl")
                st.download_button(
                    "⬇ Download questionnaire xlsx",
                    _q_xlsx_buf.getvalue(),
                    file_name=f"questionnaire_{_sel_vtype.replace(' ','_')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )

                if _rfx_id and _n_existing:
                    if st.button(
                        "🗑 Delete all questions for this vendor type",
                        key="q_delete_btn",
                        type="secondary",
                    ):
                        conn = _db_conn()
                        conn.execute(
                            "DELETE FROM rfx_questionnaire_definitions WHERE rfx_id=? AND vendor_type=?",
                            (_rfx_id, _sel_vtype),
                        )
                        conn.commit()
                        conn.close()
                        st.rerun()

            # ── Extract vendor answers ────────────────────────────────────────
            if _existing_qs:
                st.markdown("#### Extract vendor answers")
                st.caption(
                    "Upload a vendor document to extract their questionnaire answers. "
                    "The system matches free-text statements in the document to each question using LLM."
                )
                _ans_vendor_options = [
                    f"{v['vendor_id']} — {v['vendor_name'] or v['vendor_id']}"
                    for v in _tagged_vendors
                    if v["vendor_type"] == _sel_vtype
                ]
                if not _ans_vendor_options:
                    st.info(f"No vendors tagged as '{_sel_vtype}'. Tag them in Vendor Setup above.")
                else:
                    _ans_vendor_sel = st.selectbox(
                        "Select vendor to extract answers for:",
                        options=_ans_vendor_options,
                        key="q_ans_vendor",
                    )
                    _ans_vendor_id = _ans_vendor_sel.split(" — ")[0]
                    _ans_upload = st.file_uploader(
                        "Upload vendor document (PDF or xlsx)",
                        type=["pdf", "xlsx", "csv", "txt"],
                        key="q_ans_upload",
                    )
                    if _ans_upload and st.button("Extract answers", key="q_extract_btn", type="primary"):
                        with st.spinner(f"Extracting answers for {_ans_vendor_id}…"):
                            try:
                                _raw_bytes = _ans_upload.read()
                                _ans_name = _ans_upload.name.lower()
                                if _ans_name.endswith(".pdf"):
                                    import pdfplumber
                                    with pdfplumber.open(io.BytesIO(_raw_bytes)) as _pdf:
                                        _doc_text = "\n".join(
                                            (p.extract_text() or "") for p in _pdf.pages
                                        )
                                elif _ans_name.endswith((".xlsx", ".xls")):
                                    _ans_df = pd.read_excel(io.BytesIO(_raw_bytes), header=None)
                                    _doc_text = _ans_df.to_string(index=False, na_rep="")
                                else:
                                    _doc_text = _raw_bytes.decode("utf-8", errors="replace")
                                _n_ex, _ex_err = _extract_vendor_questionnaire_answers(
                                    _ans_vendor_id, _rfx_id, _sel_vtype,
                                    _doc_text, source_file=_ans_upload.name,
                                )
                                if _ex_err:
                                    st.error(f"Extraction failed: {_ex_err}")
                                else:
                                    st.success(f"Extracted {_n_ex} answers for {_ans_vendor_id}.")
                                    st.rerun()
                            except Exception as _exc:
                                st.error(f"Error: {_exc}")

            # ── Answers table ─────────────────────────────────────────────────
            if _rfx_id and _existing_qs:
                _qs_df, _resp_df = _get_questionnaire_answers_df(_rfx_id, _sel_vtype)
                if not _resp_df.empty:
                    # Coverage score: Stage 1 only
                    _s1_q_ids = set(_qs_df[_qs_df["stage"] == "stage_1_screening"]["q_id"].tolist())
                    _n_s1_total = len(_s1_q_ids)

                    st.markdown("#### Vendor answers")
                    if _n_s1_total < len(_qs_df):
                        st.caption(
                            f"Coverage % counts Stage 1 screening questions only ({_n_s1_total} of {len(_qs_df)}). "
                            "Stage 2 evaluation questions (⚑) are shown but excluded from scores."
                        )
                    _vendors_with_answers = _resp_df["vendor_id"].unique().tolist()
                    _wide_rows = []
                    for _, _qrow in _qs_df.iterrows():
                        _is_s2 = _qrow.get("stage", "stage_1_screening") == "stage_2_evaluation"
                        _r = {
                            "Q ID": _qrow["q_id"],
                            "Stage": "S2 ⚑" if _is_s2 else "S1",
                            "Dimension": _qrow["dimension"],
                            "Question": _qrow["question"][:120] + ("…" if len(_qrow["question"]) > 120 else ""),
                        }
                        for _vid in _vendors_with_answers:
                            _vname = _resp_df[_resp_df["vendor_id"] == _vid]["vendor_name"].iloc[0] if not _resp_df[_resp_df["vendor_id"] == _vid].empty else _vid
                            _match = _resp_df[
                                (_resp_df["vendor_id"] == _vid) & (_resp_df["q_id"] == _qrow["q_id"])
                            ]
                            if _match.empty:
                                _r[_vname] = "—"
                            else:
                                _passes = _match.iloc[0]["passes"]
                                _ans = _match.iloc[0]["answer"] or "(no response)"
                                _state = "✅" if _passes == 1 else ("❌" if _passes == 0 else "❓")
                                _r[_vname] = f"{_state} {_ans[:120]}"
                        _wide_rows.append(_r)

                    _wide_df = pd.DataFrame(_wide_rows)

                    # Coverage summary row (S1 only)
                    if _n_s1_total > 0:
                        _cov_row: dict = {"Q ID": "Coverage (S1)", "Stage": "", "Dimension": "", "Question": ""}
                        for _vid in _vendors_with_answers:
                            _vname = _resp_df[_resp_df["vendor_id"] == _vid]["vendor_name"].iloc[0] if not _resp_df[_resp_df["vendor_id"] == _vid].empty else _vid
                            _s1_answered = _resp_df[
                                (_resp_df["vendor_id"] == _vid) &
                                (_resp_df["q_id"].isin(_s1_q_ids)) &
                                (_resp_df["value_source"] == "extracted")
                            ].shape[0]
                            _cov_row[_vname] = f"{_s1_answered}/{_n_s1_total} ({_s1_answered*100//_n_s1_total}%)"
                        _wide_df = pd.concat([_wide_df, pd.DataFrame([_cov_row])], ignore_index=True)

                    def _q_style(df):
                        styles = pd.DataFrame("", index=df.index, columns=df.columns)
                        for col in _vendors_with_answers:
                            _vname_col = next(
                                (c for c in df.columns if _resp_df[_resp_df["vendor_id"] == col]["vendor_name"].values[0] == c
                                 if not _resp_df[_resp_df["vendor_id"] == col].empty),
                                None,
                            )
                            if _vname_col is None:
                                continue
                            for i, val in enumerate(df[_vname_col]):
                                v = str(val)
                                if v.startswith("✅"):
                                    styles.loc[i, _vname_col] = _CSS_Q_PASS
                                elif v.startswith("❌"):
                                    styles.loc[i, _vname_col] = _CSS_Q_FAIL
                                elif v.startswith("❓"):
                                    styles.loc[i, _vname_col] = _CSS_Q_AMBIG
                        return styles

                    st.dataframe(_wide_df.style.apply(_q_style, axis=None),
                                 use_container_width=True, hide_index=True)

                    _csv_wide = _wide_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇ Download answers CSV",
                        _csv_wide,
                        file_name=f"questionnaire_answers_{_sel_vtype.replace(' ','_')}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info(
                        "Questions are ready. Upload vendor documents above to extract answers."
                    )


# ── Tab 3: Data Quality ───────────────────────────────────────────────────────
with tab_dq:
    st.subheader("Per-vendor extraction overview")
    st.caption(
        "Review data completeness and confidence before running comparisons. "
        "The **value view** (bottom table) is the more important signal — a vendor can look "
        "fine by line count while their low-confidence lines are actually the highest-value items."
    )

    try:
        dq, total_lines = _data_quality_stats(st.session_state.rfx_id)

        # ── Warning banners ──────────────────────────────────────────────────
        for _, row in dq.iterrows():
            flags = []
            if row["coverage_pct"] < 70:
                flags.append(
                    f"coverage {row['coverage_pct']:.0f}% "
                    f"({int(row['lines_quoted'])}/{total_lines} lines quoted)"
                )
            if row["val_low_pct"] > 20:
                flags.append(
                    f"{row['val_low_pct']:.0f}% of quoted value is low-confidence (ec < 0.4)"
                )
            if flags:
                st.warning(
                    f"⚠️ **{row['vendor_id']} — {row['vendor_name']}**: "
                    + " · ".join(flags)
                )

        st.divider()

        # ── Coverage + confidence by COUNT ───────────────────────────────────
        st.markdown("**Coverage & confidence by line count**")
        count_df = dq[[
            "vendor_id", "vendor_name",
            "lines_quoted", "not_quoted_count", "coverage_pct",
            "ec_high_count", "ec_med_count", "ec_low_count",
        ]].copy()
        count_df.columns = [
            "Vendor", "Name",
            "Quoted", "Not Quoted", "Coverage %",
            "High (ec≥0.7)", "Med (0.4–0.69)", "Low (<0.4)",
        ]
        count_df["Coverage %"] = count_df["Coverage %"].apply(lambda x: f"{x:.0f}%")
        st.dataframe(count_df, use_container_width=True, hide_index=True)

        st.divider()

        # ── Confidence by VALUE ───────────────────────────────────────────────
        st.markdown("**Confidence by quoted value (INR) — primary trust signal**")
        st.caption(
            "Each % is share of that vendor's total quoted value (unit price × BOQ qty). "
            "A vendor with 90% high-confidence by COUNT can still have the bulk of their "
            "value sitting in the low-confidence bucket."
        )
        val_df = dq[[
            "vendor_id", "vendor_name",
            "val_total",
            "val_high_pct", "val_med_pct", "val_low_pct",
        ]].copy()
        val_df["val_total"] = val_df["val_total"].apply(
            lambda x: f"₹{x:,.0f}" if x > 0 else "—"
        )
        val_df.columns = [
            "Vendor", "Name",
            "Total Quoted (₹)",
            "High % (ec≥0.7)", "Med % (0.4–0.69)", "Low % (<0.4)",
        ]
        st.dataframe(val_df, use_container_width=True, hide_index=True)

        # ── Stacked bar: value breakdown per vendor ───────────────────────────
        st.divider()
        st.markdown("**Value confidence split — stacked bar**")
        bar_df = dq.set_index("vendor_id")[["val_high_pct", "val_med_pct", "val_low_pct"]]
        bar_df.columns = ["High (ec≥0.7) %", "Med (0.4–0.69) %", "Low (<0.4) %"]
        st.bar_chart(bar_df)

    except Exception as e:
        st.error(f"Data quality query failed: {e}")


# ── Co-pilot session helpers ──────────────────────────────────────────────────

def _reset_copilot_session():
    """Reset ALL co-pilot wizard state back to Step 1."""
    st.session_state.copilot_messages = []
    st.session_state.copilot_history = []
    st.session_state.copilot_staged_lines = []
    st.session_state.copilot_finalized_rfx_id = None
    st.session_state.copilot_file_name = None
    st.session_state.copilot_file_injected = False
    st.session_state.copilot_review_mode = False
    st.session_state.review_remarks = {}
    st.session_state.review_accepted = {}
    st.session_state.copilot_boq_xlsx = None
    st.session_state.copilot_boq_filename = None
    st.session_state.copilot_step = 1
    st.session_state.copilot_project_context = {}
    st.session_state.copilot_vision_text = ""
    st.session_state.copilot_vision_summary = ""
    _delete_draft(st.session_state.get("copilot_rfx_id", ""))
    st.session_state.pop("copilot_rfx_id", None)
    st.session_state.pop("_finalize_in_progress", None)


def _inject_step3_context():
    """
    Write Step 1+2 facts into copilot_history as a synthetic prior exchange
    so the model treats them as already established and never re-asks.
    Called exactly once when advancing from Step 2 → Step 3.
    """
    ctx = st.session_state.copilot_project_context
    vision = st.session_state.copilot_vision_text
    summary = st.session_state.copilot_vision_summary
    scope_str = ", ".join(ctx.get("scope_categories", []))
    ctx_msg = (
        "[ESTABLISHED PROJECT CONTEXT — do not ask about these again]\n"
        f"Project type: {ctx.get('project_type','')}\n"
        f"Location: {ctx.get('location','')}\n"
        f"Approximate area: {ctx.get('area','')}\n"
        f"Scope categories: {scope_str}\n"
        f"Plant selection: {ctx.get('plant_selection','')}\n"
        f"Maintenance: {ctx.get('maintenance_duration','')}\n"
        f"Payment schedule: {ctx.get('payment_schedule','')}\n"
        f"Quote validity: {ctx.get('quote_validity','')}\n"
        f"Design vision: {vision}\n"
    )
    st.session_state.copilot_history = [
        {"role": "user", "content": ctx_msg},
        {"role": "assistant", "content": (
            f"Understood. Here is my interpretation of the scope:\n\n{summary}\n\n"
            "I'll now ask only about genuine remaining gaps — species palette, "
            "budget constraints, irrigation specifics, or anything the vision left ambiguous. "
            "I will NOT ask about location, area, project type, or maintenance duration again."
        )},
    ]

    # Opening display message — what the user sees when Step 3 first loads
    plant_sel = ctx.get("plant_selection", "")
    location  = ctx.get("location", "")
    if "upload" in plant_sel.lower():
        opening = (
            "Ready to build your BOQ. **Upload your plant list** using the file uploader "
            "on the right, or paste species names here and I'll structure them into sections."
        )
    elif "suggest" in plant_sel.lower():
        opening = (
            f"I've reviewed your vision. I'll suggest a plant palette suited to **{location}**. "
            "Before I draft the list — do you have a rough budget per square metre in mind, "
            "or should I size quantities based on typical rates for this region?"
        )
    else:
        opening = (
            "Since vendors will propose the design, I'll build a minimal BOQ covering "
            "**site preparation, staking, and maintenance** terms. Ready to proceed, "
            "or is there anything specific you'd like to define upfront?"
        )
    st.session_state.copilot_messages = [{"role": "assistant", "content": opening}]


# ── Tab 4: Create New RFx (co-pilot) ─────────────────────────────────────────
with tab_new:
    st.subheader("RFQ Co-pilot — Create New RFx")
    st.caption(
        "Describe your project, upload a BOQ file (optional), and the co-pilot will "
        "help you build a complete line-item list and issue it to vendors."
    )

    # ── Finalized banner ──────────────────────────────────────────────────────
    if st.session_state.copilot_finalized_rfx_id:
        frid = st.session_state.copilot_finalized_rfx_id
        st.success(
            f"✅ RFx **{frid}** finalised and written to the database. "
            f"Select it in the sidebar to analyse it."
        )
        if st.session_state.copilot_boq_xlsx:
            st.download_button(
                label="⬇️ Download BOQ (xlsx)",
                data=st.session_state.copilot_boq_xlsx,
                file_name=st.session_state.copilot_boq_filename or f"{frid}_BOQ.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key="boq_download_btn",
            )
        if st.button("Start a new RFx", key="copilot_new"):
            _reset_copilot_session()
            st.rerun()

    elif st.session_state.copilot_review_mode:
        # ── REVIEW TABLE MODE ────────────────────────────────────────────────────
        _CATEGORY_MAP = {
            "trees":         "Trees / Palms / Bamboo",
            "palms":         "Trees / Palms / Bamboo",
            "bamboo":        "Trees / Palms / Bamboo",
            "shrubs":        "Shrubs / Climbers",
            "climbers":      "Shrubs / Climbers",
            "ground_covers": "Ground Cover / Lawn",
            "lawn":          "Ground Cover / Lawn",
            "ground cover":  "Ground Cover / Lawn",
            "soil_prep":     "Site Prep / Amendments",
            "site prep":     "Site Prep / Amendments",
            "amendments":    "Site Prep / Amendments",
            "staking":       "Staking",
        }
        _CATEGORY_ORDER = [
            "Trees / Palms / Bamboo",
            "Shrubs / Climbers",
            "Ground Cover / Lawn",
            "Site Prep / Amendments",
            "Staking",
            "Other",
        ]

        def _group_label(line: dict) -> str:
            sec = (line.get("section") or "").lower().strip()
            for key, label in _CATEGORY_MAP.items():
                if key in sec:
                    return label
            return "Other"

        # Build display rows with group, status
        lines = st.session_state.copilot_staged_lines
        needs_input_count = sum(
            1 for l in lines
            if _status_tag(l) == "needs_input" and not l.get("accepted", False)
        )

        st.markdown("### Draft RFx — Review & Confirm")
        col_hdr, col_fin = st.columns([3, 1])
        with col_hdr:
            if needs_input_count > 0:
                st.warning(
                    f"**{needs_input_count} line(s) need input** before you can finalise. "
                    "Add remarks below or accept them individually."
                )
            else:
                st.success("All lines confirmed or accepted — ready to finalise.")
        with col_fin:
            fin_disabled = needs_input_count > 0
            rfx_name_input = st.text_input(
                "RFx name (confirm or rename)",
                value=st.session_state.get("copilot_rfx_name_draft", ""),
                placeholder="e.g. Kukas Phase-1 Softscape",
                key="review_rfx_name",
                disabled=fin_disabled,
            )
            if rfx_name_input:
                st.session_state.copilot_rfx_name_draft = rfx_name_input
            if st.button(
                "Finalise RFx",
                key="review_finalize_btn",
                disabled=fin_disabled or not rfx_name_input.strip(),
                type="primary",
            ):
                with st.spinner("Finalising…"):
                    fin_result, fin_err = execute_finalize_rfx(rfx_name_input.strip())
                if fin_err:
                    st.error(fin_err)
                else:
                    st.success("RFx finalised!")
                    st.session_state.copilot_rfx_finalized = True
                    st.session_state.copilot_review_mode = False
                    st.rerun()

        st.divider()

        # Group lines by category
        from collections import defaultdict as _defaultdict
        grouped: dict = _defaultdict(list)
        for idx, line in enumerate(lines):
            grouped[_group_label(line)].append((idx, line))

        # Track edits in a parallel session key
        if "review_remarks" not in st.session_state:
            st.session_state.review_remarks = {}
        if "review_accepted" not in st.session_state:
            st.session_state.review_accepted = {}

        for cat in _CATEGORY_ORDER:
            cat_lines = grouped.get(cat, [])
            if not cat_lines:
                continue
            st.markdown(f"**{cat}** ({len(cat_lines)} lines)")

            # Build a dataframe for this group
            rows = []
            for orig_idx, l in cat_lines:
                status = _status_tag(l)
                formula_note = l.get("formula") or ""
                spec = l.get("spec_notes") or ""
                rows.append({
                    "_orig_idx": orig_idx,
                    "Description": l.get("description", ""),
                    "Spec": f"{spec} {formula_note}".strip(),
                    "Qty": l.get("quantity"),
                    "Unit": l.get("unit", ""),
                    "Source": l.get("value_source", ""),
                    "Status": status,
                    "Remarks": st.session_state.review_remarks.get(orig_idx, l.get("remarks", "")),
                    "Accepted": st.session_state.review_accepted.get(orig_idx, l.get("accepted", False)),
                })

            import pandas as _pd2
            df = _pd2.DataFrame(rows)
            # Colour-code status column via column_config
            edited = st.data_editor(
                df,
                key=f"review_editor_{cat}",
                use_container_width=True,
                hide_index=True,
                disabled=["_orig_idx", "Description", "Spec", "Qty", "Unit", "Source", "Status"],
                column_config={
                    "_orig_idx":  st.column_config.NumberColumn("_orig_idx", width="small"),
                    "Description": st.column_config.TextColumn("Description", width="large"),
                    "Spec":       st.column_config.TextColumn("Spec", width="medium"),
                    "Qty":        st.column_config.NumberColumn("Qty", width="small", format="%.2f"),
                    "Unit":       st.column_config.TextColumn("Unit", width="small"),
                    "Source":     st.column_config.TextColumn("Source", width="small"),
                    "Status":     st.column_config.TextColumn("Status", width="small"),
                    "Remarks":    st.column_config.TextColumn("Remarks", width="large"),
                    "Accepted":   st.column_config.CheckboxColumn("Accept", width="small"),
                },
                num_rows="fixed",
            )

            # Persist edits back to session state
            for _, row in edited.iterrows():
                oi = int(row["_orig_idx"])
                st.session_state.review_remarks[oi] = row["Remarks"]
                st.session_state.review_accepted[oi] = bool(row["Accepted"])
                # Write back to the canonical staged_lines list
                st.session_state.copilot_staged_lines[oi]["remarks"] = row["Remarks"]
                st.session_state.copilot_staged_lines[oi]["accepted"] = bool(row["Accepted"])

        st.divider()

        # Submit remarks as a structured diff to the co-pilot
        remarked = [
            (i, l) for i, l in enumerate(st.session_state.copilot_staged_lines)
            if l.get("remarks", "").strip()
        ]
        if remarked:
            if st.button("Send remarks to co-pilot", key="review_submit_remarks"):
                diff_lines = []
                for i, l in remarked:
                    diff_lines.append(
                        f"- Line {i} '{l['description']}' ({l.get('section','')}): {l['remarks']}"
                    )
                diff_msg = (
                    "The analyst has reviewed the staged lines and left the following remarks. "
                    "Please address each one narrowly — revise the line, ask one clarifying question about it, "
                    "or explain why it is derived that way. Do not re-summarise all lines.\n\n"
                    + "\n".join(diff_lines)
                )
                st.session_state.copilot_messages.append({"role": "user", "content": "(Remarks submitted)"})
                with st.spinner("Co-pilot reading remarks…"):
                    answer, new_history, tool_log = run_copilot_agent(
                        diff_msg, st.session_state.copilot_history
                    )
                st.session_state.copilot_history = new_history
                st.session_state.copilot_messages.append(
                    {"role": "assistant", "content": answer, "tool_log": tool_log}
                )
                # Clear remarks after submitting
                for i, l in remarked:
                    st.session_state.copilot_staged_lines[i]["remarks"] = ""
                st.session_state.review_remarks = {}
                st.rerun()

        # General chat below the table
        st.markdown("##### General chat")
        for msg in st.session_state.copilot_messages[-6:]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        copilot_input_review = st.chat_input(
            "Ask a general question (use remarks above for line-specific edits)…",
            key="copilot_input_review",
        )
        if copilot_input_review:
            st.session_state.copilot_messages.append({"role": "user", "content": copilot_input_review})
            with st.chat_message("user"):
                st.markdown(copilot_input_review)
            with st.chat_message("assistant"):
                with st.spinner("Co-pilot thinking…"):
                    answer, new_history, tool_log = run_copilot_agent(
                        copilot_input_review, st.session_state.copilot_history
                    )
                st.markdown(answer)
            st.session_state.copilot_history = new_history
            st.session_state.copilot_messages.append(
                {"role": "assistant", "content": answer, "tool_log": tool_log}
            )
            _save_draft()
            st.rerun()

        if st.button("← Back to chat / edit lines", key="review_back"):
            st.session_state.copilot_review_mode = False
            st.rerun()

    elif st.session_state.copilot_step == 1:
        # ── STEP 1 — Structured project intake form ───────────────────────────

        # Show resumable drafts before the form (only when no active wizard session)
        if not st.session_state.get("copilot_rfx_id"):
            _pending_drafts = _load_all_drafts()
            if _pending_drafts:
                with st.expander(
                    f"📋 {len(_pending_drafts)} draft(s) in progress — resume?",
                    expanded=True,
                ):
                    for _d in _pending_drafts:
                        _ts = _d["updated_at"][:16].replace("T", " ")
                        _col1, _col2, _col3 = st.columns([3, 1, 1])
                        _col1.markdown(
                            f"**{_d['rfx_name'] or '(unnamed)'}** &nbsp;·&nbsp; "
                            f"Step {_d['step']}/3 &nbsp;·&nbsp; *{_ts} UTC*"
                        )
                        if _col2.button("Resume", key=f"draft_resume_{_d['draft_id']}"):
                            _load_draft(_d["draft_id"])
                        if _col3.button("Discard", key=f"draft_discard_{_d['draft_id']}"):
                            _delete_draft(_d["draft_id"])
                            st.rerun()
                st.divider()

        st.markdown(
            "<div class='rfx-step-progress'>"
            "<span class='active-step'>● Step 1 of 3 — Project scope</span>"
            " &nbsp;›&nbsp; "
            "<span style='opacity:0.45'>Step 2 of 3 — Design vision</span>"
            " &nbsp;›&nbsp; "
            "<span style='opacity:0.45'>Step 3 of 3 — Line-item build</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        st.markdown("### Project scope")
        st.caption("Fill in what you know — all fields except Location are pre-filled with sensible defaults.")

        # ── Restore helpers for back-navigation ──────────────────────────────
        _ctx = st.session_state.copilot_project_context
        _PT_OPTIONS = [
            "Residential complex / township",
            "Office / commercial campus",
            "Hotel / resort",
            "Public / institutional",
            "Other",
        ]
        _PS_OPTIONS = [
            "I have a plant list (will upload)",
            "Suggest plants suitable for my location",
            "Let vendor propose the design",
        ]
        _MD_OPTIONS = ["Not required", "3 months", "6 months", "12 months", "24 months"]
        _PMT_OPTIONS = [
            "30% advance / 40% on delivery / 30% on handover",
            "50% advance / 50% on completion",
            "100% on completion",
            "To be negotiated",
        ]
        _QV_OPTIONS = ["30 days", "45 days", "60 days", "90 days"]

        def _idx(opts, val, default=0):
            try:
                return opts.index(val)
            except ValueError:
                return default

        with st.form("rfx_step1_form", clear_on_submit=False):
            rfx_name_s1 = st.text_input(
                "RFx name",
                value=st.session_state.get("copilot_rfx_name_draft", ""),
                placeholder="e.g. Kukas Phase-1 Softscape",
                help="Short name used to identify this RFx in the comparison views.",
            )
            project_type = st.radio(
                "Project type",
                options=_PT_OPTIONS,
                index=_idx(_PT_OPTIONS, _ctx.get("project_type", ""), 0),
                horizontal=False,
            )
            location = st.text_input(
                "Location",
                value=_ctx.get("location", ""),
                placeholder="e.g. Jaipur / semi-arid",
                help="City or region + climate type if known. Used to infer species suitability and soil prep norms.",
            )
            area = st.text_input(
                "Approximate area",
                value=_ctx.get("area", ""),
                placeholder="e.g. 2 acres  or  5000 sqm",
                help="Total landscaped area. Rough estimate is fine — used to size soil amendment quantities.",
            )
            scope_categories = st.multiselect(
                "Scope categories",
                options=["Softscape", "Hardscape", "Site preparation", "Irrigation", "Lighting", "Maintenance"],
                default=_ctx.get("scope_categories", ["Softscape", "Site preparation"]),
                help="Select all work packages this RFx should cover.",
            )
            plant_selection = st.radio(
                "Plant selection",
                options=_PS_OPTIONS,
                index=_idx(_PS_OPTIONS, _ctx.get("plant_selection", ""), 0),
                horizontal=False,
            )
            maintenance_duration = st.radio(
                "Maintenance duration",
                options=_MD_OPTIONS,
                index=_idx(_MD_OPTIONS, _ctx.get("maintenance_duration", ""), 3),
                horizontal=True,
            )

            st.divider()
            st.markdown("**Commercial terms** *(sent to vendors with the RFx)*")
            payment_schedule = st.radio(
                "Payment schedule",
                options=_PMT_OPTIONS,
                index=_idx(_PMT_OPTIONS, _ctx.get("payment_schedule", ""), 0),
                horizontal=False,
                help="Vendors will see this in the issued RFx. Select the schedule that matches your procurement policy.",
            )
            quote_validity = st.radio(
                "Quote validity required",
                options=_QV_OPTIONS,
                index=_idx(_QV_OPTIONS, _ctx.get("quote_validity", ""), 1),
                horizontal=True,
                help="How long vendors must hold their quoted prices.",
            )

            submitted1 = st.form_submit_button("Next: describe the design vision →", type="primary")

        if submitted1:
            if not rfx_name_s1.strip():
                st.error("RFx name is required.")
            elif not location.strip():
                st.error("Location is required — please enter a city or region.")
            elif not area.strip():
                st.error("Approximate area is required.")
            elif not scope_categories:
                st.error("Select at least one scope category.")
            else:
                st.session_state.copilot_rfx_name_draft = rfx_name_s1.strip()
                st.session_state.copilot_project_context = {
                    "project_type": project_type,
                    "location": location.strip(),
                    "area": area.strip(),
                    "scope_categories": scope_categories,
                    "plant_selection": plant_selection,
                    "maintenance_duration": maintenance_duration,
                    "payment_schedule": payment_schedule,
                    "quote_validity": quote_validity,
                }
                st.session_state.copilot_step = 2
                # Assign rfx_id on first forward-step so all subsequent saves use the same id
                if "copilot_rfx_id" not in st.session_state:
                    st.session_state["copilot_rfx_id"] = _gen_rfx_id(rfx_name_s1.strip())
                _save_draft()
                st.rerun()

    elif st.session_state.copilot_step == 2:
        # ── STEP 2 — Design vision ────────────────────────────────────────────
        st.markdown(
            "<div class='rfx-step-progress'>"
            "<span style='opacity:0.45'>Step 1 of 3 — Project scope</span>"
            " &nbsp;›&nbsp; "
            "<span class='active-step'>● Step 2 of 3 — Design vision</span>"
            " &nbsp;›&nbsp; "
            "<span style='opacity:0.45'>Step 3 of 3 — Line-item build</span>"
            "</div>",
            unsafe_allow_html=True,
        )
        ctx = st.session_state.copilot_project_context
        st.markdown("### Design vision")

        # Read-only Step 1 summary card
        scope_str = ", ".join(ctx.get("scope_categories", []))
        st.info(
            f"**{ctx['project_type']}** · {ctx['location']} · {ctx['area']}  \n"
            f"Scope: {scope_str}  \n"
            f"Plants: {ctx['plant_selection']}  \n"
            f"Maintenance: {ctx['maintenance_duration']}  \n"
            f"Payment: {ctx.get('payment_schedule', '—')}  \n"
            f"Quote validity: {ctx.get('quote_validity', '—')}",
        )

        # Check if an upload path will make Step 2 optional
        has_plant_list = ctx.get("plant_selection") == "I have a plant list (will upload)"

        vision_placeholder = (
            "e.g. Lush resort feel — tall palms along the driveway, flowering shrubs near the lobby, "
            "low-maintenance ground cover everywhere else. Prefer native / drought-tolerant species. "
            "Avoid anything that needs daily watering."
        )
        vision_text = st.text_area(
            "What should the landscape look and feel like?",
            value=st.session_state.copilot_vision_text,
            height=140,
            placeholder=vision_placeholder,
            help="Be as free-form as you like. Style, mood, species preferences, things to avoid.",
        )

        col_sub, col_skip = st.columns([2, 1])
        with col_sub:
            proceed_vision = st.button("Interpret & continue →", type="primary",
                                       disabled=not vision_text.strip())
        with col_skip:
            if has_plant_list:
                skip_vision = st.button("Skip (I'll upload the plant list in Step 3)")
            else:
                skip_vision = False

        if st.session_state.copilot_vision_summary:
            # Summary already generated this session — show it and allow confirmation
            st.markdown("**Here's what I understood:**")
            st.markdown(st.session_state.copilot_vision_summary)
            if st.button("Looks right — start building →", type="primary"):
                _inject_step3_context()
                st.session_state.copilot_step = 3
                _save_draft()
                st.rerun()

        elif proceed_vision and vision_text.strip():
            st.session_state.copilot_vision_text = vision_text.strip()
            with st.spinner("Interpreting your design vision…"):
                summary = _scope_interpretation_call(ctx, vision_text.strip())
            st.session_state.copilot_vision_summary = summary
            st.rerun()

        elif skip_vision:
            st.session_state.copilot_vision_text = "(plant list will be uploaded)"
            st.session_state.copilot_vision_summary = (
                "Analyst chose to upload a plant list directly — no free-text vision provided."
            )
            _inject_step3_context()
            st.session_state.copilot_step = 3
            _save_draft()
            st.rerun()

        if st.button("← Back to project scope", key="step2_back"):
            st.session_state.copilot_vision_summary = ""
            st.session_state.copilot_step = 1
            st.rerun()

    else:
        # ── STEP 3 — Agentic chat loop ────────────────────────────────────────
        st.markdown(
            "<div class='rfx-step-progress'>"
            "<span style='opacity:0.45'>Step 1 of 3 — Project scope</span>"
            " &nbsp;›&nbsp; "
            "<span style='opacity:0.45'>Step 2 of 3 — Design vision</span>"
            " &nbsp;›&nbsp; "
            "<span class='active-step'>● Step 3 of 3 — Line-item build</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Back button — only shown when no lines staged yet (going back would lose staged work)
        if not st.session_state.copilot_staged_lines:
            if st.button("← Back to design vision", key="step3_back"):
                st.session_state.copilot_step = 2
                st.session_state.copilot_history = []
                st.session_state.copilot_messages = []
                st.rerun()

        # Show collapsed Step 1+2 context at top
        ctx = st.session_state.copilot_project_context
        if ctx:
            scope_str = ", ".join(ctx.get("scope_categories", []))
            with st.expander("Project context (confirmed)", expanded=False):
                st.markdown(
                    f"**{ctx.get('project_type','')}** · {ctx.get('location','')} · {ctx.get('area','')}  \n"
                    f"Scope: {scope_str}  |  Plants: {ctx.get('plant_selection','')}  |  "
                    f"Maintenance: {ctx.get('maintenance_duration','')}"
                )
                if st.session_state.copilot_vision_summary:
                    st.markdown("---")
                    st.markdown(st.session_state.copilot_vision_summary)

        col_chat, col_staged = st.columns([2, 1])

        with col_staged:
            n_staged = len(st.session_state.copilot_staged_lines)
            st.markdown(f"**Staged lines ({n_staged})**")
            if st.session_state.copilot_staged_lines:
                staged_df = pd.DataFrame(st.session_state.copilot_staged_lines)
                display_cols = [c for c in ["section", "description", "unit", "quantity", "boq_unit_rate"]
                                if c in staged_df.columns]
                st.dataframe(staged_df[display_cols], use_container_width=True,
                             hide_index=True, height=300)
            else:
                st.caption("No lines staged yet — start chatting below.")

            st.divider()
            uploaded = st.file_uploader(
                "Upload BOQ (PDF, Excel, or text)",
                type=["pdf", "xlsx", "xls", "txt", "csv", "md"],
                key="copilot_upload",
            )
            # Detect a newly uploaded or changed file
            if uploaded and uploaded.name != st.session_state.copilot_file_name:
                st.session_state.copilot_file_name = uploaded.name
                st.session_state.copilot_file_injected = False

            if st.button("Clear co-pilot session", key="copilot_clear"):
                _reset_copilot_session()
                st.rerun()

        with col_chat:
            # Render chat history (skip the synthetic context-injection turns)
            for msg in st.session_state.copilot_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    if msg.get("tool_log"):
                        with st.expander(f"🔧 {len(msg['tool_log'])} tool call(s)", expanded=False):
                            for t in msg["tool_log"]:
                                st.caption(f"**{t['tool']}**")
                                st.code(t["code"][:300], language="json")
                                if t["error"]:
                                    st.error(t["result"][:400])
                                else:
                                    st.text(t["result"][:400])

            copilot_input = st.chat_input("Ask a question or give an instruction…",
                                          key="copilot_input")
            if copilot_input:
                # Route uploaded file through extract.py's PLANT_LIST_SYSTEM parser
                # rather than dumping raw text into the LLM context.
                user_text = copilot_input
                if (uploaded and not st.session_state.copilot_file_injected
                        and st.session_state.copilot_file_name):
                    uploaded.seek(0)
                    with st.spinner(f"Parsing {uploaded.name} with extract.py…"):
                        try:
                            import sys as _sys, os as _os
                            _src = _os.path.join(_os.path.dirname(__file__), "..", "src")
                            if _src not in _sys.path:
                                _sys.path.insert(0, _src)
                            from extract import extract_plant_list_doc
                            raw_text = _extract_file_text(uploaded)
                            parsed = extract_plant_list_doc(raw_text)
                            n_lines = len(parsed.get("line_items", []))
                            summary = parsed.get("parse_summary", "")
                            structured_json = json.dumps(parsed, indent=2)
                            user_text = (
                                f"I've uploaded a plant list / BOQ: **{uploaded.name}**\n\n"
                                f"It was parsed via extract.py (PLANT_LIST_SYSTEM). "
                                f"{n_lines} line items extracted. Parser summary: {summary}\n\n"
                                f"Structured output:\n```json\n{structured_json[:10000]}\n```\n\n"
                                f"Please call propose_line_item() for each line with "
                                f"parse_confidence ≥ 0.5, flag lower-confidence rows, "
                                f"and ask me to confirm before staging them.\n\n"
                                f"Buyer's note: {copilot_input}"
                            )
                        except Exception as _e:
                            uploaded.seek(0)
                            raw_text = _extract_file_text(uploaded)
                            user_text = (
                                f"I've uploaded a BOQ file: **{uploaded.name}** "
                                f"(extract.py unavailable: {_e})\n\n"
                                f"```\n{raw_text[:12000]}\n```\n\n"
                                f"{copilot_input}"
                            )
                    st.session_state.copilot_file_injected = True

                st.session_state.copilot_messages.append(
                    {"role": "user", "content": copilot_input}
                )
                with st.chat_message("user"):
                    st.markdown(copilot_input)

                with st.chat_message("assistant"):
                    with st.spinner("Co-pilot thinking…"):
                        answer, new_history, tool_log = run_copilot_agent(
                            user_text, st.session_state.copilot_history
                        )
                    st.markdown(answer)
                    if tool_log:
                        with st.expander(f"🔧 {len(tool_log)} tool call(s)", expanded=False):
                            for t in tool_log:
                                st.caption(f"**{t['tool']}**")
                                st.code(t["code"][:300], language="json")
                                if t["error"]:
                                    st.error(t["result"][:400])
                                else:
                                    st.text(t["result"][:400])

                st.session_state.copilot_history = new_history
                st.session_state.copilot_messages.append(
                    {"role": "assistant", "content": answer, "tool_log": tool_log}
                )
                _save_draft()
                st.rerun()

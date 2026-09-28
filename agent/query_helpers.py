"""
query_helpers.py — Plain Python functions over comparison.db.
Consumed by the analyst agent (agent/app.py) and usable standalone for sanity-checks.

Two primary surfaces:
  get_comparison_table()  — full flat DataFrame (current versions only, keyed on line_id × vendor)
  get_line(line_id)       — all vendors' data for one RFx line, with BOQ reference

Both return DataFrames with human-readable columns; no raw JSON blobs exposed.
"""
import os
import sqlite3
import json
import pandas as pd
from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "comparison.db")


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_comparison_table(
    section: Optional[str] = None,
    exclude_superseded: bool = True,
    exclude_low_confidence: bool = False,
    confidence_threshold: float = 0.70,
) -> pd.DataFrame:
    """
    Full comparison table: one row per (vendor × rfx_line), current versions only by default.

    Columns returned:
      line_id, section, description, unit, boq_unit_rate,
      vendor_id, vendor_name, source_file, document_version,
      raw_unit_price, raw_unit, normalized_unit_price,
      spec_grade_match, extraction_confidence, match_confidence,
      value_source, flags, source_snippet, labor_included, freight_included

    Filters:
      section              — filter to one section (soil_prep / trees / shrubs / ground_covers / lawn / staking)
      exclude_superseded   — True: skip phoenix v1 rows (default)
      exclude_low_confidence — True: skip rows below confidence_threshold
    """
    where_clauses = []
    params = []

    if exclude_superseded:
        where_clauses.append("(ve.superseded_by IS NULL OR ve.superseded_by = '')")

    if section:
        where_clauses.append("rl.section = ?")
        params.append(section)

    if exclude_low_confidence:
        where_clauses.append("ve.extraction_confidence >= ?")
        params.append(confidence_threshold)

    where_sql = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""

    sql = f"""
        SELECT
            rl.line_id,
            rl.section,
            rl.description,
            rl.unit                     AS boq_unit,
            rl.quantity                 AS boq_quantity,
            rl.boq_unit_rate,
            rl.spec_notes               AS boq_spec_notes,
            rl.gpt_discrepancy,
            ve.vendor_id,
            ve.vendor_name,
            ve.source_file,
            ve.document_version,
            ve.raw_unit_price,
            ve.raw_unit,
            ve.normalized_unit_price,
            ve.spec_grade_quoted,
            ve.spec_grade_match,
            ve.labor_included,
            ve.freight_included,
            ve.extraction_confidence,
            ve.match_confidence,
            ve.value_source,
            ve.flags,
            ve.source_snippet,
            ve.source_location
        FROM vendor_extractions ve
        JOIN rfx_lines rl ON ve.line_id = rl.line_id
        {where_sql}
        ORDER BY rl.section, rl.line_id, ve.vendor_id
    """
    conn = _conn()
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()

    # Parse flags JSON → list; add a human-readable summary column
    df["flags"] = df["flags"].apply(_parse_flags)
    df["flagged"] = df["flags"].apply(lambda f: len(f) > 0)
    df["flag_summary"] = df["flags"].apply(lambda f: "; ".join(f) if f else "")

    return df


def get_line(line_id: str, include_superseded: bool = False) -> pd.DataFrame:
    """
    All vendors' extractions for a single RFx line item.
    Returns a compact view with BOQ reference rate for easy eyeballing.

    Set include_superseded=True to see Phoenix v1 alongside v2.
    """
    where = "WHERE ve.line_id = ?"
    params = [line_id]
    if not include_superseded:
        where += " AND (ve.superseded_by IS NULL OR ve.superseded_by = '')"

    sql = f"""
        SELECT
            rl.description              AS rfx_description,
            rl.unit                     AS boq_unit,
            rl.quantity                 AS boq_quantity,
            rl.boq_unit_rate,
            rl.spec_notes               AS boq_spec,
            ve.vendor_id,
            ve.source_file,
            ve.document_version,
            ve.superseded_by,
            ve.raw_unit_price,
            ve.raw_unit,
            ve.normalized_unit_price,
            ve.spec_grade_quoted,
            ve.spec_grade_match,
            ve.extraction_confidence    AS ec,
            ve.match_confidence         AS mc,
            ve.value_source,
            ve.flags,
            ve.source_snippet,
            ve.normalization_note
        FROM vendor_extractions ve
        JOIN rfx_lines rl ON ve.line_id = rl.line_id
        {where}
        ORDER BY ve.vendor_id, ve.document_version
    """
    conn = _conn()
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()

    df["flags"] = df["flags"].apply(_parse_flags)
    df["flag_summary"] = df["flags"].apply(lambda f: "; ".join(f) if f else "—")

    # Add pct_vs_boq so eyeballing price vs reference is instant
    df["pct_vs_boq"] = df.apply(
        lambda r: f"{(r['normalized_unit_price'] / r['boq_unit_rate'] * 100):.0f}%"
        if (pd.notna(r['normalized_unit_price']) and pd.notna(r['boq_unit_rate'])
            and r['boq_unit_rate'] > 0 and r['value_source'] != 'unknown')
        else "—",
        axis=1
    )

    return df


def get_questionnaire_summary() -> pd.DataFrame:
    """
    Pivot: vendors × questions showing pass/fail/ambiguous.
    Useful for agent and for UI questionnaire panel.
    """
    sql = """
        SELECT
            qr.vendor_id,
            qr.q_id,
            qd.question,
            qd.pass_criteria,
            qr.answer,
            qr.passes,
            qr.confidence,
            qr.value_source,
            qr.source_location
        FROM questionnaire_responses qr
        JOIN questionnaire_definitions qd ON qr.q_id = qd.q_id
        ORDER BY qr.q_id, qr.vendor_id
    """
    conn = _conn()
    df = pd.read_sql_query(sql, conn)
    conn.close()

    df["pass_label"] = df["passes"].map({1: "PASS", 0: "FAIL"}).fillna("AMBIGUOUS")
    df.loc[df["value_source"] == "unknown", "pass_label"] = "NO_RESPONSE"

    return df


def get_flagged_lines(vendor_id: Optional[str] = None) -> pd.DataFrame:
    """All extractions with at least one flag (current versions only)."""
    where = "(ve.superseded_by IS NULL OR ve.superseded_by = '') AND ve.flags != '[]'"
    params = []
    if vendor_id:
        where += " AND ve.vendor_id = ?"
        params.append(vendor_id)

    sql = f"""
        SELECT ve.vendor_id, ve.line_id, rl.description, ve.source_file,
               ve.raw_unit_price, ve.raw_unit, ve.normalized_unit_price,
               ve.extraction_confidence, ve.value_source, ve.flags, ve.source_snippet
        FROM vendor_extractions ve
        JOIN rfx_lines rl ON ve.line_id = rl.line_id
        WHERE {where}
        ORDER BY ve.vendor_id, ve.line_id
    """
    conn = _conn()
    df = pd.read_sql_query(sql, conn, params=params)
    conn.close()
    df["flags"] = df["flags"].apply(_parse_flags)
    return df


def get_vendor_totals(exclude_flagged: bool = True) -> pd.DataFrame:
    """
    Per-vendor total (sum of normalized_unit_price × boq_quantity) for each section.
    Rows with value_source='unknown' or exclude_from_totals flag are always excluded.
    Set exclude_flagged=True (default) to also exclude any other flagged rows.
    """
    df = get_comparison_table(exclude_superseded=True)

    # Always exclude unknown / exclude_from_totals
    mask = (df["value_source"] != "unknown") & ~df["flag_summary"].str.contains("exclude_from_totals", na=False)
    if exclude_flagged:
        mask &= ~df["flagged"]
    df = df[mask].copy()

    df["line_total"] = df["normalized_unit_price"] * df["boq_quantity"]
    summary = (
        df.groupby(["vendor_id", "section"])["line_total"]
        .sum()
        .unstack(fill_value=0)
        .assign(grand_total=lambda x: x.sum(axis=1))
        .reset_index()
    )
    return summary


def get_section_totals(rfx_id: str, exclude_superseded: bool = True) -> pd.DataFrame:
    """
    Unified section-level comparison across vendors with mixed granularity.

    Returns one row per (vendor_id, section) with:
      - source: 'direct' (from vendor_section_quotes)
      - source: 'summed' (summed from vendor_extractions line items × boq_quantity)
      vendor_section_quotes always wins for a given (vendor_id, section).

    Columns: vendor_id, vendor_name, section, total_amount, value_source,
             source, estimate_low, estimate_high, package_description,
             extraction_confidence, flags, document_version, superseded_by
    """
    conn = _conn()

    # 1. Direct section quotes
    where_sq = "WHERE vsq.rfx_id = ?"
    params_sq = [rfx_id]
    if exclude_superseded:
        where_sq += " AND (vsq.superseded_by IS NULL OR vsq.superseded_by = '')"
    sq_df = pd.read_sql_query(f"""
        SELECT vsq.vendor_id, vsq.vendor_name, vsq.section,
               vsq.total_amount, vsq.value_source, vsq.estimate_low, vsq.estimate_high,
               vsq.package_description, vsq.extraction_confidence, vsq.flags,
               vsq.document_version, vsq.superseded_by,
               'direct' AS source
        FROM vendor_section_quotes vsq
        {where_sq}
        ORDER BY vsq.vendor_id, vsq.section
    """, conn, params=params_sq)

    # 2. Summed line-item totals (for vendors with line-item data)
    where_ve = "WHERE ve.rfx_id = ? AND rl.rfx_id = ?"
    params_ve = [rfx_id, rfx_id]
    if exclude_superseded:
        where_ve += " AND (ve.superseded_by IS NULL OR ve.superseded_by = '')"
    summed_df = pd.read_sql_query(f"""
        SELECT ve.vendor_id, ve.vendor_name, rl.section,
               SUM(
                   CASE WHEN ve.value_source != 'unknown'
                        THEN COALESCE(ve.normalized_unit_price, ve.raw_unit_price, 0) * COALESCE(rl.quantity, 0)
                        ELSE 0 END
               ) AS total_amount,
               'summed' AS source,
               NULL AS value_source,
               NULL AS estimate_low, NULL AS estimate_high,
               NULL AS package_description, NULL AS extraction_confidence,
               '[]' AS flags, MAX(ve.document_version) AS document_version,
               NULL AS superseded_by
        FROM vendor_extractions ve
        JOIN rfx_lines rl ON ve.line_id = rl.line_id
        {where_ve}
        GROUP BY ve.vendor_id, ve.vendor_name, rl.section
        HAVING total_amount > 0
    """, conn, params=params_ve)

    conn.close()

    if sq_df.empty and summed_df.empty:
        return pd.DataFrame()

    # Merge: direct section quotes override summed for the same (vendor_id, section)
    if not sq_df.empty and not summed_df.empty:
        direct_keys = set(zip(sq_df["vendor_id"], sq_df["section"]))
        summed_df = summed_df[
            ~summed_df.apply(lambda r: (r["vendor_id"], r["section"]) in direct_keys, axis=1)
        ]

    combined = pd.concat([sq_df, summed_df], ignore_index=True)
    combined["flags"] = combined["flags"].apply(
        lambda f: _parse_flags(f) if isinstance(f, str) else (f if isinstance(f, list) else [])
    )
    return combined.sort_values(["vendor_id", "section"]).reset_index(drop=True)


def _parse_flags(raw) -> list:
    if not raw or raw in ("[]", "", None):
        return []
    if isinstance(raw, list):
        return raw
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return [str(raw)]


# ── CLI sanity-check ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    LINE_IDS = ["L_TR_07", "L_ST_01", "L_TR_15", "L_SP_02"]

    if len(sys.argv) > 1:
        LINE_IDS = sys.argv[1:]

    for lid in LINE_IDS:
        df = get_line(lid)
        if df.empty:
            print(f"\n[{lid}] — no data")
            continue
        ref = df.iloc[0]
        print(f"\n{'='*70}")
        print(f"  {lid}: {ref['rfx_description'][:60]}")
        print(f"  BOQ: {ref['boq_quantity']} × {ref['boq_unit']} @ ₹{ref['boq_unit_rate']}  spec={ref['boq_spec']}")
        print(f"{'='*70}")
        cols = ["vendor_id", "source_file", "document_version", "raw_unit_price",
                "raw_unit", "normalized_unit_price", "pct_vs_boq",
                "spec_grade_match", "ec", "value_source", "flag_summary"]
        print(df[cols].to_string(index=False))

    print("\n" + "="*70)
    print("  Full comparison table shape:", get_comparison_table().shape)
    print("  Flagged lines:", get_flagged_lines().shape[0])
    print("  Questionnaire summary:")
    qs = get_questionnaire_summary()
    pivot = qs.pivot(index="vendor_id", columns="q_id", values="pass_label")
    print(pivot.to_string())

    print("\n  Vendor totals (clean lines only):")
    print(get_vendor_totals().to_string(index=False))

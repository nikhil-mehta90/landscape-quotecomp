"""
build_db.py — Create SQLite schema and load from JSON files.
Rebuildable at any time from the JSON layer alone.
Run after rfx_builder.py + extract.py + normalize.py.
"""
import sqlite3
import json
import os
import re
import glob
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "comparison.db")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def create_schema(conn: sqlite3.Connection):
    cur = conn.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS rfx_lines (
        rfx_id          TEXT,
        line_id         TEXT PRIMARY KEY,
        section         TEXT NOT NULL,
        description     TEXT NOT NULL,
        species_name    TEXT,
        unit            TEXT NOT NULL,
        quantity        REAL,
        boq_unit_rate   REAL,
        boq_total       REAL,
        spec_notes      TEXT,
        gpt_discrepancy TEXT
    );

    CREATE TABLE IF NOT EXISTS questionnaire_definitions (
        q_id            TEXT PRIMARY KEY,
        question        TEXT NOT NULL,
        pass_criteria   TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS vendor_extractions (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        rfx_id                  TEXT,
        vendor_id               TEXT NOT NULL,
        vendor_name             TEXT,
        source_file             TEXT,
        document_version        TEXT,
        superseded_by           TEXT,
        line_id                 TEXT REFERENCES rfx_lines(line_id),
        matched                 INTEGER,
        match_confidence        REAL,
        raw_unit_price          REAL,
        raw_unit                TEXT,
        raw_currency            TEXT DEFAULT 'INR',
        normalized_unit_price   REAL,
        normalization_note      TEXT,
        quantity_quoted         REAL,
        freight_included        TEXT DEFAULT 'unknown',
        labor_included          TEXT DEFAULT 'unknown',
        spec_grade_quoted       TEXT,
        spec_grade_match        TEXT DEFAULT 'unknown',
        source_snippet          TEXT,
        source_location         TEXT,
        extraction_confidence   REAL,
        value_source            TEXT DEFAULT 'extracted',
        flags                   TEXT DEFAULT '[]',
        extracted_at            TEXT,
        granularity             TEXT DEFAULT 'line_item'
    );

    CREATE TABLE IF NOT EXISTS vendor_section_quotes (
        id                      INTEGER PRIMARY KEY AUTOINCREMENT,
        rfx_id                  TEXT NOT NULL,
        vendor_id               TEXT NOT NULL,
        vendor_name             TEXT,
        source_file             TEXT,
        document_version        TEXT,
        superseded_by           TEXT,
        revision_type           TEXT DEFAULT 'quote',
        section                 TEXT NOT NULL,
        total_amount            REAL,
        estimate_low            REAL,
        estimate_high           REAL,
        currency                TEXT DEFAULT 'INR',
        value_source            TEXT DEFAULT 'vendor_quote',
        package_description     TEXT,
        source_snippet          TEXT,
        extraction_confidence   REAL,
        flags                   TEXT DEFAULT '[]',
        extracted_at            TEXT,
        UNIQUE(rfx_id, vendor_id, source_file, section)
    );

    CREATE TABLE IF NOT EXISTS questionnaire_responses (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        vendor_id           TEXT NOT NULL,
        source_file         TEXT,
        document_version    TEXT,
        q_id                TEXT REFERENCES questionnaire_definitions(q_id),
        question            TEXT,
        answer              TEXT,
        answer_type         TEXT DEFAULT 'text',
        passes              INTEGER,
        confidence          REAL,
        source_location     TEXT,
        value_source        TEXT DEFAULT 'extracted'
    );

    CREATE INDEX IF NOT EXISTS idx_ve_vendor_line ON vendor_extractions(vendor_id, line_id);
    CREATE INDEX IF NOT EXISTS idx_ve_line ON vendor_extractions(line_id);
    CREATE INDEX IF NOT EXISTS idx_qr_vendor ON questionnaire_responses(vendor_id);
    CREATE INDEX IF NOT EXISTS idx_vsq_rfx_vendor ON vendor_section_quotes(rfx_id, vendor_id);
    """)
    # Idempotent: add granularity column to vendor_extractions if not present
    ve_cols = {r[1] for r in conn.execute("PRAGMA table_info(vendor_extractions)").fetchall()}
    if "granularity" not in ve_cols:
        conn.execute("ALTER TABLE vendor_extractions ADD COLUMN granularity TEXT DEFAULT 'line_item'")
    conn.commit()
    print("Schema created.")


LANDSCAPE_RFX_ID = "RFX-2026-LANDSCAPE-001"


def load_rfx_lines(conn: sqlite3.Connection, rfx_id: str = LANDSCAPE_RFX_ID):
    """
    Load rfx_lines from rfx_lines.json into the DB, scoped to rfx_id.
    Only deletes rows for this rfx_id — other RFxs (created via co-pilot) are untouched.
    """
    path = os.path.join(DATA_DIR, "rfx_lines.json")
    if not os.path.exists(path):
        print(f"  SKIP rfx_lines — {path} not found")
        return 0
    with open(path) as f:
        lines = json.load(f)
    cur = conn.cursor()
    # Scope delete to this RFx only — preserves rows from other projects
    cur.execute("DELETE FROM rfx_lines WHERE rfx_id = ?", (rfx_id,))
    for row in lines:
        cur.execute("""
            INSERT OR REPLACE INTO rfx_lines
            (rfx_id, line_id, section, description, species_name, unit, quantity,
             boq_unit_rate, boq_total, spec_notes, gpt_discrepancy)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            rfx_id, row["line_id"], row["section"], row["description"],
            row.get("species_name"), row.get("unit") or "", row.get("quantity"),
            row.get("boq_unit_rate"), row.get("boq_total"),
            row.get("spec_notes"), row.get("gpt_discrepancy")
        ))
    conn.commit()
    print(f"  Loaded {len(lines)} rfx_lines for {rfx_id}.")
    return len(lines)


def _parse_spec_notes(notes: str, section: str) -> dict:
    """
    Parse free-text spec_notes into a structured JSON dict.
    Keys: height_low/high, spread_low/high, caliper_low/high_mm,
          trunk_dia_low/high_mm, stem_dia_low/high_mm, culm_dia_low/high_mm,
          pot_dia_mm, clump_spread_low/high.
    """
    if not notes:
        return {}
    d = {}

    # Height range: "3.5-4.0m ht" or "1.8m - 2.4m ht"
    m = re.search(r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*m\s+ht', notes, re.IGNORECASE)
    if m:
        d['height_low'], d['height_high'] = float(m.group(1)), float(m.group(2))
    else:
        # Single value: "0.75m ht"
        m = re.search(r'(\d+\.?\d*)\s*m\s+ht', notes, re.IGNORECASE)
        if m:
            d['height_low'] = d['height_high'] = float(m.group(1))

    # Ground covers: leading height range without "ht" keyword ("0.3m - 0.35m,")
    if 'height_low' not in d and section == 'ground_covers':
        m = re.search(r'(\d+\.?\d*)\s*m\s*[-–]\s*(\d+\.?\d*)\s*m', notes)
        if m:
            d['height_low'], d['height_high'] = float(m.group(1)), float(m.group(2))

    # Spread: second dimension after "ht," for trees/shrubs
    if section in ('trees', 'shrubs'):
        m = re.search(r'ht\s*,\s*(\d+\.?\d*)\s*[-–]?\s*(\d+\.?\d*)?\s*m', notes, re.IGNORECASE)
        if m and m.group(2):
            d['spread_low'], d['spread_high'] = float(m.group(1)), float(m.group(2))
        elif m:
            d['spread_low'] = d['spread_high'] = float(m.group(1))

    # Clump spread (bamboo): "0.8–1.0 m clump spread"
    m = re.search(r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)\s*m\s+clump\s+spread', notes, re.IGNORECASE)
    if m:
        d['clump_spread_low'], d['clump_spread_high'] = float(m.group(1)), float(m.group(2))

    # Caliper: "75–90 mm min. caliper"
    m = re.search(r'(\d+)\s*[-–]\s*(\d+)\s*mm\s+min\.?\s+caliper', notes, re.IGNORECASE)
    if m:
        d['caliper_low_mm'], d['caliper_high_mm'] = int(m.group(1)), int(m.group(2))

    # Trunk dia: "300-350 mm trunk dia"
    m = re.search(r'(\d+)\s*[-–]\s*(\d+)\s*mm\s+trunk\s+dia', notes, re.IGNORECASE)
    if m:
        d['trunk_dia_low_mm'], d['trunk_dia_high_mm'] = int(m.group(1)), int(m.group(2))

    # Stem dia: "50–60 mm stem dia"
    m = re.search(r'(\d+)\s*[-–]\s*(\d+)\s*mm\s+stem\s+dia', notes, re.IGNORECASE)
    if m:
        d['stem_dia_low_mm'], d['stem_dia_high_mm'] = int(m.group(1)), int(m.group(2))

    # Culm dia (bamboo): "12–20 mm culm dia"
    m = re.search(r'(\d+)\s*[-–]\s*(\d+)\s*mm\s+culm\s+dia', notes, re.IGNORECASE)
    if m:
        d['culm_dia_low_mm'], d['culm_dia_high_mm'] = int(m.group(1)), int(m.group(2))

    # Pot dia: "200mm dia Pots"
    m = re.search(r'(\d+)\s*mm\s+dia\s+[Pp]ot', notes)
    if m:
        d['pot_dia_mm'] = int(m.group(1))

    return d


def populate_spec_parsed(conn: sqlite3.Connection):
    """Parse spec_notes into structured JSON and write to spec_parsed column."""
    cur = conn.cursor()
    rows = cur.execute(
        "SELECT line_id, section, spec_notes FROM rfx_lines WHERE spec_notes IS NOT NULL"
    ).fetchall()
    updated = 0
    for line_id, section, spec_notes in rows:
        parsed = _parse_spec_notes(spec_notes, section)
        if parsed:
            cur.execute(
                "UPDATE rfx_lines SET spec_parsed = ? WHERE line_id = ?",
                (json.dumps(parsed), line_id),
            )
            updated += 1
    conn.commit()
    print(f"  Populated spec_parsed for {updated} rfx_lines.")
    return updated


def load_questionnaire_definitions(conn: sqlite3.Connection):
    path = os.path.join(DATA_DIR, "questionnaire_definitions.json")
    if not os.path.exists(path):
        print(f"  SKIP questionnaire_definitions — {path} not found")
        return 0
    with open(path) as f:
        defs = json.load(f)
    cur = conn.cursor()
    cur.execute("DELETE FROM questionnaire_definitions")
    for row in defs:
        cur.execute("""
            INSERT OR REPLACE INTO questionnaire_definitions (q_id, question, pass_criteria)
            VALUES (?,?,?)
        """, (row["q_id"], row["question"], row["pass_criteria"]))
    conn.commit()
    print(f"  Loaded {len(defs)} questionnaire_definitions.")
    return len(defs)


def load_vendor_section_quotes(conn: sqlite3.Connection):
    """Load vendor_section_quotes from JSON files into the vendor_section_quotes table.
    Delete-on-source_file: v1 verbal record survives independently when v2 is loaded.
    """
    vendor_dir = os.path.join(DATA_DIR, "vendor_raw")
    files = sorted(glob.glob(os.path.join(vendor_dir, "**", "*.json"), recursive=True))
    cur = conn.cursor()
    total = 0

    for fpath in files:
        with open(fpath) as f:
            vendor = json.load(f)

        section_quotes = vendor.get("section_quotes", [])
        if not section_quotes:
            continue

        vendor_id = vendor["vendor_id"]
        source_file = vendor.get("source_file", "")
        cur.execute(
            "DELETE FROM vendor_section_quotes WHERE vendor_id = ? AND source_file = ?",
            (vendor_id, source_file)
        )

        for sq in section_quotes:
            flags = sq.get("flags", [])
            cur.execute("""
                INSERT INTO vendor_section_quotes
                (rfx_id, vendor_id, vendor_name, source_file, document_version, superseded_by,
                 revision_type, section, total_amount, estimate_low, estimate_high,
                 currency, value_source, package_description, source_snippet,
                 extraction_confidence, flags, extracted_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                vendor.get("rfx_id"), vendor_id, vendor.get("vendor_name"),
                source_file, vendor.get("document_version"),
                vendor.get("superseded_by"),
                vendor.get("revision_type", "quote"),
                sq["section"], sq.get("total_amount"), sq.get("estimate_low"),
                sq.get("estimate_high"), sq.get("currency", "INR"),
                sq.get("value_source", "vendor_quote"),
                sq.get("package_description"),
                sq.get("source_snippet"), sq.get("extraction_confidence"),
                json.dumps(flags) if isinstance(flags, list) else flags,
                vendor.get("extracted_at"),
            ))
            total += 1

    conn.commit()
    print(f"  Loaded {total} vendor_section_quotes rows.")
    return total


def load_vendor_extractions(conn: sqlite3.Connection):
    vendor_dir = os.path.join(DATA_DIR, "vendor_raw")
    files = sorted(glob.glob(os.path.join(vendor_dir, "**", "*.json"), recursive=True))
    cur = conn.cursor()
    total_extractions = 0
    total_q = 0

    for fpath in files:
        with open(fpath) as f:
            vendor = json.load(f)

        vendor_id = vendor["vendor_id"]
        revision_type = vendor.get("revision_type", "quote")
        source_file = vendor.get("source_file", "")

        # questionnaire_update docs: only update questionnaire_responses for matched q_ids;
        # never delete or insert vendor_extractions for these documents.
        if revision_type == "questionnaire_update":
            for qr in vendor.get("questionnaire_responses", []):
                q_id = qr.get("q_id")
                if q_id:
                    # Replace any existing response for this vendor+q_id (from any source)
                    cur.execute(
                        "DELETE FROM questionnaire_responses WHERE vendor_id = ? AND q_id = ?",
                        (vendor_id, q_id)
                    )
                cur.execute("""
                    INSERT INTO questionnaire_responses
                    (vendor_id, source_file, document_version, q_id, question,
                     answer, answer_type, passes, confidence, source_location, value_source)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    vendor_id, source_file, vendor.get("document_version"),
                    q_id, qr.get("question"), qr.get("answer"),
                    qr.get("answer_type", "text"), qr.get("passes"),
                    qr.get("confidence"), qr.get("source_location"),
                    qr.get("value_source", "extracted"),
                ))
                total_q += 1
            print(f"  {os.path.basename(fpath)}: questionnaire_update — {len(vendor.get('questionnaire_responses', []))} Q responses updated")
            continue

        cur.execute("DELETE FROM vendor_extractions WHERE vendor_id = ? AND source_file = ?",
                    (vendor_id, source_file))
        cur.execute("DELETE FROM questionnaire_responses WHERE vendor_id = ? AND source_file = ?",
                    (vendor_id, source_file))

        for ex in vendor.get("extractions", []):
            flags = ex.get("flags", [])
            cur.execute("""
                INSERT INTO vendor_extractions
                (rfx_id, vendor_id, vendor_name, source_file, document_version, superseded_by,
                 line_id, matched, match_confidence, raw_unit_price, raw_unit, raw_currency,
                 normalized_unit_price, normalization_note, quantity_quoted,
                 freight_included, labor_included, spec_grade_quoted, spec_grade_match,
                 source_snippet, source_location, extraction_confidence, value_source, flags,
                 extracted_at, granularity)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                vendor.get("rfx_id"), vendor_id, vendor.get("vendor_name"),
                source_file, vendor.get("document_version"),
                vendor.get("superseded_by"),
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
                vendor.get("extracted_at"),
                ex.get("granularity", "line_item"),
            ))
            total_extractions += 1

        for qr in vendor.get("questionnaire_responses", []):
            cur.execute("""
                INSERT INTO questionnaire_responses
                (vendor_id, source_file, document_version, q_id, question,
                 answer, answer_type, passes, confidence, source_location, value_source)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
            """, (
                vendor_id, vendor.get("source_file"), vendor.get("document_version"),
                qr.get("q_id"), qr.get("question"), qr.get("answer"),
                qr.get("answer_type", "text"),
                qr.get("passes"),  # None for ambiguous/unknown
                qr.get("confidence"), qr.get("source_location"),
                qr.get("value_source", "extracted")
            ))
            total_q += 1

        print(f"  {os.path.basename(fpath)}: {len(vendor.get('extractions',[]))} extractions, "
              f"{len(vendor.get('questionnaire_responses',[]))} Q responses")

    conn.commit()
    print(f"  Total: {total_extractions} extractions, {total_q} questionnaire rows")
    return total_extractions


def main():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    print("=== build_db: creating schema ===")
    create_schema(conn)
    print("=== loading rfx_lines ===")
    load_rfx_lines(conn)
    print("=== parsing spec_notes → spec_parsed ===")
    populate_spec_parsed(conn)
    print("=== loading questionnaire_definitions ===")
    load_questionnaire_definitions(conn)
    print("=== loading vendor extractions ===")
    load_vendor_extractions(conn)
    print("=== loading vendor section quotes ===")
    load_vendor_section_quotes(conn)
    conn.close()
    print(f"\nDone. DB at: {DB_PATH}")


if __name__ == "__main__":
    main()

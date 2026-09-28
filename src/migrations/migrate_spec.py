"""
migrate_spec.py — One-time migration: add search_text and spec_parsed to rfx_lines.

Run once:
    python src/migrate_spec.py

    --dry-run   Print parsed JSON for each line without writing to DB.
    --preview N Show first N rows (default 5) and exit.

Idempotent: re-running will overwrite both columns with fresh values.
"""
from __future__ import annotations
import argparse
import json
import os
import sqlite3

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", "config", ".env"))

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "comparison.db")
MODEL   = "gpt-4o-mini"   # 69 simple extractions; mini is sufficient and cheap

# All possible numeric fields — only fields that actually apply are populated,
# the rest are null. Caliper/trunk/culm stored as minimum spec (lower bound of range).
SPEC_FIELDS = [
    "height_low",       # m — lower bound of height range
    "height_high",      # m — upper bound of height range
    "spread_low",       # m — canopy/clump spread lower bound
    "spread_high",      # m — canopy/clump spread upper bound
    "caliper_mm",       # mm — minimum trunk caliper (lower bound of range)
    "trunk_dia_mm",     # mm — palm trunk diameter (lower bound)
    "culm_dia_mm",      # mm — bamboo culm diameter (lower bound)
    "pot_dia_mm",       # mm — pot diameter (ground covers)
    "pit_dia_m",        # m  — planting pit diameter (soil prep)
    "pit_depth_m",      # m  — planting pit depth (soil prep)
    "depth_mm",         # mm — generic layer depth
    "thickness_mm",     # mm — material thickness
    "length_m",         # m  — material length
    "area_sqm",         # m² — area-based material quantity
]

SYSTEM_PROMPT = """\
You are a landscaping BOQ (Bill of Quantities) parser. Given a line item's section,
description, and spec_notes, extract numeric specification values into a flat JSON object.

Rules:
- Only include fields that actually apply to this item. Omit all others (null/absent).
- For ranges (e.g. "3.5-4.0m ht"), set the _low field to the lower bound and _high to the upper.
- For single values (e.g. "0.75m ht"), set both _low and _high to the same value.
- For mm ranges (e.g. "75-90mm caliper"), store the LOWER bound in the _mm field.
- Heights and spreads in metres. Caliper/dia in millimetres.
- Ground covers: the "Xm" dimension is height (height_low/high); "200mm dia Pots" → pot_dia_mm=200.
- Shrubs: "X.Xm ht, X.Xm" → height and spread.
- Trees with "trunk dia": populate trunk_dia_mm, not caliper_mm.
- Bamboo: "clump spread" → spread_low/high; "culm dia" → culm_dia_mm.
- Soil prep: pit dimensions are in the description ("0.9m dia and 0.9m deep") →
  pit_dia_m and pit_depth_m. Depth-based materials → depth_mm. Area-based → area_sqm.
- Lawn / activity-only lines (no numeric specs): return {}.
- Return ONLY valid JSON with the relevant fields as numeric values (no units, no strings).

Available fields: height_low, height_high, spread_low, spread_high, caliper_mm,
trunk_dia_mm, culm_dia_mm, pot_dia_mm, pit_dia_m, pit_depth_m, depth_mm,
thickness_mm, length_m, area_sqm.
"""


def _parse_spec(client: OpenAI, section: str, description: str, spec_notes: str | None) -> dict:
    user_msg = (
        f"section: {section}\n"
        f"description: {description}\n"
        f"spec_notes: {spec_notes or '(none)'}"
    )
    resp = client.chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0,
    )
    try:
        return json.loads(resp.choices[0].message.content)
    except (json.JSONDecodeError, AttributeError):
        return {}


def _build_search_text(description: str, species_name: str | None, spec_notes: str | None) -> str:
    parts = [p for p in [description, species_name, spec_notes] if p and p.strip()]
    return " | ".join(parts)


def run(dry_run: bool = False, preview: int | None = None) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    # Add columns if they don't exist yet
    if not dry_run:
        for col, typ in [("search_text", "TEXT"), ("spec_parsed", "TEXT")]:
            try:
                conn.execute(f"ALTER TABLE rfx_lines ADD COLUMN {col} {typ}")
                conn.commit()
                print(f"Added column: {col}")
            except sqlite3.OperationalError:
                pass  # column already exists

    rows = conn.execute(
        "SELECT line_id, section, description, species_name, spec_notes FROM rfx_lines ORDER BY line_id"
    ).fetchall()

    if preview is not None:
        rows = rows[:preview]

    updates: list[tuple] = []
    print(f"Processing {len(rows)} lines with {MODEL}...\n")

    for i, r in enumerate(rows, 1):
        line_id     = r["line_id"]
        section     = r["section"]
        description = r["description"] or ""
        species     = r["species_name"]
        spec_notes  = r["spec_notes"]

        search_text = _build_search_text(description, species, spec_notes)
        spec_parsed = _parse_spec(client, section, description, spec_notes)

        print(f"[{i:02}/{len(rows)}] {line_id} ({section})")
        print(f"  spec_notes : {spec_notes or '(none)'}")
        print(f"  search_text: {search_text[:90]}")
        print(f"  spec_parsed: {json.dumps(spec_parsed)}")
        print()

        updates.append((search_text, json.dumps(spec_parsed) if spec_parsed else None, line_id))

    if not dry_run:
        conn.executemany(
            "UPDATE rfx_lines SET search_text=?, spec_parsed=? WHERE line_id=?",
            updates,
        )
        conn.commit()
        print(f"Written {len(updates)} rows to DB.")
    else:
        print("(dry-run — nothing written)")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run",  action="store_true", help="Parse only, no DB writes")
    parser.add_argument("--preview",  type=int, default=None,
                        help="Only process first N lines (implies dry-run if not set)")
    args = parser.parse_args()
    dry = args.dry_run or (args.preview is not None and not args.dry_run is False)
    run(dry_run=args.dry_run, preview=args.preview)

"""
migrate_multi_rfx.py — One-time migration: add multi-RFx support.

Changes:
  1. Creates rfx_projects table (rfx registry).
  2. Adds rfx_id column to rfx_lines and backfills existing rows.

Safe to re-run: each step is idempotent.

Run:
    python src/migrate_multi_rfx.py
"""
from __future__ import annotations
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "comparison.db")

EXISTING_RFX_ID   = "RFX-2026-LANDSCAPE-001"
EXISTING_RFX_NAME = "Skydome Kukas Landscaping"
EXISTING_RFX_DESC = "Jaipur landscaping project — 4 vendors, 69 BOQ lines"


def run() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row

    # 1 ── Create rfx_projects if it doesn't exist
    conn.execute("""
        CREATE TABLE IF NOT EXISTS rfx_projects (
            rfx_id      TEXT PRIMARY KEY,
            name        TEXT NOT NULL,
            description TEXT,
            created_at  TEXT NOT NULL
        )
    """)
    conn.commit()
    print("rfx_projects table: ready")

    # Insert the existing project (ignore if already present)
    conn.execute("""
        INSERT OR IGNORE INTO rfx_projects (rfx_id, name, description, created_at)
        VALUES (?, ?, ?, ?)
    """, (EXISTING_RFX_ID, EXISTING_RFX_NAME, EXISTING_RFX_DESC,
          datetime.now(timezone.utc).isoformat()))
    conn.commit()
    print(f"rfx_projects: inserted/verified '{EXISTING_RFX_ID}'")

    # 2 ── Add rfx_id to rfx_lines if missing
    cols = [r["name"] for r in conn.execute("PRAGMA table_info(rfx_lines)").fetchall()]
    if "rfx_id" not in cols:
        conn.execute("ALTER TABLE rfx_lines ADD COLUMN rfx_id TEXT")
        conn.commit()
        print("rfx_lines: added rfx_id column")
    else:
        print("rfx_lines: rfx_id column already present")

    # 3 ── Backfill NULL rfx_id rows
    n = conn.execute(
        "UPDATE rfx_lines SET rfx_id=? WHERE rfx_id IS NULL", (EXISTING_RFX_ID,)
    ).rowcount
    conn.commit()
    print(f"rfx_lines: backfilled {n} rows with '{EXISTING_RFX_ID}'")

    # Verify
    total = conn.execute("SELECT COUNT(*) FROM rfx_lines").fetchone()[0]
    scoped = conn.execute(
        "SELECT COUNT(*) FROM rfx_lines WHERE rfx_id=?", (EXISTING_RFX_ID,)
    ).fetchone()[0]
    print(f"rfx_lines: {scoped}/{total} rows scoped to '{EXISTING_RFX_ID}'")

    conn.close()
    print("\nMigration complete.")


if __name__ == "__main__":
    run()

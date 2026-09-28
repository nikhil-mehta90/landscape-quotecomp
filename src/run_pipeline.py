"""
run_pipeline.py — Orchestrates the full extraction pipeline.

DEFAULT (no args): runs the full landscape pipeline, exactly as before.
  python src/run_pipeline.py

SINGLE-FILE MODE: extract one new vendor document into an existing RFx.
  python src/run_pipeline.py --file path/to/quote.pdf \
      --rfx-id RFX-2026-LANDSCAPE-001 \
      --vendor-id V_NEW --vendor-name "New Vendor"

FILE ROUTING MODE: drop a file and let the pipeline infer rfx_id/vendor_id.
  python src/run_pipeline.py --route path/to/quote.pdf
  - If the file's parent directory or name clearly maps to a known RFx/vendor,
    extraction runs automatically.
  - Otherwise it is appended to data/review_queue.json for human triage.

NOTE on routing strategy: simple pattern matching (folder name → rfx_id,
filename stem → vendor_id) is sufficient for files dropped into named project
folders. For ambiguous cases — files with generic names like "quote_v2.pdf"
arriving without folder context (e.g. via email/API) — pattern matching
cannot reliably route without risking silent misattribution. Those cases
deserve LLM-assisted routing (show the model the filename + first-page text
and ask it to match against rfx_projects + known vendor names). That is NOT
implemented here; ambiguous files go to the review queue instead. Flag this
if the volume of ambiguous intake grows.
"""
from __future__ import annotations
import json
import os
import re
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone

SRC = os.path.dirname(__file__)
DATA_DIR = os.path.join(SRC, "..", "data")
DB_PATH = os.path.join(DATA_DIR, "comparison.db")
REVIEW_QUEUE = os.path.join(DATA_DIR, "review_queue.json")

LANDSCAPE_RFX_ID = "RFX-2026-LANDSCAPE-001"

# ── Full-pipeline runner (existing behaviour) ─────────────────────────────────

def run_script(script: str, extra_args: list[str] | None = None):
    print(f"\n{'='*60}")
    print(f"RUNNING: {script}")
    print('='*60)
    cmd = [sys.executable, os.path.join(SRC, script)] + (extra_args or [])
    result = subprocess.run(cmd, cwd=os.path.join(SRC, ".."))
    if result.returncode != 0:
        print(f"\nERROR: {script} exited with code {result.returncode}")
        sys.exit(result.returncode)


def run_full_pipeline():
    """Run the complete landscape pipeline — rfx_builder → extract → normalize → build_db."""
    run_script("rfx_builder.py")
    run_script("extract.py")
    run_script("normalize.py")
    run_script("build_db.py")
    print("\n✓ Pipeline complete. DB ready at data/comparison.db")


# ── File routing ──────────────────────────────────────────────────────────────

def _load_rfx_projects() -> list[dict]:
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT rfx_id, name FROM rfx_projects").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _load_known_vendors(rfx_id: str) -> list[dict]:
    """Return distinct vendor_id/vendor_name pairs from vendor_extractions for this RFx."""
    if not os.path.exists(DB_PATH):
        return []
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT DISTINCT vendor_id, vendor_name FROM vendor_extractions WHERE rfx_id = ?",
        (rfx_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def route_file(file_path: str) -> dict | None:
    """
    Attempt to infer (rfx_id, vendor_id, vendor_name) from folder path + filename.

    Returns a dict with keys rfx_id, vendor_id, vendor_name if routing succeeds,
    or None if the file is ambiguous (caller should enqueue for review).

    Strategy:
      1. Match parent directory name against known rfx_projects names/ids (slug compare).
      2. Match filename stem against known vendor names for the resolved RFx.
      3. Both must match for auto-route; otherwise → review queue.
    """
    abs_path = os.path.abspath(file_path)
    parent_dir = os.path.basename(os.path.dirname(abs_path))
    filename_stem = os.path.splitext(os.path.basename(abs_path))[0]

    projects = _load_rfx_projects()
    if not projects:
        return None

    # Step 1: match parent dir to an RFx
    resolved_rfx_id = None
    for proj in projects:
        if _slug(proj["rfx_id"]) in _slug(parent_dir) or _slug(proj["name"]) in _slug(parent_dir):
            resolved_rfx_id = proj["rfx_id"]
            break

    if not resolved_rfx_id:
        return None

    # Step 2: match filename to a vendor within that RFx
    vendors = _load_known_vendors(resolved_rfx_id)
    for v in vendors:
        if _slug(v["vendor_name"] or "") in _slug(filename_stem) or \
           _slug(v["vendor_id"] or "") in _slug(filename_stem):
            return {
                "rfx_id": resolved_rfx_id,
                "vendor_id": v["vendor_id"],
                "vendor_name": v["vendor_name"],
            }

    return None


def enqueue_for_review(file_path: str, reason: str):
    queue = []
    if os.path.exists(REVIEW_QUEUE):
        with open(REVIEW_QUEUE) as f:
            try:
                queue = json.load(f)
            except json.JSONDecodeError:
                queue = []
    queue.append({
        "file": os.path.abspath(file_path),
        "reason": reason,
        "queued_at": datetime.now(timezone.utc).isoformat(),
    })
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(REVIEW_QUEUE, "w") as f:
        json.dump(queue, f, indent=2)
    print(f"  → Added to review queue: {REVIEW_QUEUE}")
    print(f"    Reason: {reason}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Aerchain extraction pipeline")
    parser.add_argument("--file", help="Vendor document to extract (requires --rfx-id, --vendor-id, --vendor-name)")
    parser.add_argument("--route", help="Vendor document to auto-route by folder/filename")
    parser.add_argument("--rfx-id", default=LANDSCAPE_RFX_ID, help="RFx project ID")
    parser.add_argument("--vendor-id", help="Vendor ID (e.g. V_NEW)")
    parser.add_argument("--vendor-name", help="Vendor display name")
    parser.add_argument("--version", default="v1", help="Document version (default: v1)")
    parser.add_argument("--output-name", help="Output JSON basename")
    args = parser.parse_args()

    if args.route:
        # Auto-routing mode
        routed = route_file(args.route)
        if routed:
            print(f"  Routed: {args.route}")
            print(f"    rfx_id={routed['rfx_id']}  vendor_id={routed['vendor_id']}  vendor_name={routed['vendor_name']}")
            run_script("extract.py", [
                "--file", args.route,
                "--rfx-id", routed["rfx_id"],
                "--vendor-id", routed["vendor_id"],
                "--vendor-name", routed["vendor_name"],
            ])
            run_script("normalize.py")
            run_script("build_db.py")
        else:
            enqueue_for_review(
                args.route,
                "Could not infer rfx_id or vendor_id from folder/filename. "
                "Provide --rfx-id, --vendor-id and --vendor-name explicitly, or resolve manually.",
            )

    elif args.file:
        # Explicit single-file mode
        if not args.vendor_id or not args.vendor_name:
            parser.error("--file requires --vendor-id and --vendor-name")
        run_script("extract.py", [
            "--file", args.file,
            "--rfx-id", args.rfx_id,
            "--vendor-id", args.vendor_id,
            "--vendor-name", args.vendor_name,
            "--version", args.version,
        ] + (["--output-name", args.output_name] if args.output_name else []))
        run_script("normalize.py")
        run_script("build_db.py")

    else:
        # Default: full landscape pipeline
        run_full_pipeline()

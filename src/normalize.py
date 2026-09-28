"""
normalize.py — Unit normalization and confidence scoring.
Reads vendor_raw/*.json, applies rules, updates records in-place,
then loads everything into comparison.db via build_db.
"""
from __future__ import annotations
import json
import os
import glob
import re
import sys
from typing import Optional
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.70"))
VERY_LOW_THRESHOLD = 0.40


# ── Unit normalization rules ──────────────────────────────────────────────────

# Canonical unit aliases: maps vendor variants to canonical
UNIT_ALIASES = {
    # Sqm variants — with and without trailing period/space
    "sqm": "Sqm", "sqm.": "Sqm", "sq.m": "Sqm", "sq.m.": "Sqm",
    "sq m": "Sqm", "sq m.": "Sqm", "square meter": "Sqm",
    "sq mtr": "Sqm", "sqmtr": "Sqm", "sqmtr.": "Sqm",
    "sqmt": "Sqm", "sqmt.": "Sqm",
    # Nos variants — with and without trailing period
    "nos": "Nos", "nos.": "Nos", "no.": "Nos", "no": "Nos",
    "numbers": "Nos", "number": "Nos", "nos,": "Nos",
    "each": "Each", "each.": "Each", "ea": "Each", "ea.": "Each",
    "pcs": "Nos", "pcs.": "Nos", "plants": "Nos",
    # Volume — with and without period
    "cum": "Cum", "cum.": "Cum", "cu.m": "Cum", "cu.m.": "Cum",
    "cu m": "Cum", "m3": "Cum", "m³": "Cum",
    "cft": "Cft", "cft.": "Cft",  # cubic feet — flag for conversion
    # Weight
    "kg": "Kg", "kg.": "Kg", "kgs": "Kg", "kgs.": "Kg", "kilogram": "Kg",
    # Running meter
    "rmt": "Rmt", "rmt.": "Rmt", "rm": "Rmt", "rm.": "Rmt",
    "rmt,": "Rmt", "r.mt.": "Rmt",
    # Lump
    "ls": "Lump", "l/s": "Lump", "lump sum": "Lump", "l.s.": "Lump",
    # Blank / dash / unknown
    "-": "UNKNOWN", "": "UNKNOWN", "nan": "UNKNOWN",
}

# Conversion factors to canonical unit (where defensible)
CONVERSIONS = {
    # Cft → Cum (1 Cft = 0.0283168 Cum)
    ("Cft", "Cum"): 0.0283168,
}


def canonicalize_unit(raw_unit: Optional[str]) -> tuple:
    """
    Returns (canonical_unit, conversion_note).
    conversion_note is non-None only when an actual numeric conversion was applied.
    """
    if not raw_unit:
        return "UNKNOWN", None
    normalized = raw_unit.strip().lower().rstrip(".")
    canonical = UNIT_ALIASES.get(normalized, raw_unit.strip())
    return canonical, None


def apply_unit_conversion(
    raw_price: Optional[float],
    raw_unit: str,
    canonical_unit: str
) -> tuple:
    """
    Returns (normalized_price, note, confidence_penalty).
    confidence_penalty is subtracted from extraction_confidence.
    """
    if raw_price is None:
        return None, None, 0.0

    can_raw, _ = canonicalize_unit(raw_unit)

    if can_raw == canonical_unit:
        return raw_price, None, 0.0

    key = (can_raw, canonical_unit)
    if key in CONVERSIONS:
        factor = CONVERSIONS[key]
        note = f"Converted {raw_unit} → {canonical_unit} (factor {factor})"
        return raw_price * factor, note, 0.10  # small penalty for inference

    # Incompatible units (e.g. Sqm vs Nos) — cannot normalize
    if can_raw in ("Sqm", "Rmt", "Cum") and canonical_unit in ("Nos", "Each"):
        return None, f"Cannot normalize {raw_unit} → {canonical_unit}: incompatible dimensions", 0.30
    if can_raw in ("Nos", "Each") and canonical_unit in ("Sqm", "Rmt", "Cum"):
        return None, f"Cannot normalize {raw_unit} → {canonical_unit}: incompatible dimensions", 0.30

    # Same conceptual unit, different notation — no conversion needed
    return raw_price, f"Unit alias: {raw_unit} treated as {canonical_unit}", 0.0


# ── Flag computation ──────────────────────────────────────────────────────────

_AUTO_FLAG_PREFIXES = (
    "unit_mismatch:",      # recomputed each run
    "low_extraction_",     # recomputed each run
    "low_match_",          # recomputed each run
    "very_low_",           # recomputed each run
    "exclude_from_totals", # recomputed each run
)

def compute_flags(ex: dict, rfx_line: dict | None) -> list[str]:
    # Carry forward only LLM-set semantic flags; strip auto-generated ones (recomputed below)
    flags = [
        f for f in (ex.get("flags") or [])
        if not any(f.startswith(p) for p in _AUTO_FLAG_PREFIXES)
    ]

    # Confidence flags
    ec = ex.get("extraction_confidence", 1.0) or 0.0
    mc = ex.get("match_confidence", 1.0) or 1.0
    if ec < VERY_LOW_THRESHOLD:
        if "very_low_extraction_confidence" not in flags:
            flags.append("very_low_extraction_confidence")
    elif ec < CONFIDENCE_THRESHOLD:
        if "low_extraction_confidence" not in flags:
            flags.append("low_extraction_confidence")
    if mc < CONFIDENCE_THRESHOLD:
        if "low_match_confidence" not in flags:
            flags.append("low_match_confidence")

    # Unit mismatch vs RFx canonical unit — compare canonical forms only
    if rfx_line and ex.get("raw_unit"):
        can_raw, _ = canonicalize_unit(ex["raw_unit"])
        rfx_unit = (rfx_line.get("unit") or "").strip().rstrip(".")
        can_rfx, _ = canonicalize_unit(rfx_unit)
        if can_raw not in ("UNKNOWN",) and can_raw != can_rfx:
            if not any("unit_mismatch" in f for f in flags):
                flags.append(f"unit_mismatch:{can_raw}_vs_{can_rfx}")

    # value_source unknown → exclude from totals
    if ex.get("value_source") == "unknown":
        if "exclude_from_totals" not in flags:
            flags.append("exclude_from_totals")

    return flags


# ── Per-extraction normalization ──────────────────────────────────────────────

def normalize_extraction(ex: dict, rfx_line: dict | None) -> dict:
    ex = dict(ex)  # shallow copy

    raw_price = ex.get("raw_unit_price")
    raw_unit = ex.get("raw_unit") or ""
    canonical_unit = rfx_line["unit"] if rfx_line else raw_unit

    # Normalize unit
    norm_price, conv_note, confidence_penalty = apply_unit_conversion(
        raw_price, raw_unit, canonical_unit
    )

    # Only update normalized_unit_price if we don't already have one (LLM may have set it)
    if ex.get("normalized_unit_price") is None:
        ex["normalized_unit_price"] = norm_price

    # Append to normalization_note
    existing_note = ex.get("normalization_note") or ""
    if conv_note:
        ex["normalization_note"] = (existing_note + "; " + conv_note).lstrip("; ")

    # Apply confidence penalty
    ec = (ex.get("extraction_confidence") or 1.0) - confidence_penalty
    ex["extraction_confidence"] = max(0.0, round(ec, 3))

    # If confidence < very_low and no normalization possible → force unknown
    if ex["extraction_confidence"] < VERY_LOW_THRESHOLD and ex.get("value_source") != "unknown":
        ex["value_source"] = "unknown"
        ex["normalized_unit_price"] = None

    # Compute final flags
    ex["flags"] = compute_flags(ex, rfx_line)

    return ex


# ── Per-vendor normalization ──────────────────────────────────────────────────

def load_rfx_lines() -> dict:
    path = os.path.join(DATA_DIR, "rfx_lines.json")
    with open(path) as f:
        lines = json.load(f)
    return {l["line_id"]: l for l in lines}


def normalize_vendor_file(fpath: str, rfx_index: dict) -> dict:
    with open(fpath) as f:
        vendor = json.load(f)

    updated_extractions = []
    flag_summary = {}

    for ex in vendor.get("extractions", []):
        rfx_line = rfx_index.get(ex.get("line_id"))
        normalized = normalize_extraction(ex, rfx_line)
        updated_extractions.append(normalized)

        for flag in normalized.get("flags", []):
            flag_summary[flag] = flag_summary.get(flag, 0) + 1

    vendor["extractions"] = updated_extractions

    # Write back
    with open(fpath, "w") as f:
        json.dump(vendor, f, indent=2, ensure_ascii=False)

    return flag_summary


# ── Confidence summary ────────────────────────────────────────────────────────

def print_confidence_summary(vendor_files: list, rfx_index: dict):
    print("\n=== Confidence & flag summary ===")
    for fpath in vendor_files:
        with open(fpath) as f:
            vendor = json.load(f)
        extractions = vendor.get("extractions", [])
        if not extractions:
            continue
        low = [e for e in extractions if (e.get("extraction_confidence") or 1.0) < CONFIDENCE_THRESHOLD]
        unknown = [e for e in extractions if e.get("value_source") == "unknown"]
        flagged = [e for e in extractions if e.get("flags")]
        all_flags = []
        for e in extractions:
            all_flags.extend(e.get("flags", []))
        flag_counts = {}
        for f in all_flags:
            flag_counts[f] = flag_counts.get(f, 0) + 1

        vendor_id = vendor.get("vendor_id", "?")
        src = vendor.get("source_file", "?")
        print(f"\n  {vendor_id} / {src} ({len(extractions)} lines):")
        print(f"    low confidence (<{CONFIDENCE_THRESHOLD}): {len(low)}")
        print(f"    value_source=unknown: {len(unknown)}")
        print(f"    lines with flags: {len(flagged)}")
        for flag, count in sorted(flag_counts.items(), key=lambda x: -x[1])[:8]:
            print(f"      {flag}: {count}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading rfx_lines...")
    rfx_index = load_rfx_lines()

    vendor_dir = os.path.join(DATA_DIR, "vendor_raw")
    vendor_files = sorted(glob.glob(os.path.join(vendor_dir, "**", "*.json"), recursive=True))

    if not vendor_files:
        print("No vendor JSON files found in data/vendor_raw/. Run extract.py first.")
        sys.exit(1)

    print(f"Normalizing {len(vendor_files)} vendor files...")
    for fpath in vendor_files:
        flag_summary = normalize_vendor_file(fpath, rfx_index)
        fname = os.path.basename(fpath)
        top_flags = sorted(flag_summary.items(), key=lambda x: -x[1])[:3]
        print(f"  {fname}: {dict(top_flags)}")

    print_confidence_summary(vendor_files, rfx_index)
    print("\nnormalize.py done.")


if __name__ == "__main__":
    main()

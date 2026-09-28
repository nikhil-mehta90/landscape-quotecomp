"""
anomaly.py — Two-stage anomaly detection for vendor price deviations.

Stage 1 (local, cheap) — five separate flag types:
  1. anomaly_iqr_outlier      — price outside Q1-1.5×IQR / Q3+1.5×IQR across peer
                                 quotes for that line (requires 3+ extracted quotes).
                                 Self-calibrates per line: wide natural variation on
                                 plant lines won't over-flag; tight-spec material lines
                                 still catch real outliers.
  2. anomaly_peer_deviation   — price deviates >60% from the peer median (requires
                                 3+ extracted quotes). Catches cases IQR misses when
                                 the price distribution is very wide — e.g. a 30× spread
                                 makes Tukey fences so loose (e.g. [-115, 237]) that a
                                 price at 1/17 of the median passes. Anchors to peers,
                                 not to the BOQ, so it fires independently of whether
                                 the market has moved from the client reference rate.
  3. anomaly_decimal_mismatch — vendor's price is within 10% of exactly 10× or 100×
                                 another vendor's price on the same line. Signals a
                                 likely extraction unit/decimal error, NOT a pricing
                                 issue — different remediation from IQR outliers.
  4. anomaly_vs_boq           — price deviates beyond a dynamic threshold from the
                                 client's BOQ reference rate. Base threshold is 50%; if
                                 ≥50% of vendors on the same line all exceed 50% (market
                                 consensus has moved), threshold expands to
                                 50% + |peer_median − BOQ| / BOQ (capped at 200%) so
                                 only genuine per-vendor outliers are still flagged.
  5. anomaly_extreme_low      — price is below 20% of the BOQ rate AND the peer median
                                 is above 15% of the BOQ rate (i.e. the whole market
                                 hasn't moved low; this vendor is an outlier). Catches
                                 spec/grade mismatches the dynamic-threshold logic
                                 suppresses: e.g. BOQ ₹5000, peers [550, 750, 1700,
                                 5500] — consensus uplift raises the effective BOQ
                                 threshold to 125%, letting 550 and 750 pass Flag 4,
                                 but Flag 5 catches them as likely wrong-grade plants.

  All flagged records are ranked by ₹ impact = |price - peer_median| × boq_quantity.
  Only value_source='extracted' rows participate — inferred/estimated values are never
  included in deviation math.

Stage 2 (external, selective):
  Serper web-search benchmark for lines that are BOTH anomalous (Stage 1) AND in the
  top 70% of cumulative RFx value. Falls back to GPT-4o structured estimate when
  Serper returns no usable INR pricing, tagged value_source='gpt_estimate'.
"""
from __future__ import annotations
import json
import os
import sqlite3
import statistics
from typing import Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "config", ".env"))

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "comparison.db")
MODEL_ESTIMATE = "gpt-4o"

# Stage 1 — IQR outlier requires 3+ quotes per line
MIN_QUOTES_FOR_IQR = 3

# Stage 1 — peer-median deviation: flag if price deviates this much from the peer median.
# Catches outliers that IQR misses when the distribution is wide (e.g. 30× spread makes
# Tukey fences so loose that a price at 1/17 of the median passes). Requires 3+ quotes.
PEER_DEV_THRESHOLD = 0.60     # >60% from peer median (absolute, not relative to direction)

# Stage 1 — BOQ deviation threshold (separate signal, not merged with peer flags)
DEVIATION_VS_BOQ     = 0.50   # base threshold: >50% from boq_unit_rate
MAX_BOQ_THRESHOLD    = 2.00   # cap on dynamic threshold to avoid total suppression
BOQ_CONSENSUS_UPLIFT = 0.50   # fraction of vendors that must deviate to trigger dynamic mode

# Stage 1 — decimal mismatch: flag if ratio ≈ 10× or 100× within this tolerance
DECIMAL_MISMATCH_TOL = 0.10   # ±10% of the factor (e.g. ratio in [9,11] for 10×)

# Stage 1 — extreme low: flag if vendor quotes < this fraction of the BOQ rate.
# Even if the BOQ is 3× inflated, a vendor at 20% of BOQ is at ~60% of market,
# which is unlikely without a spec/grade mismatch (wrong plant age, size, or variety).
EXTREME_LOW_BOQ_RATIO  = 0.20  # vendor price < 20% of BOQ → flag
# Guard: suppress the flag when the whole market is low (BOQ just inflated), i.e. when
# peer_median ≤ 15% of BOQ. Only fires when peers are quoting at "reasonable" levels.
EXTREME_LOW_PEER_GUARD = 0.15  # peer_median must be > 15% of BOQ for flag to apply

# Stage 2 value threshold
HIGH_VALUE_PERCENTILE = 0.70  # top 70% of cumulative RFx value eligible for Stage 2

# Flag names — prefixed 'anomaly_' to distinguish from extraction-time flags
FLAG_IQR_OUTLIER      = "anomaly_iqr_outlier"       # outside IQR fences
FLAG_PEER_DEV         = "anomaly_peer_deviation"    # >60% from peer median (catches what IQR misses on wide distributions)
FLAG_DECIMAL_MISMATCH = "anomaly_decimal_mismatch"  # likely unit/decimal extraction error
FLAG_VS_BOQ           = "anomaly_vs_boq"            # exceeds dynamic BOQ threshold
FLAG_EXTREME_LOW      = "anomaly_extreme_low"        # <20% of BOQ with peers at reasonable levels — likely wrong grade/spec

# Old flag names from the previous fixed-threshold approach — cleared on re-run
_OLD_FLAGS = frozenset({
    "anomaly_price_vs_peer_median",
    "anomaly_price_2x_min_quoted",
    "anomaly_price_vs_boq",
})


# ── Helpers ───────────────────────────────────────────────────────────────────

def _db_conn(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


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


def _iqr_fences(prices: list[float]) -> tuple[float | None, float | None]:
    """
    Tukey IQR fences via linear interpolation.
    Returns (lower_fence, upper_fence) or (None, None) if fewer than 3 prices.
    """
    n = len(prices)
    if n < MIN_QUOTES_FOR_IQR:
        return None, None
    s = sorted(prices)

    def _pctile(p: float) -> float:
        idx = p / 100.0 * (n - 1)
        lo = int(idx)
        hi = min(lo + 1, n - 1)
        return s[lo] + (idx - lo) * (s[hi] - s[lo])

    q1 = _pctile(25)
    q3 = _pctile(75)
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def _is_decimal_mismatch(price_a: float, price_b: float, tol: float = DECIMAL_MISMATCH_TOL) -> bool:
    """True if price_a / price_b is within `tol` of 10× or 100×."""
    if price_b <= 0:
        return False
    ratio = price_a / price_b
    for factor in (10.0, 100.0):
        if abs(ratio - factor) / factor <= tol:
            return True
    return False


def _clear_anomaly_flags(conn: sqlite3.Connection) -> int:
    """
    Remove ALL anomaly_* flags (old and new) from vendor_extractions, preserving
    other extraction-time flags. Returns count of rows modified.
    """
    rows = conn.execute(
        "SELECT id, flags FROM vendor_extractions WHERE flags LIKE '%anomaly_%'"
    ).fetchall()
    updates = []
    for row in rows:
        cleaned = [f for f in _parse_flags(row["flags"]) if not f.startswith("anomaly_")]
        updates.append((json.dumps(cleaned), row["id"]))
    if updates:
        conn.executemany("UPDATE vendor_extractions SET flags=? WHERE id=?", updates)
        conn.commit()
    return len(updates)


# ── Stage 1 ───────────────────────────────────────────────────────────────────

def run_stage1(db_path: str = DB_PATH) -> list[dict]:
    """
    Self-calibrating per-line anomaly detection. Rewrites anomaly_* flags in
    vendor_extractions. Returns flagged records sorted descending by ₹ impact.

    Four separate flag types (never merged into one bucket):
      anomaly_iqr_outlier      — outside Tukey fences (3+ quotes required)
      anomaly_peer_deviation   — >60% from peer median (3+ quotes required; catches
                                  wide-distribution outliers that IQR misses)
      anomaly_decimal_mismatch — price ≈ 10× or 100× a peer's price
      anomaly_vs_boq           — >50% from client BOQ reference rate

    Re-running is safe: all anomaly_* flags are cleared first.
    """
    conn = _db_conn(db_path)

    # Step 0: clear all existing anomaly flags so re-runs are idempotent
    n_cleared = _clear_anomaly_flags(conn)
    print(f"Cleared anomaly flags from {n_cleared} rows.")

    # Step 1: fetch all current extracted rows with their IDs for precise updates.
    # Exclude section-total vendors: any vendor with granularity='section_total' across
    # all their rows for this RFx has no per-line prices to flag — skip them entirely.
    section_total_vendors = {
        r[0] for r in conn.execute("""
            SELECT DISTINCT vendor_id FROM vendor_extractions
            WHERE granularity = 'section_total'
              AND (superseded_by IS NULL OR superseded_by = '')
        """).fetchall()
    }
    if section_total_vendors:
        print(f"  Skipping per-line anomaly for section-total vendors: {section_total_vendors}")

    rows = conn.execute("""
        SELECT ve.id, ve.vendor_id, ve.line_id, ve.normalized_unit_price,
               ve.flags,
               rl.boq_unit_rate, rl.quantity, rl.description,
               rl.species_name, rl.unit
        FROM vendor_extractions ve
        JOIN rfx_lines rl ON ve.line_id = rl.line_id
        WHERE ve.value_source = 'extracted'
          AND ve.normalized_unit_price IS NOT NULL
          AND (ve.superseded_by IS NULL OR ve.superseded_by = '')
          AND (ve.granularity IS NULL OR ve.granularity != 'section_total')
        ORDER BY ve.line_id, ve.vendor_id
    """).fetchall()

    # Deduplicate: keep highest-id row per (vendor_id, line_id).
    # Guards against DB duplicates when build_db.py is run multiple times or source
    # files are renamed — superseded_by only models explicit version relationships.
    seen_vl: dict[tuple, dict] = {}
    for r in rows:
        key = (r["vendor_id"], r["line_id"])
        d = dict(r)
        if key not in seen_vl or d["id"] > seen_vl[key]["id"]:
            seen_vl[key] = d
    rows = list(seen_vl.values())

    # Group by line
    by_line: dict[str, list[dict]] = {}
    for r in rows:
        by_line.setdefault(r["line_id"], []).append(r)

    flagged: list[dict] = []
    updates: list[tuple] = []  # (new_flags_json, row_id)

    for line_id, vendors in by_line.items():
        n          = len(vendors)
        prices     = [v["normalized_unit_price"] for v in vendors]
        peer_med   = statistics.median(prices)
        boq_rate   = vendors[0]["boq_unit_rate"]
        quantity   = vendors[0]["quantity"] or 0
        description = vendors[0]["description"]
        species    = vendors[0]["species_name"]
        unit       = vendors[0]["unit"]

        lower_fence, upper_fence = _iqr_fences(prices)

        # ── Dynamic BOQ threshold ─────────────────────────────────────────────
        # If ≥50% of vendors already exceed the base 50% threshold, the BOQ is
        # likely stale (market has moved). Raise the threshold to
        #   base + |peer_median − BOQ| / BOQ
        # so only genuine per-vendor outliers (further from the market than from
        # BOQ alone) are still flagged. Cap at MAX_BOQ_THRESHOLD.
        if boq_rate and boq_rate > 0:
            n_deviating = sum(
                1 for p in prices
                if abs(p - boq_rate) / boq_rate > DEVIATION_VS_BOQ
            )
            market_dev = abs(peer_med - boq_rate) / boq_rate
            if n_deviating / n >= BOQ_CONSENSUS_UPLIFT:
                effective_boq_threshold = min(
                    DEVIATION_VS_BOQ + market_dev, MAX_BOQ_THRESHOLD
                )
            else:
                effective_boq_threshold = DEVIATION_VS_BOQ
        else:
            effective_boq_threshold = DEVIATION_VS_BOQ

        for v in vendors:
            price     = v["normalized_unit_price"]
            new_flags: list[str] = []

            # ── Flag 1: IQR outlier (3+ quotes only) ─────────────────────────
            if lower_fence is not None:
                if price < lower_fence or price > upper_fence:
                    new_flags.append(FLAG_IQR_OUTLIER)

            # ── Flag 1b: peer-median deviation (3+ quotes only) ───────────────
            # Catches outliers that IQR misses when the price distribution is very
            # wide (e.g. 30× spread → Tukey fences become [-115, 237], letting a
            # price of ₹3.8 pass even though it's 94% below the peer median).
            if lower_fence is not None and peer_med > 0:
                if abs(price - peer_med) / peer_med > PEER_DEV_THRESHOLD:
                    new_flags.append(FLAG_PEER_DEV)

            # ── Flag 2: decimal mismatch (check against all other vendors) ───
            for other in vendors:
                if other["vendor_id"] == v["vendor_id"]:
                    continue
                if (_is_decimal_mismatch(price, other["normalized_unit_price"]) or
                        _is_decimal_mismatch(other["normalized_unit_price"], price)):
                    if FLAG_DECIMAL_MISMATCH not in new_flags:
                        new_flags.append(FLAG_DECIMAL_MISMATCH)
                    break

            # ── Flag 3: vs BOQ (dynamic threshold — client reference, not peers)
            if boq_rate and boq_rate > 0:
                if abs(price - boq_rate) / boq_rate > effective_boq_threshold:
                    new_flags.append(FLAG_VS_BOQ)

            # ── Flag 4: extreme low vs BOQ ────────────────────────────────────
            # Catches spec/grade mismatches that the dynamic-threshold logic
            # suppresses when most peers are also below the BOQ (e.g. consensus
            # uplift raises the effective threshold above the vendor's deviation).
            # Only fires when the peer group itself isn't uniformly low (guard).
            if boq_rate and boq_rate > 0:
                if (price < boq_rate * EXTREME_LOW_BOQ_RATIO
                        and peer_med > boq_rate * EXTREME_LOW_PEER_GUARD):
                    if FLAG_EXTREME_LOW not in new_flags:
                        new_flags.append(FLAG_EXTREME_LOW)

            if new_flags:
                existing = _parse_flags(v["flags"])
                # Remove any stale old-format flags that slipped through
                existing = [f for f in existing if not f.startswith("anomaly_")]
                merged   = list(dict.fromkeys(existing + new_flags))
                updates.append((json.dumps(merged), v["id"]))
                impact   = abs(price - peer_med) * quantity

                flagged.append({
                    "line_id":            line_id,
                    "vendor_id":          v["vendor_id"],
                    "description":        description,
                    "species":            species,
                    "unit":               unit,
                    "price":              price,
                    "peer_median":        round(peer_med, 2),
                    "boq_rate":           boq_rate,
                    "boq_threshold_used": round(effective_boq_threshold, 3),
                    "n_quotes":           n,
                    "lower_fence":        round(lower_fence, 2) if lower_fence is not None else None,
                    "upper_fence":        round(upper_fence, 2) if upper_fence is not None else None,
                    "new_flags":          new_flags,
                    "impact_inr":         round(impact, 0),
                })

    conn.executemany(
        "UPDATE vendor_extractions SET flags=? WHERE id=?",
        updates,
    )
    conn.commit()
    conn.close()

    # Sort by ₹ impact descending — high-value signal first, noise last
    # Sort by ₹ impact descending — high-value signal first, noise last.
    # extreme_low records always sort before same-impact peers so they're
    # visible even when the absolute ₹ deviation is moderate.
    def _sort_key(r):
        has_extreme = FLAG_EXTREME_LOW in r.get("new_flags", [])
        return (not has_extreme, -r["impact_inr"])

    flagged.sort(key=_sort_key)
    return flagged


# ── Stage 2 helpers ───────────────────────────────────────────────────────────

def _high_value_line_ids(db_path: str = DB_PATH, percentile: float = HIGH_VALUE_PERCENTILE) -> set[str]:
    conn = _db_conn(db_path)
    rows = conn.execute("""
        SELECT line_id, boq_unit_rate * quantity AS line_value
        FROM rfx_lines
        WHERE boq_unit_rate IS NOT NULL AND quantity IS NOT NULL AND boq_unit_rate > 0
        ORDER BY line_value DESC
    """).fetchall()
    conn.close()

    total = sum(r["line_value"] for r in rows)
    cutoff = total * percentile
    cumsum = 0.0
    eligible: set[str] = set()
    for r in rows:
        cumsum += r["line_value"]
        eligible.add(r["line_id"])
        if cumsum >= cutoff:
            break
    return eligible


def _serper_benchmark(description: str, species: str, unit: str) -> str:
    api_key = os.environ.get("SERPER_API_KEY", "")
    if not api_key:
        return ""
    query = f"{species or description} price per {unit} nursery India INR landscaping"
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            json={"q": query, "num": 5, "gl": "in", "hl": "en"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        lines = [f"Search: «{query}»"]
        for r in data.get("organic", [])[:4]:
            lines.append(f"• {r.get('title','')} — {r.get('snippet','')}")
        return "\n".join(lines)
    except Exception as exc:
        return f"[Serper error: {exc}]"


def _has_inr_price(serper_text: str) -> bool:
    if not serper_text:
        return False
    markers = ["₹", "inr", "rs.", "rs ", "rupee", "/sqm", "/nos", "/kg", "/rmt"]
    lower = serper_text.lower()
    return any(m in lower for m in markers)


def _gpt_estimate(
    description: str,
    species: str,
    unit: str,
    spec_notes: str,
    peer_prices: list[float],
    boq_rate: float,
    serper_context: str = "",
) -> dict:
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    peer_str = ", ".join(f"₹{p:.0f}" for p in sorted(peer_prices))
    prompt = (
        f"You are a landscape procurement expert in India. "
        f"Estimate a reasonable market unit price range (in INR) for the following item.\n\n"
        f"Item: {description}\n"
        f"Species/material: {species or 'N/A'}\n"
        f"Unit: {unit}\n"
        f"Spec notes: {spec_notes or 'None'}\n"
        f"Vendor quotes on file: {peer_str}\n"
        f"Client BOQ reference rate: ₹{boq_rate:.0f}\n"
    )
    if serper_context:
        prompt += f"\nWeb search context (may be unreliable):\n{serper_context[:1500]}\n"
    prompt += (
        "\nReturn ONLY valid JSON with these fields:\n"
        '{"low_inr": <number>, "high_inr": <number>, '
        '"confidence": "low|medium|high", '
        '"rationale": "<one sentence>", '
        '"caveats": ["<string>", ...]}'
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL_ESTIMATE,
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        result = json.loads(resp.choices[0].message.content)
        result["value_source"] = "gpt_estimate"
        return result
    except Exception as exc:
        return {
            "low_inr": None, "high_inr": None,
            "confidence": "none", "rationale": f"GPT estimate failed: {exc}",
            "caveats": ["estimate unavailable"], "value_source": "gpt_estimate",
        }


# ── Stage 2 ───────────────────────────────────────────────────────────────────

def run_stage2(stage1_flagged: list[dict], db_path: str = DB_PATH) -> list[dict]:
    high_value = _high_value_line_ids(db_path)
    seen: set[str] = set()
    qualifying: list[dict] = []
    for rec in stage1_flagged:
        lid = rec["line_id"]
        if lid in high_value and lid not in seen:
            seen.add(lid)
            qualifying.append(rec)

    conn = _db_conn(db_path)
    results: list[dict] = []

    for rec in qualifying:
        lid = rec["line_id"]
        peer_rows = conn.execute("""
            SELECT ve.vendor_id, ve.normalized_unit_price
            FROM vendor_extractions ve
            WHERE ve.line_id=? AND ve.value_source='extracted'
              AND (ve.superseded_by IS NULL OR ve.superseded_by='')
        """, (lid,)).fetchall()
        peer_prices = [r["normalized_unit_price"] for r in peer_rows]

        spec = conn.execute(
            "SELECT spec_notes FROM rfx_lines WHERE line_id=?", (lid,)
        ).fetchone()
        spec_notes = spec["spec_notes"] if spec else ""

        print(f"  Stage 2 — {lid} ({rec['description'][:40]})...")
        serper_text = _serper_benchmark(rec["description"], rec["species"], rec["unit"])

        if _has_inr_price(serper_text):
            benchmark = {"serper_result": serper_text, "value_source": "web_search"}
            gpt_est   = None
        else:
            print(f"    Serper: no INR data → GPT-4o estimate...")
            gpt_est   = _gpt_estimate(
                rec["description"], rec["species"], rec["unit"],
                spec_notes, peer_prices, rec["boq_rate"], serper_text,
            )
            benchmark = {"serper_result": serper_text or "(no results)", "value_source": "web_search"}

        results.append({
            **rec,
            "high_value_eligible": True,
            "peer_prices":      {r["vendor_id"]: r["normalized_unit_price"] for r in peer_rows},
            "serper_benchmark": benchmark,
            "gpt_estimate":     gpt_est,
        })

    conn.close()
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 70)
    print("STAGE 1 — Self-calibrating IQR outlier detection")
    print("=" * 70)

    flagged = run_stage1()

    flagged_lines = {r["line_id"] for r in flagged}
    print(f"\nFlagged: {len(flagged_lines)} unique lines / {len(flagged)} vendor-line pairs")

    # Breakdown by flag type
    from collections import Counter
    flag_counts: Counter = Counter()
    for r in flagged:
        for f in r["new_flags"]:
            flag_counts[f] += 1
    print("\nBy flag type:")
    for flag, count in flag_counts.most_common():
        label = {
            FLAG_IQR_OUTLIER:      "IQR outlier (outside Tukey fences)",
            FLAG_PEER_DEV:         "Peer-median deviation >60% (catches wide-distribution outliers)",
            FLAG_DECIMAL_MISMATCH: "Decimal/unit mismatch (extraction error suspect)",
            FLAG_VS_BOQ:           "vs BOQ reference rate",
            FLAG_EXTREME_LOW:      "Extreme low (<20% of BOQ, peers at reasonable levels — likely wrong grade/spec)",
        }.get(flag, flag)
        print(f"  {label}: {count} vendor-line pairs")

    # Top 5 by ₹ impact
    print("\nTop 5 by ₹ impact (|price - peer_median| × qty):")
    for i, r in enumerate(flagged[:5], 1):
        flags_short = [f.replace("anomaly_", "") for f in r["new_flags"]]
        print(
            f"  {i}. {r['line_id']} [{r['vendor_id']}]  "
            f"₹{r['price']:,.0f} vs median ₹{r['peer_median']:,.0f}  "
            f"impact ₹{r['impact_inr']:,.0f}  [{', '.join(flags_short)}]"
        )
        print(f"     {r['description'] or r['species'] or '—'}")

    # Stage 2 eligibility summary
    high_value = _high_value_line_ids()
    stage2_eligible = {r["line_id"] for r in flagged if r["line_id"] in high_value}
    print(f"\nStage 2 eligible (flagged ∩ top-70% value): {len(stage2_eligible)} lines")

    if "--stage2" in sys.argv:
        stage2_results = run_stage2(flagged)
        print(f"\nStage 2 completed: {len(stage2_results)} benchmark lookups")
        if stage2_results:
            ex2 = stage2_results[0]
            print(f"\nTop Stage 2 result: {ex2['line_id']} — {ex2['description']}")
            if ex2.get("gpt_estimate") and ex2["gpt_estimate"]["low_inr"]:
                g = ex2["gpt_estimate"]
                print(f"  GPT-4o estimate: ₹{g['low_inr']}–{g['high_inr']} ({g['confidence']} confidence)")
                print(f"  Rationale: {g['rationale']}")
                print(f"  value_source: {g['value_source']}  ← never used in totals")
            else:
                sr = ex2["serper_benchmark"]["serper_result"]
                print(f"  Serper result (first 200 chars): {sr[:200]}")
    else:
        print(f"\nRun with --stage2 to execute Serper/GPT benchmark lookups.")

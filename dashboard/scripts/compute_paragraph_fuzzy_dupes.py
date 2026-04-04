#!/usr/bin/env python3
"""
Precompute fuzzy near-duplicate paragraph clusters for the dashboard.

Exact duplicate counts are cheap and computed live in the app. Fuzzy matching (≥threshold
similarity on normalized text) is O(n × window) with a cap on comparisons — still meant to be
run **locally** after updating ``merged_analysis.csv`` (or your ``DASHBOARD_CSV``), not on every
Streamlit request.

Writes ``dashboard/paragraph_fuzzy_duplicates.json`` by default (override with ``--out``).

Example:
  .venv/bin/python dashboard/scripts/compute_paragraph_fuzzy_dupes.py \\
    --csv "old_with future types/data/processed/merged_analysis.csv"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_DASHBOARD_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _DASHBOARD_DIR not in sys.path:
    sys.path.insert(0, _DASHBOARD_DIR)

from data_utils import (  # noqa: E402
    default_csv_path,
    fuzzy_paragraph_duplicate_cluster_stats,
    load_paragraph_table,
    paragraph_exact_duplicate_metrics,
)


def main() -> None:
    root = os.path.abspath(os.path.join(_DASHBOARD_DIR, ".."))
    ap = argparse.ArgumentParser(description="Precompute fuzzy paragraph duplicate stats for AI Vision Explorer.")
    ap.add_argument(
        "--csv",
        default=default_csv_path(root),
        help="Path to merged_analysis.csv (or any dashboard-compatible table)",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(_DASHBOARD_DIR, "paragraph_fuzzy_duplicates.json"),
        help="Output JSON path",
    )
    ap.add_argument("--threshold", type=int, default=90, help="rapidfuzz ratio 0–100 (default 90)")
    ap.add_argument("--window", type=int, default=100, help="Sliding window size for candidate pairs")
    ap.add_argument("--min-len", type=int, default=40, help="Min normalized paragraph length to include")
    args = ap.parse_args()

    csv_path = os.path.abspath(args.csv)
    if not os.path.isfile(csv_path):
        print(f"Error: file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {csv_path} …")
    df = load_paragraph_table(csv_path, dashboard_dir=_DASHBOARD_DIR)

    exact = paragraph_exact_duplicate_metrics(df)
    print(
        f"Exact: {exact['para_rows_total']:,} rows · {exact['para_unique_norms']:,} unique norms · "
        f"{exact['para_exact_extra_rows']:,} extra duplicate rows · {exact['para_exact_multi_groups']:,} multi-row texts"
    )

    print(
        f"Fuzzy clustering (threshold={args.threshold}, window={args.window}, min_len={args.min_len}) …"
    )
    n_g, n_r, cmp_done = fuzzy_paragraph_duplicate_cluster_stats(
        df,
        threshold=args.threshold,
        window=args.window,
        min_len=args.min_len,
    )
    print(f"Fuzzy: {n_g:,} clusters · {n_r:,} rows in those clusters · {cmp_done:,} comparisons")

    mtime = os.path.getmtime(csv_path)
    payload = {
        "source_csv": os.path.basename(csv_path),
        "source_csv_path": csv_path,
        "source_csv_mtime": mtime,
        "fuzzy_para_threshold": args.threshold,
        "fuzzy_para_window": args.window,
        "fuzzy_para_min_len": args.min_len,
        "fuzzy_para_clusters": n_g,
        "fuzzy_para_rows_in_clusters": n_r,
        "fuzzy_para_comparisons": cmp_done,
        **exact,
    }

    out_path = os.path.abspath(args.out)
    _out_dir = os.path.dirname(out_path)
    if _out_dir:
        os.makedirs(_out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build dashboard/article_publication.csv from merged_analysis.csv by scanning full_text
for HBR / MIT SMR URLs and phrases (same logic as the live dashboard).

Run after updating merged_analysis:
  python build_article_publication.py
  python build_article_publication.py "/path/to/merged_analysis.csv"

The CSV is merged on load with a fresh scan: rows in this file override auto-detection
(useful for manual fixes).
"""
from __future__ import annotations

import os
import sys

import pandas as pd

_DASH = os.path.dirname(os.path.abspath(__file__))
if _DASH not in sys.path:
    sys.path.insert(0, _DASH)

from data_utils import (  # noqa: E402
    default_csv_path,
    publication_map_from_full_text,
)


def main() -> None:
    project_root = os.path.abspath(os.path.join(_DASH, ".."))
    csv_path = sys.argv[1] if len(sys.argv) > 1 else default_csv_path(project_root)
    if not os.path.isfile(csv_path):
        print("File not found:", csv_path)
        sys.exit(1)
    print("Scanning:", csv_path)
    pub_map = publication_map_from_full_text(csv_path)
    out_path = os.path.join(_DASH, "article_publication.csv")
    df = pd.DataFrame(
        [{"__article_key": k, "publication": v} for k, v in sorted(pub_map.items())]
    )
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows -> {out_path}")
    print(df["publication"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()

"""
Load and enrich paragraph-level rows for the academic explorer dashboard.
Does not load full_text by default (keeps memory down for large CSVs).
"""
from __future__ import annotations

import json
import math
import os
import re
import unicodedata
from collections import Counter
from typing import Any, Optional

try:
    from rapidfuzz import fuzz as _rfuzz
except ImportError:  # pragma: no cover
    _rfuzz = None

import pandas as pd

# Table 3: Generic vision archetypes = f(vision type × rhetorical stance).
# Must match the paper table (Opening / Mobilizing / Normalizing / Controlling × 5 vision types).
# Also kept in sync with `old_with future types/.../VAR_diagnostics.py` (V2A_STANCED).
V2A_STANCED: dict[tuple[str, str], str] = {
    ("Open Horizons, Unstable Ground", "Opening"): "Pioneer",
    ("Open Horizons, Unstable Ground", "Mobilizing"): "Pioneer",
    ("Open Horizons, Unstable Ground", "Normalizing"): "Pioneer",
    ("Open Horizons, Unstable Ground", "Controlling"): "Guardian",
    ("Empowered but Exposed", "Opening"): "Pioneer",
    ("Empowered but Exposed", "Mobilizing"): "Builder",
    ("Empowered but Exposed", "Normalizing"): "Guardian",
    ("Empowered but Exposed", "Controlling"): "Guardian",
    ("Seamless but Concentrated", "Opening"): "Pioneer",
    ("Seamless but Concentrated", "Mobilizing"): "Builder",
    ("Seamless but Concentrated", "Normalizing"): "Pioneer",
    ("Seamless but Concentrated", "Controlling"): "Guardian",
    ("Transformed or Left Behind", "Opening"): "Pioneer",
    ("Transformed or Left Behind", "Mobilizing"): "Builder",
    ("Transformed or Left Behind", "Normalizing"): "Pioneer",
    ("Transformed or Left Behind", "Controlling"): "Guardian",
    ("Guided but Fragile", "Opening"): "Pioneer",
    ("Guided but Fragile", "Mobilizing"): "Guardian",
    ("Guided but Fragile", "Normalizing"): "Guardian",
    ("Guided but Fragile", "Controlling"): "Guardian",
}

# Vision label → archetypes that can occur (depends on stance)
_vis_arch: dict[str, set[str]] = {}
for (_vis, _stance), _arch in V2A_STANCED.items():
    _vis_arch.setdefault(_vis, set()).add(_arch)
VISION_TO_ARCHETYPES: dict[str, list[str]] = {k: sorted(v) for k, v in sorted(_vis_arch.items())}


def compatible_stances_for_archetypes(archetypes: set[str], vision: str) -> set[str] | None:
    """
    Stances that can produce at least one of the given archetypes under the vision×stance table.
    If `vision` is ``All``, any vision row in V2A_STANCED counts.
    Returns None if `archetypes` is empty (caller: do not deprioritize any stance).
    """
    if not archetypes:
        return None
    out: set[str] = set()
    for (v, s), arch in V2A_STANCED.items():
        if vision != "All" and v != vision:
            continue
        if arch in archetypes:
            out.add(s)
    return out


def compatible_archetypes_for_stances(stances: set[str], vision: str) -> set[str] | None:
    """
    Archetypes that can appear together with at least one of the given stances under the table.
    Returns None if `stances` is empty (no deprioritization).
    """
    if not stances:
        return None
    out: set[str] = set()
    for (v, s), arch in V2A_STANCED.items():
        if vision != "All" and v != vision:
            continue
        if s in stances:
            out.add(arch)
    return out


def compatible_visions_for_filters(archetypes: set[str], stances: set[str]) -> set[str] | None:
    """
    Vision (future) types that can occur under the table given optional archetype and/or stance picks.
    If only archetypes are set: any vision that can produce one of those archetypes for some stance.
    If only stances are set: any vision that appears with one of those stances.
    If both: visions where some (vision, stance) pair uses a selected stance and maps to a selected archetype.
    Returns None if both sets are empty (no deprioritization).
    """
    if not archetypes and not stances:
        return None
    out: set[str] = set()
    for (v, s), arch in V2A_STANCED.items():
        if archetypes and arch not in archetypes:
            continue
        if stances and s not in stances:
            continue
        out.add(v)
    return out


def compatible_archetypes_for_vision(vision: str) -> set[str] | None:
    """
    Archetypes that appear in the vision×stance→archetype table for this vision (any stance).
    Used to show ●/○ on the archetype multiselect when the user picks **vision first** without stances yet.
    """
    if vision == "All":
        return None
    out: set[str] = set()
    for (v, _s), arch in V2A_STANCED.items():
        if v == vision:
            out.add(arch)
    return out if out else None


def compatible_stances_for_vision(vision: str) -> set[str] | None:
    """
    Stances that appear in the table for this vision. Used when vision is set but archetypes are still empty.
    """
    if vision == "All":
        return None
    out: set[str] = set()
    for (v, s), _arch in V2A_STANCED.items():
        if v == vision:
            out.add(s)
    return out if out else None


def compatible_archetypes_for_stances_with_visions(
    stances: set[str], visions: list[str]
) -> set[str] | None:
    """
    Like ``compatible_archetypes_for_stances``, but restrict to one or more vision labels.
    If ``visions`` is empty, any vision row counts (same as former ``vision="All"``).
    """
    if not stances:
        return None
    restrict = bool(visions)
    out: set[str] = set()
    for (v, s), arch in V2A_STANCED.items():
        if restrict and v not in visions:
            continue
        if s in stances:
            out.add(arch)
    return out if out else None


def compatible_archetypes_for_visions_union(visions: list[str]) -> set[str] | None:
    """Union of archetypes that appear in the table for any of the given visions."""
    if not visions:
        return None
    out: set[str] = set()
    for (v, _s), arch in V2A_STANCED.items():
        if v in visions:
            out.add(arch)
    return out if out else None


def compatible_stances_for_archetypes_with_visions(
    archetypes: set[str], visions: list[str]
) -> set[str] | None:
    """
    Like ``compatible_stances_for_archetypes``, but restrict to the given vision list.
    If ``visions`` is empty, any vision counts (same as ``vision="All"``).
    """
    if not archetypes:
        return None
    restrict = bool(visions)
    out: set[str] = set()
    for (v, s), arch in V2A_STANCED.items():
        if restrict and v not in visions:
            continue
        if arch in archetypes:
            out.add(s)
    return out if out else None


def compatible_stances_for_visions_union(visions: list[str]) -> set[str] | None:
    """Union of stances that appear in the table for any of the given visions."""
    if not visions:
        return None
    out: set[str] = set()
    for (v, s), _arch in V2A_STANCED.items():
        if v in visions:
            out.add(s)
    return out if out else None


def is_remote_csv(path: str) -> bool:
    """True if ``path`` is an http(s) URL (pandas can read these directly)."""
    p = (path or "").strip().lower()
    return p.startswith("http://") or p.startswith("https://")


def default_csv_path(project_root: str) -> str:
    """Prefer dashboard-local data (Streamlit / slim repo); fall back to legacy monorepo path."""
    dash_local = os.path.join(project_root, "dashboard", "data", "merged_analysis.csv")
    legacy = os.path.join(
        project_root,
        "old_with future types",
        "data",
        "processed",
        "merged_analysis.csv",
    )
    if os.path.isfile(dash_local):
        return dash_local
    if os.path.isfile(legacy):
        return legacy
    return dash_local


def infer_publication(source_file: str, title: str = "") -> str:
    """Fallback when full text is unavailable: filenames are often generic (e.g. EBSCO PDFs)."""
    s = f"{source_file or ''}\n{title or ''}".lower()
    if "hbr.org" in s or "harvard business review" in s or ("harvard" in s and "business" in s):
        return "HBR"
    if "sloanreview.mit.edu" in s or "mit smr" in s or "sloan management review" in s or "mit sloan management review" in s:
        return "MIT SMR"
    if "hbr" in s:
        return "HBR"
    return "Other / unknown"


def infer_publication_from_text(blob: str, *, scan_full_if_needed: bool = False) -> Optional[str]:
    """
    Detect HBR vs MIT SMR from article body (URLs, imprint). Returns None if unclear.
    EBSCO PDFs often bury the canonical URL after page headers; we scan a prefix first,
    then optionally the full string (cheap substring search, no regex on huge strings twice).
    """
    raw = blob or ""
    if not str(raw).strip():
        return None
    s = raw[:50000].lower() if len(raw) > 50000 else raw.lower()
    hit = _infer_pub_from_lower(s)
    if hit is not None:
        return hit
    if scan_full_if_needed and len(raw) > 50000:
        return _infer_pub_from_lower(raw.lower())
    return None


def _infer_pub_from_lower(s: str) -> Optional[str]:
    if "sloanreview.mit.edu" in s or "mit sloan management review" in s:
        return "MIT SMR"
    if "sloan management review" in s and "mit" in s:
        return "MIT SMR"
    if "magazine summer" in s and "sloan" in s and "mit" in s:
        return "MIT SMR"
    if "hbr.org" in s or "harvard business review" in s or "store.hbr.org" in s:
        return "HBR"
    if "hbr print" in s or "from hbr.org" in s:
        return "HBR"
    return None


def publication_map_from_full_text(csv_path: str, *, chunksize: int = 4000) -> dict[str, str]:
    """
    One pass over (__article_key, full_text): first row per article wins for a text snippet.
    Fills publication labels that filenames/titles miss (e.g. EBSCO PDF names).
    """
    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if "full_text" not in header or "__article_key" not in header:
        return {}
    out: dict[str, str] = {}
    usecols = ["__article_key", "full_text"]
    for chunk in pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize, low_memory=False):
        for _, row in chunk.iterrows():
            ak = str(row["__article_key"])
            if ak in out:
                continue
            ft = str(row.get("full_text", "") or "")
            guess = infer_publication_from_text(ft, scan_full_if_needed=True)
            if guess is not None:
                out[ak] = guess
    return out


def load_article_publication_file(dashboard_dir: str) -> dict[str, str]:
    """
    Optional `dashboard/article_publication.csv` with columns __article_key, publication.
    Usually produced by `build_article_publication.py` from merged_analysis (full_text scan).
    Merged with live scan on load; **file values override** auto-detection for the same key
    (useful for manual fixes).
    """
    path = os.path.join(dashboard_dir, "article_publication.csv")
    if not os.path.isfile(path):
        return {}
    try:
        t = pd.read_csv(path, dtype=str)
        if "__article_key" not in t.columns or "publication" not in t.columns:
            return {}
        t = t.dropna(subset=["__article_key", "publication"])
        return dict(zip(t["__article_key"].astype(str), t["publication"].astype(str).str.strip()))
    except Exception:
        return {}


def load_paragraph_table(
    csv_path: str,
    *,
    optional_extra_cols: Optional[list[str]] = None,
    dashboard_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read only columns needed for filtering and display (excludes full_text).
    """
    base_cols = [
        "source_file",
        "title",
        "paragraph_id",
        "paragraph",
        "year",
        "month",
        "future_type_prediction",
        "majority_stance",
        "__article_key",
    ]
    extra = optional_extra_cols or []
    usecols = list(dict.fromkeys(base_cols + extra))

    if not is_remote_csv(csv_path) and not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    present = [c for c in usecols if c in header]
    if not present:
        raise ValueError(f"No expected columns found in CSV. Header: {header[:30]}...")
    df = pd.read_csv(csv_path, usecols=present, low_memory=False)

    for c in usecols:
        if c not in df.columns:
            df[c] = pd.NA

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    if "month" in df.columns:
        df["month"] = pd.to_numeric(df["month"], errors="coerce")

    header = pd.read_csv(csv_path, nrows=0).columns.tolist()
    dd = dashboard_dir or os.path.dirname(os.path.abspath(__file__))

    scan_map: dict[str, str] = {}
    if "full_text" in header:
        scan_map = publication_map_from_full_text(csv_path)

    file_map = load_article_publication_file(dd)
    # Merge: scan from current merged_analysis, then overlay pre-built / hand-edited CSV.
    pub_by_article = {**scan_map, **file_map}

    ak_str = df["__article_key"].astype(str) if "__article_key" in df.columns else pd.Series("", index=df.index)
    fallback_pub = df.apply(
        lambda r: infer_publication(str(r.get("source_file", "")), str(r.get("title", ""))),
        axis=1,
    )

    if pub_by_article:
        df["publication"] = ak_str.map(lambda k: pub_by_article.get(k)).fillna(fallback_pub)
    else:
        df["publication"] = fallback_pub

    def _arch(row) -> Optional[str]:
        ft = row.get("future_type_prediction")
        st = row.get("majority_stance")
        if pd.isna(ft) or pd.isna(st):
            return None
        return V2A_STANCED.get((str(ft), str(st)))

    df["archetype"] = df.apply(_arch, axis=1)

    df["date"] = pd.to_datetime(
        dict(year=df["year"], month=df["month"], day=1),
        errors="coerce",
    )

    return df


def parse_keywords(s: str) -> list[str]:
    """Comma/space-separated keywords; quoted phrases optional."""
    import shlex

    s = (s or "").strip()
    if not s:
        return []
    if '"' in s or "'" in s:
        tokens = shlex.split(s)
    else:
        tokens = [t for t in re.split(r"[,\s]+", s) if t]
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        k = t.lower()
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def keyword_mask(series: pd.Series, terms: list[str], match_all: bool) -> pd.Series:
    if not terms:
        return pd.Series(True, index=series.index)
    s = series.fillna("").astype(str).str.lower()

    def row_ok(txt: str) -> bool:
        if match_all:
            return all(t.lower() in txt for t in terms)
        return any(t.lower() in txt for t in terms)

    return s.map(row_ok)


# Minimal English stopwords for exploratory word frequency (not linguistic claims)
_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "to", "of", "in", "on", "at", "by", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "must", "shall", "can", "need", "dare", "ought", "used", "it", "its", "this", "that",
    "these", "those", "i", "you", "he", "she", "we", "they", "them", "their", "what", "which", "who",
    "whom", "whose", "where", "when", "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "just",
    "also", "now", "here", "there", "then", "once", "if", "as", "but", "with", "from", "into", "about",
    "over", "after", "before", "between", "through", "during", "without", "against", "among", "per", "via",
}


_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z\-']+", re.UNICODE)


def token_counter(texts: list[str], *, min_len: int = 3) -> Counter[str]:
    """Lowercase word counts; same token rules as word_frequencies (stopwords + min length)."""
    c: Counter[str] = Counter()
    for t in texts:
        if not t or not isinstance(t, str):
            continue
        for m in _TOKEN_RE.finditer(t.lower()):
            w = m.group(0).strip("'")
            if len(w) < min_len or w in _STOPWORDS:
                continue
            c[w] += 1
    return c


def word_frequencies(texts: list[str], *, top_n: int = 40, min_len: int = 3) -> list[tuple[str, int]]:
    """Token counts from raw text (lowercase words); filters stopwords and short tokens."""
    return token_counter(texts, min_len=min_len).most_common(top_n)


def distinctive_word_scores(
    filt_texts: list[str],
    baseline: Counter[str],
    *,
    top_n: int = 40,
    min_count_filt: int = 3,
    min_len: int = 3,
) -> list[tuple[str, float]]:
    """
    Words that are over-represented in the filtered set vs the full corpus (smoothed log-odds).

    Uses the same tokenization as word_frequencies. Baseline should be token counts over the
    entire corpus (or a comparable reference). Cheap: one pass over filter + dict lookups.
    """
    filt = token_counter(filt_texts, min_len=min_len)
    t_f = sum(filt.values())
    t_b = sum(baseline.values())
    if t_f == 0 or t_b == 0:
        return []
    alpha = 0.5
    scores: list[tuple[str, float]] = []
    for w, c_f in filt.items():
        if c_f < min_count_filt:
            continue
        c_b = baseline.get(w, 0)
        p_f = (c_f + alpha) / (t_f + 2.0 * alpha)
        p_b = (c_b + alpha) / (t_b + 2.0 * alpha)
        if p_f <= 0 or p_b <= 0:
            continue
        scores.append((w, math.log(p_f / p_b)))
    scores.sort(key=lambda x: -x[1])
    return scores[:top_n]


def word_weights_for_cloud_from_scores(scores: list[tuple[str, float]]) -> dict[str, float]:
    """Map log-odds scores to positive weights suitable for WordCloud.generate_from_frequencies."""
    if not scores:
        return {}
    lo = min(s for _, s in scores)
    # Shift so smallest is ~1; keeps ordering, avoids non-positive values
    return {w: float(s - lo + 1.0) for w, s in scores}


def per_paragraph_representativeness(
    texts: list[str],
    word_weights: dict[str, float],
    *,
    normalize_length: bool = True,
    min_len: int = 3,
) -> list[float]:
    """
    Score each text using **distinctive** word weights (log-odds vs an external baseline—the same
    weights as the word cloud). Higher = loads more on terms that *contrast* with the full corpus.

    When ``normalize_length`` is True, the raw sum of matched token weights is divided by
    ``sqrt(n_tokens)`` where ``n_tokens`` counts non-stopword tokens of sufficient length in the
    paragraph (reduces bias toward long snippets). For “typical of the filter” ranking, see
    ``per_paragraph_filter_centroid_cosine``.
    """
    if not word_weights:
        return [0.0] * len(texts)
    out: list[float] = []
    for t in texts:
        if not t or not isinstance(t, str):
            out.append(0.0)
            continue
        s_sum = 0.0
        n_tok = 0
        for m in _TOKEN_RE.finditer(t.lower()):
            w = m.group(0).strip("'")
            if len(w) < min_len or w in _STOPWORDS:
                continue
            n_tok += 1
            s_sum += word_weights.get(w, 0.0)
        if normalize_length and n_tok > 0:
            s_sum /= math.sqrt(float(n_tok))
        out.append(s_sum)
    return out


def per_paragraph_filter_centroid_cosine(
    texts: list[str],
    *,
    min_len: int = 3,
) -> list[float]:
    """
    Cosine similarity between each paragraph's token distribution and the unigram distribution
    of the **entire current filter** (same token rules as ``token_counter``).

    High scores go to snippets whose wording is closest to the bulk of the selection—useful
    for “typical” or paper-ready exemplars. Contrasts with log-odds *distinctiveness* vs an
    external corpus (see ``distinctive_word_scores`` / ``per_paragraph_representativeness``).
    """
    if not texts:
        return []
    filt = token_counter(texts, min_len=min_len)
    total_f = float(sum(filt.values()))
    if total_f <= 0:
        return [0.0] * len(texts)
    c_w = {w: filt[w] / total_f for w in filt}
    norm_c = math.sqrt(sum(v * v for v in c_w.values()))
    if norm_c <= 0:
        return [0.0] * len(texts)

    out: list[float] = []
    for t in texts:
        if not t or not isinstance(t, str):
            out.append(0.0)
            continue
        pc = token_counter([t], min_len=min_len)
        total_p = float(sum(pc.values()))
        if total_p <= 0:
            out.append(0.0)
            continue
        dot = 0.0
        norm_p_sq = 0.0
        for w, cnt in pc.items():
            pw = cnt / total_p
            norm_p_sq += pw * pw
            cw = c_w.get(w)
            if cw is not None:
                dot += pw * cw
        denom = math.sqrt(norm_p_sq) * norm_c
        out.append((dot / denom) if denom > 0 else 0.0)
    return out


def article_key_series(df: pd.DataFrame) -> pd.Series:
    """Prefer __article_key; else stable key from title + source_file."""
    t = df.get("title", pd.Series("", index=df.index)).astype(str)
    s = df.get("source_file", pd.Series("", index=df.index)).astype(str)
    fallback = t + "||" + s
    if "__article_key" not in df.columns:
        return fallback
    ak = df["__article_key"]
    return ak.where(ak.notna(), other=fallback).astype(str)


def _normalize_title_for_fuzzy(s: str) -> str:
    """
    Aggressive normalization for fuzzy matching PDF-derived titles (spacing, punctuation, case).
    """
    if not s or not isinstance(s, str):
        return ""
    t = unicodedata.normalize("NFKD", s)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = t.lower().strip()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _title_similarity_ratio(a: str, b: str) -> int:
    """Return 0–100 similarity; uses rapidfuzz when available, else difflib."""
    if not a or not b:
        return 100 if a == b else 0
    if _rfuzz is not None:
        return int(_rfuzz.ratio(a, b))
    from difflib import SequenceMatcher

    return int(SequenceMatcher(None, a, b).ratio() * 100.0)


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1


def _fuzzy_title_multi_key_clusters(
    keys: list[str],
    norms: list[str],
    *,
    threshold: int = 88,
    window: int = 100,
) -> tuple[int, int]:
    """
    Cluster article keys whose normalized titles are fuzzily similar (different IDs, same work).

    Returns (n_clusters_with_at_least_2_keys, n_distinct_keys_in_those_clusters).
    """
    n = len(keys)
    if n < 2:
        return 0, 0

    # Sort by title length so likely matches are nearby; compare within a sliding window only.
    order = sorted(range(n), key=lambda i: (len(norms[i]), norms[i]))
    uf = _UnionFind(n)
    comparisons = 0
    max_comparisons = 1_200_000
    for ii in range(n):
        i = order[ii]
        la = len(norms[i])
        if la < 4:
            continue
        for kk in range(ii + 1, min(n, ii + window + 1)):
            j = order[kk]
            lb = len(norms[j])
            if lb - la > max(18, la // 3):
                break
            comparisons += 1
            if comparisons > max_comparisons:
                break
            if keys[i] == keys[j]:
                continue
            if norms[i] == norms[j]:
                uf.union(i, j)
            elif _title_similarity_ratio(norms[i], norms[j]) >= threshold:
                uf.union(i, j)
        if comparisons > max_comparisons:
            break

    root_to_keys: dict[int, set[str]] = {}
    for i in range(n):
        if len(norms[i]) < 4:
            continue
        r = uf.find(i)
        root_to_keys.setdefault(r, set()).add(keys[i])

    n_groups = 0
    keys_in = 0
    for ks in root_to_keys.values():
        if len(ks) >= 2:
            n_groups += 1
            keys_in += len(ks)
    return n_groups, keys_in


def article_reprint_heuristics(df: pd.DataFrame) -> dict[str, int]:
    """
    Heuristic duplicate / reprint signals at paragraph granularity.

    - unique_articles: distinct article identity (__article_key or fallback).
    - keys_multi_title: article IDs with more than one normalized title string
      (OCR variants, retitling, or true reprint metadata).
    - titles_multi_key: normalized titles that appear under more than one article ID
      (same title text, different keys — e.g. reprint under another filename / listing).

    These are exploratory counts, not ground truth (titles can legitimately differ slightly).
    """
    if df is None or len(df) == 0:
        return {"unique_articles": 0, "keys_multi_title": 0, "titles_multi_key": 0}

    ak = article_key_series(df)
    tit = df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
    tnorm = tit.str.strip().str.lower()

    tmp = pd.DataFrame({"_ak": ak.astype(str), "_tn": tnorm})
    per_key = tmp.groupby("_ak", sort=False)["_tn"].nunique()
    n_keys_multi = int((per_key > 1).sum())

    tmp2 = tmp[tmp["_tn"].str.len() > 0]
    if len(tmp2) == 0:
        n_titles_multi = 0
    else:
        per_title = tmp2.groupby("_tn", sort=False)["_ak"].nunique()
        n_titles_multi = int((per_title > 1).sum())

    n_unique = int(ak.nunique())
    return {
        "unique_articles": n_unique,
        "keys_multi_title": n_keys_multi,
        "titles_multi_key": n_titles_multi,
    }


def article_summary_stats(df: pd.DataFrame) -> dict[str, int]:
    """
    Summary for the dashboard: reprint heuristics, per-article snippet counts, and fuzzy title clusters.

    Extends ``article_reprint_heuristics`` with:

    - articles_*_snippet(s): how many article IDs have exactly 1 / 2 / 3 / 4+ paragraph rows
      (multiplicity within an ID — e.g. multiple snippets per PDF).
    - max_snippets_per_article: largest paragraph count for a single article ID.
    - fuzzy_title_multi_key_groups: groups of 2+ article IDs whose titles match fuzzily (PDF/OCR variants).
    - fuzzy_title_keys_in_groups: distinct article IDs involved in those groups.
    """
    base = article_reprint_heuristics(df)
    empty_tail = {
        "articles_1_snippet": 0,
        "articles_2_snippets": 0,
        "articles_3_snippets": 0,
        "articles_4plus_snippets": 0,
        "max_snippets_per_article": 0,
        "fuzzy_title_multi_key_groups": 0,
        "fuzzy_title_keys_in_groups": 0,
    }
    if df is None or len(df) == 0:
        return {**base, **empty_tail}

    ak = article_key_series(df)
    vc = ak.value_counts()
    n1 = int((vc == 1).sum())
    n2 = int((vc == 2).sum())
    n3 = int((vc == 3).sum())
    n4p = int((vc >= 4).sum())
    mx = int(vc.max()) if len(vc) else 0

    # One representative title per article key (first row) for fuzzy clustering
    tit = df.get("title", pd.Series("", index=df.index)).fillna("").astype(str)
    sub = pd.DataFrame({"_ak": ak.astype(str), "_t": tit})
    sub = sub[sub["_ak"].str.len() > 0]
    first_title = sub.groupby("_ak", sort=False).first()["_t"].astype(str)
    keys_list = list(first_title.index.astype(str))
    norms = [_normalize_title_for_fuzzy(first_title[k]) for k in keys_list]

    fz_groups, fz_keys = _fuzzy_title_multi_key_clusters(keys_list, norms, threshold=88, window=100)

    out = {
        **base,
        "articles_1_snippet": n1,
        "articles_2_snippets": n2,
        "articles_3_snippets": n3,
        "articles_4plus_snippets": n4p,
        "max_snippets_per_article": mx,
        "fuzzy_title_multi_key_groups": fz_groups,
        "fuzzy_title_keys_in_groups": fz_keys,
    }
    return out


def normalize_paragraph_strong(s: str) -> str:
    """
    Strong normalization for exact paragraph deduplication (strip, collapse whitespace, strip accents).
    Aligns with cue-batch “strong” dedupe: one row per identical normalized body.
    """
    if not s or not isinstance(s, str):
        return ""
    t = unicodedata.normalize("NFKD", s)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = re.sub(r"\s+", " ", t.strip())
    return t


def paragraph_exact_duplicate_metrics(df: pd.DataFrame) -> dict[str, int]:
    """
    Exact duplicate paragraphs: same ``normalize_paragraph_strong`` text on multiple rows.

    - para_rows_total: row count
    - para_unique_norms: distinct normalized strings (including one bucket for empty)
    - para_distinct_body_article_key: distinct (normalized body, article identity) pairs
      (``article_key_series``: prefer ``__article_key``, else title ``||`` source_file).
      Same paragraph text under a different article ID counts separately (reprints).
    - para_exact_extra_rows: rows − unique (total “copy” rows)
    - para_exact_multi_groups: normalized texts that appear on 2+ rows
    """
    if df is None or len(df) == 0:
        return {
            "para_rows_total": 0,
            "para_unique_norms": 0,
            "para_distinct_body_article_key": 0,
            "para_exact_extra_rows": 0,
            "para_exact_multi_groups": 0,
        }
    s = df["paragraph"].fillna("").astype(str).map(normalize_paragraph_strong)
    ak = article_key_series(df).astype(str)
    u_body_ak = int((s + "\x00" + ak).nunique())
    n = len(s)
    vc = s.value_counts()
    u = int(vc.size)
    extra = n - u
    multi = int((vc > 1).sum())
    return {
        "para_rows_total": n,
        "para_unique_norms": u,
        "para_distinct_body_article_key": u_body_ak,
        "para_exact_extra_rows": extra,
        "para_exact_multi_groups": multi,
    }


def dedupe_sorted_paragraph_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Collapse rows that share the same ``normalize_paragraph_strong(paragraph)`` text.

    Keeps the **first** row of each group (order follows the already-sorted ``df``).
    Adds:

    - ``reprint_count``: rows in this frame with that normalized text
    - ``_ave_para_norm``: normalized key (join to the second return for all appearances)

    Returns:
        ``(deduped_df, full_df_with_norm)`` — second value is a copy of ``df`` plus ``_ave_para_norm``.
    """
    if df is None or len(df) == 0:
        e = df.copy() if df is not None else pd.DataFrame()
        return e, e
    full = df.copy()
    pn = full["paragraph"].fillna("").astype(str).map(normalize_paragraph_strong)
    full["_ave_para_norm"] = pn
    counts = full.groupby("_ave_para_norm", sort=False).size()
    dedup = full.drop_duplicates(subset=["_ave_para_norm"], keep="first").copy()
    dedup["reprint_count"] = dedup["_ave_para_norm"].map(counts).astype(int)
    return dedup, full


def _paragraph_body_similarity_ratio(a: str, b: str) -> int:
    """0–100 similarity for long paragraphs; truncates very long strings for speed."""
    if not a or not b:
        return 100 if a == b else 0
    maxl = 480
    if len(a) > maxl:
        a = a[:maxl]
    if len(b) > maxl:
        b = b[:maxl]
    return _title_similarity_ratio(a, b)


def fuzzy_paragraph_duplicate_cluster_stats(
    df: pd.DataFrame,
    *,
    threshold: int = 90,
    window: int = 100,
    min_len: int = 40,
    max_comparisons: int = 2_000_000,
) -> tuple[int, int, int]:
    """
    Fuzzy near-duplicate **distinct** normalized paragraphs (different exact text, ratio ≥ threshold).

    Operates on unique normalized strings with length ≥ ``min_len``. Uses a length-sorted sliding
    window (same idea as title clustering) to cap pairwise work.

    Returns:
        (n_clusters_with_at_least_2_distinct_strings, total_rows_in_those_clusters, comparisons_done)
    """
    if df is None or len(df) == 0:
        return 0, 0, 0
    s = df["paragraph"].fillna("").astype(str).map(normalize_paragraph_strong)
    vc = s.value_counts()
    items = [(str(k), int(v)) for k, v in vc.items() if len(str(k)) >= min_len]
    n = len(items)
    if n < 2:
        return 0, 0, 0

    texts = [t for t, _ in items]
    freqs = [f for _, f in items]
    order = sorted(range(n), key=lambda i: (len(texts[i]), texts[i]))
    uf = _UnionFind(n)
    comparisons = 0

    for ii in range(n):
        i = order[ii]
        la = len(texts[i])
        if la < min_len:
            continue
        for kk in range(ii + 1, min(n, ii + window + 1)):
            j = order[kk]
            lb = len(texts[j])
            if lb - la > max(72, la // 4):
                break
            if texts[i] == texts[j]:
                continue
            comparisons += 1
            if comparisons > max_comparisons:
                break
            if _paragraph_body_similarity_ratio(texts[i], texts[j]) >= threshold:
                uf.union(i, j)
        if comparisons > max_comparisons:
            break

    root_members: dict[int, list[int]] = {}
    for i in range(n):
        r = uf.find(i)
        root_members.setdefault(r, []).append(i)

    n_groups = 0
    rows_in = 0
    for members in root_members.values():
        if len(members) < 2:
            continue
        n_groups += 1
        rows_in += sum(freqs[i] for i in members)

    return n_groups, rows_in, comparisons


def load_paragraph_fuzzy_duplicate_json(path: str) -> dict[str, Any] | None:
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def paragraph_duplicate_metrics(
    df: pd.DataFrame,
    *,
    csv_path: str | None = None,
    dashboard_dir: str | None = None,
) -> dict[str, Any]:
    """
    Exact paragraph duplicate counts (always computed) plus optional **precomputed** fuzzy stats
    from ``paragraph_fuzzy_duplicates.json`` next to the app or ``DASHBOARD_PARA_FUZZY_JSON``.

    Fuzzy clustering is too heavy for interactive runs on large corpora; run
    ``dashboard/scripts/compute_paragraph_fuzzy_dupes.py`` locally after updating the CSV.
    """
    exact = paragraph_exact_duplicate_metrics(df)
    out: dict[str, Any] = {
        **exact,
        "fuzzy_para_clusters": None,
        "fuzzy_para_rows_in_clusters": None,
        "fuzzy_para_comparisons": None,
        "fuzzy_para_threshold": None,
        "fuzzy_para_file_stale": False,
        "fuzzy_para_file_path": None,
    }

    jpath = (os.environ.get("DASHBOARD_PARA_FUZZY_JSON") or "").strip()
    if not jpath and dashboard_dir:
        jpath = os.path.join(dashboard_dir, "paragraph_fuzzy_duplicates.json")
    out["fuzzy_para_file_path"] = jpath or None

    blob = load_paragraph_fuzzy_duplicate_json(jpath) if jpath else None
    if not blob:
        return out

    out["fuzzy_para_clusters"] = blob.get("fuzzy_para_clusters")
    out["fuzzy_para_rows_in_clusters"] = blob.get("fuzzy_para_rows_in_clusters")
    out["fuzzy_para_comparisons"] = blob.get("fuzzy_para_comparisons")
    out["fuzzy_para_threshold"] = blob.get("fuzzy_para_threshold")

    if csv_path and blob.get("source_csv_mtime") is not None:
        if is_remote_csv(csv_path):
            out["fuzzy_para_file_stale"] = True
        else:
            try:
                mtime = os.path.getmtime(csv_path)
                if abs(float(mtime) - float(blob["source_csv_mtime"])) > 1.5:
                    out["fuzzy_para_file_stale"] = True
            except OSError:
                out["fuzzy_para_file_stale"] = True

    return out

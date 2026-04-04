"""
Private academic paragraph explorer (Streamlit).
Set password via .streamlit/secrets.toml or DASHBOARD_PASSWORD env var.
"""
from __future__ import annotations

import base64
import calendar
import html
import io
import json
import os
import sys

import altair as alt
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

# Project root = parent of dashboard/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

from collections import Counter  # noqa: E402

from data_utils import (  # noqa: E402
    VISION_TO_ARCHETYPES,
    article_key_series,
    article_summary_stats,
    compatible_archetypes_for_stances,
    compatible_archetypes_for_vision,
    compatible_stances_for_archetypes,
    compatible_stances_for_vision,
    compatible_visions_for_filters,
    default_csv_path,
    distinctive_word_scores,
    keyword_mask,
    load_paragraph_table,
    paragraph_duplicate_metrics,
    parse_keywords,
    per_paragraph_filter_centroid_cosine,
    per_paragraph_representativeness,
    token_counter,
    word_frequencies,
    word_weights_for_cloud_from_scores,
)

_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
_BRAND_ICON_PATH = os.path.join(_DASHBOARD_DIR, "assets", "app_icon.png")

_PAGE_KEY = "dash_page_num"
_FILTER_SIG_KEY = "dash_filter_sig"
_FILTER_FOCUS_KEY = "filter_focus"
_SCROLL_SIDEBAR_KEY = "dash_scroll_sidebar"

# Distinct accent colors for multi-select compare (archetypes / stances)
_COMPARE_COLORS = ("#1a3d5c", "#c45c26", "#2d6a4f", "#7b2cbf", "#c9184a", "#006d77", "#bc6c25")

# Snippet ordering (sidebar + snippet toolbar selectbox, key dash_sort)
SORT_SNIPPET_OPTIONS: list[str] = [
    "Date (newest first)",
    "Date (oldest first)",
    "Most distinctive for current filter (vs full corpus)",
    "Most representative of current filter",
    "Publication (A–Z)",
    "Title (A–Z)",
    "Archetype (A–Z)",
]

# Older UI labels → current (session state migration)
_SORT_LABEL_LEGACY: dict[str, str] = {
    "Most distinctive (vs corpus)": "Most distinctive for current filter (vs full corpus)",
    "Most representative (of filter)": "Most representative of current filter",
}


def _set_filter_focus(key: str) -> None:
    st.session_state[_FILTER_FOCUS_KEY] = key


def _nav_time_cb() -> None:
    _set_filter_focus("time")
    st.session_state[_SCROLL_SIDEBAR_KEY] = True


def _nav_primary_cb() -> None:
    _set_filter_focus("primary")
    st.session_state[_SCROLL_SIDEBAR_KEY] = True


def _nav_keywords_cb() -> None:
    _set_filter_focus("keywords")
    st.session_state[_SCROLL_SIDEBAR_KEY] = True


def _nav_display_cb() -> None:
    _set_filter_focus("display")
    st.session_state[_SCROLL_SIDEBAR_KEY] = True


def _filter_open_cb() -> None:
    """Focus the sidebar filter area (archetype / stance / vision first)."""
    _set_filter_focus("primary")
    st.session_state[_SCROLL_SIDEBAR_KEY] = True


def _trunc_ui(s: str, n: int = 26) -> str:
    t = str(s) if s is not None else ""
    return t if len(t) <= n else t[: n - 1] + "…"


def _fmt_multi_chip(selected: list[str], *, max_len: int = 28) -> str:
    """Label for multiselect filters: empty selection means all options."""
    if not selected:
        return "All"
    s = ", ".join(sorted(str(x) for x in selected))
    return _trunc_ui(s, max_len)


def _chip_panel_line(
    label: str,
    value: str,
    *,
    emoji: str = "",
    max_len: int = 26,
) -> str:
    """Two-line chip label: category (+ optional emoji, first line via CSS) + prominent value."""
    head = f"{emoji} {label}".strip() if emoji else label
    return f"{head}\n{_trunc_ui(value, max_len)}"


def _apply_filters_unsorted(
    df: pd.DataFrame,
    time_mode: str,
    archetypes: list,
    stances: list,
    vision: str,
    pub: str,
    kw: str,
    match_all: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply all sidebar filters except snippet sort. Returns (filtered rows, time-window-only copy for charts)."""
    out = _apply_time_range_filter(df, time_mode)
    df_time_only = out.copy()
    if archetypes:
        out = out[out["archetype"].isin(archetypes)]
    if stances:
        out = out[out["majority_stance"].isin(stances)]
    if vision != "All":
        out = out[out["future_type_prediction"] == vision]
    if pub != "All":
        out = out[out["publication"] == pub]
    terms = parse_keywords(kw)
    if terms:
        out = out[keyword_mask(out["paragraph"], terms, match_all)]
    return out, df_time_only


def _apply_time_range_filter(df: pd.DataFrame, time_mode: str) -> pd.DataFrame:
    """Apply only the sidebar time filter (year/month or day range)."""
    out = df.copy()
    if time_mode == "Year & month":
        rs = pd.Timestamp(int(st.session_state["dash_y0"]), int(st.session_state["dash_m0"]), 1)
        rexc = pd.Timestamp(int(st.session_state["dash_y1"]), int(st.session_state["dash_m1"]), 1) + pd.offsets.MonthEnd(0)
        if rs > rexc:
            rs, rexc = rexc, rs
        out = out[(out["date"] >= rs) & (out["date"] <= rexc)]
    else:
        d0 = st.session_state.get("dash_d0")
        d1 = st.session_state.get("dash_d1")
        if d0 is not None and d1 is not None:
            rs = pd.Timestamp(d0)
            rexc = pd.Timestamp(d1) + pd.Timedelta(days=1)
            out = out[(out["date"] >= rs) & (out["date"] < rexc)]
    return out


def _brand_icon_data_uri() -> str:
    """PNG as data URI for inline HTML (Streamlit markdown cannot load local file URLs)."""
    if not os.path.isfile(_BRAND_ICON_PATH):
        return ""
    with open(_BRAND_ICON_PATH, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def _password_from_dashboard_secrets_file() -> str:
    """Streamlit often loads CWD/.streamlit/secrets.toml; we also support dashboard/.streamlit/secrets.toml next to app.py."""
    path = os.path.join(_DASHBOARD_DIR, ".streamlit", "secrets.toml")
    if not os.path.isfile(path):
        return ""
    try:
        import tomllib

        with open(path, "rb") as f:
            data = tomllib.load(f)
        return (data.get("password") or "").strip()
    except Exception:
        return ""


def _secret_password() -> str:
    try:
        s = (st.secrets.get("password", "") or "").strip()
    except Exception:
        s = ""
    if s:
        return s
    return _password_from_dashboard_secrets_file()


def _password_ok() -> bool:
    return bool(_secret_password()) or bool(os.environ.get("DASHBOARD_PASSWORD", "").strip())


def _inject_login_css() -> None:
    st.markdown(
        """
<style>
  .ave-login-outer { max-width: 440px; margin: 0 auto; padding: 1.5rem 1rem 2rem 1rem; }
  .ave-login-card {
    background: linear-gradient(180deg, #ffffff 0%, #f7f4ef 100%);
    border: 1px solid #e0d8ce;
    border-radius: 16px;
    padding: 1.75rem 1.5rem 1.5rem 1.5rem;
    box-shadow: 0 6px 28px rgba(35, 48, 58, 0.1);
  }
  .ave-login-brand { display: flex; align-items: flex-start; gap: 0.85rem; margin-bottom: 1.15rem; }
  .ave-login-logo {
    width: 52px; height: 52px; min-width: 52px;
    border-radius: 14px;
    background: linear-gradient(145deg, #f7f4ef, #e3eef6);
    border: 1px solid #c5d8e8;
    display: flex; align-items: center; justify-content: center;
    overflow: hidden;
  }
  .ave-login-logo img { width: 100%; height: 100%; object-fit: contain; display: block; border-radius: 10px; }
  .ave-login-title { font-size: 1.4rem; font-weight: 800; color: #2f4a5e; margin: 0; line-height: 1.2; letter-spacing: -0.02em; }
  .ave-login-tagline { font-size: 0.9rem; color: #5c6670; margin: 0.45rem 0 0 0; line-height: 1.5; }
  .ave-login-divider { height: 1px; background: rgba(0,0,0,0.06); margin: 1.1rem 0 1rem 0; }
  .ave-login-actions p { font-size: 0.82rem; color: #6b7280; margin: 0 0 0.5rem 0; }
</style>
        """,
        unsafe_allow_html=True,
    )


def _check_password() -> bool:
    if st.session_state.get("_auth_ok"):
        return True
    if not _password_ok():
        st.error(
            "Dashboard password not configured. Copy `dashboard/.streamlit/secrets.toml.example` "
            "to `dashboard/.streamlit/secrets.toml` and set `password`, or set `DASHBOARD_PASSWORD`."
        )
        return False

    expected = _secret_password() or os.environ.get("DASHBOARD_PASSWORD", "").strip()
    _inject_login_css()
    icon_uri = _brand_icon_data_uri()
    logo_block = (
        f'<div class="ave-login-logo"><img src="{icon_uri}" alt="" /></div>'
        if icon_uri
        else '<div class="ave-login-logo" style="font-size:1.6rem">🔭</div>'
    )
    _, mid, _ = st.columns([1, 2.15, 1])
    with mid:
        st.markdown(
            f"""
<div class="ave-login-outer">
  <div class="ave-login-card">
    <div class="ave-login-brand">
      {logo_block}
      <div>
        <p class="ave-login-title">AI Vision Explorer</p>
        <p class="ave-login-tagline">
          Explore AI-related paragraphs from <strong>HBR</strong> and <strong>MIT Sloan Management Review</strong>—filter by time,
          archetype, stance, and vision; scan trends; and read curated excerpts.
        </p>
      </div>
    </div>
    <div class="ave-login-divider" aria-hidden="true"></div>
    <div class="ave-login-actions">
      <p>Sign in with the shared password.</p>
    </div>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
        pw = st.text_input(
            "Password",
            type="password",
            autocomplete="current-password",
            label_visibility="visible",
            placeholder="Enter password",
        )
        if st.button("Sign in", type="primary", use_container_width=True):
            if pw and pw == expected:
                st.session_state._auth_ok = True
                st.rerun()
            else:
                st.warning("Incorrect password. Try again.")
    return False


@st.cache_data(show_spinner=True)
def _load_data(csv_path: str) -> pd.DataFrame:
    dashboard_dir = os.path.dirname(os.path.abspath(__file__))
    return load_paragraph_table(csv_path, dashboard_dir=dashboard_dir)


@st.cache_data(show_spinner=False)
def _corpus_token_counter(csv_path: str) -> Counter[str]:
    """Full-corpus word counts for distinctive-term scoring (cached; one token pass per CSV)."""
    df = _load_data(csv_path)
    return token_counter(df["paragraph"].fillna("").astype(str).tolist())


def _monthly_counts_by_publication(out: pd.DataFrame) -> pd.DataFrame:
    """Long format: ym, publication, n for stacked bar chart."""
    x = out.dropna(subset=["date"]).copy()
    if x.empty:
        return pd.DataFrame(columns=["ym", "publication", "n"])
    x["ym"] = x["date"].dt.to_period("M").dt.to_timestamp()
    g = x.groupby(["ym", "publication"], as_index=False).size().rename(columns={"size": "n"})
    return g


def _monthly_filter_vs_rest_long(df_time_only: pd.DataFrame, out_filtered: pd.DataFrame) -> pd.DataFrame:
    """
    Long format for stacked bars: per month, 'This filter' vs 'Rest of month' (time-window corpus).
    Rest = snippets in that month that are not in the filtered set.
    """
    a = df_time_only.dropna(subset=["date"]).copy()
    if a.empty:
        return pd.DataFrame(columns=["ym", "layer", "n", "layer_sort"])
    a["ym"] = a["date"].dt.to_period("M").dt.to_timestamp()
    n_month = a.groupby("ym").size()

    b = out_filtered.dropna(subset=["date"]).copy()
    if b.empty:
        n_match = pd.Series(dtype=int)
    else:
        b["ym"] = b["date"].dt.to_period("M").dt.to_timestamp()
        n_match = b.groupby("ym").size()

    rows: list[dict] = []
    for ym in n_month.index:
        nm = int(n_month[ym])
        nk = int(n_match[ym]) if ym in n_match.index else 0
        rest = max(0, nm - nk)
        # layer_sort: lower = bottom of stack (rest), higher = top (filter)
        rows.append({"ym": ym, "layer": "This filter", "n": nk, "layer_sort": 1})
        rows.append({"ym": ym, "layer": "Rest of month", "n": rest, "layer_sort": 0})
    return pd.DataFrame(rows)


def _monthly_counts_by_field(
    out: pd.DataFrame, field: str, categories: list[str]
) -> pd.DataFrame:
    """Long format: ym, category, n for stacked bars (only rows where field is in categories)."""
    if not categories:
        return pd.DataFrame(columns=["ym", "category", "n"])
    x = out.dropna(subset=["date"]).copy()
    if x.empty:
        return pd.DataFrame(columns=["ym", "category", "n"])
    x["ym"] = x["date"].dt.to_period("M").dt.to_timestamp()
    x = x[x[field].isin(categories)]
    if x.empty:
        return pd.DataFrame(columns=["ym", "category", "n"])
    g = x.groupby(["ym", field], as_index=False).size().rename(columns={"size": "n", field: "category"})
    return g


def _monthly_filter_breakdown_by_values(
    df_time_only: pd.DataFrame,
    out_filtered: pd.DataFrame,
    *,
    split_field: str,
    split_values: list[str],
) -> pd.DataFrame:
    """
    Per month: 'Rest of month' plus one layer per selected category value (mutually exclusive rows).
    """
    if not split_values:
        return pd.DataFrame(columns=["ym", "layer", "n", "layer_sort"])
    a = df_time_only.dropna(subset=["date"]).copy()
    if a.empty:
        return pd.DataFrame(columns=["ym", "layer", "n", "layer_sort"])
    a["ym"] = a["date"].dt.to_period("M").dt.to_timestamp()
    n_month = a.groupby("ym").size()

    b = out_filtered.dropna(subset=["date"]).copy()
    if b.empty:
        n_by_ym_val = pd.DataFrame()
    else:
        b["ym"] = b["date"].dt.to_period("M").dt.to_timestamp()
        sub = b[b[split_field].isin(split_values)]
        n_by_ym_val = (
            sub.groupby(["ym", split_field]).size().unstack(fill_value=0)
            if len(sub)
            else pd.DataFrame()
        )

    ordered = sorted(split_values)
    rows: list[dict] = []
    for ym in n_month.index:
        nm = int(n_month[ym])
        total_match = 0
        parts: dict[str, int] = {}
        for val in ordered:
            nk = 0
            if not n_by_ym_val.empty and ym in n_by_ym_val.index:
                row = n_by_ym_val.loc[ym]
                nk = int(row[val]) if val in row.index else 0
            parts[val] = nk
            total_match += nk
        rest = max(0, nm - total_match)
        rows.append({"ym": ym, "layer": "Rest of month", "n": rest, "layer_sort": 0})
        for i, val in enumerate(ordered):
            rows.append({"ym": ym, "layer": val, "n": parts[val], "layer_sort": i + 1})
    return pd.DataFrame(rows)


def _compare_accent_for_row(row: pd.Series, archetypes: list[str], stances: list[str]) -> str | None:
    """Left-border color when multiple archetypes or multiple stances are selected."""
    if len(archetypes) > 1:
        av = row.get("archetype")
        if pd.isna(av) or str(av) not in archetypes:
            return None
        ix = sorted(str(x) for x in archetypes).index(str(av))
        return _COMPARE_COLORS[ix % len(_COMPARE_COLORS)]
    if len(stances) > 1:
        sv = row.get("majority_stance")
        if pd.isna(sv) or str(sv) not in stances:
            return None
        ix = sorted(str(x) for x in stances).index(str(sv))
        return _COMPARE_COLORS[ix % len(_COMPARE_COLORS)]
    return None


def _snippet_item_html(
    row: pd.Series,
    *,
    global_index: int,
    archetypes: list[str],
    stances: list[str],
) -> str:
    """One snippet row: large dim index outside the card (left), then the white card."""
    title = str(row.get("title", "") or "Untitled")[:200]
    dt = row.get("date", "")
    dt_s = pd.Timestamp(dt).strftime("%Y-%m-%d") if pd.notna(dt) else ""
    pub_s = str(row.get("publication", "") or "")
    arch_s = str(row.get("archetype", "") or "")
    vis_s = str(row.get("future_type_prediction", "") or "")
    st_s = str(row.get("majority_stance", "") or "")
    para_raw = str(row.get("paragraph", "") or "")
    meta_html = (
        f"{html.escape(dt_s)} · {html.escape(pub_s)} · "
        f"<strong>Vision:</strong> {html.escape(vis_s)} · "
        f"<strong>Archetype:</strong> {html.escape(arch_s)} · "
        f"<strong>Stance:</strong> {html.escape(st_s)}"
    )
    _accent = _compare_accent_for_row(row, archetypes, stances)
    open_div = (
        f'<div class="ave-snippet-card" style="border-left:4px solid {_accent}">'
        if _accent
        else '<div class="ave-snippet-card">'
    )
    return (
        f'<div class="ave-snippet-item" role="article" aria-label="Snippet {global_index}">'
        f'<div class="ave-snippet-num-col" aria-hidden="true">'
        f'<span class="ave-snippet-num">{global_index}</span></div>'
        f'<div class="ave-snippet-card-wrap">'
        f"{open_div}"
        f'<div class="ave-snippet-title">{html.escape(title)}</div>'
        f'<div class="ave-snippet-meta">{meta_html}</div>'
        f'<div class="ave-snippet-body">{html.escape(para_raw)}</div>'
        f"</div></div></div>"
    )


def _inject_branding_css() -> None:
    st.markdown(
        """
<style>
  .ave-filter-card {
    background: linear-gradient(180deg, #ffffff 0%, #f7f4ef 100%);
    border: 1px solid #e0d8ce;
    border-radius: 12px;
    padding: 1rem 1.1rem 1.05rem 1.1rem;
    margin-bottom: 1rem;
  }
  .ave-filter-title {
    font-size: 1.05rem; font-weight: 700; color: #2a3d4f;
    margin: 0 0 0.35rem 0;
  }
  .ave-filter-sub { font-size: 0.88rem; color: #5c6670; margin: 0 0 0.75rem 0; }
  .ave-count-line { font-size: 1.65rem; font-weight: 800; color: #2f4a5e; margin: 0 0 0.65rem 0; }
  .ave-count-line span.muted { font-size: 0.95rem; font-weight: 600; color: #7a7268; }
  .ave-chips { display: flex; flex-wrap: wrap; gap: 0.45rem; align-items: center; }
  .ave-flash-target {
    animation: ave-flash-bg 1.1s ease-out 1;
    border-radius: 8px;
    padding: 2px 4px 8px 4px;
    margin: -2px -4px 4px -4px;
  }
  @keyframes ave-flash-bg {
    0% { background-color: rgba(62, 130, 200, 0.45); }
    100% { background-color: transparent; }
  }
  .ave-main-strip {
    background: linear-gradient(125deg, #2f4a5e 0%, #4a6b82 55%, #6d6048 100%);
    color: #faf8f5;
    padding: 0.4rem 0.75rem;
    border-radius: 8px;
    font-size: 0.95rem;
    font-weight: 600;
    margin: 0 0 0.35rem 0;
    box-shadow: 0 2px 8px rgba(35, 48, 58, 0.12);
  }
  /*
    Snippet nav: default = in-flow card above the snippet list (same layer as the rest of the page).
    When .ave-fixed-nav-visible (after scrolling past #ave-snippet-toolbar-sentinel), pin under app header.
    Flexbox (not grid) so Streamlit column cells never collapse to ~0 width.
  */
  section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) {
    position: relative !important;
    left: auto !important;
    right: auto !important;
    width: 100% !important;
    max-width: min(100%, 72rem) !important;
    margin: 0 auto 0.65rem auto !important;
    z-index: 10 !important;
    background: rgba(250, 248, 245, 0.98) !important;
    backdrop-filter: blur(10px);
    border: 1px solid #d8cfc4 !important;
    border-radius: 12px !important;
    padding: 0.5rem 1rem 0.55rem 1rem !important;
    box-shadow: 0 1px 8px rgba(35, 48, 58, 0.08) !important;
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    align-items: center !important;
    align-content: center !important;
    gap: 0.45rem 0.6rem !important;
    box-sizing: border-box !important;
    opacity: 1 !important;
    pointer-events: auto !important;
    transition: box-shadow 0.2s ease, border-radius 0.2s ease;
  }
  section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar).ave-fixed-nav-visible {
    position: fixed !important;
    top: 3.25rem !important;
    left: var(--ave-sidebar-w, 21rem) !important;
    right: 0 !important;
    width: auto !important;
    max-width: none !important;
    margin: 0 !important;
    z-index: 100002 !important;
    border-radius: 0 !important;
    border-left: none !important;
    border-right: none !important;
    border-top: none !important;
    padding: 0.45rem 1rem 0.5rem 1rem !important;
    box-shadow: 0 2px 12px rgba(35, 48, 58, 0.14) !important;
  }
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div {
    box-sizing: border-box !important;
    flex: 0 1 auto !important;
    min-width: 4.5rem !important;
    max-width: 100% !important;
  }
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(1) {
    flex: 1 1 12rem !important;
    min-width: min(100%, 12rem) !important;
  }
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(2) {
    flex: 0 1 5.5rem !important;
    min-width: 4.75rem !important;
    max-width: 6.5rem !important;
  }
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(3) {
    flex: 1 1 11rem !important;
    min-width: min(100%, 10rem) !important;
    max-width: 26rem !important;
  }
  /* Column 4 = pager cluster (nested row: Prev · page · Next · Top — stays together on wrap) */
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(4) {
    flex: 1 1 17rem !important;
    min-width: min(100%, 14rem) !important;
    max-width: 100% !important;
  }
  /* Nested row lives inside column 4 (not always direct child of column — omit > before stHorizontalBlock) */
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(4) div[data-testid="stHorizontalBlock"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: flex-end !important;
    gap: 0.35rem 0.45rem !important;
    width: 100% !important;
    min-width: min(100%, 17.5rem) !important;
  }
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(4) div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    flex: 0 0 auto !important;
    min-width: 3rem !important;
    max-width: none !important;
  }
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(4) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) {
    flex: 1 1 6.5rem !important;
    min-width: 5.5rem !important;
  }
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(2) .stSelectbox label p,
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(2) .stSelectbox label span {
    color: #64748b !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
  }
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(2) [data-baseweb="select"] {
    opacity: 0.95 !important;
    min-height: 2.1rem !important;
    font-size: 0.85rem !important;
  }
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(3) .stSelectbox label p,
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(3) .stSelectbox label span {
    font-size: 0.78rem !important;
    color: #334155 !important;
  }
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) .stButton > button {
    white-space: nowrap !important;
  }
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) a {
    white-space: nowrap !important;
  }
  .ave-nav-toolbar-topcell {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-height: 2rem !important;
    padding: 0 !important;
    box-sizing: border-box !important;
  }
  .ave-nav-toolbar-topcell .ave-scroll-top {
    color: #2f4a5e !important;
    font-weight: 600 !important;
    text-decoration: none !important;
    font-size: 0.88rem !important;
    flex-shrink: 0 !important;
  }
  .ave-nav-toolbar-page {
    font-size: 0.82rem;
    line-height: 1.35;
    color: #475569;
    text-align: center;
    padding: 0.15rem 0.2rem 0.1rem;
    margin: 0;
  }
  .ave-nav-toolbar-page strong { color: #1e293b; font-weight: 700; }
  /* Wide: full-bleed fixed bar aligned with main column max width */
  @media (min-width: 900px) {
    section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar).ave-fixed-nav-visible {
      padding-left: max(1rem, calc((100% - min(100%, 72rem)) / 2)) !important;
      padding-right: max(1rem, calc((100% - min(100%, 72rem)) / 2)) !important;
    }
    .ave-nav-toolbar-stats {
      max-width: none !important;
    }
  }
  @media (max-width: 768px) {
    .ave-nav-toolbar-stats { font-size: 0.76rem; }
    .ave-nav-toolbar-page { font-size: 0.78rem; }
  }
  .ave-nav-toolbar-stats {
    font-size: 0.8rem;
    line-height: 1.35;
    color: #3d4f5c;
    padding: 0.15rem 0.35rem 0 0;
    text-align: left;
    max-width: 22rem;
    white-space: normal !important;
    word-break: normal !important;
  }
  .ave-nav-toolbar-stats strong { color: #1e293b; font-weight: 700; }
  /* Nested Streamlit column widgets: respect parent min-width (ratios alone can go near-zero) */
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) div[data-testid="column"] {
    flex: 1 1 auto !important;
    width: 100% !important;
    min-width: 4.5rem !important;
  }
  .ave-nav-toolbar-gap {
    width: 100%;
    min-width: 0.5rem;
  }
  @media (max-width: 768px) {
    section[data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar).ave-fixed-nav-visible {
      left: 0 !important;
    }
  }
  .ave-nav-flow-spacer {
    height: 0;
    min-height: 0;
    margin: 0;
    padding: 0;
    pointer-events: none;
    transition: min-height 0.15s ease;
  }
  /* Spacer when fixed nav is visible (matches one vs stacked toolbar height) */
  .ave-nav-flow-spacer.ave-nav-spacer-open {
    height: 7.75rem;
    min-height: 7.75rem;
  }
  @media (min-width: 900px) {
    .ave-nav-flow-spacer.ave-nav-spacer-open {
      height: 4.35rem;
      min-height: 4.35rem;
    }
  }
  @media (max-width: 899px) {
    .ave-nav-flow-spacer.ave-nav-spacer-open {
      height: 7.25rem;
      min-height: 7.25rem;
    }
  }
  .ave-jump-chips {
    margin: 0 0 0.75rem 0;
  }
  /* Stats row: same width as filter chips below (max 72rem); visually merged into one card */
  .ave-chips-dock-stats {
    max-width: min(100%, 72rem);
    margin: 0 auto;
    padding: 0;
    box-sizing: border-box;
  }
  .ave-chip-stats-summary {
    font-size: 0.84rem;
    color: #475569;
    margin: 0;
    padding: 0.55rem 0.85rem 0.6rem 0.85rem;
    background: linear-gradient(180deg, #eef3f9 0%, #e8eef4 100%);
    border: 1px solid #b0c4d8;
    border-bottom: none;
    border-radius: 12px 12px 0 0;
    width: 100%;
    line-height: 1.5;
    box-sizing: border-box;
  }
  .ave-chip-stats-lead {
    margin-right: 0.35rem;
    font-size: 1rem;
    vertical-align: -0.1em;
  }
  .ave-chip-stats-summary span {
    font-weight: 700;
    color: #0f172a;
  }
  .ave-chip-stats-summary em {
    font-style: normal;
    font-weight: 600;
    color: #1d4ed8;
  }
  /* Fixed dock: Filter + Reset — JS assigns fixed to the row containing these buttons */
  .ave-sidebar-dock-fixed {
    position: fixed !important;
    bottom: 0 !important;
    left: var(--ave-sidebar-left, 0px) !important;
    width: var(--ave-sidebar-w, 21rem) !important;
    max-width: min(100vw, var(--ave-sidebar-w, 100vw)) !important;
    z-index: 100006 !important;
    padding: 0.55rem 0.75rem 0.75rem !important;
    margin: 0 !important;
    background: linear-gradient(180deg, rgba(250, 248, 245, 0.98), #f0ebe3) !important;
    border-top: 1px solid #d8cfc4 !important;
    box-shadow: 0 -6px 24px rgba(35, 48, 58, 0.12) !important;
    box-sizing: border-box !important;
  }
  @media (max-width: 768px) {
    .ave-sidebar-dock-fixed { width: 100% !important; left: 0 !important; }
  }
  /* Space so last sidebar expander isn’t hidden behind the dock */
  section[data-testid="stSidebar"] .block-container {
    padding-bottom: 4.25rem !important;
  }
  .ave-sidebar-brand {
    padding: 0.15rem 0 0.85rem 0;
    margin-bottom: 0.5rem;
    border-bottom: 1px solid rgba(0,0,0,0.08);
  }
  .ave-sidebar-brand .sb-row { display: flex; align-items: center; gap: 0.55rem; }
  .ave-sidebar-brand .sb-logo {
    width: 40px; height: 40px; min-width: 40px;
    background: linear-gradient(145deg, #f7f4ef, #e3eef6);
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.35rem;
    border: 1px solid #c5d8e8;
    padding: 3px;
    box-sizing: border-box;
    overflow: hidden;
  }
  .ave-sidebar-brand .sb-logo img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
    border-radius: 7px;
  }
  .ave-main-strip-inner {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
  }
  .ave-strip-icon {
    width: 1.35rem;
    height: 1.35rem;
    object-fit: contain;
    border-radius: 4px;
    flex-shrink: 0;
    vertical-align: middle;
  }
  .ave-sidebar-brand .sb-title { font-weight: 800; font-size: 1.02rem; color: #2f4a5e; margin: 0; line-height: 1.2; }
  .ave-sidebar-brand .sb-sub { font-size: 0.72rem; color: #5c6670; margin: 0.15rem 0 0 0; }
  /* Prevent primary buttons from collapsing to one vertical character in narrow columns */
  section[data-testid="stMain"] .stButton > button {
    white-space: normal !important;
    line-height: 1.3 !important;
  }
  /*
    Filter chips: flat fill (no gradient seam), overlapping borders so the two rows read as one pill.
  */
  section[data-testid="stSidebar"][aria-expanded="false"] {
    pointer-events: none !important;
  }
  /* Blue chip dock: same max-width as stats row above; bottom half of merged “card” */
  section[data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-chips-panel {
    position: relative;
    z-index: 50;
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: wrap !important;
    align-items: stretch !important;
    align-content: flex-start !important;
    gap: 0.45rem !important;
    row-gap: 0.5rem !important;
    max-width: min(100%, 72rem) !important;
    margin-left: auto !important;
    margin-right: auto !important;
    margin-top: 0 !important;
    margin-bottom: 0.65rem !important;
    background: #e8eef4 !important;
    background-image: none !important;
    border: 1px solid #b0c4d8 !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    padding: 0.48rem 0.75rem 0.52rem 0.75rem !important;
    box-shadow: none !important;
  }
  /* Direct children are Streamlit column wrappers (may be element containers) */
  section[data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-chips-panel > div {
    flex: 1 1 9.25rem !important;
    width: auto !important;
    min-width: min(100%, 8.75rem) !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
  }
  section[data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-chips-panel .stButton > button {
    min-height: 3.65rem !important;
    align-items: flex-start !important;
    justify-content: flex-start !important;
    padding: 0.5rem 0.6rem !important;
    background: linear-gradient(180deg, #ffffff 0%, #f4f7fb 100%) !important;
    border: 1px solid #b9ccde !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 3px rgba(30, 58, 95, 0.07) !important;
  }
  section[data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-chips-panel .stButton > button:hover {
    border-color: #8eb4d4 !important;
    box-shadow: 0 2px 8px rgba(30, 58, 95, 0.1) !important;
  }
  section[data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-chips-panel .stButton > button p {
    white-space: pre-line !important;
    text-align: left !important;
    width: 100% !important;
    margin: 0 !important;
    line-height: 1.3 !important;
    font-size: 1.02rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
  }
  section[data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-chips-panel .stButton > button p::first-line {
    font-size: 0.62rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    color: #64748b !important;
  }
  /* Sidebar: scroll target flash (main chips / Filter button) */
  section[data-testid="stSidebar"] details.ave-sidebar-section-flash,
  section[data-testid="stSidebar"] [data-testid="stExpander"].ave-sidebar-section-flash {
    border-radius: 10px;
    outline: 2px solid rgba(59, 130, 246, 0.7);
    outline-offset: 1px;
    background: rgba(219, 234, 254, 0.4) !important;
  }
  section[data-testid="stSidebar"] [data-testid="stVerticalBlock"].ave-sidebar-section-flash {
    border-radius: 10px;
    outline: 2px solid rgba(59, 130, 246, 0.55);
    outline-offset: 1px;
  }
  /* Sidebar: matching count (custom block) */
  section[data-testid="stSidebar"] .ave-sidebar-match {
    padding: 0.2rem 0 0.75rem 0;
    margin: 0 0 0.35rem 0;
    border-bottom: 1px solid rgba(0,0,0,0.06);
  }
  section[data-testid="stSidebar"] .ave-sidebar-match .ave-sm-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #6b7280;
    letter-spacing: 0.02em;
    text-transform: uppercase;
  }
  section[data-testid="stSidebar"] .ave-sidebar-match .ave-sm-num {
    font-size: 1.38rem;
    font-weight: 700;
    color: #4b5563;
    line-height: 1.2;
    margin: 0.15rem 0 0 0;
  }
  section[data-testid="stSidebar"] .ave-sidebar-match .ave-sm-sub {
    font-size: 0.82rem;
    color: #9ca3af;
    margin-top: 0.2rem;
  }
  /* Empty filter state (replaces default blue st.info) */
  .ave-empty-filter {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    max-width: 38rem;
    margin: 0.5rem 0 1rem 0;
    padding: 1.1rem 1.25rem;
    background: linear-gradient(135deg, #eef6fb 0%, #f7f4ef 100%);
    border: 1px solid #b8cfe0;
    border-radius: 14px;
    box-shadow: 0 4px 20px rgba(47, 74, 94, 0.09);
  }
  .ave-empty-filter-icon {
    font-size: 1.75rem;
    line-height: 1;
    width: 2.65rem;
    height: 2.65rem;
    display: flex;
    align-items: center;
    justify-content: center;
    background: linear-gradient(145deg, #e3eef6, #d4e4f0);
    border-radius: 12px;
    border: 1px solid rgba(47, 74, 94, 0.14);
    flex-shrink: 0;
  }
  .ave-empty-filter-body { min-width: 0; }
  .ave-empty-filter-title {
    font-weight: 700;
    font-size: 1.06rem;
    color: #1e3a4f;
    margin: 0 0 0.4rem 0;
    letter-spacing: -0.01em;
  }
  .ave-empty-filter-hint {
    font-size: 0.92rem;
    color: #5c6670;
    margin: 0;
    line-height: 1.55;
  }
  /* Snippet grid: default 1 col; .ave-snippet-grid--two toggled by JS from main width + sidebar */
  .ave-snippet-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 1.35rem 1.5rem;
    width: 100%;
    max-width: 52rem;
    margin: 0 auto;
  }
  .ave-snippet-grid.ave-snippet-grid--two {
    grid-template-columns: 1fr 1fr;
    max-width: min(58rem, 100%);
  }
  .ave-snippet-item {
    display: grid;
    grid-template-columns: minmax(2.75rem, auto) minmax(0, 1fr);
    column-gap: 0.65rem;
    align-items: start;
    min-width: 0;
  }
  .ave-snippet-num-col {
    text-align: right;
    padding-top: 0.15rem;
    line-height: 1;
  }
  .ave-snippet-num {
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: clamp(1.65rem, 2.8vw, 2.35rem);
    font-weight: 500;
    color: rgba(92, 72, 54, 0.28);
    letter-spacing: -0.04em;
    font-variant-numeric: tabular-nums;
    user-select: none;
    pointer-events: none;
  }
  .ave-snippet-card-wrap {
    position: relative;
    min-width: 0;
    margin: 0;
  }
  /* Snippet cards: bright white, body is hero; title/meta de-emphasized */
  .ave-snippet-card {
    background: #ffffff;
    border: 1px solid #ece8e4;
    border-radius: 14px;
    padding: 0.95rem 1.1rem 1.05rem 1.1rem;
    margin: 0;
    box-shadow: 0 1px 8px rgba(45, 42, 38, 0.05);
  }
  .ave-snippet-title {
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 0.88rem;
    font-weight: 500;
    color: #94a3b8;
    margin: 0 0 0.4rem 0;
    line-height: 1.35;
  }
  .ave-snippet-meta {
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 0.74rem;
    color: #9ca3af;
    margin: 0 0 0.85rem 0;
    line-height: 1.45;
  }
  .ave-snippet-body {
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, "Noto Serif", serif;
    font-size: 1.08rem;
    font-weight: 400;
    line-height: 1.78;
    color: #3f3c38;
    margin: 0;
    white-space: pre-wrap;
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def _fixed_nav_scroll_script() -> None:
    """Pin snippet toolbar under app header only after #ave-snippet-toolbar-sentinel scrolls up (main iframe DOM)."""
    components.html(
        """
<script>
(function () {
  function appDocument() {
    try {
      return window.parent.document;
    } catch (e) {
      return document;
    }
  }
  function findToolbar() {
    const doc = appDocument();
    const pin = doc.querySelector("#ave-pin-toolbar");
    if (!pin) return null;
    return pin.closest('[data-testid="stHorizontalBlock"]');
  }
  function tick() {
    const doc = appDocument();
    const sent = doc.getElementById("ave-snippet-toolbar-sentinel");
    const bar = findToolbar();
    const sp = doc.getElementById("ave-nav-flow-spacer");
    if (!sent || !bar || !sp) return;
    const y = sent.getBoundingClientRect().top;
    const show = y < 72;
    bar.classList.toggle("ave-fixed-nav-visible", show);
    sp.classList.toggle("ave-nav-spacer-open", show);
  }
  const doc = appDocument();
  const root = doc.scrollingElement || doc.documentElement;
  window.parent.addEventListener("scroll", tick, true);
  doc.addEventListener("scroll", tick, true);
  root.addEventListener("scroll", tick, true);
  const appView = doc.querySelector('[data-testid="stAppViewContainer"]');
  if (appView) appView.addEventListener("scroll", tick, true);
  const main = doc.querySelector("section.main") || doc.querySelector('[data-testid="stMain"]');
  if (main) main.addEventListener("scroll", tick, true);
  setInterval(tick, 350);
  tick();
})();
</script>
        """,
        height=0,
    )


def _smooth_scroll_top_script() -> None:
    """Smooth-scroll to #ave-top when clicking the fixed snippet bar ↑ Top link (avoids instant hash jump)."""
    components.html(
        """
<script>
(function () {
  function appDocument() {
    try { return window.parent.document; } catch (e) { return document; }
  }
  function findTop() {
    const doc = appDocument();
    let el = doc.getElementById("ave-top");
    if (!el) {
      try { el = document.getElementById("ave-top"); } catch (e) {}
    }
    return el;
  }
  function onClick(ev) {
    const a = ev.target && ev.target.closest && ev.target.closest("a.ave-scroll-top");
    if (!a) return;
    ev.preventDefault();
    const top = findTop();
    if (top) {
      top.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }
  const doc = appDocument();
  doc.addEventListener("click", onClick, true);
  try {
    window.parent.document.addEventListener("click", onClick, true);
  } catch (e) {}
  document.addEventListener("click", onClick, true);
})();
</script>
        """,
        height=0,
    )


def _sidebar_dock_script() -> None:
    """Pin the sidebar row that contains the Filter / Reset all buttons; sync width with resizable sidebar."""
    components.html(
        """
<script>
(function () {
  function appDocument() {
    try { return window.parent.document; } catch (e) { return document; }
  }
  function syncSidebarVars() {
    const doc = appDocument();
    const root = doc.documentElement;
    const sb = doc.querySelector('[data-testid="stSidebar"]');
    if (!sb || !root) return;
    const r = sb.getBoundingClientRect();
    root.style.setProperty("--ave-sidebar-left", r.left + "px");
    root.style.setProperty("--ave-sidebar-w", r.width + "px");
  }
  function dock() {
    syncSidebarVars();
    const doc = appDocument();
    const sb = doc.querySelector('[data-testid="stSidebar"]');
    if (!sb) return;
    const blocks = sb.querySelectorAll('[data-testid="stHorizontalBlock"]');
    for (const row of blocks) {
      const btns = row.querySelectorAll("button");
      let hasFilter = false, hasReset = false;
      for (const b of btns) {
        const t = (b.innerText || "").trim();
        if (t === "Filter") hasFilter = true;
        if (t === "Reset all") hasReset = true;
      }
      if (hasFilter && hasReset) {
        row.classList.add("ave-sidebar-dock-fixed");
        return;
      }
    }
  }
  dock();
  setInterval(dock, 400);
  const doc = appDocument();
  window.parent.addEventListener("resize", syncSidebarVars);
  doc.addEventListener("click", function () { setTimeout(syncSidebarVars, 50); }, true);
  const obs = new MutationObserver(function () { dock(); });
  obs.observe(doc.body, { childList: true, subtree: true });
})();
</script>
        """,
        height=0,
    )


def _v2a_multiselect_mute_script() -> None:
    """Lower opacity for ○ (off-manifold) labels in sidebar multiselects — options and tags."""
    components.html(
        """
<script>
(function () {
  function appDocument() {
    try { return window.parent.document; } catch (e) { return document; }
  }
  function styleEl(el) {
    const t = (el.textContent || "").trim();
    if (t.indexOf("○") === 0) {
      el.style.opacity = "0.45";
      el.style.fontStyle = "italic";
    } else if (t.indexOf("●") === 0) {
      el.style.opacity = "1";
      el.style.fontStyle = "";
    }
  }
  function apply() {
    const doc = appDocument();
    const sb = doc.querySelector('[data-testid="stSidebar"]');
    if (sb) {
      sb.querySelectorAll('[data-baseweb="tag"]').forEach(styleEl);
    }
    /* Dropdown list is often portaled under Base Web popovers */
    doc.querySelectorAll('[data-baseweb="popover"] [role="option"]').forEach(function (el) {
      const t = (el.textContent || "").trim();
      if (t.indexOf("○") === 0 || t.indexOf("●") === 0) {
        styleEl(el);
      }
    });
  }
  apply();
  setInterval(apply, 600);
  const doc = appDocument();
  const obs = new MutationObserver(apply);
  obs.observe(doc.body, { childList: true, subtree: true, attributes: true });
})();
</script>
        """,
        height=0,
    )


def _chip_bar_panel_script() -> None:
    """Tag first main horizontal block (filters + sort) as the blue chip panel."""
    components.html(
        """
<script>
(function () {
  function appDocument() {
    try { return window.parent.document; } catch (e) { return document; }
  }
  function panelize() {
    const doc = appDocument();
    const main = doc.querySelector('[data-testid="stMain"]');
    if (!main) return;
    const blocks = main.querySelectorAll('[data-testid="stHorizontalBlock"]');
    const rows = [];
    for (const hb of blocks) {
      if (hb.querySelector("#ave-pin-toolbar")) continue;
      rows.push(hb);
    }
    for (let i = 0; i < rows.length; i++) {
      const hb = rows[i];
      hb.classList.remove("ave-chips-panel", "ave-sort-chip-row");
      if (i === 0) {
        hb.classList.add("ave-chips-panel");
      }
    }
  }
  panelize();
  setInterval(panelize, 800);
  const doc = appDocument();
  const obs = new MutationObserver(function () { panelize(); });
  obs.observe(doc.body, { childList: true, subtree: true });
})();
</script>
        """,
        height=0,
    )


def _snippet_grid_responsive_script() -> None:
    """Two-column snippet grid only when main column is wide enough; stricter when sidebar is expanded."""
    components.html(
        """
<script>
(function () {
  function appDocument() {
    try { return window.parent.document; } catch (e) { return document; }
  }
  function apply() {
    const doc = appDocument();
    const grid = doc.getElementById("ave-snippet-grid");
    if (!grid) return;
    const main = doc.querySelector('[data-testid="stMain"]');
    if (!main) return;
    const w = main.getBoundingClientRect().width;
    const sb = doc.querySelector('[data-testid="stSidebar"]');
    const expanded = sb && sb.getAttribute("aria-expanded") === "true";
    /* Two columns only when each card column can be comfortably wide (~min 28–30rem total). */
    const threshold = expanded ? 1200 : 1000;
    grid.classList.toggle("ave-snippet-grid--two", w >= threshold);
  }
  apply();
  setInterval(apply, 450);
  const doc = appDocument();
  window.parent.addEventListener("resize", apply, true);
  doc.addEventListener("click", function () { setTimeout(apply, 150); }, true);
  const obs = new MutationObserver(apply);
  obs.observe(doc.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["aria-expanded", "class"] });
})();
</script>
        """,
        height=0,
    )


def _main_chips_expand_sidebar_script() -> None:
    """When sidebar is collapsed, a chip click should expand it (header control lives outside sidebar)."""
    components.html(
        """
<script>
(function () {
  function appDocument() {
    try { return window.parent.document; } catch (e) { return document; }
  }
  function onClick(ev) {
    const doc = appDocument();
    const t = ev.target;
    if (!t || !t.closest) return;
    const btn = t.closest("button");
    if (!btn) return;
    const hb = btn.closest('[data-testid="stHorizontalBlock"]');
    if (!hb || !hb.classList.contains("ave-chips-panel")) return;
    const main = doc.querySelector('[data-testid="stMain"]');
    if (!main || !main.contains(btn)) return;
    const expand = doc.querySelector('[data-testid="stExpandSidebarButton"]');
    if (expand && expand.offsetParent !== null) {
      expand.click();
    }
  }
  const doc = appDocument();
  doc.addEventListener("click", onClick, true);
})();
</script>
        """,
        height=0,
    )


def _sidebar_scroll_to_focus_script(focus_key: str | None) -> None:
    """Scroll sidebar so the focused section is visible; flash-highlight it (also when sidebar already open)."""
    if not focus_key:
        return
    needles = {
        "time": "Time range",
        "primary": "Archetype, stance & vision",
        "keywords": "Keywords",
        "display": "Publication & sort",
    }
    needle = needles.get(focus_key)
    if not needle:
        return
    needle_js = json.dumps(needle)
    components.html(
        f"""
<script>
(function () {{
  const needle = {needle_js};
  function flashEl(node) {{
    if (!node) return;
    node.classList.add("ave-sidebar-section-flash");
    setTimeout(function () {{
      node.classList.remove("ave-sidebar-section-flash");
    }}, 2400);
  }}
  function run() {{
    try {{
      const doc = window.parent.document;
      const content = doc.querySelector('[data-testid="stSidebarContent"]');
      if (!content) return;
      const candidates = content.querySelectorAll("summary, button, span, p, label, div");
      for (const el of candidates) {{
        const text = (el.textContent || "").trim();
        if (text.length > 0 && text.length < 200 && text.includes(needle)) {{
          const details = el.closest("details");
          const expander = el.closest('[data-testid="stExpander"]');
          if (details && !details.open) {{
            details.open = true;
          }} else if (expander) {{
            const sum = expander.querySelector("summary");
            const aria = expander.querySelector('[aria-expanded="false"]');
            if (aria) aria.click();
            else if (sum) sum.click();
          }}
          el.scrollIntoView({{ block: "center", behavior: "smooth" }});
          if (details) {{
            flashEl(details);
          }} else if (expander) {{
            flashEl(expander);
          }} else {{
            const block = el.closest('[data-testid="stVerticalBlock"]');
            if (block) flashEl(block);
          }}
          return;
        }}
      }}
    }} catch (e) {{}}
  }}
  setTimeout(run, 80);
  setTimeout(run, 280);
  setTimeout(run, 600);
  setTimeout(run, 1100);
}})();
</script>
        """,
        height=0,
    )


def _render_keyword_wordcloud(
    texts: list[str],
    *,
    word_weights: dict[str, float] | None = None,
) -> bool:
    """Render a word cloud from paragraph texts or precomputed weights. Returns True if rendered."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        return False
    wc = WordCloud(
        width=840,
        height=340,
        background_color="#faf8f5",
        colormap="Blues",
        max_words=80,
        relative_scaling=0.45,
        min_font_size=10,
    )
    if word_weights:
        wc = wc.generate_from_frequencies(word_weights)
    else:
        blob = " ".join(t for t in texts if t)
        if not blob.strip():
            return False
        wc = wc.generate(blob)
    # Fixed physical size so Streamlit does not stretch the cloud to full viewport width on large displays.
    fig, ax = plt.subplots(figsize=(8.4, 3.4), dpi=100)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    st.pyplot(fig, clear_figure=True, use_container_width=False)
    plt.close(fig)
    return True


def _apply_sort(out: pd.DataFrame, sort_by: str, *, csv_path: str | None = None) -> pd.DataFrame:
    o = out.copy()
    sort_by = _SORT_LABEL_LEGACY.get(sort_by, sort_by)
    if sort_by == "Most distinctive for current filter (vs full corpus)":
        if csv_path is None or len(o) == 0:
            return o
        corpus_counts = _corpus_token_counter(csv_path)
        texts = o["paragraph"].fillna("").astype(str).tolist()
        scores_pairs = distinctive_word_scores(
            texts, corpus_counts, top_n=40, min_count_filt=3
        )
        if not scores_pairs:
            return o.sort_values("date", ascending=False, na_position="last")
        ww = word_weights_for_cloud_from_scores(scores_pairs)
        rep = per_paragraph_representativeness(texts, ww, normalize_length=True)
        return (
            o.assign(_ave_rep=rep)
            .sort_values(["_ave_rep", "date"], ascending=[False, False], na_position="last")
            .drop(columns=["_ave_rep"])
        )
    if sort_by == "Most representative of current filter":
        if len(o) == 0:
            return o
        texts = o["paragraph"].fillna("").astype(str).tolist()
        cos_s = per_paragraph_filter_centroid_cosine(texts)
        return (
            o.assign(_ave_cent=cos_s)
            .sort_values(["_ave_cent", "date"], ascending=[False, False], na_position="last")
            .drop(columns=["_ave_cent"])
        )
    if sort_by == "Date (newest first)":
        return o.sort_values("date", ascending=False, na_position="last")
    if sort_by == "Date (oldest first)":
        return o.sort_values("date", ascending=True, na_position="last")
    if sort_by == "Publication (A–Z)":
        return o.sort_values(["publication", "date"], ascending=[True, False], na_position="last")
    if sort_by == "Title (A–Z)":
        return o.sort_values("title", ascending=True, key=lambda s: s.astype(str).str.lower())
    if sort_by == "Archetype (A–Z)":
        return o.sort_values("archetype", ascending=True, na_position="last")
    return o


def main() -> None:
    _page_icon = _BRAND_ICON_PATH if os.path.isfile(_BRAND_ICON_PATH) else "🔭"
    st.set_page_config(
        page_title="AI Vision Explorer",
        page_icon=_page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if not _check_password():
        return

    csv_path = os.environ.get("DASHBOARD_CSV", default_csv_path(_PROJECT_ROOT))

    try:
        df = _load_data(csv_path)
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()

    para_dup_stats = paragraph_duplicate_metrics(df, csv_path=csv_path, dashboard_dir=_DASHBOARD_DIR)

    # ----- Date bounds & session defaults -----
    date_min = df["date"].min()
    date_max = df["date"].max()
    ymin = int(df["year"].min()) if df["year"].notna().any() else 2014
    ymax = int(df["year"].max()) if df["year"].notna().any() else 2025
    years = list(range(ymin, ymax + 1))
    month_options = list(range(1, 13))

    if "dash_y0" not in st.session_state:
        st.session_state["dash_y0"] = ymin
        st.session_state["dash_m0"] = 1
        st.session_state["dash_y1"] = ymax
        st.session_state["dash_m1"] = 12
    if "dash_d0" not in st.session_state:
        st.session_state["dash_d0"] = date_min.date() if pd.notna(date_min) else None
        st.session_state["dash_d1"] = date_max.date() if pd.notna(date_max) else None
    if "dash_kw" not in st.session_state:
        st.session_state["dash_kw"] = ""
    if "dash_time_mode" not in st.session_state:
        st.session_state["dash_time_mode"] = "Year & month"
    if "dash_page_size" not in st.session_state:
        st.session_state["dash_page_size"] = 50
    if "dash_archetypes" not in st.session_state:
        st.session_state["dash_archetypes"] = []
    if "dash_stances" not in st.session_state:
        st.session_state["dash_stances"] = []
    if "dash_vision" not in st.session_state:
        st.session_state["dash_vision"] = "All"
    if "dash_pub" not in st.session_state:
        st.session_state["dash_pub"] = "All"
    if "dash_sort" not in st.session_state:
        st.session_state["dash_sort"] = "Date (newest first)"
    if "dash_match_all" not in st.session_state:
        st.session_state["dash_match_all"] = False
    if "dash_chart_y_mode" not in st.session_state:
        st.session_state["dash_chart_y_mode"] = "Absolute counts (by publication)"
    if "dash_word_profile" not in st.session_state:
        st.session_state["dash_word_profile"] = "Distinctive vs full corpus"
    if st.session_state.get("dash_chart_y_mode") == "Filter share of month (%)":
        st.session_state["dash_chart_y_mode"] = "Filter vs rest of month (stacked)"
    _chart_mode_opts = frozenset({"Absolute counts (by publication)", "Filter vs rest of month (stacked)"})
    if st.session_state.get("dash_chart_y_mode") not in _chart_mode_opts:
        st.session_state["dash_chart_y_mode"] = "Absolute counts (by publication)"
    _sort_opts = frozenset(SORT_SNIPPET_OPTIONS)
    _ds = st.session_state.get("dash_sort")
    if isinstance(_ds, str) and _ds in _SORT_LABEL_LEGACY:
        st.session_state["dash_sort"] = _SORT_LABEL_LEGACY[_ds]
    if st.session_state.get("dash_sort") not in _sort_opts:
        st.session_state["dash_sort"] = "Date (newest first)"

    def _reset_all_filters() -> None:
        st.session_state["dash_y0"] = ymin
        st.session_state["dash_m0"] = 1
        st.session_state["dash_y1"] = ymax
        st.session_state["dash_m1"] = 12
        if pd.notna(date_min):
            st.session_state["dash_d0"] = date_min.date()
        if pd.notna(date_max):
            st.session_state["dash_d1"] = date_max.date()
        st.session_state["dash_kw"] = ""
        st.session_state["dash_time_mode"] = "Year & month"
        st.session_state["dash_archetypes"] = []
        st.session_state["dash_stances"] = []
        st.session_state["dash_vision"] = "All"
        st.session_state["dash_pub"] = "All"
        st.session_state["dash_sort"] = "Date (newest first)"
        st.session_state["dash_match_all"] = False

    _icon_uri = _brand_icon_data_uri()
    _sb_logo = (
        f'<div class="sb-logo" aria-hidden="true"><img src="{_icon_uri}" alt="" /></div>'
        if _icon_uri
        else '<div class="sb-logo" aria-hidden="true">🔭</div>'
    )
    st.sidebar.markdown(
        f"""
<div class="ave-sidebar-brand">
  <div class="sb-row">
    {_sb_logo}
    <div>
      <p class="sb-title">AI Vision Explorer</p>
      <p class="sb-sub">HBR · MIT SMR</p>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    page_size = int(st.session_state.get("dash_page_size", 50))

    _inject_branding_css()
    _sidebar_dock_script()
    _v2a_multiselect_mute_script()
    _chip_bar_panel_script()
    _snippet_grid_responsive_script()
    _main_chips_expand_sidebar_script()

    focus = st.session_state.get(_FILTER_FOCUS_KEY)

    with st.sidebar:
        with st.expander("🎯 Archetype, stance & vision", expanded=(focus in (None, "primary"))):
            if focus == "primary":
                st.markdown(
                    '<div class="ave-flash-target" style="font-weight:600;color:#1a3344;margin-bottom:0.5rem;">'
                    "Archetype / stance / vision — adjust below</div>",
                    unsafe_allow_html=True,
                )
            _arch_pick = set(st.session_state.get("dash_archetypes", []))
            _stance_pick = set(st.session_state.get("dash_stances", []))
            _vision_prev = st.session_state.get("dash_vision", "All")
            if _stance_pick:
                _compat_ar = compatible_archetypes_for_stances(_stance_pick, _vision_prev)
            elif _vision_prev != "All":
                _compat_ar = compatible_archetypes_for_vision(_vision_prev)
            else:
                _compat_ar = None
            _compat_vis = compatible_visions_for_filters(_arch_pick, _stance_pick)

            def _fmt_arch_label(opt: str) -> str:
                if _compat_ar is None:
                    return opt
                return f"○ {opt}" if opt not in _compat_ar else f"● {opt}"

            def _fmt_vision_label(opt: str) -> str:
                if opt == "All":
                    return "All"
                if _compat_vis is None:
                    return opt
                return f"○ {opt}" if opt not in _compat_vis else f"● {opt}"

            arch_opts = sorted(df["archetype"].dropna().unique().tolist())
            archetypes = st.multiselect(
                "Archetype",
                options=arch_opts,
                key="dash_archetypes",
                format_func=_fmt_arch_label,
                help="Empty = all archetypes. ● = can occur with your current stance selection (given vision scope); "
                "○ = off that mapping (still selectable).",
            )
            vision_opts = ["All"] + sorted(df["future_type_prediction"].dropna().unique().tolist())
            vision = st.selectbox(
                "Vision type",
                vision_opts,
                key="dash_vision",
                format_func=_fmt_vision_label,
                help="Future / division type column. ● = can occur with your current archetype and stance picks "
                "under the table; ○ = off that joint mapping (still selectable). 'All' is unlabeled.",
            )
            if vision != "All" and vision in VISION_TO_ARCHETYPES:
                st.caption(
                    "Archetypes possible for this vision (depends on stance): "
                    + ", ".join(VISION_TO_ARCHETYPES[vision])
                )

            if _arch_pick:
                _compat_st = compatible_stances_for_archetypes(_arch_pick, vision)
            elif vision != "All":
                _compat_st = compatible_stances_for_vision(vision)
            else:
                _compat_st = None

            def _fmt_stance_label(opt: str) -> str:
                if _compat_st is None:
                    return opt
                return f"○ {opt}" if opt not in _compat_st else f"● {opt}"

            stance_opts = sorted(df["majority_stance"].dropna().unique().tolist())
            stances = st.multiselect(
                "Rhetorical stance",
                options=stance_opts,
                key="dash_stances",
                format_func=_fmt_stance_label,
                help="Empty = all stances. ● = can produce your current archetype selection under the chosen vision; "
                "○ = no table row maps those archetypes to this stance (still selectable).",
            )
            if _compat_st is not None or _compat_ar is not None or _compat_vis is not None:
                st.caption(
                    "**●** = consistent with the other picks under vision×stance→archetype · "
                    "**○** = off that mapping (muted); still clickable for strict AND filters."
                )
            with st.expander("Vision → archetypes (reference table)", expanded=False):
                lines = [
                    f"- {v}: " + ", ".join(arches) for v, arches in VISION_TO_ARCHETYPES.items()
                ]
                st.markdown("\n".join(lines))

        with st.expander("⏱ Time range", expanded=(focus == "time")):
            if focus == "time":
                st.markdown(
                    '<div class="ave-flash-target" style="font-weight:600;color:#1a3344;margin-bottom:0.5rem;">'
                    "Time range — adjust below</div>",
                    unsafe_allow_html=True,
                )
            time_mode = st.radio(
                "How do you want to pick dates?",
                ["Year & month", "Exact calendar days"],
                horizontal=True,
                key="dash_time_mode",
                help="Default: year/month. Use exact days only for a specific day range.",
            )

            if time_mode == "Year & month":
                c1, c2 = st.columns(2)
                with c1:
                    y0 = st.selectbox("From · year", years, key="dash_y0")
                with c2:
                    m0 = st.selectbox(
                        "From · month",
                        month_options,
                        format_func=lambda m: calendar.month_abbr[m],
                        key="dash_m0",
                    )
                c3, c4 = st.columns(2)
                with c3:
                    y1 = st.selectbox("To · year", years, key="dash_y1")
                with c4:
                    m1 = st.selectbox(
                        "To · month",
                        month_options,
                        format_func=lambda m: calendar.month_abbr[m],
                        key="dash_m1",
                    )
                date_label = f"{calendar.month_abbr[int(m0)]} {int(y0)} → {calendar.month_abbr[int(m1)]} {int(y1)}"
            else:
                d0 = st.date_input("From (day)", key="dash_d0")
                d1 = st.date_input("To (day)", key="dash_d1")
                date_label = f"{d0} → {d1}"

        with st.expander("🔎 Keywords", expanded=(focus == "keywords")):
            if focus == "keywords":
                st.markdown(
                    '<div class="ave-flash-target" style="font-weight:600;color:#1a3344;margin-bottom:0.5rem;">'
                    "Keywords — adjust below</div>",
                    unsafe_allow_html=True,
                )
            kw = st.text_area(
                "Keywords (paragraph)",
                placeholder='e.g. AI, "generative"',
                height=88,
                key="dash_kw",
            )
            match_all = st.checkbox("Match ALL keywords", key="dash_match_all")

        with st.expander("📰 Publication & sort", expanded=(focus == "display")):
            if focus == "display":
                st.markdown(
                    '<div class="ave-flash-target" style="font-weight:600;color:#1a3344;margin-bottom:0.5rem;">'
                    "Publication & display — adjust below</div>",
                    unsafe_allow_html=True,
                )
            pubs = ["All"] + sorted(df["publication"].dropna().unique().tolist())
            pub = st.selectbox("Publication", pubs, key="dash_pub")
            st.caption(
                "Snippet **sort order** is in the bar above the list (with Previous / Next); it stays visible when you scroll."
            )
    if st.session_state.pop(_SCROLL_SIDEBAR_KEY, False):
        _sidebar_scroll_to_focus_script(st.session_state.get(_FILTER_FOCUS_KEY))

    sort_by = st.session_state.get("dash_sort", "Date (newest first)")

    out_base, df_time_only = _apply_filters_unsorted(
        df, time_mode, list(archetypes), list(stances), vision, pub, kw, match_all
    )
    n_total = len(df)
    n_match = len(out_base)
    n_art_match = int(article_key_series(out_base).nunique()) if n_match else 0
    pct_corpus = (100.0 * n_match / n_total) if n_total else 0.0
    snip_per_article_line = (n_match / n_art_match) if n_art_match else 0.0

    out = _apply_sort(out_base, sort_by, csv_path=csv_path)
    sum_stats = article_summary_stats(out)

    if _icon_uri:
        st.markdown(
            "<div class='ave-main-strip'><span class='ave-main-strip-inner'>"
            f"<img class='ave-strip-icon' src='{_icon_uri}' alt='' /> "
            "<strong>AI Vision Explorer</strong></span></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<div class='ave-main-strip'>🔭 &nbsp; <strong>AI Vision Explorer</strong></div>",
            unsafe_allow_html=True,
        )
    st.markdown('<div id="ave-top" style="height:1px;margin:0;padding:0;"></div>', unsafe_allow_html=True)
    _jm = "ALL" if match_all else "ANY"
    _kw_chip = (kw or "").strip() or "—"
    _vis_chip = vision if vision != "All" else "All"
    st.markdown(
        '<div class="ave-chips-dock-stats"><div class="ave-chip-stats-summary" title="Current filter vs full loaded corpus">'
        '<span class="ave-chip-stats-lead" aria-hidden="true">📊</span>'
        f"<span>{n_match:,}</span> / {n_total:,} snippets"
        f" · <span>{n_art_match:,}</span> articles"
        f" · <em>{pct_corpus:.1f}%</em> of corpus"
        f" · {snip_per_article_line:.2f} snip/article</div></div>",
        unsafe_allow_html=True,
    )
    # Hierarchy: archetype → vision → stance → time → publication → keywords (sort is in snippet toolbar).
    j1, j2, j3, j4, j5, j6 = st.columns([1.15, 1, 1, 1, 1, 1])
    with j1:
        st.button(
            _chip_panel_line("Archetype", _fmt_multi_chip(archetypes), emoji="🎭", max_len=22),
            on_click=_nav_primary_cb,
            key="main_nav_arch",
            use_container_width=True,
        )
    with j2:
        st.button(
            _chip_panel_line("Vision type", _vis_chip, emoji="🔮", max_len=24),
            on_click=_nav_primary_cb,
            key="main_nav_vision",
            use_container_width=True,
        )
    with j3:
        st.button(
            _chip_panel_line("Stance", _fmt_multi_chip(stances), emoji="🗣️", max_len=22),
            on_click=_nav_primary_cb,
            key="main_nav_stance",
            use_container_width=True,
        )
    with j4:
        st.button(
            _chip_panel_line("Time", date_label, emoji="📅", max_len=30),
            on_click=_nav_time_cb,
            key="main_nav_time",
            use_container_width=True,
        )
    with j5:
        st.button(
            _chip_panel_line("Publication", pub, emoji="📰", max_len=22),
            on_click=_nav_display_cb,
            key="main_nav_pub",
            use_container_width=True,
        )
    with j6:
        st.button(
            _chip_panel_line("Keywords", f"{_jm} · {_trunc_ui(_kw_chip, 20)}", emoji="🔎", max_len=26),
            on_click=_nav_keywords_cb,
            key="main_nav_kw",
            use_container_width=True,
        )
    n_articles_match = sum_stats["unique_articles"] if len(out) else 0
    st.sidebar.markdown(
        f'<div class="ave-sidebar-match">'
        f'<div class="ave-sm-label">Matching snippets</div>'
        f'<div class="ave-sm-num">{len(out):,}</div>'
        f'<div class="ave-sm-sub">from {n_articles_match:,} articles</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    sb_d1, sb_d2 = st.sidebar.columns(2)
    with sb_d1:
        st.button("Filter", type="primary", on_click=_filter_open_cb, key="sb_filter", use_container_width=True)
    with sb_d2:
        st.button("Reset all", on_click=_reset_all_filters, key="sb_reset_all", use_container_width=True)

    # Reset page when filters change
    sig = (
        time_mode,
        str(st.session_state.get("dash_y0")),
        str(st.session_state.get("dash_m0")),
        str(st.session_state.get("dash_y1")),
        str(st.session_state.get("dash_m1")),
        str(st.session_state.get("dash_d0")),
        str(st.session_state.get("dash_d1")),
        tuple(sorted(archetypes)),
        tuple(sorted(stances)),
        vision,
        kw,
        match_all,
        pub,
        sort_by,
        page_size,
    )
    if st.session_state.get(_FILTER_SIG_KEY) != sig:
        st.session_state[_FILTER_SIG_KEY] = sig
        st.session_state[_PAGE_KEY] = 1

    n_snippets = len(out)
    n_articles = sum_stats["unique_articles"] if n_snippets > 0 else 0
    snip_per_art = (n_snippets / n_articles) if n_articles else 0.0
    peak_year = None
    peak_year_n = 0
    peak_month_label = None
    peak_month_n = 0
    if n_snippets > 0:
        sub_y = out.dropna(subset=["year"])
        if len(sub_y):
            yc = sub_y.groupby("year").size()
            peak_year = int(yc.idxmax())
            peak_year_n = int(yc.max())
        sub_m = out.dropna(subset=["date"])
        if len(sub_m):
            mc = sub_m.assign(ym=sub_m["date"].dt.to_period("M").astype(str)).groupby("ym").size()
            peak_month_label = str(mc.idxmax())
            peak_month_n = int(mc.max())

    total = n_snippets
    n_pages = max(1, (total + page_size - 1) // page_size)
    if _PAGE_KEY not in st.session_state:
        st.session_state[_PAGE_KEY] = 1
    p = int(st.session_state[_PAGE_KEY])
    p = max(1, min(p, n_pages))
    st.session_state[_PAGE_KEY] = p

    if n_snippets == 0:
        st.caption(f"No snippets match · {n_total:,} in the full dataset · change filters in the sidebar.")
        st.markdown(
            """
<div class="ave-empty-filter" role="status">
  <div class="ave-empty-filter-icon" aria-hidden="true">🔍</div>
  <div class="ave-empty-filter-body">
    <p class="ave-empty-filter-title">No snippets match these filters</p>
    <p class="ave-empty-filter-hint">Widen the date range, clear keywords, or relax archetype, stance, vision, or publication in the sidebar.</p>
  </div>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    _fz_pc = para_dup_stats.get("fuzzy_para_clusters")
    _fz_thr = para_dup_stats.get("fuzzy_para_threshold") or 90
    if _fz_pc is None:
        _para_fuzzy_line = (
            "- **Paragraphs — fuzzy near-duplicates** (different exact text, high similarity; full dataset): "
            "*not precomputed* — run locally `python dashboard/scripts/compute_paragraph_fuzzy_dupes.py` "
            "(writes `dashboard/paragraph_fuzzy_duplicates.json`).\n"
        )
    else:
        _stale_n = (
            " *(precomputed JSON may be stale vs current CSV — re-run the script.)*"
            if para_dup_stats.get("fuzzy_para_file_stale")
            else ""
        )
        _fz_rows = int(para_dup_stats.get("fuzzy_para_rows_in_clusters") or 0)
        _para_fuzzy_line = (
            f"- **Paragraphs — fuzzy near-duplicates** (≥{_fz_thr}% similarity on normalized text; full dataset): "
            f"**{_fz_pc:,}** clusters · **{_fz_rows:,}** snippet rows in those clusters{_stale_n}\n"
        )
    _para_exact_line = (
        "- **Paragraphs — exact duplicate text** (same normalized body on multiple rows; full dataset): "
        f"**{para_dup_stats['para_exact_extra_rows']:,}** redundant rows · "
        f"**{para_dup_stats['para_exact_multi_groups']:,}** distinct paragraph texts appearing 2+ times.\n"
    )

    with st.expander("Summary statistics", expanded=False):
        st.markdown(
            f"- **Unique articles** (distinct article ID): **{sum_stats['unique_articles']:,}**\n"
            f"- **Article IDs with multiple title strings:** **{sum_stats['keys_multi_title']:,}** "
            f"— same corpus ID, different normalized titles (retitling, OCR noise, or listing variants).\n"
            f"- **Exact duplicate titles (different IDs):** **{sum_stats['titles_multi_key']:,}** "
            f"— identical normalized title text under more than one article ID (reprint / duplicate file / split).\n"
            f"- **Fuzzy duplicate titles (different IDs):** **{sum_stats['fuzzy_title_multi_key_groups']:,}** "
            f"groups · **{sum_stats['fuzzy_title_keys_in_groups']:,}** article IDs involved "
            f"(≥88% string similarity after normalizing punctuation/spacing for PDF/OCR variants).\n"
            f"{_para_exact_line}"
            f"{_para_fuzzy_line}"
            f"- **Snippet rows per article ID** in this filter: "
            f"**{sum_stats['articles_1_snippet']:,}** articles with 1 row · "
            f"**{sum_stats['articles_2_snippets']:,}** with 2 · "
            f"**{sum_stats['articles_3_snippets']:,}** with 3 · "
            f"**{sum_stats['articles_4plus_snippets']:,}** with 4+ · "
            f"max **{sum_stats['max_snippets_per_article']}** rows under one ID.\n"
            f"- Snippets per article (mean): **{snip_per_art:.2f}**\n"
            f"- Busiest year: **{peak_year if peak_year is not None else '—'}** ({peak_year_n:,} snippets)\n"
            f"- Busiest month: **{peak_month_label or '—'}** ({peak_month_n:,} snippets)\n"
            f"- Full dataset size: **{n_total:,}** snippet rows"
        )
        st.caption(
            "Multiplicity and reprint counts are heuristics: there is no dedicated reprint flag in the table. "
            "Fuzzy title groups use rapidfuzz (ratio) on normalized text; exact matches are listed separately. "
            "Paragraph duplicate lines use the **full dataset** (not the current filter). "
            "Fuzzy paragraph clusters are optional and CPU-heavy — precompute with `dashboard/scripts/compute_paragraph_fuzzy_dupes.py`."
        )

    if len(archetypes) > 1:
        _split_field, _split_vals = "archetype", sorted(str(x) for x in archetypes)
    elif len(stances) > 1:
        _split_field, _split_vals = "majority_stance", sorted(str(x) for x in stances)
    else:
        _split_field, _split_vals = None, None

    if _split_field:
        monthly_long = _monthly_counts_by_field(out, _split_field, list(_split_vals))
    else:
        monthly_long = _monthly_counts_by_publication(out)
    if _split_field:
        monthly_vs_rest = _monthly_filter_breakdown_by_values(
            df_time_only, out, split_field=_split_field, split_values=list(_split_vals)
        )
    else:
        monthly_vs_rest = _monthly_filter_vs_rest_long(df_time_only, out)

    with st.expander("Chart · Snippets over time", expanded=True):
        y_mode = st.radio(
            "Chart type",
            ["Absolute counts (by publication)", "Filter vs rest of month (stacked)"],
            horizontal=True,
            key="dash_chart_y_mode",
            help="By publication: stacked counts (split by publication, or by archetype/stance when several are "
            "selected). Filter vs rest: each month shows all activity in your date range — compare selected "
            "groups or see filter vs other snippets that month.",
        )
        if _split_field:
            st.caption(
                f"**Compare mode:** {_split_field.replace('_', ' ')} — each selected value has its own color "
                "(same palette as snippet cards)."
            )
        if y_mode == "Filter vs rest of month (stacked)":
            if monthly_vs_rest.empty or monthly_vs_rest["n"].sum() == 0:
                st.caption("No dated rows in the time window.")
            else:
                mvr = monthly_vs_rest.copy()
                if _split_field:
                    layer_order = ["Rest of month"] + list(_split_vals)
                    color_range = ["#c5ddf0"] + [
                        _COMPARE_COLORS[i % len(_COMPARE_COLORS)] for i in range(len(_split_vals))
                    ]
                else:
                    layer_order = ["Rest of month", "This filter"]
                    color_range = ["#c5ddf0", "#1a3d5c"]
                vs_chart = (
                    alt.Chart(mvr)
                    .mark_bar()
                    .encode(
                        x=alt.X("ym:T", title="Month", axis=alt.Axis(format="%Y-%m", labelAngle=-45)),
                        y=alt.Y("n:Q", title="Snippets", stack="zero"),
                        color=alt.Color(
                            "layer:N",
                            title="",
                            sort=layer_order,
                            scale=alt.Scale(domain=layer_order, range=color_range),
                        ),
                        order=alt.Order("layer_sort:Q"),
                        tooltip=[
                            alt.Tooltip("ym:T", title="Month"),
                            alt.Tooltip("layer:N", title="Segment"),
                            alt.Tooltip("n:Q", title="Snippets"),
                        ],
                    )
                    .properties(height=320)
                )
                st.altair_chart(vs_chart, use_container_width=True)
                if _split_field:
                    st.caption(
                        "Within your sidebar date range, each bar is all snippets in that month. "
                        "Colored segments = your selected groups; bottom = other snippets in the same month."
                    )
                else:
                    st.caption(
                        "Within your sidebar date range, each bar is all snippets in that month. "
                        "Dark segment = your current filters; light segment = other snippets in the same month."
                    )
        elif len(monthly_long) == 0:
            st.caption("No dated rows in the filtered set.")
        else:
            ml = monthly_long.copy()
            if _split_field:
                dom = list(_split_vals)
                clr = [_COMPARE_COLORS[i % len(_COMPARE_COLORS)] for i in range(len(dom))]
                chart = (
                    alt.Chart(ml)
                    .mark_bar()
                    .encode(
                        x=alt.X("ym:T", title="Month", axis=alt.Axis(format="%Y-%m", labelAngle=-45)),
                        y=alt.Y("n:Q", title="Snippets", stack="zero"),
                        color=alt.Color(
                            "category:N",
                            title=_split_field.replace("_", " ").title(),
                            scale=alt.Scale(domain=dom, range=clr),
                        ),
                        tooltip=["ym:T", "category", "n"],
                    )
                    .properties(height=320)
                )
            else:
                chart = (
                    alt.Chart(ml)
                    .mark_bar()
                    .encode(
                        x=alt.X("ym:T", title="Month", axis=alt.Axis(format="%Y-%m", labelAngle=-45)),
                        y=alt.Y("n:Q", title="Snippets", stack="zero"),
                        color=alt.Color("publication:N", title="Publication"),
                        tooltip=["ym:T", "publication", "n"],
                    )
                    .properties(height=320)
                )
            st.altair_chart(chart, use_container_width=True)

    texts = out["paragraph"].fillna("").astype(str).tolist()
    corpus_counts = _corpus_token_counter(csv_path)
    with st.expander("Chart · Keyword word cloud (filtered snippets)", expanded=False):
        word_prof = st.radio(
            "Word ranking",
            ["Distinctive vs full corpus", "Raw frequency in filter"],
            horizontal=True,
            key="dash_word_profile",
            help="Distinctive: log-odds vs the whole dataset (under-represented words drop; highlights "
            "what makes this filter different). Raw: most frequent tokens in the filter only.",
        )
        if word_prof.startswith("Distinctive"):
            scores = distinctive_word_scores(texts, corpus_counts, top_n=40, min_count_filt=3)
            wc_weights = word_weights_for_cloud_from_scores(scores) if scores else {}
            bar_rows = scores
            bar_x = "score"
            bar_title = "Log-odds vs corpus (↑ = more characteristic of this filter)"
        else:
            wc_weights = {}
            bar_rows = word_frequencies(texts, top_n=40)
            bar_x = "count"
            bar_title = "Occurrences in filtered snippets"

        if word_prof.startswith("Distinctive") and not bar_rows:
            st.caption("No distinctive terms (empty filter or too little overlap with baseline).")
        elif not bar_rows:
            st.caption("No token counts (empty text).")
        else:
            _wc_l, _wc_m, _wc_r = st.columns([1, 2.2, 1])
            with _wc_m:
                cloud_ok = _render_keyword_wordcloud(
                    texts,
                    word_weights=wc_weights if wc_weights else None,
                )
            if not cloud_ok:
                wf_df = pd.DataFrame(bar_rows, columns=["word", bar_x])
                wchart = (
                    alt.Chart(wf_df)
                    .mark_bar(color="#4a6b82")
                    .encode(
                        x=alt.X(f"{bar_x}:Q", title=bar_title),
                        y=alt.Y("word:N", sort="-x", title=None),
                        tooltip=["word", bar_x],
                    )
                    .properties(height=min(520, 14 * len(wf_df)))
                )
                st.altair_chart(wchart, use_container_width=True)
                st.caption("Install `wordcloud` (see requirements.txt) for the cloud view.")
            if word_prof.startswith("Distinctive") and bar_rows:
                st.caption(
                    "Distinctive terms use smoothed log-odds vs the full corpus (same token rules as raw counts). "
                    "Positive = relatively more common in this filter than overall. "
                    "Sort by **Most distinctive for current filter (vs full corpus)** to prioritize those words; "
                    "by **Most representative of current filter** for excerpts closest to typical wording in the "
                    "current filter (good paper quotes)."
                )

    # Snippet controls: in normal document flow above the list; same row pins under the app header
    # after scrolling past #ave-snippet-toolbar-sentinel (see _fixed_nav_scroll_script).
    st.markdown(
        '<div id="ave-snippet-toolbar-sentinel" style="height:1px;width:100%;margin:0;padding:0" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )
    _nav_stats_html = (
        f"<div class='ave-nav-toolbar-stats'><strong>{n_snippets:,}</strong> snippets · "
        f"<strong>{total:,}</strong> in filter</div>"
    )
    _nav_page_html = (
        f"<div class='ave-nav-toolbar-page'>Page <strong>{p}</strong> of <strong>{n_pages}</strong></div>"
    )
    _nav_top_html = (
        "<div class='ave-nav-toolbar-topcell'>"
        "<a href='#ave-top' class='ave-scroll-top' title='Back to top of page' "
        'aria-label="Back to top of page">↑ Top</a>'
        "</div>"
    )
    _nav_row1_html = (
        "<div class='ave-nav-toolbar-row1'>"
        '<span id="ave-pin-toolbar"></span>'
        f"{_nav_stats_html}"
        "</div>"
    )
    nv1, nv2, nv3, nv4 = st.columns([1.55, 0.82, 1.28, 2.35])
    with nv1:
        st.markdown(_nav_row1_html, unsafe_allow_html=True)
    with nv2:
        st.selectbox(
            "Per page",
            [25, 50, 100, 200],
            key="dash_page_size",
            help="How many snippet cards to show on each page (does not change your filter).",
        )
    with nv3:
        st.selectbox(
            "Order by",
            SORT_SNIPPET_OPTIONS,
            key="dash_sort",
            help=(
                "How snippets are ordered within the current filter. "
                "Distinctive: log-odds for the current filter vs the full corpus. "
                "Representative: closest to typical wording within the current filter."
            ),
        )
    with nv4:
        pc1, pc2, pc3, pc4 = st.columns([0.95, 1.05, 0.95, 0.55])
        with pc1:
            if st.button("Previous", disabled=p <= 1, key="nav_prev", use_container_width=True):
                st.session_state[_PAGE_KEY] = p - 1
                st.rerun()
        with pc2:
            st.markdown(_nav_page_html, unsafe_allow_html=True)
        with pc3:
            if st.button("Next", disabled=p >= n_pages, key="nav_next", use_container_width=True):
                st.session_state[_PAGE_KEY] = p + 1
                st.rerun()
        with pc4:
            st.markdown(_nav_top_html, unsafe_allow_html=True)

    st.markdown(
        '<div id="ave-nav-flow-spacer" class="ave-nav-flow-spacer" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )
    _fixed_nav_scroll_script()
    _smooth_scroll_top_script()

    # ----- Snippet list + table -----
    start = (p - 1) * page_size
    chunk = out.iloc[start : start + page_size]

    st.markdown(
        '<p style="margin:0 0 0.65rem 0;padding:0;font-size:0.95rem;font-weight:600;color:#2d3436">Snippets</p>',
        unsafe_allow_html=True,
    )

    _pad_l, snippet_mid, _pad_r = st.columns([1, 14, 1])
    with snippet_mid:
        rows_list = list(chunk.iterrows())
        _arch_l, _stance_l = list(archetypes), list(stances)
        _cells = [
            _snippet_item_html(
                row,
                global_index=start + i + 1,
                archetypes=_arch_l,
                stances=_stance_l,
            )
            for i, (_, row) in enumerate(rows_list)
        ]
        st.markdown(
            '<div id="ave-snippet-grid" class="ave-snippet-grid">' + "".join(_cells) + "</div>",
            unsafe_allow_html=True,
        )

    st.caption(
        f"Showing snippets {start + 1}–{min(start + page_size, total)} of {total:,}"
    )

    # Tabular view + download (bottom)
    display_cols = [
        "date",
        "publication",
        "title",
        "future_type_prediction",
        "archetype",
        "majority_stance",
        "paragraph",
    ]
    chunk_show = chunk[[c for c in display_cols if c in chunk.columns]].copy()
    if "date" in chunk_show.columns:
        chunk_show["date"] = pd.to_datetime(chunk_show["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    st.subheader("Data (current page)")
    st.dataframe(chunk_show, use_container_width=True, hide_index=True)

    buf = io.StringIO()
    chunk_show.to_csv(buf, index=False)
    st.download_button(
        label="Download current page as CSV",
        data=buf.getvalue(),
        file_name=f"snippets_page{p}.csv",
        mime="text/csv",
    )

    st.caption(f"Data file: `{csv_path}`")


if __name__ == "__main__":
    main()

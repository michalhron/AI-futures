"""
Private academic paragraph explorer (Streamlit).
Set password via .streamlit/secrets.toml or DASHBOARD_PASSWORD env var.
"""
from __future__ import annotations

import base64
import calendar
import contextlib
import html
import io
import json
import os
import re
import sys

import streamlit as st

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as _e:
    if getattr(_e, "name", None) == "matplotlib":
        _root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        _py = os.path.join(_root, ".venv", "bin", "python")
        st.set_page_config(page_title="AI Vision Explorer", layout="wide")
        st.error(
            "**Wrong Python environment.** `matplotlib` is missing because Streamlit was started with a "
            "different Python than this repo’s **`.venv`** (where `pip install -r requirements.txt` was run). "
            "Cursor sometimes uses **`.venv_preview`** — that is a separate env unless you install deps there too."
        )
        st.markdown(
            f"- **Interpreter running Streamlit:** `{sys.executable}`  \n"
            f"- **Project venv (recommended):** `{_py}`"
        )
        st.markdown("**Option A — use the project `.venv` (recommended):**")
        st.code(
            f'cd "{_root}"\n'
            f'"{_py}" -m pip install -r requirements.txt\n'
            f'"{_py}" -m streamlit run dashboard/app.py',
            language="bash",
        )
        st.markdown("**Option B — keep your current interpreter, install deps into it:**")
        st.code(
            f'cd "{_root}"\n'
            f'"{sys.executable}" -m pip install -r requirements.txt\n'
            f'"{sys.executable}" -m streamlit run dashboard/app.py',
            language="bash",
        )
        st.caption("From repo root you can also run: `./run_dashboard.sh` (always uses `.venv/bin/python`).")
        st.stop()
    raise

import altair as alt
import pandas as pd
import streamlit.components.v1 as components

# Project root = parent of dashboard/
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

from collections import Counter  # noqa: E402

from explore_filters import (  # noqa: E402
    DASH_FILTER_FLASH_KEY,
    FILTER_FOCUS_KEY as _FILTER_FOCUS_KEY,
    SCROLL_SIDEBAR_KEY as _SCROLL_SIDEBAR_KEY,
    AVE_SHOW_GUIDE_KEY,
    apply_explore_filter,
)
from guide_content import render_guide_page  # noqa: E402

from data_utils import (  # noqa: E402
    VISION_TO_ARCHETYPES,
    article_key_series,
    article_summary_stats,
    dedupe_sorted_paragraph_rows,
    compatible_archetypes_for_stances_with_visions,
    compatible_archetypes_for_visions_union,
    compatible_stances_for_archetypes_with_visions,
    compatible_stances_for_visions_union,
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


def _html_main_script(html: str) -> None:
    """Inject HTML/JS into the real Streamlit app DOM.

    ``st.components.v1.html`` renders inside a sandboxed iframe; accessing
    ``window.parent.document`` often throws ``SecurityError``, so scripts that
    style or navigate the app never ran. ``st.html(..., unsafe_allow_javascript=True)``
    inserts into the main document (Streamlit >= 1.52).
    """
    html_fn = getattr(st, "html", None)
    if html_fn is not None:
        try:
            html_fn(html, unsafe_allow_javascript=True)
            return
        except TypeError:
            pass
        except Exception:
            pass
    components.html(html, height=0)


def _safe_markdown_html(html: str, *, prefer_stretch: bool = True) -> None:
    """``unsafe_allow_html`` markdown; ``width=`` only exists on Streamlit ≥ ~1.52."""
    if prefer_stretch:
        try:
            st.markdown(html, unsafe_allow_html=True, width="stretch")
            return
        except TypeError:
            pass
    st.markdown(html, unsafe_allow_html=True)


def _render_snippet_block(html: str) -> None:
    """Render one batched HTML block per page (``st.html`` when available, else markdown).

    Huge HTML strings can **stall** the Streamlit session; truncate with a warning so the app
    always returns a response (reduce **Per page** in List options to see full cards).
    """
    if not (html or "").strip():
        return
    html = (html or "").replace("\x00", "")
    _max = 200_000
    if len(html) > _max:
        st.warning(
            f"Snippet block is very large ({len(html):,} characters). "
            f"Choose a smaller **Per page** value under **List options** so the page can render reliably."
        )
        html = html[:_max] + '<p class="ave-snippet-trunc">… truncated …</p>'
    html_fn = getattr(st, "html", None)
    if html_fn is not None:
        try:
            html_fn(html, width="stretch")
            return
        except Exception:
            pass
    _safe_markdown_html(html, prefer_stretch=True)


_DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))
_BRAND_ICON_PATH = os.path.join(_DASHBOARD_DIR, "assets", "app_icon.png")
_BRAND_SVG_PATH = os.path.join(_DASHBOARD_DIR, "assets", "app_icon.svg")
_FAVICON_PATH = os.path.join(_DASHBOARD_DIR, "assets", "favicon.png")

_PAGE_KEY = "dash_page_num"
_FILTER_SIG_KEY = "dash_filter_sig"
_COLLAPSE_SIDEBAR_KEY = "dash_collapse_sidebar"

# Multiselect selections are serialized using format_func output. Our ●/○ labels depend on
# other filters, so when compatibility changes across reruns the frontend can send stale
# formatted strings; Streamlit then returns "● Pioneer" instead of "Pioneer", and
# .isin() matches nothing. Map back to real option values before widgets read state.
_FILTER_MUTE_PREFIX_RE = re.compile(r"^[●○]\s+")
# Some clients send other bullet / invisible chars ahead of the label.
_FILTER_LEADING_JUNK_RE = re.compile(
    r"^[\s\u200b\uFEFF\u2022\u2023\u25E6\u25CB\u25CF\u25AA\u25AB●○◦•\-\*]+"
)


def _sanitize_multiselect_against_options(selected: list, allowed: list[str]) -> list[str]:
    """Map multiselect values (possibly ●/○ prefixed or unicode-mangled) to real CSV option strings."""
    if not selected or not allowed:
        return []
    allow = frozenset(allowed)
    by_lower = {a.lower(): a for a in allowed}
    out: list[str] = []
    for x in selected:
        if x is None:
            continue
        try:
            if pd.isna(x):
                continue
        except (ValueError, TypeError):
            pass
        s = str(x).strip()
        if not s or s.lower() == "nan":
            continue
        resolved: str | None = None
        for candidate in (
            s,
            _FILTER_MUTE_PREFIX_RE.sub("", s).strip(),
            _FILTER_LEADING_JUNK_RE.sub("", s).strip(),
            _FILTER_MUTE_PREFIX_RE.sub("", _FILTER_LEADING_JUNK_RE.sub("", s)).strip(),
        ):
            if not candidate:
                continue
            if candidate in allow:
                resolved = candidate
                break
            low = candidate.lower()
            if low in by_lower:
                resolved = by_lower[low]
                break
        if resolved is not None:
            out.append(resolved)
    return list(dict.fromkeys(out))


def _repair_format_multiselect_state(key: str, allowed: list[str]) -> None:
    raw = st.session_state.get(key)
    if not raw or not isinstance(raw, list):
        return
    fixed = _sanitize_multiselect_against_options(raw, allowed)
    if fixed != raw:
        st.session_state[key] = fixed


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


def _sidebar_apply_filters_cb() -> None:
    """Sidebar primary action: hide the sidebar after the rerun (filters already apply live)."""
    st.session_state[_COLLAPSE_SIDEBAR_KEY] = True


def _trunc_ui(s: str, n: int = 26) -> str:
    t = str(s) if s is not None else ""
    return t if len(t) <= n else t[: n - 1] + "…"


def _fmt_multi_chip(selected: list[str], *, max_len: int = 28) -> str:
    """Label for multiselect filters: empty selection means all options."""
    if not selected:
        return "All"
    s = ", ".join(sorted(str(x) for x in selected))
    return _trunc_ui(s, max_len)


def _apply_ave_explore_query_param(raw: str) -> bool:
    """Parse legacy ``ave_explore=arch:Pioneer`` URLs (full reload may drop sign-in — prefer in-app Explore buttons)."""
    from urllib.parse import unquote

    s = unquote(raw).strip()
    if not s or ":" not in s:
        return False
    kind, _, rest = s.partition(":")
    return apply_explore_filter(kind, rest.strip())


def _normalize_dash_visions(raw: object) -> list[str]:
    """Session value for vision filter: list of labels, or legacy str 'All' / single type."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [] if raw == "All" else [raw]
    if isinstance(raw, list):
        return [str(x) for x in raw if x is not None and str(x).strip()]
    return []


def _time_chip_display(date_label: str, *, max_len: int = 18) -> str:
    """Single-line time summary for the chip (full range stays in the sidebar)."""
    s = str(date_label).replace(" → ", "–").replace("→", "–").strip()
    # Prefer YYYY–YYYY so the value stays one line inside a fixed-height pill
    years = re.findall(r"\b(19\d{2}|20\d{2})\b", s)
    if len(years) >= 2:
        y0, y1 = years[0], years[-1]
        if y0 != y1:
            s = f"{y0}–{y1}"
        else:
            s = y0
    return _trunc_ui(s, max_len)


def _chip_panel_line(
    label: str,
    value: str,
    *,
    emoji: str = "",
    max_len: int = 26,
) -> str:
    """Two-line chip label: `st.button` renders labels as Markdown — use GFM hard break (two spaces + newline)."""
    head = f"{emoji} {label}".strip() if emoji else label
    val = _trunc_ui(value, max_len)
    return f"{head}  \n{val}"


def _apply_filters_unsorted(
    df: pd.DataFrame,
    time_mode: str,
    archetypes: list,
    stances: list,
    visions: list[str],
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
    if visions:
        out = out[out["future_type_prediction"].isin(visions)]
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
    """SVG (preferred) or PNG as data URI for inline HTML (Streamlit cannot load local file URLs)."""
    if os.path.isfile(_BRAND_SVG_PATH):
        with open(_BRAND_SVG_PATH, "rb") as f:
            return "data:image/svg+xml;base64," + base64.b64encode(f.read()).decode("ascii")
    if not os.path.isfile(_BRAND_ICON_PATH):
        return ""
    with open(_BRAND_ICON_PATH, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def _explorer_title_bar_html(icon_uri: str) -> str:
    """Brand + title strip (used alone or inside :func:`_explorer_title_bar_row_html`)."""
    brand = ""
    if os.path.isfile(_BRAND_SVG_PATH):
        with open(_BRAND_SVG_PATH, encoding="utf-8") as f:
            _svg_raw = f.read().strip()
        _svg_raw = _svg_raw.replace("<svg", '<svg class="ave-brand-mark"', 1)
        brand = f'<span class="ave-strip-brand-wrap">{_svg_raw}</span>'
    elif icon_uri:
        safe = html.escape(icon_uri, quote=True)
        brand = (
            f'<span class="ave-strip-brand-wrap ave-strip-brand-wrap--img">'
            f'<img class="ave-strip-icon" src="{safe}" alt="" /></span>'
        )
    if brand:
        inner = f'<span class="ave-main-strip-inner">{brand} <strong>AI Vision Explorer</strong></span>'
    else:
        inner = '<span class="ave-main-strip-inner">🔭 &nbsp; <strong>AI Vision Explorer</strong></span>'
    return f'<div class="ave-main-strip ave-main-strip--in-expl-row">{inner}</div>'


def _explorer_title_bar_row_html(icon_uri: str) -> str:
    """One flex row: title strip + **Explanations** button (JS clicks hidden Streamlit button; avoids page navigation which resets session/auth)."""
    strip = _explorer_title_bar_html(icon_uri)
    # onclick: prevent navigation; click the hidden trigger button in the same document.
    _onclick = (
        "event.preventDefault();"
        "(function(){var b=document.querySelector('[class*=st-key-ave_open_guide] button');"
        "if(b)b.click();}())"
    )
    return (
        f'<div class="ave-expl-title-unified">'
        f'<div class="ave-expl-title-unified__left">{strip}</div>'
        f'<a class="ave-expl-title-unified__link" href="#" role="button" onclick="{_onclick}" aria-label="Open Explanations">Explanations</a>'
        f"</div>"
    )


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
  /*
    Login shell: Streamlit container (key ave_login_shell) wraps hero HTML + password + button
    so the form stays the same width as the logo card (widgets were full column width before).
  */
  [data-testid="stMain"] [class*="st-key-ave_login_shell"] {
    box-sizing: border-box;
    width: 100%;
    max-width: 520px;
    margin-left: auto !important;
    margin-right: auto !important;
    padding: 2rem 1.75rem 1.6rem 1.75rem !important;
    background: linear-gradient(180deg, #ffffff 0%, #f7f4ef 100%);
    border: 1px solid #e0d8ce;
    border-radius: 20px;
    box-shadow: 0 8px 36px rgba(35, 48, 58, 0.12);
  }
  [data-testid="stMain"] [class*="st-key-ave_login_shell"] .stTextInput,
  [data-testid="stMain"] [class*="st-key-ave_login_shell"] .stButton {
    width: 100%;
  }
  .ave-login-hero {
    text-align: center;
    margin-bottom: 1.35rem;
  }
  .ave-login-logo {
    width: 132px; height: 132px; min-width: 132px;
    margin: 0 auto 1.15rem auto;
    border-radius: 28px;
    background: linear-gradient(145deg, #f7f4ef, #e3eef6);
    border: 1px solid #c5d8e8;
    box-shadow: 0 4px 20px rgba(47, 74, 94, 0.12);
    display: flex; align-items: center; justify-content: center;
    overflow: hidden;
  }
  .ave-login-logo img {
    width: 100%; height: 100%; object-fit: contain; display: block; border-radius: 22px; padding: 6px;
    transition: transform 0.2s ease;
  }
  .ave-login-logo:hover img { transform: scale(1.05); }
  .ave-login-logo.ave-login-logo--emoji { font-size: 4.25rem; line-height: 1; padding: 0; }
  .ave-login-title {
    font-size: 1.65rem; font-weight: 800; color: #2f4a5e;
    margin: 0; line-height: 1.2; letter-spacing: -0.03em;
  }
  .ave-login-tagline { font-size: 0.95rem; color: #5c6670; margin: 0.65rem 0 0 0; line-height: 1.55; text-align: left; }
  .ave-login-divider { height: 1px; background: rgba(0,0,0,0.06); margin: 1.15rem 0 1rem 0; }
  .ave-login-actions p { font-size: 0.82rem; color: #6b7280; margin: 0 0 0.75rem 0; }
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
        else '<div class="ave-login-logo ave-login-logo--emoji" aria-hidden="true">🔭</div>'
    )
    _, mid, _ = st.columns([1, 2.35, 1])
    with mid:
        with st.container(key="ave_login_shell"):
            st.markdown(
                f"""
<div class="ave-login-hero">
  {logo_block}
  <p class="ave-login-title">AI Vision Explorer</p>
  <p class="ave-login-tagline">
    Explore AI-related paragraphs from <strong>HBR</strong> and <strong>MIT Sloan Management Review</strong>—filter by time,
    archetype, stance, and vision; scan trends; and read curated excerpts.
  </p>
</div>
<div class="ave-login-divider" aria-hidden="true"></div>
<div class="ave-login-actions">
  <p>Sign in with the shared password.</p>
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
                key="ave_login_pw",
            )
            if st.button(
                "Access the dashboard",
                type="primary",
                use_container_width=True,
                key="ave_login_submit",
            ):
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


def _compare_accent_for_row(
    row: pd.Series,
    archetypes: list[str],
    stances: list[str],
    visions: list[str],
) -> str | None:
    """Left-border color when multiple archetypes, stances, or vision types are selected."""
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
    if len(visions) > 1:
        vv = row.get("future_type_prediction")
        if pd.isna(vv) or str(vv) not in visions:
            return None
        ix = sorted(str(x) for x in visions).index(str(vv))
        return _COMPARE_COLORS[ix % len(_COMPARE_COLORS)]
    return None


def _snippet_card_inner_html(
    row: pd.Series,
    *,
    archetypes: list[str],
    stances: list[str],
    visions: list[str],
) -> str:
    """White card only (title, meta, body) — used by HTML snippet rows and Streamlit-column rows."""
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
    _accent = _compare_accent_for_row(row, archetypes, stances, visions)
    rc = max(1, int(row.get("reprint_count", 1)))
    stack_n = min(rc, 5)
    stack_cls = f" ave-snippet-card--stack{stack_n}" if rc > 1 else ""
    open_div = (
        f'<div class="ave-snippet-card{stack_cls}" style="border-left:4px solid {_accent}">'
        if _accent
        else f'<div class="ave-snippet-card{stack_cls}">'
    )
    return (
        f'<div class="ave-snippet-card-wrap">'
        f"{open_div}"
        f'<div class="ave-snippet-title">{html.escape(title)}</div>'
        f'<div class="ave-snippet-meta">{meta_html}</div>'
        f'<div class="ave-snippet-body">{html.escape(para_raw)}</div>'
        f"</div></div>"
    )


def _snippet_item_html(
    row: pd.Series,
    *,
    global_index: int,
    archetypes: list[str],
    stances: list[str],
    visions: list[str],
    reprint_count: int = 1,
) -> str:
    """One snippet row: large dim index outside the card (left), then the white card (static HTML only).

    **Dedupe** mode with reprints uses :func:`_render_snippet_row_streamlit` instead (×N is a real
    ``st.button`` next to the index). This helper stays for non-dedupe batched ``st.html`` grids.
    """
    rc = max(1, int(reprint_count))
    card_inner = _snippet_card_inner_html(row, archetypes=archetypes, stances=stances, visions=visions)
    if rc <= 1:
        _badge = ""
    else:
        _badge = (
            f'<span class="ave-snippet-reprint-badge" title="This paragraph text appears {rc} times in the current filter">'
            f"×{rc}</span>"
        )
    return (
        f'<div class="ave-snippet-item" role="article" aria-label="Snippet {global_index}">'
        f'<div class="ave-snippet-num-col" aria-hidden="true">'
        f'<span class="ave-snippet-num">{global_index}</span>{_badge}</div>'
        f"{card_inner}</div>"
    )


def _render_snippet_row_streamlit(
    row: pd.Series,
    *,
    global_index: int,
    reprint_button_key: str,
    archetypes: list[str],
    stances: list[str],
    visions: list[str],
    para_norm: str | None,
) -> None:
    """One snippet row: index + optional ×N ``st.button`` + card via ``st.html`` (dedupe / reprints)."""
    _rc = max(1, int(row.get("reprint_count", 1)))
    # Narrow index rail + wide card; small gap keeps number close to card.
    try:
        idx_col, card_col = st.columns([1, 14], gap="small")
    except TypeError:
        idx_col, card_col = st.columns([1, 14])
    with idx_col:
        # Keyed container so CSS can always find the rail (Streamlit may wrap Column > VB with extra nodes).
        _rail_wrap_key = f"idxrail_{reprint_button_key}"

        def _rail_content() -> None:
            st.markdown(
                f'<div class="ave-snippet-num-col ave-snippet-num-col--streamlit ave-snippet-index-rail">'
                f'<span class="ave-snippet-num">{global_index}</span></div>',
                unsafe_allow_html=True,
            )
            if para_norm is None or _rc <= 1:
                return
            try:
                _rp_click = st.button(
                    f"\u00d7{_rc}",
                    key=reprint_button_key,
                    help=f"Show all {_rc} appearances of this paragraph in the current filter",
                    type="tertiary",
                )
            except TypeError:
                _rp_click = st.button(
                    f"\u00d7{_rc}",
                    key=reprint_button_key,
                    help=f"Show all {_rc} appearances of this paragraph in the current filter",
                    type="secondary",
                )
            if _rp_click:
                st.session_state["ave_reprint_open_pn"] = para_norm
                st.rerun()

        try:
            with st.container(key=_rail_wrap_key):
                _rail_content()
        except TypeError:
            _rail_content()
    with card_col:
        _render_snippet_block(
            _snippet_card_inner_html(row, archetypes=archetypes, stances=stances, visions=visions)
        )


def _inject_branding_css() -> None:
    st.markdown(
        """
<style>
  /*
    Align filter dock, summary/chart/cloud, and snippet nav with the snippet list width
    (single wide column). Updated by _snippet_grid_responsive_script.
  */
  :root {
    --ave-explore-snippet-max: min(90rem, 100%);
  }
  /*
    Sidebar expand: match slate/teal top-bar strip (.ave-main-strip); three vertical bars replace chevrons.
  */
  [data-testid="stExpandSidebarButton"] {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: auto !important;
    min-width: 2.5rem !important;
    min-height: 2.25rem !important;
    padding: 0.32rem 0.5rem !important;
    border-radius: 8px !important;
    background: linear-gradient(125deg, rgba(47, 74, 94, 0.92) 0%, rgba(74, 107, 130, 0.88) 55%, rgba(100, 96, 72, 0.82) 100%) !important;
    border: 1px solid rgba(47, 74, 94, 0.28) !important;
    color: #f8fafc !important;
    box-shadow: 0 1px 2px rgba(35, 48, 58, 0.1) !important;
  }
  [data-testid="stExpandSidebarButton"]:hover {
    filter: brightness(1.06);
    border-color: rgba(47, 74, 94, 0.42) !important;
  }
  [data-testid="stExpandSidebarButton"] svg,
  [data-testid="stExpandSidebarButton"] img {
    display: none !important;
  }
  /* Three short vertical bars (menu affordance), replaces Streamlit chevrons */
  [data-testid="stExpandSidebarButton"]::after {
    content: "";
    display: block;
    width: 2px;
    height: 11px;
    background: currentColor;
    border-radius: 1px;
    opacity: 0.92;
    box-shadow: 4px 0 0 currentColor, 8px 0 0 currentColor;
  }
  /* List options → Adjust filters: plain HTML button (client-side only — no Streamlit rerun) */
  button.ave-popover-open-sidebar {
    width: 100%;
    box-sizing: border-box;
    margin: 0;
    padding: 0.375rem 0.75rem;
    border-radius: 0.5rem;
    border: 1px solid rgb(213, 218, 224);
    background: rgb(255, 255, 255);
    color: rgb(49, 51, 63);
    font-size: 0.875rem;
    font-weight: 450;
    font-family: "Source Sans Pro", sans-serif;
    cursor: pointer;
    line-height: 1.4;
  }
  button.ave-popover-open-sidebar:hover {
    border-color: rgb(179, 184, 194);
    color: rgb(49, 51, 63);
  }
  button.ave-popover-open-sidebar:focus-visible {
    outline: 2px solid rgba(47, 74, 94, 0.45);
    outline-offset: 1px;
  }
  /* Main top padding — don’t exaggerate (large gap reads as “empty blue”) */
  [data-testid="stMain"] .block-container {
    padding-top: 3.25rem !important;
  }
  @media (max-width: 768px) {
    [data-testid="stMain"] .block-container {
      padding-top: 3rem !important;
    }
  }
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
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) {
    position: relative !important;
    left: auto !important;
    right: auto !important;
    width: 100% !important;
    max-width: var(--ave-explore-snippet-max) !important;
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
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar).ave-fixed-nav-visible {
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
  /* Col1 = counts; Col2 = pager; Col3 = popover (Per page + Order by) */
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(1) {
    flex: 1 1 10rem !important;
    min-width: min(100%, 9rem) !important;
  }
  /* Pager cluster (nested row: Prev · page · Next · Top) */
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(2) {
    flex: 1 1 12rem !important;
    min-width: min(100%, 11rem) !important;
    max-width: 100% !important;
  }
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(2) div[data-testid="stHorizontalBlock"] {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.3rem 0.4rem !important;
    width: 100% !important;
    min-width: 0 !important;
    max-width: 100% !important;
  }
  /* Direct-child column rules use min-width:0 for desktop flex; on phones they collapse sibling columns — scoped below */
  @media (min-width: 641px) {
    div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(2) div[data-testid="stHorizontalBlock"] > div[data-testid="column"] {
      flex: 0 1 auto !important;
      min-width: 0 !important;
      max-width: none !important;
    }
    div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(2) div[data-testid="stHorizontalBlock"] > div[data-testid="column"]:nth-child(2) {
      flex: 1 1 5rem !important;
      min-width: 4rem !important;
    }
  }
  /* List options popover trigger — compact chip in the bar */
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(3) {
    flex: 0 0 auto !important;
    min-width: 5.5rem !important;
    max-width: 100% !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
  }
  /* List options: match pager row height (Prev/Next use min-height:0 + tight padding) */
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(3) button {
    font-size: 0.8rem !important;
    padding: 0.14rem 0.5rem !important;
    min-height: 0 !important;
    height: auto !important;
    line-height: 1.2 !important;
    white-space: nowrap !important;
  }
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(3) .stButton > button p {
    margin: 0 !important;
    padding: 0 !important;
    line-height: inherit !important;
    font-size: inherit !important;
  }
  /* Popover panel: snippet nav is the only popover in this app */
  [data-testid="stPopover"] .stSelectbox label p,
  [data-testid="stPopover"] .stSelectbox label span {
    font-size: 0.78rem !important;
    color: #334155 !important;
  }
  [data-testid="stPopover"] [data-baseweb="select"] {
    min-height: 1.85rem !important;
    font-size: 0.85rem !important;
  }
  @media (max-width: 640px) {
    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) {
      padding: 0.45rem max(0.75rem, env(safe-area-inset-right)) 0.5rem max(0.75rem, env(safe-area-inset-left)) !important;
      gap: 0.55rem !important;
      align-items: stretch !important;
      flex-direction: column !important;
      flex-wrap: nowrap !important;
      overflow-x: visible !important;
    }
    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar).ave-fixed-nav-visible {
      padding-left: max(0.75rem, env(safe-area-inset-left)) !important;
      padding-right: max(0.75rem, env(safe-area-inset-right)) !important;
    }
    /* Undo global 4.5rem floor on the three top-level cells so the column stack doesn’t clip */
    div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div {
      min-width: 0 !important;
    }
    /* Stack: pager → counts → list options (no overlapping rows) */
    div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(2) {
      order: 1 !important;
      flex: 0 0 auto !important;
      width: 100% !important;
      min-width: 0 !important;
      max-width: 100% !important;
    }
    div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(1) {
      order: 2 !important;
      flex: 0 0 auto !important;
      width: 100% !important;
      min-width: 0 !important;
      max-width: 100% !important;
    }
    div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(3) {
      order: 3 !important;
      flex: 0 0 auto !important;
      width: 100% !important;
      min-width: 0 !important;
      justify-content: stretch !important;
      margin-top: 0 !important;
    }
    div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(3) > div {
      width: 100% !important;
    }
    div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(3) button {
      width: 100% !important;
    }
    .ave-nav-toolbar-stats {
      padding: 0 !important;
      margin: 0 !important;
      max-width: none !important;
    }
    div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(2) div[data-testid="stHorizontalBlock"] {
      flex-wrap: wrap !important;
      justify-content: center !important;
      align-content: center !important;
      gap: 0.35rem 0.45rem !important;
    }
    /*
      Do not let nested pager columns shrink to 0 width (hides Page / Next / Top).
      flex-shrink: 0 + wrap so the row reflows instead of clipping.
    */
    div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(2) [data-testid="column"] {
      flex: 0 0 auto !important;
      flex-shrink: 0 !important;
      min-width: auto !important;
      max-width: 100% !important;
      overflow: visible !important;
    }
    /* Prev/Next: content-sized so one column doesn’t dominate the row */
    div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(2) .stButton > button {
      width: auto !important;
      min-width: 2.75rem !important;
    }
    div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(2) .stButton {
      width: auto !important;
      min-width: 0 !important;
      flex-shrink: 0 !important;
    }
    .ave-nav-toolbar-page {
      padding-left: 0.15rem !important;
      padding-right: 0.15rem !important;
    }
  }
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) .stButton > button {
    white-space: nowrap !important;
  }
  /* Pager Prev/Next: override Streamlit default min-height / padding (was visually very tall) */
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(2) .stButton > button {
    padding: 0.18rem 0.5rem !important;
    min-height: 0 !important;
    height: auto !important;
    font-size: 0.8rem !important;
    line-height: 1.2 !important;
  }
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(2) .stButton > button p {
    margin: 0 !important;
    padding: 0 !important;
    line-height: inherit !important;
    font-size: inherit !important;
  }
  div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) a {
    white-space: nowrap !important;
  }
  .ave-nav-toolbar-topcell {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-height: 1.5rem !important;
    padding: 0 !important;
    box-sizing: border-box !important;
  }
  .ave-nav-toolbar-topcell .ave-scroll-top {
    color: #2f4a5e !important;
    font-weight: 600 !important;
    text-decoration: none !important;
    font-size: 0.8rem !important;
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
    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar).ave-fixed-nav-visible {
      padding-left: max(1rem, calc((100% - var(--ave-explore-snippet-max)) / 2)) !important;
      padding-right: max(1rem, calc((100% - var(--ave-explore-snippet-max)) / 2)) !important;
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
  /* Keep unique count numerically tabular but same weight/size as other stats on the line */
  .ave-nav-toolbar-stats--dedupe .ave-nav-stat-hero {
    font-size: inherit;
    font-weight: 700;
    letter-spacing: normal;
    font-variant-numeric: tabular-nums;
    color: #1e293b;
  }
  .ave-nav-toolbar-stats--dedupe {
    max-width: min(36rem, 100%) !important;
  }
  /*
    Pager row (tablet/desktop): nested columns may sit under a VerticalBlock — descendant selector.
    min-width: 0 only above mobile — on phones a later @media (max-width: 640px) block prevents zero-width columns.
  */
  @media (min-width: 641px) {
    div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(2) [data-testid="column"] {
      flex: 0 1 auto !important;
      min-width: 0 !important;
      max-width: 100% !important;
    }
    div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar) > div:nth-child(2) .stButton > button {
      max-width: 100% !important;
    }
  }
  .ave-nav-toolbar-gap {
    width: 100%;
    min-width: 0.5rem;
  }
  @media (max-width: 768px) {
    [data-testid="stMain"] div[data-testid="stHorizontalBlock"]:has(span#ave-pin-toolbar).ave-fixed-nav-visible {
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
  /* Spacer when fixed nav is visible (toolbar ≈ stats + pager + list-options popover; stacks on narrow viewports) */
  .ave-nav-flow-spacer.ave-nav-spacer-open {
    height: 6.5rem;
    min-height: 6.5rem;
  }
  @media (min-width: 900px) {
    .ave-nav-flow-spacer.ave-nav-spacer-open {
      height: 4rem;
      min-height: 4rem;
    }
  }
  @media (max-width: 640px) {
    .ave-nav-flow-spacer.ave-nav-spacer-open {
      height: 9.75rem;
      min-height: 9.75rem;
    }
  }
  .ave-jump-chips {
    margin: 0 0 0.75rem 0;
  }
  /*
    Filter dock: stats + chips live in `st.container(border=True)` — one small DOM wrapper only.
    Style that wrapper (not generic stVerticalBlock:has, which matched page-wide ancestors).
  */
  .ave-chips-dock-stats {
    width: 100%;
    margin: 0;
    padding: 0;
    box-sizing: border-box;
  }
  /* Blue dock: wrapper, JS class, and inner VerticalBlock (direct child of border — avoids :has() misses) */
  [data-testid="stMain"] div[data-testid="stVerticalBlockBorderWrapper"]:has(.ave-chips-dock-stats),
  [data-testid="stMain"] div[data-testid="stStyledVerticalBlockBorderWrapper"]:has(.ave-chips-dock-stats),
  [data-testid="stMain"] .ave-chips-dock-frame,
  [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"]:has(.ave-chips-dock-stats),
  [data-testid="stMain"] [data-testid="stStyledVerticalBlockBorderWrapper"] > [data-testid="stVerticalBlock"]:has(.ave-chips-dock-stats),
  [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"]:has(.ave-chips-dock-stats),
  [data-testid="stMain"] [data-testid="stStyledVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"]:has(.ave-chips-dock-stats) {
    background: linear-gradient(180deg, #c4daf6 0%, #a8c8f0 52%, #8eb6e8 100%) !important;
    border: 1px solid #5a8ec8 !important;
    border-radius: 12px !important;
    box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.55), 0 1px 2px rgba(30, 58, 95, 0.08) !important;
    padding: 0.45rem 0.75rem 0.6rem 0.75rem !important;
    max-width: var(--ave-explore-snippet-max) !important;
    margin-left: auto !important;
    margin-right: auto !important;
    margin-top: 0.35rem !important;
    margin-bottom: 0.65rem !important;
    box-sizing: border-box !important;
  }
  /*
    Filter chip dock + summary / chart / keyword cloud: same max width as snippet grid (key on st.container).
    Streamlit attaches st-key-* to the block; match any class substring for version tolerance.
  */
  [data-testid="stMain"] [class*="st-key-ave_snippet_width_stack"],
  [data-testid="stMain"] [class*="st-key-ave_chips_dock_width"],
  [data-testid="stMain"] [class*="st-key-ave-chips-dock-width"] {
    max-width: var(--ave-explore-snippet-max) !important;
    width: 100% !important;
    margin-left: auto !important;
    margin-right: auto !important;
    box-sizing: border-box !important;
  }
  /* Filter chip rows: equal-width columns inside the dock */
  [data-testid="stMain"] [class*="st-key-ave_chips_dock_width"] [data-testid="stHorizontalBlock"],
  [data-testid="stMain"] [class*="st-key-ave-chips-dock-width"] [data-testid="stHorizontalBlock"] {
    align-items: stretch !important;
  }
  [data-testid="stMain"] [class*="st-key-ave_chips_dock_width"] [data-testid="stHorizontalBlock"] > div[data-testid="column"],
  [data-testid="stMain"] [class*="st-key-ave-chips-dock-width"] [data-testid="stHorizontalBlock"] > div[data-testid="column"] {
    flex: 1 1 0% !important;
    min-width: 0 !important;
  }
  .ave-chip-stats-summary {
    font-size: 0.84rem;
    color: #475569;
    margin: 0;
    padding: 0 0 0.35rem 0;
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    width: 100%;
    line-height: 1.5;
    box-sizing: border-box;
    position: relative;
    z-index: 2;
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
  .ave-chip-stats-summary--dedupe .ave-chip-stat-hero {
    font-size: inherit;
    font-weight: 700;
    color: #0f172a;
    letter-spacing: normal;
    font-variant-numeric: tabular-nums;
  }
  .ave-chip-stats-summary--dedupe .ave-chip-stat-hero-label {
    font-weight: 600;
    font-size: inherit;
    color: #475569;
  }
  .ave-chip-stats-summary--dedupe .ave-chip-stat-muted {
    font-weight: 600;
    font-size: 0.82rem;
    color: #64748b;
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
    transition: transform 0.2s ease;
  }
  .ave-sidebar-brand .sb-logo:hover img {
    transform: scale(1.06);
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
  /* Vision Explorer title bar: icon on a light tile (readable on dark blue gradient) */
  .ave-strip-brand-wrap {
    display: inline-flex;
    width: 2.35rem;
    height: 2.35rem;
    min-width: 2.35rem;
    min-height: 2.35rem;
    flex-shrink: 0;
    align-items: center;
    justify-content: center;
    vertical-align: middle;
    box-sizing: border-box;
    padding: 3px;
    border-radius: 10px;
    background: linear-gradient(145deg, #ffffff 0%, #f5f0e8 55%, #eef4f9 100%);
    border: 1px solid rgba(255, 255, 255, 0.95);
    box-shadow: 0 1px 5px rgba(15, 30, 45, 0.28), inset 0 1px 0 rgba(255, 255, 255, 0.85);
  }
  .ave-strip-brand-wrap .ave-brand-mark {
    width: 100%;
    height: 100%;
    display: block;
    cursor: default;
    border-radius: 6px;
  }
  /* Ring origins for ecological scale animations (inline SVG + overlay only) */
  .ave-brand-mark .ave-ring-palisade,
  #ave-loading-overlay .ave-loading-overlay__svg-host .ave-ring-palisade {
    transform-origin: 28px 50px;
  }
  .ave-brand-mark .ave-ring-mediator,
  #ave-loading-overlay .ave-loading-overlay__svg-host .ave-ring-mediator {
    transform-origin: 50px 50px;
  }
  .ave-brand-mark .ave-ring-parasite,
  #ave-loading-overlay .ave-loading-overlay__svg-host .ave-ring-parasite {
    transform-origin: 72px 50px;
  }
  @keyframes ave-group-breathe {
    0%, 100% { opacity: 0.9; }
    50% { opacity: 1; }
  }
  @keyframes ave-palisade-eco {
    0%, 100% { transform: scale(1.12); }
    25% { transform: scale(0.86); }
    50% { transform: scale(0.82); }
    75% { transform: scale(0.86); }
  }
  @keyframes ave-parasite-eco {
    0%, 100% { transform: scale(0.82); }
    25% { transform: scale(0.86); }
    50% { transform: scale(1.12); }
    75% { transform: scale(0.86); }
  }
  @keyframes ave-mediator-eco {
    0%, 100% { transform: scale(0.92); }
    25% { transform: scale(1.14); }
    50% { transform: scale(0.92); }
    75% { transform: scale(1.14); }
  }
  /* Default: static rings (sidebar img is static in file; inline mark has no motion until hover/loading) */
  .ave-brand-mark .ave-ring-group,
  .ave-brand-mark .ave-ring-palisade,
  .ave-brand-mark .ave-ring-mediator,
  .ave-brand-mark .ave-ring-parasite {
    animation: none;
  }
  /* Hover: full ecological motion (title bar / any inline .ave-brand-mark) */
  .ave-brand-mark:hover:not(.ave-brand-mark--busy) .ave-ring-group {
    animation: ave-group-breathe 2.85s ease-in-out infinite;
  }
  .ave-brand-mark:hover:not(.ave-brand-mark--busy) .ave-ring-palisade {
    animation: ave-palisade-eco 2.85s ease-in-out infinite;
  }
  .ave-brand-mark:hover:not(.ave-brand-mark--busy) .ave-ring-mediator {
    animation: ave-mediator-eco 2.85s ease-in-out infinite;
  }
  .ave-brand-mark:hover:not(.ave-brand-mark--busy) .ave-ring-parasite {
    animation: ave-parasite-eco 2.85s ease-in-out infinite;
  }
  /* Rerun / busy: fast motion on header icon (see _expl_title_row_script) */
  .ave-brand-mark.ave-brand-mark--busy .ave-ring-group {
    animation: ave-group-breathe 0.36s linear infinite !important;
  }
  .ave-brand-mark.ave-brand-mark--busy .ave-ring-palisade {
    animation: ave-palisade-eco 0.36s linear infinite !important;
  }
  .ave-brand-mark.ave-brand-mark--busy .ave-ring-mediator {
    animation: ave-mediator-eco 0.36s linear infinite !important;
  }
  .ave-brand-mark.ave-brand-mark--busy .ave-ring-parasite {
    animation: ave-parasite-eco 0.36s linear infinite !important;
  }
  /*
    Full-screen loading overlay: injected only into the Streamlit app document (see _loading_overlay_script).
    pointer-events: none so the glass layer never steals clicks (Safari-safe); visual feedback only.
  */
  #ave-loading-overlay {
    position: fixed;
    inset: 0;
    z-index: 2147483000;
    display: flex;
    align-items: center;
    justify-content: center;
    box-sizing: border-box;
    padding: 1.5rem;
    margin: 0;
    border: none;
    background: rgba(248, 250, 252, 0.78);
    backdrop-filter: blur(3px);
    -webkit-backdrop-filter: blur(3px);
    pointer-events: none;
    opacity: 0;
    visibility: hidden;
    transition: opacity 0.22s ease, visibility 0.22s ease;
  }
  #ave-loading-overlay.ave-loading-overlay--visible {
    opacity: 1;
    visibility: visible;
  }
  #ave-loading-overlay:not(.ave-loading-overlay--visible) .ave-loading-overlay__mark {
    animation: none !important;
  }
  #ave-loading-overlay:not(.ave-loading-overlay--visible) .ave-loading-overlay__mark .ave-ring-group,
  #ave-loading-overlay:not(.ave-loading-overlay--visible) .ave-loading-overlay__mark .ave-ring-palisade,
  #ave-loading-overlay:not(.ave-loading-overlay--visible) .ave-loading-overlay__mark .ave-ring-mediator,
  #ave-loading-overlay:not(.ave-loading-overlay--visible) .ave-loading-overlay__mark .ave-ring-parasite {
    animation: none !important;
  }
  .ave-loading-overlay__inner {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.85rem;
    max-width: 22rem;
    text-align: center;
    pointer-events: none;
  }
  .ave-loading-overlay__svg-host {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 112px;
    height: 112px;
    border-radius: 18px;
    background: linear-gradient(145deg, #f7f4ef, #e3eef6);
    border: 1px solid #c5d8e8;
    box-shadow: 0 6px 28px rgba(35, 48, 58, 0.12);
    padding: 10px;
    box-sizing: border-box;
  }
  .ave-loading-overlay__svg-host .ave-loading-overlay__mark {
    width: 100%;
    height: 100%;
    display: block;
  }
  .ave-loading-overlay__msg {
    margin: 0;
    font-family: "Source Sans Pro", sans-serif;
    font-size: 1.05rem;
    font-weight: 600;
    color: #2f4a5e;
    letter-spacing: 0.01em;
    line-height: 1.35;
  }
  .ave-loading-overlay__sub {
    margin: 0;
    font-family: "Source Sans Pro", sans-serif;
    font-size: 0.88rem;
    font-weight: 450;
    color: #5c6670;
    line-height: 1.4;
  }
  .ave-sidebar-brand .sb-title { font-weight: 800; font-size: 1.02rem; color: #2f4a5e; margin: 0; line-height: 1.2; }
  .ave-sidebar-brand .sb-sub { font-size: 0.72rem; color: #5c6670; margin: 0.15rem 0 0 0; }
  .ave-sidebar-brand .sb-guide-hint { font-size: 0.68rem; color: #7a8794; margin: 0.45rem 0 0 0; line-height: 1.35; font-weight: 400; max-width: 14rem; }
  .ave-sidebar-brand .sb-guide-hint-short { display: none; }
  @media (max-width: 768px) {
    .ave-sidebar-brand .sb-guide-hint-full { display: none; }
    .ave-sidebar-brand .sb-guide-hint-short { display: block; max-width: none; }
  }
  .ave-sidebar-filters-heading {
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin: 0 0 0.45rem 0;
    padding: 0;
  }
  /* Prominent entry to the filters panel on phones (row tagged by JS) */
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-mobile-filters-strip {
    display: none !important;
  }
  @media (max-width: 768px) {
    [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-mobile-filters-strip {
      display: flex !important;
      flex-direction: row !important;
      flex-wrap: nowrap !important;
      align-items: stretch !important;
      gap: 0.5rem !important;
      width: 100% !important;
      max-width: min(100%, 72rem) !important;
      margin: 0 auto 0.5rem auto !important;
      padding: 0 !important;
      box-sizing: border-box !important;
    }
    [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-mobile-filters-strip > div {
      flex: 1 1 0 !important;
      min-width: 0 !important;
    }
    [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-mobile-filters-strip .stButton > button {
      min-height: 2.85rem !important;
      font-weight: 650 !important;
      font-size: 0.95rem !important;
      border-radius: 10px !important;
    }
  }
  /*
    Title row: single HTML flex bar (``.ave-expl-title-unified``) — avoids Streamlit column shrink / label wrap.
  */
  [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ave-expl-title-unified) {
    display: block !important;
    width: 100% !important;
    max-width: min(100%, 72rem) !important;
    margin: 0 auto 0.35rem auto !important;
    padding: 0 !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
  }
  [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ave-expl-title-unified) [data-testid="stElementContainer"],
  [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ave-expl-title-unified) [data-testid="element-container"] {
    margin: 0 !important;
    padding: 0 !important;
  }
  .ave-expl-title-unified {
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: space-between !important;
    gap: 0.75rem !important;
    width: 100% !important;
    box-sizing: border-box !important;
    margin: 0 !important;
    padding: 0.5rem 0.85rem !important;
    border-radius: 8px !important;
    /* Slightly richer than the old bar: clearer blue-teal, less muddy brown */
    background: linear-gradient(125deg, #2a4f72 0%, #3d6f9a 48%, #4f6d8a 100%) !important;
    box-shadow: 0 2px 10px rgba(25, 45, 72, 0.18) !important;
  }
  .ave-expl-title-unified__left {
    flex: 1 1 auto !important;
    min-width: 0 !important;
    display: flex !important;
    align-items: center !important;
  }
  .ave-expl-title-unified .ave-main-strip--in-expl-row,
  .ave-expl-title-unified .ave-main-strip.ave-main-strip--in-expl-row {
    background: transparent !important;
    box-shadow: none !important;
    margin: 0 !important;
    padding: 0 !important;
    border-radius: 0 !important;
  }
  .ave-expl-title-unified .ave-main-strip-inner strong {
    line-height: 1.2 !important;
    vertical-align: middle;
  }
  .ave-expl-title-unified__link {
    flex: 0 0 auto !important;
    white-space: nowrap !important;
    text-decoration: none !important;
    margin: 0 !important;
    background: rgba(255, 255, 255, 0.18) !important;
    border: 1px solid rgba(255, 255, 255, 0.58) !important;
    color: #f8fafc !important;
    font-family: "Iowan Old Style", Palatino, Georgia, serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    border-radius: 999px !important;
    padding: 0.4rem 1.1rem !important;
    line-height: 1.25 !important;
    word-break: normal !important;
    overflow-wrap: normal !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.08) !important;
  }
  .ave-expl-title-unified__link:hover {
    background: rgba(255, 255, 255, 0.28) !important;
    border-color: rgba(255, 255, 255, 0.92) !important;
    color: #ffffff !important;
  }
  /* Hidden guide trigger button — JS-clicked only, visually invisible. */
  [data-testid="stMain"] [class*="st-key-ave_open_guide"] {
    position: absolute !important;
    left: -9999px !important;
    top: -9999px !important;
    width: 0 !important;
    height: 0 !important;
    overflow: hidden !important;
    pointer-events: none !important;
  }
  @media (max-width: 560px) {
    .ave-expl-title-unified {
      flex-wrap: wrap !important;
    }
    .ave-expl-title-unified__link {
      width: 100% !important;
      text-align: center !important;
      box-sizing: border-box !important;
    }
  }
  /*
    Explanations view: back row — JS adds .ave-guide-back-row; fixed under app chrome (sticky fails in Streamlit scroll).
    #ave-guide-back-flow-spacer reserves in-flow height so guide body does not sit under the bar.
  */
  #ave-guide-back-flow-spacer {
    width: 100%;
    min-height: 3.25rem;
    margin: 0 0 0.5rem 0;
    pointer-events: none;
    flex-shrink: 0;
  }
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-guide-back-row {
    position: fixed !important;
    top: 3.25rem !important;
    left: var(--ave-sidebar-w, 21rem) !important;
    right: 0 !important;
    width: auto !important;
    max-width: none !important;
    margin: 0 !important;
    z-index: 100002 !important;
    align-self: stretch !important;
    padding: 0.55rem 1rem 0.6rem 1rem !important;
    box-sizing: border-box !important;
    background: linear-gradient(180deg, #f8f6f3 0%, #f4f1ec 100%) !important;
    border: 1px solid #e8e4dc !important;
    border-radius: 0 !important;
    border-left: none !important;
    border-right: none !important;
    border-top: none !important;
    box-shadow: 0 2px 12px rgba(45, 42, 38, 0.1) !important;
  }
  @media (min-width: 900px) {
    [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-guide-back-row {
      padding-left: max(1rem, calc((100% - min(100%, 72rem)) / 2)) !important;
      padding-right: max(1rem, calc((100% - min(100%, 72rem)) / 2)) !important;
    }
  }
  @media (max-width: 768px) {
    [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-guide-back-row {
      left: 0 !important;
    }
  }
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-guide-back-row > div[data-testid="column"] {
    min-width: 0 !important;
  }
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-guide-back-row .stButton > button {
    background: #ffffff !important;
    border: 1px solid #c4b8a8 !important;
    color: #1e293b !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    border-radius: 999px !important;
    padding: 0.4rem 1.05rem !important;
    white-space: nowrap !important;
    box-shadow: 0 1px 2px rgba(30, 41, 59, 0.06) !important;
  }
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-guide-back-row .stButton > button:hover {
    border-color: #94a3b8 !important;
    background: #fafaf8 !important;
  }
  /* Prevent primary buttons from collapsing to one vertical character in narrow columns */
  [data-testid="stMain"] .stButton > button {
    white-space: normal !important;
    line-height: 1.3 !important;
  }
  /*
    Collapsed sidebar: do **not** set pointer-events: none — it has broken click-through to the
    main column in some Streamlit / browser builds (widgets look loaded but do not respond).
  */
  /* Chip rows inside bordered dock — no second box; wrapper supplies the blue frame */
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-chips-panel {
    position: relative;
    z-index: 1;
    display: flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: stretch !important;
    align-content: stretch !important;
    gap: 0.45rem !important;
    width: 100% !important;
    min-width: 0 !important;
    max-width: 100% !important;
    margin: 0 !important;
    background: transparent !important;
    border: none !important;
    padding: 0.15rem 0 !important;
    box-shadow: none !important;
    box-sizing: border-box !important;
  }
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-chips-panel--first {
    margin: 0 !important;
    padding-bottom: 0.25rem !important;
  }
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-chips-panel--last {
    margin: 0 !important;
    padding-top: 0.15rem !important;
    padding-bottom: 0 !important;
  }
  /* Streamlit: stHorizontalBlock > element-container > column > … — flex the outer wrappers */
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-chips-panel > div {
    flex: 1 1 0 !important;
    width: auto !important;
    min-width: 0 !important;
    max-width: 100% !important;
    box-sizing: border-box !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
    align-self: stretch !important;
  }
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-chips-panel > div [data-testid="column"] {
    flex: 1 1 auto !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: stretch !important;
    min-width: min(100%, 9.25rem) !important;
    width: 100% !important;
    min-height: 0 !important;
  }
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-chips-panel [data-testid="column"] .stButton {
    flex: 1 1 auto !important;
    display: flex !important;
    width: 100% !important;
    min-height: 0 !important;
    align-items: stretch !important;
  }
  /* Same box height for every chip — stops ragged tops when one value wraps */
  /* Exclude Explanations “Explore” widgets (keys st-key-gx_ex_*) — broad :has() rules would otherwise style them like chips */
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-chips-panel [data-testid="stElementContainer"]:not([class*="st-key-gx_ex_"]) .stButton > button,
  [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"]:has(.ave-chips-dock-stats) [data-testid="stHorizontalBlock"] [data-testid="stElementContainer"]:not([class*="st-key-gx_ex_"]) .stButton > button,
  [data-testid="stMain"] [data-testid="stStyledVerticalBlockBorderWrapper"]:has(.ave-chips-dock-stats) [data-testid="stHorizontalBlock"] [data-testid="stElementContainer"]:not([class*="st-key-gx_ex_"]) .stButton > button {
    flex: 0 0 auto !important;
    height: 4.5rem !important;
    min-height: 4.5rem !important;
    max-height: 4.5rem !important;
    box-sizing: border-box !important;
    align-items: flex-start !important;
    justify-content: flex-start !important;
    padding: 0.45rem 0.55rem !important;
    overflow: hidden !important;
    background: linear-gradient(180deg, #ffffff 0%, #f4f7fb 100%) !important;
    border: 1px solid #b9ccde !important;
    border-radius: 10px !important;
    box-shadow: 0 1px 3px rgba(30, 58, 95, 0.07) !important;
  }
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-chips-panel [data-testid="stElementContainer"]:not([class*="st-key-gx_ex_"]) .stButton > button:hover,
  [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"]:has(.ave-chips-dock-stats) [data-testid="stHorizontalBlock"] [data-testid="stElementContainer"]:not([class*="st-key-gx_ex_"]) .stButton > button:hover,
  [data-testid="stMain"] [data-testid="stStyledVerticalBlockBorderWrapper"]:has(.ave-chips-dock-stats) [data-testid="stHorizontalBlock"] [data-testid="stElementContainer"]:not([class*="st-key-gx_ex_"]) .stButton > button:hover {
    border-color: #8eb4d4 !important;
    box-shadow: 0 2px 8px rgba(30, 58, 95, 0.1) !important;
  }
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-chips-panel [data-testid="stElementContainer"]:not([class*="st-key-gx_ex_"]) .stButton > button p,
  [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"]:has(.ave-chips-dock-stats) [data-testid="stHorizontalBlock"] [data-testid="stElementContainer"]:not([class*="st-key-gx_ex_"]) .stButton > button p,
  [data-testid="stMain"] [data-testid="stStyledVerticalBlockBorderWrapper"]:has(.ave-chips-dock-stats) [data-testid="stHorizontalBlock"] [data-testid="stElementContainer"]:not([class*="st-key-gx_ex_"]) .stButton > button p {
    white-space: pre-line !important;
    text-align: left !important;
    width: 100% !important;
    margin: 0 !important;
    line-height: 1.22 !important;
    font-size: 1.14rem !important;
    font-weight: 700 !important;
    color: #0f172a !important;
    overflow: hidden !important;
    word-break: break-word !important;
  }
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-chips-panel [data-testid="stElementContainer"]:not([class*="st-key-gx_ex_"]) .stButton > button p::first-line,
  [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"]:has(.ave-chips-dock-stats) [data-testid="stHorizontalBlock"] [data-testid="stElementContainer"]:not([class*="st-key-gx_ex_"]) .stButton > button p::first-line,
  [data-testid="stMain"] [data-testid="stStyledVerticalBlockBorderWrapper"]:has(.ave-chips-dock-stats) [data-testid="stHorizontalBlock"] [data-testid="stElementContainer"]:not([class*="st-key-gx_ex_"]) .stButton > button p::first-line {
    font-size: 0.58rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #64748b !important;
  }
  /* Time chip: years read larger than the small “TIME” label */
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-chips-panel--first > div:nth-child(4) .stButton > button p {
    font-size: 1.28rem !important;
    letter-spacing: 0.02em !important;
  }
  [data-testid="stMain"] div[data-testid="stHorizontalBlock"].ave-chips-panel--first > div:nth-child(4) .stButton > button p::first-line {
    font-size: 0.56rem !important;
    letter-spacing: 0.1em !important;
  }
  [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"]:has(.ave-chips-dock-stats) [data-testid="stHorizontalBlock"]:first-of-type > div:nth-child(4) [data-testid="stElementContainer"]:not([class*="st-key-gx_ex_"]) .stButton > button p,
  [data-testid="stMain"] [data-testid="stStyledVerticalBlockBorderWrapper"]:has(.ave-chips-dock-stats) [data-testid="stHorizontalBlock"]:first-of-type > div:nth-child(4) [data-testid="stElementContainer"]:not([class*="st-key-gx_ex_"]) .stButton > button p {
    font-size: 1.28rem !important;
    letter-spacing: 0.02em !important;
  }
  [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"]:has(.ave-chips-dock-stats) [data-testid="stHorizontalBlock"]:first-of-type > div:nth-child(4) [data-testid="stElementContainer"]:not([class*="st-key-gx_ex_"]) .stButton > button p::first-line,
  [data-testid="stMain"] [data-testid="stStyledVerticalBlockBorderWrapper"]:has(.ave-chips-dock-stats) [data-testid="stHorizontalBlock"]:first-of-type > div:nth-child(4) [data-testid="stElementContainer"]:not([class*="st-key-gx_ex_"]) .stButton > button p::first-line {
    font-size: 0.56rem !important;
    letter-spacing: 0.1em !important;
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
  /*
    Snippet grid: use container width (st.html iframe), not viewport — @media 1024px never fired
    inside a narrow iframe on MacBook with sidebar open. auto-fill + minmax gives 2 cols when the
    main block is wide enough.
  */
  .ave-snippet-grid.ave-snippet-grid-root {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(min(100%, 22rem), 1fr));
    gap: 1.85rem 2rem;
    width: 100%;
    max-width: min(90rem, 100%);
    margin: 0 auto;
    box-sizing: border-box;
  }
  /* Dedupe mode: explicit row gap between 2-up pairs (Streamlit ignores many margin hacks) */
  .ave-snippet-dedupe-pair-spacer {
    height: 2.5rem;
    min-height: 2.5rem;
    margin: 0;
    padding: 0;
    pointer-events: none;
  }
  /*
    2-up dedupe layout: outer HorizontalBlock (pair of snippets) — breathing room between rows.
    Selector: first column’s VerticalBlock directly wraps an inner HorizontalBlock (snippet [1,14]).
  */
  [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(
    > div[data-testid="column"]:first-child > [data-testid="stVerticalBlock"] > [data-testid="stHorizontalBlock"]
  ) {
    margin-bottom: 1.85rem !important;
  }
  /*
    Snippet cards use Streamlit columns; default stretch makes the short column as tall as the tall one,
    so the next row “jumps” unevenly. Align pair rows and inner index|card rows to the top.
  */
  [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ave-snippet-card-wrap) {
    align-items: flex-start !important;
  }
  [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ave-snippet-card-wrap) > div[data-testid="column"] {
    align-self: flex-start !important;
  }
  [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ave-snippet-num-col) {
    align-items: flex-start !important;
  }
  [data-testid="stMain"] [data-testid="stHorizontalBlock"]:has(.ave-snippet-num-col) > div[data-testid="column"] {
    align-self: flex-start !important;
  }
  /*
    Dedupe index rail: large index (plain) + tiny pill on ×N only; column stays narrow.
  */
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child:has(> [data-testid="stVerticalBlock"]:has(.ave-snippet-index-rail):not(:has(> [data-testid="stHorizontalBlock"]))) {
    flex: 0 0 3.15rem !important;
    width: 3.15rem !important;
    max-width: 3.15rem !important;
    min-width: 2.55rem !important;
    padding-right: 0.55rem !important;
    box-sizing: border-box !important;
    padding-top: 0.12rem !important;
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-end !important;
    gap: 0.2rem !important;
  }
  /* Inner VerticalBlock: stack index + ×N only — no outer box */
  [data-testid="stMain"] [data-testid="stVerticalBlock"]:has(.ave-snippet-index-rail):not(:has(> [data-testid="stHorizontalBlock"])) {
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-end !important;
    gap: 0.12rem !important;
    width: 100% !important;
    box-sizing: border-box !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
  }
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child:has(> [data-testid="stVerticalBlock"]:has(.ave-snippet-index-rail):not(:has(> [data-testid="stHorizontalBlock"]))) [data-testid="stMarkdownContainer"],
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child:has(> [data-testid="stVerticalBlock"]:has(.ave-snippet-index-rail):not(:has(> [data-testid="stHorizontalBlock"]))) [data-testid="stMarkdownContainer"] p {
    margin-bottom: 0 !important;
  }
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child:has(> [data-testid="stVerticalBlock"]:has(.ave-snippet-index-rail):not(:has(> [data-testid="stHorizontalBlock"]))) .stButton {
    width: auto !important;
    margin: 0 !important;
    padding: 0 !important;
  }
  /* ×N only: small pill; lighter neutral ink than the big index */
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child:has(> [data-testid="stVerticalBlock"]:has(.ave-snippet-index-rail):not(:has(> [data-testid="stHorizontalBlock"]))) .stButton > button {
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif !important;
    font-size: 0.62rem !important;
    font-weight: 400 !important;
    color: rgba(198, 192, 184, 0.78) !important;
    background: rgba(255, 255, 255, 0.96) !important;
    background-color: rgba(255, 255, 255, 0.96) !important;
    border: 1px solid rgba(236, 230, 222, 0.88) !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    border-radius: 999px !important;
    padding: 0.08rem 0.38rem !important;
    line-height: 1.15 !important;
    letter-spacing: 0.02em !important;
    min-height: 0 !important;
    height: auto !important;
    max-height: none !important;
    width: auto !important;
    min-width: 0 !important;
    display: inline-flex !important;
    flex-direction: row !important;
    flex-wrap: nowrap !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0 !important;
    -webkit-appearance: none !important;
    appearance: none !important;
    opacity: 1 !important;
  }
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child:has(> [data-testid="stVerticalBlock"]:has(.ave-snippet-index-rail):not(:has(> [data-testid="stHorizontalBlock"]))) .stButton > button p {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.1 !important;
    white-space: nowrap !important;
    font-size: inherit !important;
    font-weight: inherit !important;
    color: inherit !important;
    display: inline !important;
  }
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child:has(> [data-testid="stVerticalBlock"]:has(.ave-snippet-index-rail):not(:has(> [data-testid="stHorizontalBlock"]))) .stButton button[data-testid^="stBaseButton"] {
    min-height: unset !important;
  }
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child:has(> [data-testid="stVerticalBlock"]:has(.ave-snippet-index-rail):not(:has(> [data-testid="stHorizontalBlock"]))) .stButton > button:hover {
    filter: none !important;
    border-color: rgba(220, 214, 206, 0.98) !important;
    color: rgba(175, 169, 162, 0.88) !important;
    background: rgba(255, 255, 255, 1) !important;
    background-color: rgba(255, 255, 255, 1) !important;
  }
  [data-testid="stMain"] [data-testid="stHorizontalBlock"] > div[data-testid="column"]:first-child:has(> [data-testid="stVerticalBlock"]:has(.ave-snippet-index-rail):not(:has(> [data-testid="stHorizontalBlock"]))) .stButton > button:focus-visible {
    outline: 2px solid #4a6b82;
    outline-offset: 2px;
  }
  [data-testid="stMain"] [data-testid="stElementContainer"][class*="st-key-ave_reprint_badge"] .stButton > button,
  [data-testid="stMain"] [class*="st-key-ave_reprint_badge"] .stButton > button {
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif !important;
    font-size: 0.62rem !important;
    font-weight: 400 !important;
    color: rgba(198, 192, 184, 0.78) !important;
    background: rgba(255, 255, 255, 0.96) !important;
    background-color: rgba(255, 255, 255, 0.96) !important;
    border: 1px solid rgba(236, 230, 222, 0.88) !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    border-radius: 999px !important;
    padding: 0.08rem 0.38rem !important;
    line-height: 1.15 !important;
    letter-spacing: 0.02em !important;
    min-height: 0 !important;
    height: auto !important;
    width: auto !important;
    display: inline-flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: center !important;
    -webkit-appearance: none !important;
    appearance: none !important;
  }
  [data-testid="stMain"] [data-testid="stElementContainer"][class*="st-key-ave_reprint_badge"] .stButton > button p,
  [data-testid="stMain"] [class*="st-key-ave_reprint_badge"] .stButton > button p {
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.1 !important;
    white-space: nowrap !important;
    font-size: inherit !important;
    font-weight: inherit !important;
    color: inherit !important;
    display: inline !important;
  }
  [data-testid="stMain"] [data-testid="stElementContainer"][class*="st-key-ave_reprint_badge"] .stButton > button:hover,
  [data-testid="stMain"] [class*="st-key-ave_reprint_badge"] .stButton > button:hover {
    filter: none !important;
    border-color: rgba(220, 214, 206, 0.98) !important;
    color: rgba(175, 169, 162, 0.88) !important;
    background: rgba(255, 255, 255, 1) !important;
    background-color: rgba(255, 255, 255, 1) !important;
  }
  /*
    Index rail wrap: st.container(key="idxrail_*") — theme-safe ×N pill (Streamlit often wraps Column>VBlock so :> chains miss).
  */
  [data-testid="stMain"] [data-testid="stElementContainer"][class*="st-key-idxrail_"] .stButton > button,
  [data-testid="stMain"] [class*="st-key-idxrail_"] .stButton > button {
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif !important;
    font-size: 0.62rem !important;
    font-weight: 400 !important;
    color: rgba(198, 192, 184, 0.78) !important;
    background: rgba(255, 255, 255, 0.97) !important;
    background-color: rgba(255, 255, 255, 0.97) !important;
    border: 1px solid rgba(236, 230, 222, 0.9) !important;
    border-radius: 999px !important;
    padding: 0.1rem 0.42rem !important;
    line-height: 1.15 !important;
    letter-spacing: 0.02em !important;
    min-height: 0 !important;
    height: auto !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06) !important;
    width: auto !important;
    display: inline-flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: center !important;
    -webkit-appearance: none !important;
    appearance: none !important;
  }
  [data-testid="stMain"] [class*="st-key-idxrail_"] .stButton > button p,
  [data-testid="stMain"] [class*="st-key-idxrail_"] .stButton > button span {
    color: rgba(198, 192, 184, 0.78) !important;
    -webkit-text-fill-color: rgba(198, 192, 184, 0.78) !important;
  }
  [data-testid="stMain"] [class*="st-key-idxrail_"] .stButton > button * {
    color: rgba(198, 192, 184, 0.78) !important;
    -webkit-text-fill-color: rgba(198, 192, 184, 0.78) !important;
  }
  [data-testid="stMain"] [class*="st-key-idxrail_"] .stButton > button:hover {
    filter: none !important;
    border-color: rgba(220, 214, 206, 0.98) !important;
    color: rgba(175, 169, 162, 0.88) !important;
    background: rgba(255, 255, 255, 1) !important;
    background-color: rgba(255, 255, 255, 1) !important;
  }
  [data-testid="stMain"] [class*="st-key-idxrail_"] .stButton > button:hover *,
  [data-testid="stMain"] [class*="st-key-idxrail_"] .stButton > button:hover p,
  [data-testid="stMain"] [class*="st-key-idxrail_"] .stButton > button:hover span {
    color: rgba(175, 169, 162, 0.88) !important;
    -webkit-text-fill-color: rgba(175, 169, 162, 0.88) !important;
  }
  /* Fallback: markdown rail + button are siblings (Streamlit uses stElementContainer or element-container) */
  [data-testid="stMain"] [data-testid="stMarkdownContainer"]:has(.ave-snippet-index-rail) ~ [data-testid="stElementContainer"] .stButton > button,
  [data-testid="stMain"] [data-testid="stMarkdownContainer"]:has(.ave-snippet-index-rail) ~ [data-testid="element-container"] .stButton > button {
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif !important;
    font-size: 0.62rem !important;
    font-weight: 400 !important;
    color: rgba(198, 192, 184, 0.78) !important;
    background: rgba(255, 255, 255, 0.97) !important;
    background-color: rgba(255, 255, 255, 0.97) !important;
    border: 1px solid rgba(236, 230, 222, 0.9) !important;
    border-radius: 999px !important;
    padding: 0.1rem 0.42rem !important;
    min-height: 0 !important;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.06) !important;
    -webkit-appearance: none !important;
    appearance: none !important;
  }
  [data-testid="stMain"] [data-testid="stMarkdownContainer"]:has(.ave-snippet-index-rail) ~ [data-testid="stElementContainer"] .stButton > button p,
  [data-testid="stMain"] [data-testid="stMarkdownContainer"]:has(.ave-snippet-index-rail) ~ [data-testid="element-container"] .stButton > button p {
    color: rgba(198, 192, 184, 0.78) !important;
    -webkit-text-fill-color: rgba(198, 192, 184, 0.78) !important;
  }
  [data-testid="stMain"] [data-testid="stMarkdownContainer"]:has(.ave-snippet-index-rail) ~ [data-testid="stElementContainer"] .stButton > button *,
  [data-testid="stMain"] [data-testid="stMarkdownContainer"]:has(.ave-snippet-index-rail) ~ [data-testid="element-container"] .stButton > button * {
    color: rgba(198, 192, 184, 0.78) !important;
    -webkit-text-fill-color: rgba(198, 192, 184, 0.78) !important;
  }
  [data-testid="stMain"] [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"]:has(.ave-snippet-index-rail):not(:has(> [data-testid="stHorizontalBlock"])) {
    display: flex !important;
    flex-direction: column !important;
    align-items: flex-end !important;
    gap: 0.12rem !important;
    width: 100% !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
  }
  .ave-snippet-num-col--streamlit {
    width: 100%;
    align-items: flex-end;
    margin: 0 !important;
    padding-bottom: 0 !important;
    line-height: 1 !important;
  }
  .ave-snippet-item {
    display: grid;
    grid-template-columns: minmax(2.75rem, auto) minmax(0, 1fr);
    column-gap: 0.95rem;
    align-items: start;
    min-width: 0;
  }
  .ave-snippet-num-col {
    display: flex;
    flex-direction: column;
    align-items: flex-end;
    gap: 0.28rem;
    text-align: right;
    padding-top: 0.15rem;
    line-height: 1;
  }
  .ave-snippet-reprint-badge {
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: 0.62rem;
    font-weight: 400;
    color: rgba(198, 192, 184, 0.78);
    background: rgba(255, 255, 255, 0.96);
    border: 1px solid rgba(236, 230, 222, 0.85);
    border-radius: 999px;
    padding: 0.08rem 0.38rem;
    line-height: 1.15;
    letter-spacing: 0.02em;
    user-select: none;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.04);
  }
  a.ave-snippet-reprint-badge--link {
    text-decoration: none;
    color: inherit;
    cursor: pointer;
    display: inline-block;
    transition: filter 0.12s ease, border-color 0.12s ease;
  }
  a.ave-snippet-reprint-badge--link:hover {
    filter: brightness(0.97);
    border-color: #d4c8bc;
  }
  a.ave-snippet-reprint-badge--link:focus-visible {
    outline: 2px solid #4a6b82;
    outline-offset: 2px;
  }
  /* Big index = primary rail anchor (darker than ×N multiplicity). */
  .ave-snippet-num {
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    font-size: clamp(1.65rem, 2.8vw, 2.35rem);
    font-weight: 500;
    color: rgba(88, 74, 64, 0.5);
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
  /* Stacked “sheets” — visible steps but lighter so the rail + ×N don’t sit on near-black slabs */
  .ave-snippet-card.ave-snippet-card--stack2 {
    box-shadow:
      0 1px 8px rgba(45, 42, 38, 0.06),
      6px 6px 0 #e4e0da,
      13px 13px 0 #d0cbc4;
  }
  .ave-snippet-card.ave-snippet-card--stack3 {
    box-shadow:
      0 1px 8px rgba(45, 42, 38, 0.06),
      6px 6px 0 #e4e0da,
      13px 13px 0 #d0cbc4,
      20px 20px 0 #bcb6af;
  }
  .ave-snippet-card.ave-snippet-card--stack4 {
    box-shadow:
      0 1px 8px rgba(45, 42, 38, 0.06),
      6px 6px 0 #e4e0da,
      13px 13px 0 #d0cbc4,
      20px 20px 0 #bcb6af,
      27px 27px 0 #a8a29a;
  }
  .ave-snippet-card.ave-snippet-card--stack5 {
    box-shadow:
      0 1px 8px rgba(45, 42, 38, 0.06),
      6px 6px 0 #e4e0da,
      13px 13px 0 #d0cbc4,
      20px 20px 0 #bcb6af,
      27px 27px 0 #a8a29a,
      34px 34px 0 #949088;
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
    hyphens: none;
    overflow-wrap: break-word;
    word-break: normal;
  }
</style>
        """,
        unsafe_allow_html=True,
    )


def _fixed_nav_scroll_script() -> None:
    """Pin snippet toolbar under app header only after #ave-snippet-toolbar-sentinel scrolls up (main iframe DOM)."""
    _html_main_script(
        """
<script>
(function () {
  function appDocument() {
    try {
      var pd = window.parent && window.parent.document;
      if (pd && pd.querySelector('[data-testid="stMain"]')) return pd;
    } catch (e) {}
    return document;
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
    )


def _smooth_scroll_top_script() -> None:
    """Smooth-scroll to #ave-top when clicking the fixed snippet bar ↑ Top link (avoids instant hash jump)."""
    _html_main_script(
        """
<script>
(function () {
  function appDocument() {
    try {
      var pd = window.parent && window.parent.document;
      if (pd && pd.querySelector('[data-testid="stMain"]')) return pd;
    } catch (e) {}
    return document;
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
    )


def _sidebar_dock_script() -> None:
    """Pin the sidebar row that contains the Filter / Reset all buttons; sync width with resizable sidebar."""
    _html_main_script(
        """
<script>
(function () {
  function appDocument() {
    try {
      var pd = window.parent && window.parent.document;
      if (pd && pd.querySelector('[data-testid="stMain"]')) return pd;
    } catch (e) {}
    return document;
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
        if (t.includes("Apply filters") || t === "Filter") hasFilter = true;
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
  if (doc.body) {
    obs.observe(doc.body, { childList: true, subtree: true });
  }
})();
</script>
        """,
    )


def _v2a_multiselect_mute_script() -> None:
    """Lower opacity for ○ (off-manifold) labels in sidebar multiselects — options and tags."""
    _html_main_script(
        """
<script>
(function () {
  function appDocument() {
    try {
      var pd = window.parent && window.parent.document;
      if (pd && pd.querySelector('[data-testid="stMain"]')) return pd;
    } catch (e) {}
    return document;
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
  if (doc.body) {
    obs.observe(doc.body, { childList: true, subtree: true, attributes: true });
  }
})();
</script>
        """,
    )


def _guide_back_row_script() -> None:
    """Tag the back row, sync sidebar width for fixed layout, match spacer height to the bar (same idea as snippet toolbar)."""
    _html_main_script(
        r"""
<script>
(function () {
  function appDocument() {
    try {
      var pd = window.parent && window.parent.document;
      if (pd && pd.querySelector('[data-testid="stMain"]')) return pd;
    } catch (e) {}
    return document;
  }
  function syncSidebarVars() {
    const doc = appDocument();
    const root = doc.documentElement;
    const sb = doc.querySelector('[data-testid="stSidebar"]');
    if (!root) return;
    if (!sb) {
      root.style.setProperty("--ave-sidebar-w", "0px");
      return;
    }
    const r = sb.getBoundingClientRect();
    root.style.setProperty("--ave-sidebar-left", r.left + "px");
    root.style.setProperty("--ave-sidebar-w", r.width + "px");
  }
  function mark() {
    syncSidebarVars();
    const doc = appDocument();
    const main = doc.querySelector('[data-testid="stMain"]');
    if (!main) return;
    for (const b of main.querySelectorAll("button")) {
      const t = (b.textContent || "").replace(/\s+/g, " ").trim();
      if (!t.includes("Back to dashboard")) continue;
      const hb = b.closest('[data-testid="stHorizontalBlock"]');
      if (hb) hb.classList.add("ave-guide-back-row");
      break;
    }
    syncGuideSpacer();
  }
  function syncGuideSpacer() {
    const doc = appDocument();
    const bar = doc.querySelector(".ave-guide-back-row");
    const sp = doc.getElementById("ave-guide-back-flow-spacer");
    if (!bar || !sp) return;
    const h = Math.ceil(bar.getBoundingClientRect().height);
    if (h > 0) {
      sp.style.height = h + "px";
      sp.style.minHeight = h + "px";
    }
  }
  mark();
  setInterval(function () { mark(); }, 320);
  const doc = appDocument();
  if (doc.body) {
    new MutationObserver(mark).observe(doc.body, { childList: true, subtree: true });
  }
  window.parent.addEventListener("resize", function () { mark(); });
  doc.addEventListener("resize", function () { mark(); }, true);
})();
</script>
        """,
    )


def _filter_flash_script(flash: str | None) -> None:
    """After Explore-from-guide: briefly pulse the matching filter chip and sidebar control."""
    if not flash:
        return

    fj = json.dumps(flash)
    _html_main_script(
        """
<script>
(function () {
  const FLASH = """
        + fj
        + """;
  function appDocument() {
    try {
      var pd = window.parent && window.parent.document;
      if (pd && pd.querySelector('[data-testid="stMain"]')) return pd;
    } catch (e) {}
    return document;
  }
  function injectCss() {
    const doc = appDocument();
    if (doc.getElementById("ave-filter-flash-style")) return;
    const s = doc.createElement("style");
    s.id = "ave-filter-flash-style";
    s.textContent = `
@keyframes ave-filter-chip-pulse {
  0%, 100% { box-shadow: inset 0 0 0 0 rgba(37,99,235,0), 0 0 0 0 rgba(37,99,235,0); }
  35% { box-shadow: inset 0 0 0 2px rgba(37,99,235,0.55), 0 0 18px rgba(37,99,235,0.35); }
  65% { box-shadow: inset 0 0 0 1px rgba(37,99,235,0.3), 0 0 10px rgba(37,99,235,0.22); }
}
@keyframes ave-filter-sidebar-pulse {
  0%, 100% { background-color: transparent; }
  30% { background-color: rgba(59, 130, 246, 0.2); }
  100% { background-color: transparent; }
}
.ave-filter-flash-chip {
  animation: ave-filter-chip-pulse 0.75s ease-in-out 3 !important;
  border-radius: 10px !important;
}
.ave-filter-flash-sidebar {
  animation: ave-filter-sidebar-pulse 1.05s ease-out 2 !important;
  border-radius: 10px !important;
}
`;
    doc.head.appendChild(s);
  }
  /* Same idea as _chip_bar_panel_script — do not rely on .ave-chips-panel--first (that class is applied later). */
  function findPrimaryChipRow(doc) {
    const main = doc.querySelector('[data-testid="stMain"]');
    if (!main) return null;
    const blocks = main.querySelectorAll('[data-testid="stHorizontalBlock"]');
    for (const hb of blocks) {
      if (hb.querySelector("#ave-pin-toolbar")) continue;
      const cols = hb.querySelectorAll('[data-testid="column"]');
      if (cols.length < 4) continue;
      const b0 = cols[0].querySelector("button");
      if (!b0) continue;
      const t0 = (b0.textContent || "").replace(/\\s+/g, " ").trim();
      const b3 = cols[3].querySelector("button");
      const t3 = b3 ? (b3.textContent || "").replace(/\\s+/g, " ").trim() : "";
      if (/\\bARCHETYPE\\b/i.test(t0) && /\\bTIME\\b/i.test(t3)) return hb;
    }
    return null;
  }
  let chipDone = false;
  let sidebarDone = false;
  function pulseOnce() {
    injectCss();
    const doc = appDocument();
    const colIdx = { archetype: 0, vision: 1, stance: 2 };
    const idx = colIdx[FLASH];
    if (idx === undefined) return;
    const row = findPrimaryChipRow(doc);
    if (row && !chipDone) {
      const cols = row.querySelectorAll('[data-testid="column"]');
      const col = cols[idx];
      const btn = col && col.querySelector("button");
      if (btn) {
        chipDone = true;
        btn.classList.add("ave-filter-flash-chip");
        setTimeout(function () { btn.classList.remove("ave-filter-flash-chip"); }, 3200);
      }
    }
    const sb = doc.querySelector('[data-testid="stSidebar"]');
    if (sb && !sidebarDone) {
      const needles = { archetype: "Archetype", vision: "Vision type", stance: "Rhetorical stance" };
      const needle = needles[FLASH];
      if (needle) {
        const nodes = sb.querySelectorAll("label, p, span");
        for (const el of nodes) {
          const raw = (el.textContent || "").trim();
          const first = raw.split("\n")[0].trim();
          if (first !== needle) continue;
          let w = el.closest('[data-testid="stWidget"]');
          if (!w) w = el.closest("div[data-baseweb]");
          if (!w) w = el.parentElement;
          if (w) {
            sidebarDone = true;
            w.classList.add("ave-filter-flash-sidebar");
            setTimeout(function () { w.classList.remove("ave-filter-flash-sidebar"); }, 2600);
          }
          break;
        }
      }
    }
  }
  const delays = [0, 120, 350, 700, 1200, 2000, 3200, 5000];
  for (let i = 0; i < delays.length; i++) {
    setTimeout(pulseOnce, delays[i]);
  }
})();
</script>
        """,
    )


def _expl_title_row_script() -> None:
    """Mark the title + Explanations row so CSS can style the two columns as one gradient bar."""
    _html_main_script(
        """
<script>
(function () {
  function appDocument() {
    try {
      var pd = window.parent && window.parent.document;
      if (pd && pd.querySelector('[data-testid="stMain"]')) return pd;
    } catch (e) {}
    return document;
  }
  function mark() {
    const doc = appDocument();
    const main = doc.querySelector('[data-testid="stMain"]');
    if (!main) return;
    const uni = main.querySelector(".ave-expl-title-unified");
    if (!uni) return;
    const hb = uni.closest('[data-testid="stHorizontalBlock"]');
    if (hb) hb.classList.add("ave-expl-title-row");
  }
  mark();
  setInterval(mark, 1200);
  const doc = appDocument();
  if (doc.body) {
    new MutationObserver(mark).observe(doc.body, { childList: true, subtree: true });
  }
})();
</script>
        """,
    )


def _loading_overlay_script() -> None:
    """Centered glass + pulsating three-ring SVG while Streamlit reruns (cold load and filter changes)."""
    svg_inner = ""
    if os.path.isfile(_BRAND_SVG_PATH):
        with open(_BRAND_SVG_PATH, encoding="utf-8") as f:
            _raw = f.read().strip()
        svg_inner = _raw.replace(
            "<svg",
            '<svg class="ave-brand-mark ave-brand-mark--busy ave-loading-overlay__mark" aria-hidden="true"',
            1,
        )
    _svg_json = json.dumps(svg_inner)
    _html_main_script(
        """
<script>
(function () {
  const SVG_INNER = """
        + _svg_json
        + r""";
  if (window.__aveLoadingOverlayInit) {
    return;
  }
  window.__aveLoadingOverlayInit = true;
  function appDocument() {
    try {
      var pd = window.parent && window.parent.document;
      if (pd && pd.querySelector('[data-testid="stMain"]')) return pd;
    } catch (e) {}
    return document;
  }
  function ensureOverlay(doc) {
    var el = doc.getElementById("ave-loading-overlay");
    if (el) return el;
    el = doc.createElement("div");
    el.id = "ave-loading-overlay";
    el.setAttribute("role", "status");
    el.setAttribute("aria-live", "polite");
    el.setAttribute("aria-busy", "true");
    var inner = doc.createElement("div");
    inner.className = "ave-loading-overlay__inner";
    var host = doc.createElement("div");
    host.className = "ave-loading-overlay__svg-host";
    if (SVG_INNER) {
      host.innerHTML = SVG_INNER;
    } else {
      host.textContent = "…";
    }
    var msg = doc.createElement("p");
    msg.className = "ave-loading-overlay__msg";
    msg.textContent = "Working…";
    var sub = doc.createElement("p");
    sub.className = "ave-loading-overlay__sub";
    sub.textContent = "Updating the dashboard";
    inner.appendChild(host);
    inner.appendChild(msg);
    inner.appendChild(sub);
    el.appendChild(inner);
    doc.body.appendChild(el);
    return el;
  }
  function isStreamlitRunning(doc) {
    var st = doc.querySelector('[data-testid="stStatusWidget"]');
    if (st) {
      var t = (st.textContent || "").toLowerCase();
      if (t.indexOf("running") >= 0) return true;
    }
    if (doc.querySelector('[data-testid="stHeader"] [data-testid="stSpinner"]')) return true;
    return false;
  }
  function dashboardReady(doc) {
    return !!doc.querySelector('[data-testid="stMain"] .ave-expl-title-unified');
  }
  var visible = false;
  var pollTimer = null;
  function setVisible(doc, on) {
    var el = ensureOverlay(doc);
    if (on === visible) return;
    visible = on;
    if (on) el.classList.add("ave-loading-overlay--visible");
    else el.classList.remove("ave-loading-overlay--visible");
    el.setAttribute("aria-busy", on ? "true" : "false");
  }
  function clearPoll() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }
  /** Wait until Streamlit status is idle for a few ticks (avoids flicker before "Running" appears). */
  function waitIdleThenHide() {
    clearPoll();
    var steady = 0;
    pollTimer = setInterval(function () {
      var d = appDocument();
      if (isStreamlitRunning(d)) {
        steady = 0;
        return;
      }
      steady++;
      if (steady < 3) return;
      setVisible(d, false);
      clearPoll();
    }, 48);
  }
  function boot() {
    var doc = appDocument();
    ensureOverlay(doc);
    setVisible(doc, true);
    var ticks = 0;
    var steady = 0;
    var iv = setInterval(function () {
      ticks++;
      var d = appDocument();
      if (dashboardReady(d) && !isStreamlitRunning(d)) {
        steady++;
      } else {
        steady = 0;
      }
      if (steady >= 4) {
        setVisible(d, false);
        clearInterval(iv);
        return;
      }
      if (ticks > 900) {
        setVisible(d, false);
        clearInterval(iv);
      }
    }, 48);
  }
  function onRerunExpected() {
    var doc = appDocument();
    setVisible(doc, true);
    waitIdleThenHide();
  }
  function isWidgetInteraction(el) {
    if (!el || el.nodeType !== 1) return false;
    var tag = (el.tagName || "").toLowerCase();
    if (tag === "button" || tag === "input" || tag === "textarea" || tag === "select") return true;
    var role = el.getAttribute && el.getAttribute("role");
    if (role === "option" || role === "listbox" || role === "combobox" || role === "menuitem" || role === "switch" || role === "menuitemcheckbox") return true;
    if (el.closest && el.closest("[data-baseweb]")) return true;
    if (el.closest && el.closest(".stSlider")) return true;
    return false;
  }
  function onPointerDown(ev) {
    var t = ev.target;
    if (!t || !t.closest) return;
    if (!t.closest('[data-testid="stAppViewContainer"]') && !t.closest('[data-testid="stSidebar"]')) return;
    if (t.closest("button.ave-popover-open-sidebar")) return;
    var cur = t;
    for (var i = 0; i < 10 && cur; i++) {
      if (isWidgetInteraction(cur)) {
        onRerunExpected();
        return;
      }
      cur = cur.parentElement;
    }
  }
  boot();
  document.addEventListener("pointerdown", onPointerDown, true);
  try {
    var pd = window.parent && window.parent.document;
    if (pd && pd !== document) pd.addEventListener("pointerdown", onPointerDown, true);
  } catch (e0) {}
})();
</script>
        """,
    )


def _dock_blue_paint_script() -> None:
    """Force the filter dock card to the light-blue gradient (theme CSS can override stylesheet :has() rules)."""
    _html_main_script(
        """
<script>
(function () {
  var GRAD = "linear-gradient(180deg, #c4daf6 0%, #a8c8f0 52%, #8eb6e8 100%)";
  function applyDock(el) {
    el.style.setProperty("background", GRAD, "important");
    el.style.setProperty("border", "1px solid #5a8ec8", "important");
    el.style.setProperty("border-radius", "12px", "important");
    el.style.setProperty("box-shadow", "inset 0 1px 0 rgba(255,255,255,0.65), 0 1px 3px rgba(30,58,95,0.12)", "important");
    el.style.setProperty("padding", "0.45rem 0.75rem 0.6rem", "important");
    el.style.setProperty("max-width", "var(--ave-explore-snippet-max)", "important");
    el.style.setProperty("margin-left", "auto", "important");
    el.style.setProperty("margin-right", "auto", "important");
    el.style.setProperty("margin-top", "0.35rem", "important");
    el.style.setProperty("margin-bottom", "0.65rem", "important");
    el.style.setProperty("box-sizing", "border-box", "important");
    el.classList.add("ave-chips-dock-frame");
  }
  function paint() {
    var stats = document.querySelector(".ave-chips-dock-stats");
    if (!stats) return;
    var wraps = document.querySelectorAll(
      '[data-testid="stVerticalBlockBorderWrapper"],[data-testid="stStyledVerticalBlockBorderWrapper"]'
    );
    for (var i = 0; i < wraps.length; i++) {
      if (wraps[i].contains(stats)) {
        applyDock(wraps[i]);
        return;
      }
    }
    var el = stats.parentElement;
    for (var j = 0; j < 22 && el; j++) {
      var tid = el.getAttribute && el.getAttribute("data-testid");
      if (tid === "stVerticalBlockBorderWrapper" || tid === "stStyledVerticalBlockBorderWrapper") {
        applyDock(el);
        return;
      }
      el = el.parentElement;
    }
    var vb = stats.closest('[data-testid="stVerticalBlock"]');
    if (vb && vb.querySelector('[data-testid="stHorizontalBlock"]')) {
      applyDock(vb);
    }
  }
  paint();
  [0, 40, 120, 400, 1000, 2200].forEach(function (d) { setTimeout(paint, d); });
  if (document.body) {
    new MutationObserver(paint).observe(document.body, { childList: true, subtree: true });
  }
})();
</script>
        """,
    )


def _chip_bar_panel_script() -> None:
    """Tag chip rows + blue frame from `.ave-chips-dock-stats` (robust across Streamlit DOM tweaks)."""
    _html_main_script(
        """
<script>
(function () {
  function appDocument() {
    try {
      var pd = window.parent && window.parent.document;
      if (pd && pd.querySelector('[data-testid="stMain"]')) return pd;
    } catch (e) {}
    return document;
  }
  function findBorderWrap(stats) {
    let el = stats;
    for (let i = 0; i < 16 && el; i++) {
      const tid = el.getAttribute && el.getAttribute("data-testid");
      if (tid === "stVerticalBlockBorderWrapper" || tid === "stStyledVerticalBlockBorderWrapper") return el;
      el = el.parentElement;
    }
    return null;
  }
  function isChipRowFirst(hb) {
    if (hb.querySelector("#ave-pin-toolbar")) return false;
    const btns = hb.querySelectorAll('div[data-testid="column"] button');
    if (btns.length < 4) return false;
    const t0 = (btns[0].textContent || "").replace(/\\s+/g, " ").trim();
    const t3 = (btns[3].textContent || "").replace(/\\s+/g, " ").trim();
    return /Archetype/i.test(t0) && /Time/i.test(t3);
  }
  function isChipRowSecond(hb) {
    if (hb.querySelector("#ave-pin-toolbar")) return false;
    const btns = hb.querySelectorAll('div[data-testid="column"] button');
    if (btns.length !== 2) return false;
    const t0 = (btns[0].textContent || "").replace(/\\s+/g, " ").trim();
    const t1 = (btns[1].textContent || "").replace(/\\s+/g, " ").trim();
    return /Publication/i.test(t0) && /Keywords/i.test(t1);
  }
  function panelize() {
    const doc = appDocument();
    const main = doc.querySelector('[data-testid="stMain"]');
    if (!main) return;
    const stats = main.querySelector(".ave-chips-dock-stats");
    if (!stats) return;
    const wrap = findBorderWrap(stats);
    if (wrap) wrap.classList.add("ave-chips-dock-frame");
    main.querySelectorAll(".ave-chips-panel").forEach(function (hb) {
      hb.classList.remove("ave-chips-panel", "ave-chips-panel--first", "ave-chips-panel--last");
    });
    const chipRows = [];
    if (wrap) {
      wrap.querySelectorAll('[data-testid="stHorizontalBlock"]').forEach(function (hb) {
        if (hb.querySelector("#ave-pin-toolbar")) return;
        const n = hb.querySelectorAll('div[data-testid="column"] button').length;
        if (n === 4 || n === 2) chipRows.push(hb);
      });
    } else {
      main.querySelectorAll('[data-testid="stHorizontalBlock"]').forEach(function (hb) {
        if (isChipRowFirst(hb)) chipRows.push(hb);
        else if (isChipRowSecond(hb)) chipRows.push(hb);
      });
    }
    chipRows.forEach(function (hb, idx) {
      hb.classList.add("ave-chips-panel");
      if (idx === 0) hb.classList.add("ave-chips-panel--first");
      if (idx === chipRows.length - 1) hb.classList.add("ave-chips-panel--last");
    });
  }
  panelize();
  requestAnimationFrame(function () { panelize(); });
  setTimeout(panelize, 80);
  setTimeout(panelize, 280);
  setTimeout(panelize, 700);
  const doc = appDocument();
  const obs = new MutationObserver(function () { panelize(); });
  if (doc.body) {
    obs.observe(doc.body, { childList: true, subtree: true });
  }
})();
</script>
        """,
    )


def _snippet_grid_responsive_script() -> None:
    """Keep a single wide snippet column; set --ave-explore-snippet-max for toolbar/dock alignment."""
    _html_main_script(
        """
<script>
(function () {
  function appDocument() {
    try {
      var pd = window.parent && window.parent.document;
      if (pd && pd.querySelector('[data-testid="stMain"]')) return pd;
    } catch (e) {}
    return document;
  }
  function apply() {
    try {
      const doc = appDocument();
      const main = doc.querySelector('[data-testid="stMain"]');
      if (!main) return;
      const docEl = doc.documentElement;
      if (docEl) {
        docEl.style.setProperty("--ave-explore-snippet-max", "min(72rem, 100%)");
      }
      let grid = doc.getElementById("ave-snippet-grid");
      if (!grid) grid = main.querySelector(".ave-snippet-grid-root");
      if (!grid) grid = main.querySelector(".ave-snippet-grid");
      if (grid) grid.classList.remove("ave-snippet-grid--two");
      if (docEl) docEl.classList.remove("ave-snippet-pair-two-col");
    } catch (e) {}
  }
  apply();
  setInterval(apply, 450);
  const doc = appDocument();
  try {
    window.parent.addEventListener("resize", apply, true);
  } catch (e) {}
  doc.addEventListener("click", function () { setTimeout(apply, 150); }, true);
  if (doc.body) {
    const obs = new MutationObserver(apply);
    obs.observe(doc.body, { childList: true, subtree: true, attributes: true, attributeFilter: ["aria-expanded", "class"] });
  }
})();
</script>
        """,
    )


def _main_chips_expand_sidebar_script() -> None:
    """When sidebar is collapsed, a chip click should expand it (header control lives outside sidebar)."""
    _html_main_script(
        """
<script>
(function () {
  function appDocument() {
    try {
      var pd = window.parent && window.parent.document;
      if (pd && pd.querySelector('[data-testid="stMain"]')) return pd;
    } catch (e) {}
    return document;
  }
  function clickExpandControl(doc) {
    const expand =
      doc.querySelector('[data-testid="stExpandSidebarButton"]')
      || doc.querySelector('[data-testid="collapsedControl"] button')
      || doc.querySelector('[data-testid="collapsedControl"] a')
      || doc.querySelector('button[aria-label*="sidebar" i]')
      || doc.querySelector('button[title*="sidebar" i]');
    if (expand && expand.offsetParent !== null) {
      expand.click();
      return true;
    }
    return false;
  }
  function onClick(ev) {
    const doc = appDocument();
    const t = ev.target;
    if (!t || !t.closest) return;
    const btn = t.closest("button") || t.closest('[role="button"]');
    if (!btn) return;
    const main = doc.querySelector('[data-testid="stMain"]');
    if (!main || !main.contains(btn)) return;
    const stats = main.querySelector(".ave-chips-dock-stats");
    let inDock = false;
    if (stats) {
      let wrap = stats.closest('[data-testid="stVerticalBlockBorderWrapper"]')
        || stats.closest('[data-testid="stStyledVerticalBlockBorderWrapper"]');
      if (!wrap) {
        let el = stats.parentElement;
        for (let i = 0; i < 14 && el; i++) {
          const tid = el.getAttribute && el.getAttribute("data-testid");
          if (tid === "stVerticalBlockBorderWrapper" || tid === "stStyledVerticalBlockBorderWrapper") {
            wrap = el;
            break;
          }
          el = el.parentElement;
        }
      }
      if (wrap && wrap.contains(btn)) inDock = true;
    }
    const hb = btn.closest('[data-testid="stHorizontalBlock"]');
    const byClass = hb && (hb.classList.contains("ave-chips-panel") || hb.classList.contains("ave-mobile-filters-strip"));
    const txt = (btn.textContent || "").replace(/\\s+/g, " ").trim();
    const looksLikeFilterChip = /^(🎭|🔮|🗣️|📅|📰|🔎)/.test(txt)
      || /\\b(Archetype|Vision|Stance|Time|Publication|Keywords)\\b/i.test(txt);
    let go = inDock || byClass;
    if (!go && looksLikeFilterChip && hb) go = true;
    if (!go) return;
    clickExpandControl(doc);
  }
  const doc = appDocument();
  doc.addEventListener("click", onClick, true);
})();
</script>
        """,
    )


def _expand_sidebar_button_label_script() -> None:
    """Keep aria-label/title in sync with the themed menu-style expand control."""
    _html_main_script(
        """
<script>
(function () {
  function appDocument() {
    try {
      var pd = window.parent && window.parent.document;
      if (pd && pd.querySelector('[data-testid="stMain"]')) return pd;
    } catch (e) {}
    return document;
  }
  const label = "Open filter sidebar";
  function apply() {
    const doc = appDocument();
    const btn = doc.querySelector('[data-testid="stExpandSidebarButton"]');
    if (!btn) return;
    btn.setAttribute("aria-label", label);
    btn.setAttribute("title", label);
  }
  apply();
  setTimeout(apply, 40);
  setInterval(apply, 600);
  const doc = appDocument();
  if (doc.body) {
    new MutationObserver(apply).observe(doc.body, { childList: true, subtree: true });
  }
})();
</script>
        """,
    )


def _popover_adjust_filters_client_script() -> None:
    """List options → Adjust filters: expand sidebar + dismiss popover in the browser (no full-app rerun)."""
    _html_main_script(
        """
<script>
(function () {
  function appDocument() {
    try {
      var pd = window.parent && window.parent.document;
      if (pd && pd.querySelector('[data-testid="stMain"]')) return pd;
    } catch (e) {}
    return document;
  }
  var doc = appDocument();
  if (doc.documentElement.dataset.avePopoverAdjustFilters === "1") return;
  doc.documentElement.dataset.avePopoverAdjustFilters = "1";
  function clickExpandSidebar() {
    var ex =
      doc.querySelector('[data-testid="stExpandSidebarButton"]')
      || doc.querySelector('[data-testid="collapsedControl"] button')
      || doc.querySelector('[data-testid="collapsedControl"] a');
    if (ex && ex.offsetParent !== null) {
      ex.click();
      return true;
    }
    return false;
  }
  function fireEscape() {
    var ev = new KeyboardEvent("keydown", {
      key: "Escape",
      code: "Escape",
      keyCode: 27,
      which: 27,
      bubbles: true,
      cancelable: true
    });
    try {
      if (doc.activeElement) doc.activeElement.dispatchEvent(ev);
    } catch (e1) {}
    doc.body.dispatchEvent(ev);
    doc.dispatchEvent(ev);
  }
  doc.addEventListener(
    "click",
    function (ev) {
      var t = ev.target;
      if (!t || !t.closest) return;
      var btn = t.closest("button.ave-popover-open-sidebar");
      if (!btn) return;
      ev.preventDefault();
      ev.stopPropagation();
      ev.stopImmediatePropagation();
      clickExpandSidebar();
      setTimeout(fireEscape, 0);
      setTimeout(fireEscape, 40);
      setTimeout(fireEscape, 100);
    },
    true
  );
})();
</script>
        """,
    )


def _collapse_sidebar_script() -> None:
    """Programmatically collapse the Streamlit sidebar (no Python API — click the real collapse control)."""
    _html_main_script(
        """
<script>
(function () {
  function appDocument() {
    try {
      var pd = window.parent && window.parent.document;
      if (pd && pd.querySelector('[data-testid="stMain"]')) return pd;
    } catch (e) {}
    return document;
  }
  function clickCollapse(doc) {
    const sb = doc.querySelector('[data-testid="stSidebar"]');
    if (!sb || sb.getAttribute("aria-expanded") !== "true") return false;
    /* Streamlit >=1.30: wrapper is data-testid="stSidebarCollapseButton" (not stCollapseSidebarButton). */
    const collapseHost =
      doc.querySelector('[data-testid="stSidebarCollapseButton"]')
      || doc.querySelector('[data-testid="stCollapseSidebarButton"]');
    if (collapseHost) {
      const innerBtn = collapseHost.matches("button")
        ? collapseHost
        : collapseHost.querySelector("button");
      if (innerBtn) {
        innerBtn.click();
        return true;
      }
      collapseHost.click();
      return true;
    }
    for (const b of sb.querySelectorAll("button")) {
      const al = ((b.getAttribute("aria-label") || "") + " " + (b.getAttribute("title") || "")).toLowerCase();
      if (al.includes("collapse") && al.includes("sidebar")) {
        b.click();
        return true;
      }
      if (al.includes("close") && al.includes("sidebar")) {
        b.click();
        return true;
      }
    }
    const hdr = doc.querySelector('[data-testid="stHeader"]');
    if (hdr) {
      for (const b of hdr.querySelectorAll("button")) {
        const al = ((b.getAttribute("aria-label") || "") + " " + (b.getAttribute("title") || "")).toLowerCase();
        if (al.includes("collapse") && al.includes("sidebar")) {
          b.click();
          return true;
        }
      }
    }
    /* Last resort: first short-label button in sidebar header row (collapse chevron). */
    const sbr = sb.getBoundingClientRect();
    for (const b of sb.querySelectorAll("button")) {
      const br = b.getBoundingClientRect();
      if (br.top > sbr.top + 80 || br.height <= 0 || br.width <= 0) continue;
      const txt = (b.textContent || "").replace(/\\s+/g, " ").trim();
      if (txt.length > 20) continue;
      b.click();
      return true;
    }
    return false;
  }
  function run() {
    const doc = appDocument();
    clickCollapse(doc);
  }
  const delays = [0, 30, 90, 200, 450, 900];
  for (let i = 0; i < delays.length; i++) {
    setTimeout(run, delays[i]);
  }
})();
</script>
        """,
    )


def _mobile_filters_strip_script() -> None:
    """Tag the main-area Apply filters / Reset row so CSS can show it only on narrow viewports."""
    _html_main_script(
        """
<script>
(function () {
  function appDocument() {
    try {
      var pd = window.parent && window.parent.document;
      if (pd && pd.querySelector('[data-testid="stMain"]')) return pd;
    } catch (e) {}
    return document;
  }
  function mark() {
    const doc = appDocument();
    const main = doc.querySelector('[data-testid="stMain"]');
    if (!main) return;
    for (const b of main.querySelectorAll("button")) {
      const t = (b.textContent || "").replace(/\\s+/g, " ").trim();
      if (!t.includes("Apply filters")) continue;
      const hb = b.closest('[data-testid="stHorizontalBlock"]');
      if (hb) hb.classList.add("ave-mobile-filters-strip");
      return;
    }
  }
  mark();
  setTimeout(mark, 80);
  setInterval(mark, 400);
  const doc = appDocument();
  if (doc.body) {
    new MutationObserver(mark).observe(doc.body, { childList: true, subtree: true });
  }
})();
</script>
        """,
    )


def _sidebar_scroll_to_focus_script(focus_key: str | None) -> None:
    """Scroll sidebar so the focused section is visible; flash-highlight it (also when sidebar already open).

    Uses the order of **top-level** ``st.expander`` blocks in the sidebar (nested reference expander is skipped).
    Avoids text search over arbitrary nodes — that previously matched captions like “Snippet sort order” and
    mis-scrolled away from the real filter controls.
    """
    if not focus_key:
        return
    fk_js = json.dumps(focus_key)
    _html_main_script(
        f"""
<script>
(function () {{
  const FOCUS = {fk_js};
  const ORDER = ["primary", "time", "keywords", "display"];
  function flashEl(node) {{
    if (!node) return;
    /* Re-trigger animation every time (e.g. sidebar already open — user still gets feedback). */
    node.classList.remove("ave-sidebar-section-flash");
    void node.offsetWidth;
    node.classList.add("ave-sidebar-section-flash");
    setTimeout(function () {{
      node.classList.remove("ave-sidebar-section-flash");
    }}, 2400);
  }}
  function openExpander(exp) {{
    if (!exp) return;
    const det = exp.querySelector("details");
    if (det) {{
      if (!det.open) det.open = true;
      return;
    }}
    const ariaFalse = exp.querySelector('[aria-expanded="false"]');
    if (ariaFalse) ariaFalse.click();
    /* Do not click summary when already open — that toggles the section closed. */
  }}
  function appDoc() {{
    try {{
      var pd = window.parent && window.parent.document;
      if (pd && pd.querySelector('[data-testid="stMain"]')) return pd;
    }} catch (e) {{}}
    return document;
  }}
  function topLevelExpanders(content) {{
    const all = Array.from(content.querySelectorAll('[data-testid="stExpander"]'));
    return all.filter(function (exp) {{
      let p = exp.parentElement;
      while (p && p !== content) {{
        if (p.getAttribute && p.getAttribute("data-testid") === "stExpander") return false;
        p = p.parentElement;
      }}
      return true;
    }});
  }}
  function fallbackBySummary(content, ix) {{
    const hints = [
      [/Archetype/i, /Vision/i, /Stance/i],
      [/Time range/i],
      [/Keywords/i],
      [/Publication/i],
    ];
    const need = hints[ix];
    if (!need) return null;
    const top = topLevelExpanders(content);
    for (const exp of top) {{
      const sum = exp.querySelector("summary");
      const lab = ((sum && sum.textContent) || "").replace(/\\s+/g, " ");
      let ok = true;
      for (let j = 0; j < need.length; j++) {{
        if (!need[j].test(lab)) {{ ok = false; break; }}
      }}
      if (ok) return exp;
    }}
    return top[ix] || null;
  }}
  function clickExpandSidebarIfCollapsed(doc) {{
    /* Sidebar collapsed: expand control is in the header — scroll script alone cannot reveal filters. */
    const ex =
      doc.querySelector('[data-testid="stExpandSidebarButton"]')
      || doc.querySelector('[data-testid="collapsedControl"] button')
      || doc.querySelector('[data-testid="collapsedControl"] a');
    if (ex && ex.offsetParent !== null) {{
      ex.click();
      return true;
    }}
    return false;
  }}
  function run() {{
    const doc = appDoc();
    clickExpandSidebarIfCollapsed(doc);
    const content =
      doc.querySelector('[data-testid="stSidebarContent"]')
      || doc.querySelector('[data-testid="stSidebar"]');
    if (!content) return;
    const ix = ORDER.indexOf(FOCUS);
    if (ix < 0) return;
    const top = topLevelExpanders(content);
    let exp = top.length > ix ? top[ix] : null;
    if (!exp) exp = fallbackBySummary(content, ix);
    if (!exp) return;
    openExpander(exp);
    exp.scrollIntoView({{ block: "center", behavior: "smooth" }});
    flashEl(exp);
  }}
  setTimeout(run, 50);
  setTimeout(run, 200);
  setTimeout(run, 500);
  setTimeout(run, 1000);
  setTimeout(run, 1800);
}})();
</script>
        """,
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


def _reprint_occurrences_dataframe(pn: str, full_df: pd.DataFrame) -> pd.DataFrame:
    """Rows in ``full_df`` with the same normalized paragraph key (current filter)."""
    sub = full_df[full_df["_ave_para_norm"] == pn].copy()
    cols = [
        c
        for c in [
            "date",
            "publication",
            "title",
            "__article_key",
            "paragraph_id",
            "year",
            "month",
            "future_type_prediction",
            "archetype",
            "majority_stance",
            "paragraph",
        ]
        if c in sub.columns
    ]
    show = sub[cols].copy()
    if "date" in show.columns:
        show["date"] = pd.to_datetime(show["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return show


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


def _dashboard_csv_path(project_root: str) -> str:
    """Use ``DASHBOARD_CSV`` from the environment or Streamlit secrets; then default path."""
    v = (os.environ.get("DASHBOARD_CSV") or "").strip()
    if v:
        return v
    try:
        v = (st.secrets.get("DASHBOARD_CSV") or "").strip()
    except Exception:
        v = ""
    if v:
        return v
    return default_csv_path(project_root)


def main() -> None:
    # Browser tab icon: small PNG favicon (see assets/favicon.png); fallback to full icon or SVG.
    _page_icon = (
        _FAVICON_PATH
        if os.path.isfile(_FAVICON_PATH)
        else (
            _BRAND_ICON_PATH
            if os.path.isfile(_BRAND_ICON_PATH)
            else (_BRAND_SVG_PATH if os.path.isfile(_BRAND_SVG_PATH) else "🔭")
        )
    )
    st.set_page_config(
        page_title="AI Vision Explorer",
        page_icon=_page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if not _check_password():
        return

    if AVE_SHOW_GUIDE_KEY not in st.session_state:
        st.session_state[AVE_SHOW_GUIDE_KEY] = False
    # Legacy ``?guide=1`` URLs: move to session state (soft rerun; keeps sign-in).
    try:
        if st.query_params.get("guide") == "1":
            st.session_state[AVE_SHOW_GUIDE_KEY] = True
            if "guide" in st.query_params:
                del st.query_params["guide"]
            st.rerun()
    except Exception:
        pass
    # Explanations “Explore this →” uses ``<a href="?ave_explore=…">`` (see ``guide_content``) — apply filters, leave guide, strip param.
    try:
        if "ave_explore" in st.query_params:
            _ae = st.query_params.get("ave_explore")
            if isinstance(_ae, list):
                _ae = _ae[0] if _ae else ""
            _ae = str(_ae).strip()
            if _ae:
                _apply_ave_explore_query_param(_ae)
            del st.query_params["ave_explore"]
            st.rerun()
    except Exception:
        pass
    if st.session_state[AVE_SHOW_GUIDE_KEY]:
        _inject_branding_css()
        _gb_cols = st.columns([1])
        with _gb_cols[0]:
            if st.button("← Back to dashboard", key="ave_back_guide", use_container_width=False):
                st.session_state[AVE_SHOW_GUIDE_KEY] = False
                st.rerun()
        st.markdown(
            '<div id="ave-guide-back-flow-spacer" class="ave-guide-back-flow-spacer" aria-hidden="true"></div>',
            unsafe_allow_html=True,
        )
        _guide_back_row_script()
        render_guide_page()
        st.stop()

    csv_path = _dashboard_csv_path(_PROJECT_ROOT)

    try:
        df = _load_data(csv_path)
    except FileNotFoundError as e:
        st.error(str(e))
        st.info(
            "Set **DASHBOARD_CSV** in this app’s Streamlit **Secrets** (or your environment) to an "
            "**https://** URL for `merged_analysis.csv`, or ship the file as `dashboard/data/merged_analysis.csv` in the repo."
        )
        st.stop()
    except Exception as e:
        st.exception(e)
        st.stop()

    para_dup_stats = paragraph_duplicate_metrics(df, csv_path=csv_path, dashboard_dir=_DASHBOARD_DIR)

    # Distinct filter values (reuse for multiselect options + post-widget sanitization).
    _filter_opts_arch = sorted(df["archetype"].dropna().unique().tolist())
    _filter_opts_vision = sorted(df["future_type_prediction"].dropna().unique().tolist())
    _filter_opts_stance = sorted(df["majority_stance"].dropna().unique().tolist())

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
        st.session_state["dash_vision"] = []
    else:
        st.session_state["dash_vision"] = _normalize_dash_visions(st.session_state.get("dash_vision"))
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
    if "dash_dedupe_paragraphs" not in st.session_state:
        st.session_state["dash_dedupe_paragraphs"] = True
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
        st.session_state["dash_vision"] = []
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
      <p class="sb-guide-hint sb-guide-hint-full">Concepts &amp; categories: open <strong>Explanations</strong> (top right, next to the title).</p>
      <p class="sb-guide-hint sb-guide-hint-short">Use <strong>Explanations</strong> (top right) for concepts &amp; categories.</p>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<p class="ave-sidebar-filters-heading">Filters</p>',
        unsafe_allow_html=True,
    )

    _ps_raw = st.session_state.get("dash_page_size", 50)
    try:
        page_size = int(_ps_raw)
    except (TypeError, ValueError):
        page_size = 50
    page_size = max(1, min(200, page_size))

    _inject_branding_css()
    _sidebar_dock_script()
    _v2a_multiselect_mute_script()
    _chip_bar_panel_script()
    _dock_blue_paint_script()
    _expl_title_row_script()
    _loading_overlay_script()
    # Disabled: capture-phase click handlers / MutationObservers interfered with Streamlit widget
    # hit-testing in production (buttons dead, snippets not updating). Re-enable only after testing.
    # _snippet_grid_responsive_script()
    # _main_chips_expand_sidebar_script()
    _expand_sidebar_button_label_script()
    # _popover_adjust_filters_client_script()
    _mobile_filters_strip_script()

    focus = st.session_state.get(_FILTER_FOCUS_KEY)

    with st.sidebar:
        with st.expander("🎭 Archetype · Vision · Stance", expanded=(focus in (None, "primary"))):
            if focus == "primary":
                st.markdown(
                    '<div class="ave-flash-target" style="font-weight:600;color:#1a3344;margin-bottom:0.5rem;">'
                    "Archetype / stance / vision — adjust below</div>",
                    unsafe_allow_html=True,
                )
            arch_opts = _filter_opts_arch
            vision_opts = _filter_opts_vision
            stance_opts = _filter_opts_stance
            _repair_format_multiselect_state("dash_archetypes", arch_opts)
            _repair_format_multiselect_state("dash_vision", vision_opts)
            _repair_format_multiselect_state("dash_stances", stance_opts)

            _arch_pick = set(st.session_state.get("dash_archetypes", []))
            _stance_pick = set(st.session_state.get("dash_stances", []))
            _vision_prev = _normalize_dash_visions(st.session_state.get("dash_vision"))
            if _stance_pick:
                _compat_ar = compatible_archetypes_for_stances_with_visions(_stance_pick, _vision_prev)
            elif _vision_prev:
                _compat_ar = compatible_archetypes_for_visions_union(_vision_prev)
            else:
                _compat_ar = None
            _compat_vis = compatible_visions_for_filters(_arch_pick, _stance_pick)

            def _fmt_arch_label(opt: str) -> str:
                if _compat_ar is None:
                    return opt
                return f"○ {opt}" if opt not in _compat_ar else f"● {opt}"

            def _fmt_vision_label(opt: str) -> str:
                if _compat_vis is None:
                    return opt
                return f"○ {opt}" if opt not in _compat_vis else f"● {opt}"

            archetypes = st.multiselect(
                "Archetype",
                options=arch_opts,
                key="dash_archetypes",
                format_func=_fmt_arch_label,
                help="Empty = all archetypes. ● = can occur with your current stance selection (given vision scope); "
                "○ = off that mapping (still selectable).",
            )
            st.multiselect(
                "Vision type",
                vision_opts,
                key="dash_vision",
                format_func=_fmt_vision_label,
                help="Empty = all vision types. ● = can occur with your current archetype and stance picks "
                "under the table; ○ = off that joint mapping (still selectable).",
            )
            visions = _normalize_dash_visions(st.session_state.get("dash_vision"))
            if visions:
                _cap_vis = [
                    f"{v}: " + ", ".join(VISION_TO_ARCHETYPES[v])
                    for v in sorted(visions)
                    if v in VISION_TO_ARCHETYPES
                ]
                if _cap_vis:
                    st.caption("Archetypes possible per vision (depend on stance) — " + " · ".join(_cap_vis))

            if _arch_pick:
                _compat_st = compatible_stances_for_archetypes_with_visions(_arch_pick, visions)
            elif visions:
                _compat_st = compatible_stances_for_visions_union(visions)
            else:
                _compat_st = None

            def _fmt_stance_label(opt: str) -> str:
                if _compat_st is None:
                    return opt
                return f"○ {opt}" if opt not in _compat_st else f"● {opt}"

            stances = st.multiselect(
                "Rhetorical stance",
                options=stance_opts,
                key="dash_stances",
                format_func=_fmt_stance_label,
                help="Empty = all stances. ● = can produce your current archetype selection under the chosen vision(s); "
                "○ = no table row maps those archetypes to this stance (still selectable).",
            )
            if _compat_st is not None or _compat_ar is not None or _compat_vis is not None:
                st.caption(
                    "**●** = consistent with the other picks under vision×stance→archetype · "
                    "**○** = off that mapping (muted); still clickable for strict AND filters."
                )
            with st.expander("Vision → archetypes (reference table)", expanded=False):
                st.caption("Paper mapping · vision × stance → archetype (full grid under **Explanations**).")
                lines = [
                    f"- {v}: " + ", ".join(arches) for v, arches in VISION_TO_ARCHETYPES.items()
                ]
                st.markdown("\n".join(lines))

        with st.expander("📅 Time range", expanded=(focus == "time")):
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
                "Snippet **sort order** and **per page** are under **List options**; **Clear filters** and **Adjust filters** sit at the bottom of that menu. The header menu button (vertical bars) also opens the sidebar. The bar stays visible when you scroll."
            )

    # Widget return values can still carry formatted/stale labels; sanitize before filtering.
    archetypes = _sanitize_multiselect_against_options(list(archetypes), _filter_opts_arch)
    stances = _sanitize_multiselect_against_options(list(stances), _filter_opts_stance)
    visions = _sanitize_multiselect_against_options(list(visions), _filter_opts_vision)

    sort_by = st.session_state.get("dash_sort", "Date (newest first)")

    out_base, df_time_only = _apply_filters_unsorted(
        df, time_mode, list(archetypes), list(stances), visions, pub, kw, match_all
    )
    n_total = len(df)
    n_match = len(out_base)
    n_art_match = int(article_key_series(out_base).nunique()) if n_match else 0
    pct_corpus = (100.0 * n_match / n_total) if n_total else 0.0
    snip_per_article_line = (n_match / n_art_match) if n_art_match else 0.0

    out = _apply_sort(out_base, sort_by, csv_path=csv_path)
    sum_stats = article_summary_stats(out)

    dedupe_paragraphs = bool(st.session_state.get("dash_dedupe_paragraphs", True))
    _dedupe_ctx = (
        st.spinner("Preparing snippet list…")
        if (dedupe_paragraphs and len(out) > 3500)
        else contextlib.nullcontext()
    )
    with _dedupe_ctx:
        if dedupe_paragraphs:
            out_list, out_full = dedupe_sorted_paragraph_rows(out)
        else:
            out_list = out
            out_full = None

    # Hidden trigger button — JS in the Explanations link clicks this to open guide without page navigation.
    if st.button("", key="ave_open_guide"):
        st.session_state[AVE_SHOW_GUIDE_KEY] = True
        st.rerun()
    # Title + Explanations: one HTML flex row (onclick JS clicks hidden button above; avoids session-resetting navigation).
    st.markdown(_explorer_title_bar_row_html(_icon_uri), unsafe_allow_html=True)

    st.markdown('<div id="ave-top" style="height:1px;margin:0;padding:0;"></div>', unsafe_allow_html=True)
    _jm = "ALL" if match_all else "ANY"
    _kw_chip = (kw or "").strip() or "—"
    _time_chip = _time_chip_display(date_label)
    # Filter chips only at top — corpus / dedupe counts live at the bottom (see Corpus & counts expander).
    # Outer keyed container constrains width on wide viewports (same as snippet grid); border lives inside.
    with st.container(key="ave_chips_dock_width"):
        with st.container(border=True):
            # Two rows (4 + 2 cols) so chips never flex-wrap to a staggered third row.
            j1, j2, j3, j4 = st.columns(4)
            with j1:
                st.button(
                    _chip_panel_line("Archetype", _fmt_multi_chip(archetypes), emoji="🎭", max_len=22),
                    on_click=_nav_primary_cb,
                    key="main_nav_arch",
                    use_container_width=True,
                )
            with j2:
                st.button(
                    _chip_panel_line("Vision", _fmt_multi_chip(visions, max_len=18), emoji="🔮", max_len=18),
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
                    _chip_panel_line("Time", _time_chip, emoji="📅", max_len=24),
                    on_click=_nav_time_cb,
                    key="main_nav_time",
                    use_container_width=True,
                )
            j5, j6 = st.columns([1, 1])
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
            j7, j8 = st.columns(2)
            with j7:
                st.button(
                    "⚙️ Open filters",
                    on_click=_nav_primary_cb,
                    key="main_dock_open_filters",
                    use_container_width=True,
                    help="Scroll to sidebar filters (filters apply as you change them).",
                )
            with j8:
                st.button(
                    "Reset all",
                    on_click=_reset_all_filters,
                    key="main_dock_reset_all",
                    use_container_width=True,
                )
    n_articles_match = sum_stats["unique_articles"] if len(out) else 0
    _sb_sub = f"from {n_articles_match:,} articles"
    st.sidebar.markdown(
        f'<div class="ave-sidebar-match">'
        f'<div class="ave-sm-label">Matching snippets</div>'
        f'<div class="ave-sm-num">{len(out):,}</div>'
        f'<div class="ave-sm-sub">{_sb_sub}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    sb_d1, sb_d2, sb_d3 = st.sidebar.columns(3)
    with sb_d1:
        st.button(
            "⚙️ Open filters",
            on_click=_nav_primary_cb,
            key="sb_open_filters",
            use_container_width=True,
            help="Scroll to filter controls.",
        )
    with sb_d2:
        st.button(
            "Apply filters",
            type="primary",
            on_click=_sidebar_apply_filters_cb,
            key="sb_filter",
            use_container_width=True,
            help="Collapse the sidebar after reviewing filters.",
        )
    with sb_d3:
        st.button("Reset all", on_click=_reset_all_filters, key="sb_reset_all", use_container_width=True)

    if st.session_state.pop(_SCROLL_SIDEBAR_KEY, False):
        _sidebar_scroll_to_focus_script(st.session_state.get(_FILTER_FOCUS_KEY))
    if st.session_state.pop(_COLLAPSE_SIDEBAR_KEY, False):
        _collapse_sidebar_script()
    _filter_flash_script(st.session_state.pop(DASH_FILTER_FLASH_KEY, None))

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
        tuple(sorted(visions)),
        kw,
        match_all,
        pub,
        sort_by,
        page_size,
        dedupe_paragraphs,
    )
    if st.session_state.get(_FILTER_SIG_KEY) != sig:
        st.session_state[_FILTER_SIG_KEY] = sig
        st.session_state[_PAGE_KEY] = 1
        st.session_state.pop("ave_reprint_open_pn", None)

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

    total = len(out_list)
    n_pages = max(1, (total + page_size - 1) // page_size)
    if _PAGE_KEY not in st.session_state:
        st.session_state[_PAGE_KEY] = 1
    p = int(st.session_state[_PAGE_KEY])
    p = max(1, min(p, n_pages))
    st.session_state[_PAGE_KEY] = p

    if n_snippets == 0:
        st.caption(
            f"No snippets match · {n_total:,} in the full dataset · widen filters in the **sidebar**, "
            "or use the **menu** button in the app header if the sidebar is hidden · **Reset all** in the sidebar."
        )
        st.markdown(
            """
<div class="ave-empty-filter" role="status">
  <div class="ave-empty-filter-icon" aria-hidden="true">🔍</div>
  <div class="ave-empty-filter-body">
    <p class="ave-empty-filter-title">No snippets match these filters</p>
    <p class="ave-empty-filter-hint">Widen the date range, clear keywords, or relax archetype, stance, vision, or publication in <strong>Filters</strong> (sidebar). If the sidebar is hidden, open it with the <strong>menu</strong> button in the app header, or use <strong>Reset all</strong> in the sidebar.</p>
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
    _p_dba = int(para_dup_stats.get("para_distinct_body_article_key") or 0)
    _para_exact_line = (
        "- **Paragraphs — exact duplicate text** (same normalized body on multiple rows; full dataset): "
        f"**{para_dup_stats['para_exact_extra_rows']:,}** redundant rows · "
        f"**{para_dup_stats['para_exact_multi_groups']:,}** distinct paragraph texts appearing 2+ times.\n"
        f"- **Paragraphs × article identity** (distinct normalized body × article ID; full dataset): "
        f"**{_p_dba:,}** — same text in another article/reprint counts as a separate unit "
        f"(between **{para_dup_stats['para_unique_norms']:,}** unique bodies and "
        f"**{para_dup_stats['para_rows_total']:,}** snippet rows).\n"
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

    with st.container(key="ave_snippet_width_stack"):
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
    if dedupe_paragraphs and n_snippets > 0 and total < n_snippets:
        _nav_stats_html = (
            "<div class='ave-nav-toolbar-stats' title='List = distinct paragraph bodies (normalized). See Corpus & counts at bottom.'>"
            f"<strong>{total:,}</strong> in list · <strong>{n_snippets:,}</strong> rows in filter</div>"
        )
    elif dedupe_paragraphs and n_snippets > 0:
        _nav_stats_html = (
            "<div class='ave-nav-toolbar-stats'>"
            f"<strong>{total:,}</strong> in list · <strong>{n_snippets:,}</strong> rows</div>"
        )
    else:
        _nav_stats_html = (
            f"<div class='ave-nav-toolbar-stats'><strong>{n_snippets:,}</strong> snippets · "
            f"<strong>{total:,}</strong> in list</div>"
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
    # Counts · pager · compact "List options" popover (per-page + sort stay available without crowding small screens)
    nv1, nv2, nv3 = st.columns([1.45, 2.5, 1.05])
    with nv1:
        st.markdown(_nav_row1_html, unsafe_allow_html=True)
    with nv2:
        pc1, pc2, pc3, pc4 = st.columns([0.95, 1.05, 0.95, 0.55])
        with pc1:
            if st.button("Previous", disabled=p <= 1, key="nav_prev", use_container_width=False):
                st.session_state[_PAGE_KEY] = p - 1
                st.session_state.pop("ave_reprint_open_pn", None)
                st.rerun()
        with pc2:
            st.markdown(_nav_page_html, unsafe_allow_html=True)
        with pc3:
            if st.button("Next", disabled=p >= n_pages, key="nav_next", use_container_width=False):
                st.session_state[_PAGE_KEY] = p + 1
                st.session_state.pop("ave_reprint_open_pn", None)
                st.rerun()
        with pc4:
            st.markdown(_nav_top_html, unsafe_allow_html=True)
    with nv3:
        # Minimal ``st.popover`` API — extra kwargs (width, icon) can break older Streamlit builds.
        _popover = getattr(st, "popover", None)
        if _popover is not None:
            _list_opts_ctx = _popover("List options", key="dash_snippet_nav_popover")
        else:
            _list_opts_ctx = st.expander("List options", expanded=False)
        with _list_opts_ctx:
            st.selectbox(
                "Per page",
                [25, 50, 100, 200],
                key="dash_page_size",
                help="How many snippet cards to show on each page (does not change your filter).",
            )
            st.selectbox(
                "Order by",
                SORT_SNIPPET_OPTIONS,
                key="dash_sort",
                help=(
                    "How snippets are ordered within the current filter. "
                    "Distinctive: log-odds for the current filter vs the full corpus. "
                    "Representative: closest to typical wording within the current filter. "
                    "See **Explanations** for more."
                ),
            )
            st.checkbox(
                "Merge identical paragraph text (reprints)",
                key="dash_dedupe_paragraphs",
                help=(
                    "Each distinct paragraph body is shown once. Multiplicity uses a stacked card and ×N; "
                    "click **×N** to open a table of every matching row in this filter."
                ),
            )
            st.divider()
            _pf1, _pf2 = st.columns(2)
            with _pf1:
                if st.button(
                    "Clear filters",
                    key="dash_popover_clear_filters",
                    use_container_width=True,
                    help="Reset all filters to defaults (same as Reset all in the sidebar).",
                ):
                    _reset_all_filters()
                    st.rerun()
            with _pf2:
                if st.button(
                    "Adjust filters",
                    key="dash_popover_adjust_filters",
                    use_container_width=True,
                    help="Scroll to filters in the sidebar (same as the chip dock).",
                ):
                    _nav_primary_cb()
                    st.rerun()

    st.markdown(
        '<div id="ave-nav-flow-spacer" class="ave-nav-flow-spacer" aria-hidden="true"></div>',
        unsafe_allow_html=True,
    )
    _fixed_nav_scroll_script()
    _smooth_scroll_top_script()

    # ----- Snippet list + table -----
    start = (p - 1) * page_size
    chunk = out_list.iloc[start : start + page_size]

    st.subheader("Snippets")

    rows_list = list(chunk.iterrows())
    _arch_l, _stance_l, _vis_l = list(archetypes), list(stances), list(visions)
    if dedupe_paragraphs:
        _lookup = {i: str(row["_ave_para_norm"]) for i, (_, row) in enumerate(rows_list)}
        _nrows = len(rows_list)
        # Two Streamlit columns of snippet rows; ×N is a real button beside the index (opens st.dialog).
        for i in range(0, _nrows, 2):
            try:
                c1, c2 = st.columns(2, gap="large")
            except TypeError:
                c1, c2 = st.columns(2)
            with c1:
                _, row = rows_list[i]
                _render_snippet_row_streamlit(
                    row,
                    global_index=start + i + 1,
                    reprint_button_key=f"ave_reprint_badge_{p}_{start}_{i}",
                    archetypes=_arch_l,
                    stances=_stance_l,
                    visions=_vis_l,
                    para_norm=_lookup[i],
                )
            if i + 1 < _nrows:
                with c2:
                    _, row = rows_list[i + 1]
                    _render_snippet_row_streamlit(
                        row,
                        global_index=start + i + 2,
                        reprint_button_key=f"ave_reprint_badge_{p}_{start}_{i + 1}",
                        archetypes=_arch_l,
                        stances=_stance_l,
                        visions=_vis_l,
                        para_norm=_lookup[i + 1],
                    )
            if i + 2 < _nrows:
                st.markdown(
                    '<div class="ave-snippet-dedupe-pair-spacer" aria-hidden="true"></div>',
                    unsafe_allow_html=True,
                )
    else:
        _cells = [
            _snippet_item_html(
                row,
                global_index=start + i + 1,
                archetypes=_arch_l,
                stances=_stance_l,
                visions=_vis_l,
                reprint_count=max(1, int(row.get("reprint_count", 1))),
            )
            for i, (_, row) in enumerate(rows_list)
        ]
        _render_snippet_block(
            '<div id="ave-snippet-grid" class="ave-snippet-grid ave-snippet-grid-root">'
            + "".join(_cells)
            + "</div>"
        )

    _pn_open = st.session_state.get("ave_reprint_open_pn")
    if _pn_open is not None and out_full is not None and dedupe_paragraphs:

        @st.dialog("Reprints — all appearances in this filter")
        def _ave_reprint_dialog() -> None:
            show = _reprint_occurrences_dataframe(str(_pn_open), out_full)
            st.dataframe(show, use_container_width=True, hide_index=True)
            st.caption(f"{len(show)} row(s) in this filter share this paragraph text (normalized).")
            if st.button("Close", key="ave_reprint_dlg_close"):
                st.session_state.pop("ave_reprint_open_pn", None)
                st.rerun()

        _ave_reprint_dialog()

    _cap_tail = f" · {n_snippets:,} snippet rows in filter" if (dedupe_paragraphs and total < n_snippets) else ""
    st.caption(
        f"Showing items {start + 1}–{min(start + page_size, total)} of {total:,}{_cap_tail}"
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
    if dedupe_paragraphs and "reprint_count" in chunk.columns:
        chunk_show.insert(0, "reprint_count", chunk["reprint_count"])
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

    with st.expander("Corpus & how counts are built", expanded=False):
        st.markdown(
            "**How the corpus was produced.** Rows come from your merged analysis table: each row is one "
            "paragraph (snippet) that was classified for future vision, stance, and archetype. Dates and "
            "publication are attached for filtering; the same paragraph text can appear more than once when "
            "articles are reprinted or duplicated in the source files — we **do not** drop those rows by default "
            "(they carry separate dates and article IDs). **List options → Merge identical paragraph text** "
            "collapses identical normalized text in the card list so you read each wording once; the table "
            "below still reflects the current page of that list."
        )
        if dedupe_paragraphs and n_match > 0:
            _nu = len(out_list)
            st.markdown(
                f"- **Distinct paragraph bodies in this filter** (normalized text): **{_nu:,}**  \n"
                f"- **Snippet rows in this filter** (all matching rows): **{n_match:,}**  \n"
                f"- **Full loaded corpus** (snippet rows): **{n_total:,}**  \n"
                f"- **Articles** represented in this filter: **{n_art_match:,}**  \n"
                f"- **Share of corpus** in this filter: **{pct_corpus:.1f}%**  \n"
                f"- **Snippets per article** (mean in this filter): **{snip_per_article_line:.2f}**"
            )
        else:
            st.markdown(
                f"- **Snippet rows in this filter:** **{n_match:,}** / **{n_total:,}** in full corpus  \n"
                f"- **Articles** in this filter: **{n_art_match:,}** · **{pct_corpus:.1f}%** of corpus  \n"
                f"- **Snippets per article:** **{snip_per_article_line:.2f}**"
            )
        _pdba = int(para_dup_stats.get("para_distinct_body_article_key") or 0)
        st.markdown(
            "**Full dataset (not filter-specific).** "
            f"**{para_dup_stats['para_rows_total']:,}** snippet rows · "
            f"**{para_dup_stats['para_unique_norms']:,}** distinct normalized paragraph bodies · "
            f"**{_pdba:,}** distinct (body × article ID) pairs "
            "(same wording in another article counts separately). "
            "See **Summary statistics** above for reprint heuristics and optional fuzzy duplicate clusters."
        )

    st.caption(f"Data file: `{csv_path}`")


if __name__ == "__main__":
    main()

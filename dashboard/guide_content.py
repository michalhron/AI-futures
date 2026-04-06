"""
Explanations view: concepts in the same visual language as snippet cards.
“Explore this →” is a real ``<a href="?ave_explore=…">`` link so it always reads as a plain text
link (Streamlit ``st.button`` was inheriting chip / full-width styles for some keys). ``main()`` applies
the filter from the query param and strips it (same mechanism as legacy deep links).
"""
from __future__ import annotations

import html
from urllib.parse import quote

import streamlit as st

from data_utils import V2A_STANCED, VISION_TO_ARCHETYPES

VISION_NARRATIVES: dict[str, str] = {
    "Open Horizons, Unstable Ground": (
        "AI appears as a <strong>regime-shifting frontier</strong>: enormous possibility, ambiguous stakes, and an unsettled "
        "sense of what counts as plausible or urgent—often before institutions know how to absorb the change."
    ),
    "Empowered but Exposed": (
        "The future is <strong>high leverage but fragile</strong>: new powers (speed, reach, personalization) sit alongside "
        "visible exposure—to competition, surveillance, error, or reputational risk."
    ),
    "Seamless but Concentrated": (
        "AI is imagined as <strong>infrastructure that fades into the background</strong>—smooth operations—while power and "
        "value <strong>concentrate</strong> in platforms or gatekeepers. Convenience meets dependency."
    ),
    "Transformed or Left Behind": (
        "The paragraph frames <strong>sharp stratification</strong>: winners and losers, reskilling, and “adapt or perish” "
        "narratives. Who remains legitimate in the new order?"
    ),
    "Guided but Fragile": (
        "<strong>Guardrails</strong> take center stage: oversight, ethics, risk, careful adoption. The future is navigable only "
        "if institutions can steer a powerful but brittle technology without losing trust."
    ),
}

STANCE_NARRATIVES: dict[str, str] = {
    "Opening": (
        "Expands the interpretive space—possibilities, scenarios, reframings. Often speculative or wide-angle."
    ),
    "Mobilizing": (
        "Points toward <strong>action</strong>: imperatives, pilots, “what leaders should do.” A script for doing something soon."
    ),
    "Normalizing": (
        "<strong>Routinizes</strong> the new: standards, habits, “how we run X now.” Embedding more than hype."
    ),
    "Controlling": (
        "<strong>Bounds</strong> adoption: risk, compliance, governance, limits on autonomy. Who may act, under what rules."
    ),
}

_ARCH_BORDER = {
    "Pioneer": "#c45c26",
    "Builder": "#2d6a4f",
    "Guardian": "#1a3d5c",
}


def _card(
    title: str,
    body: str,
    *,
    border: str | None = None,
    filter_mark: bool = True,
    with_explore: bool = False,
) -> str:
    b = f' style="border-left:4px solid {border}"' if border else ""
    fm = " gx-filter-mark" if filter_mark else ""
    hx = " gx-snippet-card--has-explore-link" if with_explore else ""
    return (
        f'<div class="gx-snippet-card{fm}{hx}"{b}>'
        f'<div class="gx-guide-card-heading">{html.escape(title)}</div>'
        f'<div class="gx-snippet-body">{body}</div>'
        "</div>"
    )


def _explore_link(kind: str, value: str) -> None:
    """Open dashboard with this filter via ``?ave_explore=`` (handled in ``app.main``)."""

    q = quote(f"{kind}:{value}", safe="")
    st.markdown(
        f'<p class="gx-explore-line"><a class="gx-explore-href" href="?ave_explore={q}">Explore this →</a></p>',
        unsafe_allow_html=True,
    )


def render_guide_page() -> None:
    st.markdown(
        """
<style>
  .gx-hero {
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, "Noto Serif", serif;
    font-size: 1.2rem;
    line-height: 1.75;
    color: #3f3c38;
    margin: 0 0 1.75rem 0;
    max-width: 52rem;
  }
  .gx-section {
    margin: 2.25rem 0 1rem 0;
  }
  .gx-kicker {
    font-family: system-ui, -apple-system, sans-serif;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 0 0 0.35rem 0;
  }
  .gx-h2 {
    font-family: system-ui, -apple-system, sans-serif;
    font-size: clamp(1.35rem, 2.5vw, 1.75rem);
    font-weight: 800;
    color: #1e293b;
    letter-spacing: -0.03em;
    line-height: 1.2;
    margin: 0 0 0.5rem 0;
  }
  .gx-sub {
    font-size: 0.95rem;
    color: #64748b;
    margin: 0 0 1rem 0;
    line-height: 1.5;
    max-width: 48rem;
  }
  .gx-snippet-card {
    background: #ffffff;
    border: 1px solid #ece8e4;
    border-radius: 14px;
    padding: 0.95rem 1.15rem 1.05rem 1.15rem;
    margin: 0 0 0.85rem 0;
    box-shadow: 0 1px 8px rgba(45, 42, 38, 0.05);
  }
  .gx-snippet-card--has-explore-link {
    padding-bottom: 0.85rem !important;
  }
  /* Concept name reads as a real heading (larger than body), distinct from snippet cards in the explorer */
  .gx-guide-card-heading {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: clamp(1.2rem, 2.35vw, 1.55rem);
    font-weight: 800;
    color: #0f172a;
    letter-spacing: -0.025em;
    line-height: 1.25;
    margin: 0 0 0.6rem 0;
  }
  /* Real <a> links — not st.button — so every row matches (no chip / full-width chrome). */
  .gx-explore-line {
    margin: -0.35rem 0 0.55rem 0;
    padding: 0 0 0 1.1rem;
    line-height: 1.35;
  }
  .gx-explore-href {
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    font-size: 0.78rem;
    font-weight: 500;
    color: #2563eb;
    text-decoration: none;
    background: transparent;
    border: none;
    padding: 0;
    display: inline;
  }
  .gx-explore-href:hover {
    color: #1d4ed8;
    text-decoration: underline;
  }
  .gx-explore-href:focus-visible {
    outline: 2px solid rgba(37, 99, 235, 0.4);
    outline-offset: 2px;
    border-radius: 2px;
  }
  .gx-snippet-body {
    font-family: "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, "Noto Serif", serif;
    font-size: 1.05rem;
    line-height: 1.72;
    color: #3f3c38;
    margin: 0;
  }
  .gx-snippet-body strong { color: #1e293b; font-weight: 600; }
  .gx-snippet-body em { color: #475569; }
  .gx-meta {
    font-size: 0.82rem;
    color: #94a3b8;
    margin: 0.35rem 0 0 0;
    font-family: system-ui, sans-serif;
  }
  .gx-table-wrap {
    margin-top: 0.75rem;
    overflow-x: auto;
    border: 1px solid #e8e4dc;
    border-radius: 12px;
    background: #faf9f7;
  }
  .gx-table-wrap table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.82rem;
  }
  .gx-table-wrap th {
    text-align: left;
    padding: 0.5rem 0.65rem;
    background: #eef4f8;
    color: #1a3d5c;
    font-weight: 700;
    border-bottom: 1px solid #dce7f0;
  }
  .gx-table-wrap td {
    padding: 0.45rem 0.65rem;
    border-bottom: 1px solid #ece8e4;
    color: #334155;
    vertical-align: top;
  }
  .gx-table-wrap tr:last-child td { border-bottom: none; }
</style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="gx-hero">Executive-facing AI discourse (HBR &amp; MIT SMR), broken into paragraphs. Each row '
        "combines <strong>what future is imagined</strong>, <strong>how the paragraph performs it</strong>, and "
        "<strong>which rhetorical role</strong> that pairing plays in an organizing vision.</p>",
        unsafe_allow_html=True,
    )

    # --- Snippet counts & reprints ---
    _counts_body = (
        "<p><strong>Snippet row</strong> — One record in the table: a single coded paragraph excerpt plus metadata "
        "(article identity, date, publication, vision type, stance, archetype, and the paragraph text).</p>"
        "<p><strong>Normalized paragraph text</strong> — Before two bodies are compared, the app trims leading and "
        "trailing space, collapses internal whitespace to single spaces, and strips diacritics (accent marks) so minor "
        "typographic differences do not split what is substantively the same wording.</p>"
        "<p><strong>Unique paragraph</strong> — When <em>Merge identical paragraph text (reprints)</em> is on in "
        "<strong>List options</strong>, the explorer counts <strong>distinct</strong> normalized paragraph bodies "
        "in your <strong>current filter</strong>. See <strong>Corpus &amp; how counts are built</strong> at the bottom "
        "of the explorer (and the snippet toolbar when it applies).</p>"
        "<p><strong>Snippet rows in filter</strong> — The number of table rows that pass your filters, counting every "
        "occurrence separately. If the same normalized text appears ten times, that adds ten to this count but only "
        "one to unique paragraphs.</p>"
        "<p><strong>Reprints and multiplicity</strong> — The dataset does not ship a dedicated “reprint” column. When "
        "the same normalized paragraph appears on more than one row, we call that <strong>multiplicity</strong>: it "
        "can reflect magazine reuse, syndication, excerpt recycling, or duplicate file rows. The ×N pill and stacked "
        "card are visual cues; click the <strong>×N</strong> button to open a table of every matching row in the filter.</p>"
        "<p><strong>Charts</strong> — Snippets-over-time and keyword views use <strong>all snippet rows</strong> in "
        "the filter (not the deduplicated list), so repeated text still contributes to chart totals.</p>"
    )
    st.markdown(
        '<div class="gx-section">'
        '<p class="gx-kicker">📑 Snippets &amp; multiplicity</p>'
        '<h2 class="gx-h2">Unique paragraphs, snippet rows, and “reprints”</h2>'
        "<p class=\"gx-sub\">How the headline counts in the filter bar relate to the list, the ×N badge, and the charts.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(_card("Definitions", _counts_body, filter_mark=False), unsafe_allow_html=True)

    # --- Archetypes ---
    st.markdown(
        '<div class="gx-section">'
        '<p class="gx-kicker">🎭 What rhetoric performs</p>'
        '<h2 class="gx-h2">Archetypes — Pioneer, Builder, Guardian</h2>'
        '<p class="gx-sub">Derived from vision type × stance (below). They are the three ways the field does '
        "interpretation, mobilization, and legitimation—not a separate manual tag.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    for arch, blurb in [
        (
            "Pioneer",
            "Expands possibilities and reframes what the technology is <em>for</em>—speculative openings and novelty. "
            "Tied to <strong>interpretation</strong> in organizing-vision theory.",
        ),
        (
            "Builder",
            "Mobilizes coordinated action: scripts, pilots, “what to do Monday.” Tied to <strong>mobilization</strong>.",
        ),
        (
            "Guardian",
            "Establishes limits, safety, and guardrails. Tied to <strong>legitimation</strong>—securing a license to operate.",
        ),
    ]:
        st.markdown(
            _card(
                arch,
                blurb,
                border=_ARCH_BORDER.get(arch, "#cbd5e1"),
                with_explore=True,
            ),
            unsafe_allow_html=True,
        )
        _explore_link("arch", arch)

    # --- Stances ---
    st.markdown(
        '<div class="gx-section">'
        '<p class="gx-kicker">🗣️ How the paragraph moves</p>'
        '<h2 class="gx-h2">Rhetorical stances</h2>'
        '<p class="gx-sub">These describe delivery—not topic. They were coded qualitatively, then scaled. '
        "Same labels appear in the sidebar filter and in the chip row.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    for stance in ("Opening", "Mobilizing", "Normalizing", "Controlling"):
        st.markdown(
            _card(stance, STANCE_NARRATIVES[stance], with_explore=True),
            unsafe_allow_html=True,
        )
        _explore_link("stance", stance)

    # --- Vision types ---
    st.markdown(
        '<div class="gx-section">'
        '<p class="gx-kicker">🔮 What futures are imagined</p>'
        '<h2 class="gx-h2">Vision types</h2>'
        '<p class="gx-sub">Synchronic categories: what kind of AI future the paragraph conjures. '
        "Names match the dataset exactly; the prose is a readable gloss.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    for vis in sorted(VISION_TO_ARCHETYPES.keys()):
        nar = VISION_NARRATIVES[vis]
        arches = ", ".join(VISION_TO_ARCHETYPES[vis])
        inner = nar + f'<p class="gx-meta">Possible archetypes (depend on stance): {html.escape(arches)}</p>'
        st.markdown(_card(vis, inner, with_explore=True), unsafe_allow_html=True)
        _explore_link("vision", vis)

    # --- Table ---
    st.markdown(
        '<div class="gx-section">'
        '<p class="gx-kicker">📋 The coding grid</p>'
        '<h2 class="gx-h2">Vision × stance → archetype</h2>'
        '<p class="gx-sub">Every paragraph is assigned one archetype from this mapping. The dashboard’s ●/○ hints '
        "refer to it.</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    rows_html = []
    for (v, s), a in sorted(V2A_STANCED.items(), key=lambda x: (x[0][0], x[0][1])):
        rows_html.append(
            "<tr><td>{}</td><td>{}</td><td><strong>{}</strong></td></tr>".format(
                html.escape(v), html.escape(s), html.escape(a)
            )
        )
    st.markdown(
        '<div class="gx-table-wrap"><table><thead><tr>'
        "<th>Vision type</th><th>Rhetorical stance</th><th>Archetype</th></tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table></div>",
        unsafe_allow_html=True,
    )

    with st.expander("Using this explorer", expanded=False):
        st.markdown(
            """
**Filters** — Combine 🎭 archetype, 🔮 vision type(s), 🗣️ stance, 📅 time, 📰 publication, and 🔎 keywords.
Empty multiselects mean “all” for that dimension. **●/○** on selectors reflect the grid above.

**Unique paragraphs vs snippet rows** — With **Merge identical paragraph text (reprints)** on (under **List options**), the snippet toolbar and **Corpus & how counts are built** (bottom of the page) give **distinct paragraph bodies** vs **snippet rows in filter** (every row, including repeats). See **Explanations → Snippets & multiplicity** for definitions.

**Snippet order** — Distinctive vs corpus (log-odds) or representative (centroid) wording; see tooltips on the toolbar.

**Charts** — Snippets over time; word cloud (distinctive or raw frequency).

**Summary statistics** — Corpus hygiene diagnostics; not theory claims.
            """
        )

    st.caption(
        "Framing follows the manuscript’s organizing-vision and attention-ecology argument; category strings match the coded table."
    )

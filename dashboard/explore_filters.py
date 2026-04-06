"""
Apply “Explore from Explanations” filters via session state only (no URL navigation — keeps Streamlit sign-in).
"""
from __future__ import annotations

import streamlit as st

from data_utils import VISION_TO_ARCHETYPES

# Keep in sync with app.py consumers (import these instead of duplicating strings).
AVE_SHOW_GUIDE_KEY = "ave_show_guide"
FILTER_FOCUS_KEY = "filter_focus"
SCROLL_SIDEBAR_KEY = "dash_scroll_sidebar"
DASH_FILTER_FLASH_KEY = "dash_filter_flash"

_ARCH = frozenset({"Pioneer", "Builder", "Guardian"})
_STANCE = frozenset({"Opening", "Mobilizing", "Normalizing", "Controlling"})


def apply_explore_filter(kind: str, value: str) -> bool:
    """
    kind: ``arch`` | ``stance`` | ``vision``.
    Sets dashboard filters, closes Explanations, focuses sidebar, queues chip flash. Returns False if invalid.
    """
    k = kind.strip().lower()
    v = value.strip()
    if not v:
        return False
    if k == "arch":
        if v not in _ARCH:
            return False
        st.session_state["dash_archetypes"] = [v]
        st.session_state["dash_stances"] = []
        st.session_state["dash_vision"] = []
    elif k == "stance":
        if v not in _STANCE:
            return False
        st.session_state["dash_archetypes"] = []
        st.session_state["dash_stances"] = [v]
        st.session_state["dash_vision"] = []
    elif k == "vision":
        if v not in VISION_TO_ARCHETYPES:
            return False
        st.session_state["dash_archetypes"] = []
        st.session_state["dash_stances"] = []
        st.session_state["dash_vision"] = [v]
    else:
        return False
    st.session_state[AVE_SHOW_GUIDE_KEY] = False
    st.session_state[FILTER_FOCUS_KEY] = "primary"
    st.session_state[SCROLL_SIDEBAR_KEY] = True
    st.session_state[DASH_FILTER_FLASH_KEY] = {"arch": "archetype", "stance": "stance", "vision": "vision"}[k]
    return True

"""
Chat Bubble Component -- Top-bar Advisory Assistant for Streamlit
==================================================================
Renders a collapsible chat panel right below the navbar, accessible
from every page via a toggle button. Uses Groq SLM (free API) when
available, falls back to template engine.
Session_state persists chat history across reruns.

Usage:
    from monsoon_textile_app.components.chat_bubble import render_chat_bubble
    render_chat_bubble(dashboard_data=data)
"""

from __future__ import annotations
import streamlit as st


def render_chat_bubble(dashboard_data: dict | None = None) -> None:
    """Render the top-bar chat panel toggle and collapsible chat area."""

    # ── Initialise session state ──
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "advisory_context" not in st.session_state:
        st.session_state.advisory_context = {}

    # ── Build/update advisory context ──
    if not dashboard_data:
        dashboard_data = (
            st.session_state.get("_advisory_dashboard_data")
            or st.session_state.get("_dashboard_data_cache")
        )

    if not dashboard_data:
        try:
            from monsoon_textile_app.data.fetch_real_data import load_all_data
            dashboard_data = load_all_data()
        except Exception:
            pass

    if dashboard_data:
        st.session_state["_advisory_dashboard_data"] = dashboard_data
        try:
            from monsoon_textile_app.components.advisory_engine import build_context
            ctx = build_context(dashboard_data)
            st.session_state.advisory_context = ctx
        except Exception:
            pass

    # ── Check SLM availability ──
    from monsoon_textile_app.components.slm_engine import is_available as _slm_ok
    _using_slm = _slm_ok()
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = "slm" if _using_slm else "template"

    # ── Top-bar toggle button ──
    _engine_tag = "SLM" if _using_slm else "Template"
    _engine_color = "#10b981" if _using_slm else "#6366f1"

    _toggle_cols = st.columns([8, 2])
    with _toggle_cols[1]:
        _btn_text = "Close RainLoom AI" if st.session_state.chat_open else "RainLoom AI"
        if st.button(_btn_text, key="chat_toggle_btn", type="primary",
                     use_container_width=True):
            st.session_state.chat_open = not st.session_state.chat_open
            st.rerun()

    # ── Chat panel (shown when open) ──
    if st.session_state.chat_open:
        _render_chat_panel(_using_slm)

    st.markdown(_CHAT_CSS, unsafe_allow_html=True)


def _render_chat_panel(using_slm: bool = False):
    """Render the chat panel with message history and input."""

    # Engine badge
    _engine_badge = (
        '🟢 **SLM**' if using_slm else '🟣 **Template**'
    )

    with st.container(border=True):
        st.markdown(f"### RainLoom Advisory  |  {_engine_badge}")

        # Display chat history
        for msg in st.session_state.chat_history[-10:]:
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.write(msg["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(msg["content"])

        # Welcome message if no history
        if not st.session_state.chat_history:
            _mode = "Groq Llama 3.1" if using_slm else "template engine"
            with st.chat_message("assistant"):
                st.write(
                    f"**RainLoom AI** (powered by {_mode}): "
                    f"Hello! I can help you interpret the dashboard data. Try asking:"
                )
                st.markdown("""
                - What's the risk for Trident?
                - Should farmers be worried?
                - How should I position my portfolio?
                """)

        st.caption("Quick Queries:")
        # Quick action buttons
        quick_cols = st.columns(4)
        quick_prompts = [
            ("Farmer", "What should farmers do right now?"),
            ("MSME", "What should MSMEs do about cotton procurement?"),
            ("Investor", "How should I position my textile portfolio?"),
            ("Monsoon", "What's the current monsoon status?"),
        ]
        for col, (label, prompt) in zip(quick_cols, quick_prompts):
            with col:
                if st.button(label, key=f"quick_{label}", use_container_width=True):
                    _handle_query(prompt)
                    st.rerun()

        # Helper to handle submission and clear input
        def _on_chat_submit():
            val = st.session_state.chat_text_input
            if val:
                _handle_query(val)
                st.session_state.chat_text_input = "" # Clear the box

        # Text input — using st.text_input for better placement in conditional containers
        st.text_input(
            "Ask about risk, monsoon, cotton...",
            key="chat_text_input",
            on_change=_on_chat_submit
        )


def _handle_query(query: str):
    """Process a user query and append response to chat history."""

    st.session_state.chat_history.append({
        "role": "user",
        "content": query,
    })

    ctx = st.session_state.get("advisory_context", {})

    response = ""
    from monsoon_textile_app.components.slm_engine import is_available, get_slm_response
    if is_available():
        response = get_slm_response(
            query,
            ctx,
            chat_history=st.session_state.chat_history[:-1],
        )

    if not response:
        from monsoon_textile_app.components.advisory_engine import get_advisory
        response = get_advisory(query, ctx)

    st.session_state.chat_history.append({
        "role": "assistant",
        "content": response,
    })


# ═══════════════════════════════════════════════════════════════════════════
# CSS for chat button styling
# ═══════════════════════════════════════════════════════════════════════════

_CHAT_CSS = """
<style>
/* Chat input styling */
div[data-testid="stChatInput"] {
    background: rgba(15, 23, 42, 0.80) !important;
    border: 1px solid rgba(59, 130, 246, 0.25) !important;
    border-radius: 12px !important;
}
div[data-testid="stChatInput"] textarea {
    color: #e2e8f0 !important;
}
</style>
"""

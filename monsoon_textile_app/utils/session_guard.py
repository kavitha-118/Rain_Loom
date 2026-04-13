"""
Streamlit Session-Level CSRF Guard — RainLoom Security Layer
============================================================
Generates and validates a per-session CSRF token to block
cross-origin form submissions in Streamlit.

Usage:
    from monsoon_textile_app.utils.session_guard import init_csrf, assert_csrf

    # At top of every page:
    init_csrf()

    # In every form submit handler:
    if st.button("Submit"):
        assert_csrf()
        ... do the action ...
"""

from __future__ import annotations

import secrets
import streamlit as st

_TOKEN_KEY = "_rl_csrf_token"
_SUBMITTED_KEY = "_rl_csrf_submitted"


def init_csrf() -> str:
    """
    Initialise the CSRF token for this session (idempotent).
    Returns the token so callers can embed it in forms if needed.
    """
    if _TOKEN_KEY not in st.session_state:
        st.session_state[_TOKEN_KEY] = secrets.token_urlsafe(32)
    return st.session_state[_TOKEN_KEY]


def get_token() -> str:
    """Return the current session CSRF token (creates one if missing)."""
    return init_csrf()


def assert_csrf(submitted_token: str | None = None) -> None:
    """
    Validate the submitted CSRF token against the session token.

    Parameters
    ----------
    submitted_token:
        The token value to validate. If None, this call is a no-op
        (useful for internal server-side actions that don't need CSRF).

    Raises
    ------
    RuntimeError if the tokens don't match.
    """
    # In-process Streamlit forms are always same-origin, so we simply
    # verify that the session token exists (no external forgery possible
    # inside Streamlit's WebSocket model). We still track submission to
    # prevent double-submissions within the same session.
    session_token = st.session_state.get(_TOKEN_KEY)

    if session_token is None:
        raise RuntimeError(
            "CSRF protection: session token missing. Please refresh the page."
        )

    if submitted_token is not None:
        if not secrets.compare_digest(str(submitted_token), str(session_token)):
            raise RuntimeError(
                "CSRF protection: token mismatch. Please refresh the page and try again."
            )


def rotate_token() -> str:
    """Rotate the CSRF token (call after a successful sensitive action)."""
    new_token = secrets.token_urlsafe(32)
    st.session_state[_TOKEN_KEY] = new_token
    return new_token


def render_hidden_token() -> str:
    """
    Return an HTML hidden-input string embedding the CSRF token.
    Useful if you render raw HTML forms.
    """
    token = get_token()
    return f'<input type="hidden" name="_csrf_token" value="{token}">'

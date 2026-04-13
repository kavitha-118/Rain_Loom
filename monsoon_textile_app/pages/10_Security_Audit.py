"""
Page 10 — Security & Audit Log Dashboard
==========================================
Displays the real-time audit event trail, security health summary,
subscriber management, and API key table.

Access: All users see audit summary metrics.
        Full log + subscriber list + key management require admin password.
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import pandas as pd
from datetime import datetime

from monsoon_textile_app.components.navbar import render_navbar
from monsoon_textile_app.components.chat_bubble import render_chat_bubble
from monsoon_textile_app.utils.session_guard import init_csrf

st.set_page_config(
    page_title="Security & Audit — RainLoom",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed",
)

render_navbar(active_page="Overview")
render_chat_bubble()
init_csrf()

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Fira+Code:wght@400;500&display=swap');

:root {
  --bg:         #0a0f1e;
  --card:       rgba(15,23,42,0.65);
  --border:     rgba(255,255,255,0.07);
  --accent:     #3b82f6;
  --green:      #10b981;
  --red:        #ef4444;
  --gold:       #f59e0b;
  --purple:     #8b5cf6;
  --text:       #e2e8f0;
  --muted:      #64748b;
}

html, body, [data-testid="stAppViewContainer"] {
  font-family: 'Inter', sans-serif;
  background: var(--bg);
  color: var(--text);
}

[data-testid="stAppViewContainer"] {
  background: radial-gradient(ellipse 80% 50% at 50% -20%, rgba(59,130,246,.07) 0%, transparent 60%),
              radial-gradient(ellipse 60% 40% at 80% 100%, rgba(139,92,246,.04) 0%, transparent 50%),
              #0a0f1e;
}
#MainMenu, footer, header,
[data-testid="stDeployButton"],
[data-testid="stToolbar"] { display:none!important; }

.page-hero { text-align:center; padding:2rem 0 1rem; animation: fadeUp .7s ease both; }
.page-hero h1 {
  font-size:2.3rem; font-weight:800; letter-spacing:-.03em;
  background: linear-gradient(135deg,#e2e8f0 0%,#94a3b8 40%,#8b5cf6 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  background-clip:text; margin:0;
}
.page-hero p { color:var(--muted); font-size:.97rem; margin-top:.5rem; }

/* KPI strip */
.kpi-strip { display:flex; flex-wrap:wrap; gap:.75rem; margin:1.2rem 0; }
.kpi-card {
  flex:1; min-width:140px;
  background:var(--card); border:1px solid var(--border);
  border-radius:14px; padding:1.1rem 1.3rem;
  backdrop-filter:blur(16px); animation: fadeUp .5s ease both;
}
.kpi-val  { font-size:1.85rem; font-weight:800; }
.kpi-lbl  { font-size:.8rem; color:var(--muted); text-transform:uppercase; letter-spacing:.08em; margin-top:.2rem; }

/* Section divider */
.sec-div {
  height:1px; margin:1.5rem 0;
  background:linear-gradient(90deg,transparent,rgba(99,102,241,.3),transparent);
}

/* Glass panel */
.glass-panel {
  background:var(--card); border:1px solid var(--border);
  border-radius:16px; padding:1.4rem 1.6rem; margin-bottom:1rem;
  backdrop-filter:blur(16px); animation:fadeUp .5s ease both;
}
.glass-panel:hover { border-color:rgba(99,102,241,.3); }

/* Event row */
.ev-row {
  display:grid; grid-template-columns:160px 90px 120px 100px 1fr;
  gap:.5rem; padding:.55rem .9rem;
  border-bottom:1px solid var(--border); font-size:.87rem;
  transition:background .2s;
}
.ev-row:hover { background:rgba(99,102,241,.05); }
.ev-row:last-child { border-bottom:none; }
.ev-header {
  display:grid; grid-template-columns:160px 90px 120px 100px 1fr;
  gap:.5rem; padding:.5rem .9rem;
  font-size:.78rem; font-weight:700; color:var(--muted);
  text-transform:uppercase; letter-spacing:.08em;
  background:rgba(99,102,241,.08); border-radius:8px; margin-bottom:.25rem;
}

.badge {
  display:inline-block; padding:.15rem .55rem; border-radius:20px;
  font-size:.76rem; font-weight:700; letter-spacing:.04em;
}
.badge-critical { background:rgba(239,68,68,.15);  color:#ef4444; }
.badge-warning  { background:rgba(245,158,11,.15); color:#f59e0b; }
.badge-info     { background:rgba(59,130,246,.15); color:#3b82f6; }

.code-mono { font-family:'Fira Code',monospace; font-size:.84rem; color:#a5b4fc; }

/* Admin gate */
.admin-gate {
  background:rgba(139,92,246,.08); border:1px solid rgba(139,92,246,.25);
  border-radius:14px; padding:1.5rem 1.8rem; text-align:center;
}
.admin-gate h3 { color:#a78bfa; margin-bottom:.5rem; }
.admin-gate p  { color:var(--muted); font-size:.9rem; margin-bottom:1rem; }

@keyframes fadeUp {
  from { opacity:0; transform:translateY(16px); }
  to   { opacity:1; transform:translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-hero">
  <h1>🔐 Security & Audit Log</h1>
  <p>Real-time security events, subscriber management, and API key administration</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

# ── Security Summary KPIs ──────────────────────────────────────────────────────
try:
    from monsoon_textile_app.utils.audit_log import security_summary, get_recent_events
    summary = security_summary()
    events  = get_recent_events(200)
except Exception as e:
    summary = {}
    events  = []
    st.warning(f"Audit log not available yet: {e}")

col1, col2, col3, col4, col5, col6 = st.columns(6)
kpis = [
    (col1, summary.get("total_events",    0),  "#3b82f6", "Total Events"),
    (col2, summary.get("critical_events", 0),  "#ef4444", "Critical"),
    (col3, summary.get("warnings",        0),  "#f59e0b", "Warnings"),
    (col4, summary.get("rate_limit_hits", 0),  "#8b5cf6", "Rate-Limit Hits"),
    (col5, summary.get("invalid_key_attempts", 0), "#ef4444", "Bad Key Attempts"),
    (col6, summary.get("emails_dispatched",    0), "#10b981", "Emails Sent"),
]
for col, val, color, label in kpis:
    col.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-val" style="color:{color};">{val}</div>
      <div class="kpi-lbl">{label}</div>
    </div>""", unsafe_allow_html=True)

st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

# ── Live Event Feed ────────────────────────────────────────────────────────────
st.markdown("### 📋 Live Audit Event Feed")

CAT_COLORS = {
    "auth":      "#3b82f6",
    "subscribe": "#10b981",
    "alert":     "#f59e0b",
    "admin":     "#8b5cf6",
    "security":  "#ef4444",
    "system":    "#64748b",
}

def _badge(severity: str) -> str:
    cls = f"badge-{severity}" if severity in ("critical","warning","info") else "badge-info"
    return f'<span class="badge {cls}">{severity.upper()}</span>'

if events:
    with st.container():
        st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
        st.markdown("""
        <div class="ev-header">
          <span>Timestamp</span><span>Severity</span><span>Category</span>
          <span>Actor</span><span>Action / Details</span>
        </div>""", unsafe_allow_html=True)
        for ev in events[:100]:
            ts_raw  = ev.get("ts", "")
            try:
                ts_fmt = datetime.fromisoformat(ts_raw).strftime("%d %b %H:%M:%S")
            except Exception:
                ts_fmt = ts_raw[:19]
            cat     = ev.get("category", "system")
            cat_col = CAT_COLORS.get(cat, "#64748b")
            action  = ev.get("action", "")
            user    = ev.get("user", "—")[:20]
            sev     = ev.get("severity", "info")
            details = str(ev.get("details", {}))[:60]
            st.markdown(f"""
            <div class="ev-row">
              <span class="code-mono" style="color:#94a3b8;">{ts_fmt}</span>
              <span>{_badge(sev)}</span>
              <span style="color:{cat_col};font-weight:600;">{cat}</span>
              <span style="color:#94a3b8;overflow:hidden;text-overflow:ellipsis;">{user}</span>
              <span style="color:#cbd5e1;">{action} <span style="color:#475569;">{details}</span></span>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.info("No audit events recorded yet. Events appear here as users interact with the system.")

st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)

# ── Admin Panel (password-gated) ───────────────────────────────────────────────
st.markdown("### 🛡️ Admin Panel")

_admin_key = "_rl_admin_authed"
if _admin_key not in st.session_state:
    st.session_state[_admin_key] = False

if not st.session_state[_admin_key]:
    st.markdown('<div class="admin-gate">', unsafe_allow_html=True)
    st.markdown("""
    <h3>🔒 Admin Access Required</h3>
    <p>Subscriber list, full audit log, and API key management are restricted to admins.</p>
    """, unsafe_allow_html=True)
    import os
    admin_pass_input = st.text_input(
        "Admin Password", type="password", key="admin_pass_input",
        placeholder="Enter admin password…"
    )
    if st.button("Unlock Admin Panel", type="primary", key="btn_admin_unlock"):
        expected = os.environ.get("ADMIN_PASSWORD_HASH", "")
        raw_pass  = admin_pass_input.strip()
        authed = False
        if expected:
            try:
                import bcrypt
                authed = bcrypt.checkpw(raw_pass.encode(), expected.encode())
            except Exception:
                # fallback: plain comparison (dev only)
                authed = (raw_pass == expected)
        else:
            # No password set — allow access in dev mode with a warning
            authed = (raw_pass == "admin")
            st.warning("⚠️ No ADMIN_PASSWORD_HASH set in .env — using default dev password 'admin'. Set this before production.")
        if authed:
            st.session_state[_admin_key] = True
            try:
                from monsoon_textile_app.utils.audit_log import audit
                audit("admin", "admin_panel_access", severity="warning",
                      details={"page": "Security Audit"})
            except Exception:
                pass
            st.rerun()
        else:
            st.error("❌ Incorrect password.")
    st.markdown("</div>", unsafe_allow_html=True)

else:
    st.success("✅ Admin panel unlocked")
    if st.button("🔒 Lock Admin Panel", key="btn_admin_lock"):
        st.session_state[_admin_key] = False
        st.rerun()

    admin_tab1, admin_tab2, admin_tab3 = st.tabs([
        "👥 Subscribers", "🔑 API Keys", "📊 Full Audit Log"
    ])

    # ── Subscribers ──────────────────────────────────────────────────────────
    with admin_tab1:
        st.markdown("#### Email Subscribers")
        try:
            from monsoon_textile_app.api.data_bridge import (
                get_subscriber_list, remove_subscriber
            )
            subs = get_subscriber_list()
        except Exception as e:
            subs = []
            st.error(f"Could not load subscribers: {e}")

        if subs:
            df_subs = pd.DataFrame([
                {
                    "Email":       s.get("email", ""),
                    "Verified":    "✅" if s.get("verified", True) else "⏳ Pending",
                    "Alert Types": ", ".join(s.get("alert_types", [])),
                    "Subscribed":  s.get("subscribed_at", "—")[:10],
                }
                for s in subs
            ])
            st.dataframe(df_subs, use_container_width=True, hide_index=True)
            st.caption(f"Total: {len(subs)} subscriber(s)")

            st.markdown("---")
            st.markdown("**Remove a subscriber**")
            rem_email = st.text_input("Email to remove", key="admin_rem_email",
                                       placeholder="user@example.com")
            if st.button("Remove Subscriber", key="btn_admin_remove", type="secondary"):
                if rem_email:
                    result = remove_subscriber(rem_email.strip().lower())
                    if result["status"] == "unsubscribed":
                        st.success(result["message"])
                    else:
                        st.warning(result["message"])
        else:
            st.info("No subscribers yet.")

    # ── API Keys ──────────────────────────────────────────────────────────────
    with admin_tab2:
        st.markdown("#### API Keys")
        try:
            from monsoon_textile_app.api.auth import list_api_keys, generate_api_key, revoke_api_key
            keys = list_api_keys()
        except Exception as e:
            keys = []
            st.error(f"Could not load API keys: {e}")

        if keys:
            df_keys = pd.DataFrame([
                {
                    "Key ID":     k.get("key_id", ""),
                    "Name":       k.get("name", ""),
                    "Active":     "✅" if k.get("active", True) else "❌ Revoked",
                    "Calls":      k.get("call_count", 0),
                    "Created":    str(k.get("created_at", ""))[:10],
                    "Last Used":  str(k.get("last_used", "—"))[:10],
                }
                for k in keys
            ])
            st.dataframe(df_keys, use_container_width=True, hide_index=True)

        st.markdown("---")
        col_gen, col_rev = st.columns(2)
        with col_gen:
            st.markdown("**Generate new key**")
            new_key_name = st.text_input("Application name", key="admin_key_name",
                                          placeholder="e.g. ICICI Risk Engine")
            if st.button("Generate Key", key="btn_admin_gen_key", type="primary"):
                if new_key_name.strip():
                    result = generate_api_key(new_key_name.strip(), created_by="admin-panel")
                    st.success(f"Key generated for **{new_key_name}**")
                    st.code(result["raw_key"], language="text")
                    st.warning("⚠️ Copy this key now — it will NOT be shown again.")
                    try:
                        from monsoon_textile_app.utils.audit_log import audit
                        audit("auth", "api_key_generated", severity="warning",
                              user="admin-panel", details={"name": new_key_name})
                    except Exception:
                        pass
                else:
                    st.warning("Enter an application name first.")
        with col_rev:
            st.markdown("**Revoke a key**")
            rev_key_id = st.text_input("Key ID to revoke", key="admin_rev_key",
                                        placeholder="kid_xxxxxx")
            if st.button("Revoke Key", key="btn_admin_rev", type="secondary"):
                if rev_key_id.strip():
                    success = revoke_api_key(rev_key_id.strip())
                    if success:
                        st.success(f"Key `{rev_key_id}` revoked.")
                    else:
                        st.warning(f"Key `{rev_key_id}` not found.")
                else:
                    st.warning("Enter a Key ID first.")

    # ── Full Audit Log ────────────────────────────────────────────────────────
    with admin_tab3:
        st.markdown("#### Full Audit Log (disk)")
        try:
            from monsoon_textile_app.utils.audit_log import load_all_events, event_counts_by_category
            all_events = load_all_events(500)
            counts     = event_counts_by_category()
        except Exception as e:
            all_events = []
            counts     = {}
            st.error(f"Could not load audit log: {e}")

        if counts:
            count_cols = st.columns(len(counts))
            for i, (cat, cnt) in enumerate(counts.items()):
                color = CAT_COLORS.get(cat, "#64748b")
                count_cols[i].metric(cat.upper(), cnt)

        if all_events:
            df_log = pd.DataFrame([
                {
                    "Time":     ev.get("ts", "")[:19],
                    "Severity": ev.get("severity", ""),
                    "Category": ev.get("category", ""),
                    "Action":   ev.get("action", ""),
                    "User":     ev.get("user", ""),
                    "IP":       ev.get("ip", ""),
                    "Details":  str(ev.get("details", {}))[:80],
                }
                for ev in all_events
            ])
            st.dataframe(df_log, use_container_width=True, hide_index=True)

            # CSV download
            csv = df_log.to_csv(index=False)
            st.download_button(
                "⬇️ Download Audit CSV",
                data=csv,
                file_name=f"rainloom_audit_{datetime.utcnow().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No events in log file yet.")

# ── Security Info Cards ────────────────────────────────────────────────────────
st.markdown('<div class="sec-div"></div>', unsafe_allow_html=True)
st.markdown("### 🛡️ Active Security Controls")

sec_col1, sec_col2, sec_col3 = st.columns(3)

with sec_col1:
    st.markdown("""
    <div class="glass-panel">
      <div style="font-size:1.4rem;margin-bottom:.5rem;">🔑</div>
      <div style="font-weight:700;color:#e2e8f0;margin-bottom:.4rem;">API Key Auth</div>
      <div style="color:#94a3b8;font-size:.88rem;line-height:1.6;">
        All write endpoints require <code>X-Api-Key</code> header.<br>
        Keys stored as SHA-256 hashes — raw keys never persisted.<br>
        Key generation/revocation tracked in audit log.
      </div>
    </div>""", unsafe_allow_html=True)

with sec_col2:
    st.markdown("""
    <div class="glass-panel">
      <div style="font-size:1.4rem;margin-bottom:.5rem;">⏱️</div>
      <div style="font-weight:700;color:#e2e8f0;margin-bottom:.4rem;">Rate Limiting</div>
      <div style="color:#94a3b8;font-size:.88rem;line-height:1.6;">
        Sliding-window per-IP limits:<br>
        • Subscribe: 5 req / 60 s<br>
        • Read endpoints: 60 req / 60 s<br>
        • Global: 300 req / 60 s
      </div>
    </div>""", unsafe_allow_html=True)

with sec_col3:
    st.markdown("""
    <div class="glass-panel">
      <div style="font-size:1.4rem;margin-bottom:.5rem;">📧</div>
      <div style="font-weight:700;color:#e2e8f0;margin-bottom:.4rem;">Email Verification</div>
      <div style="color:#94a3b8;font-size:.88rem;line-height:1.6;">
        New subscribers receive a verification email.<br>
        Only verified emails receive alerts.<br>
        Unsubscribe links are HMAC-signed (tamper-proof).
      </div>
    </div>""", unsafe_allow_html=True)

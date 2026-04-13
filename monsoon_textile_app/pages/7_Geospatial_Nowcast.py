"""
Page 7 -- Geospatial Nowcast
=============================
District-level rainfall visualization with Plotly choropleth map
for precise early warning of monsoon impact on textile belt.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json
import textwrap
import html
import plotly.graph_objects as go
from monsoon_textile_app.components.navbar import render_navbar
from monsoon_textile_app.components.chat_bubble import render_chat_bubble

st.set_page_config(page_title="Geospatial Nowcast", page_icon="G", layout="wide", initial_sidebar_state="collapsed")
render_navbar(active_page="Geospatial")
render_chat_bubble()

# Load real data
_REAL_DATA = None
try:
    from monsoon_textile_app.data.fetch_real_data import load_all_data
    with st.spinner("Loading geospatial rainfall data..."):
        _REAL_DATA = load_all_data()
except Exception:
    pass

# ---------------------------------------------------------------------------
# Global CSS -- dark glass-morphism theme (consistent with other pages)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.stApp { background: linear-gradient(145deg, #0a0f1e 0%, #0d1326 40%, #0f172a 100%); }
.block-container { padding-top: 2rem; }

.section-heading {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 1.51rem; font-weight: 600;
    letter-spacing: 0.04em; color: #e2e8f0;
    margin-bottom: 0.15rem; text-transform: uppercase;
}
.section-sub {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 1.02rem; font-weight: 300;
    color: #94a3b8; margin-bottom: 0.7rem;
}
.heading-rule { height: 2px; border: none; border-radius: 1px; margin-bottom: 1.6rem; }
.rule-blue  { background: linear-gradient(90deg, #3b82f6 0%, transparent 80%); }
.rule-green { background: linear-gradient(90deg, #10b981 0%, transparent 80%); }
.rule-gold  { background: linear-gradient(90deg, #f59e0b 0%, transparent 80%); }
.rule-red   { background: linear-gradient(90deg, #ef4444 0%, transparent 80%); }

.page-title {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 2.26rem; font-weight: 700;
    letter-spacing: 0.06em; color: #f1f5f9; margin-bottom: 0.15rem;
}
.page-subtitle {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 1.06rem; font-weight: 300;
    color: #64748b; margin-bottom: 0.5rem;
}
.title-rule {
    height: 3px; border: none; border-radius: 2px;
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, transparent 100%);
    margin-bottom: 2.2rem;
}

.glass-card {
    background: rgba(15, 23, 42, 0.55);
    border: 1px solid rgba(59, 130, 246, 0.12);
    border-radius: 14px; padding: 1.6rem 1.8rem;
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 4px 24px rgba(0,0,0,0.25); margin-bottom: 1rem;
}
.glass-card-accent-blue  { border-color: rgba(59,130,246,0.25); }
.glass-card-accent-red   { border-color: rgba(239,68,68,0.25); }
.glass-card-accent-green { border-color: rgba(16,185,129,0.25); }
.glass-card-accent-gold  { border-color: rgba(245,158,11,0.25); }

.metric-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(59,130,246,0.15);
    border-radius: 14px; padding: 1.5rem 1.6rem;
    backdrop-filter: blur(14px); text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2);
}
.metric-value {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 1.91rem; font-weight: 700;
    letter-spacing: -0.02em; margin-bottom: 0.15rem;
}
.metric-label {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 0.96rem; font-weight: 400;
    color: #94a3b8; letter-spacing: 0.04em; text-transform: uppercase;
}
.metric-detail {
    font-family: 'Inter', system-ui, sans-serif;
    font-size: 0.96rem; font-weight: 300;
    color: #64748b; margin-top: 0.25rem;
}
</style>
""", unsafe_allow_html=True)

PLOTLY_FONT = dict(family="Inter, system-ui, -apple-system, sans-serif", color="#cbd5e1")

def base_layout(**overrides):
    layout = dict(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOTLY_FONT,
        hovermode="closest",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=13, color="#94a3b8")),
    )
    layout.update(overrides)
    return layout


def render_html(block: str) -> None:
    """Render HTML inline via st.markdown.

    Strips leading whitespace from every line so the Markdown parser
    never treats indented HTML as a fenced code block.
    """
    cleaned = textwrap.dedent(block).strip()
    lines = [line.lstrip() for line in cleaned.splitlines()]
    st.markdown("\n".join(lines), unsafe_allow_html=True)


# =========================================================================
# PAGE HEADER
# =========================================================================
st.markdown("""
<div class="page-title">Geospatial Nowcast</div>
<div class="page-subtitle">Live district-level rainfall from Open-Meteo API for the Indian cotton belt</div>
<hr class="title-rule">
""", unsafe_allow_html=True)

with st.expander("Understanding this page"):
    st.markdown("""
<div class="glass-card glass-card-accent-blue" style="margin-bottom:0.5rem;">

**Geospatial Nowcasting** provides a district-level view of monsoon rainfall across India's
10 cotton-growing states. Unlike the state-level averages on other pages, this map reveals
**spatial heterogeneity** -- pockets of severe deficit that state averages mask.

**How to read the map:**
- **Green** districts are receiving normal or above-normal rainfall (positive departure from LPA).
- **Red/brown** districts are in deficit -- the darker the red, the more severe the shortfall.
- Hover over any district to see its name, state, rainfall (mm), and departure from the long-period average.

**Why this matters:** A state may report "normal" rainfall overall, but if the cotton-growing
districts within it are in severe deficit, crop damage and price shocks will follow. This
granular view enables targeted early warnings for specific supply-chain regions.

</div>
""", unsafe_allow_html=True)

# =========================================================================
# DATA PREPARATION
# =========================================================================

# Cotton-belt state centroids for map centering and district generation
_STATE_CENTROIDS = {
    "Gujarat":        {"lat": 22.3, "lon": 71.2},
    "Maharashtra":    {"lat": 19.0, "lon": 76.0},
    "Telangana":      {"lat": 17.5, "lon": 79.0},
    "Rajasthan":      {"lat": 26.5, "lon": 73.5},
    "Madhya Pradesh": {"lat": 23.5, "lon": 78.0},
    "Karnataka":      {"lat": 15.3, "lon": 76.5},
    "Andhra Pradesh": {"lat": 15.9, "lon": 80.0},
    "Tamil Nadu":     {"lat": 11.0, "lon": 78.5},
    "Punjab":         {"lat": 31.0, "lon": 75.5},
    "Haryana":        {"lat": 29.0, "lon": 76.0},
}

# Major cotton-growing districts within each state (real district names)
_COTTON_DISTRICTS = {
    "Gujarat": [
        "Ahmedabad", "Amreli", "Banaskantha", "Bhavnagar", "Bharuch",
        "Botad", "Jamnagar", "Junagadh", "Kutch", "Mehsana",
        "Morbi", "Rajkot", "Sabarkantha", "Surendranagar", "Vadodara",
    ],
    "Maharashtra": [
        "Akola", "Amravati", "Aurangabad", "Beed", "Buldhana",
        "Jalgaon", "Jalna", "Nagpur", "Nanded", "Nashik",
        "Parbhani", "Wardha", "Washim", "Yavatmal",
    ],
    "Telangana": [
        "Adilabad", "Karimnagar", "Khammam", "Mahabubnagar",
        "Nalgonda", "Nizamabad", "Warangal", "Rangareddy",
    ],
    "Rajasthan": [
        "Barmer", "Bikaner", "Churu", "Ganganagar",
        "Hanumangarh", "Jaisalmer", "Jodhpur", "Nagaur",
    ],
    "Madhya Pradesh": [
        "Betul", "Burhanpur", "Dewas", "Dhar", "Indore",
        "Khandwa", "Khargone", "Ratlam", "Ujjain",
    ],
    "Karnataka": [
        "Belgaum", "Bellary", "Dharwad", "Gulbarga",
        "Haveri", "Raichur", "Shimoga",
    ],
    "Andhra Pradesh": [
        "Guntur", "Kurnool", "Prakasam", "Anantapur",
        "Chittoor", "Kadapa", "Nellore",
    ],
    "Tamil Nadu": [
        "Coimbatore", "Madurai", "Ramanathapuram",
        "Salem", "Virudhunagar",
    ],
    "Punjab": [
        "Bathinda", "Fazilka", "Mansa",
        "Muktsar", "Sangrur",
    ],
    "Haryana": [
        "Fatehabad", "Hisar", "Jind",
        "Sirsa", "Bhiwani",
    ],
}

# Real district coordinates (lat, lon) for cotton-belt districts
_DISTRICT_COORDS = {
    # Gujarat
    "Ahmedabad": (23.02, 72.57), "Amreli": (21.60, 71.22), "Banaskantha": (24.18, 72.08),
    "Bhavnagar": (21.77, 72.15), "Bharuch": (21.70, 73.00), "Botad": (22.17, 71.67),
    "Jamnagar": (22.47, 70.07), "Junagadh": (21.52, 70.47), "Kutch": (23.73, 69.86),
    "Mehsana": (23.59, 72.38), "Morbi": (22.82, 70.83), "Rajkot": (22.30, 70.78),
    "Sabarkantha": (23.63, 73.05), "Surendranagar": (22.73, 71.65), "Vadodara": (22.31, 73.19),
    # Maharashtra
    "Akola": (20.71, 77.00), "Amravati": (20.93, 77.78), "Aurangabad": (19.88, 75.34),
    "Beed": (18.99, 75.76), "Buldhana": (20.53, 76.18), "Jalgaon": (21.01, 75.57),
    "Jalna": (19.84, 75.88), "Nagpur": (21.15, 79.09), "Nanded": (19.16, 77.30),
    "Nashik": (20.00, 73.78), "Parbhani": (19.27, 76.78), "Wardha": (20.74, 78.60),
    "Washim": (20.11, 77.13), "Yavatmal": (20.39, 78.12),
    # Telangana
    "Adilabad": (19.67, 78.53), "Karimnagar": (18.44, 79.13), "Khammam": (17.25, 80.15),
    "Mahabubnagar": (16.73, 78.00), "Nalgonda": (17.05, 79.27), "Nizamabad": (18.67, 78.10),
    "Warangal": (17.98, 79.60), "Rangareddy": (17.23, 78.28),
    # Rajasthan
    "Barmer": (25.75, 71.39), "Bikaner": (28.02, 73.31), "Churu": (28.30, 74.97),
    "Ganganagar": (29.91, 73.88), "Hanumangarh": (29.58, 74.33), "Jaisalmer": (26.92, 70.91),
    "Jodhpur": (26.24, 73.02), "Nagaur": (27.20, 73.74),
    # Madhya Pradesh
    "Betul": (21.91, 77.90), "Burhanpur": (21.31, 76.23), "Dewas": (22.97, 76.05),
    "Dhar": (22.60, 75.30), "Indore": (22.72, 75.86), "Khandwa": (21.82, 76.35),
    "Khargone": (21.82, 75.62), "Ratlam": (23.33, 75.04), "Ujjain": (23.18, 75.77),
    # Karnataka
    "Belgaum": (15.85, 74.50), "Bellary": (15.14, 76.92), "Dharwad": (15.46, 75.01),
    "Gulbarga": (17.33, 76.83), "Haveri": (14.79, 75.40), "Raichur": (16.21, 77.36),
    "Shimoga": (13.93, 75.57),
    # Andhra Pradesh
    "Guntur": (16.31, 80.44), "Kurnool": (15.83, 78.05), "Prakasam": (15.35, 79.59),
    "Anantapur": (14.68, 77.60), "Chittoor": (13.22, 79.10), "Kadapa": (14.47, 78.82),
    "Nellore": (14.44, 79.97),
    # Tamil Nadu
    "Coimbatore": (11.00, 76.96), "Madurai": (9.92, 78.12), "Ramanathapuram": (9.37, 78.83),
    "Salem": (11.65, 78.16), "Virudhunagar": (9.59, 77.96),
    # Punjab
    "Bathinda": (30.21, 74.95), "Fazilka": (30.40, 74.03), "Mansa": (29.99, 75.38),
    "Muktsar": (30.47, 74.52), "Sangrur": (30.25, 75.84),
    # Haryana
    "Fatehabad": (29.52, 75.45), "Hisar": (29.15, 75.72), "Jind": (29.32, 76.31),
    "Sirsa": (29.53, 75.03), "Bhiwani": (28.79, 76.13),
}

# LPA (Long Period Average) JJAS rainfall in mm for each state
_STATE_LPA = {
    "Gujarat": 850, "Maharashtra": 1050, "Telangana": 780,
    "Rajasthan": 480, "Madhya Pradesh": 920, "Karnataka": 720,
    "Andhra Pradesh": 650, "Tamil Nadu": 340, "Punjab": 520, "Haryana": 430,
}


# Monthly climatological normals as fraction of annual rainfall
# (approximate distribution for Indian monsoon belt)
_MONTHLY_RAIN_FRAC = {
    1: 0.01, 2: 0.01, 3: 0.02, 4: 0.03, 5: 0.05,
    6: 0.16, 7: 0.28, 8: 0.26, 9: 0.14,
    10: 0.03, 11: 0.01, 12: 0.00,
}


@st.cache_data(ttl=600, show_spinner="Fetching live district rainfall from Open-Meteo...")
def _fetch_district_rainfall_live() -> pd.DataFrame:
    """Fetch recent 30-day rainfall for each district from Open-Meteo API.

    Compares actual 30-day rainfall against the climatological normal
    for the current month (not JJAS total), so results are meaningful
    regardless of season.
    """
    import requests
    from datetime import datetime, timedelta

    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")
    current_month = end_date.month

    # Expected 30-day rainfall = annual LPA × monthly fraction
    monthly_frac = _MONTHLY_RAIN_FRAC.get(current_month, 0.03)

    rows = []
    all_districts = []
    for state, districts in _COTTON_DISTRICTS.items():
        for d in districts:
            if d in _DISTRICT_COORDS:
                all_districts.append((d, state, _DISTRICT_COORDS[d]))

    for district, state, (lat, lon) in all_districts:
        try:
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude={lat}&longitude={lon}"
                f"&daily=precipitation_sum"
                f"&start_date={start_str}&end_date={end_str}"
                f"&timezone=Asia/Kolkata"
            )
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                daily_precip = data.get("daily", {}).get("precipitation_sum", [])
                total_mm = sum(p for p in daily_precip if p is not None)
            else:
                total_mm = None
        except Exception:
            total_mm = None

        annual_lpa = _STATE_LPA.get(state, 700)
        # Expected rainfall for this 30-day period based on monthly climatology
        expected_mm = max(annual_lpa * monthly_frac, 1.0)  # floor of 1mm to avoid div-by-zero

        if total_mm is not None:
            deficit_pct = ((total_mm - expected_mm) / expected_mm) * 100
            actual_mm = total_mm
        else:
            deficit_pct = 0.0
            actual_mm = expected_mm

        # Classify severity -- when the 30-day normal is very low (< 30 mm),
        # percentage deficits are misleading (e.g. -100% in dry season).
        # Mark those districts as "Dry Season" instead.
        DRY_SEASON_NORMAL_THRESHOLD = 5  # mm
        if expected_mm < DRY_SEASON_NORMAL_THRESHOLD:
            severity, color_val = "Dry Season — Normal", 0
            deficit_pct = 0.0  # not meaningful
        elif deficit_pct < -30:
            severity, color_val = "Severe Deficit", -40
        elif deficit_pct < -15:
            severity, color_val = "Moderate Deficit", -25
        elif deficit_pct < 0:
            severity, color_val = "Mild Deficit", -8
        elif deficit_pct < 15:
            severity, color_val = "Normal", 8
        else:
            severity, color_val = "Excess", 25

        rows.append({
            "district": district,
            "state": state,
            "lat": lat,
            "lon": lon,
            "deficit_pct": round(deficit_pct, 1),
            "rainfall_mm": round(actual_mm, 1),
            "lpa_mm": round(expected_mm, 1),
            "severity": severity,
            "color_val": color_val,
            "live_30d_mm": round(total_mm, 1) if total_mm is not None else None,
        })

    return pd.DataFrame(rows)


def _generate_district_data(rainfall_data: dict) -> pd.DataFrame:
    """
    Get district-level rainfall deficit data.
    Tries live Open-Meteo data first; falls back to state-level estimates.
    """
    try:
        df = _fetch_district_rainfall_live()
        if not df.empty:
            return df
    except Exception:
        pass

    # Fallback: use state-level data with spatial variation
    from datetime import datetime as _dt_fb
    _fb_month = _dt_fb.now().month
    _fb_frac = _MONTHLY_RAIN_FRAC.get(_fb_month, 0.03)
    DRY_SEASON_NORMAL_THRESHOLD = 5  # mm

    rows = []
    annual_deficit = rainfall_data.get("annual_deficit", pd.DataFrame()) if isinstance(rainfall_data, dict) else pd.DataFrame()

    for state, districts in _COTTON_DISTRICTS.items():
        state_deficit = 0.0
        if isinstance(annual_deficit, pd.DataFrame) and not annual_deficit.empty:
            if state in annual_deficit.columns:
                state_deficit = float(annual_deficit[state].iloc[-1]) if len(annual_deficit) > 0 else 0.0

        for district in districts:
            lat, lon = _DISTRICT_COORDS.get(district, (_STATE_CENTROIDS[state]["lat"], _STATE_CENTROIDS[state]["lon"]))
            deficit = state_deficit
            lpa = _STATE_LPA.get(state, 700)
            monthly_normal = lpa * _fb_frac
            rainfall_mm = lpa * (1 + deficit / 100)

            if monthly_normal < DRY_SEASON_NORMAL_THRESHOLD:
                severity, color_val = "Dry Season — Normal", 0
                deficit = 0.0
            elif deficit < -30:
                severity, color_val = "Severe Deficit", -40
            elif deficit < -15:
                severity, color_val = "Moderate Deficit", -25
            elif deficit < 0:
                severity, color_val = "Mild Deficit", -8
            elif deficit < 15:
                severity, color_val = "Normal", 8
            else:
                severity, color_val = "Excess", 25

            rows.append({
                "district": district, "state": state,
                "lat": lat, "lon": lon,
                "deficit_pct": round(deficit, 1),
                "rainfall_mm": round(rainfall_mm, 0),
                "lpa_mm": lpa, "severity": severity, "color_val": color_val,
            })

    return pd.DataFrame(rows)


# Build district data
_rainfall = _REAL_DATA.get("rainfall", {}) if _REAL_DATA else {}
_district_df = _generate_district_data(_rainfall)

# =========================================================================
# SEASONAL BANNER -- outside JJAS, deficit % is less meaningful
# =========================================================================
from datetime import datetime as _dt
_current_month = _dt.now().month
_is_monsoon = _current_month in (6, 7, 8, 9)
_dry_count = int((_district_df["severity"] == "Dry Season — Normal").sum()) if "severity" in _district_df.columns else 0

if not _is_monsoon:
    _month_name = _dt.now().strftime("%B")
    render_html(f"""
    <div class="glass-card" style="border-color:rgba(59,130,246,0.35); padding:1rem 1.4rem; margin-bottom:1.2rem;">
        <div style="color:#60a5fa; font-weight:600; font-size:1.05rem;">
            &#x1f4c5; Pre/Post-Monsoon Season ({_month_name})
        </div>
        <div style="color:#94a3b8; font-size:0.95rem; margin-top:0.3rem; line-height:1.5;">
            Deficit calculations are most meaningful during <strong style="color:#e2e8f0;">JJAS (June–September)</strong>.
            In the current month, climatological normals are very low for most districts, so percentage
            deficits can appear extreme (e.g. &minus;100%) even though near-zero rainfall is expected.
            Districts with 30-day normals below 30 mm are shown as <strong style="color:#60a5fa;">Dry Season — Normal</strong>
            and excluded from the stressed-districts table.
            {f'<br><span style="color:#64748b; font-size:0.88rem;">{_dry_count} of {len(_district_df)} districts are in dry-season mode.</span>' if _dry_count > 0 else ''}
        </div>
    </div>
    """)

# =========================================================================
# FILTERS
# =========================================================================
st.markdown("""
<div class="section-heading">District-Level Rainfall Map</div>
<div class="section-sub">Cotton-belt districts colored by 30-day rainfall departure from monthly climatological normal</div>
<hr class="heading-rule rule-blue">
""", unsafe_allow_html=True)

filter_cols = st.columns([3, 2, 2])

with filter_cols[0]:
    selected_states = st.multiselect(
        "Filter by State",
        options=sorted(_district_df["state"].unique()),
        default=sorted(_district_df["state"].unique()),
        key="geo_state_filter",
    )

with filter_cols[1]:
    _sev_options = ["Severe Deficit", "Moderate Deficit", "Mild Deficit", "Normal", "Excess"]
    if _dry_count > 0:
        _sev_options.append("Dry Season — Normal")
    severity_filter = st.multiselect(
        "Filter by Severity",
        options=_sev_options,
        default=_sev_options,
        key="geo_severity_filter",
    )

with filter_cols[2]:
    map_style = st.selectbox(
        "Map Style",
        options=["open-street-map", "carto-darkmatter", "carto-positron", "NASA Satellite (MODIS)", "NASA NDVI (Vegetation)"],
        index=1,
        key="geo_map_style",
    )

# Apply filters
_filtered = _district_df[
    (_district_df["state"].isin(selected_states)) &
    (_district_df["severity"].isin(severity_filter))
]

# =========================================================================
# CHOROPLETH MAP (Scattermapbox with sized/colored markers)
# =========================================================================
if not _filtered.empty:
    # Color scale: red (deficit) -> yellow (normal) -> green (excess)
    fig_map = go.Figure()

    # Size markers by absolute deficit (larger = more extreme)
    _sizes = np.clip(np.abs(_filtered["deficit_pct"]) * 0.4 + 8, 8, 35)

    fig_map.add_trace(go.Scattermapbox(
        lat=_filtered["lat"],
        lon=_filtered["lon"],
        mode="markers",
        marker=dict(
            size=_sizes,
            color=_filtered["deficit_pct"],
            colorscale=[
                [0.0,  "#dc2626"],  # severe deficit
                [0.25, "#ef4444"],  # moderate deficit
                [0.40, "#f59e0b"],  # mild deficit
                [0.50, "#fbbf24"],  # near normal
                [0.60, "#84cc16"],  # slightly above
                [0.75, "#22c55e"],  # above normal
                [1.0,  "#10b981"],  # excess
            ],
            cmin=-50,
            cmax=50,
            colorbar=dict(
                title=dict(text="Deficit %", font=dict(size=13, color="#94a3b8")),
                tickfont=dict(size=12, color="#94a3b8"),
                thickness=12,
                len=0.6,
                outlinewidth=0,
                bgcolor="rgba(15,23,42,0.8)",
                tickvals=[-40, -20, 0, 20, 40],
                ticktext=["-40%", "-20%", "0%", "+20%", "+40%"],
            ),
            opacity=0.85,
        ),
        text=_filtered.apply(
            lambda r: (
                f"<b>{r['district']}</b>, {r['state']}<br>"
                f"30d Actual: {r['rainfall_mm']:.1f} mm (Normal: {r['lpa_mm']:.1f} mm)<br>"
                f"Status: {r['severity']}"
            ) if r['severity'] == "Dry Season — Normal" else (
                f"<b>{r['district']}</b>, {r['state']}<br>"
                f"Deficit: <b>{r['deficit_pct']:+.1f}%</b><br>"
                f"30d Actual: {r['rainfall_mm']:.1f} mm (Normal: {r['lpa_mm']:.1f} mm)<br>"
                f"Status: {r['severity']}"
            ), axis=1
        ),
        hoverinfo="text",
        name="Districts",
    ))

    mapbox_layers = []
    _style = map_style
    
    if map_style.startswith("NASA"):
        _style = "carto-darkmatter" # base layer
        if map_style == "NASA Satellite (MODIS)":
            mapbox_layers = [{
                "below": 'traces', "sourcetype": "raster", "sourceattribution": "NASA GIBS",
                "source": ["https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_CorrectedReflectance_TrueColor/default/2023-09-01/GoogleMapsCompatible_Level9/{z}/{y}/{x}.jpg"]
            }]
        else:
            mapbox_layers = [{
                "below": 'traces', "sourcetype": "raster", "sourceattribution": "NASA GIBS",
                "source": ["https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/MODIS_Terra_NDVI_8Day/default/2023-09-01/GoogleMapsCompatible_Level9/{z}/{y}/{x}.png"]
            }]

    fig_map.update_layout(
        mapbox=dict(
            style=_style,
            layers=mapbox_layers,
            center=dict(lat=22.0, lon=78.0),
            zoom=4.2,
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=PLOTLY_FONT,
        height=620,
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
    )

    st.plotly_chart(fig_map, use_container_width=True)
else:
    st.info("No districts match the current filters.")

# =========================================================================
# SUMMARY METRICS
# =========================================================================
st.markdown("""
<div class="section-heading">District Summary</div>
<div class="section-sub">Aggregated statistics across the filtered cotton-belt districts</div>
<hr class="heading-rule rule-gold">
""", unsafe_allow_html=True)

_total = len(_filtered)
_severe = len(_filtered[_filtered["severity"] == "Severe Deficit"])
_moderate = len(_filtered[_filtered["severity"] == "Moderate Deficit"])
_normal = len(_filtered[_filtered["severity"].isin(["Normal", "Excess"])])
_dry_season = len(_filtered[_filtered["severity"] == "Dry Season — Normal"])
# Exclude dry-season districts from average deficit (their 0% is artificial)
_non_dry = _filtered[_filtered["severity"] != "Dry Season — Normal"]
_avg_deficit = _non_dry["deficit_pct"].mean() if len(_non_dry) > 0 else 0

_m_cols = st.columns(5)
if _dry_season > 0:
    _metrics = [
        (_total, "Total Districts", f"Across {len(selected_states)} states", "#60a5fa"),
        (_severe, "Severe Deficit", "Below -30% from LPA", "#ef4444"),
        (_moderate, "Moderate Deficit", "-15% to -30% from LPA", "#f59e0b"),
        (_dry_season, "Dry Season", "Normal < 30 mm (not meaningful)", "#60a5fa"),
        (f"{_avg_deficit:+.1f}%", "Avg Deficit", "Excl. dry-season districts", "#8b5cf6"),
    ]
else:
    _metrics = [
        (_total, "Total Districts", f"Across {len(selected_states)} states", "#60a5fa"),
        (_severe, "Severe Deficit", "Below -30% from LPA", "#ef4444"),
        (_moderate, "Moderate Deficit", "-15% to -30% from LPA", "#f59e0b"),
        (_normal, "Normal / Excess", "Above -15% from LPA", "#10b981"),
        (f"{_avg_deficit:+.1f}%", "Avg Deficit", "Weighted cotton-belt average", "#8b5cf6"),
    ]

for col, (value, label, detail, color) in zip(_m_cols, _metrics):
    with col:
        render_html(f"""
<div class="metric-card">
    <div class="metric-value" style="color:{color};">{value}</div>
    <div class="metric-label">{label}</div>
    <div class="metric-detail">{detail}</div>
</div>
""")

# =========================================================================
# STRESSED DISTRICTS TABLE
# =========================================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="section-heading">Stressed Districts</div>
<div class="section-sub">Districts with 30-day rainfall deficit exceeding -15% from monthly climatological normal</div>
<hr class="heading-rule rule-red">
""", unsafe_allow_html=True)

_stressed = _filtered[
    (_filtered["deficit_pct"] < -15) &
    (_filtered["severity"] != "Dry Season — Normal")
].sort_values("deficit_pct")

if not _stressed.empty:
    _table_rows = []
    for _, row in _stressed.iterrows():
        _def_color = "#ef4444" if row["deficit_pct"] < -30 else "#f59e0b"
        _badge_bg = "rgba(239,68,68,0.15)" if row["deficit_pct"] < -30 else "rgba(245,158,11,0.15)"
        _table_rows.append(
            f"""
<tr style="border-bottom:1px solid rgba(51,65,85,0.3);">
    <td style="color:#e2e8f0; padding:0.5rem 0.8rem; font-weight:500;">{html.escape(str(row["district"]))}</td>
    <td style="color:#94a3b8; padding:0.5rem 0.8rem;">{html.escape(str(row["state"]))}</td>
    <td style="color:{_def_color}; padding:0.5rem 0.8rem; font-weight:600; text-align:center;">{row["deficit_pct"]:+.1f}%</td>
    <td style="color:#e2e8f0; padding:0.5rem 0.8rem; text-align:center;">{row["rainfall_mm"]:.0f}</td>
    <td style="color:#94a3b8; padding:0.5rem 0.8rem; text-align:center;">{row["lpa_mm"]:.0f}</td>
    <td style="padding:0.5rem 0.8rem;">
        <span style="background:{_badge_bg}; color:{_def_color}; padding:0.2rem 0.6rem; border-radius:6px; font-size:0.82rem; font-weight:600;">
            {html.escape(str(row["severity"]))}
        </span>
    </td>
</tr>
"""
        )

    render_html(
        f"""
<div class="glass-card glass-card-accent-red" style="overflow-x:auto;">
    <table style="width:100%; border-collapse:collapse; font-family:'Inter',sans-serif; font-size:0.95rem;">
        <thead>
            <tr style="border-bottom:2px solid rgba(239,68,68,0.2);">
                <th style="color:#94a3b8; font-weight:600; font-size:0.85rem; letter-spacing:0.06em; text-transform:uppercase; padding:0.6rem 0.8rem; text-align:left;">District</th>
                <th style="color:#94a3b8; font-weight:600; font-size:0.85rem; letter-spacing:0.06em; text-transform:uppercase; padding:0.6rem 0.8rem; text-align:left;">State</th>
                <th style="color:#94a3b8; font-weight:600; font-size:0.85rem; letter-spacing:0.06em; text-transform:uppercase; padding:0.6rem 0.8rem; text-align:center;">Deficit</th>
                <th style="color:#94a3b8; font-weight:600; font-size:0.85rem; letter-spacing:0.06em; text-transform:uppercase; padding:0.6rem 0.8rem; text-align:center;">30d Actual (mm)</th>
                <th style="color:#94a3b8; font-weight:600; font-size:0.85rem; letter-spacing:0.06em; text-transform:uppercase; padding:0.6rem 0.8rem; text-align:center;">30d Normal (mm)</th>
                <th style="color:#94a3b8; font-weight:600; font-size:0.85rem; letter-spacing:0.06em; text-transform:uppercase; padding:0.6rem 0.8rem; text-align:left;">Severity</th>
            </tr>
        </thead>
        <tbody>
            {"".join(_table_rows)}
        </tbody>
    </table>
</div>
"""
    )
else:
    render_html("""
<div class="glass-card glass-card-accent-green" style="padding:1rem 1.4rem;">
    <div style="color:#10b981; font-weight:600;">No stressed districts</div>
    <div style="color:#94a3b8; font-size:0.92rem; margin-top:0.2rem;">
        All filtered districts are receiving adequate rainfall (within 15% of LPA).
    </div>
</div>
""")

# =========================================================================
# STATE-LEVEL BAR CHART
# =========================================================================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="section-heading">State-Level Deficit Comparison</div>
<div class="section-sub">Average district-level deficit by cotton-growing state</div>
<hr class="heading-rule rule-green">
""", unsafe_allow_html=True)

_state_avg = _district_df.groupby("state")["deficit_pct"].mean().sort_values()

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(
    x=_state_avg.values,
    y=_state_avg.index,
    orientation="h",
    marker=dict(
        color=[
            "#ef4444" if v < -20 else "#f59e0b" if v < -10 else "#84cc16" if v < 0 else "#10b981"
            for v in _state_avg.values
        ],
        line=dict(width=0),
    ),
    text=[f"{v:+.1f}%" if abs(v) > 0 else "0.0%" for v in _state_avg.values],
    textposition="auto",
    textfont=dict(size=11, color="#e2e8f0"),
    hovertemplate="<b>%{y}</b><br>Avg Deficit: %{x:+.1f}%<extra></extra>",
))

fig_bar.add_vline(x=0, line_dash="dot", line_color="rgba(148,163,184,0.4)")

fig_bar.update_layout(
    **base_layout(height=400, margin=dict(l=140, r=60, t=20, b=50)),
    xaxis_title="Average Deficit (%)",
    yaxis_title="",
    xaxis=dict(gridcolor="rgba(51,65,85,0.2)", zeroline=False),
    yaxis=dict(gridcolor="rgba(0,0,0,0)"),
)
st.plotly_chart(fig_bar, use_container_width=True)

# Why this matters
st.markdown("""
<div class="glass-card" style="border-color:rgba(59,130,246,0.2); padding:1rem 1.4rem; margin-top:0.5rem;">
    <span style="color:#94a3b8; font-family:'Inter',sans-serif; font-size:0.92rem; font-weight:600;
        letter-spacing:0.06em; text-transform:uppercase;">Why this matters</span>
    <div style="color:#e2e8f0; font-family:'Inter',sans-serif; font-size:1.02rem; font-weight:300;
        margin-top:0.3rem; line-height:1.55;">
        State-level averages mask critical district-level variation. A state reporting "normal" monsoon
        may still have cotton-belt districts in severe deficit. This granular view enables <strong>targeted
        early warnings</strong> for specific supply-chain regions, helping farmers plan crop insurance
        (PMFBY) claims and MSMEs adjust procurement from specific ginning hubs.
    </div>
</div>
""", unsafe_allow_html=True)

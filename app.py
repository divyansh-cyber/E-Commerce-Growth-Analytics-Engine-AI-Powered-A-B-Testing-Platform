"""
app.py
------
Main Streamlit dashboard for the E-Commerce Growth Analytics platform.

Tabs
────
  1. Funnel Analysis      — PostgreSQL-powered drop-off visualisation
  2. Cohort Retention     — Weekly cohort heatmap
  3. A/B Test Results     — Statistical experiment engine
  4. AI Executive Readout — Groq/Llama-generated summary

Run:
    streamlit run app.py
"""

import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from dotenv import load_dotenv

warnings.filterwarnings("ignore")
load_dotenv()

# Ensure src/ is importable when launched from project root
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.ab_testing import ABTestEngine
from src.gemini_insights import generate_executive_readout

# ── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Growth Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS (dark premium theme) ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: linear-gradient(145deg, #060b18 0%, #0d1526 40%, #070d1f 100%);
    color: #e2e8f0;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none !important; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg,
        rgba(56,189,248,0.12) 0%,
        rgba(139,92,246,0.12) 50%,
        rgba(16,185,129,0.08) 100%);
    border: 1px solid rgba(56,189,248,0.2);
    border-radius: 24px;
    padding: 2.5rem 2.5rem 2rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse at 20% 50%, rgba(56,189,248,0.06) 0%, transparent 60%),
                radial-gradient(ellipse at 80% 50%, rgba(139,92,246,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.1rem; font-weight: 900; margin: 0;
    background: linear-gradient(120deg, #38bdf8, #a78bfa, #34d399);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.2;
}
.hero-sub {
    color: #94a3b8; font-size: .95rem; margin-top: .5rem; font-weight: 400;
}
.hero-badges { display: flex; gap: .5rem; margin-top: 1rem; flex-wrap: wrap; }
.hero-badge {
    background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.1);
    border-radius: 20px; padding: .25rem .8rem; font-size: .75rem;
    color: #94a3b8; font-weight: 500;
}

/* ── Metric Cards ── */
.kpi-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 1rem; margin-bottom:1.5rem; }
.kpi-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px; padding: 1.25rem 1.5rem;
    transition: all .25s ease; position: relative; overflow: hidden;
}
.kpi-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, #38bdf8, #a78bfa);
}
.kpi-card:hover {
    border-color: rgba(56,189,248,0.3);
    background: rgba(56,189,248,0.04);
    transform: translateY(-3px);
    box-shadow: 0 12px 40px rgba(56,189,248,0.1);
}
.kpi-val {
    font-size: 1.9rem; font-weight: 800;
    background: linear-gradient(135deg, #38bdf8, #a78bfa);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
}
.kpi-label { color: #64748b; font-size: .78rem; font-weight: 600;
             text-transform: uppercase; letter-spacing: .06em; margin-top:.2rem; }
.kpi-delta-pos { color: #34d399; font-size: .8rem; font-weight: 600; margin-top:.3rem; }
.kpi-delta-neg { color: #f87171; font-size: .8rem; font-weight: 600; margin-top:.3rem; }
.kpi-delta-neu { color: #94a3b8; font-size: .8rem; font-weight: 600; margin-top:.3rem; }

/* ── Stat Result Cards ── */
.stat-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 1.25rem; margin-bottom: .75rem;
}
.stat-card-sig {
    background: rgba(16,185,129,0.05);
    border-color: rgba(16,185,129,0.25);
}
.stat-card-insig {
    background: rgba(239,68,68,0.05);
    border-color: rgba(239,68,68,0.2);
}

/* ── Badges ── */
.badge { border-radius:20px; padding:.3rem .9rem; font-size:.78rem;
         font-weight:700; display:inline-block; }
.badge-green { background:rgba(16,185,129,.15); border:1px solid rgba(16,185,129,.4); color:#34d399; }
.badge-red   { background:rgba(239,68,68,.15);  border:1px solid rgba(239,68,68,.4);  color:#f87171; }
.badge-yellow{ background:rgba(245,158,11,.15); border:1px solid rgba(245,158,11,.4); color:#fbbf24; }
.badge-blue  { background:rgba(56,189,248,.15); border:1px solid rgba(56,189,248,.4); color:#38bdf8; }
.badge-purple{ background:rgba(139,92,246,.15); border:1px solid rgba(139,92,246,.4); color:#a78bfa; }

/* ── Section labels ── */
.section-label {
    font-size: .7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: .1em; color: #38bdf8; margin-bottom: .4rem;
}
.divider { height:1px; background:linear-gradient(90deg,transparent,rgba(56,189,248,.3),rgba(139,92,246,.3),transparent); margin:1.5rem 0; }

/* ── AI readout ── */
.ai-box {
    background: linear-gradient(135deg,rgba(139,92,246,.07) 0%,rgba(56,189,248,.07) 100%);
    border: 1px solid rgba(139,92,246,.25);
    border-radius: 18px; padding: 2rem; line-height: 1.8;
    color: #e2e8f0; font-size: .95rem;
}
.ai-box h2 { color: #a78bfa; font-size: 1.1rem; font-weight: 700; margin-top:1.5rem; }
.ai-box h3 { color: #38bdf8; font-size: 1rem;   font-weight: 600; margin-top:1.2rem; }
.ai-box strong { color: #f1f5f9; }
.ai-box code   { background: rgba(255,255,255,.07); border-radius:4px; padding:.1rem .4rem; font-family:'JetBrains Mono'; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,.025);
    border-radius: 12px; padding: 4px;
    border: 1px solid rgba(255,255,255,.06); gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px; color: #64748b; font-weight: 500;
    padding: .55rem 1.3rem; background: transparent;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,rgba(56,189,248,.2),rgba(139,92,246,.2)) !important;
    color: #38bdf8 !important; border-bottom: none !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0a0f1e 0%,#0d1526 100%);
    border-right: 1px solid rgba(255,255,255,.05);
}
[data-testid="stSidebar"] .stTextInput > div > div > input {
    background: rgba(255,255,255,.04) !important;
    border: 1px solid rgba(255,255,255,.1) !important;
    color: #e2e8f0 !important; border-radius: 8px !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg,#3b82f6,#8b5cf6);
    color: white; border: none; border-radius: 10px;
    font-weight: 600; transition: all .3s ease;
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(59,130,246,.4);
}

/* ── SQL viewer ── */
.sql-block {
    background: #0d1117;
    border: 1px solid rgba(56,189,248,.2);
    border-radius: 12px; padding: 1.2rem 1.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: .8rem; color: #e2e8f0; overflow-x: auto;
    line-height: 1.65; white-space: pre;
}
</style>
""", unsafe_allow_html=True)

# ── Colour palette for Plotly ────────────────────────────────────────────────
COLORS = {
    "blue"  : "#38bdf8",
    "purple": "#a78bfa",
    "green" : "#34d399",
    "orange": "#fb923c",
    "red"   : "#f87171",
    "yellow": "#fbbf24",
    "slate" : "#94a3b8",
}
GRAD   = [COLORS["blue"], COLORS["purple"], COLORS["green"], COLORS["orange"]]
CTRL_C = COLORS["blue"]
TRT_C  = COLORS["purple"]


def plotly_dark(fig: go.Figure, height: int = 420) -> go.Figure:
    """Apply the project's dark theme to any Plotly figure."""
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor ="rgba(0,0,0,0)",
        font=dict(family="Inter", color=COLORS["slate"]),
        title_font=dict(family="Inter", color="#f1f5f9", size=15),
        height=height,
        margin=dict(l=40, r=20, t=50, b=40),
        legend=dict(
            bgcolor="rgba(255,255,255,0.03)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
        ),
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.05)",
        linecolor="rgba(255,255,255,0.08)",
        tickcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.08)",
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.05)",
        linecolor="rgba(255,255,255,0.08)",
        tickcolor="rgba(255,255,255,0.08)",
        zerolinecolor="rgba(255,255,255,0.08)",
    )
    return fig


# ── Demo-mode data loaders (pandas — no DB needed) ───────────────────────────

@st.cache_data(show_spinner=False)
def load_demo_events() -> pd.DataFrame:
    from src.data_generator import generate_events
    return generate_events()


@st.cache_data(show_spinner=False)
def load_demo_ab() -> pd.DataFrame:
    from src.data_generator import generate_ab_test
    return generate_ab_test()


def compute_funnel_pandas(events: pd.DataFrame) -> pd.DataFrame:
    """Replicate the funnel SQL query using pandas (demo mode)."""
    stages   = ["page_view", "add_to_cart", "checkout", "purchase"]
    counts   = {s: events[events["event_type"] == s]["user_id"].nunique() for s in stages}
    top      = counts["page_view"]
    rows, prev = [], None
    for i, s in enumerate(stages):
        n    = counts[s]
        row  = {
            "stage_order"     : i + 1,
            "stage_name"      : s,
            "unique_users"    : n,
            "overall_conv_pct": round(100 * n / top, 2),
            "step_conv_pct"   : round(100 * n / prev, 2) if prev else 100.0,
            "step_dropoff_pct": round(100 * (prev - n) / prev, 2) if prev else 0.0,
        }
        rows.append(row)
        prev = n
    return pd.DataFrame(rows)


def compute_cohort_pandas(events: pd.DataFrame) -> pd.DataFrame:
    """Replicate the cohort retention SQL query using pandas (demo mode)."""
    ev = events.copy()
    ev["event_timestamp"] = pd.to_datetime(ev["event_timestamp"])
    ev["week"] = ev["event_timestamp"].dt.to_period("W").dt.start_time

    # cohort = first week seen
    cohort_map = ev.groupby("user_id")["week"].min().rename("cohort_week")
    ev = ev.merge(cohort_map, on="user_id")

    purchases = ev[ev["event_type"] == "purchase"][["user_id", "week", "cohort_week"]].drop_duplicates()
    purchases["week_number"] = (
        (purchases["week"] - purchases["cohort_week"]).dt.days // 7
    ).astype(int)

    cohort_sizes = cohort_map.reset_index().groupby("cohort_week")["user_id"].nunique()

    retention = (
        purchases[purchases["week_number"].between(0, 8)]
        .groupby(["cohort_week", "week_number"])["user_id"]
        .nunique()
        .reset_index(name="active_users")
    )
    retention = retention.merge(
        cohort_sizes.rename("cohort_size"), on="cohort_week"
    )
    retention["retention_pct"] = (
        100 * retention["active_users"] / retention["cohort_size"]
    ).round(1)
    retention["cohort_week"] = retention["cohort_week"].astype(str).str[:10]
    return retention


def compute_device_pandas(events: pd.DataFrame) -> pd.DataFrame:
    def count_by_device(etype):
        return (events[events["event_type"] == etype]
                .groupby("device_type")["user_id"].nunique())

    visitors = count_by_device("page_view")
    buyers   = count_by_device("purchase")
    revenue  = (events[events["event_type"] == "purchase"]
                .groupby("device_type")["revenue"].sum().round(2))
    df = pd.DataFrame({"visitors": visitors, "buyers": buyers, "revenue": revenue}).reset_index()
    df["overall_conv_pct"] = (100 * df["buyers"] / df["visitors"]).round(2)
    return df.sort_values("overall_conv_pct", ascending=False)


def compute_category_pandas(events: pd.DataFrame) -> pd.DataFrame:
    df = (events[events["event_type"] == "purchase"]
          .groupby("category")
          .agg(buyers=("user_id", "nunique"),
               purchases=("event_id", "count"),
               total_revenue=("revenue", "sum"),
               avg_order_value=("revenue", "mean"))
          .reset_index())
    df["total_revenue"]   = df["total_revenue"].round(2)
    df["avg_order_value"] = df["avg_order_value"].round(2)
    return df.sort_values("total_revenue", ascending=False)


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 .5rem">
        <div style="font-size:2rem">📈</div>
        <div style="font-weight:700;font-size:1.05rem;color:#f1f5f9">Growth Analytics</div>
        <div style="font-size:.75rem;color:#64748b">E-Commerce · A/B Testing · AI</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()

    # ── Database mode ───────────────────────────────────────────────────────
    st.markdown('<div class="section-label">⚙️ Data Source</div>', unsafe_allow_html=True)
    use_pg = st.toggle("Connect to PostgreSQL", value=False, key="use_pg")

    pg_engine = None
    if use_pg:
        db_url = st.text_input(
            "DATABASE_URL",
            value=os.getenv("DATABASE_URL", "postgresql://analyst:analyst123@localhost:5433/ecommerce_analytics"),
            type="password",
            label_visibility="collapsed",
        )
        if st.button("🔌 Connect", key="connect_btn"):
            with st.spinner("Connecting …"):
                try:
                    from src.sql_engine import PostgreSQLEngine
                    eng = PostgreSQLEngine(db_url)
                    if eng.ping():
                        st.session_state["pg_engine"] = eng
                        st.success("Connected ✅")
                    else:
                        st.error("Could not reach database")
                except Exception as e:
                    st.error(f"Error: {e}")

        if "pg_engine" in st.session_state:
            st.markdown('<span class="badge badge-green">🟢 PostgreSQL live</span>',
                        unsafe_allow_html=True)
            pg_engine = st.session_state["pg_engine"]
        else:
            st.markdown('<span class="badge badge-yellow">⚡ Demo mode (pandas)</span>',
                        unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-blue">⚡ Demo mode (pandas)</span>',
                    unsafe_allow_html=True)

    st.divider()

    # ── Groq API key ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-label">🤖 Groq AI (Free)</div>', unsafe_allow_html=True)
    gemini_key = st.text_input(
        "API Key",
        value=os.getenv("GROQ_API_KEY", ""),
        type="password",
        help="Free key at console.groq.com — no credit card needed",
        label_visibility="collapsed",
        placeholder="Paste Groq API key …",
    )
    if gemini_key:
        os.environ["GROQ_API_KEY"] = gemini_key
        st.markdown('<span class="badge badge-green">🤖 Groq ready</span>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-yellow">🔑 Key not set</span>',
                    unsafe_allow_html=True)
    st.caption("Free key → console.groq.com")

    st.divider()

    # ── A/B test CSV upload ─────────────────────────────────────────────────
    st.markdown('<div class="section-label">📂 Custom A/B CSV</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload CSV",
        type="csv",
        help="Columns needed: user_id, group_name, converted, order_value",
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("""
    <div style="font-size:.72rem;color:#475569;text-align:center;line-height:1.8">
        Tech Stack<br>
        <span style="color:#64748b">PostgreSQL · Python · Streamlit<br>
        SciPy · Statsmodels · Groq AI · Plotly</span>
    </div>
    """, unsafe_allow_html=True)


# ── Hero banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">📈 E-Commerce Growth Analytics</div>
    <div class="hero-sub">
        End-to-end pipeline · Conversion Funnel · Cohort Retention ·
        A/B Experimentation · AI Executive Summaries
    </div>
    <div class="hero-badges">
        <span class="hero-badge">🐘 PostgreSQL</span>
        <span class="hero-badge">📊 Plotly</span>
        <span class="hero-badge">🧪 SciPy + Statsmodels</span>
        <span class="hero-badge">🤖 Groq Llama 3.3 70B</span>
        <span class="hero-badge">⚡ Streamlit</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Load data (DB or demo) ────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def get_events_data(_eng):
    if _eng:
        return None   # we'll query directly each time
    return load_demo_events()

with st.spinner("Loading data …"):
    demo_events = load_demo_events() if not pg_engine else None
    if uploaded:
        ab_df = pd.read_csv(uploaded)
    elif pg_engine:
        ab_df = pg_engine.get_ab_test_data()
    else:
        ab_df = load_demo_ab()


# ── Quick-stats banner ───────────────────────────────────────────────────────

if demo_events is not None:
    _ev = demo_events
else:
    try:
        _ev = pg_engine.query("SELECT user_id, event_type, revenue FROM events")
    except Exception:
        _ev = load_demo_events()

total_users   = _ev["user_id"].nunique()
total_sessions= _ev["session_id"].nunique() if "session_id" in _ev.columns else _ev["user_id"].nunique()
total_orders  = (_ev["event_type"] == "purchase").sum()
total_revenue = _ev["revenue"].sum()

st.markdown(f"""
<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-val">{total_users:,}</div>
    <div class="kpi-label">Unique Users</div>
    <div class="kpi-delta-pos">↑ 90-day window</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-val">{total_sessions:,}</div>
    <div class="kpi-label">Total Sessions</div>
    <div class="kpi-delta-pos">↑ All devices</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-val">{total_orders:,}</div>
    <div class="kpi-label">Purchases</div>
    <div class="kpi-delta-pos">↑ Completed orders</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-val">${total_revenue:,.0f}</div>
    <div class="kpi-label">Total Revenue</div>
    <div class="kpi-delta-pos">↑ Gross revenue</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Main Tabs ────────────────────────────────────────────────────────────────

tab_funnel, tab_cohort, tab_ab, tab_ai = st.tabs([
    "🔽 Funnel Analysis",
    "📅 Cohort Retention",
    "🧪 A/B Test Results",
    "🤖 AI Executive Readout",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Funnel Analysis
# ════════════════════════════════════════════════════════════════════════════
with tab_funnel:
    st.markdown("### 🔽 Conversion Funnel Analysis")
    st.markdown(
        "Each bar shows **unique users** who reached that stage. "
        "Drop-off rates are calculated with **`LAG()`** window functions in PostgreSQL."
    )

    with st.spinner("Running funnel query …"):
        if pg_engine:
            funnel_df = pg_engine.get_funnel_metrics()
            device_df = pg_engine.get_device_breakdown()
            cat_df    = pg_engine.get_category_revenue()
        else:
            funnel_df = compute_funnel_pandas(demo_events)
            device_df = compute_device_pandas(demo_events)
            cat_df    = compute_category_pandas(demo_events)

    # ── Funnel chart ────────────────────────────────────────────────────────
    stage_labels = ["Page View", "Add to Cart", "Checkout", "Purchase"]
    stage_emoji  = ["👁️ Page View", "🛒 Add to Cart", "💳 Checkout", "✅ Purchase"]

    col_funnel, col_bar = st.columns([1, 1])

    with col_funnel:
        st.markdown('<div class="section-label">Funnel Visualisation</div>', unsafe_allow_html=True)
        fig_funnel = go.Figure(go.Funnel(
            y         = stage_emoji,
            x         = funnel_df["unique_users"].tolist(),
            textinfo  = "value+percent initial",
            textfont  = dict(family="Inter", size=13, color="#f1f5f9"),
            marker    = dict(
                color=GRAD,
                line=dict(width=1, color="rgba(255,255,255,0.1)")
            ),
            connector = dict(line=dict(color="rgba(255,255,255,0.05)", width=1)),
        ))
        fig_funnel = plotly_dark(fig_funnel, height=380)
        fig_funnel.update_layout(title="Funnel — Unique Users per Stage")
        st.plotly_chart(fig_funnel, use_container_width=True)

    with col_bar:
        st.markdown('<div class="section-label">Step Drop-off Rate (%)</div>', unsafe_allow_html=True)
        dropoffs = funnel_df["step_dropoff_pct"].tolist()
        bar_colors = [
            COLORS["green"] if d < 30 else COLORS["yellow"] if d < 60 else COLORS["red"]
            for d in dropoffs
        ]
        fig_drop = go.Figure(go.Bar(
            x          = stage_emoji,
            y          = dropoffs,
            marker_color=bar_colors,
            text       = [f"{d:.1f}%" for d in dropoffs],
            textposition="outside",
            textfont   = dict(color="#f1f5f9", size=13, family="Inter"),
        ))
        fig_drop = plotly_dark(fig_drop, height=380)
        fig_drop.update_layout(
            title="Drop-off at Each Funnel Stage",
            yaxis_title="Drop-off %",
        )
        st.plotly_chart(fig_drop, use_container_width=True)

    # ── Metrics table ───────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 📋 Detailed Funnel Metrics")
    display_funnel = funnel_df.copy()
    display_funnel.columns = [c.replace("_", " ").title() for c in display_funnel.columns]
    st.dataframe(
        display_funnel.style.background_gradient(
            subset=[c for c in display_funnel.columns if "%" in c],
            cmap="RdYlGn"
        ),
        use_container_width=True, hide_index=True
    )

    # ── Device + Category breakdown ─────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col_dev, col_cat = st.columns(2)

    with col_dev:
        st.markdown("#### 📱 Conversion by Device")
        fig_dev = go.Figure(go.Bar(
            x=device_df["device_type"],
            y=device_df["overall_conv_pct"],
            marker_color=[COLORS["blue"], COLORS["purple"], COLORS["green"]],
            text=[f"{v:.1f}%" for v in device_df["overall_conv_pct"]],
            textposition="outside",
            textfont=dict(color="#f1f5f9", family="Inter"),
        ))
        plotly_dark(fig_dev, 320)
        fig_dev.update_layout(yaxis_title="Conversion Rate (%)", showlegend=False)
        st.plotly_chart(fig_dev, use_container_width=True)

    with col_cat:
        st.markdown("#### 🏷️ Revenue by Category")
        fig_cat = go.Figure(go.Bar(
            x=cat_df["category"],
            y=cat_df["total_revenue"],
            marker=dict(
                color=cat_df["total_revenue"],
                colorscale=[[0, "#1e3a5f"], [0.5, "#38bdf8"], [1, "#a78bfa"]],
                showscale=False,
            ),
            text=[f"${v:,.0f}" for v in cat_df["total_revenue"]],
            textposition="outside",
            textfont=dict(color="#f1f5f9", family="Inter"),
        ))
        plotly_dark(fig_cat, 320)
        fig_cat.update_layout(yaxis_title="Revenue ($)", showlegend=False)
        st.plotly_chart(fig_cat, use_container_width=True)

    # ── SQL snippet ─────────────────────────────────────────────────────────
    with st.expander("🐘 View PostgreSQL Query (Funnel)"):
        st.markdown("""
<div class="sql-block">
<span style="color:#a78bfa">WITH</span> funnel_stages <span style="color:#a78bfa">AS</span> (
    <span style="color:#38bdf8">SELECT</span>
        event_type,
        <span style="color:#34d399">COUNT</span>(<span style="color:#fb923c">DISTINCT</span> user_id) <span style="color:#a78bfa">AS</span> unique_users,
        <span style="color:#38bdf8">CASE</span> event_type
            <span style="color:#38bdf8">WHEN</span> <span style="color:#fbbf24">'page_view'</span>   <span style="color:#38bdf8">THEN</span> 1
            <span style="color:#38bdf8">WHEN</span> <span style="color:#fbbf24">'add_to_cart'</span> <span style="color:#38bdf8">THEN</span> 2
            <span style="color:#38bdf8">WHEN</span> <span style="color:#fbbf24">'checkout'</span>    <span style="color:#38bdf8">THEN</span> 3
            <span style="color:#38bdf8">WHEN</span> <span style="color:#fbbf24">'purchase'</span>    <span style="color:#38bdf8">THEN</span> 4
        <span style="color:#38bdf8">END AS</span> stage_order
    <span style="color:#a78bfa">FROM</span> events
    <span style="color:#a78bfa">WHERE</span> event_type <span style="color:#a78bfa">IN</span> (<span style="color:#fbbf24">'page_view','add_to_cart','checkout','purchase'</span>)
    <span style="color:#a78bfa">GROUP BY</span> event_type
),
funnel_with_context <span style="color:#a78bfa">AS</span> (
    <span style="color:#38bdf8">SELECT</span>
        stage_order, event_type <span style="color:#a78bfa">AS</span> stage_name, unique_users,
        <span style="color:#34d399">LAG</span>(unique_users) <span style="color:#a78bfa">OVER</span> (<span style="color:#a78bfa">ORDER BY</span> stage_order)  <span style="color:#a78bfa">AS</span> prev_stage_users,
        <span style="color:#34d399">FIRST_VALUE</span>(unique_users) <span style="color:#a78bfa">OVER</span> (<span style="color:#a78bfa">ORDER BY</span> stage_order
            <span style="color:#a78bfa">ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING</span>) <span style="color:#a78bfa">AS</span> top_of_funnel
    <span style="color:#a78bfa">FROM</span> funnel_stages
)
<span style="color:#38bdf8">SELECT</span> stage_order, stage_name, unique_users,
    <span style="color:#34d399">ROUND</span>(100.0 * unique_users / top_of_funnel, 2)                              <span style="color:#a78bfa">AS</span> overall_conv_pct,
    <span style="color:#34d399">COALESCE</span>(<span style="color:#34d399">ROUND</span>(100.0 * unique_users / <span style="color:#34d399">NULLIF</span>(prev_stage_users,0), 2), 100.00) <span style="color:#a78bfa">AS</span> step_conv_pct,
    <span style="color:#34d399">COALESCE</span>(<span style="color:#34d399">ROUND</span>(100.0 * (prev_stage_users - unique_users)
             / <span style="color:#34d399">NULLIF</span>(prev_stage_users,0), 2), 0.00)                             <span style="color:#a78bfa">AS</span> step_dropoff_pct
<span style="color:#a78bfa">FROM</span> funnel_with_context <span style="color:#a78bfa">ORDER BY</span> stage_order;
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Cohort Retention
# ════════════════════════════════════════════════════════════════════════════
with tab_cohort:
    st.markdown("### 📅 Weekly Cohort Retention Analysis")
    st.markdown(
        "Users are grouped by their **acquisition week** (first event). "
        "Retention = % of cohort that made a **purchase** in each subsequent week."
    )

    with st.spinner("Running cohort query …"):
        cohort_long = (
            pg_engine.get_cohort_retention() if pg_engine
            else compute_cohort_pandas(demo_events)
        )

    # Pivot to matrix format
    pivot = cohort_long.pivot(
        index="cohort_week", columns="week_number", values="retention_pct"
    ).fillna(0)
    # Keep last 10 cohorts to avoid a cluttered heatmap
    pivot = pivot.tail(10)
    pivot.columns = [f"Wk {c}" for c in pivot.columns]

    col_heat, col_line = st.columns([3, 2])

    with col_heat:
        st.markdown('<div class="section-label">Retention Heatmap (%)</div>', unsafe_allow_html=True)
        fig_heat = go.Figure(go.Heatmap(
            z          = pivot.values,
            x          = pivot.columns.tolist(),
            y          = pivot.index.tolist(),
            colorscale = [
                [0.0,  "#0d1526"],
                [0.15, "#0c2a4a"],
                [0.35, "#0e4272"],
                [0.60, "#1a69a4"],
                [0.80, "#38bdf8"],
                [1.0,  "#a78bfa"],
            ],
            text            = pivot.values.round(1),
            texttemplate    = "%{text}%",
            textfont        = dict(size=11, family="Inter", color="white"),
            hoverongaps     = False,
            colorbar        = dict(
                tickfont=dict(color=COLORS["slate"]),
                outlinecolor="rgba(255,255,255,0.1)",
            ),
        ))
        plotly_dark(fig_heat, height=400)
        fig_heat.update_layout(
            title="Cohort Retention Heatmap",
            xaxis_title="Weeks Since Acquisition",
            yaxis_title="Cohort (Acquisition Week)",
        )
        st.plotly_chart(fig_heat, use_container_width=True)

    with col_line:
        st.markdown('<div class="section-label">Retention Curves</div>', unsafe_allow_html=True)
        fig_ret = go.Figure()
        cohorts_to_plot = pivot.index.tolist()[-5:]   # last 5 cohorts
        palette = [COLORS["blue"], COLORS["purple"], COLORS["green"],
                   COLORS["orange"], COLORS["yellow"]]

        for i, cw in enumerate(cohorts_to_plot):
            row_vals = pivot.loc[cw].tolist()
            week_nums = list(range(len(row_vals)))
            fig_ret.add_trace(go.Scatter(
                x    = week_nums,
                y    = row_vals,
                mode = "lines+markers",
                name = str(cw)[:10],
                line = dict(color=palette[i % len(palette)], width=2.5),
                marker= dict(size=6),
            ))

        plotly_dark(fig_ret, height=400)
        fig_ret.update_layout(
            title="Retention Curves (Last 5 Cohorts)",
            xaxis_title="Week Number",
            yaxis_title="Retention %",
        )
        st.plotly_chart(fig_ret, use_container_width=True)

    # Summary stats
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    wk0 = pivot["Wk 0"].mean() if "Wk 0" in pivot.columns else 0
    wk1 = pivot["Wk 1"].mean() if "Wk 1" in pivot.columns else 0
    wk4 = pivot["Wk 4"].mean() if "Wk 4" in pivot.columns else 0
    wk8 = pivot["Wk 8"].mean() if "Wk 8" in pivot.columns else 0

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-val">{wk0:.1f}%</div>
        <div class="kpi-label">Week 0 Retention</div>
        <div class="kpi-delta-neu">Acquisition week</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">{wk1:.1f}%</div>
        <div class="kpi-label">Week 1 Retention</div>
        <div class="kpi-delta-{'pos' if wk1 > 10 else 'neg'}">{'Good' if wk1 > 10 else 'Needs attention'}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">{wk4:.1f}%</div>
        <div class="kpi-label">Week 4 Retention</div>
        <div class="kpi-delta-{'pos' if wk4 > 5 else 'neg'}">{'Healthy' if wk4 > 5 else 'High churn'}</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-val">{wk8:.1f}%</div>
        <div class="kpi-label">Week 8 Retention</div>
        <div class="kpi-delta-neu">Long-term loyalty</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🐘 View PostgreSQL Query (Cohort Retention)"):
        st.markdown("""
<div class="sql-block">
<span style="color:#a78bfa">WITH</span> user_cohorts <span style="color:#a78bfa">AS</span> (
    <span style="color:#38bdf8">SELECT</span> user_id,
           <span style="color:#34d399">DATE_TRUNC</span>(<span style="color:#fbbf24">'week'</span>, <span style="color:#34d399">MIN</span>(event_timestamp))::<span style="color:#38bdf8">DATE</span> <span style="color:#a78bfa">AS</span> cohort_week
    <span style="color:#a78bfa">FROM</span> events <span style="color:#a78bfa">GROUP BY</span> user_id
),
user_purchase_weeks <span style="color:#a78bfa">AS</span> (
    <span style="color:#38bdf8">SELECT DISTINCT</span> user_id,
           <span style="color:#34d399">DATE_TRUNC</span>(<span style="color:#fbbf24">'week'</span>, event_timestamp)::<span style="color:#38bdf8">DATE</span> <span style="color:#a78bfa">AS</span> activity_week
    <span style="color:#a78bfa">FROM</span> events <span style="color:#a78bfa">WHERE</span> event_type = <span style="color:#fbbf24">'purchase'</span>
),
cohort_activity <span style="color:#a78bfa">AS</span> (
    <span style="color:#38bdf8">SELECT</span> uc.cohort_week,
           (<span style="color:#34d399">DATE_PART</span>(<span style="color:#fbbf24">'day'</span>, upw.activity_week::<span style="color:#38bdf8">TIMESTAMP</span>
                      - uc.cohort_week::<span style="color:#38bdf8">TIMESTAMP</span>)/7)::<span style="color:#38bdf8">INTEGER</span> <span style="color:#a78bfa">AS</span> week_number,
           <span style="color:#34d399">COUNT</span>(<span style="color:#fb923c">DISTINCT</span> upw.user_id) <span style="color:#a78bfa">AS</span> active_users
    <span style="color:#a78bfa">FROM</span> user_cohorts uc
    <span style="color:#a78bfa">JOIN</span> user_purchase_weeks upw <span style="color:#a78bfa">USING</span> (user_id)
    <span style="color:#a78bfa">GROUP BY</span> uc.cohort_week, week_number
),
cohort_sizes <span style="color:#a78bfa">AS</span> (
    <span style="color:#38bdf8">SELECT</span> cohort_week, <span style="color:#34d399">COUNT</span>(*) <span style="color:#a78bfa">AS</span> cohort_size,
           <span style="color:#34d399">DENSE_RANK</span>() <span style="color:#a78bfa">OVER</span> (<span style="color:#a78bfa">ORDER BY</span> cohort_week <span style="color:#a78bfa">DESC</span>) <span style="color:#a78bfa">AS</span> cohort_rank
    <span style="color:#a78bfa">FROM</span> user_cohorts <span style="color:#a78bfa">GROUP BY</span> cohort_week
)
<span style="color:#38bdf8">SELECT</span> ca.cohort_week, cs.cohort_size, ca.week_number, ca.active_users,
    <span style="color:#34d399">ROUND</span>(100.0 * ca.active_users / cs.cohort_size, 1) <span style="color:#a78bfa">AS</span> retention_pct
<span style="color:#a78bfa">FROM</span> cohort_activity ca
<span style="color:#a78bfa">JOIN</span> cohort_sizes cs <span style="color:#a78bfa">USING</span> (cohort_week)
<span style="color:#a78bfa">WHERE</span>  ca.week_number <span style="color:#a78bfa">BETWEEN</span> 0 <span style="color:#a78bfa">AND</span> 8
<span style="color:#a78bfa">ORDER BY</span> ca.cohort_week, ca.week_number;
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — A/B Test Results
# ════════════════════════════════════════════════════════════════════════════
with tab_ab:
    st.markdown("### 🧪 A/B Experiment Results — Checkout Redesign")
    st.markdown(
        "**Hypothesis**: A redesigned checkout page (high-contrast CTA + dynamic pricing banner) "
        "will improve conversion rate and average order value."
    )

    # Run analysis
    with st.spinner("Running statistical tests …"):
        engine = ABTestEngine(ab_df)
        results = engine.run_full_analysis(alpha=0.05)
        st.session_state["ab_results"] = results

    srm  = results["srm_check"]
    conv = results["conversion_rate"]
    aov  = results["average_order_value"]
    summ = results["experiment_summary"]

    # ── SRM guard-rail ──────────────────────────────────────────────────────
    st.markdown("#### 🛡️ Guard-Rail: Sample Ratio Mismatch (SRM)")
    srm_color = "red" if srm["srm_detected"] else "green"
    srm_label = "⚠️ SRM DETECTED" if srm["srm_detected"] else "✅ No SRM"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Control n", f"{srm['n_control']:,}")
    c2.metric("Treatment n", f"{srm['n_treatment']:,}")
    c3.metric("SRM χ²", f"{srm['chi2_statistic']}")
    c4.metric("SRM p-value", f"{srm['p_value']}")

    st.markdown(
        f'<span class="badge badge-{srm_color}">{srm_label}</span>&nbsp;&nbsp;'
        f'<span style="color:#94a3b8;font-size:.9rem">{srm["interpretation"]}</span>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Conversion Rate ─────────────────────────────────────────────────────
    st.markdown("#### 📊 Primary Metric: Conversion Rate (Chi-Square Test)")
    col_cr, col_aov_plot = st.columns(2)

    with col_cr:
        # Bar chart with CIs
        fig_cr = go.Figure()
        groups   = ["Control", "Treatment"]
        cr_vals  = [conv["ctrl_rate"] * 100, conv["trt_rate"] * 100]
        ci_lows  = [conv["ctrl_ci_lower"] * 100, conv["trt_ci_lower"] * 100]
        ci_highs = [conv["ctrl_ci_upper"] * 100, conv["trt_ci_upper"] * 100]
        err_lo   = [cr_vals[i] - ci_lows[i]  for i in range(2)]
        err_hi   = [ci_highs[i] - cr_vals[i] for i in range(2)]

        fig_cr.add_trace(go.Bar(
            x            = groups,
            y            = cr_vals,
            error_y      = dict(type="data", symmetric=False,
                                array=err_hi, arrayminus=err_lo,
                                color="rgba(255,255,255,0.6)", thickness=2),
            marker_color = [CTRL_C, TRT_C],
            text         = [f"{v:.2f}%" for v in cr_vals],
            textposition = "outside",
            textfont     = dict(color="#f1f5f9", family="Inter", size=13),
            width        = 0.45,
        ))
        plotly_dark(fig_cr, 350)
        fig_cr.update_layout(
            title="Conversion Rate with 95% Wilson CI",
            yaxis_title="Conversion Rate (%)",
            showlegend=False,
        )
        st.plotly_chart(fig_cr, use_container_width=True)

        # Result card
        sig_class = "stat-card-sig" if conv["is_significant"] else "stat-card-insig"
        badge_c   = "green" if conv["is_significant"] else "red"
        st.markdown(f"""
        <div class="stat-card {sig_class}">
            <span class="badge badge-{badge_c}">
                {'✅ Significant' if conv['is_significant'] else '❌ Not Significant'}
            </span>&nbsp;&nbsp;
            <span class="badge badge-purple">χ² = {conv['chi2_statistic']}</span>&nbsp;&nbsp;
            <span class="badge badge-blue">p = {conv['p_value']}</span>
            <br><br>
            <b style="color:#f1f5f9">Relative uplift:</b>
            <span style="color:{'#34d399' if conv['relative_uplift_pct'] > 0 else '#f87171'};font-weight:700">
                {conv['relative_uplift_pct']:+.1f}%
            </span>
            &nbsp;|&nbsp;
            <b style="color:#f1f5f9">Absolute:</b>
            <span style="color:#38bdf8;font-weight:700">
                {conv['absolute_uplift']*100:+.2f} pp
            </span>
            &nbsp;|&nbsp;
            <b style="color:#f1f5f9">Power:</b>
            <span style="color:#a78bfa;font-weight:700">{conv['statistical_power']*100:.0f}%</span>
            &nbsp;|&nbsp;
            <b style="color:#f1f5f9">Cohen's h:</b>
            <span style="color:#94a3b8">{conv['cohens_h']}</span>
        </div>
        """, unsafe_allow_html=True)

    # ── AOV Distribution ─────────────────────────────────────────────────────
    with col_aov_plot:
        ctrl_aov_vals = aov["ctrl_values"]
        trt_aov_vals  = aov["trt_values"]

        fig_aov = go.Figure()
        for vals, name, color in [
            (ctrl_aov_vals, "Control", CTRL_C),
            (trt_aov_vals,  "Treatment", TRT_C),
        ]:
            # KDE approximation using histogram
            fig_aov.add_trace(go.Histogram(
                x          = vals,
                name       = name,
                opacity    = 0.65,
                nbinsx     = 40,
                marker_color=color,
                histnorm   = "probability density",
            ))
        plotly_dark(fig_aov, 350)
        fig_aov.update_layout(
            title    = "AOV Distribution — Converters Only",
            xaxis_title="Order Value ($)",
            yaxis_title="Density",
            barmode  ="overlay",
            legend   =dict(orientation="h", y=1.08, x=0),
        )
        st.plotly_chart(fig_aov, use_container_width=True)

        sig_class2 = "stat-card-sig" if aov["is_significant"] else "stat-card-insig"
        badge_c2   = "green" if aov["is_significant"] else "red"
        st.markdown(f"""
        <div class="stat-card {sig_class2}">
            <span class="badge badge-{badge_c2}">
                {'✅ Significant' if aov['is_significant'] else '❌ Not Significant'}
            </span>&nbsp;&nbsp;
            <span class="badge badge-purple">t = {aov['t_statistic']}</span>&nbsp;&nbsp;
            <span class="badge badge-blue">p = {aov['p_value']}</span>
            <br><br>
            <b style="color:#f1f5f9">Ctrl AOV:</b> <span style="color:#38bdf8;font-weight:700">${aov['ctrl_mean']:.2f}</span>
            &nbsp;→&nbsp;
            <b style="color:#f1f5f9">Trt AOV:</b> <span style="color:#a78bfa;font-weight:700">${aov['trt_mean']:.2f}</span>
            &nbsp;|&nbsp;
            <b style="color:#f1f5f9">Diff:</b>
            <span style="color:{'#34d399' if aov['mean_difference'] > 0 else '#f87171'};font-weight:700">
                ${aov['mean_difference']:+.2f}
            </span>
            <br>
            <b style="color:#f1f5f9">95% CI:</b>
            <span style="color:#94a3b8">[${aov['ci_lower']:.2f}, ${aov['ci_upper']:.2f}]</span>
            &nbsp;|&nbsp;
            <b style="color:#f1f5f9">Cohen's d:</b> <span style="color:#94a3b8">{aov['cohens_d']}</span>
            &nbsp;|&nbsp;
            <b style="color:#f1f5f9">MDE:</b> <span style="color:#fbbf24">${aov['mde_dollars']:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Full metrics table ──────────────────────────────────────────────────
    st.markdown("#### 📋 Full Statistical Summary")
    metrics_rows = [
        {"Metric": "Conversion Rate (Control)",  "Value": f"{conv['ctrl_rate']*100:.3f}%"},
        {"Metric": "Conversion Rate (Treatment)","Value": f"{conv['trt_rate']*100:.3f}%"},
        {"Metric": "Absolute Uplift (conv.)",    "Value": f"{conv['absolute_uplift']*100:+.3f} pp"},
        {"Metric": "Relative Uplift (conv.)",    "Value": f"{conv['relative_uplift_pct']:+.2f}%"},
        {"Metric": "Chi-Square Statistic",       "Value": str(conv["chi2_statistic"])},
        {"Metric": "p-value (conversion)",       "Value": str(conv["p_value"])},
        {"Metric": "Cohen's h",                  "Value": str(conv["cohens_h"])},
        {"Metric": "Statistical Power (conv.)",  "Value": f"{conv['statistical_power']*100:.1f}%"},
        {"Metric": "─────────────────",          "Value": ""},
        {"Metric": "AOV — Control",              "Value": f"${aov['ctrl_mean']:.2f} ± ${aov['ctrl_std']:.2f}"},
        {"Metric": "AOV — Treatment",            "Value": f"${aov['trt_mean']:.2f} ± ${aov['trt_std']:.2f}"},
        {"Metric": "Mean Difference",            "Value": f"${aov['mean_difference']:+.2f}"},
        {"Metric": "95% CI (difference)",        "Value": f"[${aov['ci_lower']:.2f}, ${aov['ci_upper']:.2f}]"},
        {"Metric": "t-Statistic (Welch's)",      "Value": str(aov["t_statistic"])},
        {"Metric": "p-value (AOV)",              "Value": str(aov["p_value"])},
        {"Metric": "Degrees of Freedom",         "Value": str(aov["degrees_of_freedom"])},
        {"Metric": "Cohen's d",                  "Value": str(aov["cohens_d"])},
        {"Metric": "MDE at 80% Power",           "Value": f"${aov['mde_dollars']:.2f}"},
        {"Metric": "Statistical Power (AOV)",    "Value": f"{aov['statistical_power']*100:.1f}%"},
    ]
    st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True, hide_index=True)

    # ── Overall verdict ─────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("#### 🏁 System Recommendation")
    rec = summ["overall_recommendation"]
    rec_color = (
        "green"  if "SHIP" in rec and "NOT" not in rec else
        "red"    if "NOT" in rec or "INVALID" in rec else
        "yellow"
    )
    st.markdown(
        f'<div class="stat-card" style="border-color:rgba(56,189,248,.3);text-align:center;font-size:1.1rem;font-weight:700;color:#f1f5f9">'
        f'<span class="badge badge-{rec_color}" style="font-size:.9rem">{rec}</span>'
        f'</div>',
        unsafe_allow_html=True
    )


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — AI Executive Readout
# ════════════════════════════════════════════════════════════════════════════
with tab_ai:
    st.markdown("### 🤖 AI-Powered Executive Readout")
    st.markdown(
        "Pass all statistical outputs to **Groq (Llama 3.3 70B)** and receive a "
        "structured, plain-English business recommendation."
    )

    if "ab_results" not in st.session_state:
        st.info("Run the **A/B Test Results** tab first to compute statistics.")
    else:
        col_btn, col_status = st.columns([1, 3])
        with col_btn:
            gen_btn = st.button("✨ Generate AI Readout", key="gen_ai", use_container_width=True)
        with col_status:
            if not os.getenv("GROQ_API_KEY"):
                st.markdown(
                    '<span class="badge badge-yellow">⚠️  Add Groq API key in sidebar to enable AI</span>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    '<span class="badge badge-green">🟢 Groq ready — click Generate</span>',
                    unsafe_allow_html=True
                )

        if gen_btn:
            with st.spinner("🤖 Groq / Llama 3.3 is writing your executive readout …"):
                readout = generate_executive_readout(st.session_state["ab_results"])
                st.session_state["readout"] = readout

        if "readout" in st.session_state:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="ai-box">{st.session_state["readout"]}</div>',
                unsafe_allow_html=True
            )
            st.download_button(
                label      = "⬇️  Download Readout (Markdown)",
                data       = st.session_state["readout"],
                file_name  = "executive_readout.md",
                mime       = "text/markdown",
                key        = "download_readout",
            )
        else:
            # Show prompt preview
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown("#### 🔍 What the AI prompt includes")
            prompt_preview = """
| Input Block | Contents |
|---|---|
| **Experiment Brief** | Name, hypothesis, duration, total users |
| **SRM Guard-rail** | Chi-square stat, p-value, SRM status |
| **Conversion Rate** | Rates, uplift, CI, chi2, power, Cohen's h |
| **Average Order Value** | Means, CIs, t-stat, df, Cohen's d, MDE |
| **System Recommendation** | Pre-computed verdict |

The AI is instructed to respond with 6 structured sections:
`Validity → Snapshot → Statistical Interpretation → Business Impact → Decision → Next Steps`
"""
            st.markdown(prompt_preview)

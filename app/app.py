"""
SmartWaste AI — Advanced Streamlit App
YOLOv8-based waste detection, classification & analytics
Author: Aman Kumar Gupta (BTP Final Year Project)
"""

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import io
import textwrap
from datetime import datetime

# PDF generation (optional dependency)
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table,
        TableStyle, HRFlowable, KeepTogether,
    )
    from reportlab.lib.enums import TA_CENTER, TA_LEFT
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# ══════════════════════════════════════════════
#  PAGE CONFIG  (must be first Streamlit call)
# ══════════════════════════════════════════════
st.set_page_config(
    page_title="SmartWaste AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════
#  CONSTANTS & DATA
# ══════════════════════════════════════════════
WASTE_BINS: dict[str, dict] = {
    "plastic bottle": {"bin": "♻️ Blue Bin",  "color": "#3b82f6", "type": "Recyclable",    "tip": "Rinse before recycling"},
    "plastic bag":    {"bin": "🚫 Landfill",   "color": "#6b7280", "type": "Non-Recyclable","tip": "Most plastic bags clog recycling machines"},
    "aluminum can":   {"bin": "♻️ Blue Bin",  "color": "#3b82f6", "type": "Recyclable",    "tip": "Crush to save space"},
    "glass bottle":   {"bin": "♻️ Blue Bin",  "color": "#3b82f6", "type": "Recyclable",    "tip": "Separate by colour if required"},
    "paper":          {"bin": "📄 Paper Bin",  "color": "#f59e0b", "type": "Recyclable",    "tip": "Keep dry and clean"},
    "cardboard":      {"bin": "📄 Paper Bin",  "color": "#f59e0b", "type": "Recyclable",    "tip": "Flatten before disposal"},
    "newspaper":      {"bin": "📄 Paper Bin",  "color": "#f59e0b", "type": "Recyclable",    "tip": "Bundle together"},
    "tin can":        {"bin": "♻️ Blue Bin",  "color": "#3b82f6", "type": "Recyclable",    "tip": "Remove lids and rinse"},
    "metal":          {"bin": "♻️ Blue Bin",  "color": "#3b82f6", "type": "Recyclable",    "tip": "Scrap metal is valuable"},
    "organic waste":  {"bin": "🌱 Green Bin", "color": "#22c55e", "type": "Organic",       "tip": "Great for composting"},
    "food waste":     {"bin": "🌱 Green Bin", "color": "#22c55e", "type": "Organic",       "tip": "Compost if possible"},
    "vegetable":      {"bin": "🌱 Green Bin", "color": "#22c55e", "type": "Organic",       "tip": "Compostable"},
    "fruit":          {"bin": "🌱 Green Bin", "color": "#22c55e", "type": "Organic",       "tip": "Compostable"},
    "leaves":         {"bin": "🌱 Green Bin", "color": "#22c55e", "type": "Organic",       "tip": "Ideal for garden compost"},
    "battery":        {"bin": "⚠️ Hazardous", "color": "#ef4444", "type": "Hazardous",     "tip": "Never throw in a regular bin"},
    "syringe":        {"bin": "⚠️ Hazardous", "color": "#ef4444", "type": "Hazardous",     "tip": "Use a sharps disposal box"},
    "chemical":       {"bin": "⚠️ Hazardous", "color": "#ef4444", "type": "Hazardous",     "tip": "Contact your local waste authority"},
    "medicine":       {"bin": "⚠️ Hazardous", "color": "#ef4444", "type": "Hazardous",     "tip": "Return to pharmacy"},
    "paint":          {"bin": "⚠️ Hazardous", "color": "#ef4444", "type": "Hazardous",     "tip": "Take to a hazardous waste facility"},
    "electronics":    {"bin": "💻 E-Waste",   "color": "#8b5cf6", "type": "E-Waste",       "tip": "Take to a certified e-waste centre"},
    "phone":          {"bin": "💻 E-Waste",   "color": "#8b5cf6", "type": "E-Waste",       "tip": "Wipe data before disposal"},
    "computer":       {"bin": "💻 E-Waste",   "color": "#8b5cf6", "type": "E-Waste",       "tip": "Certified recycler required"},
    "cable":          {"bin": "💻 E-Waste",   "color": "#8b5cf6", "type": "E-Waste",       "tip": "Contains recyclable copper"},
    "_default":       {"bin": "🚫 Landfill",  "color": "#6b7280", "type": "General Waste", "tip": "Check local guidelines"},
}

TYPE_COLORS = {
    "Recyclable":    "#3b82f6",
    "Organic":       "#22c55e",
    "Hazardous":     "#ef4444",
    "E-Waste":       "#8b5cf6",
    "Non-Recyclable":"#6b7280",
    "General Waste": "#94a3b8",
}

BIN_COLORS = {
    "♻️ Blue Bin":  "#3b82f6",
    "📄 Paper Bin": "#f59e0b",
    "🌱 Green Bin": "#22c55e",
    "⚠️ Hazardous": "#ef4444",
    "💻 E-Waste":   "#8b5cf6",
    "🚫 Landfill":  "#6b7280",
}

# ══════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════
_DEFAULTS = {
    "history":          [],
    "theme":            "dark",
    "total_detections": 0,
    "session_start":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ══════════════════════════════════════════════
#  THEME — single CSS block using data-theme attr
# ══════════════════════════════════════════════
THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');

/* ── CSS variables per theme ── */
[data-theme="dark"] {
    --bg:          #0a0e17;
    --bg2:         #0d1a2e;
    --bg3:         #0f1f35;
    --border:      #1e3a5f;
    --border2:     #1e4a7a;
    --text:        #e2e8f0;
    --text2:       #94a3b8;
    --text3:       #64748b;
    --accent:      #38bdf8;
    --accent2:     #818cf8;
    --card-shadow: 0 4px 24px rgba(0,0,0,0.4);
    --tip-bg:      #0c1f38;
    --tip-border:  #38bdf8;
    --tip-text:    #94a3b8;
}
[data-theme="light"] {
    --bg:          #f8fafc;
    --bg2:         #f0f9ff;
    --bg3:         #ffffff;
    --border:      #e2e8f0;
    --border2:     #bfdbfe;
    --text:        #1e293b;
    --text2:       #475569;
    --text3:       #94a3b8;
    --accent:      #0369a1;
    --accent2:     #7c3aed;
    --card-shadow: 0 2px 12px rgba(0,0,0,0.06);
    --tip-bg:      #f0f9ff;
    --tip-border:  #0284c7;
    --tip-text:    #475569;
}

/* ── Base ── */
html, body, .stApp {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'Syne', sans-serif !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * { color: var(--text2) !important; }
section[data-testid="stSidebar"] h3 { color: var(--text) !important; }
section[data-testid="stSidebar"] code { color: var(--accent) !important; background: var(--bg3) !important; }

/* ── Main content text ── */
.stMarkdown, .stMarkdown p, .stMarkdown li,
.element-container p, .element-container span,
[data-testid="stText"] { color: var(--text) !important; }
h1, h2, h3, h4 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    color: var(--text) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: var(--bg2) !important;
    border-radius: 12px; padding: 4px; gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--text3) !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg3) !important;
    color: var(--accent) !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.12) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--accent2)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    padding: 10px 24px !important;
    transition: opacity 0.2s ease !important;
}
.stButton > button:hover { opacity: 0.88 !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg2) !important;
    border: 2px dashed var(--border2) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
}
[data-testid="stFileUploaderDropzone"] * { color: var(--text2) !important; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 10px !important; overflow: hidden; }
.stDataFrame * { color: var(--text) !important; }

/* ── Expander ── */
details { border: 1px solid var(--border) !important; border-radius: 10px !important; }
details summary { color: var(--text) !important; background: var(--bg2) !important; padding: 10px 14px !important; border-radius: 10px !important; }
details[open] summary { border-radius: 10px 10px 0 0 !important; }

/* ── Slider ── */
.stSlider label { color: var(--text2) !important; }
.stSlider [data-testid="stSliderThumb"] { background: var(--accent) !important; }

/* ── Radio ── */
.stRadio label { color: var(--text2) !important; }
.stRadio [data-testid="stMarkdownContainer"] p { color: var(--text2) !important; }

/* ── Progress bar ── */
.stProgress > div > div { background: linear-gradient(90deg, var(--accent), var(--accent2)) !important; }

/* ── Alerts ── */
[data-testid="stNotification"] { border-radius: 10px !important; }

/* ── Custom components ── */
.waste-card {
    background: var(--bg3);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px;
    margin: 10px 0;
    box-shadow: var(--card-shadow);
    color: var(--text);
}
.waste-card * { color: var(--text) !important; }

.metric-card {
    background: var(--bg2);
    border: 1px solid var(--border2);
    border-radius: 12px;
    padding: 18px 22px;
    text-align: center;
    box-shadow: var(--card-shadow);
}
.metric-number {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    font-weight: 700;
    color: var(--accent) !important;
    line-height: 1.1;
}
.metric-label {
    font-size: 0.72rem;
    color: var(--text3) !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-top: 5px;
}

.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 50%, #22c55e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    letter-spacing: -2px;
    line-height: 1.1;
}
.hero-sub {
    color: var(--text3);
    font-size: 0.88rem;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-top: 6px;
}

.section-divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 22px 0;
}

.tip-box {
    background: var(--tip-bg);
    border-left: 3px solid var(--tip-border);
    border-radius: 0 8px 8px 0;
    padding: 9px 13px;
    font-size: 0.81rem;
    color: var(--tip-text) !important;
    margin-top: 8px;
}

.bin-label {
    font-size: 1.15rem;
    font-weight: 700;
    margin-bottom: 2px;
}
.bin-type {
    font-size: 0.73rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 10px;
    color: var(--text3) !important;
}
.bin-item {
    font-size: 0.85rem;
    margin: 4px 0;
    color: var(--text) !important;
}
.bin-item span { color: var(--text3) !important; }
</style>
"""

def inject_theme(theme: str):
    """Inject CSS and set data-theme attribute via JS."""
    st.markdown(THEME_CSS, unsafe_allow_html=True)
    st.markdown(
        f'<script>document.documentElement.setAttribute("data-theme", "{theme}");</script>',
        unsafe_allow_html=True,
    )
    # Fallback: also set on body (Streamlit wraps in stApp)
    st.markdown(
        f"""<style>
        html, body, .stApp, [data-testid="stAppViewContainer"],
        [data-testid="stHeader"], section[data-testid="stSidebar"],
        .main, .block-container {{
            --bg:          {'#0a0e17' if theme=='dark' else '#f8fafc'};
            --bg2:         {'#0d1a2e' if theme=='dark' else '#f0f9ff'};
            --bg3:         {'#0f1f35' if theme=='dark' else '#ffffff'};
            --border:      {'#1e3a5f' if theme=='dark' else '#e2e8f0'};
            --border2:     {'#1e4a7a' if theme=='dark' else '#bfdbfe'};
            --text:        {'#e2e8f0' if theme=='dark' else '#1e293b'};
            --text2:       {'#94a3b8' if theme=='dark' else '#475569'};
            --text3:       {'#64748b' if theme=='dark' else '#94a3b8'};
            --accent:      {'#38bdf8' if theme=='dark' else '#0369a1'};
            --accent2:     {'#818cf8' if theme=='dark' else '#7c3aed'};
            --card-shadow: {'0 4px 24px rgba(0,0,0,0.4)' if theme=='dark' else '0 2px 12px rgba(0,0,0,0.06)'};
            --tip-bg:      {'#0c1f38' if theme=='dark' else '#f0f9ff'};
            --tip-border:  {'#38bdf8' if theme=='dark' else '#0284c7'};
            --tip-text:    {'#94a3b8' if theme=='dark' else '#475569'};
        }}
        html, body, .stApp {{
            background-color: var(--bg) !important;
            color: var(--text) !important;
        }}
        section[data-testid="stSidebar"] {{
            background: var(--bg2) !important;
            border-right: 1px solid var(--border) !important;
        }}
        </style>""",
        unsafe_allow_html=True,
    )

inject_theme(st.session_state.theme)

# ══════════════════════════════════════════════
#  MODEL LOADER
# ══════════════════════════════════════════════
@st.cache_resource(show_spinner="Loading YOLOv8 model…")
def load_model() -> YOLO:
    return YOLO("models/best.pt")

model = load_model()

# ══════════════════════════════════════════════
#  PURE UTILITY FUNCTIONS
# ══════════════════════════════════════════════
def get_waste_info(label: str) -> dict:
    label_lower = label.lower().strip()
    for key in WASTE_BINS:
        if key != "_default" and (key in label_lower or label_lower in key):
            return WASTE_BINS[key]
    return WASTE_BINS["_default"]

def process_detections(results, conf_threshold: float) -> tuple[dict, list]:
    boxes  = results[0].boxes
    names  = model.names
    counts: dict = {}
    records: list = []
    for box in boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        if conf < conf_threshold:
            continue
        label = names[cls_id]
        counts[label] = counts.get(label, 0) + 1
        info = get_waste_info(label)
        records.append({
            "Class":      label,
            "Confidence": round(conf, 3),
            "Bin":        info["bin"],
            "Type":       info["type"],
            "Tip":        info["tip"],
        })
    return counts, records

def run_detection(img_array: np.ndarray, conf_threshold: float):
    results   = model(img_array, conf=conf_threshold, verbose=False)
    annotated = results[0].plot()
    counts, records = process_detections(results, conf_threshold)
    return annotated, counts, records

def image_to_bytes(img_bgr: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", img_bgr)
    return buf.tobytes()

def add_to_history(filename: str, records: list, counts: dict):
    st.session_state.history.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "date":      datetime.now().strftime("%Y-%m-%d"),
        "file":      filename,
        "total":     sum(counts.values()),
        "classes":   counts,
        "records":   records,
    })
    st.session_state.total_detections += sum(counts.values())

def session_export_csv() -> str:
    rows = []
    for h in st.session_state.history:
        for r in h["records"]:
            rows.append({**r, "File": h["file"], "Time": h["timestamp"]})
    return pd.DataFrame(rows).to_csv(index=False) if rows else ""

# ══════════════════════════════════════════════
#  CHART HELPERS  (theme-aware, DRY)
# ══════════════════════════════════════════════
def _chart_layout(theme: str, height: int = 300) -> dict:
    dark = theme == "dark"
    return dict(
        paper_bgcolor = "#0d1a2e" if dark else "#f8fafc",
        plot_bgcolor  = "#0a0e17" if dark else "#ffffff",
        font          = dict(color="#e2e8f0" if dark else "#1e293b", family="Space Mono"),
        margin        = dict(t=10, b=10, l=10, r=10),
        height        = height,
        showlegend    = False,
        xaxis         = dict(gridcolor="#1e3a5f" if dark else "#e2e8f0"),
        yaxis         = dict(gridcolor="#1e3a5f" if dark else "#e2e8f0"),
    )

def chart_pie(counts: dict, theme: str, height: int = 300):
    if not counts:
        return None
    fg = "#e2e8f0" if theme == "dark" else "#1e293b"
    fig = px.pie(
        values=list(counts.values()),
        names=list(counts.keys()),
        color_discrete_sequence=[get_waste_info(k)["color"] for k in counts],
        hole=0.45,
    )
    fig.update_traces(textposition="outside", textinfo="label+percent",
                      textfont=dict(color=fg, size=11))
    fig.update_layout(**_chart_layout(theme, height))
    return fig

def chart_bar(counts: dict, theme: str, color_map: dict | None = None, height: int = 300):
    if not counts:
        return None
    colors = [color_map.get(k, "#64748b") if color_map else get_waste_info(k)["color"]
              for k in counts]
    fg = "#e2e8f0" if theme == "dark" else "#1e293b"
    fig = go.Figure(go.Bar(
        x=list(counts.keys()), y=list(counts.values()),
        marker=dict(color=colors, line=dict(width=0)),
        text=list(counts.values()), textposition="outside",
        textfont=dict(color=fg),
    ))
    layout = _chart_layout(theme, height)
    layout["yaxis"]["title"] = "Count"
    layout["xaxis"]["tickangle"] = -30
    fig.update_layout(**layout)
    return fig

def chart_histogram(confs: list[float], theme: str, height: int = 250):
    if not confs:
        return None
    fig = go.Figure(go.Histogram(
        x=confs, nbinsx=15,
        marker=dict(color="#38bdf8" if theme=="dark" else "#0369a1", line=dict(width=0)),
    ))
    layout = _chart_layout(theme, height)
    layout["xaxis"] = dict(title="Confidence", range=[0, 1],
                           gridcolor="#1e3a5f" if theme=="dark" else "#e2e8f0")
    layout["yaxis"]["title"] = "Count"
    fig.update_layout(**layout)
    return fig

def chart_trend(history: list, theme: str, height: int = 250):
    if len(history) < 2:
        return None
    fig = go.Figure(go.Scatter(
        x=[h["timestamp"] for h in history],
        y=[h["total"]     for h in history],
        mode="lines+markers",
        line=dict(color="#38bdf8" if theme=="dark" else "#0369a1", width=2),
        marker=dict(color="#818cf8", size=8),
        fill="tozeroy",
        fillcolor="rgba(56,189,248,0.08)",
    ))
    layout = _chart_layout(theme, height)
    layout["yaxis"]["title"] = "Objects Detected"
    fig.update_layout(**layout)
    return fig

def _plotly(fig, key: str = ""):
    if fig:
        st.plotly_chart(fig, use_container_width=True,
                        config={"displayModeBar": False}, key=key)

# ══════════════════════════════════════════════
#  REUSABLE UI COMPONENTS
# ══════════════════════════════════════════════
def metric_card(value, label: str, color: str | None = None):
    style = f"color:{color} !important;" if color else ""
    st.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-number" style="{style}">{value}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

def divider():
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

def metrics_row(items: list[tuple]):
    """items = [(value, label), ...] or (value, label, color)"""
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            metric_card(item[0], item[1], item[2] if len(item) > 2 else None)

def bin_cards(records: list, theme: str):
    """Render one card per bin group."""
    bin_groups: dict[str, list] = {}
    for r in records:
        bin_groups.setdefault(r["Bin"], []).append(r)
    if not bin_groups:
        return
    cols = st.columns(min(len(bin_groups), 4))
    for i, (bin_name, items) in enumerate(bin_groups.items()):
        info = get_waste_info(items[0]["Class"])
        with cols[i % len(cols)]:
            item_html = "".join(
                f'<div class="bin-item">• {it["Class"]} '
                f'<span>({it["Confidence"]})</span></div>'
                for it in items
            )
            st.markdown(
                f'<div class="waste-card" style="border-left:4px solid {info["color"]};">'
                f'<div class="bin-label" style="color:{info["color"]};">{bin_name}</div>'
                f'<div class="bin-type">{info["type"]}</div>'
                f'{item_html}'
                f'<div class="tip-box">💡 {items[0]["Tip"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

def detection_summary_section(annotated, counts, records, filename, elapsed, theme):
    """Full detection output: metrics → charts → bin cards → table → downloads."""

    divider()
    avg_conf = round(sum(r["Confidence"] for r in records) / len(records), 3) if records else 0
    metrics_row([
        (sum(counts.values()), "Objects Found"),
        (len(counts),          "Unique Classes"),
        (avg_conf,             "Avg Confidence"),
        (f"{elapsed:.2f}s",   "Inference Time"),
    ])
    divider()

    if not records:
        st.warning("⚠️ No objects detected. Try lowering the confidence threshold.")
        return

    # Charts
    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown("**Distribution**")
        _plotly(chart_pie(counts, theme), f"pie_{filename}")
    with c2:
        st.markdown("**Object Counts**")
        _plotly(chart_bar(counts, theme), f"bar_{filename}")

    divider()
    st.markdown("#### 🗑️ Bin Classification")
    bin_cards(records, theme)

    divider()
    st.markdown("#### 📋 Detection Details")
    df = pd.DataFrame(records)[["Class", "Confidence", "Bin", "Type"]]
    st.dataframe(df, use_container_width=True, hide_index=True)

    dl1, dl2, dl3 = st.columns(3)
    with dl1:
        st.download_button(
            "📸 Download Image",
            data=image_to_bytes(annotated),
            file_name=f"detected_{filename}",
            mime="image/png", use_container_width=True,
        )
    with dl2:
        st.download_button(
            "📄 Download CSV",
            data=df.to_csv(index=False),
            file_name=f"detections_{filename}.csv",
            mime="text/csv", use_container_width=True,
        )
    with dl3:
        if PDF_AVAILABLE:
            pdf_bytes = generate_pdf_report(
                st.session_state.history,
                st.session_state.total_detections,
                st.session_state.session_start,
            )
            if pdf_bytes:
                st.download_button(
                    "📑 Download PDF Report",
                    data=pdf_bytes,
                    file_name=f"waste_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf", use_container_width=True,
                )
        else:
            st.button("📑 PDF (install reportlab)", disabled=True,
                      use_container_width=True,
                      help="Run: pip install reportlab")

    divider()
    st.markdown("**Confidence Distribution**")
    _plotly(chart_histogram([r["Confidence"] for r in records], theme), f"hist_{filename}")



# ══════════════════════════════════════════════
#  PDF REPORT GENERATOR
# ══════════════════════════════════════════════
def generate_pdf_report(history: list, total_detections: int, session_start: str) -> bytes | None:
    """Generate a full-session PDF report using ReportLab."""
    if not PDF_AVAILABLE:
        return None

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2*cm, bottomMargin=2*cm,
    )

    # ── Styles ──────────────────────────────────────────────
    base = getSampleStyleSheet()
    DARK  = rl_colors.HexColor("#0f172a")
    GREEN = rl_colors.HexColor("#16a34a")
    BLUE  = rl_colors.HexColor("#0369a1")
    GRAY  = rl_colors.HexColor("#64748b")
    LIGHT = rl_colors.HexColor("#f1f5f9")
    RED   = rl_colors.HexColor("#dc2626")
    PURP  = rl_colors.HexColor("#7c3aed")

    title_style = ParagraphStyle("Title", parent=base["Title"],
        fontSize=22, textColor=DARK, spaceAfter=4, alignment=TA_CENTER,
        fontName="Helvetica-Bold")
    sub_style = ParagraphStyle("Sub", parent=base["Normal"],
        fontSize=9, textColor=GRAY, alignment=TA_CENTER, spaceAfter=14)
    h2_style = ParagraphStyle("H2", parent=base["Heading2"],
        fontSize=13, textColor=DARK, spaceBefore=16, spaceAfter=6,
        fontName="Helvetica-Bold")
    h3_style = ParagraphStyle("H3", parent=base["Heading3"],
        fontSize=10, textColor=BLUE, spaceBefore=10, spaceAfter=4,
        fontName="Helvetica-Bold")
    body_style = ParagraphStyle("Body", parent=base["Normal"],
        fontSize=9, textColor=DARK, spaceAfter=3, leading=14)
    small_style = ParagraphStyle("Small", parent=base["Normal"],
        fontSize=8, textColor=GRAY, spaceAfter=2)
    tip_style = ParagraphStyle("Tip", parent=base["Normal"],
        fontSize=8, textColor=rl_colors.HexColor("#0c4a6e"),
        backColor=rl_colors.HexColor("#e0f2fe"),
        leftIndent=6, rightIndent=6, spaceBefore=4, spaceAfter=6,
        borderPad=4)

    def hr():
        return HRFlowable(width="100%", thickness=0.5,
                          color=rl_colors.HexColor("#e2e8f0"), spaceAfter=8)

    def tbl_style(header_color=BLUE):
        return TableStyle([
            ("BACKGROUND",  (0, 0), (-1, 0),  header_color),
            ("TEXTCOLOR",   (0, 0), (-1, 0),  rl_colors.white),
            ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
            ("FONTSIZE",    (0, 0), (-1, 0),  8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [LIGHT, rl_colors.white]),
            ("FONTSIZE",    (0, 1), (-1, -1), 8),
            ("TEXTCOLOR",   (0, 1), (-1, -1), DARK),
            ("ALIGN",       (0, 0), (-1, -1), "LEFT"),
            ("VALIGN",      (0, 0), (-1, -1), "MIDDLE"),
            ("ROWHEIGHT",   (0, 0), (-1, -1), 16),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",(0, 0), (-1, -1), 6),
            ("GRID",        (0, 0), (-1, -1), 0.3, rl_colors.HexColor("#e2e8f0")),
            ("ROUNDEDCORNERS", (0, 0), (-1, -1), [3, 3, 3, 3]),
        ])

    story = []

    # ── COVER ────────────────────────────────────────────────
    story.append(Spacer(1, 1*cm))
    story.append(Paragraph("♻ SmartWaste AI", title_style))
    story.append(Paragraph("Detection Session Report", sub_style))
    story.append(Paragraph(
        f"Generated: {datetime.now().strftime('%d %B %Y, %H:%M')}  |  "
        f"Session started: {session_start}",
        sub_style,
    ))
    story.append(hr())

    # ── SESSION SUMMARY ──────────────────────────────────────
    story.append(Paragraph("Session Summary", h2_style))

    agg_counts: dict = {}
    agg_types:  dict = {}
    agg_bins:   dict = {}
    all_confs:  list = []

    for h in history:
        for r in h["records"]:
            agg_counts[r["Class"]] = agg_counts.get(r["Class"], 0) + 1
            agg_types[r["Type"]]   = agg_types.get(r["Type"],   0) + 1
            agg_bins[r["Bin"]]     = agg_bins.get(r["Bin"],     0) + 1
            all_confs.append(r["Confidence"])

    avg_conf = round(sum(all_confs)/len(all_confs), 3) if all_confs else 0
    top_cls  = max(agg_counts, key=agg_counts.get) if agg_counts else "—"
    total_t  = sum(agg_types.values()) or 1

    summary_data = [
        ["Metric", "Value"],
        ["Images Processed",   str(len(history))],
        ["Total Objects Detected", str(total_detections)],
        ["Unique Waste Classes",   str(len(agg_counts))],
        ["Average Confidence",     str(avg_conf)],
        ["Most Detected Class",    top_cls],
        ["Session Start",          session_start],
        ["Report Generated",       datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
    ]
    t = Table(summary_data, colWidths=[8*cm, 8*cm])
    t.setStyle(tbl_style(DARK))
    story.append(t)
    story.append(Spacer(1, 0.5*cm))

    # ── WASTE TYPE BREAKDOWN ──────────────────────────────────
    story.append(hr())
    story.append(Paragraph("Waste Type Breakdown", h2_style))

    type_color_map = {
        "Recyclable": BLUE, "Organic": GREEN, "Hazardous": RED,
        "E-Waste": PURP, "Non-Recyclable": GRAY, "General Waste": GRAY,
    }
    type_rows = [["Waste Type", "Count", "Percentage", "Disposal"]]
    type_disposal = {
        "Recyclable":    "Blue / Paper Bin",
        "Organic":       "Green / Compost Bin",
        "Hazardous":     "Hazardous Waste Facility",
        "E-Waste":       "Certified E-Waste Centre",
        "Non-Recyclable":"Landfill",
        "General Waste": "Landfill",
    }
    for wtype, cnt in sorted(agg_types.items(), key=lambda x: -x[1]):
        pct = f"{round(cnt*100/total_t)}%"
        type_rows.append([wtype, str(cnt), pct, type_disposal.get(wtype, "Check guidelines")])

    t2 = Table(type_rows, colWidths=[4.5*cm, 2.5*cm, 3*cm, 7*cm])
    t2.setStyle(tbl_style(GREEN))
    story.append(t2)
    story.append(Spacer(1, 0.3*cm))

    recyclable_pct = round(agg_types.get("Recyclable", 0) * 100 / total_t)
    story.append(Paragraph(
        f"Recyclability score: <b>{recyclable_pct}%</b> of detected waste is recyclable.",
        body_style,
    ))

    # ── BIN DISTRIBUTION ─────────────────────────────────────
    story.append(hr())
    story.append(Paragraph("Bin Distribution", h2_style))

    bin_rows = [["Bin", "Items Detected"]]
    for bin_name, cnt in sorted(agg_bins.items(), key=lambda x: -x[1]):
        # strip emoji for clean PDF text
        clean = bin_name.encode("ascii", "ignore").decode().strip()
        bin_rows.append([clean if clean else bin_name, str(cnt)])

    t3 = Table(bin_rows, colWidths=[12*cm, 4*cm])
    t3.setStyle(tbl_style(BLUE))
    story.append(t3)

    # ── CLASS DETAIL TABLE ────────────────────────────────────
    story.append(hr())
    story.append(Paragraph("Detected Class Details", h2_style))

    class_rows = [["Class", "Count", "Bin", "Type", "Disposal Tip"]]
    for cls_name, cnt in sorted(agg_counts.items(), key=lambda x: -x[1]):
        info = get_waste_info(cls_name)
        clean_bin = info["bin"].encode("ascii", "ignore").decode().strip()
        tip = textwrap.shorten(info["tip"], width=40, placeholder="...")
        class_rows.append([cls_name, str(cnt),
                           clean_bin if clean_bin else info["bin"],
                           info["type"], tip])

    t4 = Table(class_rows, colWidths=[4*cm, 1.5*cm, 3.5*cm, 3*cm, 5*cm])
    t4.setStyle(tbl_style(rl_colors.HexColor("#0f172a")))
    story.append(t4)

    # ── PER-IMAGE LOG ─────────────────────────────────────────
    story.append(hr())
    story.append(Paragraph("Per-Image Detection Log", h2_style))

    for h in history:
        top = max(h["classes"], key=h["classes"].get) if h["classes"] else "—"
        story.append(KeepTogether([
            Paragraph(f"{h['timestamp']}  —  {h['file']}", h3_style),
            Paragraph(
                f"Objects detected: <b>{h['total']}</b>  |  "
                f"Top class: <b>{top}</b>  |  "
                f"Unique classes: <b>{len(h['classes'])}</b>",
                body_style,
            ),
        ]))
        if h["records"]:
            img_rows = [["Class", "Confidence", "Bin", "Type"]]
            for r in h["records"]:
                clean_bin = r["Bin"].encode("ascii","ignore").decode().strip()
                img_rows.append([r["Class"], str(r["Confidence"]),
                                 clean_bin if clean_bin else r["Bin"], r["Type"]])
            ti = Table(img_rows, colWidths=[5*cm, 3*cm, 5*cm, 4*cm])
            ti.setStyle(tbl_style(GRAY))
            story.append(ti)
        story.append(Spacer(1, 0.3*cm))

    # ── FOOTER ────────────────────────────────────────────────
    story.append(hr())
    story.append(Paragraph(
        "SmartWaste AI  |  YOLOv8-based Waste Detection  |  "
        "B.Tech Final Year Project — Aman Kumar Gupta",
        sub_style,
    ))

    doc.build(story)
    return buf.getvalue()

# ══════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════
with st.sidebar:
    h_col, btn_col = st.columns([4, 1])
    with h_col:
        st.markdown("### ⚙️ Controls")
    with btn_col:
        lbl = "☀️" if st.session_state.theme == "dark" else "🌙"
        if st.button(lbl, key="theme_btn", help="Toggle light/dark mode"):
            st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
            st.rerun()

    st.markdown("---")
    raw_mode = st.radio(
        "**Detection Mode**",
        ["🖼️ Image", "📦 Batch", "📷 Webcam"],
        label_visibility="visible",
    )
    mode = raw_mode.split(" ", 1)[1]

    st.markdown("---")
    confidence = st.slider(
        "**Confidence Threshold**", 0.05, 1.0, 0.45, 0.05,
        help="Detections below this score are ignored",
    )

    st.markdown("---")
    st.markdown("**📊 Session Stats**")
    st.markdown(
        f'<div style="font-size:0.8rem;line-height:2.1;color:var(--text2);">'
        f'🕐 Started: <code>{st.session_state.session_start}</code><br>'
        f'🖼️ Images: <code>{len(st.session_state.history)}</code><br>'
        f'🗑️ Detections: <code>{st.session_state.total_detections}</code>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    if st.button("🗑️ Clear Session", use_container_width=True):
        st.session_state.history = []
        st.session_state.total_detections = 0
        st.rerun()

    csv_data = session_export_csv()
    if csv_data:
        st.download_button(
            "📥 Export Session CSV",
            data=csv_data,
            file_name=f"waste_session_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv", use_container_width=True,
        )

# ══════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════
st.markdown(
    '<div style="padding:10px 0 24px;">'
    '<div class="hero-title">♻️ SmartWaste AI</div>'
    '<div class="hero-sub">YOLOv8 · Real-Time Waste Detection & Classification</div>'
    '</div>',
    unsafe_allow_html=True,
)

# ══════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════
tab_detect, tab_analytics, tab_history, tab_about = st.tabs(
    ["🔍 Detect", "📊 Analytics", "🗂️ History", "ℹ️ About"]
)
theme = st.session_state.theme  # local alias for readability

# ══════════════════════════════════════════════════════════
#  TAB 1 — DETECT
# ══════════════════════════════════════════════════════════
with tab_detect:

    # ── SINGLE IMAGE ──────────────────────────────────────
    if mode == "Image":
        st.markdown("#### Upload an image for waste detection")
        uploaded = st.file_uploader(
            "Upload image", type=["jpg", "png", "jpeg", "webp"],
            label_visibility="collapsed",
        )
        if uploaded:
            image     = Image.open(uploaded).convert("RGB")
            img_array = np.array(image)

            with st.spinner("🔍 Running detection…"):
                t0 = time.perf_counter()
                annotated, counts, records = run_detection(img_array, confidence)
                elapsed = time.perf_counter() - t0

            add_to_history(uploaded.name, records, counts)

            col1, col2 = st.columns(2, gap="medium")
            with col1:
                st.markdown("**Original**")
                st.image(image, use_container_width=True)
            with col2:
                st.markdown("**Detected**")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

            detection_summary_section(annotated, counts, records, uploaded.name, elapsed, theme)

    # ── BATCH ─────────────────────────────────────────────
    elif mode == "Batch":
        st.markdown("#### Upload multiple images for batch processing")
        uploads = st.file_uploader(
            "Upload images", type=["jpg", "png", "jpeg", "webp"],
            accept_multiple_files=True,
            label_visibility="collapsed",
        )
        if uploads:
            st.info(f"📦 {len(uploads)} image(s) queued…")
            prog = st.progress(0)
            status = st.empty()

            batch:        list = []
            agg_counts:   dict = {}
            agg_records:  list = []

            for i, f in enumerate(uploads):
                status.markdown(f"🔍 Processing `{f.name}` ({i+1}/{len(uploads)})")
                img   = Image.open(f).convert("RGB")
                arr   = np.array(img)
                t0    = time.perf_counter()
                ann, counts, records = run_detection(arr, confidence)
                elapsed = time.perf_counter() - t0

                add_to_history(f.name, records, counts)

                for k, v in counts.items():
                    agg_counts[k] = agg_counts.get(k, 0) + v
                for r in records:
                    agg_records.append({**r, "File": f.name})

                batch.append(dict(name=f.name, image=img, annotated=ann,
                                  counts=counts, records=records, elapsed=elapsed))
                prog.progress((i + 1) / len(uploads))

            status.success("✅ Batch complete!")
            divider()

            avg_t = sum(r["elapsed"] for r in batch) / len(batch)
            metrics_row([
                (len(uploads),              "Images"),
                (sum(agg_counts.values()),  "Total Objects"),
                (len(agg_counts),           "Unique Classes"),
                (f"{avg_t:.2f}s",           "Avg Time/Image"),
            ])
            divider()

            if agg_counts:
                c1, c2 = st.columns(2, gap="medium")
                with c1:
                    st.markdown("**Overall Distribution**")
                    _plotly(chart_pie(agg_counts, theme), "batch_pie")
                with c2:
                    st.markdown("**Overall Counts**")
                    _plotly(chart_bar(agg_counts, theme), "batch_bar")

            divider()
            st.markdown("#### 🖼️ Per-Image Results")
            for res in batch:
                with st.expander(f"📷 {res['name']} — {sum(res['counts'].values())} objects · {res['elapsed']:.2f}s"):
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(res["image"], use_container_width=True, caption="Original")
                    with c2:
                        st.image(cv2.cvtColor(res["annotated"], cv2.COLOR_BGR2RGB),
                                 use_container_width=True, caption="Detected")
                    if res["records"]:
                        st.dataframe(
                            pd.DataFrame(res["records"])[["Class","Confidence","Bin","Type"]],
                            use_container_width=True, hide_index=True,
                        )
                    st.download_button(
                        f"📸 Download {res['name']}",
                        data=image_to_bytes(res["annotated"]),
                        file_name=f"detected_{res['name']}",
                        mime="image/png",
                        key=f"dl_batch_{res['name']}",
                    )

            if agg_records:
                divider()
                st.download_button(
                    "📄 Download Full Batch CSV",
                    data=pd.DataFrame(agg_records).to_csv(index=False),
                    file_name="batch_results.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

    # ── WEBCAM ────────────────────────────────────────────
    elif mode == "Webcam":
        st.markdown("#### 📷 Live Webcam Detection")
        st.info("ℹ️ Uses OpenCV. Uncheck **Start Webcam** to stop.")

        run = st.checkbox("▶️ Start Webcam", value=False)
        frame_win = st.empty()
        stats_win = st.empty()

        if run:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("❌ Cannot open webcam. Check your camera connection.")
            else:
                t_start    = time.perf_counter()
                frame_cnt  = 0
                while run:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Frame capture failed.")
                        break

                    results  = model(frame, conf=confidence, verbose=False)
                    ann      = results[0].plot()
                    names    = model.names
                    boxes    = results[0].boxes
                    counts: dict = {}
                    for box in boxes:
                        lbl = names[int(box.cls[0])]
                        if float(box.conf[0]) >= confidence:
                            counts[lbl] = counts.get(lbl, 0) + 1

                    frame_cnt += 1
                    fps = frame_cnt / max(time.perf_counter() - t_start, 1e-6)

                    cv2.putText(ann, f"FPS: {fps:.1f}",           (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,128), 2)
                    cv2.putText(ann, f"Objects: {sum(counts.values())}", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,128), 2)

                    frame_win.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                                    channels="RGB", use_container_width=True)
                    stats_win.markdown(
                        f"**FPS:** `{fps:.1f}` &nbsp;|&nbsp; "
                        f"**Frames:** `{frame_cnt}` &nbsp;|&nbsp; "
                        f"**Objects Now:** `{sum(counts.values())}`"
                    )
                cap.release()

# ══════════════════════════════════════════════════════════
#  TAB 2 — ANALYTICS
# ══════════════════════════════════════════════════════════
with tab_analytics:
    st.markdown("#### 📊 Session Analytics")

    if not st.session_state.history:
        st.info("🔍 Process some images first to see analytics.")
    else:
        # Aggregate session data once
        agg_counts: dict = {}
        agg_types:  dict = {}
        agg_bins:   dict = {}
        all_confs:  list = []

        for h in st.session_state.history:
            for r in h["records"]:
                agg_counts[r["Class"]] = agg_counts.get(r["Class"], 0) + 1
                agg_types[r["Type"]]   = agg_types.get(r["Type"],   0) + 1
                agg_bins[r["Bin"]]     = agg_bins.get(r["Bin"],     0) + 1
                all_confs.append(r["Confidence"])

        total = sum(agg_counts.values()) or 1
        top_class = max(agg_counts, key=agg_counts.get) if agg_counts else "—"
        avg_c = round(sum(all_confs) / len(all_confs), 3) if all_confs else 0

        metrics_row([
            (len(st.session_state.history), "Images Processed"),
            (st.session_state.total_detections, "Total Detections"),
            (avg_c,     "Avg Confidence"),
            (top_class, "Most Detected"),
        ])
        divider()

        # Row 1
        r1, r2 = st.columns(2, gap="medium")
        with r1:
            st.markdown("**Waste by Class**")
            _plotly(chart_pie(agg_counts, theme), "an_pie_class")
        with r2:
            st.markdown("**Waste by Type**")
            _plotly(chart_bar(agg_types, theme, TYPE_COLORS), "an_bar_type")

        divider()

        # Row 2
        r3, r4 = st.columns(2, gap="medium")
        with r3:
            st.markdown("**Bin Distribution**")
            _plotly(chart_pie(agg_bins, theme), "an_pie_bin")
        with r4:
            st.markdown("**Confidence Distribution**")
            _plotly(chart_histogram(all_confs, theme), "an_hist")

        divider()
        st.markdown("**📈 Detection Trend**")
        _plotly(chart_trend(st.session_state.history, theme), "an_trend")
        if len(st.session_state.history) < 2:
            st.caption("Process at least 2 images to see the trend.")

        divider()
        st.markdown("**♻️ Recyclability Breakdown**")
        recyclable = agg_types.get("Recyclable", 0)
        organic    = agg_types.get("Organic", 0)
        special    = agg_types.get("Hazardous", 0) + agg_types.get("E-Waste", 0)

        rb1, rb2, rb3 = st.columns(3)
        for col, val, label, color in [
            (rb1, recyclable, "♻️ Recyclable",        "#3b82f6"),
            (rb2, organic,    "🌱 Organic",             "#22c55e"),
            (rb3, special,    "⚠️ Special Disposal",   "#ef4444"),
        ]:
            pct = round(val * 100 / total)
            with col:
                metric_card(f"{pct}%", label, color)
                st.progress(pct / 100)

# ══════════════════════════════════════════════════════════
#  TAB 3 — HISTORY
# ══════════════════════════════════════════════════════════
with tab_history:
    st.markdown("#### 🗂️ Detection History")

    if not st.session_state.history:
        st.info("📭 No detections yet.")
    else:
        summary = [
            {
                "Time":          h["timestamp"],
                "File":          h["file"],
                "Objects":       h["total"],
                "Top Class":     max(h["classes"], key=h["classes"].get) if h["classes"] else "—",
                "Classes Found": ", ".join(h["classes"].keys()),
            }
            for h in st.session_state.history
        ]
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)
        divider()

        st.markdown("**Detailed Records**")
        for h in reversed(st.session_state.history):
            with st.expander(f"[{h['timestamp']}] {h['file']} — {h['total']} object(s)"):
                if h["records"]:
                    df_r = pd.DataFrame(h["records"])[["Class","Confidence","Bin","Type","Tip"]]
                    st.dataframe(df_r, use_container_width=True, hide_index=True)
                    _plotly(chart_bar(h["classes"], theme), f"hist_{h['timestamp']}_{h['file']}")
                else:
                    st.caption("No objects detected in this image.")

# ══════════════════════════════════════════════════════════
#  TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════
with tab_about:
    st.markdown(
        '<div class="waste-card"><h3 style="margin-top:0;">♻️ About SmartWaste AI</h3>'
        '<p>An AI-powered waste detection and classification system built with YOLOv8 and Streamlit, '
        'developed as a B.Tech Final Year Project (BTP) to contribute to smart city waste management.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2, gap="medium")
    with c1:
        st.markdown(
            '<div class="waste-card"><h4 style="margin-top:0;">🧠 Model Info</h4>'
            '<ul style="line-height:2.1;">'
            '<li>Architecture: <strong>YOLOv8n (Nano)</strong></li>'
            '<li>Dataset: Kaggle Garbage Detection</li>'
            '<li>Train / Val / Test: 5 000 / 1 000 / 500</li>'
            '<li>Epochs: 30 · Image size: 416 · Batch: 4</li>'
            '<li>Classes: 43 waste categories</li>'
            '</ul></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div class="waste-card"><h4 style="margin-top:0;">🗑️ Bin Guide</h4>'
            '<ul style="line-height:2.3;">'
            f'<li><span style="color:#3b82f6;">♻️ Blue Bin</span> — Plastic, Metal, Glass</li>'
            f'<li><span style="color:#f59e0b;">📄 Paper Bin</span> — Paper, Cardboard</li>'
            f'<li><span style="color:#22c55e;">🌱 Green Bin</span> — Organic, Food Waste</li>'
            f'<li><span style="color:#ef4444;">⚠️ Hazardous</span> — Batteries, Chemicals</li>'
            f'<li><span style="color:#8b5cf6;">💻 E-Waste</span> — Electronics, Cables</li>'
            f'<li><span style="color:#6b7280;">🚫 Landfill</span> — Non-recyclable waste</li>'
            '</ul></div>',
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="waste-card" style="margin-top:10px;">'
        '<h4 style="margin-top:0;">👨‍💻 Author</h4>'
        '<p style="margin:0;"><strong>Aman Kumar Gupta</strong> — B.Tech Final Year Project (BTP)</p>'
        '<p style="font-size:0.82rem;margin:6px 0 0;">Stack: Python · YOLOv8 · OpenCV · Streamlit · Plotly</p>'
        '</div>',
        unsafe_allow_html=True,
    )
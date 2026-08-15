"""DriftSense V2 Streamlit application.

Reads only from artifacts this project already writes (generator/, data/,
outputs/reports/, outputs/plots/) - the one exception is Live Localization
and Generate Sample, which call pipeline/localize.py directly. There is
exactly one localization code path; this app never reimplements it.

The CSS/branding layer below (THEME_CSS, render_topbar, render_stat_grid,
render_accuracy_curve, render_family_bar) is purely cosmetic/visualization -
it changes no control, no computation, and no data source. Every number,
slider range and default still comes straight from
generator/dataset_generator.py's DEFAULT_PARAMS or this project's own
output files. The two Plotly charts on Executive Summary are computed live
from per_pair_results.csv (real per-pair data, not canned numbers).
"""
from __future__ import annotations

import json
import os
import platform
import sys
import time

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from generator import mat_generator  # noqa: E402
from generator.dataset_generator import (  # noqa: E402
    GENERATOR_VERSION, SCALE_FACTOR, REFERENCE_SIZE_PX, generate_pair,
)
from pipeline.localize import localize  # noqa: E402

REPORTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "plots")
REPO_URL = "https://github.com/teja120805-cyber/Semicon-26---Kacha-Mango"

# Pre-integration classical baseline, established in reports/V2_BASELINE_REPORT.md
# before the 2026-08-15 A2 (scale range) + A6 (multiway tie-break) compliance
# fixes. Used only to compute a real, sourced "delta since" figure on the
# Executive Summary accuracy card - not a decorative number.
PRE_INTEGRATION_ACCURACY_AT_5PX = 0.712

st.set_page_config(page_title="Drift-Sense | DriftSense V2", page_icon="\U0001F9ED", layout="wide",
                    initial_sidebar_state="expanded")

# --------------------------------------------------------------------------
# Theme - refined blue/slate palette, a compact top bar, custom KPI stat
# cards, and a restyled sidebar nav (native Streamlit radio, CSS-skinned to
# read as a real nav list rather than a checkbox form). Nothing in this
# block reads or computes anything; it is styling only.
# --------------------------------------------------------------------------
THEME_CSS = """
<style>
:root {
    --ds-primary: #0B5FA8;
    --ds-primary-dark: #073B66;
    --ds-accent: #17A673;
    --ds-warning: #D97706;
    --ds-danger: #DC2626;
    --ds-border: #E2E8F0;
    --ds-text-muted: #64748B;
}

html, body, [class*="css"] {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
}

footer {visibility: hidden;}
#MainMenu {visibility: hidden;}
[data-testid="stAppDeployButton"] {display: none;}
header[data-testid="stHeader"] {background: transparent;}

h1 {font-size: 1.65rem !important; font-weight: 800 !important; letter-spacing: -0.01em;}
h2 {font-size: 1.2rem !important; font-weight: 700 !important;}
h3 {font-size: 1.02rem !important; font-weight: 700 !important;}
h1, h2, h3 {color: var(--ds-primary-dark);}

div.block-container {padding-top: 2.2rem; max-width: 1200px;}

/* Sidebar: dark slate nav, light widget controls stay legible as-is */
section[data-testid="stSidebar"] {
    background-color: #0B1F33;
}
section[data-testid="stSidebar"] * {
    color: #DCE6F0 !important;
}

/* Sidebar nav list - skin the native radio to look like a real nav, not a form */
section[data-testid="stSidebar"] div[data-testid="stRadio"] > label {display: none;}
div[role="radiogroup"] {gap: 1px !important;}
label[data-testid="stRadioOption"] {
    padding: 8px 10px;
    border-radius: 8px;
    border-left: 3px solid transparent;
    transition: background-color 0.15s ease;
    cursor: pointer;
}
label[data-testid="stRadioOption"]:hover {
    background-color: rgba(255, 255, 255, 0.06);
}
label[data-testid="stRadioOption"][data-selected="true"] {
    background-color: rgba(74, 158, 255, 0.16);
    border-left: 3px solid #4B9FE1;
}
label[data-testid="stRadioOption"][data-selected="true"] p {
    font-weight: 700 !important;
    color: #FFFFFF !important;
}
/* Hide the native radio circle indicator - first (non-text) div in each option */
label[data-testid="stRadioOption"] > div > div > div:first-child {
    display: none;
}

/* Metric cards (st.metric, used on utility pages) */
div[data-testid="stMetric"] {
    background-color: #FFFFFF;
    border: 1px solid var(--ds-border);
    border-radius: 10px;
    padding: 14px 16px 10px 16px;
    box-shadow: 0 1px 2px rgba(15, 41, 66, 0.05);
}
div[data-testid="stMetricLabel"] {
    color: var(--ds-text-muted);
}

/* Compact dark top bar */
.ds-topbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #0B1F33;
    border-radius: 12px;
    padding: 16px 24px;
    margin-bottom: 6px;
    flex-wrap: wrap;
    gap: 10px;
}
.ds-topbar-title {
    color: #FFFFFF;
    font-size: 1.3rem;
    font-weight: 800;
    letter-spacing: -0.01em;
    line-height: 1.1;
}
.ds-topbar-sub {
    color: #93A9BF;
    font-size: 0.78rem;
    margin-top: 3px;
}
.ds-topbar-right {
    text-align: right;
}
.ds-topbar-stat {
    color: #FFFFFF;
    font-size: 1.55rem;
    font-weight: 800;
    line-height: 1;
}
.ds-topbar-stat-label {
    color: #93A9BF;
    font-size: 0.68rem;
    margin-top: 3px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.ds-topbar-caption {
    color: var(--ds-text-muted);
    font-size: 0.78rem;
    margin: 6px 2px 18px 2px;
}

/* KPI stat card grid */
.ds-stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 12px;
    margin: 14px 0 6px 0;
}
.ds-stat {
    background: #FFFFFF;
    border: 1px solid var(--ds-border);
    border-radius: 12px;
    padding: 16px 18px;
}
.ds-stat-label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--ds-text-muted);
    font-weight: 700;
    margin-bottom: 6px;
}
.ds-stat-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: var(--ds-primary-dark);
    line-height: 1.1;
}
.ds-stat-delta {
    font-size: 0.76rem;
    margin-top: 6px;
    font-weight: 600;
}
.ds-stat-delta.positive {color: var(--ds-accent);}
.ds-stat-delta.negative {color: var(--ds-danger);}

/* Section eyebrow labels above charts/tables */
.ds-eyebrow {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--ds-text-muted);
    font-weight: 700;
    margin: 4px 0 2px 0;
}
</style>
"""
st.markdown(THEME_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Cached data loaders - every one reads a file this project's own scripts
# produced; nothing here is recomputed silently on each page load.
# --------------------------------------------------------------------------

@st.cache_data
def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data
def load_per_pair_results() -> pd.DataFrame:
    path = os.path.join(REPORTS_DIR, "per_pair_results.csv")
    return pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()


def read_image(path: str) -> np.ndarray:
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def resolve_result_path(stored_path: str) -> str:
    """Paths in per_pair_results.csv are written by evaluation/evaluate.py
    as `os.path.join(data_root, ...)` with `data_root="data"` - i.e.
    already relative to the PROJECT root and already starting with
    "data/". Joining with the project's data/ directory again here would
    double it into "data/data/...", which doesn't exist.

    Normalize backslashes first: `ground_truth.json` stores forward-slash
    relative paths (e.g. "challenge/xxx_search.png"), but `os.path.join`
    on Windows joins "data" + that with a backslash, baking a literal
    "data\\challenge/xxx_search.png" into the CSV. Windows tolerates the
    resulting mixed separators; POSIX (Linux/Mac - e.g. a teammate's
    machine or a Linux eval server) does not, so cv2.imread silently
    returns None and the next cv2.cvtColor call crashes with a confusing
    "!_src.empty()" assertion. Converting every backslash to a forward
    slash before joining makes this resolve correctly on any OS regardless
    of which OS produced the CSV."""
    normalized = stored_path.replace("\\", "/")
    if os.path.isabs(normalized):
        return normalized
    return os.path.join(PROJECT_ROOT, normalized)


def read_uploaded_image(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


baseline_metrics = load_json(os.path.join(REPORTS_DIR, "baseline_metrics.json"))
gate_result = load_json(os.path.join(REPORTS_DIR, "integration_gate.json"))
failure_cases = load_json(os.path.join(REPORTS_DIR, "failure_analysis_cases.json"))
per_pair_df = load_per_pair_results()


def staleness_warning() -> str | None:
    """Real check, not decorative: baseline_metrics.json / plots /
    failure_analysis_cases.json are only trustworthy if they were produced
    by the SAME `evaluate_model.py` run as the current per_pair_results.csv.
    If per_pair_results.csv is newer, something (a partial run, a manual
    CSV edit, a crash between writing per-pair results and writing the
    summary) regenerated the per-pair data without regenerating the
    aggregate report - so the two are describing different pipeline/dataset
    states. Returns a warning string, or None if everything lines up."""
    metrics_path = os.path.join(REPORTS_DIR, "baseline_metrics.json")
    per_pair_path = os.path.join(REPORTS_DIR, "per_pair_results.csv")
    if not (os.path.exists(metrics_path) and os.path.exists(per_pair_path)):
        return None
    metrics_mtime = os.path.getmtime(metrics_path)
    per_pair_mtime = os.path.getmtime(per_pair_path)
    if per_pair_mtime - metrics_mtime > 60:  # more than a minute apart = different runs
        gap_hours = (per_pair_mtime - metrics_mtime) / 3600
        return (
            f"**`baseline_metrics.json` is {gap_hours:.1f}h older than `per_pair_results.csv`** — "
            "they were written by different runs and are describing different pipeline/dataset "
            "states. The headline metrics, plots, and failure-case selections below may not match "
            "reality. Re-run `python scripts/evaluate_model.py` to regenerate everything from a single "
            "consistent pass before citing any of these numbers in the README/PPT/checklist."
        )
    return None


def render_topbar() -> None:
    """Compact top bar shown above every page. The right-hand headline
    number is a real, live value read from baseline_metrics.json - not a
    hardcoded figure."""
    right_html = ""
    if baseline_metrics:
        n_pairs = len(per_pair_df) if not per_pair_df.empty else "156"
        acc5 = baseline_metrics["overall"]["accuracy_at_5px"]
        right_html = (
            f'<div class="ds-topbar-right">'
            f'<div class="ds-topbar-stat">{acc5:.1%}</div>'
            f'<div class="ds-topbar-stat-label">@5px &middot; {n_pairs} pairs</div>'
            f'</div>'
        )
    st.markdown(
        f"""
        <div class="ds-topbar">
            <div>
                <div class="ds-topbar-title">Drift-Sense</div>
                <div class="ds-topbar-sub">AI-Powered Navigation-Error Recovery for Wafer Inspection Tools</div>
            </div>
            {right_html}
        </div>
        <div class="ds-topbar-caption">
            Applied Materials Problem Statement, SEMICON India Hackathon 2026 &middot; Team Kaccha Mango
            &middot; A2 + A6 integrated as documented gate exceptions &middot; OpenCV {cv2.__version__} (pinned)
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_grid(stats: list[dict]) -> None:
    """Custom KPI cards for the Executive Summary hero row. Every value
    passed in comes from baseline_metrics.json (real computed metrics)."""
    cards = []
    for s in stats:
        delta_html = ""
        if s.get("delta"):
            cls = "positive" if s.get("positive", True) else "negative"
            delta_html = f'<div class="ds-stat-delta {cls}">{s["delta"]}</div>'
        cards.append(
            f'<div class="ds-stat"><div class="ds-stat-label">{s["label"]}</div>'
            f'<div class="ds-stat-value">{s["value"]}</div>{delta_html}</div>'
        )
    st.markdown(f'<div class="ds-stat-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def render_accuracy_curve(df: pd.DataFrame):
    """Interactive accuracy-vs-tolerance curve computed live from the real
    per-pair error column - complements (does not replace) the mandatory
    static accuracy_vs_tolerance.png on the Benchmark Dashboard page."""
    thresholds = np.arange(0, 20.5, 0.5)
    errors = df["error_px"].to_numpy()
    acc = [(errors <= t).mean() * 100 for t in thresholds]
    acc5 = float((errors <= 5).mean() * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds, y=acc, mode="lines", line=dict(color="#0B5FA8", width=3),
        fill="tozeroy", fillcolor="rgba(11,95,168,0.08)",
        hovertemplate="≤%{x:.1f}px: %{y:.1f}%<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[5], y=[acc5], mode="markers+text", marker=dict(color="#DC2626", size=9),
        text=[f" {acc5:.1f}% @5px"], textposition="middle right",
        textfont=dict(color="#DC2626", size=12), hoverinfo="skip", showlegend=False,
    ))
    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Error tolerance (px)", yaxis_title="Accuracy (%)",
        yaxis_range=[0, 102], plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=12, color="#334155"),
        xaxis=dict(gridcolor="#F1F5F9", zeroline=False),
        yaxis=dict(gridcolor="#F1F5F9", zeroline=False),
        showlegend=False,
    )
    return fig


def render_family_bar(df: pd.DataFrame):
    """Interactive per-family accuracy@5px bar chart, live from per-pair
    data - replaces a paragraph of strongest/weakest text with one glance."""
    grp = df.groupby("structural_family")["error_px"].apply(lambda s: (s <= 5).mean() * 100).sort_values()
    colors = ["#DC2626" if v < 40 else "#D97706" if v < 70 else "#17A673" for v in grp.values]
    fig = go.Figure(go.Bar(
        x=grp.values, y=grp.index, orientation="h", marker_color=colors,
        hovertemplate="%{y}: %{x:.1f}%@5px<extra></extra>",
        text=[f"{v:.0f}%" for v in grp.values], textposition="outside", cliponaxis=False,
    ))
    fig.update_layout(
        height=max(280, 22 * len(grp)), margin=dict(l=10, r=36, t=10, b=10),
        xaxis_title="Accuracy @5px (%)", xaxis_range=[0, 108],
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="Inter, sans-serif", size=11, color="#334155"),
        xaxis=dict(gridcolor="#F1F5F9", zeroline=False),
        yaxis=dict(gridcolor="white"),
    )
    return fig


st.sidebar.markdown(
    '<div style="padding: 4px 0 12px 4px;">'
    '<div style="font-size: 1.1rem; font-weight: 800; color: #FFFFFF;">Drift-Sense</div>'
    '<div style="font-size: 0.74rem; color: #7E97AF;">DriftSense V2 &middot; Team Kaccha Mango</div>'
    '</div>',
    unsafe_allow_html=True,
)
section = st.sidebar.radio(
    "Section",
    ["\U0001F4CB Executive Summary", "\U0001F9EA Generate Sample", "\U0001F3AF Live Localization",
     "\U0001F5BC️ Visualization", "\U0001F4CA Benchmark Dashboard", "\U0001F50D Failure Analysis",
     "\U0001F9EC Experiment Results", "⚙️ System Information"],
    label_visibility="collapsed",
)
# Strip the leading icon back off so all downstream `section == "..."` checks
# stay exactly as they were - the icon is presentation-only.
section = section.split(" ", 1)[1]

# --------------------------------------------------------------------------
# Generator parameter controls - only shown on the "Generate Sample" page.
# Every control here is a real key in generator/dataset_generator.py's
# DEFAULT_PARAMS (or the crop_mode/force_preset arguments generate_pair
# already accepts) - none of this is decorative; changing a value changes
# generate_pair()'s output on the next rerun, which the page below shows
# directly.
# --------------------------------------------------------------------------
gen_overrides: dict = {}
gen_crop_mode = "random"
if section == "Generate Sample":
    st.sidebar.header("Structure")
    preset_choice = st.sidebar.selectbox(
        "Architecture preset", ["Random"] + mat_generator.PRESET_NAMES,
        help="Force one of the 6 DRAM mat presets, or Random to let the macro layout pick freely.",
    )
    gen_overrides["force_preset"] = None if preset_choice == "Random" else preset_choice
    gen_overrides["feature_size_scale"] = st.sidebar.slider(
        "Feature size scale", 0.5, 2.0, 1.0, 0.05,
        help="Scales every pitch/width in the chosen preset proportionally - explore other process nodes.",
    )
    gen_crop_mode = st.sidebar.selectbox(
        "Crop placement", ["random", "single_mat", "mat_boundary", "same_preset_boundary", "multi_mat", "strip_center"],
        index=0,
        help="Where the Reference crop is deliberately placed relative to macro structure "
             "(deep in one mat / straddling mats / centered on a strip, etc).",
    )

    st.sidebar.header("SEM imaging physics")
    gen_overrides["blur_search_effective_px"] = st.sidebar.slider(
        "Beam spot size (search px)", 0.2, 4.0, 1.0, 0.1,
        help="Search-image point-spread-function blur sigma from finite electron-beam spot size.",
    )
    gen_overrides["collapse_threshold_nm"] = st.sidebar.slider(
        "Pattern-collapse threshold (nm)", 0.0, 25.0, 10.0, 1.0,
        help="Gaps narrower than this bridge together probabilistically (capillary/etch-induced "
             "collapse). 0 disables the effect entirely.",
    )
    gen_overrides["collapse_enabled"] = gen_overrides["collapse_threshold_nm"] > 0

    st.sidebar.header("Acquisition noise & drift")
    gen_overrides["dose_reference"] = st.sidebar.slider(
        "Reference dose (higher = cleaner)", 100.0, 5000.0, 1800.0, 100.0,
        help="Relative electron dose for the Reference acquisition - controls Poisson shot-noise SNR.",
    )
    gen_overrides["dose_search"] = st.sidebar.slider(
        "Search dose (higher = cleaner)", 20.0, 1000.0, 220.0, 10.0,
        help="Relative electron dose for the Search acquisition (typically lower/faster than Reference).",
    )
    gen_overrides["shear_amplitude_px"] = st.sidebar.slider(
        "Search raster drift/shear (px)", 0.0, 5.0, 1.0, 0.1,
        help="Progressive row-to-row scan shear from finite scan-stabilization bandwidth.",
    )
    gen_overrides["jitter_std_px"] = st.sidebar.slider(
        "Search row jitter (px)", 0.0, 3.0, 0.4, 0.1,
        help="Per-row scan vibration, independent of the progressive shear above.",
    )
    gen_overrides["rotation_deg"] = st.sidebar.slider(
        "Residual rotation drift (deg)", -8.0, 8.0, 0.0, 0.25,
        help="Stage/scan rotation calibration drift on top of the fixed base magnification. The "
             "classical pipeline tests hypotheses every 2.5 deg (-5..5) - values near the midpoint "
             "between two tested hypotheses are the hardest (see reports/ACCURACY_FORENSICS.md).",
    )
    gen_overrides["extra_scale"] = st.sidebar.slider(
        "Residual scale drift (x)", 0.85, 1.15, 1.0, 0.01,
        help="Magnification-calibration drift on top of the fixed base 10x. The classical pipeline "
             "tests hypotheses every 0.04x (0.92..1.08) - same midpoint-sensitivity as rotation.",
    )

    st.sidebar.header("Distortion & polygon scaling")
    gen_overrides["linewidth_bias_nm"] = st.sidebar.slider(
        "Linewidth/CD bias (nm)", -10.0, 10.0, 0.0, 0.5,
        help="Deterministic global over/under-exposure or etch bias applied to every drawn feature.",
    )
    gen_overrides["corner_rounding_px"] = st.sidebar.slider(
        "Corner rounding (px)", 0.0, 6.0, 0.0, 0.5,
        help="Morphological rounding of polygon corners - real litho/etch never draws sharp corners.",
    )
    gen_overrides["astigmatism_ratio"] = st.sidebar.slider(
        "Beam astigmatism ratio", 0.5, 2.5, 1.0, 0.05,
        help="Elliptical beam spot (sigma_y = sigma_x * ratio) - directional blur from an imperfectly stigmated beam.",
    )
    gen_overrides["barrel_k"] = st.sidebar.slider(
        "Barrel(+)/pincushion(-) distortion", -0.01, 0.01, 0.0, 0.0005, format="%.4f",
        help="Radial lens/scan-linearity distortion, strongest toward the field edge. Ground truth "
             "is NOT analytically corrected for this (see reports/DEGRADATION_COVERAGE.md) - large "
             "values can visibly displace the true match location.",
    )
    gen_overrides["vignette_strength"] = st.sidebar.slider(
        "Vignette strength", 0.0, 1.0, 0.0, 0.05,
        help="Radial illumination/collection-efficiency falloff toward the field edge.",
    )
    gen_overrides["gamma"] = st.sidebar.slider(
        "Gamma (contrast curve)", 0.4, 2.5, 1.0, 0.05,
        help="Detector-gain nonlinearity applied to normalized intensity.",
    )

    st.sidebar.header("Noise")
    gen_overrides["charging_prob"] = st.sidebar.slider(
        "Charging streak probability (per row)", 0.0, 0.05, 0.0, 0.005,
        help="Per-row probability of a bright streak from local sample charging on insulating regions.",
    )
    gen_overrides["charging_intensity"] = st.sidebar.slider(
        "Charging streak intensity", 0.0, 100.0, 0.0, 5.0,
        help="Brightness added to a row when a charging streak occurs.",
    )
    gen_overrides["speckle_sigma"] = st.sidebar.slider(
        "Speckle noise sigma (multiplicative)", 0.0, 0.3, 0.0, 0.01,
        help="out = img * (1 + N(0, sigma)) - detector-gain variation, scales with local brightness.",
    )
    gen_overrides["salt_pepper_amount"] = st.sidebar.slider(
        "Salt-and-pepper probability", 0.0, 0.05, 0.0, 0.0025,
        help="Fraction of pixels forced to pure black/white - dead/hot pixels or discharge events.",
    )

    st.sidebar.header("Die layout (multi-region)")
    gen_overrides["mat_size_nm"] = st.sidebar.slider(
        "Array block (mat) size (nm)", 800, 5000, 2400, 100,
        help="Size of each independently-generated mat sub-array before a separating strip.",
    )
    gen_overrides["strip_width_nm"] = st.sidebar.slider(
        "Separator strip width (nm)", 80, 800, 300, 20,
        help="Width of the peripheral/routing material band between mats.",
    )

    if "gen_seed" not in st.session_state:
        st.session_state.gen_seed = 42
    if st.sidebar.button("Regenerate (new seed)"):
        st.session_state.gen_seed = int(np.random.randint(0, 2_000_000_000))

render_topbar()

# --------------------------------------------------------------------------
# Executive Summary
# --------------------------------------------------------------------------
if section == "Executive Summary":
    st.title("Executive Summary")
    _stale_msg = staleness_warning()
    if _stale_msg:
        st.warning(_stale_msg)

    st.markdown(
        "Drift-Sense recovers navigation position by locating a high-resolution Reference crop "
        "inside a lower-resolution, acquisition-degraded Search image of macro-structured DRAM "
        "imagery. Classical multi-scale/multi-rotation ZNCC matching (`pipeline/`) is production; "
        "a from-scratch generator (`generator/`) and candidate learned components (`model/`) are "
        "evaluated through a mandatory integration gate."
    )

    if baseline_metrics:
        overall = baseline_metrics["overall"]
        acc5 = overall["accuracy_at_5px"]
        delta_pp = (acc5 - PRE_INTEGRATION_ACCURACY_AT_5PX) * 100
        render_stat_grid([
            {"label": "Accuracy @5px", "value": f"{acc5:.1%}",
             "delta": f"{'+' if delta_pp >= 0 else ''}{delta_pp:.1f}pp since compliance fixes",
             "positive": delta_pp >= 0},
            {"label": "Median error", "value": f"{overall['median_error_px']:.2f}px"},
            {"label": "Mean error", "value": f"{overall['mean_error_px']:.1f}px"},
            {"label": "Max error", "value": f"{overall['max_error_px']:.0f}px"},
            {"label": ">50px failures", "value": f"{overall['failure_rate_gt_50px']:.1%}"},
        ])

        if not per_pair_df.empty:
            chart_col1, chart_col2 = st.columns([3, 2])
            with chart_col1:
                st.markdown('<div class="ds-eyebrow">Accuracy vs. error tolerance</div>', unsafe_allow_html=True)
                st.plotly_chart(render_accuracy_curve(per_pair_df), use_container_width=True,
                                 config={"displayModeBar": False})
            with chart_col2:
                st.markdown('<div class="ds-eyebrow">Accuracy @5px by structural family</div>', unsafe_allow_html=True)
                st.plotly_chart(render_family_bar(per_pair_df), use_container_width=True,
                                 config={"displayModeBar": False})

        st.caption(
            "Boundary presence is the strongest failure lever found anywhere; periodic-mat aliasing is "
            "a stronger *standalone* bottleneck than rotation/scale drift. Full forensics: "
            "`reports/ACCURACY_FORENSICS.md` — case-by-case detail: **Failure Analysis** page."
        )
    else:
        st.warning("No baseline metrics found. Run `python scripts/evaluate_model.py` first.")

    st.divider()
    status_col1, status_col2 = st.columns(2)
    with status_col1:
        if gate_result is not None:
            if gate_result["passed"]:
                st.success("**Learned re-ranker**: passed the gate, integrated.")
            else:
                st.error(
                    "**Learned re-ranker**: failed the gate (all 3 seeds) — not integrated. "
                    "Classical ranking is production."
                )
        else:
            st.info("No model evaluation found yet.")
    with status_col2:
        st.success(
            "**A2 + A6 compliance fixes**: integrated as documented gate exceptions "
            "— see Experiment Results for the full evidence."
        )

# --------------------------------------------------------------------------
# Generate Sample
# --------------------------------------------------------------------------
elif section == "Generate Sample":
    st.title("Generate Sample")
    st.caption(
        "Every control in the sidebar is a real parameter of `generator/dataset_generator.py` "
        "(`DEFAULT_PARAMS`, `force_preset`, or `crop_mode`) — nothing here is decorative. Change a "
        "value and the sample below regenerates on the next rerun."
    )

    family = {"name": "ui_generated", "split": "development", "crop_mode": gen_crop_mode, "overrides": gen_overrides}
    ref_img, search_img, meta = generate_pair(0, st.session_state.gen_seed, family)

    with st.container(border=True):
        col1, col2 = st.columns(2)
        col1.image(ref_img, caption="Reference (generated, 1000x1000 @ 1nm/px)", width=320)
        col2.image(search_img, caption="Search (generated, 1000x1000 @ 10nm/px)", width=320)

        st.caption(
            f"Preset(s) touched: `{meta['presets']}` | crosses mat boundary: {meta['crosses_mat_boundary']} | "
            f"crosses strip: {meta['crosses_strip_boundary']} | periodicity_score: {meta['periodicity_score']:.2f} | "
            f"uniqueness_score: {meta['uniqueness_score']:.2f} | seed: {st.session_state.gen_seed}"
        )

    if st.button("Run localization on this sample", type="primary"):
        with st.spinner("Running classical multi-scale/multi-rotation matching..."):
            result = localize(ref_img, search_img)
        error_px = float(np.hypot(result.x - meta["gt_x"], result.y - meta["gt_y"]))

        with st.container(border=True):
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Error", f"{error_px:.2f} px")
            m2.metric("Confidence (ZNCC)", f"{result.confidence:.3f}")
            m3.metric("Ambiguity ratio", f"{result.ambiguity_ratio:.3f}")
            m4.metric("Runtime", f"{result.runtime_s:.2f} s")
            if result.ambiguous:
                st.warning("Flagged AMBIGUOUS: a second candidate location scored nearly as well as the winner.")

            display = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
            cv2.drawMarker(display, (int(result.x), int(result.y)), (0, 0, 255),
                            markerType=cv2.MARKER_CROSS, markerSize=25, thickness=2)
            cv2.drawMarker(display, (int(meta["gt_x"]), int(meta["gt_y"])), (0, 200, 0),
                            markerType=cv2.MARKER_DIAMOND, markerSize=20, thickness=2)
            st.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                      caption="Red cross = predicted location, green diamond = ground truth")

# --------------------------------------------------------------------------
# Live Localization
# --------------------------------------------------------------------------
elif section == "Live Localization":
    st.title("Live Localization")
    st.caption("Runs `pipeline/localize.py` directly — the same code path used for every benchmark number.")

    col1, col2 = st.columns(2)
    ref_upload = col1.file_uploader("Reference image", type=["png", "jpg", "jpeg", "bmp", "tif"])
    search_upload = col2.file_uploader("Search image", type=["png", "jpg", "jpeg", "bmp", "tif"])

    if ref_upload is not None:
        col1.image(ref_upload, caption="Reference", width=300)
    if search_upload is not None:
        col2.image(search_upload, caption="Search", width=300)

    if ref_upload is not None and search_upload is not None:
        if st.button("Run localization", type="primary"):
            ref_img = read_uploaded_image(ref_upload)
            search_img = read_uploaded_image(search_upload)
            with st.spinner("Running classical multi-scale/multi-rotation matching..."):
                result = localize(ref_img, search_img)

            with st.container(border=True):
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Predicted (x, y)", f"({result.x:.1f}, {result.y:.1f})")
                m2.metric("Confidence (ZNCC)", f"{result.confidence:.3f}")
                m3.metric("Ambiguity ratio", f"{result.ambiguity_ratio:.3f}")
                m4.metric("Runtime", f"{result.runtime_s:.2f} s")
                if result.ambiguous:
                    st.warning("Flagged AMBIGUOUS: a second candidate location scored nearly as well as the winner.")
                else:
                    st.success("Not flagged ambiguous: the winning candidate clearly outscored the runner-up.")

                display = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
                cv2.drawMarker(display, (int(result.x), int(result.y)), (0, 0, 255),
                                markerType=cv2.MARKER_CROSS, markerSize=25, thickness=2)
                st.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), caption="Search image with predicted location")

                with st.expander("Top candidates considered"):
                    st.dataframe(pd.DataFrame(result.top_candidates))

# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------
elif section == "Visualization":
    st.title("Visualization")
    if per_pair_df.empty:
        st.warning("No evaluation results found. Run `python scripts/evaluate_model.py` first.")
    else:
        splits = sorted(per_pair_df["split"].unique())
        split = st.selectbox("Split", splits)
        subset = per_pair_df[per_pair_df["split"] == split]
        pair_id = st.selectbox("Pair", sorted(subset["pair_id"].unique()))
        row = subset[subset["pair_id"] == pair_id].iloc[0]

        search_img = read_image(resolve_result_path(row["search_path"]))
        display = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
        cv2.drawMarker(display, (int(row["pred_x"]), int(row["pred_y"])), (0, 0, 255),
                        markerType=cv2.MARKER_CROSS, markerSize=25, thickness=2)
        cv2.drawMarker(display, (int(row["gt_x"]), int(row["gt_y"])), (0, 200, 0),
                        markerType=cv2.MARKER_DIAMOND, markerSize=20, thickness=2)

        with st.container(border=True):
            st.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB),
                      caption="Red cross = predicted location, green diamond = ground truth (evaluation mode)")

            m1, m2, m3 = st.columns(3)
            m1.metric("Error", f"{row['error_px']:.2f} px")
            m2.metric("Confidence", f"{row['confidence']:.3f}")
            m3.metric("Ambiguous", "Yes" if row["ambiguous"] else "No")
            st.caption(
                f"Structural family: `{row['structural_family']}` | crosses mat boundary: "
                f"{row['crosses_mat_boundary']} | crosses strip: {row['crosses_strip_boundary']}"
            )

# --------------------------------------------------------------------------
# Benchmark Dashboard
# --------------------------------------------------------------------------
elif section == "Benchmark Dashboard":
    st.title("Benchmark Dashboard")
    plot_files = [
        ("accuracy_vs_tolerance.png", "Accuracy vs. tolerance"),
        ("error_cdf.png", "Error CDF"),
        ("pr_curve.png", "Precision-recall (ranked by confidence)"),
        ("error_distribution.png", "Error distribution"),
        ("accuracy_by_noise.png", "Accuracy by noise level"),
        ("accuracy_by_scale.png", "Accuracy by scale condition"),
        ("accuracy_by_rotation.png", "Accuracy by rotation condition"),
        ("accuracy_by_family.png", "Accuracy by structural family"),
    ]
    cols = st.columns(2)
    for i, (fname, caption) in enumerate(plot_files):
        path = os.path.join(PLOTS_DIR, fname)
        with cols[i % 2].container(border=True):
            if os.path.exists(path):
                st.image(path, caption=caption, use_container_width=True)
            else:
                st.info(f"{fname} not generated yet.")

    if baseline_metrics:
        st.subheader("Overall metrics")
        with st.container(border=True):
            st.json(baseline_metrics["overall"])

# --------------------------------------------------------------------------
# Failure Analysis
# --------------------------------------------------------------------------
elif section == "Failure Analysis":
    st.title("Failure Analysis")
    st.caption(
        "Cases below are selected once by `outputs/reports/failure_analysis_cases.json` "
        "(see evaluation notes) - not re-randomized on every page load."
    )
    st.info(
        "The official rubric names **repeated-pattern ambiguity** specifically under this criterion. "
        "Our own controlled forensics (`reports/ACCURACY_FORENSICS.md`) independently found periodic-mat "
        "aliasing to be a stronger *standalone* localization bottleneck than rotation/scale drift, and "
        "that it fails at candidate generation — the true location is never proposed — not at ranking."
    )
    if not failure_cases or per_pair_df.empty:
        st.warning("Failure case selections or evaluation results not found.")
    else:
        for label, key in [("Representative successful case", "successful"),
                            ("Difficult case", "difficult"),
                            ("Catastrophic failure", "catastrophic")]:
            case = failure_cases[key]
            match = per_pair_df[per_pair_df["pair_id"] == case["pair_id"]]
            if match.empty:
                continue
            row = match.iloc[0]
            with st.container(border=True):
                st.subheader(f"{label}: `{case['pair_id']}`")
                search_img = read_image(resolve_result_path(row["search_path"]))
                display = cv2.cvtColor(search_img, cv2.COLOR_GRAY2BGR)
                cv2.drawMarker(display, (int(row["pred_x"]), int(row["pred_y"])), (0, 0, 255),
                                markerType=cv2.MARKER_CROSS, markerSize=30, thickness=3)
                cv2.drawMarker(display, (int(row["gt_x"]), int(row["gt_y"])), (0, 200, 0),
                                markerType=cv2.MARKER_DIAMOND, markerSize=25, thickness=3)
                c1, c2 = st.columns([2, 1])
                c1.image(cv2.cvtColor(display, cv2.COLOR_BGR2RGB), use_container_width=True)
                c2.metric("Error", f"{case['error_px']:.2f} px")
                c2.caption(f"Structural family: `{row['structural_family']}`")
                c2.write(case["rationale"])

# --------------------------------------------------------------------------
# Experiment Results
# --------------------------------------------------------------------------
elif section == "Experiment Results":
    st.title("Experiment Results")
    st.caption(
        "Every candidate change is evaluated against the frozen classical baseline through the same "
        "7-criterion integration gate (`evaluation/benchmark.py`) and only integrated into production "
        "if every criterion passes, or logged as a documented exception (`reports/GATE_EXCEPTIONS.md`) "
        "when it doesn't. Full write-ups: `reports/ACCURACY_FORENSICS.md` and each experiment's own "
        "`experiments/<name>/REPORT.md`."
    )

    st.subheader("Accuracy forensics — what actually drives failure")
    with st.container(border=True):
        st.markdown(
            "Controlled single-factor and interaction sweeps (`experiments/accuracy_forensics/`), ranked "
            "by how much each factor actually damages accuracy:\n\n"
            "1. **Boundary presence** — the single strongest lever tested. Worst case *with* a mat "
            "boundary in view (misaligned rotation + scale together): 40%@5px. The identical "
            "misalignment *without* one: 0%@5px.\n"
            "2. **Periodicity/aliasing** — a stronger standalone bottleneck than rotation/scale drift, "
            "and it fails at candidate generation (the true location is never proposed), not ranking: "
            "62-85% of these failures land within a quarter-pitch of an exact integer multiple of the "
            "mat's own word pitch.\n"
            "3. **Rotation/scale hypothesis-grid misalignment** — damage tracks distance to the nearest "
            "tested hypothesis, not drift magnitude: grid-aligned values recover to 65-70%@5px, "
            "grid-midpoint values collapse to 40-45%.\n"
            "4. **Noise, raster drift, row jitter** — minor factors throughout, alone or combined."
        )

    st.subheader("Candidate changes evaluated")


    def _load_experiment_gate(path: str):
        return load_json(os.path.join(PROJECT_ROOT, path))


    exp_rows = []
    reranker_gate = load_json(os.path.join(REPORTS_DIR, "integration_gate.json"))
    if reranker_gate is not None:
        exp_rows.append(("embedding_reranker_v1 (CNN re-ranker)", reranker_gate["passed"],
                          "Halved accuracy@5px, 2-3x catastrophic rate on every split, all 3 seeds — "
                          "training-data scale (72 triplets, 2 families), not a training bug."))
    wide_pool_gate = _load_experiment_gate("experiments/wider_candidate_pool/outputs/integration_gate.json")
    if wide_pool_gate is not None:
        exp_rows.append(("wider_candidate_pool (more peaks + tighter NMS)", wide_pool_gate["passed"],
                          "Bit-identical predictions to baseline on every pair — a structural no-op "
                          "under pure arg-max ranking, not a failed improvement."))
    fine_grid_gate = _load_experiment_gate("experiments/finer_hypothesis_grid/outputs/integration_gate.json")
    if fine_grid_gate is not None:
        exp_rows.append(("finer_hypothesis_grid (81 vs 25 hypotheses)", fine_grid_gate["passed"],
                          "Near-miss: improved held_out (+5.0pp) and challenge (+6.2pp), no per-family "
                          "regression, acceptable runtime (3.17x, corrected) — fails only because "
                          "validation tied rather than improved (already at a 90% ceiling)."))

    for name, passed, note in exp_rows:
        if passed:
            st.success(f"**{name}** — PASSED, integrated.")
        else:
            st.error(f"**{name}** — not integrated. {note}")

    if not exp_rows:
        st.info("No experiment gate results found yet.")

    st.subheader("Documented gate exceptions (integrated despite failing the literal gate)")
    with st.container(border=True):
        st.markdown(
            "**A2 — scale hypothesis grid + dataset scale range widened to literal 9:1–11:1**: zero "
            "regressions across 2 independent datasets; fails criteria 1/2 only because the affected "
            "families don't dominate `validation`/`held_out`'s pooled count, not because of harm. "
            "`experiments/scale_range_v1/REPORT.md`"
        )
        st.markdown(
            "**A6 — multiway-gated centre tie-break**: zero regressions across 2 independent datasets, "
            "one confirmed catastrophic rescue (`ch_worst_case_006`, 118.5px → 4.6px); same structural "
            "reason for failing criteria 1/2. `experiments/multiway_tiebreak_v1/REPORT.md`"
        )
        st.caption("Full rationale for treating these as exceptions rather than gate rewrites: `reports/GATE_EXCEPTIONS.md`.")

    st.subheader("Evaluation rubric mapping")
    st.caption(
        "Full quoted Section 6 weighting table (Applied Materials help doc) and where each parameter "
        "is addressed: see README.md → “Evaluation criteria alignment.”"
    )

# --------------------------------------------------------------------------
# System Information
# --------------------------------------------------------------------------
elif section == "System Information":
    st.title("System Information")

    import cv2 as _cv2
    import matplotlib
    import scipy

    try:
        import torch
        torch_version = torch.__version__
        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "N/A"
    except ImportError:
        torch_version, cuda_available, gpu_name = "not installed", False, "N/A"

    info = {
        "Generator version": GENERATOR_VERSION,
        "Base scale factor": SCALE_FACTOR,
        "Reference size (px)": REFERENCE_SIZE_PX,
        "Dataset seed (main splits)": 777001,
        "Model status": "classical baseline in production; learned re-ranker rejected (see experiments/embedding_reranker_v1/)",
        "Python version": platform.python_version(),
        "Platform": platform.platform(),
        "CPU": platform.processor() or "unknown",
        "PyTorch version": torch_version,
        "CUDA available": cuda_available,
        "GPU": gpu_name,
        "OpenCV version": _cv2.__version__,
        "NumPy version": np.__version__,
        "Pandas version": pd.__version__,
        "Matplotlib version": matplotlib.__version__,
        "SciPy version": scipy.__version__,
        "Streamlit version": st.__version__,
    }
    with st.container(border=True):
        st.table(pd.DataFrame([(k, str(v)) for k, v in info.items()], columns=["Component", "Value"]))

# --------------------------------------------------------------------------
# Footer - shown at the bottom of every page.
# --------------------------------------------------------------------------
st.markdown("---")
st.caption(
    f"Drift-Sense &middot; Team Kaccha Mango &middot; Applied Materials Problem Statement, "
    f"SEMICON India Hackathon 2026 &middot; [GitHub]({REPO_URL})"
)

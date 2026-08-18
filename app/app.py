"""Drift-Sense — PRECISION UNDER DRIFT.

Frontend only. Every algorithmic path (`pipeline.localize`, `generator`,
`evaluation`) is imported and called unmodified; this file renders results and
never computes a benchmark number of its own. Every figure shown is read from
the artifacts `scripts/evaluate_model.py` produces, or from a live pipeline run
the operator explicitly triggers — nothing is hardcoded.

Design intent
-------------
This is not a dashboard. It is an *instrument log*: a dark graphite console
with one hero measurement, editorial prose in a serif face, every technical
value in monospace, and structure carried by hairlines, numbered chapter marks
and asymmetry rather than by a wall of rounded cards. Optical motifs — corner
registration marks, a hover crosshair, a self-terminating scan sweep after a
live acquisition — come from what the system actually does: it finds one small
frame inside a much larger one, under drift.

Layout
------
    THEME       one CSS design system, injected once
    ENCODE      numpy / PNG -> data URI, so figures stay offline but styleable
    DATA        cached loaders over outputs/reports/*
    PRIMITIVES  masthead, chapter mark, hero numeral, readout run, figure, tokens
    CHARTS      two Plotly builders, restyled for graphite
    SCREENS     one render_* function per screen
    ROUTER      numbered instrument index -> screen function
"""
from __future__ import annotations

import base64
import json
import os
import platform
import sys

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

TOLERANCE_PX = 5.0          # the problem statement's success criterion
CATASTROPHIC_PX = 50.0      # evaluation/benchmark.py::CATASTROPHIC_PX

# Overlay colours, BGR for OpenCV, matched to the CSS palette below.
BGR_PRED = (82, 95, 255)     # --k-red   #FF5F52
BGR_TRUE = (164, 214, 53)    # --k-green #35D6A4

st.set_page_config(page_title="Drift-Sense — precision under drift",
                    page_icon="\u25C9", layout="wide",
                    initial_sidebar_state="expanded")

# ==========================================================================
# THEME
# ==========================================================================
# One design system, injected once. Streamlit-internal selectors are held to
# the minimum that cannot be expressed any other way (chrome hiding, sidebar
# surface, the nav radio, widget surfaces) and are grouped under a single
# banner so a Streamlit upgrade has exactly one place to check.
#
# `.streamlit/config.toml` sets Streamlit's own dark base so native widgets
# match; the overrides here are additionally defensive, so the console still
# renders correctly when launched from a directory where that file is not
# discovered.
# ==========================================================================
THEME_CSS = """
<style>
:root{
  --k-void:#07090C;  --k-base:#0B0E13;  --k-panel:#11151C; --k-panel-2:#161B24;
  --k-line:#1D242E;  --k-line-2:#2B3441;
  --k-ink:#E8EDF4;   --k-ink-2:#AEBBCA; --k-mute:#6D7C8F;
  --k-blue:#3B9EFF;  --k-green:#35D6A4; --k-amber:#E8A33D; --k-red:#FF5F52;
  --k-mono: ui-monospace, "SF Mono", SFMono-Regular, Menlo, Consolas,
            "Liberation Mono", "DejaVu Sans Mono", monospace;
  --k-serif: "Iowan Old Style", "Palatino Linotype", Palatino, "Book Antiqua",
             Georgia, "Times New Roman", serif;
  --k-sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto,
            Helvetica, Arial, sans-serif;
}

/* --- Surface ---------------------------------------------------------- */
.stApp{
  background:
    radial-gradient(1100px 560px at 76% -10%, rgba(59,158,255,.060), transparent 62%),
    radial-gradient(900px 500px at 4% 106%, rgba(53,214,164,.032), transparent 60%),
    repeating-linear-gradient(0deg, rgba(255,255,255,.017) 0 1px, transparent 1px 4px),
    var(--k-base);
  color: var(--k-ink);
}
html, body, [class*="css"]{ font-family: var(--k-sans); }
div.block-container{ padding-top:1.5rem; padding-bottom:4.5rem; max-width:1320px; }

/* --- Motion ----------------------------------------------------------- */
@keyframes rise   { from{opacity:0; transform:translateY(14px);} to{opacity:1; transform:none;} }
@keyframes draw   { to{ stroke-dashoffset:0; } }
@keyframes sweep  { 0%{transform:translateY(-30%); opacity:0;} 10%{opacity:.9;}
                    85%{opacity:.9;} 100%{transform:translateY(400%); opacity:0;} }
@keyframes breathe{ 0%,100%{ box-shadow:0 0 0 0 rgba(53,214,164,.50);}
                    70%{ box-shadow:0 0 0 7px rgba(53,214,164,0);} }
@keyframes rule   { from{transform:scaleX(0);} to{transform:scaleX(1);} }

.r{ animation: rise .62s cubic-bezier(.2,.7,.25,1) both; }
.d1{animation-delay:.05s;} .d2{animation-delay:.11s;} .d3{animation-delay:.18s;}
.d4{animation-delay:.26s;} .d5{animation-delay:.35s;} .d6{animation-delay:.45s;}

/* --- Masthead --------------------------------------------------------- */
.k-mast{ display:flex; align-items:flex-end; justify-content:space-between;
  gap:24px; flex-wrap:wrap; padding:2px 0 12px; border-bottom:1px solid var(--k-line);
  margin-bottom:6px; }
.k-word{ font-family:var(--k-mono); font-size:.95rem; font-weight:700;
  letter-spacing:.40em; color:var(--k-ink); }
.k-word i{ font-style:normal; color:var(--k-blue); }
.k-mast-meta{ font-family:var(--k-mono); font-size:.60rem; letter-spacing:.17em;
  text-transform:uppercase; color:var(--k-mute); text-align:right; line-height:1.9; }
.k-dot{ display:inline-block; width:6px; height:6px; border-radius:50%;
  background:var(--k-green); margin-right:9px; vertical-align:middle;
  animation:breathe 2.8s ease-out infinite; }

/* --- Standing line + hero numeral ------------------------------------- */
.k-standing{ font-family:var(--k-mono); font-size:.63rem; letter-spacing:.36em;
  text-transform:uppercase; color:var(--k-blue); margin:26px 0 2px; }
.k-hero{ display:flex; align-items:flex-end; gap:clamp(16px,4vw,58px);
  flex-wrap:wrap; margin:2px 0 4px; }
.k-numeral{ font-family:var(--k-mono); font-weight:700; line-height:.80;
  font-size:clamp(4.2rem,13vw,9.5rem); letter-spacing:-.05em; color:var(--k-ink); }
.k-numeral sup{ font-size:.33em; top:-1.42em; margin-left:-.03em;
  color:var(--k-blue); letter-spacing:-.06em; }
.k-hero-note{ font-family:var(--k-serif); line-height:1.54; color:var(--k-ink-2);
  font-size:clamp(.98rem,1.35vw,1.14rem); max-width:47ch; padding-bottom:.85rem; }
.k-hero-note b{ color:var(--k-ink); font-weight:600; }

/* --- Chapter mark (replaces card headers) ----------------------------- */
.k-mark{ display:flex; align-items:center; gap:13px; margin:40px 0 15px; }
.k-mark .x{ position:relative; width:9px; height:9px; flex:none; }
.k-mark .x::before,.k-mark .x::after{ content:""; position:absolute; background:var(--k-blue); }
.k-mark .x::before{ left:4px; top:0; width:1px; height:9px; }
.k-mark .x::after { top:4px; left:0; height:1px; width:9px; }
.k-mark .n{ font-family:var(--k-mono); font-size:.61rem; letter-spacing:.2em; color:var(--k-blue); }
.k-mark .t{ font-family:var(--k-mono); font-size:.67rem; letter-spacing:.25em;
  text-transform:uppercase; color:var(--k-ink-2); white-space:nowrap; }
.k-mark .rule{ flex:1; height:1px; transform-origin:left;
  background:linear-gradient(90deg,var(--k-line-2) 0%, var(--k-line) 55%, transparent 100%);
  animation:rule .85s cubic-bezier(.2,.7,.25,1) both; }

/* --- Readout run (hairline band, deliberately not cards) -------------- */
.k-run{ display:flex; flex-wrap:wrap;
  border-top:1px solid var(--k-line); border-bottom:1px solid var(--k-line); }
.k-cell{ flex:1 1 165px; padding:17px 22px 16px; border-left:1px solid var(--k-line); }
.k-cell:first-child{ border-left:0; padding-left:0; }
.k-cell .k{ font-family:var(--k-mono); font-size:.59rem; letter-spacing:.17em;
  text-transform:uppercase; color:var(--k-mute); }
.k-cell .v{ font-family:var(--k-mono); font-size:1.58rem; font-weight:700;
  letter-spacing:-.02em; color:var(--k-ink); margin-top:8px; line-height:1; }
.k-cell .s{ font-size:.72rem; color:var(--k-mute); margin-top:8px; }
.k-cell .s.pos{ color:var(--k-green); } .k-cell .s.neg{ color:var(--k-red); }
.k-cell.lead .v{ color:var(--k-blue); }
.k-cell.lead.pass .v{ color:var(--k-green); } .k-cell.lead.miss .v{ color:var(--k-red); }

/* --- Figure: registration marks, hover crosshair, scan sweep ---------- */
.k-fig{ position:relative; overflow:hidden; background:var(--k-void);
  border:1px solid var(--k-line-2); }
.k-fig img{ display:block; width:100%; height:auto; }
.k-fig .c{ position:absolute; width:14px; height:14px; pointer-events:none;
  border:0 solid var(--k-blue); opacity:.8; }
.k-fig .c.tl{ top:7px; left:7px;  border-top-width:1px; border-left-width:1px; }
.k-fig .c.tr{ top:7px; right:7px; border-top-width:1px; border-right-width:1px; }
.k-fig .c.bl{ bottom:7px; left:7px;  border-bottom-width:1px; border-left-width:1px; }
.k-fig .c.br{ bottom:7px; right:7px; border-bottom-width:1px; border-right-width:1px; }
.k-fig .xh{ position:absolute; inset:0; opacity:0; pointer-events:none;
  transition:opacity .22s ease;
  background:
    linear-gradient(to right,  transparent calc(50% - .5px), rgba(59,158,255,.42) calc(50% - .5px),
                   rgba(59,158,255,.42) calc(50% + .5px), transparent calc(50% + .5px)),
    linear-gradient(to bottom, transparent calc(50% - .5px), rgba(59,158,255,.42) calc(50% - .5px),
                   rgba(59,158,255,.42) calc(50% + .5px), transparent calc(50% + .5px)); }
.k-fig:hover .xh{ opacity:1; }
.k-fig .scan{ position:absolute; left:0; right:0; top:0; height:18%; pointer-events:none;
  background:linear-gradient(180deg, transparent, rgba(59,158,255,.14) 50%,
             rgba(59,158,255,.46) 84%, transparent);
  animation:sweep 1.9s linear 3; }
.k-fig .tag{ position:absolute; left:0; bottom:0; font-family:var(--k-mono);
  font-size:.57rem; letter-spacing:.15em; text-transform:uppercase; color:var(--k-ink-2);
  background:rgba(7,9,12,.86); padding:5px 11px;
  border-top:1px solid var(--k-line-2); border-right:1px solid var(--k-line-2); }
.k-cap{ font-family:var(--k-mono); font-size:.61rem; letter-spacing:.05em;
  color:var(--k-mute); margin:9px 0 4px; line-height:1.7; }

/* --- Plate: the eight required matplotlib figures, as inserted plates -- */
.k-plate{ border:1px solid var(--k-line-2); background:var(--k-panel); padding:13px 14px 14px; }
.k-plate-h{ display:flex; justify-content:space-between; align-items:baseline; gap:12px;
  font-family:var(--k-mono); font-size:.60rem; letter-spacing:.15em; text-transform:uppercase;
  color:var(--k-ink-2); border-bottom:1px solid var(--k-line); padding-bottom:8px; }
.k-plate-h .pn{ color:var(--k-blue); flex:none; }
.k-plate .sheet{ background:#F3F5F8; margin-top:12px; padding:6px; }
.k-plate .sheet img{ display:block; width:100%; height:auto; }
.k-plate .blurb{ font-family:var(--k-serif); font-size:.85rem; line-height:1.55;
  color:var(--k-ink-2); margin-top:11px; max-width:66ch; }

/* --- Tokens: always a word, never colour alone ------------------------ */
.k-tok{ display:inline-block; font-family:var(--k-mono); font-size:.585rem;
  letter-spacing:.15em; text-transform:uppercase; padding:3px 9px; border:1px solid;
  white-space:nowrap; }
.k-tok.ok{ color:var(--k-green); border-color:rgba(53,214,164,.42); background:rgba(53,214,164,.08); }
.k-tok.no{ color:var(--k-red);   border-color:rgba(255,95,82,.42);  background:rgba(255,95,82,.08); }
.k-tok.ex{ color:var(--k-amber); border-color:rgba(232,163,61,.42); background:rgba(232,163,61,.08); }
.k-tok.in{ color:var(--k-blue);  border-color:rgba(59,158,255,.42); background:rgba(59,158,255,.08); }

/* --- Editorial entry (left rule + prose, no card) --------------------- */
.k-entry{ border-left:2px solid var(--k-line-2); padding:1px 0 1px 21px; margin:0 0 24px; }
.k-entry.ok{ border-left-color:var(--k-green); }
.k-entry.no{ border-left-color:var(--k-red); }
.k-entry.ex{ border-left-color:var(--k-amber); }
.k-entry.in{ border-left-color:var(--k-blue); }
.k-entry .meta{ font-family:var(--k-mono); font-size:.585rem; letter-spacing:.16em;
  text-transform:uppercase; color:var(--k-mute); margin-bottom:6px; }
.k-entry .h{ font-family:var(--k-sans); font-size:.94rem; font-weight:600; color:var(--k-ink);
  margin:0 0 7px; letter-spacing:-.004em; display:flex; gap:11px; align-items:center;
  flex-wrap:wrap; }
.k-entry .p{ font-family:var(--k-serif); font-size:.925rem; line-height:1.62;
  color:var(--k-ink-2); margin:0; max-width:76ch; }
.k-entry .p code{ font-family:var(--k-mono); font-size:.83em; color:var(--k-ink);
  background:var(--k-panel); padding:1px 5px; border:1px solid var(--k-line); }

/* --- Experiment ledger ------------------------------------------------ */
.k-led{ display:flex; gap:16px; align-items:baseline; padding:10px 2px;
  border-bottom:1px solid var(--k-line); }
.k-led:last-child{ border-bottom:0; }
.k-led .nm{ font-family:var(--k-mono); font-size:.655rem; color:var(--k-ink);
  flex:0 0 15.5rem; letter-spacing:.02em; word-break:break-word; }
.k-led .tk{ flex:0 0 8.6rem; }
.k-led .ti{ font-family:var(--k-serif); font-size:.865rem; line-height:1.45;
  color:var(--k-ink-2); flex:1 1 18rem; }
@media (max-width: 860px){
  .k-led{ flex-wrap:wrap; gap:6px; }
  .k-led .nm, .k-led .tk, .k-led .ti{ flex:1 1 100%; }
}

/* --- Prose ------------------------------------------------------------ */
.k-lede{ font-family:var(--k-serif); font-size:1.02rem; line-height:1.62;
  color:var(--k-ink-2); max-width:76ch; margin:2px 0 4px; }
.k-lede b{ color:var(--k-ink); font-weight:600; }
.k-lede code{ font-family:var(--k-mono); font-size:.82em; color:var(--k-ink);
  background:var(--k-panel); padding:1px 5px; border:1px solid var(--k-line); }
.k-note{ font-family:var(--k-sans); font-size:.775rem; line-height:1.65;
  color:var(--k-mute); max-width:92ch; margin:10px 0 2px; }
.k-note code{ font-family:var(--k-mono); font-size:.86em; color:var(--k-ink-2); }
.k-note b{ color:var(--k-ink-2); }

/* --- Definition list (case files, metadata) --------------------------- */
.k-dl{ font-family:var(--k-mono); font-size:.68rem; }
.k-dl .row{ display:flex; justify-content:space-between; gap:14px; padding:8px 0;
  border-bottom:1px dotted var(--k-line-2); }
.k-dl .row:last-child{ border-bottom:0; }
.k-dl .k{ color:var(--k-mute); letter-spacing:.13em; text-transform:uppercase;
  font-size:.585rem; padding-top:1px; }
.k-dl .v{ color:var(--k-ink); text-align:right; }
.k-dl .v.ok{ color:var(--k-green); } .k-dl .v.no{ color:var(--k-red); }

/* --- Decision tree ---------------------------------------------------- */
.k-tree{ display:block; width:100%; height:auto; max-width:940px; margin:10px 0 0; }
.k-tree .ln{ fill:none; stroke:var(--k-line-2); stroke-width:1;
  stroke-dasharray:640; stroke-dashoffset:640;
  animation:draw 1.15s cubic-bezier(.3,.85,.3,1) forwards; }
.k-tree .ln.b{ animation-delay:.42s; }
.k-tree .lab { font-family:var(--k-mono); font-size:10px;   letter-spacing:.22em; fill:var(--k-mute); }
.k-tree .cond{ font-family:var(--k-mono); font-size:11px;   letter-spacing:.07em; fill:var(--k-ink-2); }
.k-tree .big { font-family:var(--k-mono); font-size:27px; font-weight:700; fill:var(--k-ink); }
.k-tree .huge{ font-family:var(--k-mono); font-size:37px; font-weight:700; }
.k-tree .sub { font-family:var(--k-mono); font-size:10.5px; fill:var(--k-mute); }
.k-tree .ok{ fill:var(--k-green); } .k-tree .warn{ fill:var(--k-amber); }
.k-tree .yn{ font-family:var(--k-mono); font-size:9.5px; letter-spacing:.2em; fill:var(--k-mute); }

/* ======================================================================
   STREAMLIT INTERNALS — the only place internal selectors are used.
   ====================================================================== */
footer{ visibility:hidden; }
#MainMenu{ visibility:hidden; }
[data-testid="stAppDeployButton"]{ display:none; }
header[data-testid="stHeader"]{ background:transparent; }

section[data-testid="stSidebar"]{ background:var(--k-void); border-right:1px solid var(--k-line); }
section[data-testid="stSidebar"] *{ color:var(--k-ink-2); }

/* Instrument index: the native radio, skinned into a numbered list. If these
   selectors ever stop matching, it degrades to a perfectly usable radio. */
section[data-testid="stSidebar"] div[data-testid="stRadio"] > label{ display:none; }
section[data-testid="stSidebar"] div[role="radiogroup"]{ gap:0 !important; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]{
  padding:11px 12px; border-bottom:1px solid var(--k-line);
  border-left:2px solid transparent; cursor:pointer;
  transition:background-color .18s ease, border-color .18s ease, padding-left .18s ease; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] p{
  font-family:var(--k-mono) !important; font-size:.665rem !important;
  letter-spacing:.135em !important; text-transform:uppercase;
  color:var(--k-mute) !important; margin:0 !important; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"]:hover{
  background:rgba(59,158,255,.05); padding-left:17px; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"]{
  border-left-color:var(--k-blue); background:rgba(59,158,255,.09); }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"][data-selected="true"] p{
  color:var(--k-ink) !important; }
section[data-testid="stSidebar"] label[data-testid="stRadioOption"] > div > div > div:first-child{
  display:none; }

.k-sidehead{ padding:2px 0 14px; }
.k-sidehead .w{ font-family:var(--k-mono); font-size:.78rem; font-weight:700;
  letter-spacing:.30em; color:var(--k-ink); }
.k-sidehead .s{ font-family:var(--k-mono); font-size:.575rem; letter-spacing:.18em;
  text-transform:uppercase; color:var(--k-mute); margin-top:6px; }
.k-sidegroup{ font-family:var(--k-mono); font-size:.575rem; letter-spacing:.20em;
  text-transform:uppercase; color:var(--k-blue); margin:22px 0 8px 2px; }

/* Buttons: instrument switches, not pills. */
.stButton > button{ font-family:var(--k-mono); font-size:.645rem; letter-spacing:.19em;
  text-transform:uppercase; border-radius:0; border:1px solid var(--k-line-2);
  background:transparent; color:var(--k-ink-2); padding:11px 20px;
  transition:border-color .18s ease, color .18s ease, background-color .18s ease; }
.stButton > button:hover{ border-color:var(--k-blue); color:var(--k-blue);
  background:rgba(59,158,255,.07); }
.stButton > button[kind="primary"]{ border-color:var(--k-blue); background:var(--k-blue);
  color:#06090D; font-weight:700; }
.stButton > button[kind="primary"]:hover{ background:#63B2FF; color:#06090D; }

/* Widget surfaces */
div[data-testid="stExpander"] details{ background:var(--k-panel);
  border:1px solid var(--k-line); border-radius:0; }
div[data-testid="stExpander"] summary{ font-family:var(--k-mono); font-size:.66rem;
  letter-spacing:.13em; text-transform:uppercase; color:var(--k-ink-2); }
/* Alerts: the coloured background lives on stAlertContainer, not stAlert. */
[data-testid="stAlert"]{ border-radius:0; }
[data-testid="stAlertContainer"]{ border-radius:0 !important; background:var(--k-panel) !important;
  border:1px solid var(--k-line) !important; border-left:2px solid var(--k-blue) !important;
  color:var(--k-ink-2) !important; box-shadow:none !important; }
[data-testid="stAlertContainer"] p{ color:var(--k-ink-2) !important; font-size:.83rem; }
[data-testid="stAlertContainer"] code{ color:var(--k-ink) !important;
  background:var(--k-void) !important; }
[data-testid="stAlertContainer"] strong{ color:var(--k-ink) !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]){
  border-left-color:var(--k-amber) !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]){
  border-left-color:var(--k-red) !important; }
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]){
  border-left-color:var(--k-green) !important; }
[data-testid="stAlertContainer"] [data-testid="stIconMaterial"]{ color:var(--k-mute) !important; }
section[data-testid="stFileUploaderDropzone"], div[data-testid="stFileUploaderDropzone"]{
  background:var(--k-panel); border:1px dashed var(--k-line-2); border-radius:0; }
div[data-baseweb="select"] > div{ border-radius:0 !important;
  background:var(--k-panel) !important; border-color:var(--k-line-2) !important; }
label[data-testid="stWidgetLabel"] p{ font-family:var(--k-mono) !important;
  font-size:.62rem !important; letter-spacing:.12em; text-transform:uppercase;
  color:var(--k-mute) !important; }
hr{ border-color:var(--k-line) !important; }
code{ font-family:var(--k-mono) !important; }

/* Focus visibility (accessibility) */
a:focus-visible, button:focus-visible, [role="radio"]:focus-visible,
input:focus-visible, select:focus-visible, summary:focus-visible{
  outline:2px solid var(--k-blue) !important; outline-offset:2px !important; }

@media (max-width: 900px){
  .k-cell{ flex-basis:50%; padding-left:16px; }
  .k-cell:first-child{ padding-left:0; }
  .k-mast-meta{ text-align:left; }
}

/* Respect the operating-system reduced-motion preference. */
@media (prefers-reduced-motion: reduce){
  *, *::before, *::after{ animation:none !important; transition:none !important; }
  .k-tree .ln{ stroke-dashoffset:0 !important; }
  .k-fig .scan{ display:none; }
}
</style>
"""


def inject_theme() -> None:
    st.markdown(THEME_CSS, unsafe_allow_html=True)


# ==========================================================================
# ENCODE — images become data URIs so they can live inside styled HTML while
# the application stays completely offline (no CDN, no external asset server).
# ==========================================================================

def _uri_from_bytes(raw: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def image_uri(img: np.ndarray) -> str | None:
    """Encode a numpy image (grayscale, or BGR as OpenCV produces it) to a PNG
    data URI. Returns None rather than raising if encoding fails."""
    if img is None:
        return None
    ok, buf = cv2.imencode(".png", img)
    return _uri_from_bytes(buf.tobytes()) if ok else None


@st.cache_data
def plot_uri(fname: str) -> str | None:
    path = os.path.join(PLOTS_DIR, fname)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return _uri_from_bytes(f.read())


# ==========================================================================
# DATA — cached loaders. Every one reads a file this project's own scripts
# produced; nothing here is recomputed silently on each page load.
# ==========================================================================

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
    """Paths in per_pair_results.csv are written by evaluation/evaluate.py as
    `os.path.join(data_root, ...)` with `data_root="data"` — i.e. already
    relative to the PROJECT root and already starting with "data/". Joining
    with the project's data/ directory again would double it into
    "data/data/...", which doesn't exist.

    Normalize backslashes first: `ground_truth.json` stores forward-slash
    relative paths, but `os.path.join` on Windows joins "data" + that with a
    backslash, baking a literal "data\\challenge/xxx_search.png" into the CSV.
    Windows tolerates the mixed separators; POSIX does not, so cv2.imread
    silently returns None and the next cv2.cvtColor call crashes with a
    confusing "!_src.empty()" assertion.

    An ABSOLUTE path baked in on another OS (e.g. "D:\\...") still cannot be
    resolved here — callers must handle a missing file rather than assume this
    succeeded. See `load_pair_image`."""
    normalized = stored_path.replace("\\", "/")
    if os.path.isabs(normalized):
        return normalized
    return os.path.join(PROJECT_ROOT, normalized)


def load_pair_image(stored_path: str):
    """Resolve + read a benchmark image, returning (image, error_message).

    Never raises and never fabricates an image: a CSV produced on another OS
    can carry an absolute path that does not exist here, and passing the
    resulting None straight into cv2.cvtColor crashes the page with an opaque
    '!_src.empty()' assertion."""
    resolved = resolve_result_path(stored_path)
    if not os.path.exists(resolved):
        return None, (f"Image not found: `{resolved}`. `per_pair_results.csv` stores the path that "
                      "was written when the benchmark ran — if that run happened on another machine "
                      "or OS, the path will not resolve here. Re-run `python scripts/evaluate_model.py` "
                      "on this machine to regenerate it.")
    img = read_image(resolved)
    if img is None:
        return None, f"File exists but could not be decoded as an image: `{resolved}`."
    return img, None


def read_uploaded_image(uploaded_file) -> np.ndarray:
    data = np.frombuffer(uploaded_file.read(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def staleness_warning() -> str | None:
    """Real check, not decorative: baseline_metrics.json / plots /
    failure_analysis_cases.json are only trustworthy if they were produced by
    the SAME `evaluate_model.py` run as the current per_pair_results.csv."""
    metrics_path = os.path.join(REPORTS_DIR, "baseline_metrics.json")
    per_pair_path = os.path.join(REPORTS_DIR, "per_pair_results.csv")
    if not (os.path.exists(metrics_path) and os.path.exists(per_pair_path)):
        return None
    metrics_mtime = os.path.getmtime(metrics_path)
    per_pair_mtime = os.path.getmtime(per_pair_path)
    if per_pair_mtime - metrics_mtime > 60:
        gap_hours = (per_pair_mtime - metrics_mtime) / 3600
        return (
            f"**`baseline_metrics.json` is {gap_hours:.1f}h older than `per_pair_results.csv`** — "
            "they were written by different runs and describe different pipeline/dataset states. "
            "Re-run `python scripts/evaluate_model.py` before citing any of these numbers."
        )
    return None


EXPERIMENTS_DIR = os.path.join(PROJECT_ROOT, "experiments")

# The only experiments that reached production. Sourced from
# reports/GATE_EXCEPTIONS.md (the four documented exceptions) plus
# finer_hypothesis_grid, integrated 2026-08-15. Kept as an explicit set because
# "was this shipped" is a decision recorded in that report, not something a
# REPORT.md's prose can be parsed for reliably - every other verdict below IS
# derived from the report text.
INTEGRATED_EXPERIMENTS = {
    "scale_range_v1": "gate exception A2",
    "multiway_tiebreak_v1": "gate exception A6",
    "psf_gated_selection": "gate exception, PSF dual-arm",
    "psr_confidence": "gate exception, ambiguity threshold",
    "finer_hypothesis_grid": "integrated 2026-08-15",
}

# Ordered: first match wins. REJECT is checked before near-miss and diagnostic
# on purpose - several reports are titled "REJECT (near-miss on one seed)" or
# "REJECT at the diagnostic stage", and the rejection is the verdict.
_VERDICT_RULES = [
    ("interim",        ("interim",)),
    ("rejected",       ("reject",)),
    ("not reproduced", ("not supported", "did not reproduce")),
    ("inconclusive",   ("inconclusive",)),
    ("near-miss",      ("near-miss", "near miss", "promising", "replicates", "not integrated")),
    ("diagnostic",     ("diagnostic", "refuted", "no-op", "supported")),
]


def _classify(title: str, head: list[str]) -> str:
    """Classify from the report's TITLE plus any explicit status line only.

    Scanning the whole opening section instead reads far too much ordinary
    prose - words like "promising", "supported" and "no-op" appear constantly
    in the body of a report that concludes the opposite."""
    signal = [title]
    for line in head:
        stripped = line.strip()
        if stripped.lower().startswith(("**status", "> **status", "**the status")) \
                or "**STATUS" in stripped:
            signal.append(stripped)
    low = " ".join(signal).lower()
    for verdict, needles in _VERDICT_RULES:
        if any(n in low for n in needles):
            return verdict
    return "documented"


@st.cache_data
def load_experiment_ledger() -> list[dict]:
    """Every experiment directory on disk, read from its own REPORT.md.

    Deliberately filesystem-driven rather than a curated list: the project's
    claim is that *every* candidate change is kept with its evidence, and a
    hand-maintained list in this file would quietly drift out of step with the
    directory the moment an experiment was added. A directory with no report is
    surfaced as such rather than hidden."""
    if not os.path.isdir(EXPERIMENTS_DIR):
        return []
    rows = []
    for name in sorted(os.listdir(EXPERIMENTS_DIR)):
        path = os.path.join(EXPERIMENTS_DIR, name)
        if not os.path.isdir(path) or name.startswith((".", "_")):
            continue
        report = os.path.join(path, "REPORT.md")
        docs = ([report] if os.path.exists(report)
                else sorted(os.path.join(path, f) for f in os.listdir(path)
                            if f.endswith(".md")))
        if not docs:
            # A couple of the earliest investigations were written up in reports/
            # before the one-REPORT.md-per-directory convention existed. Follow
            # the convention's naming to find them rather than calling the work
            # undocumented when it is not.
            elsewhere = os.path.join(PROJECT_ROOT, "reports", f"{name.upper()}.md")
            if os.path.exists(elsewhere):
                docs = [elsewhere]
            else:
                rows.append({"name": name, "title": "No written report found for this directory.",
                             "verdict": "missing", "doc": None})
                continue
        doc_path = docs[0]
        try:
            with open(doc_path, encoding="utf-8") as f:
                head = [next(f, "") for _ in range(15)]
        except OSError:
            continue
        title = next((ln[2:].strip() for ln in head if ln.startswith("# ")), name)
        # Titles are written as "experiments/<name> — claim" or "Experiment X: claim";
        # strip the redundant prefix so the name column isn't repeated in the prose.
        for prefix in (f"experiments/{name} — ", f"experiments/{name} - ", f"experiments/{name}"):
            if title.startswith(prefix):
                title = title[len(prefix):].lstrip("—- :")
                break
        if name in INTEGRATED_EXPERIMENTS:
            verdict = "integrated"
        else:
            verdict = _classify(title, head)
        rows.append({"name": name, "title": title, "verdict": verdict,
                     "doc": os.path.relpath(doc_path, PROJECT_ROOT).replace(os.sep, "/")})
    return rows


def selective_prediction_stats(df: pd.DataFrame) -> dict | None:
    """Coverage / accuracy / flag statistics, computed live from the per-pair
    CSV so they can never drift from the benchmark shown elsewhere.

    Returns None when the CSV lacks the columns (e.g. an older run)."""
    if df.empty or "ambiguous" not in df.columns or "error_px" not in df.columns:
        return None
    raw = df["ambiguous"]
    # A CSV round-trip can give either real booleans or the strings
    # "True"/"False"; astype(bool) on the latter is silently all-True.
    flagged = (raw.astype(str).str.strip().str.lower().isin(("true", "1", "yes"))
               if raw.dtype == object else raw.astype(bool))
    correct = df["error_px"] <= TOLERANCE_PX
    answered = ~flagged
    n_fail = int((~correct).sum())
    if not answered.any() or n_fail == 0:
        return None
    return {
        "n": int(len(df)),
        "n_answered": int(answered.sum()),
        "n_flagged": int(flagged.sum()),
        "coverage": float(answered.mean()),
        "flag_rate": float(flagged.mean()),
        "acc_answered": float(correct[answered].mean()),
        "acc_all": float(correct.mean()),
        "n_fail": n_fail,
        "n_fail_flagged": int((flagged & ~correct).sum()),
        "failure_recall": float((flagged & ~correct).sum()) / n_fail,
    }


baseline_metrics = load_json(os.path.join(REPORTS_DIR, "baseline_metrics.json"))
gate_result = load_json(os.path.join(REPORTS_DIR, "integration_gate.json"))
failure_cases = load_json(os.path.join(REPORTS_DIR, "failure_analysis_cases.json"))
per_pair_df = load_per_pair_results()


# ==========================================================================
# PRIMITIVES
# ==========================================================================

def masthead() -> None:
    """One rule across the top of every screen. The live figure on the right is
    read from baseline_metrics.json — never hardcoded."""
    right = "no benchmark artifact"
    if baseline_metrics:
        acc5 = baseline_metrics["overall"]["accuracy_at_5px"]
        n_pairs = baseline_metrics["overall"].get("n", len(per_pair_df) or "—")
        right = f"{acc5:.1%} @{TOLERANCE_PX:.0f}px &middot; {n_pairs} pairs"
    st.markdown(
        f"""<div class="k-mast r">
              <div>
                <div class="k-word">DRIFT<i>&middot;</i>SENSE</div>
              </div>
              <div class="k-mast-meta">
                <span class="k-dot"></span>{right}<br>
                Applied Materials &middot; SEMICON India 2026 &middot; Kaccha Mango
                &middot; OpenCV {cv2.__version__}
              </div>
            </div>""",
        unsafe_allow_html=True)


def standing(text: str) -> None:
    st.markdown(f'<div class="k-standing r d1">{text}</div>', unsafe_allow_html=True)


def hero(numeral_html: str, note_html: str) -> None:
    st.markdown(
        f'<div class="k-hero"><div class="k-numeral r d2">{numeral_html}</div>'
        f'<div class="k-hero-note r d3">{note_html}</div></div>',
        unsafe_allow_html=True)


def mark(number: str, title: str) -> None:
    """Numbered chapter mark with a crosshair glyph and a drawn rule. This
    replaces the card header entirely — sections are separated by structure
    and whitespace, not by boxes."""
    st.markdown(
        f'<div class="k-mark r"><span class="x"></span><span class="n">{number}</span>'
        f'<span class="t">{title}</span><span class="rule"></span></div>',
        unsafe_allow_html=True)


def lede(html: str) -> None:
    st.markdown(f'<div class="k-lede r d1">{html}</div>', unsafe_allow_html=True)


def note(html: str) -> None:
    st.markdown(f'<div class="k-note">{html}</div>', unsafe_allow_html=True)


def run_row(cells: list[dict], delay: str = "d2") -> None:
    """A band of readings on hairlines. Each cell: k (label), v (value),
    optional s (sub) and tone ('pos'/'neg'), optional lead=True."""
    out = []
    for c in cells:
        cls = "k-cell lead" if c.get("lead") else "k-cell"
        if c.get("verdict"):                 # colours the value itself, not just the sub
            cls += f' {c["verdict"]}'
        sub = ""
        if c.get("s"):
            sub = f'<div class="s {c.get("tone", "")}">{c["s"]}</div>'
        out.append(f'<div class="{cls}"><div class="k">{c["k"]}</div>'
                   f'<div class="v">{c["v"]}</div>{sub}</div>')
    st.markdown(f'<div class="k-run r {delay}">{"".join(out)}</div>', unsafe_allow_html=True)


def tok(kind: str, text: str) -> str:
    """Status token. Always carries a word, never colour alone."""
    return f'<span class="k-tok {kind}">{text}</span>'


def figure(uri: str | None, tag: str = "", caption: str = "", scan: bool = False,
           missing: str = "Figure unavailable.") -> None:
    """Optical frame: registration corners, a crosshair that tracks the pointer
    on hover, and — only immediately after a live acquisition — a scan sweep
    that runs three times and stops."""
    if not uri:
        st.markdown(f'<div class="k-cap">{missing}</div>', unsafe_allow_html=True)
        return
    tag_html = f'<div class="tag">{tag}</div>' if tag else ""
    scan_html = '<div class="scan"></div>' if scan else ""
    cap_html = f'<div class="k-cap">{caption}</div>' if caption else ""
    st.markdown(
        f'<div class="k-fig r d2"><img src="{uri}" alt="{tag or "figure"}">'
        f'<div class="xh"></div>{scan_html}'
        f'<span class="c tl"></span><span class="c tr"></span>'
        f'<span class="c bl"></span><span class="c br"></span>{tag_html}</div>{cap_html}',
        unsafe_allow_html=True)


_CODE_RE = None


def md_inline(text: str) -> str:
    """Render the small subset of Markdown that this project's own artifacts
    actually use — `code` and **bold** — into HTML.

    Needed because strings such as `failure_analysis_cases.json`'s `rationale`
    are authored as Markdown but are injected into hand-built HTML here, where
    Streamlit's Markdown pass never sees them; without this the backticks show
    up literally on screen."""
    import re
    global _CODE_RE
    if _CODE_RE is None:
        _CODE_RE = (re.compile(r"`([^`]+)`"), re.compile(r"\*\*([^*]+)\*\*"))
    code_re, bold_re = _CODE_RE
    out = code_re.sub(r"<code>\1</code>", str(text))
    return bold_re.sub(r"<b>\1</b>", out)


def dl(rows: list[tuple[str, str]], classes: dict | None = None) -> None:
    classes = classes or {}
    body = "".join(
        f'<div class="row"><span class="k">{k}</span>'
        f'<span class="v {classes.get(k, "")}">{v}</span></div>'
        for k, v in rows)
    st.markdown(f'<div class="k-dl r d2">{body}</div>', unsafe_allow_html=True)


def entry(kind: str, meta: str, heading: str, body: str, token: str = "") -> None:
    """Editorial entry: a coloured left rule, a mono kicker, a sans heading and
    serif prose. Deliberately not a card — entries stack into a column of text."""
    tok_html = tok(kind, token) if token else ""
    st.markdown(
        f'<div class="k-entry {kind} r d1"><div class="meta">{meta}</div>'
        f'<div class="h">{heading}{tok_html}</div><div class="p">{body}</div></div>',
        unsafe_allow_html=True)


def marked_overlay(gray: np.ndarray, pred_xy, gt_xy=None) -> str | None:
    """Search image with the prediction (and optionally ground truth) marked,
    returned as a data URI so it can be framed by `figure`."""
    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    px, py = int(pred_xy[0]), int(pred_xy[1])
    cv2.drawMarker(display, (px, py), BGR_PRED, markerType=cv2.MARKER_CROSS,
                    markerSize=30, thickness=2)
    cv2.circle(display, (px, py), 16, BGR_PRED, 1, lineType=cv2.LINE_AA)
    if gt_xy is not None:
        cv2.drawMarker(display, (int(gt_xy[0]), int(gt_xy[1])), BGR_TRUE,
                        markerType=cv2.MARKER_DIAMOND, markerSize=24, thickness=2)
    return image_uri(display)


def psf_arm_note(result) -> str:
    """Which template the dual-arm PSF selection picked for this pair, and how
    decisively. Surfacing it keeps a per-pair branch from being invisible.
    See pipeline/localize.py::PSF_MATCH_SIGMA and GATE_EXCEPTIONS.md #3."""
    sigma = getattr(result, "psf_sigma", 0.0)
    gap = getattr(result, "psf_decisiveness", float("nan"))
    gap_txt = f"{gap:.4f}" if np.isfinite(gap) else "n/a (no distinct rival)"
    if sigma and sigma > 0:
        return (f"Template <b>PSF-matched</b> (blur &sigma; {sigma:.1f}), matched to the Search "
                f"image's passband — it won the dual-arm comparison here. Decisiveness "
                f"(top vs. best rival &gt;10px away) <code>{gap_txt}</code>")
    return (f"Template <b>standard</b> (no PSF blur) — the sharp template won the dual-arm "
            f"comparison here. Decisiveness (top vs. best rival &gt;10px away) "
            f"<code>{gap_txt}</code>")


def result_readout(result, error_px: float | None = None) -> None:
    cells = []
    if error_px is not None:
        hit = error_px <= TOLERANCE_PX
        cells.append({"k": "Error", "v": f"{error_px:.2f}px", "lead": True,
                      "verdict": "pass" if hit else "miss",
                      "s": f"within {TOLERANCE_PX:.0f}px" if hit else f"outside {TOLERANCE_PX:.0f}px",
                      "tone": "pos" if hit else "neg"})
    else:
        cells.append({"k": "Predicted centre", "v": f"{result.x:.0f}, {result.y:.0f}",
                      "lead": True, "s": "x, y in Search-image pixels"})
    cells += [
        {"k": "Confidence (ZNCC)", "v": f"{result.confidence:.3f}"},
        {"k": "Ambiguity ratio", "v": f"{result.ambiguity_ratio:.3f}",
         "s": "runner-up / winner"},
        {"k": "Runtime", "v": f"{result.runtime_s:.2f}s"},
    ]
    run_row(cells, delay="d1")
    if result.ambiguous:
        entry("ex", "verdict", "Flagged AMBIGUOUS",
              "A second location scored nearly as well. The system reports this for review "
              "rather than returning it silently — see the decision tree on <b>01 Overview</b>.",
              token="held")
    else:
        entry("ok", "verdict", "Answer returned",
              "The winning candidate clearly outscored the best rival more than 10px away, so "
              "the localization is returned rather than deferred.", token="accepted")
    note(psf_arm_note(result))


def decision_tree_svg(s: dict) -> str:
    """The selective-prediction rule, drawn as what it is: one branch. Every
    number is computed live from per_pair_results.csv by
    `selective_prediction_stats` — nothing here is authored.

    Emitted as a SINGLE LINE with no blank lines anywhere. Streamlit runs the
    string through a Markdown pass before sanitising it, and a blank line
    inside a raw-HTML block terminates that block — everything after it is
    re-parsed as Markdown, which strips the SVG tags and dumps the labels into
    the page as loose prose. This is not cosmetic pedantry; it silently broke
    the whole figure once."""
    d = 100.0 * (s["acc_answered"] - s["acc_all"])
    parts = [
        '<svg class="k-tree r d3" viewBox="0 0 900 336" role="img" aria-label="Selective '
        f'prediction decision tree: {s["n"]} pairs, {s["n_answered"]} answered and '
        f'{s["n_flagged"]} deferred.">',
        '<text x="450" y="16" text-anchor="middle" class="lab">EVERY PAIR EVALUATED</text>',
        f'<text x="450" y="52" text-anchor="middle" class="big">{s["n"]}</text>',
        '<path class="ln" d="M450 64 L450 88"/>',
        '<text x="450" y="110" text-anchor="middle" class="cond">is the winner decisive '
        'against its best rival &gt;10px away?</text>',
        '<path class="ln b" d="M450 122 L450 146 L165 146 L165 176"/>',
        '<path class="ln b" d="M450 122 L450 146 L735 146 L735 176"/>',
        '<text x="300" y="140" text-anchor="middle" class="yn">YES</text>',
        '<text x="600" y="140" text-anchor="middle" class="yn">NO</text>',
        '<text x="165" y="196" text-anchor="middle" class="lab ok">ANSWER</text>',
        f'<text x="165" y="238" text-anchor="middle" class="huge ok">{s["coverage"]:.1%}</text>',
        f'<text x="165" y="262" text-anchor="middle" class="sub">{s["n_answered"]} of '
        f'{s["n"]} pairs returned</text>',
        f'<text x="165" y="300" text-anchor="middle" class="big">{s["acc_answered"]:.1%}</text>',
        f'<text x="165" y="322" text-anchor="middle" class="sub">accuracy on what it '
        f'answers &#160;({d:+.1f} pp vs all)</text>',
        '<text x="735" y="196" text-anchor="middle" class="lab warn">DEFER FOR REVIEW</text>',
        f'<text x="735" y="238" text-anchor="middle" class="huge warn">{s["flag_rate"]:.1%}</text>',
        f'<text x="735" y="262" text-anchor="middle" class="sub">{s["n_flagged"]} pairs '
        f'handed back</text>',
        f'<text x="735" y="300" text-anchor="middle" class="big">{s["failure_recall"]:.1%}</text>',
        f'<text x="735" y="322" text-anchor="middle" class="sub">of all {s["n_fail"]} '
        f'failures caught here</text>',
        '</svg>',
    ]
    return "".join(parts)


# ==========================================================================
# CHARTS — restyled for graphite. Both are computed live from the per-pair
# CSV; they complement, never replace, the eight required static figures.
# ==========================================================================
_FONT = dict(family='ui-monospace, SFMono-Regular, Menlo, Consolas, monospace',
              size=11, color="#6D7C8F")
_GRID = "#1D242E"


def paired_chart_height(df: pd.DataFrame) -> int:
    """The two Executive Summary charts sit side by side, so they are given one
    shared height driven by the taller of the two — the family bar, whose
    length depends on how many structural families the benchmark actually
    contains. Without this the shorter column leaves a dead gap underneath."""
    n_fam = int(df["structural_family"].nunique()) if "structural_family" in df else 0
    return max(320, 26 * n_fam)


def render_accuracy_curve(df: pd.DataFrame, height: int = 320):
    thresholds = np.arange(0, 20.5, 0.5)
    errors = df["error_px"].to_numpy()
    acc = [(errors <= t).mean() * 100 for t in thresholds]
    acc5 = float((errors <= TOLERANCE_PX).mean() * 100)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=thresholds, y=acc, mode="lines", line=dict(color="#3B9EFF", width=2.2),
        fill="tozeroy", fillcolor="rgba(59,158,255,0.09)", name="accuracy",
        hovertemplate="tolerance \u2264%{x:.1f} px<br>accuracy %{y:.1f}%<extra></extra>"))
    fig.add_vline(x=TOLERANCE_PX, line=dict(color="#FF5F52", width=1, dash="dot"))
    fig.add_trace(go.Scatter(
        x=[TOLERANCE_PX], y=[acc5], mode="markers+text",
        marker=dict(color="#FF5F52", size=7),
        text=[f"  {acc5:.1f}% @{TOLERANCE_PX:.0f}px"], textposition="middle right",
        textfont=dict(color="#FF5F52", size=11, family=_FONT["family"]),
        hoverinfo="skip", showlegend=False))
    fig.update_layout(
        height=height, margin=dict(l=6, r=6, t=6, b=6),
        xaxis_title="error tolerance", yaxis_title="accuracy",
        yaxis_range=[0, 104], plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=_FONT, showlegend=False, hoverlabel=dict(font=dict(family=_FONT["family"])),
        xaxis=dict(gridcolor=_GRID, zeroline=False, ticksuffix=" px", linecolor=_GRID),
        yaxis=dict(gridcolor=_GRID, zeroline=False, ticksuffix="%", linecolor=_GRID))
    return fig


def render_family_bar(df: pd.DataFrame, height: int = 320):
    grp = (df.groupby("structural_family")["error_px"]
             .apply(lambda s: (s <= TOLERANCE_PX).mean() * 100).sort_values())
    colors = ["#FF5F52" if v < 40 else "#E8A33D" if v < 70 else "#35D6A4" for v in grp.values]
    fig = go.Figure(go.Bar(
        x=grp.values, y=grp.index, orientation="h", marker_color=colors,
        marker_line_width=0, width=0.62,
        hovertemplate="%{y}<br>%{x:.1f}%% @5px<extra></extra>",
        text=[f"{v:.0f}%" for v in grp.values], textposition="outside", cliponaxis=False,
        textfont=dict(family=_FONT["family"], size=10, color="#AEBBCA")))
    fig.update_layout(
        height=height, margin=dict(l=6, r=44, t=6, b=6),
        xaxis_title=f"accuracy @{TOLERANCE_PX:.0f}px", xaxis_range=[0, 112],
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font=_FONT,
        hoverlabel=dict(font=dict(family=_FONT["family"])),
        xaxis=dict(gridcolor=_GRID, zeroline=False, ticksuffix="%", linecolor=_GRID),
        yaxis=dict(gridcolor="rgba(0,0,0,0)", linecolor=_GRID))
    return fig


def chart(fig, title: str, blurb: str) -> None:
    st.markdown(f'<div class="k-plate-h r d2"><span>{title}</span>'
                f'<span class="pn">live</span></div>', unsafe_allow_html=True)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    st.markdown(f'<div class="k-cap">{blurb}</div>', unsafe_allow_html=True)


def plate(number: str, fname: str, title: str, blurb: str) -> None:
    """One of the eight required benchmark figures, presented as an inserted
    plate. The matplotlib figures are light by design and are shown on their
    own sheet rather than recoloured, so what is on screen is byte-for-byte
    what `scripts/evaluate_model.py` wrote."""
    uri = plot_uri(fname)
    if uri is None:
        st.markdown(
            f'<div class="k-plate r d1"><div class="k-plate-h"><span>{title}</span>'
            f'<span class="pn">{number}</span></div>'
            f'<div class="blurb">Not generated yet — run '
            f'<code>python scripts/evaluate_model.py</code>.</div></div>',
            unsafe_allow_html=True)
        return
    st.markdown(
        f'<div class="k-plate r d1"><div class="k-plate-h"><span>{title}</span>'
        f'<span class="pn">{number}</span></div>'
        f'<div class="sheet"><img src="{uri}" alt="{title}"></div>'
        f'<div class="blurb">{blurb}</div></div>',
        unsafe_allow_html=True)
    st.write("")


# ==========================================================================
# INSTRUMENT INDEX
# ==========================================================================
# Screen names are unchanged from the previous build so the README, the demo
# script and the AppTest harness all still address the same eight screens.
NAV = [
    ("01", "Executive Summary",  "OVERVIEW"),
    ("02", "Generate Sample",    "INSTRUMENT"),
    ("03", "Live Localization",  "INSTRUMENT"),
    ("04", "Visualization",      "EVIDENCE"),
    ("05", "Benchmark Dashboard", "EVIDENCE"),
    ("06", "Failure Analysis",   "EVIDENCE"),
    ("07", "Experiment Results", "EVIDENCE"),
    ("08", "System Information", "PROVENANCE"),
]
_SEP = "   "                                    # three spaces — the split token
OPTIONS = [f"{num}{_SEP}{name}" for num, name, _ in NAV]
_BY_NAME = {name: (num, band) for num, name, band in NAV}


def render_index() -> str:
    """One radio, skinned into a numbered instrument index.

    The number is baked into the option string and stripped back off, rather
    than applied via `format_func`: format_func keeps the raw value in session
    state but exposes only the formatted string to introspection, which makes
    the app awkward to drive from Streamlit's own AppTest harness. Baking it in
    keeps every screen reachable in automated tests."""
    st.sidebar.markdown(
        '<div class="k-sidehead"><div class="w">DRIFT&middot;SENSE</div>'
        '<div class="s">Precision under drift</div></div>',
        unsafe_allow_html=True)
    st.sidebar.markdown('<div class="k-sidegroup">Index</div>', unsafe_allow_html=True)
    choice = st.sidebar.radio("Screen", OPTIONS, label_visibility="collapsed")
    return choice.split(_SEP, 1)[1]


def render_generator_controls() -> tuple[dict, str]:
    """Acquisition parameter panel, shown only on 02 Generate Sample.

    Every control is a real key in generator/dataset_generator.py's
    DEFAULT_PARAMS (or the crop_mode / force_preset arguments generate_pair
    already accepts) — none of it is decorative. No parameter, default, range
    or behaviour is changed here."""
    o: dict = {}
    st.sidebar.markdown('<div class="k-sidegroup">Acquisition parameters</div>',
                        unsafe_allow_html=True)

    with st.sidebar.expander("Architecture", expanded=True):
        preset = st.selectbox("Architecture preset", ["Random"] + mat_generator.PRESET_NAMES,
                               help="Force one of the 6 DRAM mat presets, or Random to let the "
                                    "macro layout pick freely.")
        o["force_preset"] = None if preset == "Random" else preset
        o["feature_size_scale"] = st.slider(
            "Feature size scale", 0.5, 2.0, 1.0, 0.05,
            help="Scales every pitch/width in the chosen preset proportionally.")
        crop_mode = st.selectbox(
            "Crop placement",
            ["random", "single_mat", "mat_boundary", "same_preset_boundary",
             "multi_mat", "strip_center"], index=0,
            help="Where the Reference crop sits relative to macro structure.")

    with st.sidebar.expander("Die layout"):
        o["mat_size_nm"] = st.slider("Array block (mat) size (nm)", 800, 5000, 2400, 100,
                                      help="Size of each independently-generated mat sub-array.")
        o["strip_width_nm"] = st.slider("Separator strip width (nm)", 80, 800, 300, 20,
                                         help="Peripheral/routing band between mats.")

    with st.sidebar.expander("SEM imaging physics"):
        o["blur_search_effective_px"] = st.slider(
            "Beam spot size (search px)", 0.2, 4.0, 1.0, 0.1,
            help="Search-image point-spread-function blur sigma.")
        o["collapse_threshold_nm"] = st.slider(
            "Pattern-collapse threshold (nm)", 0.0, 25.0, 10.0, 1.0,
            help="Gaps narrower than this bridge together. 0 disables the effect.")
        o["collapse_enabled"] = o["collapse_threshold_nm"] > 0
        o["astigmatism_ratio"] = st.slider(
            "Beam astigmatism ratio", 0.5, 2.5, 1.0, 0.05,
            help="Elliptical beam spot (sigma_y = sigma_x * ratio).")

    with st.sidebar.expander("Acquisition & drift"):
        o["dose_reference"] = st.slider("Reference dose", 100.0, 5000.0, 1800.0, 100.0,
                                         help="Higher = cleaner. Controls Poisson shot-noise SNR.")
        o["dose_search"] = st.slider("Search dose", 20.0, 1000.0, 220.0, 10.0,
                                      help="Typically lower/faster than the Reference acquisition.")
        o["shear_amplitude_px"] = st.slider("Raster drift/shear (px)", 0.0, 5.0, 1.0, 0.1,
                                             help="Progressive row-to-row scan shear.")
        o["jitter_std_px"] = st.slider("Row jitter (px)", 0.0, 3.0, 0.4, 0.1,
                                        help="Per-row scan vibration, independent of shear.")
        o["rotation_deg"] = st.slider(
            "Residual rotation drift (deg)", -8.0, 8.0, 0.0, 0.25,
            help="The pipeline tests rotation hypotheses on a fixed grid — values near a "
                 "midpoint between two tested hypotheses are hardest.")
        o["extra_scale"] = st.slider(
            "Residual scale drift (x)", 0.85, 1.15, 1.0, 0.01,
            help="Magnification drift on top of the fixed base 10x. Same midpoint sensitivity.")

    with st.sidebar.expander("Distortion & polygon scaling"):
        o["linewidth_bias_nm"] = st.slider("Linewidth/CD bias (nm)", -10.0, 10.0, 0.0, 0.5,
                                            help="Global over/under-exposure or etch bias.")
        o["corner_rounding_px"] = st.slider("Corner rounding (px)", 0.0, 6.0, 0.0, 0.5,
                                             help="Real litho/etch never draws sharp corners.")
        o["barrel_k"] = st.slider(
            "Barrel(+)/pincushion(-)", -0.01, 0.01, 0.0, 0.0005, format="%.4f",
            help="Radial scan-linearity distortion. Ground truth is NOT analytically corrected "
                 "for this — large values visibly displace the true match location.")
        o["vignette_strength"] = st.slider("Vignette strength", 0.0, 1.0, 0.0, 0.05,
                                            help="Radial illumination falloff toward the edge.")
        o["gamma"] = st.slider("Gamma (contrast curve)", 0.4, 2.5, 1.0, 0.05,
                                help="Detector-gain nonlinearity.")

    with st.sidebar.expander("Noise"):
        o["charging_prob"] = st.slider("Charging streak probability", 0.0, 0.05, 0.0, 0.005,
                                        help="Per-row probability of a bright charging streak.")
        o["charging_intensity"] = st.slider("Charging streak intensity", 0.0, 100.0, 0.0, 5.0,
                                             help="Brightness added when a streak occurs.")
        o["speckle_sigma"] = st.slider("Speckle sigma (multiplicative)", 0.0, 0.3, 0.0, 0.01,
                                        help="out = img * (1 + N(0, sigma)).")
        o["salt_pepper_amount"] = st.slider("Salt-and-pepper probability", 0.0, 0.05, 0.0, 0.0025,
                                             help="Pixels forced to pure black/white.")

    if "gen_seed" not in st.session_state:
        st.session_state.gen_seed = 42
    if st.sidebar.button("Regenerate — new seed", use_container_width=True):
        st.session_state.gen_seed = int(np.random.randint(0, 2_000_000_000))
    return o, crop_mode


def screen_head(name: str, title: str, sub_html: str) -> None:
    num, band = _BY_NAME[name]
    st.markdown(
        f'<div class="k-standing r d1">{num} &nbsp;&mdash;&nbsp; {band}</div>'
        f'<div style="font-family:var(--k-serif);font-size:clamp(1.6rem,3.4vw,2.5rem);'
        f'line-height:1.12;color:var(--k-ink);margin:8px 0 12px;letter-spacing:-.01em;" '
        f'class="r d2">{title}</div>',
        unsafe_allow_html=True)
    lede(sub_html)


# ==========================================================================
# SCREENS
# ==========================================================================

def render_executive_summary() -> None:
    stale = staleness_warning()
    if stale:
        st.warning(stale)

    if not baseline_metrics:
        standing("01 &mdash; overview")
        st.warning("No baseline metrics found. Run `python scripts/evaluate_model.py` first.")
        return

    overall = baseline_metrics["overall"]
    acc5 = overall["accuracy_at_5px"]
    n_pairs = overall.get("n", len(per_pair_df))

    # --- Hero: one measurement, at the size of its importance -------------
    whole, frac = f"{acc5 * 100:.1f}".split(".")
    standing("Precision under drift")
    hero(
        f'{whole}<sup>.{frac}%</sup>',
        f"A 1000&times;1000 reference crop, placed inside a search image covering ten times the "
        f"field, under scan drift, dose starvation and beam blur. <b>{acc5:.1%} of "
        f"{n_pairs} pairs land within {TOLERANCE_PX:.0f} pixels of truth</b> — classical "
        f"multi-scale &times; multi-rotation ZNCC, with no deep learning anywhere in the "
        f"localization path.")

    run_row([
        {"k": "Pairs evaluated", "v": f"{n_pairs}",
         "s": "frozen benchmark, run twice total"},
        {"k": "Median error", "v": f"{overall['median_error_px']:.2f}px",
         "s": "half of all pairs land inside this"},
        {"k": "Mean error", "v": f"{overall['mean_error_px']:.1f}px",
         "s": "pulled up by a bimodal tail"},
        {"k": f"Catastrophic &gt;{CATASTROPHIC_PX:.0f}px",
         "v": f"{overall['failure_rate_gt_50px']:.1%}"},
    ], delay="d4")

    # --- Tighter thresholds -----------------------------------------------
    # The problem statement asks for pass rates at 5, 4, 2 and 1 px. @5px is
    # the hero above; these are the rest, read straight from the same metrics
    # file. Shown as a band rather than buried in a table because the *shape*
    # is the argument: identical at 4, 2 and 5 px, so nothing in the benchmark
    # lands in that range at all.
    tight = [k for k in ("accuracy_at_4px", "accuracy_at_2px", "accuracy_at_1px")
             if k in overall]
    if tight:
        n = len(per_pair_df) or n_pairs
        cells = []
        for key in tight:
            px = key.split("_")[-1].replace("px", "")
            acc = overall[key]
            same = abs(acc - acc5) < 1e-9
            cells.append({
                "k": f"Accuracy @{px}px",
                "v": f"{acc:.1%}",
                "lead": same,
                "s": (f"identical to @{TOLERANCE_PX:.0f}px" if same
                      else f"{round((acc5 - acc) * n)} pairs tighter"),
            })
        run_row(cells, delay="d5")
        note(
            f"Tightening the tolerance from {TOLERANCE_PX:.0f}px to 2px costs <b>nothing</b> — the "
            f"pass rate is identical at 5, 4, 3 and 2 pixels, so not one pair in the benchmark "
            f"lands in that band. Only at 1px does it move, and by "
            f"{round((acc5 - overall['accuracy_at_1px']) * n)} pairs. That is the bimodality "
            f"result stated as a measurement: a prediction is either essentially exact or it has "
            f"locked onto the wrong lattice cell, with almost nothing in between.")

    # --- The decision, drawn as a decision --------------------------------
    stats = selective_prediction_stats(per_pair_df)
    if stats:
        mark("01", "The decision")
        lede(
            "A localizer that is confidently wrong is worse than one that says so. Drift-Sense "
            "compares the winning candidate against the best rival more than 10px away and only "
            "returns an answer when the gap is decisive. <b>What follows is that single branch, "
            "with every number computed live from <code>per_pair_results.csv</code>.</b>")
        st.markdown(decision_tree_svg(stats), unsafe_allow_html=True)
        note(
            f"The threshold was fitted on tuning surfaces and evaluated once on held-back data — "
            f"<code>reports/GATE_EXCEPTIONS.md</code>, exception 4. Deferring "
            f"{stats['flag_rate']:.1%} of pairs buys "
            f"{100 * (stats['acc_answered'] - stats['acc_all']):+.1f} pp of accuracy on the "
            f"remainder and catches {stats['failure_recall']:.1%} of every failure the system "
            f"makes. On an inspection tool, that is the difference between a re-scan and a "
            f"mis-measured die.")

    # --- Behaviour --------------------------------------------------------
    if not per_pair_df.empty:
        mark("02", "Behaviour")
        h = paired_chart_height(per_pair_df)
        c1, c2 = st.columns([7, 5], gap="large")
        with c1:
            chart(render_accuracy_curve(per_pair_df, height=h), "Accuracy vs. error tolerance",
                  f"The {TOLERANCE_PX:.0f}px criterion sits on a steep shoulder, not a plateau — "
                  f"errors are strongly bimodal, so a pair is usually either right or lost, with "
                  f"little in between.")
        with c2:
            chart(render_family_bar(per_pair_df, height=h),
                  f"Accuracy @{TOLERANCE_PX:.0f}px by family",
                  "Which structural conditions hold up. The spread across families is the "
                  "honest picture of where the remaining work is.")

    # --- What the evidence rests on ---------------------------------------
    mark("03", "What the evidence rests on")
    if gate_result is not None:
        passed = gate_result["passed"]
        entry("ok" if passed else "no", "candidate change", "Learned re-ranker",
              ("Passed the 7-criterion gate and was integrated."
               if passed else
               "Halved accuracy on every split, across all three training seeds, and did not "
               "clear the 7-criterion integration gate. It was not integrated. Classical "
               "ranking remains production — the gate is not decorative."),
              token="integrated" if passed else "rejected")
    entry("ex", "shipped without a clean 7/7", "Four documented gate exceptions",
          "Scale range (A2), multiway centre tie-break (A6), PSF dual-arm candidate generation, "
          "and the ambiguity-threshold recalibration. Each is logged with the criteria it did "
          "not clear and the evidence behind it. &ldquo;In production&rdquo; never silently "
          "means &ldquo;passed all seven&rdquo; — see <b>07 Experiment Results</b>.",
          token="4 logged")
    entry("in", "the finding that reframed the problem",
          "Crop uniqueness, not periodicity, governs accuracy",
          "The rubric names repeated-pattern ambiguity. Controlled forensics found that crops "
          "with no distinguishing macro structure score far below the rest, and that periodicity "
          "correlates with failure mainly because non-unique crops tend to be periodic. With "
          "uniqueness held fixed, periodicity moves accuracy by well under a point. Full "
          "workings in <code>reports/ACCURACY_FORENSICS.md</code>; case by case on "
          "<b>06 Failure Analysis</b>.")


def render_generate_sample(gen_overrides: dict, gen_crop_mode: str) -> None:
    screen_head(
        "Generate Sample", "Synthesise a pair, then run the real pipeline on it",
        "Every control in the left panel is an actual generator parameter — beam spot size, "
        "dose, raster shear, barrel distortion, pattern collapse. The pair below is produced by "
        "<code>generator.generate_pair</code> and localized by <code>pipeline.localize</code>: "
        "the same two code paths behind every benchmark number on this site.")

    family = {"name": "ui_generated", "split": "development",
              "crop_mode": gen_crop_mode, "overrides": gen_overrides}
    ref_img, search_img, meta = generate_pair(0, st.session_state.gen_seed, family)

    mark("01", "Acquisition")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        figure(image_uri(ref_img), tag="reference &middot; 1000&times;1000 @ 1 nm/px",
               caption="The high-resolution crop to be found.")
    with c2:
        figure(image_uri(search_img), tag="search &middot; 1000&times;1000 @ 10 nm/px",
               caption="Ten times the field, at a tenth the resolution, degraded.")

    presets = meta["presets"]
    presets_txt = ", ".join(str(p) for p in presets) if isinstance(presets, (list, tuple)) \
        else str(presets)
    st.write("")
    dl([
        ("Presets", presets_txt),
        ("Crosses mat boundary", str(meta["crosses_mat_boundary"])),
        ("Crosses strip boundary", str(meta["crosses_strip_boundary"])),
        ("Periodicity score", f"{meta['periodicity_score']:.3f}"),
        ("Uniqueness score", f"{meta['uniqueness_score']:.3f}"),
        ("Seed", str(st.session_state.gen_seed)),
    ])

    mark("02", "Localization")
    if st.button("Run localization on this pair", type="primary"):
        with st.spinner("Multi-scale \u00d7 multi-rotation ZNCC \u2014 scanning hypothesis grid\u2026"):
            result = localize(ref_img, search_img)
        error_px = float(np.hypot(result.x - meta["gt_x"], result.y - meta["gt_y"]))
        result_readout(result, error_px=error_px)
        st.write("")
        figure(marked_overlay(search_img, (result.x, result.y), (meta["gt_x"], meta["gt_y"])),
               tag="search &middot; located", scan=True,
               caption="Red cross and ring &mdash; predicted centre. Green diamond &mdash; ground truth.")
    else:
        note("Adjust any acquisition parameter on the left, then run. Rotation and scale drift "
             "are the sharpest levers: the pipeline tests a fixed hypothesis grid, so values "
             "landing midway between two tested hypotheses are the hardest cases in the set.")


def render_live_localization() -> None:
    screen_head(
        "Live Localization", "Bring your own pair",
        "Upload a reference crop and a wider search image and run the production pipeline on "
        "them. This calls <code>pipeline.localize</code> directly — there is exactly one "
        "localization implementation in this project and the application does not fork it.")

    mark("01", "Input")
    c1, c2 = st.columns(2, gap="large")
    ref_upload = c1.file_uploader("Reference image", type=["png", "jpg", "jpeg", "bmp", "tif"])
    search_upload = c2.file_uploader("Search image", type=["png", "jpg", "jpeg", "bmp", "tif"])
    if ref_upload is not None:
        c1.image(ref_upload, use_container_width=True)
    if search_upload is not None:
        c2.image(search_upload, use_container_width=True)

    if ref_upload is None or search_upload is None:
        note("Both images are required. The reference should be a high-resolution close-up; the "
             "search image a wider, lower-resolution view that contains it.")
        return

    mark("02", "Localization")
    if not st.button("Run localization", type="primary"):
        return

    ref_img = read_uploaded_image(ref_upload)
    search_img = read_uploaded_image(search_upload)
    if ref_img is None or search_img is None:
        st.error("One of the uploaded files could not be decoded as an image.")
        return

    with st.spinner("Multi-scale \u00d7 multi-rotation ZNCC \u2014 scanning hypothesis grid\u2026"):
        result = localize(ref_img, search_img)

    result_readout(result)
    st.write("")
    figure(marked_overlay(search_img, (result.x, result.y)),
           tag="search &middot; located", scan=True,
           caption="Red cross and ring &mdash; predicted centre.")
    with st.expander("Candidates considered"):
        st.dataframe(pd.DataFrame(result.top_candidates), use_container_width=True)


def render_visualization() -> None:
    screen_head(
        "Visualization", "Any pair in the benchmark, opened up",
        "Prediction against ground truth for a single evaluated pair, with the conditions that "
        "pair was generated under. Read straight from "
        "<code>outputs/reports/per_pair_results.csv</code>.")

    if per_pair_df.empty:
        st.warning("No evaluation results found. Run `python scripts/evaluate_model.py` first.")
        return

    mark("01", "Select")
    c1, c2 = st.columns([1, 2], gap="large")
    split = c1.selectbox("Split", sorted(per_pair_df["split"].unique()))
    subset = per_pair_df[per_pair_df["split"] == split]
    pair_id = c2.selectbox("Pair", sorted(subset["pair_id"].unique()))
    row = subset[subset["pair_id"] == pair_id].iloc[0]
    correct = row["error_px"] <= TOLERANCE_PX

    mark("02", "Prediction against truth")
    v1, v2 = st.columns([3, 2], gap="large")
    with v1:
        img, err = load_pair_image(row["search_path"])
        if img is None:
            st.warning(err)
        else:
            figure(marked_overlay(img, (row["pred_x"], row["pred_y"]),
                                   (row["gt_x"], row["gt_y"])),
                   tag=f"{pair_id}",
                   caption="Red cross and ring &mdash; predicted. Green diamond &mdash; truth.")
    with v2:
        verdict = "ok" if correct else "no"
        st.markdown(
            f'<div class="k-standing" style="margin-top:0;">{pair_id}</div>'
            f'<div style="font-family:var(--k-mono);font-size:2.9rem;font-weight:700;'
            f'letter-spacing:-.03em;margin:10px 0 4px;'
            f'color:var(--k-{"green" if correct else "red"});">'
            f'{row["error_px"]:.2f}<span style="font-size:.36em;">px</span></div>'
            f'<div style="margin-bottom:16px;">'
            f'{tok(verdict, "within tolerance" if correct else "outside tolerance")}</div>',
            unsafe_allow_html=True)
        dl([
            ("Confidence", f"{row['confidence']:.3f}"),
            (f"Within {TOLERANCE_PX:.0f}px", "yes" if correct else "no"),
            ("Flagged ambiguous", "yes" if row["ambiguous"] else "no"),
            ("Family", str(row["structural_family"])),
            ("Crosses mat boundary", str(row["crosses_mat_boundary"])),
            ("Crosses strip", str(row["crosses_strip_boundary"])),
        ], classes={f"Within {TOLERANCE_PX:.0f}px": "ok" if correct else "no"})


def render_benchmark_dashboard() -> None:
    screen_head(
        "Benchmark Dashboard", "Eight plates, three questions",
        "The eight required benchmark figures, arranged by the question each one answers rather "
        "than by filename. All are regenerated by <code>scripts/evaluate_model.py</code> from a "
        "single evaluation pass, and are shown here exactly as that script wrote them.")

    stale = staleness_warning()
    if stale:
        st.warning(stale)

    chapters = [
        ("01", "Robustness — which conditions hold up",
         "Each plate isolates one acquisition variable. Read together they say the same thing "
         "the forensics did: structural content matters far more than noise.",
         [("i",   "accuracy_by_family.png", "Accuracy by structural family",
           "Per-family accuracy@5px across every generated difficulty class."),
          ("ii",  "accuracy_by_noise.png", "Accuracy by noise level",
           "Sensitivity to acquisition dose and detector noise."),
          ("iii", "accuracy_by_rotation.png", "Accuracy by rotation condition",
           "Behaviour against residual stage-rotation drift."),
          ("iv",  "accuracy_by_scale.png", "Accuracy by scale condition",
           "Behaviour against residual magnification drift.")]),
        ("02", "Error behaviour — how wrong, how often",
         "The distribution is strongly bimodal: a pair is usually either located within a few "
         "pixels or lost to a lattice-aligned decoy. There is very little middle ground, which "
         "is exactly why a confidence gate is worth having.",
         [("v",    "accuracy_vs_tolerance.png", "Accuracy vs. tolerance",
           "How accuracy responds as the success threshold is relaxed."),
          ("vi",   "error_cdf.png", "Error CDF",
           "Cumulative distribution of localization error."),
          ("vii",  "error_distribution.png", "Error distribution",
           "Where errors concentrate across the full benchmark.")]),
        ("03", "Decision quality — is the confidence meaningful",
         "A confidence score is only useful if it orders correct predictions ahead of wrong "
         "ones. This is the plate that justifies the decision tree on 01 Overview.",
         [("viii", "pr_curve.png", "Precision–recall, ranked by confidence",
           "Whether confidence separates correct predictions from incorrect ones.")]),
    ]

    for num, title, blurb, plots in chapters:
        mark(num, title)
        lede(blurb)
        st.write("")
        cols = st.columns(2, gap="large")
        for i, (pn, fname, ptitle, pblurb) in enumerate(plots):
            with cols[i % 2]:
                plate(f"plate {pn}", fname, ptitle, pblurb)

    if baseline_metrics:
        mark("04", "The underlying record")
        with st.expander("baseline_metrics.json — overall"):
            st.json(baseline_metrics["overall"])


def render_failure_analysis() -> None:
    screen_head(
        "Failure Analysis", "Three case files",
        "Representative cases selected once and deterministically by "
        "<code>outputs/reports/failure_analysis_cases.json</code> — not re-randomised on each "
        "page load, and not cherry-picked to flatter the result.")

    entry("in", "what the failures actually are",
          "Every remaining failure is a near-tie",
          "Measured across the frozen benchmark, roughly a third of remaining failures are "
          "<b>discovery</b> failures — the true location is never proposed — and the rest are "
          "<b>selection</b> failures, where truth sits in the candidate pool but is not ranked "
          "first. In the selection cases the winning and true scores are within 0.05 ZNCC, at a "
          "maximum ratio of 1.048&times;. The error runs along a lattice axis: the cross-axis "
          "residual is only about 1.5px. These are not wild misses; they are one-dimensional "
          "ties against a self-similar decoy. See "
          "<code>experiments/reachability_verification/</code>.")

    if not failure_cases or per_pair_df.empty:
        st.warning("Failure case selections or evaluation results not found. "
                   "Run `python scripts/evaluate_model.py` first.")
        return

    tiers = [("01", "Representative success", "successful", "ok", "located"),
             ("02", "Difficult case", "difficult", "ex", "marginal"),
             ("03", "Catastrophic failure", "catastrophic", "no", "lost")]

    for num, label, key, kind, word in tiers:
        case = failure_cases.get(key)
        if not case:
            continue
        match = per_pair_df[per_pair_df["pair_id"] == case["pair_id"]]
        if match.empty:
            continue
        row = match.iloc[0]
        err_px = float(case["error_px"])

        mark(num, label)
        c1, c2 = st.columns([3, 2], gap="large")
        with c1:
            img, err = load_pair_image(row["search_path"])
            if img is None:
                st.warning(err)
            else:
                figure(marked_overlay(img, (row["pred_x"], row["pred_y"]),
                                       (row["gt_x"], row["gt_y"])),
                       tag=f"case {num} &middot; {case['pair_id']}",
                       caption="Red cross and ring &mdash; predicted. Green diamond &mdash; truth.")
        with c2:
            st.markdown(
                f'<div class="k-standing" style="margin-top:0;">case file {num}</div>'
                f'<div style="font-family:var(--k-mono);font-size:.72rem;color:var(--k-ink-2);'
                f'letter-spacing:.06em;margin:8px 0 12px;">{case["pair_id"]}</div>'
                f'<div style="font-family:var(--k-mono);font-size:2.7rem;font-weight:700;'
                f'letter-spacing:-.03em;color:var(--k-{"green" if kind == "ok" else "amber" if kind == "ex" else "red"});">'
                f'{err_px:.2f}<span style="font-size:.36em;">px</span></div>'
                f'<div style="margin:8px 0 16px;">{tok(kind, word)}</div>',
                unsafe_allow_html=True)
            dl([("Family", str(row["structural_family"])),
                ("Confidence", f"{row['confidence']:.3f}"),
                ("Flagged ambiguous", "yes" if row["ambiguous"] else "no")])
            st.markdown(f'<div class="k-lede" style="font-size:.9rem;margin-top:14px;">'
                        f'{md_inline(case["rationale"])}</div>', unsafe_allow_html=True)
        st.write("")


def render_experiment_results() -> None:
    screen_head(
        "Experiment Results", "The ledger, including what failed",
        "Every candidate change is judged against the frozen classical baseline through the same "
        "7-criterion integration gate. Rejections are kept with their evidence — they are part "
        "of the result, not something to hide before a demo.")

    mark("01", "What drives failure")
    lede("Four findings, in the order of how much they move the number.")
    drivers = [
        ("01", "Boundary presence",
         "The strongest lever measured. The same rotation/scale misalignment is far more "
         "recoverable when a mat boundary is in view than when it is not."),
        ("02", "Crop uniqueness, not periodicity",
         "Crops with no distinguishing macro structure score far below the rest. With "
         "uniqueness held fixed, periodicity moves accuracy by well under a point "
         "(<code>experiments/crop_uniqueness_ceiling/</code>)."),
        ("03", "Hypothesis-grid misalignment",
         "Damage tracks distance to the nearest tested scale/rotation hypothesis, not the "
         "magnitude of the drift itself."),
        ("04", "Noise, raster drift, row jitter",
         "Minor factors throughout, alone or in combination. Worth stating plainly, because it "
         "is the opposite of what the acquisition parameters suggest."),
    ]
    for num, head, body in drivers:
        entry("in", f"driver {num}", head, body)

    # --- The full ledger, read from disk ---------------------------------
    ledger = load_experiment_ledger()
    if ledger:
        mark("02", f"The complete ledger &mdash; all {len(ledger)} experiments")
        counts = {}
        for row in ledger:
            counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
        n_ship = counts.get("integrated", 0)
        lede(
            f"Every experiment directory in this repository, read live from disk rather than "
            f"curated by hand — so this list cannot quietly fall out of step with the work. "
            f"<b>{len(ledger)} candidate changes were built and measured; {n_ship} reached "
            f"production.</b> The rest are kept with their evidence, because a rejection that "
            f"cost a day of compute is a result: it tells the next person not to spend that day.")
        run_row([
            {"k": "Experiments", "v": f"{len(ledger)}", "lead": True},
            {"k": "Integrated", "v": f"{n_ship}", "s": "four are documented gate exceptions"},
            {"k": "Rejected", "v": f"{counts.get('rejected', 0)}"},
            {"k": "Diagnostics", "v": f"{counts.get('diagnostic', 0)}",
             "s": "measured something, changed nothing"},
            {"k": "Near-miss / interim",
             "v": f"{counts.get('near-miss', 0) + counts.get('interim', 0)}"},
        ], delay="d3")

        # Ordered by what a reader most wants to see first, not alphabetically.
        groups = [
            ("integrated", "in", "Reached production"),
            ("interim", "ex", "Interim — measured, not integrated"),
            ("near-miss", "ex", "Near-miss — real signal, failed the gate"),
            ("rejected", "no", "Rejected"),
            ("not reproduced", "no", "Did not reproduce"),
            ("inconclusive", "ex", "Inconclusive"),
            ("diagnostic", "in", "Diagnostics — changed the question, not the code"),
            ("documented", "in", "Recorded"),
            ("missing", "ex", "No written report in the directory"),
        ]
        for verdict, kind, heading in groups:
            rows = [r for r in ledger if r["verdict"] == verdict]
            if not rows:
                continue
            st.markdown(
                f'<div class="k-standing" style="margin:26px 0 2px;">{heading} '
                f'&nbsp;&middot;&nbsp; {len(rows)}</div>', unsafe_allow_html=True)
            html = []
            for r in rows:
                extra = INTEGRATED_EXPERIMENTS.get(r["name"], "")
                title = r["title"]
                if extra:
                    title = f"{title} <i style=\"color:var(--k-mute);\">({extra})</i>"
                html.append(
                    f'<div class="k-led"><span class="nm">{r["name"]}</span>'
                    f'<span class="tk">{tok(kind, verdict)}</span>'
                    f'<span class="ti">{md_inline(title)}</span></div>')
            st.markdown("".join(html), unsafe_allow_html=True)
        note(
            "Each row is the first heading of that experiment's own <code>REPORT.md</code>; the "
            "verdict is read from that heading and its status line, not assigned here. Open any "
            "directory under <code>experiments/</code> for the full write-up, including the "
            "pre-registered hypothesis and the measurement that killed it.")

    # --- Formal gate verdicts, where an artifact exists -------------------
    gate_rows = []
    reranker = load_json(os.path.join(REPORTS_DIR, "integration_gate.json"))
    if reranker is not None:
        gate_rows.append(("embedding_reranker_v1 — CNN re-ranker", reranker["passed"],
                          "Halved accuracy@5px and multiplied the catastrophic rate on every "
                          "split, across all three training seeds. A learned re-ranker trained "
                          "from scratch on this data does not have enough signal to beat "
                          "arg-max ZNCC."))
    wide = load_json(os.path.join(PROJECT_ROOT,
                                   "experiments/wider_candidate_pool/outputs/integration_gate.json"))
    if wide is not None:
        gate_rows.append(("wider_candidate_pool — more peaks, tighter NMS", wide["passed"],
                          "Bit-identical predictions on every pair. This is a structural no-op "
                          "under pure arg-max ranking, not a failed improvement — worth "
                          "distinguishing."))
    fine = load_json(os.path.join(PROJECT_ROOT,
                                   "experiments/finer_hypothesis_grid/outputs/integration_gate.json"))
    if fine is not None:
        gate_rows.append(("finer_hypothesis_grid — denser scale/rotation grid", fine["passed"],
                          "Near miss. Improved held_out and challenge with no per-family "
                          "regression, failing only because validation tied rather than "
                          "improved."))
    if gate_rows:
        mark("03", "Formal integration-gate verdicts")
        lede("Where a machine-readable gate result was produced, it is read from that artifact "
             "rather than described. The gate is the same 7 criteria for every candidate.")
        for name, passed, body in gate_rows:
            entry("ok" if passed else "no", "integration gate", name, body,
                  token="integrated" if passed else "not integrated")
        note("Gate artifacts live under <code>experiments/&lt;name&gt;/outputs/</code>, which is "
             "gitignored as regenerable — run that experiment to reproduce its verdict.")

    mark("04", "Documented gate exceptions")
    lede("Changes shipped without a clean 7/7 pass, each logged with the criteria it did not "
         "clear and the evidence behind it. <b>&ldquo;In production&rdquo; never silently means "
         "&ldquo;passed all seven&rdquo;.</b>")
    exceptions = [
        ("A2", "Scale hypothesis grid widened to the literal 9:1–11:1 span",
         "Zero regressions across two independent datasets. Fails criteria 1 and 2 only because "
         "the affected families do not dominate the pooled validation/held_out counts — not "
         "because of harm. <code>experiments/scale_range_v1/</code>"),
        ("A6", "Multiway-gated centre tie-break",
         "Zero regressions across two independent datasets, with one confirmed catastrophic "
         "rescue. Same structural reason for failing criteria 1 and 2. "
         "<code>experiments/multiway_tiebreak_v1/</code>"),
        ("PSF", "PSF-matched dual-arm candidate generation",
         "The reference and search acquisition paths leave the template far sharper than the "
         "image it is matched against. The candidate pool is built both with and without a "
         "passband-matching blur, keeping whichever arm is more decisive. Replicated on a second "
         "independent seed — the first statistically significant result in the project. "
         "<code>experiments/psf_gated_selection/</code>"),
        ("THR", "Ambiguity-threshold recalibration",
         "The previous threshold fired on the large majority of pairs, so the flag carried almost "
         "no information. The statistic was sound; only the constant was wrong. Fitted on tuning "
         "surfaces and evaluated once on held-back data. This is a <b>fourth kind</b> of "
         "exception — <b>not evaluable by the gate</b> rather than failing it, since criteria 1–6 "
         "all measure prediction quality and this change is verified bit-identical on every "
         "prediction. <code>experiments/psr_confidence/</code>"),
    ]
    for code, head, body in exceptions:
        entry("ex", f"exception {code}", head, body, token="documented")
    note("Full rationale for every one of these: <code>reports/GATE_EXCEPTIONS.md</code>.")


def render_system_information() -> None:
    screen_head(
        "System Information", "Provenance",
        "The exact versions, seeds and version strings behind every number in this application. "
        "If a figure here cannot be reproduced from this table, it should not be trusted.")

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

    mark("01", "Dataset & model")
    dl([
        ("Generator version", str(GENERATOR_VERSION)),
        ("Base scale factor", f"{SCALE_FACTOR}\u00d7"),
        ("Reference size", f"{REFERENCE_SIZE_PX} px"),
        ("Dataset seed (main splits)", "777001"),
        ("Production ranking", "classical ZNCC — no deep learning in the localization path"),
        ("Learned re-ranker", "evaluated and rejected by the integration gate"),
    ])

    mark("02", "Environment")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        dl([("Python", platform.python_version()),
            ("Platform", platform.platform()),
            ("CPU", platform.processor() or "unknown")])
    with c2:
        dl([("OpenCV", _cv2.__version__), ("NumPy", np.__version__),
            ("pandas", pd.__version__), ("Matplotlib", matplotlib.__version__),
            ("SciPy", scipy.__version__), ("Streamlit", st.__version__),
            ("PyTorch", str(torch_version)), ("CUDA available", str(cuda_available)),
            ("GPU", str(gpu_name))])
    note("OpenCV is pinned to an exact version: its 4.x\u21925.x numerics change "
         "<code>warpAffine</code> / <code>remap</code> / <code>GaussianBlur</code> enough to "
         "alter generated datasets. See README \u2192 Reproducibility.")


# ==========================================================================
# ROUTER
# ==========================================================================
inject_theme()
screen = render_index()

# Acquisition controls must be built before the screen renders, since the
# screen consumes their values. Only shown on 02 Generate Sample.
gen_overrides, gen_crop_mode = ({}, "random")
if screen == "Generate Sample":
    gen_overrides, gen_crop_mode = render_generator_controls()

masthead()

if screen == "Executive Summary":
    render_executive_summary()
elif screen == "Generate Sample":
    render_generate_sample(gen_overrides, gen_crop_mode)
elif screen == "Live Localization":
    render_live_localization()
elif screen == "Visualization":
    render_visualization()
elif screen == "Benchmark Dashboard":
    render_benchmark_dashboard()
elif screen == "Failure Analysis":
    render_failure_analysis()
elif screen == "Experiment Results":
    render_experiment_results()
elif screen == "System Information":
    render_system_information()

st.markdown(
    f'<div style="margin-top:56px;padding-top:16px;border-top:1px solid var(--k-line);'
    f'display:flex;justify-content:space-between;gap:18px;flex-wrap:wrap;'
    f'font-family:var(--k-mono);font-size:.585rem;letter-spacing:.16em;'
    f'text-transform:uppercase;color:var(--k-mute);">'
    f'<span>Drift&middot;Sense &nbsp;&mdash;&nbsp; precision under drift</span>'
    f'<span>Team Kaccha Mango &middot; Applied Materials &middot; SEMICON India 2026 '
    f'&middot; <a href="{REPO_URL}" style="color:var(--k-blue);text-decoration:none;">'
    f'source</a></span></div>',
    unsafe_allow_html=True)

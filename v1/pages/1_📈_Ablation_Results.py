"""Ablation Results — separate Streamlit page.

Streamlit auto-discovers files in the `pages/` directory and adds them to
the sidebar navigation. The numeric prefix in the filename controls order.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from _styles import inject_css, hero_header

st.set_page_config(
    page_title="Ablation Results — Probabilistic CRAG",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
hero_header(subtitle="Ablation results · v1 / v2 / v3 · Llama-3.1-8B & Gemini-2.5-Flash generators")

# ---------------------------------------------------------------------------
# Version selector
# ---------------------------------------------------------------------------
# Each version of the system has its own ablation results CSV. v1 is the
# version presented in the live class demo (22 questions). v2 and v3 are
# the iteration ablation runs on the expanded 44-question eval bank.
# Default to v1 so this page loads to exactly what was shown in the demo
# recording when no specific version is selected.

VERSION_RESULTS: dict[str, dict] = {
    "v1 — presented in demo (22q)": {
        "csv": "data/eval_results_v1_22q.csv",
        "n_questions": 22,
        "show_v1_findings": True,
        "blurb": "Original Probabilistic CRAG results as presented in our class demo and recorded video. 22 questions × 4 conditions.",
    },
    "v1 — 44q rerun": {
        "csv": "data/eval_results_v1_44q_judged.csv",
        "n_questions": 44,
        "show_v1_findings": False,
        "blurb": "v1 system code re-evaluated on the expanded 44-question eval bank for direct apples-to-apples comparison with v2 and v3. Same crag_pipeline.py as v1 demo; harmonized evaluation harness. Falls back to data/eval_results.csv if the labeled version isn't saved yet.",
    },
    "v2 — 44q with web noise": {
        "csv": "data/eval_results_v2_websnoise.csv",
        "n_questions": 44,
        "show_v1_findings": False,
        "blurb": "Iteration with modality-aware decomposition + per-sub-question coverage check + numerical fidelity check + bootstrap CIs. Web fallback still fires on decomposed sub-questions (the un-fixed behavior).",
    },
    "v3 — 44q web denoised (recommended)": {
        "csv": "data/eval_results_v3_judged.csv",
        "n_questions": 44,
        "show_v1_findings": False,
        "blurb": "v2 + the `_suppress_web=True` fix that prevents off-topic web content from polluting decomposed-query answers. Recommended version. See ../README.md for the v2→v3 delta breakdown.",
    },
}

selected_version = st.selectbox(
    "Select a version to view its ablation results",
    list(VERSION_RESULTS.keys()),
    index=0,
    help="Default = v1 (the version presented in the live demo). v2 and v3 are the iteration ablation runs on the expanded 44-question eval bank.",
)
version_meta = VERSION_RESULTS[selected_version]
results_path = Path(version_meta["csv"])
n_questions = version_meta["n_questions"]
show_v1_findings = version_meta["show_v1_findings"]

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.markdown(
    f"<div style='display:flex; align-items:baseline; justify-content:space-between; "
    f"margin-bottom:0.6rem; border-bottom:2px solid #1E2761; padding-bottom:0.5rem;'>"
    f"<h2 style='margin:0; font-size:1.4rem;'>Ablation Study Results — {selected_version}</h2>"
    f"<span style='color:#94A3B8; font-size:0.85rem;'>{n_questions} questions · 4 conditions</span>"
    f"</div>",
    unsafe_allow_html=True,
)

st.markdown(
    f"<div style='font-size:0.85rem; color:#475569; line-height:1.55; margin-bottom:1rem;'>"
    f"{version_meta['blurb']}</div>",
    unsafe_allow_html=True,
)

if not results_path.exists():
    st.markdown(
        f"<div style='background:#FEF3C7; border:1px solid #F4A261; "
        f"padding:1rem 1.25rem; border-radius:0.5rem; color:#78350F; margin-top:1rem;'>"
        f"Results for <b>{selected_version}</b> not found at <code>{results_path}</code>. "
        f"For v1, run <code>python evaluate.py</code> from this folder; "
        f"for v2/v3, those CSVs live in their respective folders' <code>data/</code> directory. "
        f"Switch back to <i>v1 — presented in demo</i> in the dropdown above to see the recorded results."
        f"</div>",
        unsafe_allow_html=True,
    )
    st.stop()

df = pd.read_csv(results_path)

# ---------------------------------------------------------------------------
# Per-condition KPI tiles
# ---------------------------------------------------------------------------

summary = (
    df.groupby("condition")
    .agg(
        avg_keyword_hit=("keyword_hit_rate", "mean"),
        routing_precision=("routing_correct", lambda x: x.dropna().mean()),
        avg_latency_ms=("latency_ms", "mean"),
    )
    .round(3)
    .reset_index()
)

PALETTE = {
    "baseline_no_tables": "#94A3B8",
    "baseline_tables":    "#94A3B8",
    "crag_no_tables":     "#475569",
    "crag_tables":        "#2A9D8F",  # winner accent
}

st.markdown(
    "<div style='font-size:0.78rem; color:#94A3B8; letter-spacing:0.08em; "
    "text-transform:uppercase; margin-top:1rem; margin-bottom:0.5rem; font-weight:600;'>"
    "Headline metrics by condition</div>",
    unsafe_allow_html=True,
)

kpi_cols = st.columns(len(summary))
for kc, (_, r) in zip(kpi_cols, summary.iterrows()):
    cond = r["condition"]
    accent = PALETTE.get(cond, "#94A3B8")
    khr = r["avg_keyword_hit"]
    rp = r["routing_precision"]
    rp_str = f"{rp:.2f}" if pd.notna(rp) else "—"
    is_winner = (cond == "crag_tables")
    with kc:
        st.markdown(
            f"<div style='background:white; border:1px solid #E2E8F0; "
            f"border-top:4px solid {accent}; padding:1rem 1.1rem; border-radius:0.5rem; "
            f"box-shadow: 0 1px 3px rgba(0,0,0,0.04); height:160px;'>"
            f"<div style='font-size:0.7rem; color:#94A3B8; letter-spacing:0.08em; "
            f"text-transform:uppercase; font-weight:600;'>{cond.replace('_', ' ')}"
            f"{' ★' if is_winner else ''}</div>"
            f"<div style='font-size:1.95rem; font-weight:700; color:{accent}; "
            f"margin-top:0.4rem; line-height:1; font-family: \"Source Serif Pro\", serif;'>"
            f"{khr:.3f}</div>"
            f"<div style='font-size:0.75rem; color:#64748B; margin-top:0.2rem;'>"
            f"keyword hit rate</div>"
            f"<div style='font-size:0.78rem; color:#475569; margin-top:0.5rem;'>"
            f"Routing prec: <b>{rp_str}</b></div>"
            f"<div style='font-size:0.78rem; color:#475569;'>"
            f"Latency: <b>{r['avg_latency_ms']:.0f} ms</b></div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Two side-by-side charts: KHR by condition (overall) and KHR by question type
# ---------------------------------------------------------------------------

ch_left, ch_right = st.columns(2, gap="large")

with ch_left:
    st.markdown(
        "<div style='font-size:0.78rem; color:#94A3B8; letter-spacing:0.08em; "
        "text-transform:uppercase; margin-bottom:0.4rem; font-weight:600;'>"
        "Keyword hit rate vs routing precision</div>",
        unsafe_allow_html=True,
    )
    fig1 = go.Figure()
    fig1.add_bar(
        x=summary["condition"],
        y=summary["avg_keyword_hit"],
        name="Keyword hit rate",
        marker_color="#1E2761",
    )
    fig1.add_bar(
        x=summary["condition"],
        y=summary["routing_precision"],
        name="Routing precision",
        marker_color="#2A9D8F",
    )
    fig1.update_layout(
        barmode="group",
        height=360,
        margin=dict(t=10, b=20, l=10, r=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, tickfont=dict(color="#475569")),
        yaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#475569"), range=[0, 1.0]),
    )
    st.plotly_chart(fig1, use_container_width=True)

with ch_right:
    st.markdown(
        "<div style='font-size:0.78rem; color:#94A3B8; letter-spacing:0.08em; "
        "text-transform:uppercase; margin-bottom:0.4rem; font-weight:600;'>"
        "Keyword hit rate by question type</div>",
        unsafe_allow_html=True,
    )
    by_type = (
        df.groupby(["condition", "question_type"])
        .agg(avg_khr=("keyword_hit_rate", "mean"))
        .reset_index()
    )
    type_labels = {1: "Type 1 (narrative + OOC)", 3: "Type 3 (table)", 4: "Type 4 (multimodal)"}
    type_colors = {1: "#CADCFC", 3: "#1E2761", 4: "#E76F51"}
    fig2 = go.Figure()
    for qt in sorted(by_type["question_type"].unique()):
        sub = by_type[by_type["question_type"] == qt]
        fig2.add_bar(
            x=sub["condition"],
            y=sub["avg_khr"],
            name=type_labels.get(qt, f"Type {qt}"),
            marker_color=type_colors.get(qt, "#94A3B8"),
        )
    fig2.update_layout(
        barmode="group",
        height=360,
        margin=dict(t=10, b=20, l=10, r=10),
        paper_bgcolor="white",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(showgrid=False, tickfont=dict(color="#475569")),
        yaxis=dict(gridcolor="#E2E8F0", tickfont=dict(color="#475569"), range=[0, 1.0]),
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Findings cards
# ---------------------------------------------------------------------------

st.markdown(
    "<div style='font-size:0.78rem; color:#94A3B8; letter-spacing:0.08em; "
    "text-transform:uppercase; margin-bottom:0.6rem; font-weight:600;'>"
    "Findings</div>",
    unsafe_allow_html=True,
)

# v1 findings are hardcoded with the specific numbers reported in the
# class demo. They only apply to the "v1 — presented in demo" view
# (22-question eval) and would be misleading numerically if shown for
# v1-44q-rerun, v2, or v3. For those, the report's headline-results
# section in ../README.md gives the up-to-date interpretation; we
# show a brief pointer here instead.
if show_v1_findings:
    findings = [
        {"icon": "✓", "color": "#2A9D8F",
         "text": "<b>CRAG-tables wins overall</b> by +17% relative (0.621 vs 0.530 best baseline)."},
        {"icon": "✓", "color": "#2A9D8F",
         "text": "<b>Type 1 (narrative + OOC):</b> CRAG-tables 0.556 vs best baseline 0.333 — <b>+67% relative.</b>"},
        {"icon": "✓", "color": "#2A9D8F",
         "text": "<b>Type 4 (multimodal synthesis):</b> CRAG-tables 0.733 vs best baseline 0.667 — <b>+10% relative.</b>"},
        {"icon": "≈", "color": "#F4A261",
         "text": "<b>Type 3 (table lookup):</b> tied — naive RAG is strong because numerical figures appear redundantly across MD&A and statements."},
        {"icon": "✓", "color": "#2A9D8F",
         "text": "<b>91% routing precision</b> on CRAG-tables — a quality signal only the corrective architecture can produce."},
    ]
    for f in findings:
        st.markdown(
            f"<div style='display:flex; align-items:flex-start; background:white; "
            f"border:1px solid #E2E8F0; border-left:4px solid {f['color']}; "
            f"padding:0.85rem 1.1rem; border-radius:0.4rem; margin-bottom:0.5rem; "
            f"box-shadow: 0 1px 2px rgba(0,0,0,0.03);'>"
            f"<div style='font-size:1.2rem; color:{f['color']}; font-weight:700; "
            f"margin-right:0.7rem; min-width:1.3rem;'>{f['icon']}</div>"
            f"<div style='color:#1E293B; font-size:0.92rem; line-height:1.45;'>"
            f"{f['text']}</div></div>",
            unsafe_allow_html=True,
        )
else:
    st.markdown(
        "<div style='background:#F1F5F9; border:1px solid #CBD5E1; "
        "padding:0.85rem 1.1rem; border-radius:0.4rem; color:#475569; "
        "font-size:0.88rem; line-height:1.5;'>"
        "Findings for this version (v1-44q rerun, v2, or v3) are reported in the "
        "project root <code>README.md</code> under <i>Headline results</i>, "
        "with bootstrap 95% CIs, numerical fidelity, and sub-question coverage "
        "metrics. The KPI tiles and charts above use the <i>same</i> CSV that "
        "produced those numbers — they will agree."
        "</div>",
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Per-question detail (collapsible)
# ---------------------------------------------------------------------------

with st.expander("📋 Per-question detail (88 rows)"):
    # Surface only the most useful columns for readability
    display_cols = [
        "condition", "question_id", "question_type", "query_type",
        "routing_decision", "tier_used",
        "keyword_hit_rate", "routing_correct",
        "confidence_score", "latency_ms",
        "answer_snippet",
    ]
    cols_present = [c for c in display_cols if c in df.columns]
    st.dataframe(df[cols_present], use_container_width=True, height=420)

# ---------------------------------------------------------------------------
# Sidebar — minimal nav cue
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        "<div style='border-left:3px solid #1E2761; padding-left:0.7rem; margin-bottom:1.2rem;'>"
        "<div style='font-family:\"Source Serif Pro\", serif; font-size:1.15rem; "
        "font-weight:700; color:#1E2761;'>Probabilistic CRAG</div>"
        "<div style='font-size:0.72rem; color:#94A3B8; letter-spacing:0.06em; "
        "text-transform:uppercase;'>Ablation results</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<div style='font-size:0.85rem; color:#475569; line-height:1.55;'>"
        "Aggregate metrics from <code>data/eval_results.csv</code> — "
        "22 questions × 4 conditions. Click <b>Query Console</b> in the navigation "
        "above to run live queries.</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='margin-top:2rem; padding-top:1rem; border-top:1px solid #E2E8F0; "
        "font-size:0.72rem; color:#94A3B8; line-height:1.5;'>"
        "<div style='font-weight:600; color:#475569;'>STAT 5293 GenAI</div>"
        "Yang · Chen · He · Spring 2026</div>",
        unsafe_allow_html=True,
    )

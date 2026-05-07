"""
Streamlit demo app — Probabilistic CRAG vs Baseline RAG side-by-side.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
import re
import time
import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dotenv import load_dotenv

from _styles import inject_css, hero_header, kpi_strip

load_dotenv()
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Citation prettification
# ---------------------------------------------------------------------------
# CRAG's pipeline tags every retrieved chunk with a raw inline marker like
# "[TEXT | data/sec_filings/sec-edgar-filings/AAPL/10-K/0000320193-25-000079/
# full-submission.txt]". The generator sometimes echoes these markers verbatim
# into the answer text, which looks unprofessional in a recorded demo. The
# helpers below convert filesystem paths to human-readable filing names and
# strip the inline markers from generated answers.

TICKER_TO_COMPANY = {
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "AMZN": "Amazon",
    "META": "Meta",
    "GOOGL": "Alphabet",
    "GOOG": "Alphabet",
}

_PATH_RX = re.compile(r"sec-edgar-filings/([A-Z]+)/(10-[KQ])/(\d+)-(\d+)-(\d+)")
_INLINE_MARKER_RX = re.compile(r"\[(?:TEXT|TABLE)\s*\|\s*([^\]]+)\]")


def format_source(path: str) -> str:
    """Convert a SEC filing path (or 'web_search') to a human-readable name.

    Examples
    --------
    'data/.../AAPL/10-K/0000320193-25-000079/full-submission.txt'
        -> 'Apple 10-K (2025)'
    'data/.../META/10-Q/0001628280-26-003942/full-submission.txt'
        -> 'Meta 10-Q (2026)'
    'web_search' -> 'Web search'
    """
    if not path:
        return ""
    if path == "web_search":
        return "Web search"
    m = _PATH_RX.search(path)
    if not m:
        return Path(path).name  # fallback to bare filename
    ticker, filing_type, _cik, yy, _seq = m.groups()
    company = TICKER_TO_COMPANY.get(ticker, ticker)
    year = f"20{yy}" if len(yy) == 2 else yy
    return f"{company} {filing_type} ({year})"


def clean_answer(text: str) -> str:
    """Strip raw filesystem-path markers from generated answers, replacing
    them with human-readable filing names so the demo doesn't show
    'data/sec_filings/sec-edgar-filings/...' inline."""
    if not text:
        return text
    return _INLINE_MARKER_RX.sub(
        lambda m: format_source(m.group(1).strip()), text
    )

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Query Console — Probabilistic CRAG",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
hero_header()
kpi_strip()

st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    # Sidebar branding strip
    st.markdown(
        "<div style='border-left:3px solid #1E2761; padding-left:0.7rem; margin-bottom:1.2rem;'>"
        "<div style='font-family:\"Source Serif Pro\", serif; font-size:1.15rem; "
        "font-weight:700; color:#1E2761;'>Probabilistic CRAG</div>"
        "<div style='font-size:0.72rem; color:#94A3B8; letter-spacing:0.06em; "
        "text-transform:uppercase;'>Demo console</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<div style='font-size:0.72rem; color:#94A3B8; letter-spacing:0.08em; "
                "text-transform:uppercase; margin-bottom:0.3rem; font-weight:600;'>"
                "RETRIEVAL</div>", unsafe_allow_html=True)
    vector_db_dir = st.text_input("Vector DB path", value="data/vectordb_crag_tables",
                                   label_visibility="collapsed")

    # Note: top_k slider was removed because it only affected baseline.
    # CRAG retrieval is governed by RETRIEVAL_COMPOSITION (per-query-type
    # text_k/table_k counts) inside crag_pipeline.py — the slider was a
    # no-op for the system the demo is showcasing. Baseline uses a fixed
    # value below to match the ablation conditions.
    BASELINE_TOP_K = 5

    st.markdown("<div style='font-size:0.72rem; color:#94A3B8; letter-spacing:0.08em; "
                "text-transform:uppercase; margin-top:0.8rem; margin-bottom:0.3rem; "
                "font-weight:600;'>ROUTING THRESHOLDS</div>", unsafe_allow_html=True)
    st.caption("Live-tunable. τ_high controls when routing reads CORRECT; "
               "τ_low gates web fallback.")
    threshold_high = st.slider("τ_high (Correct)", 0.5, 1.0, 0.75, 0.01)
    threshold_low = st.slider("τ_low (Ambiguous)", 0.1, 0.7, 0.40, 0.01)

    st.markdown("<div style='font-size:0.72rem; color:#94A3B8; letter-spacing:0.08em; "
                "text-transform:uppercase; margin-top:0.8rem; margin-bottom:0.3rem; "
                "font-weight:600;'>EVALUATOR BLEND</div>", unsafe_allow_html=True)
    st.caption("α weights the cosine-vs-cross-encoder mix in "
               "P(Relevant | q, d) = α·cos + (1-α)·σ(CE).")
    alpha = st.slider("α (cosine weight)", 0.0, 1.0, 0.40, 0.05)

    st.markdown("<div style='font-size:0.72rem; color:#94A3B8; letter-spacing:0.08em; "
                "text-transform:uppercase; margin-top:0.8rem; margin-bottom:0.3rem; "
                "font-weight:600;'>DEMO CONTROLS</div>", unsafe_allow_html=True)
    web_fallback_enabled = st.toggle(
        "Enable web fallback",
        value=True,
        help="OFF: CRAG can't access the web. OOC questions and low-confidence "
             "queries stay corpus-only — directly demonstrates what CRAG "
             "looks like without its web-fallback claim.",
    )
    query_type_override_label = st.selectbox(
        "Query-type override",
        ["Auto-classify (LLM)", "table_lookup", "narrative", "multimodal", "out_of_corpus"],
        index=0,
        help="Bypass the LLM classifier and force a specific routing path. "
             "Useful for showing per-route retrieval composition or saving "
             "the ~500 ms classifier call.",
    )
    query_type_override = (
        None if query_type_override_label == "Auto-classify (LLM)"
        else query_type_override_label
    )

    show_chunks = st.checkbox("Show retrieved chunks", value=False)

    st.divider()

    st.markdown("<div style='font-size:0.72rem; color:#94A3B8; letter-spacing:0.08em; "
                "text-transform:uppercase; margin-bottom:0.5rem; font-weight:600;'>"
                "DATA OPERATIONS</div>", unsafe_allow_html=True)
    if st.button("⬇  Download SEC filings", use_container_width=True):
        with st.spinner("Downloading from SEC EDGAR..."):
            from data_fetcher import download_filings
            results = download_filings(num_filings=1)
            st.success(f"Downloaded filings for: {', '.join(results.keys())}")

    if st.button("⟳  Rebuild FAISS indexes", use_container_width=True):
        with st.spinner("Building FAISS index..."):
            from rag_baseline import build_index_from_dir
            build_index_from_dir("data/sec_filings", table_aware=True, save_path=Path("data/vectordb_crag_tables"))
            build_index_from_dir("data/sec_filings", table_aware=False, save_path=Path("data/vectordb_baseline"))
            st.success("Indexes built!")

    # Footer attribution — normal flow, sits below the buttons
    st.markdown(
        "<div style='margin-top:2rem; padding-top:1rem; border-top:1px solid #E2E8F0; "
        "font-size:0.72rem; color:#94A3B8; line-height:1.5;'>"
        "<div style='font-weight:600; color:#475569;'>STAT 5293 GenAI</div>"
        "Yang · Chen · He · Spring 2026</div>",
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Load systems (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading RAG systems...")
def load_systems(
    vdb_crag: str,
    vdb_baseline: str,
    t_high: float,
    t_low: float,
    k: int,
    alpha_blend: float,
    web_enabled: bool,
    qtype_override: str | None,
):
    from rag_baseline import VectorStore, BaselineRAG
    from crag_pipeline import CorrectedRAG, RetrievalEvaluator

    vdb_crag_path = Path(vdb_crag)
    vdb_base_path = Path(vdb_baseline)

    if not vdb_crag_path.exists() or not vdb_base_path.exists():
        return None, None

    vs_crag = VectorStore.load(vdb_crag_path)
    vs_base = VectorStore.load(vdb_base_path)

    evaluator = RetrievalEvaluator(
        threshold_high=t_high, threshold_low=t_low, alpha=alpha_blend
    )
    crag = CorrectedRAG(
        vs_crag,
        evaluator=evaluator,
        top_k=k,
        web_fallback_enabled=web_enabled,
        query_type_override=qtype_override,
    )
    baseline = BaselineRAG(vs_base, top_k=k)
    return crag, baseline


crag_system, baseline_system = load_systems(
    vector_db_dir,
    "data/vectordb_baseline",
    threshold_high,
    threshold_low,
    BASELINE_TOP_K,
    alpha,
    web_fallback_enabled,
    query_type_override,
)

if crag_system is None:
    st.warning(
        "Vector index not found. Use the sidebar to download SEC filings and build the index first."
    )

# ---------------------------------------------------------------------------
# Query input
# ---------------------------------------------------------------------------

# Section header
st.markdown(
    "<div style='margin-top:0.5rem; margin-bottom:0.7rem;'>"
    "<h2 style='margin-bottom:0.2rem; font-size:1.4rem;'>Query Console</h2>"
    "<p style='color:#64748B; margin-top:0; font-size:0.9rem;'>"
    "Ask a question about Apple, Microsoft, Alphabet, Amazon, or Meta — "
    "or test out-of-corpus questions to see the web fallback."
    "</p></div>",
    unsafe_allow_html=True,
)

example_questions = [
    "What are the principal risk factors disclosed in Apple's fiscal year 2025 10-K?",
    "What were Apple's total net sales in fiscal year 2025?",
    "How does Microsoft describe its Azure business in the fiscal year 2025 10-K?",
    "How did Apple's iPhone revenue change between fiscal year 2024 and 2025, and what does management cite as the reason?",
    "What was Amazon's AWS growth rate in 2025, and how does management explain it?",
    "What is the current federal funds rate?",
    "Who is the current US Treasury Secretary?",
]

# Use a Streamlit container so styling actually wraps the widgets.
with st.container(border=True):
    selected = st.selectbox(
        "Pick a demo question",
        ["(custom)"] + example_questions,
        index=0,
    )
    q_col, btn_col = st.columns([5, 1], gap="small")
    with q_col:
        question = st.text_input(
            "Your question",
            value="" if selected == "(custom)" else selected,
            placeholder="e.g. What was Amazon's AWS net sales in 2025?",
        )
    with btn_col:
        # Spacer to vertically align with the text input (which has a label)
        st.markdown("<div style='height:1.85rem;'></div>", unsafe_allow_html=True)
        run = st.button(
            "Run query →",
            type="primary",
            disabled=(crag_system is None or not question.strip()),
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if run and question.strip():
    # Display the asked question prominently above the answer panels — so
    # viewers (and the recording) always have the question on screen alongside
    # the answers, not just in the input field above.
    st.markdown(
        f"<div style='background:white; border:1px solid #E2E8F0; "
        f"border-left:4px solid #1E2761; padding:1rem 1.4rem; border-radius:0.5rem; "
        f"margin-bottom:1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.04);'>"
        f"<div style='font-size:0.7rem; color:#94A3B8; letter-spacing:0.12em; "
        f"font-weight:600; margin-bottom:0.35rem;'>QUESTION</div>"
        f"<div style='font-family: \"Source Serif Pro\", Georgia, serif; "
        f"font-size:1.2rem; color:#1E2761; font-weight:600; line-height:1.4;'>"
        f"{question}</div></div>",
        unsafe_allow_html=True,
    )

    col_crag, col_base = st.columns(2, gap="large")

    # Helper: visually distinct badge box (works on video, doesn't depend on
    # streamlit's flaky color shorthand which doesn't render in screen recordings)
    def _badge_html(label: str, color: str, fg: str = "white") -> str:
        return (
            f"<span style='display:inline-block;background:{color};color:{fg};"
            f"padding:0.35rem 0.85rem;border-radius:0.45rem;font-weight:600;"
            f"font-size:0.95rem;margin-right:0.4rem;letter-spacing:0.02em;'>"
            f"{label}</span>"
        )

    DECISION_COLORS = {
        "correct":   "#2A9D8F",  # teal
        "ambiguous": "#F4A261",  # amber
        "incorrect": "#E76F51",  # coral / red
    }
    QTYPE_COLORS = {
        "table_lookup":  "#1E2761",  # navy
        "narrative":     "#475569",  # slate
        "multimodal":    "#6D2E46",  # berry
        "out_of_corpus": "#E76F51",  # coral
    }

    # ===== CRAG panel =====
    with col_crag:
        st.markdown(
            "<h2 style='color:#1E2761;margin-bottom:0.2rem;'>🔁 CRAG</h2>"
            "<p style='color:#64748B;margin-top:0;font-style:italic;'>"
            "Probabilistic routing + dynamic query classification + tiered fallback</p>",
            unsafe_allow_html=True,
        )
        with st.spinner("CRAG · classifying → retrieving → evaluating → generating…"):
            t0 = time.perf_counter()
            crag_result = crag_system.query(question)
            crag_latency = (time.perf_counter() - t0) * 1000

        decision = crag_result.get("routing_decision", "?")
        confidence = crag_result.get("confidence_score", 0.0)
        query_type = crag_result.get("query_type", "?")
        tier_used = crag_result.get("tier_used", "?")

        # Top row of badges — each one big and on its own visual line so
        # they read clearly on a recorded video.
        decision_color = DECISION_COLORS.get(decision, "#64748B")
        qtype_color = QTYPE_COLORS.get(query_type, "#64748B")
        st.markdown(
            _badge_html(f"Query type · {query_type}", qtype_color)
            + _badge_html(f"Routing · {decision.upper()}", decision_color),
            unsafe_allow_html=True,
        )
        st.markdown(
            _badge_html(f"Tier · {tier_used}", "#475569")
            + _badge_html(f"Confidence · {confidence:.2f}", "#1E2761")
            + _badge_html(f"Latency · {crag_latency:.0f} ms", "#94A3B8"),
            unsafe_allow_html=True,
        )

        # Answer card — bigger text + accent border. clean_answer() strips
        # raw filesystem-path markers like "[TEXT | data/sec_filings/...]"
        # that the generator sometimes echoes from the context, replacing
        # them with human-readable filing names.
        st.markdown(
            "<div style='background:#F8FAFC;border-left:5px solid #2A9D8F;"
            "padding:1rem 1.25rem;border-radius:0.5rem;margin-top:1rem;"
            "font-size:1.0rem;line-height:1.55;color:#1E293B;'>"
            f"{clean_answer(crag_result['answer'])}</div>",
            unsafe_allow_html=True,
        )

        # Sources used (if any) — small footer chip. format_source() maps
        # raw paths to readable names (e.g. "Apple 10-K (2025)") and
        # de-duplicates so we don't render the same filing 3x.
        sources = crag_result.get("sources_used", [])
        if sources:
            seen: set[str] = set()
            unique_sources: list[str] = []
            for s in sources:
                pretty = format_source(s)
                if pretty and pretty not in seen:
                    seen.add(pretty)
                    unique_sources.append(pretty)
            src_chips = " ".join(
                f"<span style='display:inline-block;background:#E2E8F0;color:#475569;"
                f"padding:0.18rem 0.55rem;border-radius:0.3rem;font-size:0.78rem;"
                f"margin-right:0.3rem;'>{name}</span>"
                for name in unique_sources[:5]
            )
            st.markdown(
                f"<div style='margin-top:0.7rem;color:#94A3B8;font-size:0.85rem;'>"
                f"Sources: {src_chips}</div>",
                unsafe_allow_html=True,
            )

        if show_chunks:
            with st.expander("📊 Per-chunk retrieval scores"):
                scores_df = pd.DataFrame(crag_result.get("chunk_scores", []))
                if not scores_df.empty:
                    st.dataframe(scores_df, use_container_width=True)

    # ===== Baseline panel =====
    with col_base:
        st.markdown(
            "<h2 style='color:#475569;margin-bottom:0.2rem;'>📄 Baseline RAG</h2>"
            "<p style='color:#64748B;margin-top:0;font-style:italic;'>"
            "Top-K text retrieval → generate. No router, no fallback.</p>",
            unsafe_allow_html=True,
        )
        with st.spinner("Baseline · retrieving → generating…"):
            t0 = time.perf_counter()
            base_result = baseline_system.query(question)
            base_latency = (time.perf_counter() - t0) * 1000

        st.markdown(
            _badge_html(f"Tier · text only", "#475569")
            + _badge_html(f"Latency · {base_latency:.0f} ms", "#94A3B8"),
            unsafe_allow_html=True,
        )

        st.markdown(
            "<div style='background:#F8FAFC;border-left:5px solid #94A3B8;"
            "padding:1rem 1.25rem;border-radius:0.5rem;margin-top:1rem;"
            "font-size:1.0rem;line-height:1.55;color:#1E293B;'>"
            f"{clean_answer(base_result['answer'])}</div>",
            unsafe_allow_html=True,
        )

        if show_chunks:
            with st.expander("📊 Top retrieved chunks"):
                for c in base_result.get("retrieved_chunks", []):
                    st.markdown(
                        f"<div style='padding:0.4rem;background:#F1F5F9;"
                        f"border-radius:0.3rem;margin-bottom:0.3rem;'>"
                        f"<b>cosine={c['score']:.3f}</b> · "
                        f"{c['text'][:200]}…</div>",
                        unsafe_allow_html=True,
                    )

    # Confidence gauge — restyled for the demo recording
    st.divider()
    gauge_col, legend_col = st.columns([3, 2])
    with gauge_col:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=confidence,
                number={"font": {"size": 44, "color": "#1E2761"}, "suffix": ""},
                gauge={
                    "axis": {"range": [0, 1], "tickwidth": 1, "tickcolor": "#94A3B8"},
                    "bar": {"color": "#1E2761", "thickness": 0.7},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "#E2E8F0",
                    "steps": [
                        {"range": [0, threshold_low],            "color": "#FCE7E0"},  # incorrect zone
                        {"range": [threshold_low, threshold_high], "color": "#FEF3C7"},  # ambiguous zone
                        {"range": [threshold_high, 1],           "color": "#D1FAE5"},  # correct zone
                    ],
                    "threshold": {
                        "line": {"color": "#1E2761", "width": 3},
                        "thickness": 0.85,
                        "value": confidence,
                    },
                },
                title={"text": "P(Relevant | q, d)", "font": {"size": 16, "color": "#475569"}},
            )
        )
        fig.update_layout(height=270, margin=dict(t=50, b=10, l=20, r=20), paper_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with legend_col:
        st.markdown(
            "<div style='padding-top:1.5rem;'>"
            "<h4 style='color:#1E2761;margin-bottom:0.6rem;'>Routing zones</h4>"
            "<div style='display:flex;align-items:center;margin-bottom:0.45rem;'>"
            "<span style='display:inline-block;width:14px;height:14px;background:#D1FAE5;"
            "border:1px solid #2A9D8F;margin-right:0.55rem;border-radius:3px;'></span>"
            f"<span style='color:#1E293B;font-size:0.95rem;'>≥ {threshold_high:.2f} · "
            "<b style='color:#2A9D8F;'>CORRECT</b> → generate from corpus</span></div>"
            "<div style='display:flex;align-items:center;margin-bottom:0.45rem;'>"
            "<span style='display:inline-block;width:14px;height:14px;background:#FEF3C7;"
            "border:1px solid #F4A261;margin-right:0.55rem;border-radius:3px;'></span>"
            f"<span style='color:#1E293B;font-size:0.95rem;'>{threshold_low:.2f}–{threshold_high:.2f} · "
            "<b style='color:#F4A261;'>AMBIGUOUS</b> → re-retrieve / accept</span></div>"
            "<div style='display:flex;align-items:center;'>"
            "<span style='display:inline-block;width:14px;height:14px;background:#FCE7E0;"
            "border:1px solid #E76F51;margin-right:0.55rem;border-radius:3px;'></span>"
            f"<span style='color:#1E293B;font-size:0.95rem;'>&lt; {threshold_low:.2f} · "
            "<b style='color:#E76F51;'>INCORRECT</b> → web fallback</span></div>"
            "</div>",
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Ablation results viewer
# ---------------------------------------------------------------------------

# Footer cue — direct viewer to the Ablation Results page
st.markdown("<div style='height:1.5rem;'></div>", unsafe_allow_html=True)
st.markdown(
    "<div style='background:white; border:1px solid #E2E8F0; border-left:4px solid #2A9D8F; "
    "padding:1rem 1.25rem; border-radius:0.5rem; "
    "display:flex; align-items:center; justify-content:space-between; gap:1rem;'>"
    "<div>"
    "<div style='font-size:0.95rem; color:#1E2761; font-weight:600;'>"
    "📈 Want the quantitative story?</div>"
    "<div style='font-size:0.85rem; color:#64748B; margin-top:0.2rem;'>"
    "Open <b>Ablation Results</b> in the sidebar nav to see KHR, routing precision, "
    "and per-type charts across all 4 conditions × 22 questions."
    "</div></div>"
    "</div>",
    unsafe_allow_html=True,
)

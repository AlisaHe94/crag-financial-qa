# Prompt Summary — Streamlit Demo App (`app.py`, `_styles.py`, `pages/`)

## Goal

A live demo UI that renders CRAG and Baseline answers side-by-side,
exposes CRAG's internal state via badges, and lets a presenter tune
key parameters (thresholds, α, web fallback, query-type override) at
runtime.

## AI assistance

The team designed the UI direction (side-by-side comparison, visible
badges, financial-services aesthetic, the live demo controls), chose
the visual palette and typography, and selected the demo questions and
narration. AI was used as a coding assistant for the Streamlit
implementation, the visual styling, and the citation cleanup pass; the
team reviewed and modified all generated code before integration.

## High-level prompts used

- *Build a Streamlit app with two columns (CRAG and Baseline) and a
  shared question input. Show CRAG's routing decision, query type,
  tier, confidence, and latency as labeled badges above the answer.*
- *Apply a financial-services aesthetic: navy/teal/coral palette,
  Source Serif Pro headers, Inter body. Hero header with project name
  and team. KPI strip below.*
- *Sidebar: τ_high and τ_low sliders, with captions explaining what
  they control. Pass through to `RetrievalEvaluator` via Streamlit
  caching.*
- *Citation cleanup: regex-replace inline `[TEXT | path]` and
  `[TABLE | path]` markers in answer text with readable filing names
  (e.g., "Apple 10-K (2025)"). De-duplicate the source chips at the
  bottom of the answer card.*
- *Move the ablation results to a separate page in `pages/`. Render
  KPIs, KHR-by-condition charts, and per-condition findings.*
- *Add an α slider (cosine-vs-CE blend), a web-fallback toggle, and a
  query-type override dropdown to the sidebar. Pass them through to
  `CorrectedRAG`.*

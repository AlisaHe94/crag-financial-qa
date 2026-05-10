"""Shared visual styling for the Streamlit app pages.

Importing and calling `inject()` at the top of each page file applies the
financial-services aesthetic (navy/teal/coral palette, Source Serif Pro
headers, branded sidebar header, Inter body text) consistently across the
multi-page app.
"""

from __future__ import annotations

import streamlit as st


def inject_css() -> None:
    """Inject the shared CSS rules. Call once per page after set_page_config()."""
    st.markdown(
        """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Source+Serif+Pro:wght@400;600;700&family=Inter:wght@400;500;600;700&display=swap');

  html, body, [class*="st-"], [class*="css-"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  }

  .stApp { background: #FAFBFC; }

  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { visibility: hidden; }

  [data-testid="stSidebar"] {
    background: #FFFFFF;
    border-right: 1px solid #E2E8F0;
  }
  [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'Source Serif Pro', Georgia, serif;
    color: #1E2761;
    letter-spacing: -0.01em;
  }
  [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
    font-size: 0.85rem;
    color: #475569;
  }

  .block-container { padding-top: 1.5rem; padding-bottom: 4rem; max-width: 1400px; }

  h1, h2, h3 {
    font-family: 'Source Serif Pro', Georgia, serif !important;
    color: #1E2761;
    letter-spacing: -0.015em;
  }

  .stButton > button {
    background: #1E2761;
    color: white;
    font-weight: 600;
    padding: 0.55rem 1.5rem;
    border-radius: 0.4rem;
    border: none;
    transition: background 0.15s ease;
    box-shadow: 0 1px 2px rgba(30, 39, 97, 0.1);
  }
  .stButton > button:hover {
    background: #2A3578;
    color: white;
    box-shadow: 0 2px 6px rgba(30, 39, 97, 0.2);
  }
  .stButton > button:disabled {
    background: #CBD5E1;
    color: #94A3B8;
  }

  .stTextInput > div > div > input, .stSelectbox > div > div > div {
    border: 1px solid #E2E8F0;
    border-radius: 0.35rem;
    padding: 0.5rem 0.75rem;
    font-size: 0.95rem;
  }
  .stTextInput > div > div > input:focus { border-color: #1E2761; }

  .stSlider [role="slider"] { background: #1E2761; }
  .stSlider > div > div > div > div { background: #E2E8F0; }

  .stDataFrame { border: 1px solid #E2E8F0; border-radius: 0.4rem; }

  hr { border-color: #E2E8F0; margin: 1.5rem 0; }
</style>
        """,
        unsafe_allow_html=True,
    )


def hero_header(subtitle: str = "Side-by-side comparison · 10 SEC EDGAR filings · CRAG vs Baseline RAG") -> None:
    """Render the navy gradient hero header. Call once per page."""
    st.markdown(
        f"""
<div style="background: linear-gradient(135deg, #1E2761 0%, #2A3578 100%);
            padding: 2rem 2.25rem; margin: -1rem -2rem 1.75rem -2rem;
            border-radius: 0; color: white;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
  <div style="max-width: 1340px; margin: 0 auto;">
    <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:1rem;">
      <div>
        <div style="font-size: 0.78rem; letter-spacing: 0.18em; text-transform: uppercase;
                    color: #CADCFC; margin-bottom: 0.4rem;">
          STAT 5293 GenAI · Final Project
        </div>
        <h1 style="font-family: 'Source Serif Pro', Georgia, serif !important;
                   font-size: 2.1rem; line-height: 1.15; margin: 0; color: white;
                   letter-spacing: -0.02em;">
          Probabilistic Corrective RAG <span style="color:#E76F51;">·</span>
          Financial Document Q&amp;A
        </h1>
        <div style="margin-top: 0.55rem; font-size: 0.95rem; color: #E2E8F0; font-style: italic;">
          {subtitle}
        </div>
      </div>
      <div style="display:flex; gap:0.5rem; flex-wrap:wrap; align-items:center;">
        <span style="background: rgba(255,255,255,0.12); color:#CADCFC;
                     padding: 0.35rem 0.8rem; border-radius:0.35rem;
                     font-size:0.78rem; font-weight:500; letter-spacing:0.04em;">
          DISHEN YANG
        </span>
        <span style="background: rgba(255,255,255,0.12); color:#CADCFC;
                     padding: 0.35rem 0.8rem; border-radius:0.35rem;
                     font-size:0.78rem; font-weight:500; letter-spacing:0.04em;">
          SIWEN CHEN
        </span>
        <span style="background: rgba(255,255,255,0.12); color:#CADCFC;
                     padding: 0.35rem 0.8rem; border-radius:0.35rem;
                     font-size:0.78rem; font-weight:500; letter-spacing:0.04em;">
          JIAYI HE
        </span>
      </div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def kpi_strip() -> None:
    """The four corpus/model KPI tiles shown on the home page."""
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.markdown(
            "<div style='border-left:3px solid #1E2761; padding:0.4rem 0 0.4rem 0.85rem;'>"
            "<div style='font-size:0.7rem; color:#94A3B8; letter-spacing:0.12em;'>CORPUS</div>"
            "<div style='font-size:1.55rem; font-weight:700; color:#1E2761; line-height:1;'>10</div>"
            "<div style='font-size:0.78rem; color:#64748B;'>SEC filings (10-K + 10-Q)</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with mc2:
        st.markdown(
            "<div style='border-left:3px solid #2A9D8F; padding:0.4rem 0 0.4rem 0.85rem;'>"
            "<div style='font-size:0.7rem; color:#94A3B8; letter-spacing:0.12em;'>TEXT CHUNKS</div>"
            "<div style='font-size:1.55rem; font-weight:700; color:#2A9D8F; line-height:1;'>631</div>"
            "<div style='font-size:0.78rem; color:#64748B;'>semantic, bge-base 768d</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with mc3:
        st.markdown(
            "<div style='border-left:3px solid #6D2E46; padding:0.4rem 0 0.4rem 0.85rem;'>"
            "<div style='font-size:0.7rem; color:#94A3B8; letter-spacing:0.12em;'>TABLE CHUNKS</div>"
            "<div style='font-size:1.55rem; font-weight:700; color:#6D2E46; line-height:1;'>505</div>"
            "<div style='font-size:0.78rem; color:#64748B;'>structured key-value</div>"
            "</div>",
            unsafe_allow_html=True,
        )
    with mc4:
        st.markdown(
            "<div style='border-left:3px solid #E76F51; padding:0.4rem 0 0.4rem 0.85rem;'>"
            "<div style='font-size:0.7rem; color:#94A3B8; letter-spacing:0.12em;'>GENERATOR</div>"
            "<div style='font-size:1.55rem; font-weight:700; color:#E76F51; line-height:1;'>Llama 8B</div>"
            "<div style='font-size:0.78rem; color:#64748B;'>via Groq · Gemini fallback</div>"
            "</div>",
            unsafe_allow_html=True,
        )

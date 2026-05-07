# Demo Recording Script — Probabilistic CRAG

**Target length: 4-5 minutes**  ·  **Audience: STAT 5293 graders (rubric §6)**

---

## Pre-recording checklist (do this 5 min before pressing Record)

- [ ] Streamlit running (`./run.sh`), browser tab open at `http://localhost:8501`
- [ ] **Pre-warm all 4 demo queries by running them once** — this caches the embeddings, the cross-encoder, and Groq's connection. Live queries will then return in 2-4 seconds instead of 30+.
- [ ] Terminal visible somewhere on screen (so the diagnostic logs are partially visible — adds credibility)
- [ ] Sidebar: confirm `vectordb_crag_tables` is the path, slider top-K = 4
- [ ] Quit Slack, email, notifications — turn on Do Not Disturb (Mac: Cmd-Option-D or Notification Center)
- [ ] Recording tool ready (recommended: Cmd+Shift+5 → "Record Selected Portion" → frame the browser + a sliver of terminal)
- [ ] Glass of water nearby (sounds dumb, helps with vocal fry)

---

## Script (read aloud or paraphrase)

### 0:00–0:15 — Open

**[Show: Streamlit landing page, both panels visible]**

> "Hi. This is Probabilistic Corrective RAG for Financial Document QA — our STAT 5293 final project. The system answers questions over 10 SEC 10-K and 10-Q filings from Apple, Microsoft, Alphabet, Amazon, and Meta. On the left you'll see our CRAG pipeline — on the right, a standard naive RAG baseline running over the same corpus. Let's see how they handle four different question types."

---

### 0:15–0:30 — Quick architecture orientation

**[Optional: briefly cut to slide 4 (Architecture) for 5 seconds, then back to UI]**

> "Quick context — CRAG adds three things on top of naive RAG: an LLM-based query classifier that picks the right retrieval strategy per question type, a probabilistic evaluator that combines bi-encoder cosine and a cross-encoder re-ranker, and a 3-tier fallback that escalates to live web search when the corpus doesn't have the answer."

---

### 0:30–1:15 — Query 1: Type 1 (narrative)

**[Click: example dropdown → "What are the principal risk factors disclosed in Apple's fiscal year 2025 10-K?"]**

> "First — a narrative question. What are Apple's principal risk factors in their FY2025 10-K?"

**[Click "Ask". Wait. When results appear, point at:]**

- **Query type badge: `narrative`** → "The classifier routed this as a narrative query, so retrieval is text-heavy — six text chunks, one table chunk."
- **Routing decision: CORRECT** → "Confidence is high enough to route directly to generation, no web fallback needed."
- **CRAG answer** → "It correctly extracts the actual Item 1A risk categories: macroeconomic and industry risks, supply chain reliance, design and manufacturing defects."
- **Baseline answer** → "Baseline gets the same chunks but produces a less structured answer."

---

### 1:15–2:00 — Query 2: Type 3 (table lookup)

**[Click: dropdown → "What were Apple's total net sales in fiscal year 2025?"]**

> "Now a table lookup — what were Apple's total net sales in FY2025?"

**[Ask. When results appear:]**

- **Query type: `table_lookup`** → "Routed differently this time — six table chunks, two text chunks. The classifier knows tabular data is more useful here."
- **Both answers should give $416,161 million** → "Both pipelines find the right number, but notice the source. CRAG cites the structured income-statement table chunk we extracted with our HTML parser. Baseline cites flattened text — the number happens to appear in narrative."

---

### 2:00–3:00 — Query 3: Type 4 (multimodal synthesis)

**[Click: dropdown → "How did Apple's iPhone revenue change between fiscal year 2024 and 2025, and what does management cite as the reason?"]**

> "The harder case — a multimodal synthesis question. iPhone revenue change PLUS the reason management gives."

**[Ask. When results appear:]**

- **Query type: `multimodal`** → "Balanced retrieval — four text plus four table chunks. The system needs both."
- **CRAG answer** → "iPhone revenue went from $201.183 billion in FY2024 to $209.586 billion in FY2025. Management attributes the increase to higher net sales of Pro models. Both the number and the reason in one answer."

---

### 3:00–3:45 — Query 4: Out-of-corpus (the marquee moment)

**[Click: dropdown → "What is the current federal funds rate?"]**

> "Last query — and this is the one we built CRAG for. What's the current federal funds rate?"

**[Ask. When results appear:]**

- **Query type: `out_of_corpus`** → "The classifier instantly recognizes this isn't a question SEC filings can answer."
- **Tier: `web (OOC)`** → "Retrieval is bypassed entirely — straight to live web search via Tavily."
- **Routing: INCORRECT** in red → "The badge correctly says INCORRECT — meaning 'not in corpus,' not 'wrong answer.'"
- **CRAG answer**: "3.50% to 3.75%, with citations from NerdWallet and Trading Economics — current as of today."
- **Baseline answer**: "Insufficient information."

> "This is the failure mode standard RAG can't recover from — a question that needs information beyond the indexed documents. CRAG handles it gracefully by recognizing the gap and pivoting to live search."

---

### 3:45–4:00 — Close

**[Show: full UI one more time, maybe cut briefly to slide 13]**

> "To recap — Probabilistic CRAG with hybrid retrieval, dynamic query routing, and a 3-tier fallback. Code is on GitHub at our repo. Thanks for watching."

---

## If something goes wrong while recording

| Problem | Recovery |
|---|---|
| Streamlit shows "Connection error" | Don't panic — refresh tab. If still broken, save the take, restart streamlit, re-record from that question |
| LLM call takes 20+ seconds | Just keep talking — narrate the architecture while it works ("…the cross-encoder is now scoring each chunk's relevance to the query…") |
| Routing decision badge is unexpected (e.g. CORRECT when you said it'd be AMBIGUOUS) | Acknowledge it: "Interesting — the routing here is showing CORRECT, which means our confidence threshold cleared even on this borderline query. That's the kind of behavior the probabilistic evaluator gives us." Don't pretend it didn't happen. |
| Web fallback returns a stale or weird Tavily result | Briefly note it: "Tavily here pulled an aggregator page — in production we'd add source-quality filtering." Then move on. |
| You stutter or fluff a sentence | Pause for 1 second (gives editor a clean cut point) and just re-say the sentence. Most viewers won't notice; clean cuts are easy. |

---

## Recording technical setup (Mac)

**Built-in (recommended for speed):**
- **Cmd + Shift + 5** → click "Record Selected Portion" → drag to frame your browser + sliver of terminal
- Click Options → make sure "Show Mouse Clicks" is ON
- Microphone: Built-in Mic (or your AirPods) — check the menu bar before clicking Record

**Loom (recommended if you want a webcam corner):**
- Free tier records up to 5 min per video
- Webcam in corner adds personality; good for the rubric's "engagement" criterion

**OBS:** overkill for tonight — only worth it if you want to do scene transitions

---

## Submission notes

- Export as **MP4** (broadest compatibility)
- Aim for under 100 MB if uploading to a course platform — recording at 1080p compresses well
- Name the file something like `STAT5293_CRAG_Demo_Yang_Chen_He.mp4`
- If your team submits an unlisted YouTube link, double-check the share permissions

---

## After recording

1. Watch it back once at 1.5x speed — flag anything that needs re-shooting
2. Re-shoot only individual segments if needed (not the whole thing) — you can stitch in iMovie or QuickTime trim
3. Upload to wherever the course wants
4. Add the link to your README + slide 13 (Q&A slide footer or appendix)

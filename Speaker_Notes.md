# Speaker Notes — Probabilistic CRAG Final Presentation

**Total target time: ~10–12 minutes** (≈ 50–55 seconds per slide, with the demo + Q&A absorbing extra).

Suggested team split:
- **Person A** — Slides 1–4 (problem, prior work, architecture)
- **Person B** — Slides 5–8 (contributions, implementation, data, evaluator)
- **Person C** — Slides 9–13 (demo, evaluation, results, limitations, Q&A)

---

## Slide 1 — Title (15 sec)

> "We're Yang, Chen, and He, and our project is **Probabilistic Corrective RAG for Financial Document QA**. We're attacking three things at once: making RAG admit when it doesn't know, treating tables as first-class citizens in retrieval, and gracefully falling back to live web search when the corpus runs out. Let me set the stage with the problem."

---

## Slide 2 — The Problem (60 sec)

> "Financial analysts spend hours pulling specific numbers out of 200-page SEC filings. The natural fit is RAG — but standard RAG breaks in four characteristic ways on these documents.
>
> First, it **hallucinates numbers**. In an M&A context, a fabricated revenue figure is catastrophic.
>
> Second, **tables get flattened**. A 10-K's income statement loses row-column structure when chunked at 512 characters — the model sees `'Revenue 100 200 300'` and can't tell which year is which.
>
> Third, there's **no quality gate**. The retriever returns top-K chunks no matter how irrelevant they are.
>
> And fourth, the corpus is **stale by definition** — it can't tell you today's federal funds rate.
>
> The stakes are real: hallucinated numbers in finance translate directly to bad capital allocation."

**Transition:** "So how have others tried to fix this? Let's look at the state of the art."

---

## Slide 3 — State of the Art & Research Questions (50 sec)

> "Three lines of prior work matter here. **Standard RAG** from Lewis 2020 — the foundation, no quality gate. **CRAG** from Yan 2024 added a binary relevance evaluator and web fallback, but the binary threshold loses nuance. And **MultiFinRAG** from earlier this year built modality-aware indexes for SEC filings — text, table, and image — but didn't add probabilistic routing.
>
> Our four research questions sit at the intersection: Can a continuous probabilistic evaluator beat CRAG's binary one? Does a modality-aware index actually help on table questions? Does three-way routing reduce both hallucination and over-refusal? And can we gracefully handle out-of-corpus questions through web fallback?"

**Transition:** "Here's how we put it together architecturally."

---

## Slide 4 — Architecture (90 sec — the centerpiece slide)

> "This is the full system. A query enters at the top and fans out into a three-tier retrieval. Tier 1 hits the **text FAISS index** with a cosine threshold. Tier 2 hits the **table FAISS index** in parallel — this is the key MultiFinRAG-style innovation: separate indexes per modality so table embeddings don't compete with text embeddings.
>
> All retrieved chunks then pass through our **probabilistic evaluator**. The score is a convex combination — alpha-weighted FAISS cosine plus a sigmoid-normalized cross-encoder logit. Alpha is 0.4, so 60 percent of the score comes from the precise but slow cross-encoder.
>
> The score then drives a **three-way soft router**. Above tau-high, we accept the chunks and generate. In the ambiguous middle band, we re-retrieve. Below tau-low — or when the best chunk's cosine is too weak — we trigger Tavily web search. This is the marquee fallback for out-of-corpus questions."

**Anchor stats to memorize:**
- alpha = 0.4
- tau_high = 0.55, tau_low = 0.30
- text threshold = 0.50, table threshold = 0.45

**Transition:** "Three innovations make this work — let me walk through them."

---

## Slide 5 — Technical Contributions (60 sec)

> "Three contributions beyond CRAG and MultiFinRAG.
>
> **One — continuous probabilistic evaluator.** CRAG used a single binary threshold; we use a `[0,1]` score combining cosine and cross-encoder. The continuous score lets us tune two thresholds for asymmetric risk tolerance — analysts can set them tighter or looser depending on whether the cost of a wrong answer or a refusal is worse.
>
> **Two — three-way soft routing.** This is what distinguishes 'I retrieved weak evidence' (ambiguous → re-retrieve) from 'I retrieved nothing relevant' (incorrect → web fallback). It cuts both hallucination and unnecessary refusal.
>
> **Three — HTML-native table extraction.** Most pipelines assume PDF input. SEC's actual delivery format is inline-XBRL HTML wrapped in a multi-part submission archive. We wrote a BeautifulSoup parser that pulls each financial table out as a Markdown chunk and stores it in the table FAISS index — separately from the body text."

---

## Slide 6 — Implementation (40 sec)

> "Open stack, runs on a single laptop. **Llama-3.1-8B-Instant on Groq** is our primary generator — about 500 tokens per second, free tier — with **Gemini 2.5 Flash** as an automatic fallback when Groq throttles. Embeddings are bge-base-en-v1.5, 768-dim. The cross-encoder is MS-MARCO MiniLM. Vector store is FAISS, separate text and table indexes. Tavily handles web fallback. BeautifulSoup plus lxml parses the SEC submissions. Streamlit serves the demo. End-to-end reproducible with one command — `./run.sh` — and zero paid API keys required."

---

## Slide 7 — Data & Preprocessing (40 sec)

> "Our corpus is 10 SEC filings — the most recent 10-K and 10-Q for Apple, Microsoft, Alphabet, Amazon, and Meta. All FY2025-period documents. The pipeline has five steps: **extract** the primary document from SEC's multi-part archive, **clean** with BeautifulSoup, **split modality** using a numeric-density heuristic that distinguishes real data tables from layout-only tables, **chunk semantically** using sentence-embedding distance breakpoints à la MultiFinRAG, and **index** into two FAISS stores. End result: 631 text chunks plus 505 structured table chunks."

---

## Slide 8 — The Probabilistic Evaluator (60 sec)

> "Zooming into the evaluator. The score equation is here: alpha-weighted bi-encoder cosine plus 1-minus-alpha-weighted sigmoid of the cross-encoder logit. Why two signals? **Cosine** is fast and recall-friendly — it surfaces candidates. **Cross-encoder** does a joint encoding of query and chunk together — it catches subtle precision losses that cosine misses. The convex combination lets us tune the trade-off.
>
> The three-way decision is in the middle column. And in the right column — what makes this 'beyond CRAG' — is that the score is exposed to the analyst alongside every answer. It's auditable, calibratable, and tunable per task without a single code change."

---

## Slide 9 — Demo Walkthrough (90 sec — switch to live demo here if possible)

> "Three queries showing the system end-to-end. *(If running live: switch to Streamlit at this point. Otherwise narrate from the slide.)*
>
> First a **Type-3 table query**: 'What were Apple's total net sales in FY 2025?' Both pipelines find $416.161 billion. CRAG sources it from a structured income-statement table chunk; baseline gets it from flattened text — same answer, but CRAG's path is auditable.
>
> Second a **Type-1 narrative query** about Microsoft's Azure description. CRAG returns a detailed answer — Server products and cloud services revenue $98.4 billion, plus 23% growth driven by Azure plus 34%. Baseline returns 'unable to verify' because top-4 text chunks miss the segment-revenue paragraph.
>
> Third — and this is the marquee moment — an **out-of-corpus question**: 'What is the current federal funds rate?' CRAG's evaluator detects no strong matches, the badge flips to INCORRECT, and Tavily returns 3.50–3.75% from Trading Economics. Baseline says 'I cannot verify.' This is the exact rubric edge case the system is designed for."

---

## Slide 10 — Evaluation Methodology (40 sec)

> "We're using the four-question-type taxonomy from MultiFinRAG — text, image, table, and combined — crossed with our four ablation conditions. The bottom row is the full system: semantic chunking, table-aware index, CRAG router. Three primary metrics: **keyword hit rate** as a proxy for accuracy, **routing precision** for the CRAG-only conditions, and **end-to-end latency**."

---

## Slide 11 — Results & Qualitative Findings (60 sec)

> "Two important caveats here. The chart is **illustrative**, based on the pattern from six manual probe queries — the full 30-question ablation is in progress and will replace these numbers in the final report. *(If you've run the real ablation by then, replace this slide.)*
>
> The pattern we expect to hold: **CRAG-tables wins across all question types**, with the largest delta on Type-3 (table) and Type-4 (mixed) — exactly where modality-aware retrieval matters most.
>
> Five qualitative findings on the right. CRAG correctly handles narrative queries Baseline can't. Web fallback fires reliably on out-of-corpus. Adding the bge query prefix lifted retrieval scores from a tight cluster around 0.5 to a healthy 0.6–0.85 spread. Two challenges: routing badge calibration is tight, and free-tier rate limits are real but mitigated by the Gemini fallback."

---

## Slide 12 — Limitations & Future Work (45 sec)

> "Four current limitations. **Threshold calibration** — confidence sits in a narrow band so the routing badge is too often AMBIGUOUS to be informative. **Cross-encoder ceiling** — MS-MARCO MiniLM was trained on web search, not financial filings. **No image modality** for chart questions. And **free-tier rate limits** that throttle the ablation.
>
> Four next steps directly addressing each: **Dynamic LLM-classified routing** — let an LLM pre-classify queries as text / table / OOC. **Fine-tune the cross-encoder** on a small SEC-derived dataset. **Add a vision pipeline** for Type-2 questions. And **calibrate thresholds** on a held-out set."

---

## Slide 13 — Recap & Q&A (30 sec)

> "To recap. Continuous probabilistic evaluator that replaces CRAG's binary threshold with a tunable, auditable score. Modality-aware retrieval — 631 text plus 505 table chunks across 10 SEC filings. Tiered fallback that handles out-of-corpus questions. And a fully open, reproducible stack on a single laptop with one command. Happy to take questions."

---

## Likely Q&A — prepared answers

**Q: "Why not just use a bigger LLM?"**
> A: "Three reasons. First, cost — Llama-8B on Groq's free tier handles our entire ablation for $0; GPT-4-class models would be $5–20 per run. Second, latency — Groq's LPU does ~500 tokens/sec on 8B, faster than any GPT-4 deployment we've used. Third, the proposal's research thesis is specifically about *small open models* — improving retrieval and routing so the generator doesn't have to shoulder the burden."

**Q: "Why three thresholds (τ_high, τ_low, max-cosine)?"**
> A: "Two thresholds came from CRAG's binary correct/incorrect — we needed a third for the AMBIGUOUS middle case where a query rewrite might help. The max-cosine check was added later when we observed the cross-encoder gives moderate scores even to irrelevant chunks, so the mean confidence wasn't bottoming out enough to trigger web fallback for out-of-corpus questions. Looking at the strongest individual match — not the mean — fixed it."

**Q: "How do you know your numeric-density heuristic for tables actually catches the right tables?"**
> A: "Two checks. The empirical one — our table index ended up with 505 chunks across 10 filings, ~50 per 10-K, which matches the typical count of real financial tables in SEC filings (income statement, balance sheet, cash flows, segment breakdowns, exec comp, etc.). And the negative test — when we accidentally extracted layout tables in an early version, our text body became near-empty and baseline got 0 chunks. The heuristic clearly does the right partition."

**Q: "What if the question is about something the model already knows from pretraining?"**
> A: "Good question. Our system prompt explicitly constrains the model to answer from the provided context. If retrieval surfaces nothing relevant, the model says so and we trigger web fallback rather than letting the model fall back to its prior knowledge. This is a deliberate choice — for finance, audit trail matters more than fluency."

**Q: "What's the latency overhead of CRAG vs Baseline?"**
> A: "CRAG adds the cross-encoder pass — about 50ms per chunk × 4 chunks = 200ms. The web fallback when triggered adds about 1–2 seconds for the Tavily round trip. Net: CRAG is ~200ms slower than baseline on in-corpus queries, ~1.5s slower when fallback fires. That's the cost of auditability."

**Q: "Could you adapt this to non-financial domains?"**
> A: "Yes — the architecture is domain-agnostic. The two domain-specific pieces are the SEC archive parser (one specific input format) and the numeric-density table heuristic (specific to financial tables). For legal docs, we'd swap the parser; for medical literature, we'd loosen the heuristic. The probabilistic evaluator and three-way routing transfer directly."

---

## Practical tips for the live presentation

1. **Open Streamlit BEFORE you start the deck.** Models take 30–60s to warm up; have it ready.
2. **Pre-warm the demo queries.** Run each of the three demo questions once before presenting so the answers are already in the cache and feel snappy.
3. **Keep the terminal visible during the demo.** The diagnostic log line (`Eval @ '...': mean=0.X → ...`) is real proof the system is doing what you say it is.
4. **The federal funds rate query is the moment.** Let it land — pause, point at the routing decision flipping to INCORRECT, then read the Tavily-sourced answer.
5. **If something breaks live**, narrate it: "This is exactly the kind of free-tier rate limit we mentioned — let me retry, and notice the Gemini fallback engages." Embraces the failure mode rather than hiding it.

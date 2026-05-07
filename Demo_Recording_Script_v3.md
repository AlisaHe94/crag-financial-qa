# Demo Recording Script v3 — Probabilistic CRAG vs Baseline RAG

**Target length:** ~5 minutes. **Format:** screen recording with voiceover.
**Premise:** four corpus / OOC scenarios + one architectural toggle proof,
each demonstrating a different CRAG mechanism.

This version supersedes v2. Key updates: dropped the iPhone-only corpus
question, added a *partial-OOC* multi-part question that exercises the
completeness check + web augmentation path, and integrated the
web-fallback toggle demo as the architectural closer.

---

## Pre-recording setup

1. **Restart Streamlit fresh.** `./run.sh` or `streamlit run app.py`. Loads
   the latest `crag_pipeline.py` edits, the citation polish, and the new
   sidebar controls (α, web toggle, query-type override).
2. **Confirm sidebar defaults.** τ_high=0.75, τ_low=0.40, α=0.40, web
   fallback ON, query-type override = "Auto-classify (LLM)".
3. **Browser zoom ~100%** so sidebar + answer panels both fit in frame.
4. **Pre-load Ablation Results tab** once so it's cached when you switch
   to it at the close.
5. **Single continuous take.** Each retry burns Groq tokens; you've been
   hitting 429s. If you flub a sentence, keep going.

---

## Question lineup (recording order)

| # | Question | Type | Mechanism shown | Expected outcome |
|---|---|---|---|---|
| 1 | "How did Meta's family of apps daily active users change in 2025, and what does management say about Reality Labs spending?" | multimodal | Hybrid retrieval — table chunk for DAU + text chunk for Reality Labs | Both correct. CRAG narration leans on `Tier · text+table` and the visible badges as the interpretability win. |
| 2 | "What was Microsoft's Azure revenue growth in fiscal year 2025, and what is Microsoft's current stock price?" | partial-OOC (force multimodal via override) | Completeness check fires → web augmentation fills the missing half | CRAG: corpus answer for Azure + web-sourced stock price. Tier escalates to `text+table+web`. Baseline: no web access, refuses or hallucinates the price. |
| 3 | "What is the current price of Bitcoin?" | out_of_corpus | Pure web fallback — bypass corpus entirely | CRAG: live web answer, `Tier · web (OOC)`. Baseline: refuses or hallucinates. |
| 4 | (Toggle "Enable web fallback" OFF, re-ask Bitcoin) | OOC with web disabled | Architectural proof — CRAG degrades to baseline-like behavior without web | CRAG: refuses, just like baseline. Demonstrates the web fallback was *the* feature doing the work. Toggle back ON to leave the system clean. |

> Optional 5th beat (skip if running long): Microsoft Azure pure narrative
> ("How does Microsoft describe its Azure business in the FY2025 10-K?").
> Both pipelines correct, CRAG broader because it surfaces the OpenAI/Azure
> exclusivity quote. Use only if the recording feels short and you want
> another corpus example.

---

## Badge interpretation (memorize before recording)

Five visible CRAG badges, in the order they appear:

- **Query type** — LLM classifier label: `table_lookup` / `narrative` /
  `multimodal` / `out_of_corpus`. Drives retrieval composition.
- **Routing** — probabilistic verdict: `CORRECT` (green) / `AMBIGUOUS`
  (amber) / `INCORRECT` (coral). From P(Relevant) ≥ τ_high / ≥ τ_low / below.
- **Tier** — what retrieval path actually fired: `text only` /
  `text+table` / `text+table+web` / `web (OOC)`. The architectural claim
  is *visible* here.
- **Confidence** — mean P(Relevant | q, d) across retrieved chunks.
- **Latency** — wall-clock query time. Useful contrast with baseline.

Below the answer: **Sources** chips show readable filing names (e.g.
`Meta 10-K (2026)`) — citation polish landed pre-recording.

Below both panels: **confidence gauge** with τ_high / τ_low zones.

---

## The script (with timings and stage directions)

> Speak conversationally. Bracketed text is direction, not narration.

### [0:00–0:25] Open

> "Hi, I'm Alisa, with Dishen Yang and Siwen Chen — STAT 5293. We're
> presenting Probabilistic Corrective RAG for Financial Document QA. The
> system retrieves over ten SEC EDGAR filings — Apple, Microsoft, Amazon,
> Meta, Alphabet — and we're showing it head-to-head against a naive
> baseline RAG. Same corpus, same chunks, same generator. Only the
> retrieval architecture differs."

[Pan over the page so the hero header, sidebar controls, and query input
are all briefly visible.]

### [0:25–1:30] Question 1 — Meta DAU + Reality Labs

[Select question 1 from dropdown. Click Run.]

> "Multi-part question. Asks about Meta's daily active users — that's a
> figure in a table — and management's commentary on Reality Labs spending
> — that's prose. Watch CRAG."

[Wait for results render. Gesture at the badges row.]

> "CRAG classified this as multimodal. Routing came back CORRECT at
> confidence 0.60. Critically — look at the Tier badge: `text+table`.
> CRAG retrieved text chunks AND table chunks. Baseline only retrieves
> text. That distinction matters here because the 3.58 billion DAU figure
> lives in a table inside Meta's 10-K, not in a paragraph."

[Gesture at the CRAG answer.]

> "CRAG synthesizes both halves: 3.58 billion DAU up 7% year over year,
> AND the 70/30 wearables-versus-VR Reality Labs split. And notice the
> sources — `Meta 10-K (2026)` and `Meta 10-Q (2026)` — readable filing
> names, not raw paths."

[Gesture at the baseline panel.]

> "Baseline at top_k=5 actually finds the DAU number too in this run.
> Both pipelines get the answer. But notice what CRAG additionally tells
> you: the routing decision, the tier, the confidence score, the
> classified query type. CRAG is *interpretable*. Baseline is opaque."

### [1:30–3:00] Question 2 — Azure growth + current Microsoft stock price

[Open the sidebar. Set "Query-type override" to `multimodal`. Close
sidebar.]

> "Now a partial out-of-corpus question. Asks two things — Microsoft's
> Azure revenue growth in fiscal 2025, which is in the corpus, and
> Microsoft's current stock price, which is *not* in any annual filing.
> I'm forcing the query type to multimodal in the sidebar so the
> completeness-check path runs."

[Paste or type the question. Click Run.]

> "First-pass answer would have given the Azure growth from the 10-K and
> admitted the stock price isn't in the context. CRAG's completeness
> check audits the answer — does it address every part of the question?
> The answer was: no. So CRAG augments."

[Wait for results. Gesture at the Tier badge.]

> "Look at the tier — `text+table+web`. CRAG retrieved corpus first,
> generated, audited the answer for completeness, found a missing part,
> then ran a web search to fill the gap and regenerated. The Azure 34%
> growth comes from the 10-K. The current stock price comes from the web.
> One answer, two sources, transparently labeled."

[Gesture at baseline.]

> "Baseline has no completeness check, no web access. It either refuses
> the second half or hallucinates a price from the LLM's training data
> cutoff. This is the multi-source augmentation claim — CRAG knows when
> the corpus is incomplete and patches the gap."

[Reset query-type override to "Auto-classify (LLM)" before next question.]

### [3:00–3:45] Question 3 — Bitcoin (pure OOC)

[Paste: "What is the current price of Bitcoin?" Click Run.]

> "Now a fully out-of-corpus question. Bitcoin prices have never been in
> any SEC filing — by construction, baseline cannot answer this."

[Wait for results.]

> "Watch the badges. CRAG classified it as out_of_corpus. Routing went
> straight to INCORRECT. Tier reads `web (OOC)` — CRAG bypassed the
> corpus retrieval entirely, went straight to Tavily web search, and
> returned a live price with sources. Baseline either refuses or
> hallucinates a stale price. CRAG knows when the corpus can't help and
> reroutes accordingly."

### [3:45–4:25] Question 4 — Toggle web fallback OFF, re-ask Bitcoin

[Open sidebar. Toggle "Enable web fallback" to OFF. Close sidebar.]

> "Quick architectural proof. I'm turning off CRAG's web fallback in the
> sidebar. Now CRAG has no escape hatch."

[Re-run the Bitcoin question.]

> "Same question. Watch what CRAG does now."

[Wait for results.]

> "CRAG can't reach the web, falls through to corpus retrieval, and
> refuses — just like baseline does. The web fallback was *the* feature
> that made CRAG capable of OOC questions. Without it, CRAG degrades to
> baseline-like behavior. This is the architectural claim, made
> interactively visible."

[Toggle web fallback BACK ON to leave the system in working state.]

### [4:25–5:00] Close — gesture at Ablation Results

[Click into the Ablation Results page in the sidebar nav.]

> "Quantitatively — 22 questions across 4 conditions: baseline-no-tables,
> baseline-tables, CRAG-no-tables, CRAG-tables. CRAG-tables wins overall
> by 17 percent relative on keyword hit rate, with 91 percent routing
> precision. Largest gains on Type 1 narrative — 67 percent relative —
> and Type 4 multimodal synthesis. Type 3 ties because numerical figures
> repeat across MD&A and statements. Thanks for watching."

[End recording.]

---

## Rapid-fire Q&A cheat sheet

**"Why did the routing badge say AMBIGUOUS on a question CRAG got right?"**
The mean confidence over 6+ chunks gets dragged down by the long tail. One
or two chunks score high cosine, but the mean dips below τ_high. A
strong-corpus-match guard overrides the low mean when at least one chunk
clearly fits — that's why the corpus answer still happens despite the
amber badge.

**"Why is CRAG slower than baseline?"**
CRAG runs an LLM classifier (~500 ms), a cross-encoder pass (~1 s for 8
chunks), and on multimodal queries optionally a completeness check + a
re-prompt. Baseline does one retrieval + one generate. Cost is real;
question is whether the routing/quality gains justify it.

**"What if the LLM classifier mis-routes a question?"**
Two safety nets in code: the strong-corpus-match guard (prevents accidental
web fallback when corpus has the answer) and the narrative refusal rescue
(re-prompts on the same context with a sharper instruction if the LLM
refuses despite strong corpus signal). Both are visible as guards on top
of the classifier's decision. Plus, the demo controls let you manually
override the classifier in the sidebar.

**"Couldn't you just use a bigger model and skip the routing?"**
A bigger model wouldn't tell you the current Bitcoin price — that's an
architectural problem, not a generator problem. You'd also lose all the
interpretability (no routing badges, no confidence, no source provenance),
pay for the bigger model on every query including simple lookups, and have
no way to handle multi-source augmentation cleanly. The routing/fallback
machinery is doing work the generator alone can't do.

**"Why is α set to 0.4 instead of 0.5?"**
Slight bias toward cosine. Cross-encoder scores are uncalibrated and noisy,
especially on out-of-distribution queries. Future work would calibrate the
blend on a labeled validation set rather than hand-tune.

**"Is this real CRAG or a simplified version?"**
We implemented the routing, the probabilistic evaluator, and the tiered
fallback from the original paper. Strip-level knowledge refinement — where
the evaluator scores sentences within chunks rather than whole chunks — is
v2 work. Honest acknowledgment if asked.

---

## Failure modes to volunteer (if asked)

- Cross-encoder scores are uncalibrated. We treat the evaluator as noisy
  signal and lean on multiple guard rails. Calibrating on labeled data
  would tighten routing decisions.
- N=22 ablation has high variance. We'd want N≥100 for tighter intervals.
- Llama-8B's reading comprehension limits what CRAG can do on
  attribution-buried-in-prose questions. A bigger generator closes that
  gap; bigger generator also costs more per query.
- The web fallback inherits Tavily's freshness/quality. We don't control
  what the search engine returns.

These are honest limitations and worth volunteering rather than letting an
audience member surface them as gotchas.

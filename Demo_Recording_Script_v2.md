# Demo Recording Script — Probabilistic CRAG vs Baseline RAG

**Target length:** 4–5 minutes. **Format:** screen recording with voiceover.
**System under demo:** Streamlit app at `app.py`, four pre-validated questions plus
one toggle demonstration.

---

## Pre-recording setup (do this first)

1. Restart Streamlit fresh: `./run.sh` or `streamlit run app.py`. This loads
   the latest `crag_pipeline.py` edits and the new sidebar controls.
2. Confirm sliders at their defaults: τ_high = 0.75, τ_low = 0.40, α = 0.40,
   web fallback ON, query-type override = "Auto-classify (LLM)".
3. Set your browser zoom to ~100% so the sidebar and answer panels both fit
   in the recording frame.
4. Open the **Ablation Results** tab once in advance so it's pre-loaded — you'll
   gesture at it briefly at the end.
5. Recommend recording in one continuous take to conserve Groq tokens. Don't
   re-run questions during recording; if you need to retry, restart Streamlit
   first to re-cache.

---

## The four demo questions, in recording order

| # | Question | Type | Why this question | Outcome |
|---|---|---|---|---|
| 1 | "How did Meta's family of apps daily active users change in 2025, and what does management say about Reality Labs spending?" | multimodal | CRAG strict win — answer requires both a table figure (DAU) and prose (Reality Labs commentary). Demonstrates hybrid retrieval. | CRAG: both halves correct. Baseline: misses DAU. |
| 2 | "How does Microsoft describe its Azure business in the fiscal year 2025 10-K?" | narrative | CRAG slight edge — both correct, CRAG additionally surfaces the OpenAI/Azure exclusivity quote. Demonstrates that broad retrieval picks up complementary chunks. | Both correct, CRAG broader. |
| 3 | "What is the current price of Bitcoin?" | out_of_corpus | CRAG categorical win — baseline structurally cannot answer (no SEC filing has Bitcoin prices). Demonstrates web fallback claim. | CRAG: live web answer. Baseline: refuses or hallucinates. |
| 4 | "What did the DOJ rule in the Google antitrust case?" | out_of_corpus | Reinforces the OOC routing isn't a fluke — same pattern on a different OOC topic. | CRAG: web-sourced answer. Baseline: refuses or hallucinates. |

Optional 5th beat: toggle web fallback off, re-ask Bitcoin, watch CRAG
degrade to baseline-like behavior. Time-permitting only.

---

## What to point at on screen

CRAG produces five visible badges per query. Know what each one means so you
can narrate without hesitation:

- **Query type** — the LLM classifier's label: `table_lookup` / `narrative` /
  `multimodal` / `out_of_corpus`. Drives retrieval composition.
- **Routing** — the probabilistic evaluator's verdict: `CORRECT` (green) /
  `AMBIGUOUS` (amber) / `INCORRECT` (coral). Comes from
  P(Relevant) ≥ τ_high, ≥ τ_low, or below.
- **Tier** — what retrieval path actually fired: `text only` / `text+table` /
  `text+table+web` / `web (OOC)`. The architectural claim is *visible* here.
- **Confidence** — the mean P(Relevant | q, d) across retrieved chunks.
- **Latency** — wall-clock query time. Useful contrast with baseline.

Below the answer, **Sources** chips show which filings (or `Web search`)
contributed. Below both panels, the **confidence gauge** shows where the
mean score sits relative to τ_high / τ_low zones.

---

## The script (with timings and stage directions)

> Speak conversationally. The script below is a guide — paraphrase rather
> than read. Bracketed text is a stage direction, not narration.

### [0:00–0:20] Open

> "Hi, this is Alisa, Dishen, and Siwen from STAT 5293. We're presenting
> Probabilistic Corrective RAG for Financial Document Question Answering.
> The system retrieves from 10 SEC EDGAR filings — Apple, Microsoft, Amazon,
> Meta, Alphabet — and we're comparing it head-to-head against a naive
> baseline RAG over the same corpus, same generator, same chunks. The only
> thing that varies is the architecture."

[Brief pan over the page: hero header, sidebar, query input.]

### [0:20–1:30] Question 1 — Meta DAU + Reality Labs (the strongest win)

[Select question 1 from the dropdown. Click Run.]

> "First question is multi-part: it asks about a Meta DAU number AND about
> management commentary on Reality Labs spending. Watch the CRAG side."

[Wait for results to render. Gesture at CRAG badges.]

> "CRAG classified this as multimodal — synthesis across numerical and
> narrative content. Routing came back CORRECT at confidence 0.60. And
> notice the tier badge: text+table. CRAG retrieved table chunks alongside
> text chunks. That matters here because the 3.58 billion DAU figure lives
> in a table inside Meta's 10-K — not in prose."

[Gesture at the CRAG answer.]

> "CRAG answers both halves: it states the DAU change — 3.58 billion, up 7%
> year over year — and it summarizes management's Reality Labs commentary,
> the 70/30 wearables/VR split."

[Gesture at the baseline panel.]

> "Now look at baseline. Same generator. Same question. Baseline retrieves
> text only — that's its tier badge. And because the DAU number isn't in any
> text chunk, baseline has to admit it: 'the provided context does not
> contain information about how these figures changed.' Baseline got the
> Reality Labs half but lost the DAU half. This is exactly what
> modality-aware retrieval was designed to fix, and you can see the
> mechanism on the screen."

### [1:30–2:30] Question 2 — Microsoft Azure (narrative)

[Select question 3 (Microsoft Azure) from dropdown. Click Run.]

> "Second question — pure narrative this time. How does Microsoft describe
> Azure in the fiscal 2025 10-K?"

[Wait for results.]

> "CRAG classified this as narrative. The interesting badge here is routing —
> AMBIGUOUS, with confidence only 0.29. Mean cross-encoder relevance across
> seven retrieved chunks is genuinely low. But notice tier still reads
> text+table — CRAG didn't fall back to web. That's the strong-corpus-match
> guard: even when the average confidence is low, if at least one chunk has
> high cosine similarity, CRAG trusts the corpus. This is a routing nuance
> that baseline literally has no machinery to express."

[Compare answer panels.]

> "Both pipelines got the headline 23 percent / 34 percent Azure revenue
> growth. But look at CRAG's answer: it additionally surfaced the OpenAI
> exclusivity quote — 'The OpenAI API is exclusive to Azure, runs on Azure.'
> Baseline didn't include that. Same corpus, but the broader retrieval
> composition picked up an additional relevant chunk. That's the kind of
> answer-breadth gain that comes from CRAG's per-query-type retrieval
> tuning."

### [2:30–3:10] Question 3 — Bitcoin (OOC)

[Select question 6 (federal funds rate) — wait, replace with Bitcoin if
present. If not present in the dropdown, paste manually: "What is the
current price of Bitcoin?"]

> "Now an out-of-corpus question. SEC filings don't contain Bitcoin prices —
> by construction, baseline cannot answer this."

[Wait for results.]

> "Watch the badges. CRAG classified this as out_of_corpus. Routing went
> straight to INCORRECT. Tier reads `web (OOC)` — CRAG bypassed corpus
> retrieval entirely, went to Tavily web search, and returned a live price
> with sources. Baseline, on the right, has no concept of corpus boundary.
> It either refuses or hallucinates a price from the LLM's training data
> cutoff. CRAG knows when to leave the corpus."

### [3:10–3:40] Question 4 — DOJ Google (second OOC)

[Select / paste: "What did the DOJ rule in the Google antitrust case?"]

> "One more out-of-corpus question to confirm this isn't a coincidence."

[Wait for results.]

> "Same pattern. CRAG routes to web, returns sourced facts. Baseline doesn't
> have the concept. The architectural claim — that CRAG knows when the
> corpus can't help — is repeatable across topics."

### [3:40–4:10] Optional: web toggle demonstration

[Open sidebar, flip "Enable web fallback" OFF. Re-ask Bitcoin.]

> "Quick illustration. If I turn off CRAG's web fallback in the sidebar
> and ask the Bitcoin question again — CRAG now degrades to baseline-like
> behavior. The web tier is gone. CRAG refuses, just like baseline does.
> The web fallback is *the* feature that handles OOC. Toggle it back on,
> CRAG works again."

[Toggle back ON to leave the system in its working state.]

### [4:10–4:35] Close

[Click into the **Ablation Results** tab in the sidebar nav.]

> "And here's the quantitative picture. We ran 22 questions across 4
> conditions — baseline-no-tables, baseline-tables, CRAG-no-tables,
> CRAG-tables. CRAG-tables wins overall by 17 percent relative on keyword
> hit rate, with 91 percent routing precision. The biggest gains are on
> Type 1 narrative queries — 67 percent relative — and Type 4 multimodal
> synthesis. Type 3 ties because numerical figures repeat across MD&A and
> financial statements. Thanks for watching."

[End recording.]

---

## Quick interpretation cheat sheet (for Q&A after the recording)

**"Why is the routing AMBIGUOUS on Azure when the answer is correct?"**
Mean cross-encoder relevance across seven chunks is dragged down by the long
tail. Even though one or two chunks score high cosine, the average is below
τ_high. Routing reflects *aggregate uncertainty*, not whether any single
chunk is strong. That's why we have the strong-corpus-match guard: it
overrides the average when a single chunk clearly fits.

**"Why does CRAG sometimes take longer than baseline?"**
CRAG runs an LLM classifier (~500 ms), a cross-encoder pass (~1 s for 8
chunks), and on multimodal queries it may run a completeness check + a
sharper re-prompt. Baseline does one retrieval and one generate. The cost
is real; the question is whether the routing/quality gains justify it for
your use case.

**"What if the LLM classifier mis-routes a question?"**
Two safety nets: the strong-corpus-match guard (prevents accidental web
fallback when corpus has the answer) and the narrative refusal rescue
(re-prompts on the same context if the LLM refuses despite strong corpus
signal). Both are visible in the code as guards on top of the classifier's
decision.

**"Could you just use a bigger model and skip the routing?"**
Maybe — but you'd lose interpretability (no routing badges, no confidence
score, no source provenance), you'd pay for a bigger model on every query
including simple table lookups, and you'd still hit the OOC failure mode
(a bigger LLM doesn't know the current Bitcoin price). The routing/fallback
machinery is doing work the generator alone can't do.

**"Why is α 0.4 instead of 0.5?"**
Slight bias toward cosine because cross-encoder scores can be noisy and
uncalibrated on out-of-distribution queries. Future work would calibrate
the blend on a labeled validation set instead of hand-tuning.

---

## Failure modes to acknowledge if asked

- The cross-encoder produces uncalibrated scores; we treat it as a noisy
  signal and lean on multiple guard rails. Calibrating the evaluator would
  make the routing more defensible.
- N=22 ablation has high variance. We'd want N≥100 for tighter intervals.
- Llama-8B's reading comprehension limits what CRAG can do on
  attribution-buried-in-prose questions (e.g., iPhone Pro models). A bigger
  generator would close that gap.

These are honest limitations and worth volunteering rather than letting an
audience member surface them as gotchas.

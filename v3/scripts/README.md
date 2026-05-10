# Reproducibility scripts

Utility scripts for reproducing the methodology results reported in the
paper. All scripts assume:

- `data/vectordb_baseline/` and `data/vectordb_crag_tables/` are built
  (see top-level README "Step 2 — Build the vector index")
- `.env` is populated with `GROQ_API_KEY` (or fallback `GEMINI_API_KEY`)
  and `TAVILY_API_KEY`
- Python dependencies are installed (`pip install -r requirements.txt`)

## `alpha_sweep.py`

Sweeps the probabilistic-evaluator blend coefficient α ∈ {0.0, 0.4,
0.6, 1.0} across all 22 evaluation questions, runs CRAG-tables on each
combination, and writes per-row keyword-hit-rate / latency results to
`alpha_sweep_results.csv`.

Runtime: ~10–15 min on Groq free tier (88 LLM calls + cross-encoder
passes). At the end the script prints three summary tables:

1. **KHR by α × question_type** — primary quality metric. Shows
   whether different α values shift performance asymmetrically across
   the four question types.
2. **Latency by α** — mean / stdev / min / max in milliseconds.
   Latency varies because different routing decisions (corpus vs
   corpus+web) have different cost profiles, and different α can flip
   routing on borderline queries.
3. **Mean latency by α × question_type** — useful for spotting
   whether α changes shift the latency profile asymmetrically (e.g.,
   α values that route more multimodal queries through the
   completeness-check + re-prompt path push their latency up).

```bash
python scripts/alpha_sweep.py
```

Use the KHR table to defend the α=0.4 default with empirical
sensitivity data. Use the latency tables to discuss the cost-vs-quality
trade-off — α values that shift routing decisions affect both metrics
simultaneously.

## `alpha_ensemble.py`

α-ensemble inference. Runs each evaluation question through CRAG-tables at
three α values simultaneously (α ∈ {0.0, 0.4, 1.0}) and uses the agreement
pattern across the three routing decisions as an inference-time confidence
signal:

- 3-way agreement → "stable" (high confidence)
- 2-way majority → "majority" (moderate confidence)
- full disagreement → "disagree" (low confidence)

α=0.4 is the primary; the answer returned to the user always comes from that
run. The other two α values exist purely as a confidence probe. α=0.6 is
deliberately excluded (worst-performing in the offline sweep, KHR 0.712 vs.
α=0.4's 0.803).

This reframes α from "knob to tune offline" to "ensemble axis exposed at
runtime" — a contribution distinct from Yan et al. (2024), Self-RAG, and
MultiFinRAG, none of which perturb their own scoring hyperparameters at
inference time.

Runtime: ~3× the headline ablation per question — but only one α-set instead
of four, so wall-clock is roughly 50% of `alpha_sweep.py`.

```bash
python scripts/alpha_ensemble.py
```

Outputs `alpha_ensemble_results.csv` plus five summary tables:

1. **Routing-stability distribution** — how often the three α values agree.
2. **Mean KHR by stability bucket** — the headline validation. If routing
   stability is a real confidence signal, KHR should be higher on stable
   rows than on disagree rows.
3. **Stability × question_type** — which question types trigger more
   disagreement (we expect multimodal and OOC to be less stable than
   table-lookup).
4. **Latency overhead** — primary-α latency vs. full-ensemble latency.
5. **Unstable rows** — the actual questions where the ensemble did real
   work, useful for the failure-mode taxonomy in the report.

## `compare_answers.py`

Side-by-side qualitative comparison of CRAG and baseline answers on a
curated subset of high-signal questions. Pulls 6–8 questions from the
existing `data/eval_results.csv` (no new eval runs needed) and prints
all four conditions' answers for each, along with KHR, numerical
fidelity score, sub-question coverage rate, and any unverified numbers.

Used to spot-check whether quantitative KHR improvements correspond to
real qualitative improvement (CRAG's longer, more grounded answer) vs.
artifactual keyword-density gain. Also surfaces concrete failure-mode
examples (e.g., `balance_sheet_lookup` rows where every condition
struggles) for the report's Limitations section.

```bash
python scripts/compare_answers.py > data/answer_comparison.md
```

The output is markdown so it can be pasted directly into the report
as "Worked Examples" or saved as a presentation appendix. Edit the
`QUESTIONS_TO_INSPECT` list at the top of the script to focus on
different question IDs.

## `alpha_ensemble_ci.py`

Post-hoc bootstrap 95% confidence intervals on the α-ensemble
stability-bucket KHR means. Loads `alpha_ensemble_results.csv` and
quantifies whether the "stable rows have higher KHR than majority rows"
finding holds up statistically on our n=22 eval set. Reuses the
`_bootstrap_ci` helper from `evaluate.py` (1000 resamples, seed=42,
percentile method) so the methodology is consistent with the headline
ablation.

```bash
python scripts/alpha_ensemble_ci.py
```

Reports per-bucket CIs and a CI-overlap flag — the report can claim
"stability predicts quality" only if the bucket CIs do not overlap.

## `alpha_sweep_demo_questions.py`

Faster variant: sweeps α ∈ {0.2, 0.4, 0.6} on just the 4 demo
questions, dumping full answers to `alpha_sensitivity_answers.md` for
qualitative inspection.

Runtime: ~2–3 min.

```bash
python scripts/alpha_sweep_demo_questions.py
```

## Bootstrap confidence intervals

Bootstrap CIs on the headline ablation results are computed
automatically when you run `evaluate.py` from the repo root — the
output CSV (`data/eval_results.csv`) plus the printed summary include
2.5–97.5 percentile bounds for **both KHR and latency** in each cell,
plus routing precision CIs for CRAG conditions. See the top-level
README "Step 4 — Run the ablation study" for usage.

Reported metrics in `_print_summary`:

- **KHR mean + 95% bootstrap CI** — primary quality metric
- **Routing precision mean + 95% bootstrap CI** (CRAG only) — quality
  signal exposed by the architecture
- **Latency mean (ms) + 95% bootstrap CI** — runtime cost; CRAG is
  expected to be slower than baseline due to additional LLM calls
  (classifier, optional completeness check + re-prompt)

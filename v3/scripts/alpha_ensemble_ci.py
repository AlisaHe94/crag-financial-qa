"""Bootstrap 95% CIs on the α-ensemble stability-bucket KHR means.

Loads `alpha_ensemble_results.csv` (produced by `alpha_ensemble.py`), splits
rows by routing-stability bucket, and reports the bootstrap CI on the KHR
mean of each bucket. Used to test whether the headline "stable rows have
higher KHR than majority rows" finding survives statistical scrutiny on
our n=22 eval set.

Reuses the `_bootstrap_ci` helper already defined in `evaluate.py` so the
methodology stays consistent with the headline ablation (1000 resamples,
seed=42, percentile method).

Usage:
    python scripts/alpha_ensemble_ci.py
"""

from __future__ import annotations
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluate import _bootstrap_ci  # noqa: E402


def main() -> None:
    csv_path = Path("alpha_ensemble_results.csv")
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run scripts/alpha_ensemble.py first.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows from {csv_path}\n")

    print("=== KHR by stability bucket (95% bootstrap CI, 1000 resamples) ===")
    print(f"{'stability':<12}{'n':>4}  {'mean':>6}  {'95% CI':>16}")
    print("-" * 42)
    for stability, group in df.groupby("stability"):
        khr_mean, khr_lo, khr_hi = _bootstrap_ci(group["khr"])
        print(
            f"{stability:<12}{len(group):>4}  "
            f"{khr_mean:>6.3f}  "
            f"[{khr_lo:.3f}, {khr_hi:.3f}]"
        )

    # Also report the overall (pooled) KHR for context — useful for the
    # report so we can compare bucket means to the population mean.
    overall_mean, overall_lo, overall_hi = _bootstrap_ci(df["khr"])
    print("-" * 42)
    print(
        f"{'(pooled)':<12}{len(df):>4}  "
        f"{overall_mean:>6.3f}  [{overall_lo:.3f}, {overall_hi:.3f}]"
    )

    # Headline gap: how far apart are the two buckets, with their CIs?
    # If the CIs overlap substantially, the gap is not statistically tight
    # and we should hedge in the report. If they don't overlap, we have
    # a defensible claim.
    if {"stable", "majority"}.issubset(set(df["stability"].unique())):
        stable_khr = df[df["stability"] == "stable"]["khr"]
        majority_khr = df[df["stability"] == "majority"]["khr"]
        s_mean, s_lo, s_hi = _bootstrap_ci(stable_khr)
        m_mean, m_lo, m_hi = _bootstrap_ci(majority_khr)
        gap = s_mean - m_mean
        ci_overlap = s_lo <= m_hi and m_lo <= s_hi
        print()
        print(f"Stable − Majority KHR gap: {gap:+.3f}")
        if ci_overlap:
            print(
                "  CIs overlap. The gap is suggestive but NOT statistically "
                "tight on n=22 — report should hedge ('directional evidence', "
                "not 'significant difference')."
            )
        else:
            print(
                "  CIs do NOT overlap. The gap is statistically meaningful on "
                "this sample — report can claim the stability label predicts "
                "answer quality."
            )


if __name__ == "__main__":
    main()

# program_pilotB.md: Pilot B (shared-error lineage), confirmatory. Frozen at G1; the loop never edits it.

## Question

Does a sampler that uses **shared-transcript lineage** certify the same recall guarantee as Stretto-style global sampling with clearly **fewer ground-truth labels**?

## Data (fixed; human labels exist, so no new annotation)

- **Corpus:** the AMI dev meetings in `configs/ami_meetings.txt` that have dialogue-act annotations: 13 of the 18. IB4001, IB4002, IB4004, IB4010 and IB4011 have none; see deviation D1 in `reports/deviations.md`, dated 2026-10-28.
- **Unit:** an AMI dialogue act (manual annotation, with its time span). Acts with no DA type are skipped.
- **Shared transcript:** the Whisper large-v3 words whose midpoint falls in the act's span. Both predicates read this **same** transcript. ASR confidence = the mean Whisper word probability; acts with no word are marked empty.
- **Predicates:** each is scored by the Pilot A machinery (Qwen2.5-1.5B fp32, margin = logit(" yes") − logit(" no")) over that one transcript:
  - **P1:** "Is this meeting utterance a suggestion (proposing an idea or a course of action)?" Gold: DA class `sug`.
  - **P2:** "Does this meeting utterance ask someone for information or for their opinion or assessment?" Gold: classes `el.inf` and `el.ass`.
- **Query:** P1 OR P2.
  - Score = max(margin1, margin2).
  - Returned set R(τ) = {units with score ≥ τ}.
- **Ground truth:** AMI's manual dialogue-act labels, i.e. human truth, not an LLM oracle. One "label" = reading the gold value of one unit.

## Guarantee and methods (`pilots/guarantee_eval.py`, locked)

**Guarantee:** recall of R(τ) ≥ 0.9 with probability ≥ 1 − δ, δ = 0.1.

**Label budgets:** n ∈ {800, 1120, 1440, 1600}, i.e. 0.5×, 0.7×, 0.9× and 1.0× of the largest budget.

**Estimator (identical for both methods; only the sampling design differs):**

- **Recall estimate:** the stratified ratio estimator, recall = TP / positives, with each stratum weighted by its size.
- **Lower bound:** a linearised variance with finite-population correction gives a one-sided normal lower bound. It is built at **δ_bound = 0.05**: the approximation is not exact, so the bound is made tighter than the δ = 0.1 it is judged against.
- **Minimum:** no certificate from fewer than 30 sampled gold positives.
- **Threshold choice:** τ runs over a grid of K = 50 score quantiles, tested by **fixed-sequence testing** from the largest set to the smallest, stopping at the first threshold that fails. This is valid because recall(τ) cannot increase with τ, so no /K correction is needed.
- **No certificate:** R = all units.

| method | sampling |
|---|---|
| **global** (the Stretto-style baseline) | a uniform sample without replacement (one stratum) |
| **lineage** | strata from system outputs only: ASR-confidence tercile × (score ≥ 0) × empty transcript. Allocation ∝ stratum size × √(1 − the stratum's mean ASR confidence), with at least 2 per stratum. |

**Trials:** 100 per (method, n, seed), with seeds 0, 1 and 2 (300 trials per method and budget). Per trial: the precision of R, the true recall from the full gold labels, and violation = (true recall < 0.9).

**Design history:** the design was checked on synthetic data before any real label was computed (2026-10-27).

- **Exact bounds were degenerate.** Per-τ Bonferroni with exact Clopper–Pearson bounds never certified at these budgets, and the per-stratum union bound never certified for lineage. So both methods now share one estimator.
- **The 400 budget was dropped.** With about 48 positives the normal approximation was unreliable there.

## Verdict (`pilots/verdict.py`, locked)

Let `viol(m, n)` = the violation share over 300 trials, and `CI(a − b)` = the 95% bootstrap CI of the difference in mean precision, resampling trials.

- **PASS:**
  - `viol(lineage, n)` ≤ 0.1 at every n, **and**
  - lo `CI(prec(lineage, 1120) − prec(global, 1600))` ≥ 0, i.e. lineage matches global with ≤ 0.7× the labels.
- **FAIL:**
  - `viol(lineage, n)` > 0.1 at any n, **or**
  - hi `CI(prec(lineage, 1440) − prec(global, 1600))` < 0, i.e. it saves ≤ 10% of the labels.
- **INCONCLUSIVE:** anything else.

## Anomalies (stop and report to a human)

- `viol(global, n)` > 0.1 at any n: the baseline's guarantee itself is broken (a bug, or the approximation fails on this data).
- `viol` = 0 for lineage at every n while prec(lineage, 800) ≥ prec(global, 1600): a label saving of ≥ 50% with no violation is suspiciously good.
- A mean precision of 1.0.
- Fewer than 13 meetings (deviation D1).
- The eval lock check fails.

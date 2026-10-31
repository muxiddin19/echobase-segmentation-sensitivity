# E0 checks for G0 (plan v2.1 §E0; dev data only; 2026-10-28)

All numbers below come from generated files:
- `reports/e0_checks_auto.md`, from `echobase/e0_checks.py`;
- `reports/power_sim.md`;
- `reports/e0_determinism_gpu{0,1}.json`;
- `reports/e0_model2_*.json`;
- `configs/frozen_lists.json`.

AMI eval and the Wikipedia test split were not touched.

| # | check | rule | result | status |
|---|---|---|---|---|
| 1 | determinism under stress (500 dev segments; shuffled order; budgets 1k/3k/6k; both GPUs; 3 columns) | labels identical | 0 label mismatches in 4,500 rescorings per GPU; max margin drift 1.4e-4 | **pass** |
| 2 | replay causality and paired same-cell tests | unit tests pass | `echobase/tests/test_replay.py`: 3 tests (15/15 overall) | **pass** |
| 3 | W2 placebo on dev | code works (a small value is a result) | median D, Q1 = 0.244 (n = 18) | **pass** |
| 4 | W3 diarizer change on dev | change share ≥ 0.10 | pyannote 3.0 crashed gpulab2 (D2). The 3.1 threshold variant: share 0.004, so it failed. **D3:** H3 dropped; W3 becomes exploratory, estimated vs known speaker count: share **0.193** pooled, all-or-nothing by meeting (see note) | **resolved (D3)** |
| 5 | W4 Wikipedia chunker change | split/merge/n:m share ≥ 0.15; Qd1(old) ≥ 0.02 in ≥ 150/200 | share 0.940; 197 / 200 | **pass** |
| 6 | headroom | Q(old) ≥ 0.02 in ≥ 14/18 for every primary query | the minimum is Q4 on w1old, 14 / 18 (exactly at the rule); all others are 17–18 / 18 | **pass** |
| 7 | second-model admission | yes-rate in [0.05, 0.60]; distinct answer tokens | SmolLM2-1.7B: yes-rate 0.000 / 0.002, **not admitted**. Plan fallback Qwen2.5-0.5B: 0.132 / 0.160, **admitted** | **resolved (fallback)** |
| 8 | pyannote training data | determine | eval = AMI full-corpus-ASR test = the BUT `AMI-diarization-setup` test list (verified). segmentation-3.0 was trained on "the training sets of … AMI …" (model card) | **resolved** |
| 9 | power simulation | report P(rule met) | H1 0.96, H2 0.99, H4 1.00 (500 smoothed-bootstrap simulations, see note) | **done** |
| 10 | frozen lists | file written | `configs/frozen_lists.json`: 16 eval meetings, placebo seeds, 200 test article ids, the articles sha256, chunker parameters, model revisions, code sha256 (matches the repo) | **pass** |

## Notes

- **Power simulation fix.** The first run gave P = 0.64 / 0.63 for H1 / H2. 68 of 200 diagnostic draws had a NaN
  BCa interval: resampling 18 values with replacement creates ties, and BCa's jackknife then degenerates. Only one draw
  of 200 failed on the actual bound. The simulation now uses a smoothed bootstrap. The decision rule now falls back
  to the percentile interval when BCa is undefined. That fallback is part of the preregistered rule and is needed on
  eval too. With these changes there were 0 fallbacks in 1,500 simulated tests.
- **The placebo drifts as much as the real workload.** On dev, the W2 placebo (manual → length-shuffled segmentation,
  same words) has median D, Q1 = 0.244. W1 (diarizer → manual) has 0.211. This fits the paper's claim that
  segmentation alone moves answers. It also means H1 says nothing about diarization errors as such. The paper must not
  read W1 drift as an error effect. "W1 exceeds placebo" is not a planned test and should stay exploratory.
- **Why W3 failed.** Both sides use the known speaker count. In pyannote 3.1 the clustering threshold then only picks
  the starting point of the dendrogram search for that count (`pipelines/clustering.py`). The turns come out almost
  identical.
- **Second model.** The admitted model is in the same family as the primary (Qwen2.5). This is a limitation for E4,
  as plan §E0.7 anticipated.
- **Cost.** W3 diarization took 149–415 s per meeting (19 min in total); w3old scoring took 5,110 calls.

## W3 exploratory result (D3; e06 in `echobase/results.tsv`)

The change is all-or-nothing by meeting:
- **ES2011a and ES2011d:** share 0 and D = 0.
- **ES2011b:** share 0.098. All 8 relabels leave the segment text unchanged, and D for Q1 is 0.
- **ES2011c:** share 0.727. The system estimated 5 speakers against 4 in the gold; there are 42 edited segments, 36
  split or merge segments and 41 new ones. D is 0.505 for Q1 and 0.107 for Q2.

So the median over the 4 meetings is about 0. A configuration change moves answers only where it changes the
segmentation, and then it can move them a lot. This is reported as exploratory.

## Decision record

- **2026-10-28:** the human chose "Drop H3, W3 exploratory". The confirmatory family is H1 → H2 → H4 (plan v2.2, D3).

# Preregistration: Segmentation Sensitivity of LLM-Derived Views (EchoBase, ICDE 2027 EAB)

- **Drafted:** 2026-10-28 by Claude, after G0 was approved by the human.
- **Frozen:** at G1 (planned 2026-11-03). The human writes the eval lock (`python tools/eval_lock.py --write`), which
  covers this file, and records "G1" in the `STATUS.md` header.
- **Source:** `EchoBase Research Plan.md` v2.2, §3.2 and §8. Where this file and the plan differ, this file governs
  the confirmatory analysis.
- **PI:** Prof. Wookey Lee, BMSE, Inha University.

## 1. What is known before the confirmatory data are touched

Everything so far used AMI **dev** (18 meetings) and Wikipedia **dev** (200 articles). The AMI eval meetings and
the Wikipedia test articles were read only to list their names and ids and to hash the article file
(`configs/frozen_lists.json`). Nothing on them was diarized, scored or aggregated.

The thresholds below were set in plan v2 **after** the dev drift values were seen (e05: dev median D for Q1 = 0.211).
They sit at about half of the dev median, so that a real effect of roughly the dev size passes. The dev results
themselves are not evidence for the hypotheses.

Dev values for reference (`reports/e0_checks.md`, `reports/power_sim.md`):

| workload | dev median D | dev 95% CI |
|---|---|---|
| W1 | 0.211 | [0.143, 0.302] |
| W2 | 0.244 | [0.103, 0.337] |
| W4 | 0.213 | [0.172, 0.248] |

## 2. Data (frozen in `configs/frozen_lists.json`)

**AMI eval.** 16 meetings, Mix-Headset audio, AMI manual annotations NXT 1.6.2:
- EN2002a–d, ES2004a–d, IS1009a–d, TS3003a–d;
- this is the full-corpus-ASR test partition, the same as the BUT `AMI-diarization-setup` test list;
- pyannote's segmentation-3.0 was trained on AMI *train*.

**Wikipedia test.** 200 articles:
- source: `wikimedia/wikipedia` `20231101.en` shard `train-00000-of-00041`;
- sampling: 800–3,000 words, seed 0; articles 201–400 of the sample;
- ids in the frozen lists; `articles.jsonl` sha256 `01798f55…6cc943`;
- license: CC BY-SA 3.0 and GFDL.

## 3. Versions (the same words in old and new; only the segmentation differs)

| workload | old | new | role | code |
|---|---|---|---|---|
| **W1** | pyannote 3.1 turns, known speaker count, filled with AMI reference words | AMI manual segments with the same words | confirmatory (H1) | `pilots/run_diar.py` (frozen); `echobase/lineage.versions` |
| **W2** | AMI manual | placebo: per speaker, the same word order, segment count and length multiset, with the lengths shuffled. Seed `crc32("placebo:<meeting>")` | confirmatory (H2) | `echobase/workloads.placebo` |
| **W4** | fixed 120-word windows | paragraphs merged forward to ≥ 40 words | confirmatory (H4) | `echobase/wiki.py` |
| W3 | pyannote 3.1, estimated speaker count | pyannote 3.1, known count (= W1 old) | exploratory | `echobase/run_diar_v.py --estimate-speakers` |

## 4. Measurement

**Scorer.** Qwen/Qwen2.5-1.5B-Instruct, revision `989aa798…`, fp32, on a TITAN Xp. The answer is the first-token logit
argmax over a fixed answer set, so there is no sampling. The prompts and answer sets are in `echobase/llm.py`,
whose sha256 is in the frozen lists. Determinism was verified at E0.1: 0 label changes over 4,500 rescorings per GPU.

**Columns.**
- AMI: action_item, question (yes/no); entity_count (0–9); sentiment (3 labels); topic (5 labels).
- Wikipedia: doc_date (yes/no); doc_entity_count (0–9); doc_topic (7 labels).

**Queries.**
- **Q1** (AMI primary): the share of words in segments labeled action_item = yes.
- **Qd1** (Wikipedia primary): the share of words in passages labeled doc_date = yes.
- Shares are word-weighted, so they do not depend on the segment count.

**Unit drift.** D(u) = |Q(new) − Q(old)| / Q(old). A unit with Q(old) < 0.02 is excluded for that query and counted
(`echobase/drift.py`).

## 5. Hypotheses and decision rules (confirmatory family, fixed sequence H1 → H2 → H4)

| id | statement | units (min valid) | rule for "supported" |
|---|---|---|---|
| **H1** | Re-segmenting identical meeting words from diarizer turns to manual segments changes Q1 | 16 AMI eval meetings (14) | median D ≥ **0.10** and 95% CI lower ≥ **0.05** |
| **H2** | A placebo re-segmentation (same words, segment count and lengths) changes Q1 | 16 AMI eval meetings (14) | median D ≥ **0.05** and CI lower ≥ **0.02** |
| **H4** | Changing the document chunker changes Qd1 | 200 Wikipedia test articles (180) | median D ≥ **0.10** and CI lower ≥ **0.05** |

- **Statistic and interval.** The median of D over the valid units. The CI is BCa over units, with 10,000 resamples,
  `numpy.random.default_rng(0)` and `scipy.stats.bootstrap`. If BCa is undefined (NaN), the percentile interval is
  used and the report says so.
- **Verdicts:**
  - **supported:** the rule is met;
  - **refuted:** CI upper < T;
  - **inconclusive:** anything else, or fewer valid units than the minimum.
- **Interpretation labels** (plan §3.2):
  - H1 with CI lower ≥ 0.05 but median < 0.10 is reported as "moderate drift";
  - H1 with CI lower < 0.05 is reported as "no material drift shown";
  - a failed H2 means the drift is explained by granularity;
  - a failed H4 means the effect is specific to speech.
- **Multiplicity.** Fixed sequence, each test at α = 0.05. A hypothesis is *claimed* only if all earlier ones are
  supported. All three are computed and reported regardless.
- **Robustness (reported, not decisive).** A one-sided sign test of D > T.
- **Implementation.** `echobase/confirm.py`, unit-tested in `echobase/tests/test_confirm.py`. It refuses to run
  before G1.
- **Power.** E0.9 simulated each rule from the dev units: a smoothed bootstrap with 500 simulations of the exact rule.
  P(rule met) is 0.96 for H1, 0.99 for H2 and 1.00 for H4 (`reports/power_sim.md`).

## 6. Missing data and anomalies

- **Failed jobs.** A failed diarization or scoring job is retried once. A unit that fails twice has no cache. It is
  removed, listed in the report and counts against the minimum (plan §8.8).
- **Stop and report to the human** without running `confirm.py` when:
  - the eval lock check fails;
  - any eval cache is empty or unparsable after a host crash. gpulab2 hard-reset twice on 2026-10-28, so every cache
    is checked for 0-byte files before analysis;
  - a determinism spot check shows any label change. The check is
    `python -m echobase.e0_determinism --split eval --n 100 --out reports/e0_determinism_eval.json`, which rescores 100
    eval manual segments against the E2 cache.
- **One look.** `confirm.py` runs once, after E1–E3 are complete. If it has to be re-run because of a bug, that is a
  deviation: the bug, the fix and both outputs are logged.

## 7. Execution order (after G1; gpulab2 only; one job per free card)

1. **E1.** `python -m pilots.run_diar --split eval` (W1 old, frozen pilot code), then
   `python -m echobase.run_diar_v --pipeline 3.1 --estimate-speakers --split eval` (W3, exploratory).
2. **E2.** `python -m echobase.score_versions --split eval --version {manual, w1old, placebo, w3old}`.
3. **E3.** `python -m echobase.score_wiki --split test`.
4. Integrity checks (§6): the lock check, the 0-byte scan and the determinism spot check. Then `python -m echobase.confirm` writes `reports/confirmatory.{md,json}`.
5. **Exploratory work, after step 4:**
   - E4 second model: the same commands with `--model Qwen/Qwen2.5-0.5B-Instruct`;
   - E5 maintenance replay;
   - E6 granularity dose–response;
   - the W3 drift.

## 8. Exploratory analyses (labeled as such in the paper; no decision rules)

- Q2–Q6 and Qd2–Qd3.
- W3: estimated vs known speaker count. Dev showed an all-or-nothing effect (e06).
- "W1 exceeds placebo". On dev the placebo is not smaller (0.244 vs 0.211). So W1 drift is **not** interpreted as a
  diarization-error effect.
- Second model: Qwen2.5-0.5B, admitted at E0.7. It is from the same family as the primary, which is a limitation.
- Granularity dose–response with merge windows k = 2, 4, 8.
- Per-cell flip rates and their predictors.
- Maintenance consequences (RQ4):
  - full / hash / anchor-cert / EchoBase-conf replay;
  - ε ∈ {0.02, 0.05, 0.10, 0.15}, δ = 0.05, confidence bins (2, 5);
  - the policy re-run inside each bootstrap resample;
  - κ, tokens and wall-clock;
  - audit validity on injected strata.

## 9. Deviations before G1 (`reports/deviations.md`)

- **D1** (pilot phase): Pilot B meeting coverage.
- **D2:** pyannote 3.0 crashed the host, so W3 moved to a 3.1 threshold variant. That variant failed E0.4 with a
  change share of 0.004.
- **D3:** H3 was dropped and W3 made exploratory.

After G1, every change to a locked file or to this protocol is a deviation. It is logged with the date, the reason
and the results it touches, and it needs the human's approval.

## 10. Locked at G1 (`tools/eval_lock.py`, LOCKED)

- The pilot files already locked.
- `pilots/run_diar.py`.
- `echobase/`: llm.py, lineage.py, workloads.py, wiki.py, drift.py, power_sim.py, confirm.py, score_versions.py,
  score_wiki.py, e0_determinism.py, rules.py, certify.py, replay.py, and tests/*.py.
- `configs/frozen_lists.json`.
- `PREREGISTRATION.md`.
- `program_echobase.md`.

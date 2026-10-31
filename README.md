# Artifact: Same Words, Different Answers (ICDE 2027, EAB)

This package reproduces every number, table and figure of the paper. It has two levels:

1. **From the released caches, CPU only (about 30 minutes).** Every model output used in the paper is cached, so all
   statistics, tables and figures regenerate without a GPU.
2. **From scratch (about 8 GPU-hours on a 12 GB card).** Download AMI and the Wikipedia shard, diarize, score, then
   run level 1.

## Note on the freeze date

`PREREGISTRATION.md` says the freeze was planned for 2026-11-03. It happened earlier, on 2026-10-28: the eval lock was
written that day and re-written after deviation D4 (`reports/deviations.md`), before any held-out unit was scored. The
file itself is hash-locked, so its header is left unchanged.

## Contents

| Path | What |
|---|---|
| `PREREGISTRATION.md` | hypotheses, thresholds, units, analysis plan; frozen before any held-out data were scored |
| `reports/deviations.md` | every deviation from the plan, with date, reason and approval |
| `.claude/eval.lock.json` | SHA-256 of the 45 files that decide a score, written at the freeze; `python tools/eval_lock.py --check` verifies them |
| `configs/frozen_lists.json` | AMI dev/eval meetings, Wikipedia dev/test article ids, placebo seeds, model revisions, file hashes |
| `pilots/`, `echobase/` | segmentation, versions, scoring, drift, confirmatory analysis, maintenance replay, exploratory analyses |
| `outputs/` | caches: diarizer turns (`diar/`, `diar_v/`), per-segment values and confidence margins for every version and model (`echobase/v2/`, `echobase/v2-Qwen2.5-0.5B-Instruct/`) |
| `reports/` | confirmatory and exploratory results (JSON and Markdown), run journal, GPU ledger |
| `paper/` | LaTeX source; `gen_numbers.py` writes every number in the text from `reports/`; `figs/*.py` build every table and figure |

The AMI audio and annotations and the Wikipedia text are not redistributed; the scripts download them from their
original sources; their licenses apply (AMI: see the AMI Meeting Corpus site; Wikipedia: CC BY-SA 3.0 and GFDL).

## Environment

Tested on Ubuntu 18.04, Python 3.10, NVIDIA TITAN Xp (Pascal, CUDA 11.8 wheels), with torch 2.1.2, transformers
4.44.2, pyannote.audio 3.1.1, numpy 1.26.4, scipy 1.15.3, pyarrow 17.0.0. `scripts/setup_env.sh` builds the environment
with `uv`. Models: `Qwen/Qwen2.5-1.5B-Instruct` and `Qwen/Qwen2.5-0.5B-Instruct` at the revisions in
`configs/frozen_lists.json`; post-hoc scorers `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct` and
`microsoft/Phi-3.5-mini-instruct`, all in fp32 (`echobase/post_score.py`; 7B ran on one NVIDIA A100 80GB); pyannote 3.1 needs a Hugging Face token with its terms accepted.

## Level 1: from the caches (CPU)

    bash scripts/reproduce_paper.sh

It checks the file hashes, then runs the confirmatory analysis (`echobase.confirm`), the exploratory analyses
(`echobase.explore_eval` for both models, `echobase.explore_flips`, `echobase.e5_maintenance` for W1 and W3), the
post-hoc analyses of deviations D5, D6 and D7 for all five scorers (`explore_signed`, `explore_gold`, `explore_eval`,
`explore_calib`, `explore_ctxcost`, `explore_formal`, and `explore_scale` over the 136 meetings in
`configs/ami_scale_done.txt`, for the 0.5B and 1.5B models; `explore_confident`, `explore_pooled`,
`explore_chunksize`, `explore_crosshost`), the table and figure scripts and `paper/gen_numbers.py`, and builds `paper/main.pdf`.
GPU-only measurements (determinism checks, throughput) are not rerun; their reports are released as measured. The confirmatory script refuses to
run unless the STATUS.md header carries the freeze marker, which the released copy does.

## Level 2: from scratch (GPU)

    bash scripts/setup_env.sh && bash scripts/fetch_ami.sh   # environment, AMI audio and annotations
    python -m echobase.wiki prepare                   # Wikipedia sample (shard 0 of wikimedia/wikipedia 20231101.en)
    python -m pilots.run_diar --split eval --gpu 0    # W1 old side: pyannote 3.1, known speaker count
    python -m echobase.run_diar_v --pipeline 3.1 --estimate-speakers --split eval --gpu 0   # W3 old side
    for v in manual w1old placebo w3old merge2 merge4 merge8; do
        python -m echobase.score_versions --split eval --version $v --gpu 0; done
    python -m echobase.score_wiki --split test --gpu 0
    # second model: the same two scoring commands with --model Qwen/Qwen2.5-0.5B-Instruct (not for merge*)
    bash scripts/reproduce_paper.sh

Scoring is deterministic: `python -m echobase.e0_determinism --split eval --n 100 --out <file>` rescores held-out
segments under several batch budgets and reports label mismatches (0 in our runs).

## Mapping from the paper to the code

| Paper (LaTeX label; figure and table numbers follow float placement) | Code | Output |
|---|---|---|
| workload table (`tab:workloads`) | `paper/figs/workloads.py` | `paper/figs/workloads.tex`, `reports/tables/workloads.csv` |
| example figure (`fig:example`) | `paper/figs/example.py` | `paper/figs/example.tex` |
| H1/H2/H4, drift-per-workload figure (`fig:drift`) | `echobase/confirm.py`, `paper/figs/drift.py` | `reports/confirmatory.{json,md}` |
| E6 granularity figure (`fig:gran`) | `echobase/explore_eval.py`, `paper/figs/drift_all.py` | `reports/explore_eval.json` |
| per-meeting figure (`fig:perunit`) | `paper/figs/per_unit.py` | from `reports/confirmatory.json` |
| all-queries table (`tab:all`), E4 | `echobase/explore_eval.py --model ...`, `paper/figs/drift_all.py` | `reports/explore_eval_model2.json` |
| maintenance table (`tab:maint`), E5 | `echobase/e5_maintenance.py`, `paper/figs/maintenance.py` | `reports/e5_w{1,3}.json` |
| word flips | `echobase/explore_flips.py` | `reports/explore_flips.json` |
| throughput | `echobase/throughput.py` (reads the job journals) | `reports/throughput.json` |
| direction of drift (`sec:dir`, post-hoc, D5) | `echobase/explore_signed.py --model ...` | `reports/explore_signed_<model>.json` |
| human-label table (`tab:gold`, post-hoc, D5) | `echobase/explore_gold.py --model ...`, `paper/figs/gold.py` | `reports/explore_gold_<model>.json` |
| 3B model (post-hoc, D5) | `echobase/post_score.py` (check / ami / wiki), `echobase/explore_eval.py --model Qwen/Qwen2.5-3B-Instruct` | `reports/post_determinism_3b.json`, `reports/explore_eval_model3.json` |
| scale figure (`fig:scale`, post-hoc, D5) | `echobase/run_diar_scale.py`, `scripts/scale_loop.sh`, `echobase/explore_scale.py`, `paper/figs/scale.py` | `reports/explore_scale.json` |
| cross-host determinism | rescoring on a second server; `echobase/explore_crosshost.py` (1.5B) | `reports/post_crosshost_{0.5b,1.5b}.json` |
| models table (`tab:models`), 7B and Phi (post-hoc, D6) | `echobase/post_score.py`, `echobase/explore_eval.py`, `paper/figs/models.py` | `reports/explore_eval_{7b,phi}.json`, `reports/post_determinism_{7b,phi}.json` |
| calibration and context (`sec:calib`, post-hoc, D6, D7) | `echobase/explore_calib.py`, `echobase/explore_ctxcost.py` | `reports/explore_calib.json`, `reports/explore_ctxcost.json` |
| monotonicity, power curve, anchor accuracy (post-hoc, D6) | `echobase/explore_formal.py` | `reports/explore_formal.json` |
| confident-label drift (post-hoc, D7) | `echobase/explore_confident.py` | `reports/explore_confident.json` |
| pooled audit (post-hoc, D7) | `echobase/explore_pooled.py` | `reports/explore_pooled_w{1,3}.json` |
| chunk window 100/150 (post-hoc, D7) | `echobase/post_score.py wikisize`, `echobase/explore_chunksize.py` | `reports/explore_chunksize.json` |
| 1.5B at scale (post-hoc, D7) | `echobase/post_score.py --cache-name v2-scale-1.5B`, `echobase/explore_scale.py` | `reports/explore_scale_1.5b.json` |
| cost per workload (`tab:cost`, post-hoc, after review 4) | `echobase/explore_cost.py`, `echobase/throughput.py` (run times from the gpulab2 job journals), `paper/figs/cost.py` | `reports/explore_cost.json`, `reports/throughput.json`, `reports/tables/cost.csv` |

Post-hoc analyses were added after the confirmatory analysis and are logged as deviations D5, D6 and D7 in
`reports/deviations.md`. The level-1 script reproduces all of them from the caches.

# Artifact: Same Words, Different Answers

**Status (2026-11-14).** The paper was submitted to the ICDE 2027 research track (Experiment, Analysis and Benchmark)
and was withdrawn by the authors on 2026-11-14 for an administrative reason (a co-author's per-round submission
limit). The withdrawal does not concern the results: this package and its numbers are unchanged from v1.3, and the
work continues in an extended version. This release (v1.3.1) changes only this note and the metadata.

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
The canonical-unit caches (`outputs/echobase/v2-canon*`, D18/D19) are keyed by the text of each unit, so they contain
short spans of the AMI reference transcripts (AMI Meeting Corpus, CC BY 4.0). The second
corpora of D17-C/D19 are not redistributed either; `scripts/fetch_d17.sh` downloads them and checks their SHA-256: the
ICSI Meeting Corpus core NXT annotations (CC BY 4.0, https://groups.inf.ed.ac.uk/ami/icsi/) and the GovReport CRS test
split (US government works, https://huggingface.co/datasets/launch/gov_report).

## Environment

Tested on Ubuntu 18.04, Python 3.10, NVIDIA TITAN Xp (Pascal, CUDA 11.8 wheels), with torch 2.1.2, transformers
4.44.2, pyannote.audio 3.1.1, numpy 1.26.4, scipy 1.15.3, pyarrow 17.0.0. `scripts/setup_env.sh` builds the environment
with `uv`. Models: `Qwen/Qwen2.5-1.5B-Instruct` and `Qwen/Qwen2.5-0.5B-Instruct` at the revisions in
`configs/frozen_lists.json`; post-hoc scorers `Qwen/Qwen2.5-3B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-14B-Instruct`,
`Qwen/Qwen2.5-32B-Instruct` (D15; action item, question and dated text only) and
`microsoft/Phi-3.5-mini-instruct`, all in fp32 (`echobase/post_score.py`; 7B ran on one NVIDIA A100 80GB; 14B on two TITAN Xp
with CPU offload, `POST_MAX_MEM=7GiB POST_CPU_MEM=75GiB POST_BUDGET=6144`, about 90 GB of RAM; 32B on one A100 80GB with CPU offload, `POST_MAX_MEM=64GiB POST_CPU_MEM=140GiB POST_BUDGET=6144`, `scripts/d15_gp146.sh`); pyannote 3.1 needs a Hugging Face token with its terms accepted.

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

## Post-submission analyses (D17-D19)

The paper version uploaded before the deadline (Sec. X and the post-hoc paragraphs marked D17-D19) adds analyses whose
rules were logged in `reports/deviations.md` before any code or result. Level 1 reproduces them from the caches:

| Analysis | Code | Caches | Output |
|---|---|---|---|
| learned non-LLM scorer, TF-IDF + logistic regression on AMI train (D17-A, CPU) | `echobase/d17_learned.py`, then `explore_eval --partial`, `explore_signed`, `explore_gold`, the D14 comparison with `--cache v2-learned` | `v2-learned` | `reports/d17_learned_{fit,eval,signed,gold,oracle}.json` |
| word-F1 ceiling of segment labels (D17-A, CPU) | `echobase/d17_ceiling.py` | all scorer caches | `reports/d17_ceiling.json` |
| bf16 rerun noise vs segmentation drift (D17-B) | `scripts/d17b_bf16.sh` (GPU), `echobase/d17_bf16.py` | `v2-bf16-b3072`, `v2-bf16-b1024` | `reports/d17_bf16.json` |
| second corpora: ICSI placebo and merges, GovReport CRS chunker (D17-C: 1.5B; D19: 3B and 0.5B) | `scripts/fetch_d17.sh`, `echobase/d17_corpora.py` (`check`, `icsi`, `crs` on GPU; `analyze`), `scripts/d17c_corpora.sh`, `scripts/d19_second.sh`, `scripts/d19_half.sh`; the scorer is set by `EB_MODEL` | `v2-d17`, `v2-d17-3b`, `v2-d17-0.5b` | `reports/d17_corpora.json`, `reports/d19_corpora_{3b,half}.json`, `reports/{d17_corpora,d19_3b,d19_half}_determinism.json` |
| canonical units (`sec:canon`, `tab:canon`; D18: 1.5B; D19: 3B and 0.5B) | `echobase/d18_canonical.py` (`score` on GPU, `analyze` on CPU; `EB_MODEL`), `echobase/d18_wordsets.py` | `v2-canon`, `v2-canon-3b`, `v2-canon-0.5b` | `reports/d18_canonical.json`, `reports/d19_canonical_{3b,half}.json`, `reports/d18_wordsets.json` |

The 3B scorer runs in fp32 split over two 12 GB cards (`--gpus 0,1`); 0.5B and 1.5B fit on one. Determinism checks
(`check`) gave 0 of 600 label changes for each scorer. `d18_canonical analyze` needs several GB of RAM.

## Mapping from the paper to the code

| Paper (LaTeX label; figure and table numbers follow float placement) | Code | Output |
|---|---|---|
| workload table (`tab:workloads`) | `paper/figs/workloads.py` | `paper/figs/workloads.tex`, `reports/tables/workloads.csv` |
| example figure (`fig:example`) | `paper/figs/example.py` | `paper/figs/example.tex` |
| H1/H2/H4, drift-per-workload figure (`fig:drift`) | `echobase/confirm.py`, `paper/figs/drift.py` | `reports/confirmatory.{json,md}` |
| E6 granularity figure (`fig:gran`) | `echobase/explore_eval.py`, `paper/figs/drift_all.py` | `reports/explore_eval.json` |
| per-meeting figure (`fig:perunit`; not in the uploaded version) | `paper/figs/per_unit.py` | from `reports/confirmatory.json` |
| all-queries table (`tab:all`), E4 | `echobase/explore_eval.py --model ...`, `paper/figs/drift_all.py` | `reports/explore_eval_model2.json` |
| maintenance table (`tab:maint`), E5 | `echobase/e5_maintenance.py`, `paper/figs/maintenance.py` | `reports/e5_w{1,3}.json` |
| word flips | `echobase/explore_flips.py` | `reports/explore_flips.json` |
| throughput | `echobase/throughput.py` (reads the job journals) | `reports/throughput.json` |
| direction of drift (`sec:dir`, post-hoc, D5) | `echobase/explore_signed.py --model ...` | `reports/explore_signed_<model>.json` |
| human-label table (`tab:gold`, post-hoc, D5) | `echobase/explore_gold.py --model ...`, `paper/figs/gold.py` | `reports/explore_gold_<model>.json` |
| 3B model (post-hoc, D5) | `echobase/post_score.py` (check / ami / wiki), `echobase/explore_eval.py --model Qwen/Qwen2.5-3B-Instruct` | `reports/post_determinism_3b.json`, `reports/explore_eval_model3.json` |
| scale figure (`fig:scale`, post-hoc, D5; not in the uploaded version) | `echobase/run_diar_scale.py`, `scripts/scale_loop.sh`, `echobase/explore_scale.py`, `paper/figs/scale.py` | `reports/explore_scale.json` |
| cross-host determinism | rescoring on a second server; `echobase/explore_crosshost.py` (1.5B) | `reports/post_crosshost_{0.5b,1.5b}.json` |
| models table (`tab:models`), 7B and Phi (post-hoc, D6) | `echobase/post_score.py`, `echobase/explore_eval.py`, `paper/figs/models.py` | `reports/explore_eval_{7b,phi}.json`, `reports/post_determinism_{7b,phi}.json` |
| models table, 14B (post-hoc, D9), cross-hardware label check | `echobase/post_score.py`, `scripts/d9_worker.sh`, `echobase/explore_eval.py`, `paper/figs/models.py` | `reports/explore_eval_14b.json`, `reports/post_determinism_14b{,_gp8}.json`, `reports/d9_crosshw_{ami_gp231,wiki_gp8}_vs_gpulab2.json`, `reports/d9_verify_gp231.json` |
| calibration and context (`sec:calib`, post-hoc, D6, D7) | `echobase/explore_calib.py`, `echobase/explore_ctxcost.py` | `reports/explore_calib.json`, `reports/explore_ctxcost.json` |
| monotonicity, power curve, anchor accuracy (post-hoc, D6) | `echobase/explore_formal.py` | `reports/explore_formal.json` |
| confident-label drift (post-hoc, D7) | `echobase/explore_confident.py` | `reports/explore_confident.json` |
| pooled audit (post-hoc, D7) | `echobase/explore_pooled.py` | `reports/explore_pooled_w{1,3}.json` |
| chunk window 100/150 (post-hoc, D7) | `echobase/post_score.py wikisize`, `echobase/explore_chunksize.py` | `reports/explore_chunksize.json` |
| 1.5B at scale (post-hoc, D7) | `echobase/post_score.py --cache-name v2-scale-1.5B`, `echobase/explore_scale.py` | `reports/explore_scale_1.5b.json` |
| non-LLM reference scorers, series check (Sec. VII, post-hoc, D14, CPU) | `echobase/explore_oracle.py` | `reports/explore_oracle.json` |
| 32B scorer (post-hoc, D15) | `echobase/post_score.py --cols`, `scripts/d15_gp146.sh`, `echobase/explore_eval.py --partial` | `reports/explore_eval_32b.json`, `reports/explore_{signed,gold}_Qwen2.5-32B-Instruct.json`, `reports/post_determinism_32b_gp146.json`, `reports/d15_env_check.json` |
| aggregate certification (Sec. IX, post-hoc, D12, CPU) | `echobase/explore_aggcert.py` | `reports/explore_aggcert.json` |
| replay scalability (`fig:scalability`, post-hoc, D11, CPU; figure not in the uploaded version) | `echobase/explore_scalability.py`, `paper/figs/scalability.py` | `reports/explore_scalability.json` (wall times depend on the machine; cells, calls and shares are exact) |
| cost per workload (`tab:cost`, post-hoc, after review 4) | `echobase/explore_cost.py`, `echobase/throughput.py` (run times from the gpulab2 job journals), `paper/figs/cost.py` | `reports/explore_cost.json`, `reports/throughput.json`, `reports/tables/cost.csv` |

Post-hoc analyses were added after the confirmatory analysis and are logged as deviations D5-D19 in
`reports/deviations.md`. The level-1 script reproduces all of them from the caches.

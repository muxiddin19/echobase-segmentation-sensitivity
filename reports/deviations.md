# Deviations from the frozen design (G1, 2026-10-27)

Each entry gives the date, what changed, why, who approved it, and which results it touches. These entries are for the
supplementary material.

## D1: Pilot B meeting coverage (2026-10-28)

- **What:** Pilot B's data is "the dev meetings that have AMI dialogue-act annotations": 13 of the 18 dev meetings and
  10,312 dialogue acts. The meeting check in `pilots/verdict.py` (`verdict_pilotB`) and in `program_pilotB.md` changes
  from `< 18` to `< 13`.
- **Why:** AMI 1.6.2 has no dialogue-act annotation for IB4001, IB4002, IB4004, IB4010 or IB4011. Those 5 meetings
  cannot contribute ground truth to Pilot B. Under the frozen rule (`< 18 → ANOMALY`) the verdict would be ANOMALY
  whatever the sampler result, so the rule tested a coverage fact instead of the pilot question. The gap was missed
  before G1. The units file `runs/pilotB/labels/s0/run/units.json` reports `meetings=13`.
- **What does not change:**
  - the estimator, budgets, trials, seeds and δ;
  - the PASS, FAIL and INCONCLUSIVE thresholds;
  - every other anomaly rule;
  - Pilot A, which uses segments and ran on all 18 meetings.
- **When it was decided:** before any Pilot B sampler result was read. At decision time the 6 sampler jobs were
  still running after the plumbing fix. One `result.json` (global, seed 0) had been written, but none had been opened.
- **Approval:** the user approved option 1 ("the one which fits our target aim") on 2026-10-28. The human re-writes the eval lock
  after this change.
- **Results touched:** the Pilot B verdict only.

## D2: W3 diarizer-upgrade workload uses the plan's fallback (2026-10-28, before G1)

- **What:** W3 is "pyannote 3.1 with clustering threshold 0.55 (old) → pyannote 3.1 default, threshold 0.7046 (new)".
  Both sides use the known speaker count and the same reference words. Plan v2 §E0.4 named this fallback in advance:
  "replace W3 by pyannote 3.1 with a different clustering threshold". Code: `echobase/run_diar_v.py --threshold`, with
  `W3_OLD` in `echobase/score_versions.py`. The value 0.55 was fixed before any W3 output existed.
- **Why:** gpulab2 hard-reset twice, each time within seconds of the pyannote 3.0 job starting:
  - the boot ended at about 13:10, when `e0-diar30` started after `e0-wiki`;
  - the boot ended at 15:18:35, 6 s after the job was relaunched, with no other job running.

  No kernel message was logged. gpulab2 is shared, so 3.0 is not run there again. No 3.0 output was ever produced.
- **What does not change:**
  - the H3 threshold, test and position in the fixed sequence;
  - the queries;
  - the E0.4 rule (change share ≥ 0.10 on dev), which still applies to the replacement.
- **Caveat for the paper:** W3 is now a configuration retune of one diarizer, not a release upgrade. The text calls
  it that and does not claim a version change.
- **Approval:** the user chose "3.1 threshold variant (Recommended)" on 2026-10-28.
- **Results touched:** H3 and every W3 analysis.

## D3: H3 dropped from the confirmatory family; W3 becomes exploratory (2026-10-28, before G1)

- **What:**
  - The confirmatory family becomes H1 → H2 → H4, a fixed sequence of 3 tests.
  - W3 is redefined as pyannote 3.1 with an estimated speaker count (old) → pyannote 3.1 with the known count (new),
    both filled with the same reference words.
  - W3 is reported with the H2 statistic but with no decision rule (plan v2.2 §8.2).
  - Code: `run_diar_v.py --estimate-speakers`, and `W3_OLD` in `score_versions.py` becomes `outputs/diar_v/3.1-est`.
- **Why:** both pre-specified W3 definitions failed E0.4 on dev:
  - pyannote 3.0 crashed the host (D2);
  - 3.1 at threshold 0.55 → default changed 4 of 1,021 segments, a share of 0.004 against the rule of ≥ 0.10.

  A third definition chosen after two dev failures would be a forking path if made confirmatory. So the new
  definition is exploratory only.
- **What does not change:**
  - H1, H2 and H4: thresholds, tests and order;
  - the queries, models and statistics.
- **Approval:** the user chose "Drop H3, W3 exploratory (Recommended)" on 2026-10-28.
- **Results touched:** H3 is removed. The W3 analyses become exploratory. The paper states the diarizer release
  upgrade as a limitation.

## D4: Eval gate fixed after the G1 lock was written (2026-10-28, before any eval run)

- **What:**
  - New file `echobase/gate.py`. Eval and test work now needs a dated marker in the `STATUS.md` header:
    `- **EchoBase G1:** YYYY-MM-DD`.
  - The marker check replaces the bare `"G1" in header` check in `confirm.py`, `score_versions.py`, `score_wiki.py`
    and `run_diar_v.py`. `e0_determinism.py --split eval` is gated too.
  - `gate.py` is added to LOCKED.
  - Tests are in `echobase/tests/test_gate.py`.
- **Why:** the header already said "G1 freeze done 2026-10-27" from the pilot phase. So the old check had always
  passed and never blocked eval or test work.
- **When:**
  - The human wrote the lock at 2026-10-28 08:07:54 UTC.
  - The bug was found afterwards. The fix edited 4 locked files (8 changed lines, gate checks only) before the
    deviation was approved. `eval_lock.py --check` reported the violation.
  - The hook that should have blocked the edit was inactive: the Claude session was started from another directory,
    so the project's `.claude/settings.json` hooks were not loaded.
- **Effect on results:** none.
  - Before the fix, the server held no eval-meeting outputs, no Wikipedia test outputs and no eval or test log
    (checked on gpulab2).
  - No rule, statistic, data, model or prompt changed.
- **Approval:** the user chose "Keep fix as D4, re-lock" on 2026-10-28. The human re-writes the lock.

## D5: Post-hoc exploratory additions after an internal review (2026-10-29, after the confirmatory analysis)

- **What:** four analyses that are not in PREREGISTRATION §8, all labeled *post-hoc exploratory* in the paper:
  1. signed and absolute drift per query (existing caches, CPU);
  2. correctness against AMI gold dialogue acts for the question column under each segmentation (CPU);
  3. a larger scorer, Qwen/Qwen2.5-3B-Instruct, fp32 split over both GPUs, on W1, W2 and W4, after its own
     determinism check;
  4. scale: the AMI meetings outside the dev and eval lists (training split), diarized with the frozen W1 pipeline and
     scored with Qwen2.5-0.5B, for a drift-versus-corpus-size and cost curve.
- **Why:** an independent referee-style review (2026-10-29) judged the draft a weak reject. Its main points were the
  missing direction of drift, no link to correctness, small models and small scale.
- **What does not change:**
  - the confirmatory results, which ran once (`reports/confirmatory.*`);
  - every locked file (new scripts only; the program's "no new models/units after G1" is set aside for these
    exploratory analyses only, by this deviation);
  - the paper's claims from H1, H2 and H4.
- **Approval:** the user selected all four on 2026-10-29, and chose to present the maintenance analysis as a measured
  negative result without a system name.
- **Results touched:** none of the confirmatory results; adds exploratory sections and floats.
- **Addendum (2026-10-29 08:20):**
  - The user offered gp231 GPU 0 for the scale analysis (item 4), so it runs there, not on gpulab2. Audio is stored on
    the lab NAS (`/nas/muhiddin/datasets/AMI/audio`) at the user's invitation. GPU 1 of gp231 belongs to another user
    and is not used.
  - The AMI mirror served about 25 KB/s per connection, so not all 137 meetings may arrive in time. Meetings are
    fetched in a fixed random order (seed 0, `logs/scale_order.txt`), so the subset finished at the cut-off
    (results freeze, Nov 4) is a random sample. The number used will be reported.
  - The 3B scorer's determinism check: 0 label mismatches over 200 segments × 3 columns under two batch budgets
    (`reports/post_determinism_3b.json`).

## D6: Post-hoc additions after a second internal review (2026-10-29, after the confirmatory analysis)

- **What:** four further *post-hoc exploratory* analyses, settings fixed here before any of them is run:
  1. **Mitigation with cost:** context-padded scoring. Each segment is scored with the two preceding and two following
     segments of the same version shown as context, and only the target segment is labeled. Primary model, AMI eval
     versions manual, w1old and placebo. Reported: drift, agreement with the human labels of D5, extra tokens.
  2. **Calibration baselines** on the fixed manual segmentation:
     - D between models (1.5B vs 0.5B vs 3B) from the existing caches;
     - D between the frozen prompt and one paraphrased prompt (same answer sets and scoring), for the 1.5B and 0.5B
       models.
  3. **Other-family model:** microsoft/Phi-3.5-mini-instruct, fp32 split over two cards. Runs a determinism check
     first, then AMI eval manual, w1old and placebo, and Wikipedia test.
  4. **CPU analyses:**
     - a monotonicity proposition for "any part" predicates and the LLM's violation rate on the merge caches;
     - an audit power curve (injected error from eps-0.05 to eps+0.05);
     - per-column anchor accuracy.
- **Why:** a second referee-style review (2026-10-29, weak reject) asked for a remedy, a scale against which to read
  D, a model from another family, and formal content.
- **What does not change:** the confirmatory results, every locked file (new code paths only), and the claims from
  H1, H2 and H4.
- **Approval:** the user selected all four on 2026-10-29.
- **Addendum (2026-10-29 14:50):**
  - A colleague offered his A100 80GB card (a third lab host, GPU 0), which he is still using. The user approved
    adding **Qwen/Qwen2.5-7B-Instruct** in bf16 on that shared card for the same post-hoc analyses as 3B: a
    determinism check, then AMI eval manual, w1old and placebo, then Wikipedia test.
  - Its labels come from different hardware and precision, so they are compared only within the 7B run, never mixed
    with TITAN Xp caches.
  - GPU 1 on that host belongs to another user and is not used.
  - Phi's batch budget was lowered to 1536 tokens after two out-of-memory failures. Its determinism check passed:
    0 label mismatches out of 600.
- **Addendum (2026-10-29, 7B precision):** Qwen2.5-7B in bf16 failed the determinism check: under two batch budgets,
  entity count changed 2 of 200 labels and topic 3 of 200 (action item 0; `reports/post_determinism_7b_bf16.json`).
  The bf16 run was stopped and its partial cache deleted before any analysis. 7B was restarted in fp32 on the same
  card and must pass the same check before its labels are used.
  Result: the fp32 run passed (0 of 600 labels changed, `reports/post_determinism_7b.json`) and ran 15:45-22:19 KST
  (6.57 GPU-h, logged). The short bf16 attempt before it is not in the ledger; its time is not recorded.

## D7: Post-hoc additions after a third internal review (2026-10-30, after the confirmatory analysis)

- **What:** four further *post-hoc exploratory* analyses, settings fixed here before any of them is run:
  1. **More rewordings:** three further paraphrases of every AMI column question (`PARAPHRASES` para2-para4 in
     `echobase/post_score.py`; answer sets and scoring unchanged). Primary model, AMI eval manual segments only.
     Reported: D between the frozen prompt and each rewording (`explore_calib`), so the prompt scale is a range of four.
  2. **Realistic chunk sizes:** the locked fixed chunker (120 words) at 100 and 150 words, primary model, Wikipedia
     test. Reported: drift of the dated share (Qd1) for 120 -> 100 and 120 -> 150, as a pipeline-style revision.
  3. **CPU analyses:**
     - drift restricted to confident labels: Q1/Q2 shares recomputed over segments whose first-token logit margin is
       at least 2.0 (fixed here), for W1 and W2 with every scorer whose cache has margins;
     - a pooled audit in the maintenance replay: anchor values for every non-identical cell, one stratum over all
       columns, audited with the locked `certify.audit` at alpha = delta (no union bound). Its guarantee is weaker
       (overall cell error <= eps, not per column), which the paper must state. W1 and W3, eps 0.02-0.15, 100 trials.
  4. **1.5B at scale:** the primary model on the 136 scale meetings (`configs/ami_scale_done.txt`), versions manual,
     w1old and placebo, on gp231, cache `outputs/echobase/v2-scale-1.5B` (kept out of the confirmatory v2 cache).
     Before it, on the same card: the determinism check (`post_score check`) and a cross-host check (the 16 eval
     manual meetings rescored on gp231 into `v2-gp231-check` and compared with the gpulab2 v2 cache).
- **Why:** a third referee-style review (2026-10-30, weak reject) said the prompt scale rests on one rewording, the
  realistic revisions are small, a weak scorer drifts by construction, the audit baseline cannot win against hashing,
  and the scale run used only the second model.
- **What does not change:** the confirmatory results, every locked file (new code paths only), and the claims from
  H1, H2 and H4. Results that weaken a claim are reported as they come.
- **Approval:** the user selected all four on 2026-10-30.
- **Addendum (2026-10-30 11:40):**
  - Rewordings para2-para4 and the cross-host rescoring are done; the 1.5B determinism check on gp231 gave 0 of 600
    mismatches, the cross-host check 0 of 37,450.
  - The chunk-size run writes to `outputs/echobase/v2/wiki/fixed100` and `fixed150`, new subfolders of the primary
    model's cache (post_score uses cache_root for wikisize). No existing file is changed; only the scale run uses its
    own cache.
  - The first chunk-size launch failed (no Wikipedia articles in gp231's checkout) and was relaunched as unit d7-wiki.

## D8: Cost per workload and window-size means after a fourth internal review (2026-10-31, CPU only)

- **What:** no new model runs. (1) `echobase/explore_cost.py` counts scorer calls and prompt tokens (primary tokenizer,
  frozen prompt) for every held-out version; `paper/figs/cost.py` turns them into GPU-minutes at the measured rates.
  (2) `echobase/throughput.py` now also reads the Wikipedia test run's time from the e1-gpu1 journal (start = the `date`
  line before it, end = its "done test" line, model loading included); the existing entries are reproduced exactly.
  The 100/150-word windows were scored on gp231 without timing and are converted at the Wikipedia run's token rate,
  marked in the table. (3) `explore_chunksize.py` also reports means and the number of articles whose dated share
  changes; the existing fields are reproduced exactly. (4) A table of the status of every analysis (confirmatory,
  preregistered exploratory, D5-D8).
- **Why:** a fourth referee-style review (2026-10-31, weak reject) asked for cost next to drift per workload, a named
  artifact, and means for the lattice-valued window drifts.
- **What does not change:** every result already reported, every locked file. The two self-citations in the threats
  paragraph were removed as off-topic (two reviews); the user delegated this editorial decision on 2026-10-31.
- **Approval:** the user delegated scholarly decisions on 2026-10-31 ("feel free to do some decision").

## D9: A 14B scorer with CPU offload (2026-11-03, after the confirmatory analysis)

- **What:** Qwen/Qwen2.5-14B-Instruct, fp32, same prompts and scoring, on gpulab2 (both TITAN Xp cards, 11 GiB of
  weights each) with the remaining layers in CPU RAM (`POST_CPU_MEM`, accelerate offload). Same protocol as 7B:
  first the determinism check (`post_score check`, two batch budgets, 200 segments x 3 columns; any mismatch stops
  the run and the model is not used), then AMI eval manual, w1old and placebo, then Wikipedia test. Cache
  `outputs/echobase/v2-Qwen2.5-14B-Instruct`. Reported like the other post-hoc scorers (Tab. models, signed drift,
  agreement with human labels), labeled post-hoc.
- **Why:** reviews 3-5 named the weak agreement of every scorer with the human labels (question F1 at most 0.30) as
  the main limitation; the user asked for a stronger scorer "if it helps" (2026-10-31) and offered gpulab2
  (2026-11-03). The A100 has no room (about 41 GB free of the ~60 GB needed).
- **Feasibility rule (fixed here):** a timing test of the first batches decides. The run starts only if its estimated
  total finishes by 2026-11-05 18:00 KST; otherwise D9 is abandoned and reported as not run. If it starts but has not
  finished by the 2026-11-06 freeze, its partial results are not used.
- **What does not change:** the confirmatory results, every locked file, all earlier results.
- **Approval:** the user, 2026-10-31 and 2026-11-03.
- **Addendum (2026-11-04 11:30 KST):** the first check launch ran out of GPU memory while loading (10 GiB of weights per
  card left no room for a 2.9 GiB embedding tensor); relaunched with 7 GiB per card and the rest in RAM. Determinism
  check passed: 0 of 600 labels changed under two batch budgets (`reports/post_determinism_14b.json`), 1,200 calls
  in about 12 minutes. Estimated full run 20-26 h, finishing before 2026-11-05 18:00 KST, so by the rule above the
  run started (unit d9-run, AMI eval manual/w1old/placebo then Wikipedia test, batch budget 6,144 tokens, which the
  check covered). The download needed two attempts (CDN errors); gpulab2's disk is at 99% (7.5 GB free).
- **Addendum (2026-11-04 12:00 KST):** to finish earlier, the Wikipedia part runs in parallel on GPU3 (host gp8, one
  RTX 3090 24 GB, card 1, free; offered by its user) with CPU offload; the AMI part stays on gpulab2. gp8 runs its own
  determinism check first (same rule: any mismatch and its labels are not used, and gpulab2 scores Wikipedia as
  planned). Both chunkers of an article are scored on the same host, so every Wikipedia drift compares labels from one
  host. Code: `post_score` takes the offload path for a single card when `POST_CPU_MEM` is set (earlier runs, which
  never set it, are unchanged). Environment copied from gp231 (same Python and package versions); model files
  downloaded to the NAS.

- 2026-11-04 15:10 KST: the first gp8 launch failed before scoring (outputs/diar and data/ami/annotations were not on gp8). Both were copied from gpulab2, manual and w1old segment counts were checked against gpulab2 (EN2002a 746/590, EN2002b 488/419), and d9-gp8c was relaunched on card 1. No labels were produced by the failed launch.
- 2026-11-04 15:40 KST (user request "exploit all available GPUs"): extra D9 workers. scripts/d9_worker.sh first rescores the manual version of IS1009a (already scored on gpulab2) and requires every label in all five columns to equal gpulab2's (reference copy in outputs/echobase/d9-ref); only then does it score its share. Each comparison stays within labels that are identical across hosts, so hosts can be mixed. Shares: gp231 card 1 (TITAN Xp, the only free card there) placebo for the first 8 eval meetings (configs/d9_placebo_a.txt); gpulab2 continues manual, then w1old; placebo for the last 8 meetings and w1old from the end (configs/d9_placebo_b.txt, d9_w1old_rev.txt) are reserved for the university cluster. gp8 card 1 is pinned with CUDA_VISIBLE_DEVICES=1, after a 254 MB CUDA context was seen on the user's card 0; the job was restarted before any label was written. Busy cards were not used: gp114 (both cards in use by other users' jobs and the user's own), xvoice (both A100s in use by two other users).
- University cluster: code, data and a Python 3.10 environment were started under the shared xvoice account; the PyTorch install failed with "Disk quota exceeded". Paused there: the account is shared with other lab members, and filling its quota (venv ~5 GB plus 28 GB of weights) could make their jobs fail. Resumes only once the quota is known.
- 2026-11-04 16:30 KST: gp231 verification passed (IS1009a manual, 195 segments x 5 columns: 0 label mismatches with gpulab2; largest margin difference 3.7e-4). gp8 determinism check passed (0/600). gp231 then takes placebo for the last 8 meetings in reverse order (configs/d9_placebo_b_rev.txt, queued as d9-gp231b); its files are copied into gpulab2's cache before gpulab2 reaches placebo, so gpulab2 skips them.
- 2026-11-05 06:30 KST: AMI part done on gpulab2 alone (manual, w1old, placebo: 16 meetings x 5 columns, 240 files; finished 01:15). gp231 stopped at 16:57 on Nov 4 (GPU out of memory at the third placebo meeting); its 11 completed placebo files match gpulab2 exactly (6,804 labels, 0 mismatches, largest margin difference 7.1e-4) and are kept only as a cross-host check. gp8 rebooted at 22:30 on Nov 4 and its Wikipedia job stopped after 145 of 200 articles (no error logged); both gp8 cards have since been in use by the user's other project, so it was not restarted. Wikipedia is scored on gpulab2 alone (123/200 at 06:20), so every reported comparison comes from one host; gp8's partial Wikipedia labels serve as a cross-hardware check (RTX 3090 vs TITAN Xp). AMI cache backed up to /nas/muhiddin/icde2027_echobase/combined.
- 2026-11-05 07:30 KST: cross-hardware check, Wikipedia: gp8 (RTX 3090, card 1 + CPU offload) vs gpulab2 (2x TITAN Xp + CPU offload) on the 867 files both had (145 articles, both chunkers, 3 columns): 13,056 labels, 0 mismatches, largest margin difference 2.0e-3 (reports/d9_crosshw_wiki_gp8_vs_gpulab2.json). With the AMI check (gp231 vs gpulab2, 6,804 labels, 0 mismatches), fp32 first-token labels of the 14B model do not depend on the card or on the GPU/CPU split.
- 2026-11-05 10:40 KST, D9 result: Wikipedia finished on gpulab2 at 09:31 (18,165 calls). Qwen2.5-14B (fp32) added to Tab. models, the confident-label analysis and the text; all 1,733 earlier macros unchanged (171 added). Base action-item share 4.2%; W1 D 0.27 (|change| 0.9 pts), placebo D 0.38 (1.3 pts), chunker D 0.11 (6.0 pts); question F1 0.29. Every relative drift positive, as for the other five scorers. One claim changed: the placebo raises the question share for 14B as for 7B (+7.0 pts, 15 of 16 meetings), so the text now says the two largest models reverse the direction instead of calling 7B the single exception. explore_confident was rerun on gpulab2 after copying the 7B cache there from the NAS (the first rerun lacked it and dropped the 7B row; caught by the LaTeX build).

## D10: Editorial corrections after an integrity audit (2026-11-05, no new runs)
- **What:** two independent audits (references; venue rules, preregistration fidelity and fair comparison) and checks by the assistant (eval lock, every literal number in the text, re-verification of each reference on Crossref/arXiv/publisher pages). Changes:
  - References: Cyclex removed from the intro sentence on LLM-maintenance work (it is not LLM work); the intro's operator claim now cites Evaporate, LOTUS, Palimpzest; the cost-scaling preprint is described as a synthetic study of embedding indexes, with its venue (IES 2026); LOTUS title corrected; Enzyme cited as SIGMOD Companion 2026 and Palimpzest as CIDR 2025, ZenDB as ICDE 2025 (author spelling Zeighami); AMI year/LNCS/pages; DOIs added; two uncited self-citations deleted from the bib (they never printed); added and verified: Bredin 2023, the BUTSpeechFIT AMI diarization lists, the pyannote/segmentation-3.0 model card, the Phi-3 report, SemBench (PVLDB 19), Kassaie and Tompa (PVLDB 16), SUPG (PVLDB 13). BARGAIN was suggested but not verified and not added.
  - Labels: "exploratory" added where preregistered-exploratory results stood next to confirmatory ones (other queries, the paired placebo-vs-W1 difference, granularity and maintenance findings); the abstract no longer calls the preregistered maintenance replay post-hoc.
  - Stale text after D9: four of six scorers / the two largest reverse; six scorers; five post-hoc rounds (D5-D9).
  - Fairness and validity: the anchor policy is described as our adaptation of stable-row reuse (SemReuse and sIVM were not run); prompts were designed with the primary model only; scorer validity, multiplicity and review-driven post-hoc rounds added to Threats; dated freeze timeline (hashes 17:14 KST, held-out scoring 17:19, confirmatory 20:21 on 2026-10-28).
  - Data: AMI license (CC BY 4.0, checked on the corpus site), no new human-subject data; Wikipedia sampling frame (first of 41 shards); pyannote training-data claim now sourced (model card; AMI standard test list = our 16 held-out meetings, verified on the BUTSpeechFIT lists).
  - AI declaration: states that the assistant checked references, that D8 editorial changes were delegated, and that assistant instances audited references and preregistration fidelity.
  - Figures embed TrueType fonts (pdf.fonttype 42).
- **Numbers:** no number changed; every changed sentence uses existing macros.
- **Approval:** the user asked for this audit on 2026-11-05 ("fits all ICDE 2027 requirements ... research ethics followed, including fair comparison, not hallucinating reference, data").
- **Reproduction check (2026-11-05):** `scripts/reproduce_paper.sh` was rerun CPU-only from the cached labels (gpulab2; the
  steps whose caches live only on gp231, i.e. scale, cross-host, context, paraphrase and wiki chunk-size, on gp231 with
  sha256-identical code). All 36 reports match the released ones (floats to rel. 1e-9) except `explore_calib.json`, which
  differs only by the pair added below; every earlier pair is identical.
- **Addendum to D9 (2026-11-05):** the calibration (`explore_calib`) adds the model swap 1.5B->14B on the fixed manual
  segmentation, from the existing 14B cache (no new scoring). The 14B labels are identical across the hosts that produced
  them (0 of 19,860), so the hardware caveat that keeps 7B out of this comparison does not apply. Result: action-item D
  0.842 [0.785, 0.902]; the text now says four model swaps. No other number changed.

## D11: Scalability of the maintenance replay and a cross-hardware check for 7B (2026-11-05 19:00 KST, before either run)

- **Why:** the audit of 2026-11-05 found two gaps. (1) The paper has no scalability experiment for the maintenance
  machinery, which ICDE expects of a data-engineering contribution. (2) The 7B scorer is the only one whose labels were
  never compared across hardware (A100 vs TITAN Xp), so the model-swap calibration leaves it out. Both are post-hoc and
  exploratory; neither touches a locked file, the confirmatory results or any earlier number. The user approved both on
  2026-11-05 ("yes, please do them accordingly").
- **(a) Scalability (CPU only, no new model outputs).** New script `echobase/explore_scalability.py`.
  - Data: the 136 scale meetings (`configs/ami_scale_done.txt`), workload W1 (old = w1old, new = manual), labels of
    the primary 1.5B model (`outputs/echobase/v2-scale-1.5B`, D7). The cell records, policies and audit are the locked
    code (`e5_maintenance.cells` logic, `replay.run_policy`, `certify.audit`), unchanged.
  - Sizes: nested prefixes of the seed-0 meeting order (8, 16, 32, 64, 136 meetings); then the 136-meeting cell set
    replicated x2, x4, x8 (copies get new meeting ids; this tests the algorithm's scaling, not new data, and is labeled
    as replication).
  - Measured per size: wall time of stage 1 (lineage, change map, rule derivation, anchoring; real prefixes only) and
    of stage 2 per policy (hash; anchor_cert, echobase, echobase_conf at eps 0.05; one trial, median of 5 repeats with
    seeds 0-4), changed cells, LLM calls, peak RSS. Single thread (OMP/MKL threads 1), host gp231 (CPU model reported).
  - LLM time of the calls each policy makes is computed from the measured 1.5B throughput on TITAN Xp
    (`reports/throughput.json`, manual, 43.1 calls/s); the report gives the overhead as a fraction of that time.
  - Rules fixed now: the log-log slope of stage-2 time vs cells (least squares over all sizes) is reported per policy;
    the text may call the replay "linear" only if every slope is <= 1.10, otherwise it reports the slopes. If the
    overhead exceeds 10% of the LLM time of the calls for any policy at 136 meetings, the text says so.
- **(b) 7B cross-hardware labels.** Qwen/Qwen2.5-7B-Instruct, the same revision as the A100 run
  (a09a35458c702b33eeacc393d103063234e8bc28), fp32, on gpulab2 (both TITAN Xp, idle; remaining layers in CPU RAM as in
  D9; weights held in /dev/shm because the disk is full, removed afterwards). It rescores the manual segmentation of the
  AMI eval meetings in list order, starting with IS1009a, into a separate cache (`d11-7b-gpulab2`), until 6 h of wall
  clock pass (stopping at a meeting boundary); every label is compared with the A100 cache.
  - Rule fixed now: 0 mismatches over all compared labels -> 7B joins the model-swap calibration (pair 1.5B->7B, already
    computed in `explore_calib.json`) and the text drops its hardware caveat. Any mismatch -> 7B stays out, and the
    paper reports the mismatch count instead. Either way the count is reported.
  - GPU time is logged in `reports/gpu_ledger.csv`.
- **Result (a), 2026-11-05 19:05 KST** (gp231, Intel i9-7900X, one thread, 9 min; `reports/explore_scalability.json`):
  323,535 changed cells on the 136 real meetings, 2,588,280 with x8 replication. Row mapping 169 s; policies with audit
  at most 0.9 s (real) and 4.2 s (x8), 3.7 GB peak memory. Log-log slopes: mapping 0.83, policies 0.66-0.98, all
  <= 1.10, so by the rule above the text may say the replay grows no faster than linearly. Overhead at most 3.2% of the
  LLM time of the calls left (rule: report if > 10%; it is not). Call shares at eps 0.05: hash 0.742, anchor + audit
  0.742, rules/anchor + audit 0.714 (0.703 at x8), + confidence strata 0.702 (0.692): the savings do not grow with
  size. Paper: new paragraph and Fig. `fig:scalability` in Sec. RQ4, Tab. status row D11; `reproduce_paper.sh` compares
  only cells, calls, shares and errors of this report (timings depend on the machine).
- **Addendum (b), 2026-11-05 19:25 KST, before any 7B label was produced:** gpulab2's network failed for bulk data
  (Hugging Face download stalled at 0.35 MB/s; copies from the lab reset after 58 MB), which would put the weights there
  after the freeze. No lab GPU is free (gp231, gp8, gp114 and the A100 host all run other users' or the user's jobs).
  The check therefore runs on **CPU** (gp231, Intel i9-7900X, fp32, 8 threads, nice 19), a larger hardware difference
  than GPU vs GPU. Weights: the A100 host's own snapshot a09a3545..., copied file by file; sha256 of all four shards
  identical to the source. New opt-in switch `POST_DEVICE=cpu` in `echobase/post_score.py` (not locked; GPU paths
  unchanged). A watchdog stops the run (by PID) if the host's available memory falls below 8 GB, to protect other users'
  jobs. Meetings, 6 h budget, separate cache (`d11-7b-cpu`) and the decision rule are unchanged from (b) above.
- **Result (b), 2026-11-05 20:30 KST** (`reports/d11_7b_crosshw.json`): gp231 rebooted without a clean shutdown at
  about 20:05 KST, 45 min into the run, during EN2002a. The system log ends at 20:05:01 with no out-of-memory, thermal,
  machine-check or panic message; available memory was 42 GB and the watchdog had not fired. The cause is unknown; the
  run had kept 8 CPU threads at full load for 40 min, so a contribution cannot be ruled out. A colleague's jobs on the
  host were also ended. The run was not restarted on gp231. Every file written before the reboot is complete (no
  empty or truncated JSON) and compared: IS1009a, all five columns (975 labels), and EN2002a, action item and question
  (1,492 labels). **2,467 labels, 0 mismatches**; margins differ by at most 0.0004. By the rule above, 7B joins the
  model-swap calibration (1.5B->7B: action-item D 0.944, already in `explore_calib.json`); the text states the count
  and that the run was cut short. The coverage (2,467 labels) is smaller than the planned 6 h and than 14B's 19,860.
  CPU only; no GPU time to log. The 7B weights remain in `~/muhiddin/d11_hf` on gp231 (15 GB) until the user decides.
- **Addendum (b), 2026-11-06 11:05 KST:** the host's journal shows further unclean crashes after the run, with none of
  our jobs running: 11-05 22:23, 11-06 00:22, 04:45 (51 s after boot) and 05:17 (41 s after boot); it was powered on
  again at 10:53. Crashes within a minute of boot point to a host hardware or power fault rather than the run's load;
  the cause is still unconfirmed. The result above is unaffected (files verified complete). No further runs on gp231
  until it is repaired.

## D12: Sample-based certification of the aggregate answer (2026-11-06 13:00 KST, before any code or result)

**Why.** A sixth independent review (an AI assistant acting as an ICDE referee, 2026-11-06) judged the maintenance
section a purely negative result: per-cell reuse saves nothing over content hashing. It asked whether the *query
answer*, rather than every cell, can be re-certified after a re-segmentation for fewer LLM calls. This is post-hoc and
exploratory, CPU only, from the cached labels of the primary model (no new scoring, no locked file changed).

**Design (fixed now).**
- Data: the changed cells of the maintenance replay (`e5_maintenance.cells`), workloads W1 and W3, AMI eval (16
  meetings), binary columns action item and question. Query per meeting and column: the word-weighted yes-share
  Q(S_new) over all new segments. Unchanged segments and changed cells with identical text (rule `hash`) are known
  exactly at no cost; the remaining cells R are unknown until scored.
- Estimators of the R part, from a simple random sample without replacement of n cells of R per meeting and column:
  (a) **sample**: the expansion estimator N_R x mean(w y);
  (b) **difference**: the free proxy p (the structural rule's value where a rule fires, else the anchor's old value,
  else 0) is summed over R and the sample estimates the correction: sum_R(w p) + N_R x mean(w (y - p)).
  Both with a normal 95% interval and finite-population correction; the interval is reported on Q in percentage
  points.
- Sampling fractions f = 0.05, 0.10, 0.20, 0.40 of R (at least 10 cells, all of R if fewer); 1,000 resamples per
  meeting, column and fraction; seed 0.
- Reported per workload, column, estimator and fraction: calls as a share of the changed cells (the denominator of
  the replay's call shares; hash = 0.742 on W1), median interval half-width (points), and empirical coverage of the
  true Q (pooled over meetings and resamples).

**Rules fixed before results.**
- An estimator "certifies the answer at 95%" at a fraction only if its pooled coverage is at least 0.93 for every
  workload and column; otherwise the text reports the coverage as a failure, and no saving is claimed for it.
- The proxy "helps" only if the difference estimator's median half-width is at most 0.8 times the sample estimator's
  at the same fraction, for both workloads; otherwise the text says the proxy does not help.
- The headline is the smallest fraction at which a certifying estimator's median half-width is at most 2 points; its
  call share is compared with hashing (0.742) and full recomputation (1.0). If no fraction reaches 2 points, the text
  says so.
- The guarantee is per query answer (a share with a confidence interval), not per cell; the text states this
  difference from the per-cell contract of Sec. IX.
- Output: `reports/explore_aggcert.json`, script `echobase/explore_aggcert.py`, added to `scripts/reproduce_paper.sh`.
- **Result, 2026-11-06 14:30 KST** (`reports/explore_aggcert.json`, 85 s on one CPU): by the rules above, **neither
  estimator certifies** a per-meeting answer: pooled coverage of the nominal 95% interval is 0.62-0.93 across workloads,
  columns and fractions (below 0.93 for at least one workload and column at every fraction), because a meeting has only
  a few hundred unknown cells with skewed labels. The proxy **helps** (difference half-width 0.45-0.77 of the sample
  estimator's at every fraction, both workloads). No certifying fraction reaches 2 points, so no saving over hashing is
  claimed. W1 at f = 0.40 (0.30 of the calls): median half-width 7.9 points with the proxy, 11.0 without. The paper
  adds one paragraph to Sec. IX and a row to Table III.
- **Reproduction note (D11 a), 2026-11-06 16:10 KST:** the clean-copy reproduction of release v1.1 on a Windows PC
  failed at `explore_scalability` (`import resource` is Unix-only); the peak-memory helper now falls back to the
  Windows peak working set (no effect on cells, calls or shares). Rerun there: cells, calls and shares identical; wall
  times differ by machine as expected (mapping slope 1.03 on that PC against 0.83 on the i9-7900X; policy slopes
  0.67-0.96; overhead at most 3.7%), all within the pre-fixed rule for "linear" (slope <= 1.10). The paper reports the
  i9-7900X run and names its hardware.

## D13: Editorial rewrite after the sixth review (2026-11-07, no new runs)
Text only; no data, code under the eval lock, or analysis changed. Abstract, findings and conclusion now lead with
absolute changes across all six scorers (new macros in `paper/gen_numbers.py` are minima and maxima of existing
per-scorer macros: chunker 5.5-16.0 points per article, placebo 1.0-6.0, W1 0.5-4.4, every median relative drift
>= 0.10, chunker sign from -3.4 (14B) to +14.4 (0.5B)); W1 is presented as a stress test that is small for the 3B-14B
scorers; the protocol narrative is shortened (the freeze-check defect is still stated); the per-scorer number wall
is replaced by statements read from Table IV; one sentence written during the rewrite ("the scorers that agree best
with human labels drift no less on the chunker") was found false against the table before the build (7B and 14B
drift least) and corrected. Related work adds eight references, each checked against Crossref or its arXiv page:
UDA-Bench, Murin 2026 (sensitivity of LLM extraction to prompt, model, schema), BlazeIt, ABae, prediction-powered
inference, Decibel, OrpheusDB, ModelHub. A second novelty search (12 fixed queries, 2025-2026) found no work that
measures answer drift under re-segmentation with fixed words; SemReuse and sIVM are unchanged (v1).

## D14: Non-LLM reference scorers and a series-level check (2026-11-07 15:00 KST, before any code or result)
**Why.** A seventh independent review (an AI assistant acting as an ICDE referee) noted that word-weighted shares of
"any part" predicates change with segmentation even under a perfect scorer (a unit-of-aggregation effect), so the
measured drift mixes a part caused by the query definition with a part caused by the LLM; and that the 16 held-out
meetings come from 4 series of 4 sessions. Post-hoc, exploratory, CPU only, from cached labels and existing data; no
locked file changes.

**Design (fixed now).**
- (a) *Documents.* A deterministic date detector applied to exactly the chunks the LLMs scored (the cached `spans`
  of each chunker over the article's words): a chunk is "yes" if it contains a year 1000-2099 as a standalone
  four-digit token (optionally followed by "s" or punctuation), or an English month name (January..December; "May"
  only when adjacent to a number). Pairs: fixed -> para (W4), fixed -> fixed150, fixed -> fixed100; 200 held-out
  articles. Per article: Qd1 under each chunking, signed change in points, relative drift D with the 0.02 guard.
- (b) *Meetings.* A segment is "yes" if any of its words lies in a human dialogue act of the column's type (question
  <- elicit acts, action item <- suggestions; the word projection of `explore_gold`). Pairs: W1 (w1old -> manual),
  W2 (manual -> placebo), W3 (w3old -> w1old), E6 (manual -> merge k = 2, 4, 8); 16 held-out meetings.
- For every pair and every scorer with cached labels on it (six on W1, W2, W4; primary on W3, E6 and the window
  sizes): per unit the LLM change dL and the reference change dR (signed, points); reported are the median |dR|,
  median |dL|, the median of |dL - dR| (the part not explained by the query definition), each with a 95% BCa
  interval over units (10,000 resamples, seed 0), and the Spearman correlation of dL and dR.
- (c) *Series.* For H1 and H2 (per-meeting drifts from `reports/confirmatory.json`): the median within each of the
  four meeting series and the four leave-one-series-out medians.

**Rules fixed before results.**
- "The LLM adds drift beyond the query definition" on a pair only if the interval of the median |dL - dR| lies
  entirely above 1 point; "the definition explains most of the drift" only if the median |dR| is at least half the
  median |dL|; otherwise the text reports the numbers without either claim.
- H1 and H2 are "robust to dropping any one series" only if all four leave-one-series-out medians reach their
  thresholds (0.10 and 0.05); otherwise the text names the series whose removal breaks it.
- The reference scorers are not ground truth for the LLM columns (the date detector is a rule; dialogue acts are a
  proxy for action items); the text says so.
- Output: `reports/explore_oracle.json`, script `echobase/explore_oracle.py`, added to `scripts/reproduce_paper.sh`.
- **Result, 2026-11-07 16:00 KST** (`reports/explore_oracle.json`, 8 min on one CPU). (a) Documents: the date rule
  moves an article's dated share by a median 6.0 points under the chunker change (signed -5.0); each scorer's change
  departs from it by a median 4.0 (14B) to 20.3 (0.5B) points, every interval above 1 point (rule 1 holds for all six;
  the departure is in either direction, so the text says the LLM "departs from" the reference rather than "adds");
  Spearman of LLM and reference changes 0.06-0.48. Window sizes (primary): departure 11.6 and 10.4 points. (b) Meetings:
  the reference moves by 2.3-6.8 points under W1 and the placebo; departures 2.9-20.4 points, every interval above 1
  for all six scorers and both columns; W3: reference 0.1, departure 1.4 with interval from 0.4 (rule 1 not met);
  merges: at k = 8 the reference rises by +30.4 against the scorer's +30.9 (action items), so the merge effect is
  definitional (rule 2 holds). (c) H1 and H2 are robust to dropping any one series (smallest leave-one-series-out
  medians 0.123 and 0.190; series medians for H1 0.079-0.265). The paper adds one paragraph (Sec. VII), clauses in
  the abstract, findings, threats and lessons, and a row to Table III.

## D15: A 32B scorer on the university cluster (2026-11-08 11:00 KST, before any run)
**Why.** Every independent review names the size of the scorers (0.5B-14B) as the main remaining weakness. The
user gave access to the university SLURM cluster (A100 80 GB nodes, partition p2 of the lab account) on 2026-11-08.
Post-hoc and exploratory; the confirmatory family and every locked file are unchanged.

**Design (fixed now).**
- Model: Qwen/Qwen2.5-32B-Instruct, fp32 (bf16 failed the determinism check at 7B), the locked prompts, answer sets
  and first-token scoring through `echobase.post_score` (split path over the allocated cards, CPU offload only if
  the allocation is smaller than the fp32 weights). One addition to `post_score`: `--cols` restricts the columns
  scored (default: all, so earlier commands are unchanged).
- Gates, in order; a failed gate stops the run and is reported: (g1) environment check: the primary 1.5B scorer
  on the cluster re-labels meeting IS1009a (manual, all five columns) identically to its released cache; (g2) the
  32B `check` mode: 0 label mismatches between two batch budgets on 200 segments x 3 columns.
- Scope, in priority order: (1) Wikipedia W4 (fixed -> para), column doc_date, the 200 held-out articles;
  (2) AMI W1 and W2 (versions manual, w1old, placebo), columns action_item and question, the 16 held-out meetings.
- Hard stop: 2026-11-09 20:00 KST. A workload is reported only if all its units are scored; an incomplete one is
  reported as not completed, with its unit count, and none of its numbers enter the paper.
- One job at a time on p2, wall-time limit at most 24 h, cancelled at the hard stop; the model files are deleted
  from the shared account after the run.

**Rules fixed before results.**
- Completed workloads are reported with the same quantities as the other scorers (median relative drift with the
  0.02 guard and the preregistered unit minima, median |change| in points, median signed change, sign counts, and
  the D14 departure from the reference scorer), as a seventh scorer in Table models and in the text.
- The ranges stated over "all scorers" (abstract, findings, conclusion) then include the 32B scorer whether it
  widens or narrows them; the text says "seven scorers" for those workloads.
- If the 32B scorer's median |change| on the chunker is below the smallest of the six (5.5 points), the text says
  that drift shrinks at this size; if above, that it does not; no extrapolation beyond 32B.
- Outputs: cache `outputs/echobase/v2-Qwen2.5-32B-Instruct`, gate reports `reports/d15_env_check.json` and
  `reports/post_determinism_32b.json`, job logs under `journal/`.

### D15 amendment (2026-11-09 17:40 KST, before any 32B drift number is computed or seen)
- **Host.** The university jobs never started (all A100s busy; the administrator cancelled the pending jobs of the
  shared account on 2026-11-09 11:50). The run moved to the lab server gp146 (one A100 80 GB, shared by agreement;
  torch 2.1.2+cu118, transformers 4.44.2, accelerate 0.33.0, the same pins as the cluster plan), fp32 with up to
  64 GiB of weights on the card and the rest in RAM. Script `scripts/d15_gp146.sh`; it stops itself if the card's
  free memory falls below 3 GiB.
- **Gates, as fixed.** g1 passed on gp146 (`reports/d15_env_check.json`: 0 label mismatches, IS1009a, 5 columns x
  195 segments). g2 passed (`reports/post_determinism_32b_gp146.json`: 0 mismatches between two batch budgets,
  3 columns x 200 segments). Resumed runs skip label files already written, so every unit is scored once; the
  batch budget does not change labels (g2).
- **Hard stop extended** to 2026-11-10 12:00 KST, by the user's decision, so that scope (2) (AMI) can finish while
  the card is available. Reason: GPU access, not results; at this time only label files exist (Wikipedia 159 of 200
  articles) and no drift statistic of the 32B scorer has been computed. Every other rule of D15 is unchanged.

### D15 result (2026-11-10 08:30 KST)
- Both scopes completed on gp146 (Wikipedia 200 of 200 articles, 2026-11-09 17:55 KST; AMI 16 of 16 meetings x 3
  versions x 2 columns, 41,890 calls, 2026-11-10 01:58 KST), so both are reported. Model files deleted from gp146
  afterwards; the university jobs never ran (their node-local staging files are pending deletion by the administrator).
- Applied rules: the 32B scorer is the seventh row of Table models; the ranges over all scorers now cover seven
  (chunker |change| 5.5-16.0 and placebo 1.0-6.0 points unchanged; the signed chunker range widens to -4.1..+14.4).
  Its chunker |change| is 5.7 points, not below the smallest of the six (5.5), so the text says that drift does not
  shrink at this size.
- D14 comparison with the reference scorers: on meetings the 32B scorer departs from the reference like the others
  (every interval above 1 point); on the chunker it does not (excess 0.7 points, interval 0.0-2.3, Spearman 0.75).
  The abstract, findings and Sec. dir now say this ("on documents all but the largest"); this narrows the earlier
  claim that every scorer adds a scorer-specific part on documents.

## D16: Editorial corrections after the eighth review (2026-11-10, no new runs)
Text and generated macros only; no data, cache, or code under the eval lock changed. A fresh independent review of the
D15 changes found these, each checked against the reports before it was applied:
- The D14 excess on the chunker change falls monotonically with size in the Qwen family (20.3, 14.2, 11.7, 6.7, 4.0,
  0.7 points; Phi 15.0); calling 32B "the exception" hid this. The paper now reports the measured sequence (new macros
  `OrWfourExQwenSeq`, `OrWfourExHalf`, `OrWfourExPhi`), states that the scorer-specific part on documents shrinks with
  size and is not detectable at 32B, and claims nothing beyond 32B. On meetings it reports that the placebo
  action-item excess shows no such trend (2.9, 12.2, 7.5, 5.2, 7.6, 9.1) while the question excess does (20.4 to 3.7),
  and that 32B departs from the reference in every meeting cell. The abstract and the Threats paragraph are updated.
- "Drift does not shrink at this size" -> "does not shrink further from 7B to 32B" (5.5 vs 5.7 points); the D15 rule
  is unchanged, the old wording read as "no size effect at all", which the 0.5B-7B values contradict.
- "On meetings every LLM scorer adds a part" -> "on the main meeting workloads": on W3 the reference explains the
  0.5B and 1.5B change (`llm_adds` false).
- The 32B determinism check covered action item, entity count and topic, not question or doc date; the text now says
  so, and that its host and stop time changed before any drift was computed (D15 amendment).
- "3B-32B" -> "Qwen 3B-32B" (Phi's W1 change is 2.8 points and its base share 15.8%); `SixDMin` printed with three
  decimals (0.096, the 32B W4 value, had printed as 0.10); post-hoc round count eight -> nine; "three of the six" ->
  "seven"; "five model swaps" -> "the five model swaps measured (0.5B-14B, Phi)"; "larger scorers do not remove the
  effect" scoped to "up to 32B".
- Merge claim: the k=8 reference and scorer agree in the median (30.4 vs 30.9 points) but differ per meeting by a
  median 13.2 points (8.2-29.7); the text now says "median" and gives the per-meeting difference (new macros
  `OrKeightEx*`).
Build: 13 pages, body ends on page 12, no undefined references.

## D17: Addressing the stated limitations after submission (2026-11-11 07:00 KST, before any code or result)
Post-hoc and exploratory; for the rebuttal (Jan 8-15) and any later version. The submitted paper (55a2671) is frozen.
No locked file changes; the confirmatory analysis is not rerun. Three parts, each with rules fixed here.

**D17-A: a high-validity, non-LLM learned scorer (limitation: weak labels; CPU only).**
- Training data: AMI meetings with dialogue-act annotation, excluding the 18 dev and 16 eval meetings and every
  meeting of their series (ES2011, IB4*, IS1008, TS3004, EN2002, ES2004, IS1009, TS3003). Units: AMI manual segments
  over reference words. Label: a segment is "yes" if any of its words lies in a dialogue act of the column's types
  (D14 reference definition: question <- el.inf, el.ass; action item <- sug).
- Model: TF-IDF of word 1-2 grams (lowercase, min_df 2, sublinear tf) + logistic regression (liblinear, balanced
  class weights). C in {0.1, 1, 10} and the probability threshold in {0.10, 0.15, ..., 0.90} are chosen on the 18 AMI
  dev meetings (manual segments) by word-level F1; then applied once to the 16 eval meetings.
- Validity: word-level F1 against the dialogue acts on eval manual segments (explore_gold metric). It counts as
  "valid" for a column if F1 >= 0.50 (best LLM question F1: 0.30). Reported either way.
- Drift: on the 16 eval meetings, the same statistics as Table models and D14 (relative D, |change|, signed change,
  excess over the D14 reference) for W1, W2, W3 and merges k = 2, 4, 8, both columns.
- Rules: (i) "a valid scorer still drifts" if, for a valid column, the W2 |change| interval lies above 1 point;
  (ii) "it adds a scorer-specific part" by the D14 rule (excess interval above 1 point). Since it is trained toward
  the reference definition, a small excess is expected and will be stated as such.

**D17-B: rerun noise under reduced-precision, batch-varying inference (limitation: determinism needs fp32).**
- Primary scorer (1.5B) in bf16 on a TITAN Xp of gpulab2 (the only idle cards; Pascal runs bf16 through emulated
  kernels, stated as a caveat), two batch budgets (3072 and 1024 tokens: different batch composition and padding).
  AMI eval: manual, placebo, w1old x action item, question; Wikipedia test: fixed, para x doc date.
- Per unit: rerun change = |s(manual or fixed, 3072) - s(same, 1024)|; segmentation change = |s(placebo or w1old or
  para, 3072) - s(manual or fixed, 3072)|; plus the label flip rate between budgets.
- Rule: segmentation drift exceeds rerun noise if the per-unit paired difference (segmentation change - rerun
  change) has a median interval (same BCa routine) above 0. Otherwise the text will say that, at bf16, this
  workload's drift is not separable from rerun noise. bf16 vs fp32 segmentation drift is reported descriptively.

**D17-C: a second corpus per domain (limitation: one corpus per domain).**
- Meetings: ICSI Meeting Corpus core NXT annotations (CC BY 4.0). 20 of the 75 meetings, drawn with Python
  random.Random(crc32("icsi:d17")).sample(sorted(meetings), 20). Manual segments = ICSI transcription segments;
  words with times only (no punctuation tokens). Workloads: W2 placebo (same algorithm, seed crc32("placebo:<m>"))
  and merges k = 2, 4, 8. No W1 (no diarization of ICSI audio). Reference (D14 style): question <- MRDA tags qy, qw,
  qr, qrr, qo, qh (any component of the label); action item <- cs.
- Documents: GovReport CRS test split (US government works, public domain; launch/gov_report, 362 reports): 200
  drawn with random.Random(crc32("crs:d17")).sample(sorted(ids), 200); text = paragraphs in order (section titles
  dropped). W4 as for Wikipedia (fixed 120-word windows vs paragraphs merged forward to >= 40 words), doc date column,
  rule-based date reference.
- Scorer: the primary 1.5B in fp32 (frozen prompts, unchanged), determinism spot check 0 mismatches between two batch
  budgets on 200 segments before use.
- Rules: the H2 and H4 decision rules (median D >= 0.05 with CI lower >= 0.02; median D >= 0.10 with CI lower
  >= 0.05), applied as a post-hoc replication, with unit minima of 18 of 20 meetings and 180 of 200 reports passing
  the 0.02 guard; plus |change|, signed change and the D14 excess. A failed rule is reported as a failed replication.
- D17-B amendment (2026-11-11 07:40 KST, before any label): bf16 attention failed to launch on Pascal ("cutlassF: no
  kernel found"); the run uses PyTorch's math attention backend (memory-efficient and flash attention disabled via
  torch.backends), unchanged code otherwise. No label had been written.
- D17-A addendum (2026-11-11 08:30 KST, after the dev F1 of 0.33 and the eval F1 of 0.35 were seen; no rule changed):
  word-level F1 credits a "yes" segment with all its words, while the dialogue act covers only some, so even a labeler
  that is exactly right about every segment cannot reach F1 = 1. We therefore also compute, on the eval meetings and
  per version, (a) the word-level F1 of the reference labeling itself (a segment is "yes" iff any of its words is in
  an act of the column's type), the ceiling for any segment-level scorer, and (b) segment-level precision, recall and
  F1 of every scorer against that reference labeling. Descriptive only; the >= 0.50 validity rule of D17-A is
  reported as written.

## D18: Segmentation-invariant scoring by canonical units (2026-11-11 09:10 KST, before any code or label)
Post-hoc, exploratory; addresses the limitation that the paper offers no constructive data-engineering remedy (its
conclusion names "scoring operators whose answers are stable under re-segmentation by construction" as future work).
- Operator: score fixed canonical units derived from the words alone, never from the segmenter: per speaker, the
  reference words in time order, cut wherever the gap to the speaker's previous word is >= 0.5 s. (Dev only, no
  labels: 91% of gaps are 0 and ~9% exceed 0.3 s, with nearly the same split at 0.3, 0.5 or 1.0 s, so 0.5 s is fixed
  here and not tuned.) A segmentation version's view is the word-weighted share over canonical units. Because the
  units do not depend on segment boundaries, W1 (on reference words), W2 and the merges give identical answers by
  construction, and one scoring pass serves every version (all labels reusable by content hash). Caveat stated in the
  text: invariance covers boundary moves, not speaker-attribution errors of a real diarizer.
- Run: primary scorer (1.5B, fp32, frozen prompts), action item and question, 16 eval meetings, gpulab2.
- Measured: (a) drift across versions (zero by construction; verified by computation); (b) validity: word-level F1
  against the dialogue acts (explore_gold metric), compared with the same scorer on manual segments and with the
  word-F1 ceiling of the canonical units (D17-A addendum method); (c) cost: calls and prompt tokens of one canonical
  pass against scoring each version separately.
- Rules: the operator "keeps validity" for a column if its word-level F1 is at least 0.9 x the manual-segment F1 of
  the same scorer (pooled over the 16 meetings); otherwise the text reports the validity cost of invariance. No
  threshold is tuned on eval.

## Availability paragraph (2026-11-11 15:00 KST, text only, before the submission deadline)
At the user's request the paper names its artifacts explicitly: the abstract ends with the repository URL, and a new
"Availability" paragraph after the contributions gives the code (MIT), the Zenodo concept DOI, the AMI and Wikipedia
sources with licenses and URLs, and the model sources. No number, analysis or claim changed; 13 pages, body ends on
page 12. For re-upload in place of the 55a2671 PDF before the deadline.

## D19: A second scorer for D17-C and D18 (2026-11-11 18:05 KST, before any code change or label)
Post-hoc, exploratory; for a possible revised upload before the deadline (Nov 12 10:00 KST), the rebuttal or a later
version. Limitation addressed: the D17-C replications and the D18 operator rest on one scorer (the 1.5B).
- Scorer: Qwen2.5-3B-Instruct, fp32, frozen prompts, split over both TITAN Xp cards of gpulab2 (as in D5). Before
  use, the D17-C determinism spot check (two batch budgets, 600 labels) must show 0 mismatches; otherwise no label is
  used and the text says so.
- Scope, in this order: D17-C ICSI (20 meetings; manual, placebo, merges 2/4/8; action item, question), D17-C CRS
  (200 reports; fixed, para; doc date), D18 canonical units (16 AMI eval meetings; action item, question). Same
  meeting and report samples, same code paths; only the model id and the cache directory change (suffix "-3b").
- Rules: exactly those of D17-C (H2 rule on ICSI with >= 18/20 meetings valid, H4 rule on CRS with >= 180/200
  reports valid; |change|, signed change, D14 excess) and of D18 (canonical F1 >= 0.9 x the same scorer's
  manual-segment F1; the manual-segment labels are the 3B's D5 cache v2-Qwen2.5-3B-Instruct). A failed rule is
  reported as a failed replication. If the 3B runs end before 01:00 KST, the same scope runs with Qwen2.5-0.5B
  (one card each for ICSI and CRS); otherwise not.

## Correction: words outside diarizer turns (2026-11-11 19:00 KST, found while writing up D18)
The submitted text (55a2671, c638790) says that both versions of a meeting "contain exactly the same words in the same
order". That holds for the placebo, the merges and the chunkers, but not for the diarizer versions: pilots/lineage.py
`_assign` puts a reference word into the turn that contains its midpoint, and a word whose midpoint lies in no turn is
left out. On the 16 held-out meetings, W1's diarizer version lacks 876 of 88,826 manual-version words (1.0%), W3's
880 (reports/d18_wordsets.json, echobase/d18_wordsets.py). No code or number changes; the revised text states this.
The main conclusions are unaffected: the placebo, which keeps every word, drifts more than W1 for all seven scorers.
It also explains why canonical units (D18) leave a small residual drift under W1 and W3 but none under boundary-only
changes.
- D19 result, ICSI and CRS (2026-11-11 20:45 KST): Qwen2.5-3B determinism 0/600. CRS chunker D 0.232 [0.199, 0.262],
  192/200 valid: H4 rule met. ICSI placebo, action items: only 3 of 20 meetings pass the 0.02 guard (the 3B labels
  almost no ICSI speech as action items), below the minimum of 18, so the H2 rule is not met; reported in the text as
  a failed replication for lack of measurable units. ICSI placebo, questions (descriptive): D 0.285 [0.203, 0.395],
  20/20 valid. Report: reports/d19_corpora_3b.json (analysis run on the PC from the copied caches; sha256 of the
  archive checked).
- D19 result, canonical units (2026-11-11 20:55 KST): Qwen2.5-3B, 16 eval meetings. D18 rule met for both columns
  (word F1 0.114 vs 0.109 on manual segments for action items; 0.203 vs 0.198 for questions). Placebo and merges:
  D exactly 0 (max 0). W1 residue: action items D 0.010 (12 meetings pass the guard), questions D 0.112 (16).
  Analysis run on gpulab2 (reports/d19_canonical_3b.json); caches copied with sha256 check. The 3B runs ended at
  20:55 KST, before the 01:00 cutoff, so the logged 0.5B extension (same scope; ICSI and CRS on one card each, then
  canonical units) runs next.

### D19 result: Qwen2.5-0.5B extension (logged 2026-11-12 06:45 KST)

Run per the D19 rule (3B runs ended 20:55, before 01:00): unit d19h on gpulab2, 20:49-21:07 KST, all parts EXIT 0.
Determinism 0/600 (`reports/d19_half_determinism.json`).
- D17-C (`reports/d19_corpora_half.json`): ICSI placebo action items D 0.197 [0.132, 0.224], 20/20 valid: H2 rule met.
  ICSI placebo questions D 0.405. CRS chunker D 0.274 [0.233, 0.304], 200/200 valid: H4 rule met.
  Across the three scorers: primary (1.5B) both met; 3B chunker met, placebo not met (3/20 measurable); 0.5B both met.
- D18 canonical (`reports/d19_canonical_half.json`): F1 0.169 vs 0.170 (action items), 0.059 vs 0.060 (questions):
  rule met for both columns. Placebo and merges k=2,4,8: D exactly 0. W1 residue 0.031 (action items), 0.039 (questions).
Paper: one clause in the findings, one sentence in "Second corpora", one clause in Sec. canon results; conclusion's
limitations sentence shortened to keep the body within 12 pages.

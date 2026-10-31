# program_pilotA.md: Pilot A (EchoBase), confirmatory. Frozen at G1; the loop never edits it.

## Question

Is remapping cell identity under speech re-segmentation genuinely needed? The verdict is taken on the **structural** condition.

The test: on real re-segmentation changes, does the **strong baseline** either reuse cells wrongly or recompute far more than needed? The strong baseline has two parts:

- **Anchoring:** each new segment takes the cell of the most-overlapping old segment of the same speaker, or of any speaker if none overlaps.
- **Content-hash invalidation:** the cell is reused iff the anchor's text is identical, even when boundaries moved, because the extraction reads only text. Otherwise it is recomputed.

## Data (fixed)

- **Corpus:** AMI Mix-Headset, the **18 dev meetings** in `configs/ami_meetings.txt`.
- **Version 1** (the "after" state, i.e. the human revision): AMI manual segments with their reference words (annotations v1.6.2).
- **Version 0** (the "before" state): system segmentation. These are pyannote/speaker-diarization-3.1 turns with the known speaker count:
  - same-speaker turns less than 0.5 s apart are merged;
  - speakers are mapped to AMI letters by maximum total overlap.
- **Conditions** (decided 2026-10-27 after the smoke test, before any confirmatory run):
  - **structural (primary, decides the verdict):** version 0 is filled with the **reference** words. Only segmentation and speaker attribution differ from version 1.
  - **asr (secondary, context only):** version 0 is filled with Whisper large-v3 words, so segmentation and transcript both differ. Its text changes are the per-cell reuse problem that sIVM covers.
- **LLM column:** "Does this meeting utterance assign or commit to an action item?"
  - Model: Qwen2.5-1.5B-Instruct, fp32. The label is yes iff logit(" yes") > logit(" no") in one forward pass.
  - The model was chosen by `pilots/bench_extract.py` before any Pilot A result: 22 segments/s, fp32 and fp16 labels agree on 99.6%, and the 3B model answered yes on only 2/256 segments.
- **Change mapping:** segments link only within the same speaker (≥ 20% of the shorter segment). Relabels are a separate pass requiring ≥ 80% mutual overlap.

## Measurement (locked)

- `pilots/segmap.py`: classifies each group of segments as unchanged, edited, split, merge, n:m, relabel, new or deleted.
- `pilots/metrics.py`: computes two rates against a full recompute on version 1:
  - **wrong-reuse rate:** among the cells the baseline reuses, the share whose recomputed value differs;
  - **over-recompute ratio:** the cells the baseline recomputes, divided by the cells whose value actually changed.
- **Structural subset:** the new segments in split, merge and n:m groups.
- **Self-disagreement floor:** the version-0 segments labelled a second time in a different order (exposes GPU-numeric noise).
- **Uncertainty** (`pilots/stats.py`): a 95% bootstrap CI that resamples **meetings** with replacement, 2,000 resamples with seed 0. Each resample pools the counts again.

## Runs

The matrix jobs `pilotA/structural/s0/run` and `pilotA/asr/s0/run` run once each. Extraction is deterministic, so seeds would only repeat the same numbers.

## Verdict (structural condition, the structural subset)

Let `share` = the split + merge + n:m share of changed new segments, `W` = the wrong-reuse rate and `R` = the over-recompute ratio. Let `lo` and `hi` be their 95% bootstrap bounds.

- **PASS:**
  - `share` ≥ 0.15, **and**
  - either `lo(W)` > max(3 × floor, 0.05) **or** `lo(R)` ≥ 2.
  - The 5% absolute minimum was set before any result, because a deterministic extraction can make the floor close to 0.
- **FAIL:**
  - `share` < 0.15, **or**
  - both `hi(W)` ≤ max(1.5 × floor, 0.05) **and** `hi(R)` < 1.3.
- **INCONCLUSIVE:** anything else. It is reported as such; a human decides at G2.
- **Reading a PASS on R alone:** it shows that the baseline wastes recomputation on structural changes. It does **not** yet show that EchoBase can avoid that waste safely. The verdict report must say so.

## Anomalies (stop and report to a human)

- `W` = 1.0, or `R` < 1.0, or any negative count.
- Fewer than 18 meetings in the run.
- The eval lock check fails.

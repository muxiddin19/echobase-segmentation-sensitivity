# program.md: ICDE 2027 go/no-go pilots (CONFIRMATORY)

Written by the human (drafted by Claude 2026-10-27, approved at G1). The loop executes and records it; it never edits it.

## Goal

Decide which finalist becomes the main ICDE topic. Two pilots each give a PASS or FAIL against thresholds fixed before any result:

- **Pilot A: EchoBase.** `program_pilotA.md` sets the question, data, measurement and thresholds.
- **Pilot B: shared-transcript lineage.** `program_pilotB.md` sets the same.

**Decision rule** (from `topic_selection/05_final_claude.md`):

| Pilot A | Pilot B | decision |
|---|---|---|
| PASS | any | **A** |
| FAIL | PASS | **B** |
| FAIL | FAIL | neither; keep SpeechQBench and re-scope |

The loop **reports** the verdicts. A human makes the decision.

## Mode

Confirmatory. The job matrix `configs/run_matrix.toml` is frozen at G1:

- no keep/revert;
- no tuning on results;
- no new arms, seeds, conditions, prompts, models or thresholds after G1.

Any change after G1 is a **deviation**. It is logged in `reports/deviations.md` (date, what, why, which results it touches) and needs the human's approval.

## Data

- **Corpus:** AMI meeting corpus, Mix-Headset audio.
- **Split:** `configs/ami_meetings.txt`. The **dev** meetings (18) are used for the verdicts. The **eval** meetings (16) are run once, after both verdicts, by a human decision.
- **Fixed upstream outputs:**
  - Whisper large-v3 (faster-whisper int8, `condition_on_previous_text=False`, temperature 0);
  - pyannote speaker-diarization-3.1 with the known speaker count.
- **LLM column:** Qwen2.5-1.5B-Instruct in fp32, scored with yes/no logits (`pilots/extract.py`).

## Allowed edits (the loop)

- Job plumbing only: `scripts/run_job.py` and `scripts/*.sh`.
- `STATUS.md`.
- Appending to `reports/`.
- Plumbing fixes that do not change any locked file. Each fix is logged as a journal note.

## Forbidden edits

- Every file in `tools/eval_lock.py` LOCKED:
  - the measurement, statistics, extraction, version building and ground-truth parsing;
  - the tests, the split and the matrix;
  - the program files.
- Writing the eval lock. **Only a human runs `python tools/eval_lock.py --write`.**
- Anything on hosts other than `gpulab2`.
- Any process of another user. Never start a job on a card where another user's process is running.

## One cycle (desk patrol `.claude/loop.md`, or the queue service)

1. `python tools/matrix.py reconcile`, then `python tools/eval_lock.py --check`, then `python tools/matrix.py budget`.
2. Check both cards on gpulab2: the owner of every process.
3. `python tools/matrix.py launch` (the queue does this when a card is completely empty).
4. Report one line per job. Numbers come only from `runs/*/status.json` and `outputs/`, through scripts.

## Stop conditions

Stop and set line 1 of `STATUS.md` to `STOP` or `DONE`:

- **DONE:** every job in the matrix is `ok`.
- **STOP:** the eval lock check fails.
- **STOP:** GPU-h used ≥ the cap of **24**. At the plan of **12**, the loop launches nothing new and asks the human.
- **STOP:** the same error in 3 cycles in a row, or a job fails twice (`blocked`).
- **STOP, a human is needed (anomaly):**
  - any result that is too good or impossible;
  - Pilot B with a violation rate of 0 together with a label saving above 50%;
  - Pilot A with a wrong-reuse rate of 1.0 or an over-recompute ratio below 1.0;
  - a negative count;
  - a pilot that runs on fewer than 18 dev meetings.

## Human gates

| gate | when | what the human does |
|---|---|---|
| **G1: freeze** | before any confirmatory job | approves `program.md`, `program_pilotA.md`, `program_pilotB.md`, `configs/run_matrix.toml` and the LOCKED list; then runs `python tools/eval_lock.py --write` and commits `.claude/eval.lock.json` |
| **G2: verdicts** | **Thu 2026-10-29** | reads both verdicts in `reports/verdicts.md` and decides A, B or neither (with the professor) |
| **G3: eval meetings** | after G2 | decides whether to run the 16 eval meetings once, for the chosen topic only |

## Budget

GPU-h plan **12**, hard cap **24** (`configs/run_matrix.toml`). The ledger is `reports/gpu_ledger.csv`.

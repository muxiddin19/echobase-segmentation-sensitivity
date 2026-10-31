# program_echobase.md: EchoBase confirmatory phase (ICDE 2027 EAB)

Drafted by Claude on 2026-10-28 and approved by the human at G1. The loop executes and records it; it never edits it.

## Goal

Run the confirmatory analysis in `PREREGISTRATION.md` exactly once:
- H1 → H2 → H4, on AMI eval (16 meetings) and Wikipedia test (200 articles);
- then the exploratory analyses (§8 there).

`echobase/confirm.py` reports the verdicts. The human decides what the paper claims.

## Mode

Confirmatory:
- no keep/revert;
- no tuning on eval or test results;
- no new arms, queries, models, prompts, thresholds or units after G1.

Any change to a locked file or to the protocol is a deviation. It is logged in `reports/deviations.md` and approved
by the human before it takes effect.

## Jobs (PREREGISTRATION §7; gpulab2 only)

| step | command (`python -m …`) | card | expected cost |
|---|---|---|---|
| E1a | `pilots.run_diar --split eval --gpu 0` | 0 | ≈ 1.3 GPU-h (dev: 2.5–7 min per meeting) |
| E1b | `echobase.run_diar_v --pipeline 3.1 --estimate-speakers --split eval --gpu 1` | 1 | ≈ 1.3 GPU-h |
| E2 | `echobase.score_versions --split eval --version manual`, then `w1old`, `placebo`, `w3old` | free card | ≈ 2.4 GPU-h |
| E3 | `echobase.score_wiki --split test` | free card | ≈ 0.5 GPU-h |
| check | `tools/eval_lock.py --check`; a 0-byte scan of `outputs/`; `echobase.e0_determinism --split eval --n 100 --out reports/e0_determinism_eval.json` | free card | minutes |
| confirm | `echobase.confirm` (CPU) | — | minutes |
| E4 | E2 and E3 again with `--model Qwen/Qwen2.5-0.5B-Instruct` (exploratory) | free card | ≈ 2 GPU-h |

## Rules for running jobs

- Each job runs as a `systemd-run --user` unit and only on a card with no process at all.
  - Never use a card where another user's process is running.
  - Never kill another user's process. Kill our own jobs by PID.
- Never run pyannote 3.0 on gpulab2: it hard-reset the host twice on 2026-10-28.
- After any reboot (`last -x`), scan for 0-byte files before using the outputs.
  - Delete a 0-byte cache of ours and rerun it (that counts as the retry).
  - Never delete anything else.
- Code reaches the server through a staging directory with sha256 checks. Never run `tar` straight into the repo.
- Append each job to `reports/run_journal.tsv` and `reports/gpu_ledger.csv`.

## Allowed edits

- `STATUS.md`.
- Appending to `reports/`.
- Job plumbing in `scripts/`.
- New exploratory scripts that do not modify locked files.

## Forbidden edits

- Every file in `tools/eval_lock.py` LOCKED.
- Writing the eval lock: only the human runs `--write`.
- Any host other than gpulab2.
- Paid APIs.
- `git push`.

## Stop conditions (stop, write STATUS line 1 = STOP, report to the human)

- the lock check fails;
- PREREGISTRATION §6 anomalies;
- the GPU-h ledger passes 40 (plan); 80 is the hard cap;
- a host crash during an eval job;
- any result that looks too good. Examples: a CI width of 0; identical D in every unit; all three hypotheses with
  D > 3× the dev median.

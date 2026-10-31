# Author checks before submission (for the AI-generated content declaration)

The ICDE 2027 CFP requires AI-generated content to be disclosed in the acknowledgments. Reviewers asked that the
declaration name concrete checks the authors performed themselves. Only list a check in the paper after you have
actually done it. About 1-1.5 hours in total.

| # | Check | How | Time | Done by / date | Result |
|---|---|---|---|---|---|
| 1 | Proposition 1 proof (`prop:trivial`, Sec. II) | Re-derive both directions on paper | 10 min | | |
| 2 | Proposition 2 proof (`prop:mono`, Sec. VIII) | Re-derive; check the counterexample direction | 10 min | | |
| 3 | Ten numbers in the text against the result files | Pick 10 numbers in the PDF at random; find each in `reports/*.json` (the macro name in `paper/numbers.tex` tells which file) | 20 min | | |
| 4 | Level-1 reproduction | `bash scripts/reproduce_paper.sh` on gp231 from a fresh checkout and the NAS caches; `git diff reports/` must be empty | 15 min (runs ~10 min) | | |
| 5 | Workload construction | Open one eval meeting's manual and w1old versions (`echobase/score_versions.py build`); confirm the word sequences are identical | 10 min | | |
| 6 | Labels by hand | Read 30 segments with their action-item / question labels from `outputs/echobase/v2/manual`; note how many you would label differently | 20 min | | |
| 7 | Citations | Open each of the ~25 references and confirm the cited claim | 20 min | | |

When done, the declaration's last sentences can say, for example: "The authors re-derived both proofs, checked ten
randomly chosen numbers against the released result files, reran the Level-1 reproduction, and verified every cited
source." Use only the checks actually done.

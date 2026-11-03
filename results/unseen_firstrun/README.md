# First run of the pre-registered evaluation on new proteins (superseded)

The pre-registered ESM-1b evaluation (`docs/preregistration_unseen.md`) was first run on
2026-11-02 at commit 14e961c (raw outputs: `results/raw/unseen_firstrun/`, not committed). While
it was running, an adversarial review of the proofs found that the solver behind ProtCOVER's
thresholds could, at near-ties, return a threshold below the supremum of program (3). The fix
(commit 23bf9b0, pushed before these files were opened) certifies every threshold in exact
arithmetic and makes the weighted quantile exact at ties. As the pre-registration requires, the
evaluation was rerun with the fixed code (`results/unseen/`), and both runs are reported in
`docs/unseen_results.md`. The files here are the outputs of `scripts/summarize.py` and
`scripts/score_unseen.py` for the first run, unchanged (`data.csv` is the file the first run
wrote as `results/unseen_data.csv`).

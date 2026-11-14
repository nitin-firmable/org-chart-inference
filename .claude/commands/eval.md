---
description: Regenerate data, run inference and evaluation, compare with the baseline
---
From the repo root, using the virtual environment's python, as modules:

1. App data: `python -m data.synthetic.generate`, `python -m judge.rules`, `python -m harness.evaluate`
2. Held out: `python -m data.synthetic.generate --seed 7 --out data/synthetic/output/seed7`,
   `python -m judge.rules data/synthetic/output/seed7`,
   `python -m harness.evaluate data/synthetic/output/seed7`
3. If the change touches features or rules, retrain first: `python -m judge.train_confidence`

Report a short table: precision, coverage, wrong edges and Brier score for both datasets,
next to the baseline in CLAUDE.md. Call out any scenario or level whose precision dropped,
say in one line whether the change is worth keeping, and log it in docs/experiments.md.
$ARGUMENTS

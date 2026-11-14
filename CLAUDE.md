# Org chart inference — project notes for Claude Code

Take-home task for Firmable (B2B sales intelligence). Deadline: Sat 7 Nov 2026.

## The brief (summary)
- For every company, infer who reports to whom from a flat list of people + job titles,
  with a confidence on each reporting line.
- **A wrong edge is worse than no edge.** Abstain when the manager can't be resolved.
- No dataset is given: we generate one. >= 500 companies, >= 100 employees each, realistic
  mix of titles / functions / seniority / industries / sizes, and messy data.
  "Think about how you'll know if your inference is right" -> the generator keeps an answer key.
- Firmable is AI-native: how the work is built (dev loop) matters as much as the result.
- Deliverables: hosted app (pick a company, see its org chart) + public repo with code,
  generator and a short write-up (approach, key decisions, how it was measured).

## Layout
- `data/synthetic/generate.py` -> `output/<seed>/truth.csv` (answer key) + `employees.csv` (messy)
- `data/scraped/uk_organograms.py`, `asx_leadership.py` (+ hand-checked `asx_leadership.csv`)
  -> `data/scraped/output/uk|asx/` (real data, same columns)
- `judge/rules.py` (was infer.py) -> `<data>/predictions.csv` (manager_id, confidence, reason,
  f_* features). `judge/train_confidence.py` trains the edge-confidence tree on seed 1,
  saves `models/confidence_tree.json` (used by judge/rules.py) + `.txt` (readable rules)
- Model judges, same decision format (Choice / Score / Noul, `judge/decisions.py`), each writing
  `<data>/runs/<judge>/raw.csv`: `judge/claude_judge.py` (Claude Code CLI, headless, staged:
  level -> domain -> manager), `judge/laya_judge.py` and `judge/laya_staged.py` (free local
  Laya; doesn't know seniority, kept as a baseline). Laya needs requirements-research.txt.
- `harness/run.py` -> any judge on any dataset, one table (precision, chain, coverage, wrong,
  Brier). `harness/calibrate.py` fits each model judge's confidence on the UK answer key
  (leave one organisation out); maps saved in `models/calibration/`.
- `harness/evaluate.py` -> precision / coverage / wrong edges, Brier, sweep, calibration, scenarios
- `app/app.py` -> Streamlit app (builds seed 42 itself if missing or stale); `app/org_chart.py`

Every dataset folder holds `employees.csv`, `truth.csv`, `predictions.csv`. `data/**/output/` is
git-ignored; everything is reproducible (app data seed 42, held-out seed 7, training seed 1).
`models/` IS committed (the hosted app needs it).

## Commands (from the repo root, as modules)
- Full loop: `python -m data.synthetic.generate`, `python -m judge.rules`,
  `python -m harness.evaluate` (default dataset: `data/synthetic/output/seed42`)
- Held out: `python -m data.synthetic.generate --seed 7 --out data/synthetic/output/seed7`,
  `python -m judge.rules data/synthetic/output/seed7`,
  `python -m harness.evaluate data/synthetic/output/seed7`
- Real data: `python -m data.scraped.uk_organograms` / `python -m data.scraped.asx_leadership`,
  then `python -m judge.rules data/scraped/output/uk` and the harness
- Retrain confidence: `python -m judge.train_confidence` (`python -m judge.rules --rules` = no tree)
- App: `streamlit run app/app.py` (Streamlit Cloud: main file path `app/app.py`)

## Current baseline (threshold 0.75), see docs/experiments.md
Decision tree confidence, threshold 0.8 (wrong edge = 4x a missing edge):
- v2 data (default, seed 42): precision 96.3%, coverage 58.7%, 3,465 wrong edges, Brier 0.0433.
- v2 held-out (`--seed 7 --out data/synthetic/output/seed7`): 96.3% / 60.6%, Brier 0.0420.
- Weakest: shared-manager teams (63% precision), manager missing from the data.
Any change must be reported against this baseline, on all three datasets,
and logged in docs/experiments.md.

## Rules for working in this repo
- The user is new to Git/terminals: explain each change, then give exact git commands.
- Never change code without saying what changed and why.
- Evaluate on held-out companies when anything is learned or tuned (split by company_id).
- Keep the app light: Streamlit Cloud free tier has ~1 GB RAM. No torch in requirements.txt.
- Research notes: `docs/research_models.md`.

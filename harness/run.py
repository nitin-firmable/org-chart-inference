"""
harness/run.py  -  run judges on datasets and score them side by side

    python -m harness.run --judge rules laya --data uk asx
    python -m harness.run --judge rules laya --data uk --companies UK02 UK10

Every judge is scored the same way, on the same people (with --companies, both judges are
scored on those companies only). Columns:
  precision   share of drawn lines that name the true direct manager
  chain       share of drawn lines that name the true manager or someone above them
  coverage    share of people (excluding the top of each organisation) who got a line
  wrong       lines that are wrong; skip = of those, how many point to a boss further up
  brier       how honest the confidences are (0 = perfect, lower = better)
Datasets without an answer key (ASX: 2 known managers) still get coverage and line counts.
"""
import sys
from pathlib import Path

import pandas as pd

from harness import calibrate
from judge import claude_judge, laya_judge, laya_staged

DATASETS = {"seed42": "data/synthetic/output/seed42", "seed7": "data/synthetic/output/seed7",
            "uk": "data/scraped/output/uk", "asx": "data/scraped/output/asx"}
THRESHOLD = 0.8                 # wrong line = 4x a missing line, as in judge/rules.py
CALIBRATION_DATA = "uk"         # the only real answer key: every calibration map is fitted here


def rules(data_dir, companies):
    if not (data_dir / "predictions.csv").exists():
        from judge import rules as rules_judge
        rules_judge.main(data_dir)
    return pd.read_csv(data_dir / "predictions.csv")


def model_judge(name, module):
    """A decision-model judge: run it for missing companies, calibrate, apply the threshold."""
    def run(data_dir, companies):
        employees = pd.read_csv(data_dir / "employees.csv")
        out = data_dir / "runs" / name
        wanted = set(companies or employees.company_id)
        have = set(pd.read_csv(out / "raw.csv").employee_id) if (out / "raw.csv").exists() \
            else set()
        missing = employees[employees.company_id.isin(wanted) & ~employees.employee_id.isin(have)]
        if len(missing):
            module.main(data_dir, sorted(missing.company_id.unique()))
        raw = pd.read_csv(out / "raw.csv")
        pred = employees.merge(raw, on="employee_id")

        if data_dir == Path(DATASETS[CALIBRATION_DATA]):
            truth = pd.read_csv(data_dir / "truth.csv")[["employee_id", "true_manager_id"]]
            pred = pred.merge(truth, on="employee_id", how="left")
            correct = (pred.choice == pred.true_manager_id).astype(int)
            conf = calibrate.leave_one_company_out(pred.raw_confidence, correct, pred.company_id)
            calibrate.save(name, calibrate.fit(pred.raw_confidence, correct))  # for other data
            if "inside" in pred:
                candidate_recall(pred)
            pred = pred.drop(columns="true_manager_id")
        else:
            mapping = calibrate.load(name)
            if mapping is None:
                sys.exit(f"No {name} calibration yet: run the harness on --data uk first.")
            conf = calibrate.apply(mapping, pred.raw_confidence)
        pred = laya_judge.to_predictions(employees[employees.employee_id.isin(pred.employee_id)],
                                         raw, conf)
        pred["manager_id"] = pred.pred_manager_id.where(pred.confidence >= THRESHOLD)
        pred.loc[pred.pred_manager_id.notna() & pred.manager_id.isna(), "reason"] = \
            f"{name}: confidence below threshold"
        pred.to_csv(out / "predictions.csv", index=False)
        return pred
    return run


def candidate_recall(pred):
    """Staged judges: did the true manager survive the level and domain filters?"""
    known = pred[pred.true_manager_id.notna()]
    def has(col):
        return pd.Series([isinstance(c, str) and m in c.split(";")
                          for c, m in zip(known[col], known.true_manager_id)], index=known.index)
    inside, outside = has("inside"), has("outside")
    print(f"  true manager among same-domain candidates: {inside.mean():.1%}, "
          f"among all candidates (domain + fallback): {(inside | outside).mean():.1%}, "
          f"of {len(known)} people")


JUDGES = {"rules": rules, "laya": model_judge("laya", laya_judge),
          "laya_staged": model_judge("laya_staged", laya_staged),
          "claude": model_judge("claude", claude_judge)}


def score(truth, pred):
    boss_of = dict(zip(truth.employee_id, truth.true_manager_id))

    def above(person, candidate):           # is candidate somewhere above person?
        seen, cur = set(), boss_of.get(person)
        while isinstance(cur, str) and cur not in seen:
            if cur == candidate:
                return True
            seen.add(cur)
            cur = boss_of.get(cur)
        return False

    t = truth[truth.in_dataset.astype(bool) & truth.true_manager_id.notna()]
    df = t[["employee_id", "true_manager_id"]].merge(
        pred[["employee_id", "pred_manager_id", "manager_id", "confidence"]],
        on="employee_id", how="left")
    given = df.manager_id.notna()
    right = given & (df.manager_id == df.true_manager_id)
    up = pd.Series([above(p, m) for p, m in zip(df.employee_id, df.manager_id)], index=df.index)
    chain = right | (given & up)
    proposed = df[df.pred_manager_id.notna()]
    brier = ((proposed.confidence - (proposed.pred_manager_id == proposed.true_manager_id))
             ** 2).mean()
    return dict(people=len(df), lines=int(given.sum()),
                precision=right.sum() / max(given.sum(), 1),
                chain=chain.sum() / max(given.sum(), 1), coverage=given.mean(),
                wrong=int(given.sum() - right.sum()), skip=int(chain.sum() - right.sum()),
                brier=brier)


def main(judges, datasets, companies=None):
    rows = []
    for name in datasets:
        data_dir = Path(DATASETS[name])
        truth = pd.read_csv(data_dir / "truth.csv")
        if companies:
            truth = truth[truth.company_id.isin(companies)]
        for judge in judges:
            pred = JUDGES[judge](data_dir, companies)
            pred = pred[pred.employee_id.isin(truth.employee_id)]
            drawn = pred.manager_id.notna()
            rows.append(dict(data=name, judge=judge, all_people=len(pred),
                             lines_drawn=int(drawn.sum()), **score(truth, pred)))
    table = pd.DataFrame(rows)
    for col in ["precision", "chain", "coverage"]:
        table[col] = (table[col] * 100).round(1)
    table["brier"] = table.brier.round(4)
    print(table.to_string(index=False))
    return table


if __name__ == "__main__":
    def opt(flag):
        if flag not in sys.argv:
            return None
        rest = sys.argv[sys.argv.index(flag) + 1:]
        return [a for a in rest[:next((i for i, a in enumerate(rest) if a.startswith("--")),
                                      len(rest))]]
    main(opt("--judge") or ["rules"], opt("--data") or ["uk"], opt("--companies"))

"""
harness/export.py  -  build app/org_chart.duckdb, the one database the app reads

The hosted app can't run Claude or Laya, and data/**/output/ isn't in git, so finished results
are shipped with the app in a single DuckDB file (tracked; public data only):

  people      one row per (dataset, judge, person): title, unit, proposed manager, confidence,
              reason, level, domain, is_top, source_url
  answer_key  one row per (dataset, person) whose real manager is known
  datasets    what each dataset is, and which judges it has

    python -m harness.run --judge rules laya laya_staged claude --data uk asx
    python -m harness.export

Synthetic data (seed 42, rules judge) is included, so the app never rebuilds it on the server.
"""
from pathlib import Path

import duckdb
import pandas as pd

from harness.run import DATASETS
from judge.decisions import TOP

DB = Path(__file__).parent.parent / "app" / "org_chart.duckdb"
JUDGE_ORDER = ["claude", "rules", "laya_staged", "laya"]
COLUMNS = ["company_id", "company_name", "industry", "employee_id", "full_name", "title",
           "department", "pred_manager_id", "confidence", "reason", "level", "domain",
           "is_top", "source_url"]


def judge_frame(data_dir, judge):
    """One judge's predictions in the app's columns, or None if it hasn't been run."""
    if judge == "rules":
        path = data_dir / "predictions.csv"
        if not path.exists():
            return None
        pred = pd.read_csv(path)
        pred["domain"] = pred.function
        pred["is_top"] = pred.level == 1
        return pred
    path = data_dir / "runs" / judge / "predictions.csv"
    if not path.exists():
        return None
    pred = pd.read_csv(path)
    pred["is_top"] = pred.choice == TOP
    if "level" not in pred:
        pred["level"] = float("nan")         # plain Laya never asks for a level
    pred["level"] = pd.to_numeric(pred.level).round()
    pred.loc[pred.is_top, "level"] = 1       # whoever heads the organisation is level 1
    if "domain" not in pred:
        pred["domain"] = None
    return pred


def main():
    people, keys, meta = [], [], []
    for name in ["synthetic", "uk", "asx"]:
        data_dir = Path(DATASETS["seed42" if name == "synthetic" else name])
        truth = pd.read_csv(data_dir / "truth.csv")
        in_data = truth[truth.in_dataset.astype(bool)] if "in_dataset" in truth else truth
        known = in_data[in_data.true_manager_id.notna()]
        keys.append(known[["company_id", "employee_id", "true_manager_id"]].assign(dataset=name))
        sources = truth[["employee_id", "source_url"]] if "source_url" in truth else None
        judges = []
        for judge in JUDGE_ORDER:
            pred = judge_frame(data_dir, judge)
            if pred is None:
                continue
            if sources is not None:
                pred = pred.drop(columns=["source_url"], errors="ignore").merge(
                    sources, on="employee_id", how="left")
            for col in COLUMNS:
                if col not in pred:
                    pred[col] = None
            people.append(pred[COLUMNS].assign(dataset=name, judge=judge))
            judges.append(judge)
        meta.append(dict(dataset=name, judges=",".join(judges), people=len(in_data),
                         known_managers=len(known)))

    DB.unlink(missing_ok=True)
    con = duckdb.connect(str(DB))
    all_people = pd.concat(people, ignore_index=True)
    no_name = all_people.full_name.isna()        # e.g. a source name of "NA", read as missing
    all_people.loc[no_name, "full_name"] = "Undisclosed (" + all_people.employee_id[no_name] + ")"
    all_people["is_top"] = all_people.is_top.fillna(False).astype(bool)
    for col in ["confidence", "level"]:
        all_people[col] = pd.to_numeric(all_people[col])
    all_keys, all_meta = pd.concat(keys, ignore_index=True), pd.DataFrame(meta)
    con.execute("CREATE TABLE people AS SELECT * FROM all_people ORDER BY dataset, judge, "
                "company_id, employee_id")
    con.execute("CREATE TABLE answer_key AS SELECT * FROM all_keys ORDER BY dataset, employee_id")
    con.execute("CREATE TABLE datasets AS SELECT * FROM all_meta")
    con.close()
    for row in all_meta.itertuples():
        print(f"{row.dataset}: {row.people:,} people, {row.known_managers:,} known managers, "
              f"judges {row.judges}")
    print(f"{DB.relative_to(DB.parent.parent)}: {DB.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()

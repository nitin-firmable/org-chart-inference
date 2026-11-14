"""
harness/export.py  -  copy finished results into app/results/ for the app

The hosted app can't run Claude or Laya, and data/**/output/ isn't in git. So after the
harness has scored a dataset, this writes one small CSV per (dataset, judge) with just what
the app shows, into app/results/<dataset>/<judge>.csv (tracked; public data only).

    python -m harness.run --judge rules laya laya_staged claude --data uk asx
    python -m harness.export

Synthetic data isn't exported: it is 158k people, and the app rebuilds it itself.
"""
from pathlib import Path

import pandas as pd

from harness.run import DATASETS
from judge.decisions import TOP

OUT = Path(__file__).parent.parent / "app" / "results"
REAL = ["uk", "asx"]
COLUMNS = ["company_id", "company_name", "industry", "employee_id", "full_name", "title",
           "department", "pred_manager_id", "confidence", "reason", "level", "domain",
           "is_top", "true_manager_id", "source_url"]


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
    for name in REAL:
        data_dir = Path(DATASETS[name])
        truth = pd.read_csv(data_dir / "truth.csv")
        keep = ["employee_id", "true_manager_id"] + \
            (["source_url"] if "source_url" in truth else [])
        for judge in ["rules", "claude", "laya", "laya_staged"]:
            pred = judge_frame(data_dir, judge)
            if pred is None:
                continue
            pred = pred.drop(columns=[c for c in keep[1:] if c in pred]).merge(
                truth[keep], on="employee_id", how="left")
            for col in COLUMNS:
                if col not in pred:
                    pred[col] = None
            (OUT / name).mkdir(parents=True, exist_ok=True)
            path = OUT / name / f"{judge}.csv"
            pred[COLUMNS].to_csv(path, index=False)
            print(f"{path.relative_to(OUT.parent.parent)}: {len(pred)} people, "
                  f"{pred.pred_manager_id.notna().sum()} proposed lines")


if __name__ == "__main__":
    main()

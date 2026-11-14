"""
judge/laya_judge.py  -  Laya (free, open decision model) as the judge

For every person, Laya answers one Choice question: who in the same company is their direct
manager (or "top of the organisation", or "not listed")? It reads the titles and units of
everyone in the company, so it judges by context, not by a fixed keyword list.

Laya's raw probabilities are not calibrated (its model card says so), so this file writes
the RAW answers only. harness/calibrate.py turns them into honest confidences, and the
harness applies the threshold.

    python -m judge.laya_judge data/scraped/output/asx        # -> <data>/runs/laya/raw.csv
    python -m judge.laya_judge data/scraped/output/uk --companies UK02 UK10

Needs requirements-research.txt (laya, torch). First run downloads the model (~1.7 GB).
"""
import os
import sys
import time
from pathlib import Path

import pandas as pd

from judge.decisions import NOT_LISTED, TOP, manager_question

os.environ.setdefault("USE_TF", "0")     # the model card: TensorFlow probing can hang loading
MODEL = "convaiinnovations/laya"
GROUP_SIZE = 16                          # model card: keep each choice under ~20 options
ROWS_PER_CALL = 64                       # question-groups per forward pass: bounds memory use


def load_agent():
    import laya
    import torch
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return laya, laya.load(MODEL, device=device)


def ask(laya, agent, state, questions):
    """Answer many questions over one state, a few per forward pass to bound memory.

    Long option lists (everyone in a 600-person department) go through a knockout of
    16-option groups, so each option keeps a fair share of the model's reading budget.
    Every group is one row of the forward pass.
    """
    answers, chunk, rows = {}, {}, 0
    for qid, q in questions.items():
        n = len(q["criteria"])
        q_rows = -(-n // GROUP_SIZE) if q["type"] == "choice" else 1
        if chunk and rows + q_rows > ROWS_PER_CALL:
            answers.update(laya.predict_tournament(agent, state, chunk,
                                                   group_size=GROUP_SIZE)["answers"])
            chunk, rows = {}, 0
        chunk[qid], rows = q, rows + q_rows
    if chunk:
        answers.update(laya.predict_tournament(agent, state, chunk,
                                               group_size=GROUP_SIZE)["answers"])
    return answers


def choice_row(pid, ans):
    probs = sorted(ans["probabilities"].values(), reverse=True)
    return dict(employee_id=pid, choice=ans["choice"],
                raw_confidence=round(float(ans["confidence"]), 4),
                raw_margin=round(float(probs[0] - (probs[1] if len(probs) > 1 else 0)), 4),
                finalists=len(ans["probabilities"]))


def judge_company(laya, agent, co):
    people = list(co.itertuples())
    state = {"organisation": co.company_name.iloc[0], "people": len(people)}
    questions = {p.employee_id: manager_question(p, people) for p in people}
    return [choice_row(pid, ans) for pid, ans in ask(laya, agent, state, questions).items()]


def main(data_dir, companies=None):
    data_dir = Path(data_dir)
    out = data_dir / "runs" / "laya"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_dir / "employees.csv")
    if companies:
        df = df[df.company_id.isin(companies)]
    done = pd.read_csv(out / "raw.csv") if (out / "raw.csv").exists() else pd.DataFrame()
    todo = df[~df.employee_id.isin(done.get("employee_id", []))]

    laya, agent = load_agent()
    for cid, co in todo.groupby("company_id"):
        start = time.time()
        rows = pd.DataFrame(judge_company(laya, agent, co))
        done = pd.concat([done, rows], ignore_index=True)
        done.to_csv(out / "raw.csv", index=False)       # saved per company: safe to stop
        print(f"{cid} {co.company_name.iloc[0]}: {len(co)} people in {time.time() - start:.0f}s")
    print(f"{len(done)} answers in {out / 'raw.csv'}")


def to_predictions(employees, raw, confidence, name="Laya"):
    """Raw answers + a (calibrated) confidence per answer -> predictions.csv columns."""
    pred = employees.merge(raw, on="employee_id", how="left")
    is_person = pred.choice.notna() & ~pred.choice.isin([TOP, NOT_LISTED])
    pred["pred_manager_id"] = pred.choice.where(is_person)
    pred["confidence"] = pd.Series(confidence, index=pred.index).where(is_person, 0.0).round(3)
    pred["rule_confidence"] = pred.raw_confidence.where(is_person, 0.0)
    pred["reason"] = pred.choice.map({TOP: f"{name}: heads the organisation",
                                      NOT_LISTED: f"{name}: manager not in the data"})
    return pred


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    picked = sys.argv[sys.argv.index("--companies") + 1:] if "--companies" in sys.argv else None
    main(args[0] if args else "data/scraped/output/asx",
         [c for c in picked if not c.startswith("--")] if picked else None)

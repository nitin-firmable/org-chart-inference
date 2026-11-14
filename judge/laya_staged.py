"""
judge/laya_staged.py  -  Laya in stages: level first, then the manager within the domain

Asking "who is the manager?" with everyone as a candidate fails: Laya matches wording, not
seniority, and picks a Deputy Director as the boss of a Director General. So, per company:

1. Level (Score, 1-10): how senior is each person? Laya's expected level, e.g. 4.6.
2. Domain: the person's unit when the data has one; otherwise a Choice over business areas.
3. Manager (Choice) among people in the SAME domain at the SAME or a HIGHER level, or "none".
4. Fallback (Choice) when step 3 finds nobody: everyone OUTSIDE the domain at the same or a
   higher level, or "heads the organisation". The normal path for the head of a domain.

    python -m judge.laya_staged data/scraped/output/uk     # -> <data>/runs/laya_staged/raw.csv

Writes raw answers plus each person's candidate lists, so the harness can measure how often
the true manager survived steps 1-2 (if not, no later step can find them).
"""
import sys
import time
from pathlib import Path

import pandas as pd

from judge.decisions import NOT_LISTED, TOP, domain_question, level_question, manager_among
from judge.laya_judge import ask, choice_row, load_agent

SAME_LEVEL_SLACK = 0.5          # "same or higher": allow candidates up to half a level below


def judge_company(laya, agent, co):
    people = list(co.itertuples())
    by_id = {p.employee_id: p for p in people}
    state = {"organisation": co.company_name.iloc[0], "people": len(people)}

    levels = ask(laya, agent, state, {p.employee_id: level_question(p) for p in people})
    level = {pid: a["score"] + 1 for pid, a in levels.items()}      # 0-9 -> 1-10

    domain = {p.employee_id: p.department for p in people if isinstance(p.department, str)}
    unknown = [p for p in people if p.employee_id not in domain]
    if unknown:
        asked = ask(laya, agent, state, {p.employee_id: domain_question(p) for p in unknown})
        domain.update({pid: a["choice"] for pid, a in asked.items()})

    def senior_enough(p, c):
        return c.employee_id != p.employee_id and \
            level[c.employee_id] <= level[p.employee_id] + SAME_LEVEL_SLACK

    inside = {p.employee_id: [c for c in people if senior_enough(p, c)
                              and domain[c.employee_id] == domain[p.employee_id]]
              for p in people}
    outside = {p.employee_id: [c for c in people if senior_enough(p, c)
                               and domain[c.employee_id] != domain[p.employee_id]]
               for p in people}

    first = ask(laya, agent, state, {p.employee_id: manager_among(p, inside[p.employee_id])
                                     for p in people if inside[p.employee_id]})
    answers, stage = {}, {}
    for pid, a in first.items():
        if a["choice"] != NOT_LISTED:
            answers[pid], stage[pid] = a, "domain"
    retry = [p for p in people if p.employee_id not in answers]
    second = ask(laya, agent, state, {p.employee_id: manager_among(p, outside[p.employee_id],
                                                                   top=True) for p in retry})
    for pid, a in second.items():
        answers[pid], stage[pid] = a, "fallback"

    rows = []
    for pid, a in answers.items():
        row = choice_row(pid, a)
        row.update(stage=stage[pid], level=round(level[pid], 2), domain=domain[pid],
                   inside=";".join(c.employee_id for c in inside[pid]),
                   outside=";".join(c.employee_id for c in outside[pid]))
        rows.append(row)
    return rows


def main(data_dir, companies=None):
    data_dir = Path(data_dir)
    out = data_dir / "runs" / "laya_staged"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_dir / "employees.csv")
    if companies:
        df = df[df.company_id.isin(companies)]
    done = pd.read_csv(out / "raw.csv") if (out / "raw.csv").exists() else pd.DataFrame()
    todo = df[~df.employee_id.isin(done.get("employee_id", []))]

    laya, agent = load_agent()
    for cid, co in todo.groupby("company_id"):
        start = time.time()
        done = pd.concat([done, pd.DataFrame(judge_company(laya, agent, co))],
                         ignore_index=True)
        done.to_csv(out / "raw.csv", index=False)       # saved per company: safe to stop
        print(f"{cid} {co.company_name.iloc[0]}: {len(co)} people in {time.time() - start:.0f}s")
    print(f"{len(done)} answers in {out / 'raw.csv'}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    picked = sys.argv[sys.argv.index("--companies") + 1:] if "--companies" in sys.argv else None
    main(args[0] if args else "data/scraped/output/asx",
         [c for c in picked if not c.startswith("--")] if picked else None)

"""
what_if.py  -  one visitor's "what if" edits, as session tables over the shared DuckDB file

Each session gets its own in-memory DuckDB with the shared app/org_chart.duckdb attached
read-only, plus four small tables: added, removed, overrides (re-placed reports) and log.
Every action gets an edit number, so Undo deletes the latest one. Company and employee ids are
unique across datasets, so one set of edits covers every dataset. Nothing is ever written to
the shared file.
"""
import duckdb
import pandas as pd

from judge import live
from judge.decisions import NOT_LISTED, TOP

NEW_ORG = "__new__"
PERSON_COLUMNS = ["employee_id", "full_name", "title", "department"]


class WhatIf:
    def __init__(self, db_path):
        self.con = duckdb.connect()
        self.con.execute(f"ATTACH '{db_path}' AS shared (READ_ONLY)")
        self.con.execute("CREATE TABLE added AS SELECT 0 AS edit_no, * FROM shared.people LIMIT 0")
        self.con.execute("CREATE TABLE removed (edit_no INT, dataset VARCHAR, "
                         "employee_id VARCHAR, destination VARCHAR)")
        self.con.execute("CREATE TABLE overrides (edit_no INT, dataset VARCHAR, "
                         "employee_id VARCHAR, pred_manager_id VARCHAR, confidence DOUBLE, "
                         "reason VARCHAR, old_manager_id VARCHAR)")
        self.con.execute("CREATE TABLE log (edit_no INT, dataset VARCHAR, employee_id VARCHAR, "
                         "action VARCHAR, judge VARCHAR)")
        self.last_no = 0

    # ------------------------------------------------------------------ reading
    def counts(self):
        return {t: self.con.execute(f"SELECT count(*) FROM {t}").fetchall()[0][0]
                for t in ["added", "removed", "overrides"]}

    def has_edits(self):
        return any(self.counts().values())

    def changed(self):
        return set(self.con.execute("""
            SELECT employee_id FROM added UNION SELECT employee_id FROM removed
            UNION SELECT employee_id FROM overrides""").df().employee_id)

    def changed_companies(self, people):
        ids = self.changed()
        return set(people[people.employee_id.isin(ids)].company_id)

    def people(self, pairs):
        """Every (dataset, judge) in pairs, with this session's edits applied."""
        self.con.execute("CREATE OR REPLACE TEMP TABLE shown (dataset VARCHAR, judge VARCHAR)")
        self.con.executemany("INSERT INTO shown VALUES (?, ?)", list(pairs))
        return self.con.execute("""
            WITH base AS (
                SELECT p.*, FALSE AS is_new FROM shared.people p
                JOIN shown s ON s.dataset = p.dataset AND s.judge = p.judge
                UNION ALL BY NAME
                SELECT * EXCLUDE (edit_no), TRUE AS is_new FROM added
            ), o AS (
                SELECT * FROM overrides
                QUALIFY row_number() OVER (PARTITION BY employee_id ORDER BY edit_no DESC) = 1
            )
            SELECT b.* REPLACE (
                       CASE WHEN o.employee_id IS NULL THEN b.pred_manager_id
                            ELSE o.pred_manager_id END AS pred_manager_id,
                       CASE WHEN o.employee_id IS NULL THEN b.confidence
                            ELSE o.confidence END AS confidence,
                       CASE WHEN o.employee_id IS NULL THEN b.reason ELSE o.reason END AS reason),
                   o.old_manager_id AS moved_from, r.destination AS gone, k.true_manager_id
            FROM base b
            LEFT JOIN o ON o.employee_id = b.employee_id
            LEFT JOIN removed r ON r.employee_id = b.employee_id
            LEFT JOIN shared.answer_key k ON k.dataset = b.dataset AND k.employee_id = b.employee_id
            """).df()

    def last_change(self):
        """The latest action's people: what happened to each, for the result summary."""
        no = self.con.execute("SELECT max(edit_no) FROM log").fetchall()[0][0]
        if no is None:
            return None, pd.DataFrame()
        return no, self.con.execute("SELECT employee_id, action, judge FROM log "
                                    "WHERE edit_no = ?", [no]).df()

    # ------------------------------------------------------------------ writing
    def _next(self):
        self.last_no += 1
        return self.last_no

    def add(self, company, rows, backend, secrets, current, edit_no=None):
        """company: dict(company_id, company_name, industry, dataset); rows: [(name, title)]."""
        no = edit_no or self._next()
        new = pd.DataFrame([dict(
            company_id=company["company_id"], company_name=company["company_name"],
            industry=company["industry"], employee_id=f"ADD{no:03d}-{i}",
            full_name=name or f"New person {no}.{i}", title=title, department=None,
            source_url=None) for i, (name, title) in enumerate(rows, 1)])
        pool = pd.concat([current[(current.company_id == company["company_id"]) &
                                  current.gone.isna()][PERSON_COLUMNS], new[PERSON_COLUMNS]],
                         ignore_index=True)
        answers = live.judge(backend, company["company_name"], pool, list(new.employee_id),
                             secrets)
        for col in ["pred_manager_id", "confidence", "reason", "level", "domain", "is_top"]:
            new[col] = None
        for i, pid in enumerate(new.employee_id):
            a = answers.get(pid, {"choice": NOT_LISTED, "confidence": 0, "level": None,
                                  "domain": "", "reason": ""})
            person = a["choice"] not in (TOP, NOT_LISTED)
            new.loc[i, "pred_manager_id"] = a["choice"] if person else None
            new.loc[i, "confidence"] = a["confidence"] if person else 0.0
            new.loc[i, "is_top"] = a["choice"] == TOP
            new.loc[i, "level"] = 1 if a["choice"] == TOP else a["level"]
            new.loc[i, "domain"] = a["domain"]
            new.loc[i, "reason"] = (f"{live.LABELS[backend]}: heads the organisation"
                                    if a["choice"] == TOP else
                                    f"{live.LABELS[backend]}: manager not in the data"
                                    if not person else f"{live.LABELS[backend]}: new person")
        new["confidence"] = pd.to_numeric(new.confidence)
        new["level"] = pd.to_numeric(new.level)
        new["is_top"] = new.is_top.astype(bool)
        new = new.assign(edit_no=no, dataset=company["dataset"], judge="live")
        self.con.execute("INSERT INTO added BY NAME SELECT * FROM new")
        entries = pd.DataFrame({"edit_no": no, "dataset": company["dataset"],
                                "employee_id": new.employee_id,
                                "action": "joined" if edit_no else "added", "judge": backend})
        self.con.execute("INSERT INTO log BY NAME SELECT * FROM entries")
        return no

    def remove(self, company_id, leavers, backend, secrets, current, companies):
        """leavers: {employee_id: dict(dest_id, new_org, designation)}. Their direct reports
        are re-judged among the people who stay; movers are added to their new company."""
        no = self._next()
        co = current[(current.company_id == company_id) & current.gone.isna()]
        data = companies[company_id]["dataset"]
        name_of = dict(zip(co.employee_id, co.full_name))
        title_of = dict(zip(co.employee_id, co.title))
        for pid, dest in leavers.items():
            where = dest["new_org"] if dest["dest_id"] == NEW_ORG else \
                companies.get(dest["dest_id"], {}).get("company_name", "unknown")
            text = where + (f", {dest['designation']}" if dest["designation"] else "")
            self.con.execute("INSERT INTO removed VALUES (?, ?, ?, ?)", [no, data, pid, text])
            self.con.execute("INSERT INTO log VALUES (?, ?, ?, 'removed', ?)",
                             [no, data, pid, backend])

        stay = co[~co.employee_id.isin(leavers)]
        reports = stay[stay.pred_manager_id.isin(leavers)]
        if len(reports):
            answers = live.judge(backend, co.company_name.iloc[0], stay[PERSON_COLUMNS],
                                 list(reports.employee_id), secrets)
            left = " & ".join(name_of[p] for p in leavers)
            for r in reports.itertuples():
                a = answers.get(r.employee_id, {"choice": NOT_LISTED, "confidence": 0})
                person = a["choice"] not in (TOP, NOT_LISTED)
                reason = (f"{live.LABELS[backend]}: re-placed after {left} left" if person else
                          f"{live.LABELS[backend]}: no clear new manager after {left} left")
                self.con.execute("INSERT INTO overrides VALUES (?, ?, ?, ?, ?, ?, ?)",
                                 [no, data, r.employee_id, a["choice"] if person else None,
                                  a["confidence"] if person else 0.0, reason,
                                  r.pred_manager_id])
                self.con.execute("INSERT INTO log VALUES (?, ?, ?, 'moved', ?)",
                                 [no, data, r.employee_id, backend])

        for pid, dest in leavers.items():          # movers join their new company
            if not dest["dest_id"]:
                continue
            if dest["dest_id"] == NEW_ORG:
                company = dict(company_id=f"NEW{no:03d}", company_name=dest["new_org"],
                               industry="Added (what-if)", dataset="added")
            else:
                company = companies[dest["dest_id"]]
            self.add(company, [(name_of[pid], dest["designation"] or title_of[pid])],
                     backend, secrets, current, edit_no=no)
        return no

    def undo(self):
        no = self.con.execute("SELECT max(edit_no) FROM log").fetchall()[0][0]
        if no is not None:
            for table in ["added", "removed", "overrides", "log"]:
                self.con.execute(f"DELETE FROM {table} WHERE edit_no = ?", [no])

    def reset(self):
        for table in ["added", "removed", "overrides", "log"]:
            self.con.execute(f"DELETE FROM {table}")

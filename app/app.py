"""
app.py  -  Step 7
Streamlit app. Front page: one list of every company across the datasets (generated benchmark,
UK government organograms, ASX/NZX leadership teams), with a dataset filter. Click a company's
name (or use the sidebar) to open its org chart. Each dataset uses the judge that works on it.
Run locally from the repo root with:  streamlit run app/app.py

Everything comes from one shared DuckDB file, app/org_chart.duckdb (built by
`python -m harness.export`), opened read-only: the hosted app never runs a model judge.
"""
import sys
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # Streamlit only puts app/ on the path; we need the root
from app import org_chart  # noqa: E402
from app.what_if import NEW_ORG, WhatIf  # noqa: E402
from judge import live  # noqa: E402

DB = ROOT / "app/org_chart.duckdb"
FLOW = ROOT / "app/flow.svg"
DATASETS = {"uk": "UK government", "asx": "ASX/NZX leadership",
            "synthetic": "Benchmark", "added": "Added (what-if)"}
ABOUT = {
    "synthetic": "Generated benchmark companies (not real firms) with a hidden answer key.",
    "uk": "Senior civil servants from data.gov.uk organograms (Open Government Licence). "
          "Each post's real 'reports to' line is the answer key.",
    "asx": "Executive teams copied from each company's own leadership page (source on each "
           "company page). No answer key: only 2 managers are publicly stated.",
    "added": "An organisation you added in this session.",
}
HAS_TRUTH = {"synthetic": True, "uk": True, "asx": False, "added": False}
# The judge each dataset is shown with. Rules and Laya can't read real-world titles (blank or
# nonsense charts), so real data uses Claude; all judges stay in the DuckDB file and harness.
APP_JUDGE = {"synthetic": "rules", "uk": "claude", "asx": "claude"}
JUDGES = {"claude": "Claude", "rules": "Rules", "live": "live judge"}
DEFAULT_THRESHOLD = 0.7                # app default; the harness scores at 0.8 (wrong line = 4x)

st.set_page_config(page_title="Org Chart Inference", layout="wide")


@st.cache_resource
def db():
    """One read-only connection shared by every visitor; each query uses its own cursor."""
    return duckdb.connect(str(DB), read_only=True)


@st.cache_data
def load():
    """Returns (people, scored) for every dataset with its judge: everyone with the judge's
    answer, and the people who have a known manager to find (built like the harness)."""
    con = db().cursor()
    con.execute("CREATE OR REPLACE TEMP TABLE shown AS SELECT * FROM (VALUES "
                + ", ".join(f"('{d}', '{j}')" for d, j in APP_JUDGE.items())
                + ") t(dataset, judge)")
    people = con.execute("""
        SELECT p.*, k.true_manager_id
        FROM people p JOIN shown s ON s.dataset = p.dataset AND s.judge = p.judge
        LEFT JOIN answer_key k ON k.dataset = p.dataset AND k.employee_id = p.employee_id
        """).df()
    scored = con.execute("""
        SELECT k.dataset, k.company_id, k.employee_id, k.true_manager_id,
               p.pred_manager_id, p.confidence
        FROM answer_key k JOIN shown s ON s.dataset = k.dataset
        LEFT JOIN people p ON p.dataset = k.dataset AND p.employee_id = k.employee_id
                          AND p.judge = s.judge""").df()
    return people, scored


def break_cycles(frame):
    """A model can say A reports to B and B to A; those people would vanish from the chart.
    In each loop, drop the least confident line (every person has one manager, so each
    connected group holds at most one loop)."""
    boss = {e: m for e, m in zip(frame.employee_id, frame.manager_id) if isinstance(m, str)}
    conf = dict(zip(frame.employee_id, frame.confidence))
    state, dropped = {}, set()
    for start in boss:
        path, cur = [], start
        while cur in boss and cur not in state:
            state[cur] = "open"
            path.append(cur)
            cur = boss[cur]
        if state.get(cur) == "open":                  # walked back into this path: a loop
            dropped.add(min(path[path.index(cur):], key=lambda p: conf.get(p, 0)))
        for p in path:
            state[p] = "done"
    if dropped:
        hit = frame.employee_id.isin(dropped)
        frame.loc[hit, "manager_id"] = None
        if "reason" in frame:
            frame.loc[hit, "reason"] = "circular reporting line (weakest line dropped)"
    return frame


def apply_threshold(frame, threshold):
    """Lines below the threshold are not drawn: the manager is left blank."""
    frame = frame.copy()
    frame["manager_id"] = frame.pred_manager_id.where(frame.confidence >= threshold)
    if "reason" in frame:
        below = frame.pred_manager_id.notna() & frame.manager_id.isna()
        frame.loc[below, "reason"] = "confidence below threshold"
    frame = break_cycles(frame)
    frame["given"] = frame.manager_id.notna()
    frame["right"] = frame.given & (frame.manager_id == frame.true_manager_id)
    return frame


def eligible(people, scored, threshold):
    """The people coverage is measured over. With an answer key: people whose manager is known
    (as in the harness). Without one: everyone except heads and people who left."""
    truth = [d for d in people.dataset.unique() if HAS_TRUTH.get(d)]
    known = apply_threshold(scored[scored.dataset.isin(truth)], threshold)
    rest = apply_threshold(people[~people.dataset.isin(truth)], threshold)
    rest = rest[~rest.is_top.astype(bool) & rest.gone.isna()]
    return known, rest


def measure(people, scored, threshold):
    """(coverage, precision); precision only counts companies with an answer key."""
    known, rest = eligible(people, scored, threshold)
    n = len(known) + len(rest)
    coverage = (known.given.sum() + rest.given.sum()) / n if n else 0.0
    precision = known.right.sum() / max(known.given.sum(), 1) if len(known) else None
    return coverage, precision


ss = st.session_state
ss.setdefault("table_version", 0)

# ---------------------------------------------------------------- everyone, with session edits
base, scored = load()
if "what_if" not in ss:
    ss.what_if = WhatIf(DB)
wi = ss.what_if
editing = wi.has_edits()
if editing:
    df = wi.people(APP_JUDGE.items())
    scored = scored[~scored.employee_id.isin(wi.changed())]
else:
    df = base.assign(is_new=False, moved_from=None, gone=None)

# Largest first, then by name; the table's clicked row is looked up in the filtered list
companies = (df[["company_id", "company_name", "industry", "dataset"]]
             .drop_duplicates("company_id")
             .merge(df.groupby("company_id").size().rename("size"), on="company_id")
             .sort_values(["size", "company_name"], ascending=[False, True])
             .reset_index(drop=True))
company_label = dict(zip(companies.company_id,
                         companies.company_name + " (" + companies.company_id + ")"))
company_info = {r.company_id: dict(company_id=r.company_id, company_name=r.company_name,
                                   industry=r.industry, dataset=r.dataset)
                for r in companies.itertuples()}
present = [d for d in DATASETS if d in set(companies.dataset)]
ss.setdefault("dataset_filter", present)
new_datasets = [d for d in present if d not in ss.get("seen_datasets", present)]
if new_datasets:                       # e.g. "Added" after a new organisation: show it
    ss.dataset_filter = list(ss.dataset_filter) + new_datasets
ss.seen_datasets = present
picked = [d for d in ss.dataset_filter if d in present] or present
shown = companies[companies.dataset.isin(picked)].reset_index(drop=True)

# ---------------------------------------------------------------- which company is open?
# One source of truth: ss.company_select (also the sidebar dropdown's value).
# Table clicks and the Back button write to it; the URL (?company=UK10) mirrors it.
if "company_select" not in ss:
    from_url = st.query_params.get("company", "")
    ss.company_select = from_url if from_url in company_label else ""
if "open_company" in ss:
    ss.company_select = ss.pop("open_company")
if ss.company_select not in company_label:
    ss.company_select = ""
table_key = f"company_table_{ss.table_version}"
clicked = ss.get(table_key)
cells = clicked["selection"].get("cells", []) if clicked else []
if cells and cells[0][1] == "company_name" and cells[0][0] < len(shown):
    ss.company_select = shown.company_id.iloc[cells[0][0]]   # a click on a name opens it
    ss.table_version += 1          # fresh table next time, so the old click doesn't stick


def back_to_all():
    ss.company_select = ""
    ss.table_version += 1


# ---------------------------------------------------------------- sidebar
st.sidebar.selectbox("Company", [""] + sorted(companies.company_id, key=company_label.get),
                     key="company_select",
                     format_func=lambda c: company_label.get(c, "All companies"))
threshold = st.sidebar.slider(
    "Confidence threshold", 0.5, 1.0, DEFAULT_THRESHOLD, 0.05,
    help="How sure a judge must be before a reporting line is drawn. Benchmark: a decision "
         "tree's confidence; UK and ASX: Claude's confidence, calibrated on the UK answer key. "
         "Default 0.7 shows more lines; 0.8 is the strict setting where a wrong line counts "
         "4x worse than a missing one (the reported results use 0.8).")
show_truth = st.sidebar.checkbox("Show accuracy vs. answer key", value=True)
cid = ss.company_select
data = company_info[cid]["dataset"] if cid else None
for old in ["data", "judge"]:                       # links from before the combined list
    st.query_params.pop(old, None)
if cid:
    st.query_params["company"] = cid
else:
    st.query_params.pop("company", None)

st.title("Org Chart Inference")
if cid:
    judge = APP_JUDGE.get(data, "live")
    st.caption(f"{DATASETS[data]} · judge: {JUDGES[judge]}. {ABOUT[data]} No line is drawn "
               "when confidence is below the threshold — a wrong line is worse than no line.")
else:
    st.caption("Reporting lines inferred from names and job titles only. Benchmark: generated "
               "companies with a hidden answer key (rules judge). UK government: real "
               "organograms with each post's real manager as the answer key (Claude). ASX/NZX: "
               "real leadership teams, no answer key (Claude). No line is drawn when confidence "
               "is below the threshold — a wrong line is worse than no line.")


# ---------------------------------------------------------------- what if: add / remove people
def secrets():
    try:
        return dict(st.secrets)
    except StreamlitSecretNotFoundError:
        return {}


def live_judge_choice(key):
    """Best available live judge by default; changeable under Advanced."""
    found = live.available(secrets())
    with st.expander("Advanced: which judge places these people"):
        return st.radio("Live judge", found, format_func=live.LABELS.get, key=key,
                        horizontal=True)


def company_option(c):
    return "＋ New organisation…" if c == NEW_ORG else company_label.get(c, c)


@st.dialog("Add people", width="large")
def add_dialog(default_cid):
    ids = sorted(companies.company_id, key=company_label.get)
    pick = st.selectbox("Company", ids + [NEW_ORG], format_func=company_option,
                        index=ids.index(default_cid) if default_cid in ids else len(ids))
    new_org = st.text_input("New organisation name", placeholder="e.g. Acme Pty Ltd") \
        if pick == NEW_ORG else ""
    one, many = st.tabs(["One by one", "Paste a list"])
    with one:
        table = st.data_editor(
            pd.DataFrame({"Name (optional)": [""] * 3, "Designation": [""] * 3}),
            num_rows="dynamic", hide_index=True, width="stretch", key="add_table")
    with many:
        pasted = st.text_area("One person per line: Name | Designation, or just the designation",
                              placeholder="Priya Shah | Head of Data Engineering\n"
                                          "Senior Data Analyst", height=150)
    backend = live_judge_choice("add_judge")
    if not st.button("Add and place them", type="primary"):
        return
    rows = [(str(n or "").strip(), str(d or "").strip())
            for n, d in table.itertuples(index=False) if str(d or "").strip()]
    for line in pasted.splitlines():
        parts = [x.strip() for x in line.replace("\t", "|").split("|")]
        if parts[-1]:
            rows.append((parts[0] if len(parts) > 1 else "", parts[-1]))
    if not rows:
        st.error("Add at least one designation.")
        return
    if pick == NEW_ORG and not new_org.strip():
        st.error("Give the new organisation a name.")
        return
    company = company_info.get(pick) or dict(company_id=f"NEW{wi.last_no + 1:03d}",
                                             company_name=new_org.strip(),
                                             industry="Added (what-if)", dataset="added")
    try:
        with st.spinner(f"{live.LABELS[backend]} is placing {len(rows)} people…"):
            wi.add(company, rows, backend, secrets(), df)
    except live.LiveJudgeError as exc:
        st.error(str(exc))
        return
    ss.pop("seen_change", None)
    if pick == NEW_ORG:
        ss.open_company = company["company_id"]      # applied before the sidebar is drawn
    st.rerun()


@st.dialog("Remove people", width="large")
def remove_dialog(cid):
    co = df[(df.company_id == cid) & df.gone.isna()]
    who = dict(zip(co.employee_id, co.full_name + " — " + co.title.fillna("(no title)")))
    picks = st.multiselect("Who is leaving?", list(who), format_func=who.get,
                           placeholder="Pick one or more people")
    others = [c for c in sorted(companies.company_id, key=company_label.get) if c != cid]
    leavers = {}
    for pid in picks:
        with st.container(border=True):
            st.markdown(f"**{who[pid]}**")
            c1, c2, c3 = st.columns(3)
            dest = c1.selectbox("Moving to (optional)", [""] + others + [NEW_ORG], key=f"to_{pid}",
                                format_func=lambda c: "Unknown / not in the data" if c == ""
                                else company_option(c))
            org = c2.text_input("New organisation name", key=f"org_{pid}",
                                disabled=dest != NEW_ORG, placeholder="e.g. Google")
            role = c3.text_input("New designation (optional)", key=f"role_{pid}",
                                 placeholder="e.g. Staff Engineer")
            leavers[pid] = dict(dest_id=dest, new_org=org.strip(), designation=role.strip())
    backend = live_judge_choice("remove_judge")
    if not st.button("Remove and re-place their team", type="primary", disabled=not picks):
        return
    if any(d["dest_id"] == NEW_ORG and not d["new_org"] for d in leavers.values()):
        st.error("Give each new organisation a name.")
        return
    try:
        with st.spinner(f"{live.LABELS[backend]} is re-placing their team…"):
            wi.remove(cid, leavers, backend, secrets(), df, company_info)
    except live.LiveJudgeError as exc:
        st.error(str(exc))
        return
    ss.pop("seen_change", None)
    st.rerun()


def what_if_bar():
    if not editing:
        return
    c = wi.counts()
    touched = df[df.company_id.isin(wi.changed_companies(df))]
    with st.container(border=True):
        b1, b2, b3, b4 = st.columns([6, 1.3, 1, 1.7], vertical_alignment="center")
        b1.markdown(f"**What-if mode** · {c['added']} added · {c['removed']} removed · "
                    f"{c['overrides']} re-placed  \nOnly in your session; the shared data is "
                    "unchanged. Accuracy leaves out the people you changed.")
        b2.button("Undo last", on_click=wi.undo, width="stretch")
        b3.button("Reset", on_click=wi.reset, width="stretch")
        b4.download_button("Download CSV", touched.to_csv(index=False).encode(),
                           "what_if_changed_companies.csv", "text/csv", width="stretch",
                           help="The companies you changed, with your edits applied")


def last_change_summary():
    no, change = wi.last_change()
    if no is None or ss.get("seen_change") == no:
        return
    drawn = apply_threshold(df[df.employee_id.isin(change.employee_id) |
                               df.company_id.isin(wi.changed_companies(df))], threshold)
    drawn = drawn.set_index("employee_id")
    name_of = dict(zip(df.employee_id, df.full_name))
    verbs = {"added": "added", "joined": "joined", "removed": "left", "moved": "re-placed"}
    rows = []
    for r in change.drop_duplicates(["employee_id", "action"]).itertuples():
        if r.employee_id not in drawn.index:
            continue
        p = drawn.loc[r.employee_id]
        if r.action == "removed":
            result, boss, conf = f"left → {p.gone}", "", None
        else:
            boss = name_of.get(p.pred_manager_id, "")
            conf = p.confidence if isinstance(p.pred_manager_id, str) else None
            result = ("line drawn" if p.given else "heads the organisation" if p.is_top
                      else f"not drawn: below {threshold:.2f}" if boss
                      else str(p.reason).split(": ", 1)[-1])
        rows.append({"Person": p.full_name, "Designation": p.title, "Company": p.company_name,
                     "Change": verbs[r.action], "Reports to": boss, "Confidence": conf,
                     "Result": result})
    table = pd.DataFrame(rows)
    if table.empty:
        return
    if table.Company.nunique() == 1:
        table = table.drop(columns="Company")
    with st.container(border=True):
        h1, h2 = st.columns([8, 1], vertical_alignment="center")
        backend = change.judge.iloc[0]
        h1.markdown(f"**Last change** · judged by {live.LABELS[backend]} · lines are drawn at "
                    f"confidence ≥ {threshold:.2f}" +
                    (" · Gemini's confidence isn't calibrated yet, so lower the threshold to "
                     "see its proposals" if backend == "gemini" else ""))
        h2.button("Dismiss", on_click=lambda: ss.update(seen_change=no), width="stretch")
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"Confidence": st.column_config.NumberColumn(format="%.2f")})


@st.dialog("Docs", width="large")
def docs_dialog():
    svg = FLOW.read_text().replace('width="1200" height="880"', 'width="100%"', 1)
    st.iframe(f"<div>{svg}</div>", height=880)


def sidebar_docs():
    st.sidebar.divider()
    if st.sidebar.button("Docs: how it works", icon=":material/account_tree:", width="stretch"):
        docs_dialog()


what_if_bar()
last_change_summary()

# ================================================================ front page: all companies
if not cid:
    st.write("Click a company's name to see its org chart. Filter by dataset, search with the "
             "magnifying glass (top right of the table), sort by clicking a column header.")
    if st.button("＋ Add people", help="Add people to a company, or to a new organisation"):
        add_dialog(None)
    view_people = df[df.dataset.isin(picked)]
    view_scored = scored[scored.dataset.isin(picked)]
    coverage, precision = measure(view_people, view_scored, threshold)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Companies", f"{len(shown):,}")
    c2.metric("People", f"{len(view_people):,}")
    c3.metric("Coverage", f"{coverage:.1%}",
              help="Share of people (excluding the head of each organisation) who got a "
                   "manager")
    if show_truth and precision is not None:
        c4.metric("Precision", f"{precision:.1%}",
                  help="Of the reporting lines drawn, the share that match the answer key "
                       "(companies with an answer key only)")

    st.subheader("Companies")
    st.pills("Dataset", present, selection_mode="multi", key="dataset_filter",
             format_func=DATASETS.get, label_visibility="collapsed",
             on_change=lambda: ss.update(table_version=ss.table_version + 1))
    known, rest = eligible(view_people, view_scored, threshold)
    eligible_rows = pd.concat([known[["company_id", "given"]], rest[["company_id", "given"]]])
    table = shown.set_index("company_id").join(pd.DataFrame({
        "lines": apply_threshold(view_people, threshold).groupby("company_id").given.sum(),
        "coverage": 100 * eligible_rows.groupby("company_id").given.mean(),
        "precision": 100 * known.groupby("company_id").right.sum()
        / known.groupby("company_id").given.sum().clip(lower=1),
    })).reset_index(drop=True)
    table["dataset"] = table.dataset.map(DATASETS)
    for col in ["coverage", "precision"]:          # numbers, so the columns sort correctly
        table[col] = pd.to_numeric(table[col], errors="coerce").astype(float)
    has_precision = show_truth and table.precision.notna().any()
    columns = ["company_name", "dataset", "industry", "size", "lines", "coverage"] + \
        (["precision"] if has_precision else [])
    st.dataframe(
        table, key=table_key, on_select="rerun", selection_mode="single-cell",
        hide_index=True, width="stretch", column_order=columns,
        height=min(560, 35 * (len(table) + 1) + 3),     # fit the rows: no empty lines
        column_config={
            "company_name": st.column_config.TextColumn(
                "Company", help="Click a company's name to open its org chart", width=300),
            "dataset": st.column_config.TextColumn("Dataset", width=135),
            "industry": st.column_config.TextColumn("Industry", width=215),
            "size": st.column_config.NumberColumn("Employees", format="%d", width=85),
            "lines": st.column_config.NumberColumn("Lines drawn", format="%d", width=85),
            "coverage": st.column_config.NumberColumn("Coverage", format="%.0f%%", width=80),
            "precision": st.column_config.NumberColumn(
                "Precision", format="%.1f%%", width=80,
                help="Share of drawn lines that match the answer key; None where there is none"),
        })
    sidebar_docs()
    st.stop()

# ================================================================ company page
b1, b2, b3, _ = st.columns([1.6, 1.3, 1.5, 4])
b1.button("← Back to all companies", on_click=back_to_all)
if b2.button("＋ Add people"):
    add_dialog(cid)
if b3.button("− Remove people"):
    remove_dialog(cid)
co = apply_threshold(df[df.company_id == cid], threshold)
# Re-placed after someone left: "was under X", and "interim" if moved up to X's own manager
boss_before = dict(zip(co.employee_id, co.pred_manager_id))
name_of = dict(zip(co.employee_id, co.full_name))
co["moved"] = [(f"was under {name_of.get(m, '?')}" +
                (" · interim" if p == boss_before.get(m) else "")) if isinstance(m, str) else ""
               for m, p in zip(co.moved_from, co.pred_manager_id)]
domains = ["All"] + sorted(co.domain.dropna().astype(str).unique().tolist())
domain = st.sidebar.selectbox("Domain", domains,
                              help="Function, unit or division, as the judge read it")
show_truth = show_truth and HAS_TRUTH.get(data, False)
if not HAS_TRUTH.get(data, False):
    st.sidebar.caption("No answer key for this company, so no accuracy to show.")

# ---------------------------------------------------------------- headline numbers
info = co.iloc[0]
coverage, precision = measure(df[df.company_id == cid], scored[scored.company_id == cid],
                              threshold)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Employees", len(co), help="People listed for this company")
c2.metric("Industry", info.industry)
c3.metric("Coverage", f"{coverage:.0%}",
          help="Share of people (excluding the head) who got a manager at this threshold")
if show_truth and precision is not None:
    c4.metric("Precision", f"{precision:.1%}",
              help="Of the reporting lines drawn, the share that match the answer key")
sources = co.source_url.dropna().unique() if "source_url" in co else []
if len(sources):
    st.caption("Source: " + " · ".join(f"[{u}]({u})" for u in sources))

# ---------------------------------------------------------------- the org chart
view = co if domain == "All" else co[(co.domain == domain) | co.is_top.astype(bool)]
st.subheader(f"{info.company_name} — {domain}")
if not view.is_top.astype(bool).any() and not view.given.any():
    # Nothing to draw: say why instead of showing an empty chart
    with st.container(border=True):
        st.markdown(f"**Nothing could be placed in this view.** The {JUDGES[judge]} found "
                    "no head of the organisation and drew no reporting lines"
                    + (" for this domain. Try **All** domains." if domain != "All" else
                       f" at confidence ≥ {threshold:.2f}. It may not be able to read these "
                       "titles, or it isn't confident enough: try a lower threshold."))
# Our own HTML: names and titles are escaped inside org_chart.py before display
st.iframe(org_chart.render_html(view, show_truth), height=720)

# ---------------------------------------------------------------- lookups and gaps
st.subheader("Who does this person report to?")
person = st.selectbox("Person", co.full_name + " — " + co.title.fillna("(no title)"))
row = co[(co.full_name + " — " + co.title.fillna("(no title)")) == person].iloc[0]
chain, cur = [], row
while pd.notna(cur.manager_id) and len(chain) < 10:
    edge_conf = cur.confidence                     # confidence of THIS reporting line
    cur = co[co.employee_id == cur.manager_id].iloc[0]
    chain.append(f"{cur.full_name} ({cur.title}) — confidence {edge_conf:.2f}")
if chain:
    st.write("Reporting chain, bottom to top:")
    for i, c in enumerate(chain, 1):
        st.write(f"{i}. {c}")
elif row.is_top:
    st.write("Judged to head the organisation.")
else:
    st.write(f"No manager assigned. Reason: {row.reason or 'below threshold'}")

st.subheader("People left without a manager")
st.dataframe(co[co.manager_id.isna() & ~co.is_top.astype(bool) & co.gone.isna()][
    ["full_name", "title", "department", "reason"]], width="stretch", hide_index=True)
sidebar_docs()

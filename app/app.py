"""
app.py  -  Step 7
Streamlit app. Pick a dataset (synthetic, or real: UK government, ASX/NZX leadership) and a
judge (rules, Claude, Laya). Front page: overall results and a searchable,
sortable table of companies. Click a row (or use the sidebar) to open that company's org chart.
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
DATASETS = {"synthetic": "Benchmark · 500 companies", "uk": "UK government",
            "asx": "ASX/NZX leadership"}
ABOUT = {
    "synthetic": "Generated benchmark companies (not real firms) with a hidden answer key.",
    "uk": "Senior civil servants from data.gov.uk organograms (Open Government Licence). "
          "Each post's real 'reports to' line is the answer key.",
    "asx": "Executive teams copied from each company's own leadership page (source on each "
           "company page). No answer key: only 2 managers are publicly stated.",
}
HAS_TRUTH = {"synthetic": True, "uk": True, "asx": False}
# Judges shown in the app. Rules and Laya can't read real-world titles (blank or nonsense
# charts), so on real data only Claude is shown; all judges stay in the DuckDB file and harness.
APP_JUDGES = {"synthetic": ["rules"], "uk": ["claude"], "asx": ["claude"]}
JUDGES = {"claude": "Claude", "rules": "Rules", "laya_staged": "Laya (staged)", "laya": "Laya"}
DEFAULT_THRESHOLD = 0.7                # app default; the harness scores at 0.8 (wrong line = 4x)

st.set_page_config(page_title="Org Chart Inference", layout="wide")


@st.cache_resource
def db():
    """One read-only connection shared by every visitor; each query uses its own cursor."""
    return duckdb.connect(str(DB), read_only=True)


def judges_for(data):
    row = db().cursor().execute("SELECT judges FROM datasets WHERE dataset = ?",
                                [data]).fetchone()
    have = row[0].split(",") if row else []
    return [j for j in APP_JUDGES.get(data, have) if j in have] or have


@st.cache_data
def load(data, judge):
    """Returns (people, scored): everyone with the judge's answer, and the people who have
    a known manager to find (built like harness/evaluate.py, so the numbers match)."""
    con = db().cursor()
    people = con.execute("""
        SELECT p.* EXCLUDE (dataset, judge), k.true_manager_id
        FROM people p
        LEFT JOIN answer_key k ON k.dataset = p.dataset AND k.employee_id = p.employee_id
        WHERE p.dataset = ? AND p.judge = ?""", [data, judge]).df()
    scored = con.execute("""
        SELECT k.company_id, k.employee_id, k.true_manager_id, p.pred_manager_id, p.confidence
        FROM answer_key k
        LEFT JOIN people p ON p.dataset = k.dataset AND p.employee_id = k.employee_id
                          AND p.judge = ?
        WHERE k.dataset = ?""", [judge, data]).df()
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


def score(frame):
    """Coverage and precision, as defined in evaluate.py."""
    return frame.given.mean(), frame.right.sum() / max(frame.given.sum(), 1)


def measure(people, scored, threshold):
    """(coverage, precision). With an answer key, over people whose manager is known (as in
    the harness); without one, coverage over everyone but the head, and no precision."""
    if HAS_TRUTH[data]:
        return score(apply_threshold(scored, threshold))
    p = apply_threshold(people, threshold)
    keep = ~p.is_top.astype(bool)
    if "gone" in p:
        keep &= p.gone.isna()                      # people who left don't count
    return p[keep].given.mean(), None


ss = st.session_state

# ---------------------------------------------------------------- dataset and judge
# Defaults come from the URL (?data=uk&judge=claude&company=UK02), so links are shareable.
if "data_select" not in ss:
    from_url = st.query_params.get("data", "")
    ss.data_select = from_url if from_url in DATASETS else "synthetic"
ss.setdefault("table_version", 0)


def dataset_changed():
    ss.company_select = ""
    ss.table_version += 1
    ss.pop("judge_select", None)


st.sidebar.selectbox("Dataset", list(DATASETS), key="data_select", format_func=DATASETS.get,
                     on_change=dataset_changed)
data = ss.data_select
judges = judges_for(data)
if ss.get("judge_select") not in judges:
    from_url = st.query_params.get("judge", "")
    ss.judge_select = from_url if from_url in judges else judges[0]
st.sidebar.selectbox("Judge", judges, key="judge_select", format_func=JUDGES.get,
                     on_change=lambda: ss.update(table_version=ss.table_version + 1))
judge = ss.judge_select

df, scored = load(data, judge)
# This visitor's "what if" edits: session tables over the shared file, per dataset
if "what_if" not in ss:
    ss.what_if = WhatIf(DB)
wi = ss.what_if
editing = wi.has_edits(data)
if editing:
    df = wi.people(data, judge)
    scored = scored[~scored.employee_id.isin(wi.changed(data))]
else:
    df = df.assign(is_new=False, moved_from=None, gone=None)
# Fixed order (largest first, then by name): the table's clicked row is looked up here
companies = (df[["company_id", "company_name", "industry"]].drop_duplicates("company_id")
             .merge(df.groupby("company_id").size().rename("size"), on="company_id")
             .sort_values(["size", "company_name"], ascending=[False, True])
             .reset_index(drop=True))
company_label = dict(zip(companies.company_id,
                         companies.company_name + " (" + companies.company_id + ")"))
company_info = {r.company_id: dict(company_id=r.company_id, company_name=r.company_name,
                                   industry=r.industry) for r in companies.itertuples()}

# ---------------------------------------------------------------- which company is open?
# One source of truth: ss.company_select (also the sidebar dropdown's value).
# Table clicks and the Back button write to it; the URL (?company=C001) mirrors it.
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
if cells and cells[0][1] == "company_name":   # a click on a company's name opens it
    ss.company_select = companies.company_id.iloc[cells[0][0]]
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
    help=("Confidence comes from a decision tree trained on a separate dataset."
          if judge == "rules" else
          "The model's confidence, calibrated on the UK answer key (each organisation by a "
          "map fitted on the others).") +
    " Default 0.7 shows more lines; 0.8 is the strict setting where a wrong line counts "
    "4x worse than a missing one (the reported results use 0.8).")
if HAS_TRUTH[data]:
    show_truth = st.sidebar.checkbox("Show accuracy vs. answer key", value=True)
else:
    show_truth = False
    st.sidebar.caption("No answer key for this dataset, so no accuracy to show.")
cid = ss.company_select
st.query_params.update(data=data, judge=judge)
if cid:
    st.query_params["company"] = cid
else:
    st.query_params.pop("company", None)

st.title("Org Chart Inference")
st.caption(f"{DATASETS[data]} · judge: {JUDGES[judge]}. {ABOUT[data]} "
           "No line is drawn when confidence is below the threshold — "
           "a wrong line is worse than no line.")


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
    ids = companies.company_id.tolist()
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
                                             industry="Added (what-if)")
    try:
        with st.spinner(f"{live.LABELS[backend]} is placing {len(rows)} people…"):
            wi.add(data, judge, company, rows, backend, secrets(), df)
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
    others = [c for c in companies.company_id if c != cid]
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
            wi.remove(data, judge, cid, leavers, backend, secrets(), df, company_info)
    except live.LiveJudgeError as exc:
        st.error(str(exc))
        return
    ss.pop("seen_change", None)
    st.rerun()


def what_if_bar():
    if not editing:
        return
    c = wi.counts(data)
    with st.container(border=True):
        b1, b2, b3, b4 = st.columns([6, 1.3, 1, 1.7], vertical_alignment="center")
        b1.markdown(f"**What-if mode** · {c['added']} added · {c['removed']} removed · "
                    f"{c['overrides']} re-placed  \nOnly in your session; the shared data is "
                    "unchanged. Accuracy leaves out the people you changed.")
        b2.button("Undo last", on_click=wi.undo, args=(data,), width="stretch")
        b3.button("Reset", on_click=wi.reset, args=(data,), width="stretch")
        b4.download_button("Download CSV", df.to_csv(index=False).encode(),
                           f"what_if_{data}.csv", "text/csv", width="stretch")


def last_change_summary():
    no, change = wi.last_change(data)
    if no is None or ss.get("seen_change") == (data, no):
        return
    drawn = apply_threshold(df, threshold).set_index("employee_id")
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
    if table.Company.nunique() == 1:
        table = table.drop(columns="Company")
    with st.container(border=True):
        h1, h2 = st.columns([8, 1], vertical_alignment="center")
        backend = change.judge.iloc[0]
        h1.markdown(f"**Last change** · judged by {live.LABELS[backend]} · lines are drawn at "
                    f"confidence ≥ {threshold:.2f}" +
                    (" · Gemini's confidence isn't calibrated yet, so lower the threshold to "
                     "see its proposals" if backend == "gemini" else ""))
        h2.button("Dismiss", on_click=lambda: ss.update(seen_change=(data, no)), width="stretch")
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
    st.write("Click a company's name to see its org chart. Search with the magnifying "
             "glass (top right of the table), sort by clicking a column header.")
    if st.button("＋ Add people", help="Add people to a company, or to a new organisation"):
        add_dialog(None)
    s = apply_threshold(scored, threshold)
    coverage, precision = measure(df, scored, threshold)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Companies", f"{len(companies):,}")
    c2.metric("People", f"{len(df):,}")
    c3.metric("Coverage (all companies)", f"{coverage:.1%}",
              help="Share of people (excluding the head of each organisation) who got a "
                   "manager")
    if show_truth:
        c4.metric("Precision (all companies)", f"{precision:.1%}",
                  help="Of the reporting lines drawn, the share that match the answer key")

    st.subheader("Companies")
    everyone = apply_threshold(df, threshold)
    g = everyone.groupby("company_id")
    gs = s.groupby("company_id")
    cov = gs.given.mean() if show_truth else \
        everyone[~everyone.is_top.astype(bool)].groupby("company_id").given.mean()
    table = companies.set_index("company_id").join(pd.DataFrame({
        "employees": df.groupby("company_id").size(),
        "lines": g.given.sum(),
        "coverage": 100 * cov,
        "precision": 100 * gs.right.sum() / gs.given.sum().clip(lower=1),
    })).reset_index(drop=True)
    columns = ["company_name", "industry", "employees", "lines", "coverage"] + \
        (["precision"] if show_truth else [])
    st.dataframe(
        table, key=table_key, on_select="rerun", selection_mode="single-cell",
        hide_index=True, width="stretch", column_order=columns,
        height=min(560, 35 * (len(table) + 1) + 3),     # fit the rows: no empty lines
        column_config={
            "company_name": st.column_config.TextColumn(
                "Company", help="Click a company's name to open its org chart"),
            "industry": st.column_config.TextColumn("Industry"),
            "employees": st.column_config.NumberColumn("Employees", format="%d"),
            "lines": st.column_config.NumberColumn("Lines drawn", format="%d"),
            "coverage": st.column_config.NumberColumn("Coverage", format="%.0f%%"),
            "precision": st.column_config.NumberColumn("Precision", format="%.1f%%"),
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

# ---------------------------------------------------------------- headline numbers
info = co.iloc[0]
coverage, precision = measure(df[df.company_id == cid], scored[scored.company_id == cid],
                              threshold)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Employees", len(co), help="People listed for this company")
c2.metric("Industry", info.industry)
c3.metric("Coverage", f"{coverage:.0%}",
          help="Share of people (excluding the head) who got a manager at this threshold")
if show_truth:
    c4.metric("Precision", f"{precision:.1%}",
              help="Of the reporting lines drawn, the share that match the answer key")
sources = co.source_url.dropna().unique() if "source_url" in co else []
if len(sources):
    st.caption("Source: " + " · ".join(f"[{u}]({u})" for u in sources))

# ---------------------------------------------------------------- the org chart
view = co if domain == "All" else co[(co.domain == domain) | co.is_top.astype(bool)]
st.subheader(f"{info.company_name} — {domain}")
# Our own HTML: names and titles are escaped inside org_chart.py before display
if not view.is_top.astype(bool).any() and not view.given.any():
    # Nothing to draw: say why instead of showing an empty chart
    others = [j for j in judges if j != judge]
    with st.container(border=True):
        st.markdown(f"**Nothing could be placed in this view.** The {JUDGES[judge]} judge found "
                    "no head of the organisation and drew no reporting lines"
                    + (" for this domain. Try **All** domains." if domain != "All" else
                       f" at confidence ≥ {threshold:.2f}. It may not be able to read these "
                       "titles, or it isn't confident enough: try a lower threshold"
                       + (" or another judge." if others else ".")))
        if "claude" in others:
            st.button("Switch to Claude", type="primary",
                      on_click=lambda: ss.update(judge_select="claude"))
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

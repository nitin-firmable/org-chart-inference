"""
app.py  -  Step 7
Streamlit app. Pick a dataset (synthetic, or real: UK government, ASX/NZX leadership) and a
judge (rules, Claude, Laya). Front page: overall results, judges compared, and a searchable,
sortable table of companies. Click a row (or use the sidebar) to open that company's org chart.
Run locally from the repo root with:  streamlit run app/app.py

Real-data results come from app/results/ (written by `python -m harness.export`), because the
hosted app can't run the model judges itself. Synthetic data is rebuilt here when missing.
"""
import sys
import threading
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))          # Streamlit only puts app/ on the path; we need the root
from app import org_chart  # noqa: E402

SYNTHETIC = ROOT / "data/synthetic/output/seed42"
RESULTS = ROOT / "app/results"
DATASETS = {"synthetic": "Synthetic (500 companies)", "uk": "UK government (real)",
            "asx": "ASX/NZX leadership (real)"}
ABOUT = {
    "synthetic": "Generated companies with a hidden answer key (seed 42).",
    "uk": "Senior civil servants from data.gov.uk organograms (Open Government Licence). "
          "Each post's real 'reports to' line is the answer key.",
    "asx": "Executive teams copied from each company's own leadership page (source on each "
           "company page). No answer key: only 2 managers are publicly stated.",
}
HAS_TRUTH = {"synthetic": True, "uk": True, "asx": False}
JUDGES = {"claude": "Claude", "rules": "Rules", "laya_staged": "Laya (staged)", "laya": "Laya"}
DEFAULT_THRESHOLD = 0.8                # wrong line = 4x a missing line

st.set_page_config(page_title="Org Chart Inference", layout="wide")


@st.cache_resource
def build_lock():
    return threading.Lock()        # one lock shared by every visitor's session


def is_stale():
    # Missing files, or files made by an older version of the code
    return (not (SYNTHETIC / "predictions.csv").exists()
            or not (SYNTHETIC / "truth.csv").exists()
            or "scenario" not in pd.read_csv(SYNTHETIC / "truth.csv", nrows=1).columns
            or "rule_confidence" not in pd.read_csv(SYNTHETIC / "predictions.csv",
                                                    nrows=1).columns)


def judges_for(data):
    if data == "synthetic":
        return ["rules"]
    return [j for j in JUDGES if (RESULTS / data / f"{j}.csv").exists()]


@st.cache_data
def load(data, judge):
    """Returns (people, scored): everyone with the judge's answer, and the people who have
    a known manager to find (built like harness/evaluate.py, so the numbers match)."""
    if data != "synthetic":
        people = pd.read_csv(RESULTS / data / f"{judge}.csv")
        scored = people[people.true_manager_id.notna()][
            ["company_id", "employee_id", "true_manager_id", "pred_manager_id", "confidence"]]
        return people, scored
    # On a fresh server (e.g. Streamlit Cloud) the CSVs don't exist yet: build them once.
    # The lock stops two visitors from rebuilding at the same time; the second one waits,
    # then finds fresh files and skips the rebuild.
    with build_lock():
        if is_stale():
            from data.synthetic import generate as generate_data
            from judge import rules as infer
            generate_data.main(out=SYNTHETIC)
            infer.main(SYNTHETIC)
    pred = pd.read_csv(SYNTHETIC / "predictions.csv")
    truth = pd.read_csv(SYNTHETIC / "truth.csv")
    pred["domain"], pred["is_top"] = pred.function, pred.level == 1
    scored = truth[truth.in_dataset.astype(bool) & truth.true_manager_id.notna()][
        ["company_id", "employee_id", "true_manager_id"]].merge(
        pred[["employee_id", "pred_manager_id", "confidence"]], on="employee_id", how="left")
    people = pred.merge(truth[["employee_id", "true_manager_id"]], on="employee_id", how="left")
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
    return p[~p.is_top.astype(bool)].given.mean(), None


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
# Fixed order: the table's selected row number is looked up in this list
companies = (df[["company_id", "company_name", "industry"]].drop_duplicates("company_id")
             .sort_values("company_name").reset_index(drop=True))
company_label = dict(zip(companies.company_id,
                         companies.company_name + " (" + companies.company_id + ")"))

# ---------------------------------------------------------------- which company is open?
# One source of truth: ss.company_select (also the sidebar dropdown's value).
# Table clicks and the Back button write to it; the URL (?company=C001) mirrors it.
if "company_select" not in ss:
    from_url = st.query_params.get("company", "")
    ss.company_select = from_url if from_url in company_label else ""
if ss.company_select not in company_label:
    ss.company_select = ""
table_key = f"company_table_{ss.table_version}"
clicked = ss.get(table_key)
if clicked and clicked["selection"]["rows"]:
    ss.company_select = companies.company_id.iloc[clicked["selection"]["rows"][0]]
    ss.table_version += 1          # fresh table next time, so the old click doesn't stick


def back_to_all():
    ss.company_select = ""
    ss.table_version += 1


# ---------------------------------------------------------------- sidebar
st.sidebar.selectbox("Company", [""] + companies.company_id.tolist(), key="company_select",
                     format_func=lambda c: company_label.get(c, "All companies"))
threshold = st.sidebar.slider(
    "Confidence threshold", 0.5, 1.0, DEFAULT_THRESHOLD, 0.05,
    help=("Confidence comes from a decision tree trained on a separate dataset."
          if judge == "rules" else
          "The model's confidence, calibrated on the UK answer key (each organisation by a "
          "map fitted on the others).") +
    " Default 0.8: a wrong line is treated as 4x worse than a missing one.")
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

# ================================================================ front page: all companies
if not cid:
    st.write("Pick a company in the table to see its org chart. Search with the magnifying "
             "glass (top right of the table), sort by clicking a column header.")
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

    if len(judges) > 1:
        st.subheader("Judges compared")
        rows = []
        for j in judges:
            people, sc = load(data, j)
            drawn = apply_threshold(people, threshold).given.sum()
            cov, prec = measure(people, sc, threshold)
            rows.append({"Judge": JUDGES[j], "Lines drawn": int(drawn),
                         "Coverage": 100 * cov, "Precision": 100 * prec if show_truth else None})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="content",
                     column_order=["Judge", "Lines drawn", "Coverage"] +
                     (["Precision"] if show_truth else []),
                     column_config={
                         "Coverage": st.column_config.NumberColumn(format="%.1f%%"),
                         "Precision": st.column_config.NumberColumn(format="%.1f%%")})

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
        table, key=table_key, on_select="rerun", selection_mode="single-row",
        hide_index=True, width="stretch", height=560, column_order=columns,
        column_config={
            "company_name": st.column_config.TextColumn("Company"),
            "industry": st.column_config.TextColumn("Industry"),
            "employees": st.column_config.NumberColumn("Employees", format="%d"),
            "lines": st.column_config.NumberColumn("Lines drawn", format="%d"),
            "coverage": st.column_config.NumberColumn("Coverage", format="%.0f%%"),
            "precision": st.column_config.NumberColumn("Precision", format="%.1f%%"),
        })
    st.stop()

# ================================================================ company page
st.button("← Back to all companies", on_click=back_to_all)
co = apply_threshold(df[df.company_id == cid], threshold)
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
st.dataframe(co[co.manager_id.isna() & ~co.is_top.astype(bool)][
    ["full_name", "title", "department", "reason"]], width="stretch", hide_index=True)

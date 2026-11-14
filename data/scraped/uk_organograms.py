"""
Real data: UK government organograms (data.gov.uk, Open Government Licence).

Each organisation publishes its senior posts with name, job title, unit and the post they
report to. That last column is a real answer key, so the model can be scored on real people.
Output has the same columns as the synthetic data, so the judges and harness run unchanged:

    python -m data.scraped.uk_organograms        # -> data/scraped/output/uk/
    python -m judge.rules data/scraped/output/uk
    python -m harness.evaluate data/scraped/output/uk

Only senior posts are named (the "junior" files are headcounts), so each organisation has
tens to hundreds of people, 3-5 levels deep. The files used are listed in sources.csv.
"""
import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

API = "https://ckan.publishing.service.gov.uk/api/action/package_show?id="
OUT = Path("data/scraped/output/uk")
MIN_POSTS = 20                  # smaller files are too thin to say anything
MINISTERIAL, NON_MINISTERIAL = "Ministerial department", "Non-ministerial department"
ORG_TYPE = {                    # GOV.UK's classification of each organisation
    "Cabinet Office": MINISTERIAL, "HM Treasury": MINISTERIAL, "Home Office": MINISTERIAL,
    "Department for Education": MINISTERIAL, "Ministry of Justice": MINISTERIAL,
    "Department for Digital, Culture, Media and Sport": MINISTERIAL,
    "Department for Environment, Food and Rural Affairs": MINISTERIAL,
    "HM Revenue and Customs": NON_MINISTERIAL, "HM Land Registry": NON_MINISTERIAL,
    "Government Legal Department": NON_MINISTERIAL, "Crown Prosecution Service": NON_MINISTERIAL,
    "Environment Agency": "Executive non-departmental public body",
    "High Speed 2 Limited": "Executive non-departmental public body",
}
DATASETS = [
    "organogram-cabinet-office", "organogram-hm-treasury", "organogram-home-office",
    "organogram-department-for-education", "organogram-ministry-of-justice",
    "organogram-hm-revenue-and-customs", "organogram-environment-agency",
    "organogram-land-registry", "organogram-government-legal-department",
    "organogram-department-for-culture-media-and-sport", "organogram-high-speed-2",
    "organogram-crown-prosecution-service", "organogram-care-quality-commission",
    "organogram-department-for-environment-food-and-rural-affairs",
    "organogram-ministry-of-defence", "organogram-hm-passport-office",
]
HIDDEN = {"", "N/D", "N/A", "NA", "REDACTED", "NOT DISCLOSED", "-"}
EMPTY_POST = {"VACANT", "VACANCY", "ELIMINATED", "VACANT POST"}


def fetch(url, cache):
    # Download once, then reuse the local copy (data.gov.uk is slow and files don't change)
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(urllib.request.urlopen(url, timeout=120).read())
    return cache.read_bytes()


def latest_senior_file(dataset):
    meta = json.loads(fetch(API + dataset, OUT / "raw" / f"{dataset}.json"))["result"]
    files = [r for r in meta["resources"]
             if "senior" in (r.get("name") or "").lower() and r.get("url")]
    if not files:
        return None, None
    publisher = (meta.get("organization") or {}).get("title") or dataset[11:]
    return publisher, max(files, key=lambda r: r.get("created", ""))


def clean(value):
    value = (value or "").strip()
    return None if value.upper() in HIDDEN else value


def convert(cid, company_name, rows):
    """One organogram -> people rows with the true manager, in the synthetic data's columns."""
    rows = [r for r in rows if (r.get("Post Unique Reference") or "").strip()]
    seen, ids = {}, []
    for r in rows:                      # job shares repeat a post reference: keep both people
        ref = r["Post Unique Reference"].strip()
        seen[ref] = seen.get(ref, 0) + 1
        ids.append(f"{cid}-{ref}" + (f"-{seen[ref]}" if seen[ref] > 1 else ""))
    post_id = {}                        # a post's first holder stands in as "the manager"
    for r, pid in zip(rows, ids):
        post_id.setdefault(r["Post Unique Reference"].strip(), pid)

    people = []
    for r, pid in zip(rows, ids):
        name = (r.get("Name") or "").strip()
        title = clean(r.get("Job Title"))
        empty = name.upper() in EMPTY_POST or (title or "").upper() == "NOT IN POST"
        boss = post_id.get((r.get("Reports to Senior Post") or "").strip())
        people.append(dict(
            company_id=cid, company_name=company_name,
            industry=ORG_TYPE.get(company_name, "Public sector"),
            size=len(rows), employee_id=pid, full_name=None if empty else clean(name),
            title=None if empty else title, department=clean(r.get("Unit")),
            location=clean(r.get("Office Region")), grade=clean(r.get("Grade (or equivalent)")),
            true_manager_id=boss if boss != pid else None,
            in_dataset=not empty,       # vacant posts have nobody to find
            dotted_manager_id=None))
    for p in people:                    # undisclosed names must stay unique (judge.rules dedupes)
        p["full_name"] = p["full_name"] or f"Undisclosed ({p['employee_id']})"

    by_id = {p["employee_id"]: p for p in people}
    for p in people:
        depth, cur = 1, p               # level = steps below the top post (top = 1)
        while cur["true_manager_id"] in by_id and depth < 20:
            depth, cur = depth + 1, by_id[cur["true_manager_id"]]
        p["true_level"] = depth
        boss = by_id.get(p["true_manager_id"])
        p["scenario"] = ("manager_missing" if boss and not boss["in_dataset"] else
                         "title_missing" if not p["title"] else "normal")
    return people


def main(out=OUT):
    out = Path(out)
    people, sources = [], []
    for dataset in DATASETS:
        publisher, file = latest_senior_file(dataset)
        if file is None:
            print(f"skip {dataset}: no senior file")
            continue
        raw = fetch(file["url"], out / "raw" / f"{dataset}-senior.csv")
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig", "replace"))))
        if len(rows) < MIN_POSTS:
            print(f"skip {dataset}: only {len(rows)} posts")
            continue
        cid = f"UK{len(sources) + 1:02d}"
        people.extend(convert(cid, publisher, rows))
        sources.append(dict(company_id=cid, company_name=publisher, dataset=dataset,
                            published=file.get("created", "")[:10], posts=len(rows),
                            url=file["url"]))
        print(f"{cid} {publisher}: {len(rows)} posts ({file.get('created', '')[:10]})")

    truth = pd.DataFrame(people)
    truth.to_csv(out / "truth.csv", index=False)
    keep = ["company_id", "company_name", "industry", "size", "employee_id",
            "full_name", "title", "department", "location"]
    truth[truth.in_dataset][keep].to_csv(out / "employees.csv", index=False)
    pd.DataFrame(sources).to_csv(out / "sources.csv", index=False)
    print(f"{len(sources)} organisations, {truth.in_dataset.sum()} people in employees.csv, "
          f"{(~truth.in_dataset).sum()} vacant posts")
    print("scenarios:", truth[truth.in_dataset].scenario.value_counts().to_dict())


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else OUT)

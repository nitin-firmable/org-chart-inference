"""
Real data: ASX/NZX leadership teams, copied from each company's own website.

asx_leadership.csv (next to this file) is the hand-checked sheet (company_name, full_name, title,
reports_to, evidence, source_url). A manager is filled in only when the source says so;
everyone else has no answer key and isn't scored. Same output columns as the synthetic data:

    python -m data.scraped.asx_leadership        # -> data/scraped/output/asx/
    python -m judge.rules data/scraped/output/asx
"""
import sys
from pathlib import Path

import pandas as pd

SHEET = Path(__file__).parent / "asx_leadership.csv"
OUT = Path("data/scraped/output/asx")
INDUSTRY = {                            # sector of each company (GICS-style, kept short)
    "Commonwealth Bank of Australia": "Banking", "Fisher & Paykel Healthcare": "Medical devices",
    "Fletcher Building": "Building materials", "Origin Energy": "Energy",
    "Qantas": "Airlines", "SEEK": "Online marketplaces", "Telstra": "Telecommunications",
    "Wesfarmers": "Conglomerate (retail)", "WiseTech Global": "Software (logistics)",
    "Woolworths Group": "Retail (supermarkets)", "Xero": "Software (accounting)",
}


def main(out=OUT):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    sheet = pd.read_csv(SHEET, dtype=str).fillna("")
    rows = []
    for i, (name, co) in enumerate(sheet.groupby("company_name", sort=True), start=1):
        cid = f"AU{i:02d}"
        ids = {p.full_name.strip(): f"{cid}-E{j:03d}" for j, p in enumerate(co.itertuples(), 1)}
        for p in co.itertuples():
            boss = p.reports_to.strip()
            if boss and boss not in ids:
                print(f"warning: {name}: '{boss}' (manager of {p.full_name}) is not in the sheet")
            rows.append(dict(
                company_id=cid, company_name=name, industry=INDUSTRY.get(name, "Listed company"),
                size=len(co),
                employee_id=ids[p.full_name.strip()], full_name=p.full_name.strip(),
                title=p.title.strip(), department=None, location=None,
                true_manager_id=ids.get(boss), true_level=2 if boss else 1,
                in_dataset=True, scenario="stated" if boss else "unknown",
                dotted_manager_id=None, evidence=p.evidence, source_url=p.source_url))

    truth = pd.DataFrame(rows)
    truth.to_csv(out / "truth.csv", index=False)
    keep = ["company_id", "company_name", "industry", "size", "employee_id",
            "full_name", "title", "department", "location"]
    truth[keep].to_csv(out / "employees.csv", index=False)
    print(f"{truth.company_id.nunique()} companies, {len(truth)} people, "
          f"{truth.true_manager_id.notna().sum()} with a stated manager")


if __name__ == "__main__":
    main(sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else OUT)

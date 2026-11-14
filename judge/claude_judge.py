"""
judge/claude_judge.py  -  Claude (via the Claude Code CLI) as the judge

Runs `claude -p` headless: no tools, a short judge-only system prompt, and a JSON schema so
every answer is structured. Uses whatever Claude Code is logged in with (a subscription
uses the plan's allowance; no separate API bill).

Per company, Claude sees EVERYONE (id, title, unit) and judges each person in stages:
level (1-10) -> domain -> manager among same-domain people at the same or a higher level ->
otherwise someone more senior elsewhere, "TOP" or "NOT_LISTED". Big organisations are judged
in chunks of people, but every chunk still sees the whole organisation.

    python -m judge.claude_judge data/scraped/output/uk --companies UK02 UK10
    python -m judge.claude_judge data/scraped/output/asx --model opus
    python -m judge.claude_judge data/scraped/output/uk --wait     # sleeps through usage limits

Resumable: answers are saved after every call and anyone already answered is skipped, so
rerunning the same command continues where it stopped. If the plan's usage limit is hit it
stops cleanly (or, with --wait, retries every 30 minutes until the limit resets).

Its confidence is self-reported, so the harness calibrates it on the UK answer key.
"""
import json
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from judge.decisions import NOT_LISTED, TOP, label

MODEL = "sonnet"
CHUNK = 80                      # people judged per call (each call sees the whole organisation)
WORKERS = 3                     # calls in parallel
WAIT_MINUTES = 30               # --wait: how often to retry after the plan's usage limit

SYSTEM = f"""You judge organisation charts. You get everyone in one organisation (id | title |
unit) and a list of ids to judge. For each id, decide in stages:
1. level: 1 = head of the whole organisation ... 10 = intern, judged in THIS organisation's
   own terms (e.g. Permanent Secretary > Director General > Director > Deputy Director).
2. domain: the area they work in (their unit, division or function).
3. manager_id: their direct line manager. Look first among people in the same domain at the
   same or a higher level; if none fits, someone more senior elsewhere. Use "{TOP}" if they
   head the organisation, "{NOT_LISTED}" if their manager is clearly not in the list.
4. confidence: probability (0-1) that manager_id is exactly right. Many people share a
   title, so be honest: guessing between look-alikes is low confidence.
5. runner_up: your second choice for manager_id ("" if none).
Answer only in the requested JSON, one entry per id you were asked about."""

SCHEMA = {"type": "object", "required": ["people"], "properties": {"people": {
    "type": "array", "items": {"type": "object",
                               "required": ["id", "level", "domain", "manager_id",
                                            "confidence", "runner_up"],
                               "properties": {"id": {"type": "string"},
                                              "level": {"type": "integer", "minimum": 1,
                                                        "maximum": 10},
                                              "domain": {"type": "string"},
                                              "manager_id": {"type": "string"},
                                              "confidence": {"type": "number", "minimum": 0,
                                                             "maximum": 1},
                                              "runner_up": {"type": "string"}}}}}}


class UsageLimit(Exception):
    """The plan's usage limit was hit: stop, and continue after it resets."""


def call_claude(prompt, model):
    cmd = ["claude", "-p", "--system-prompt", SYSTEM, "--setting-sources", "",
           "--strict-mcp-config", "--no-session-persistence", "--tools", "",
           "--output-format", "json", "--model", model, "--json-schema", json.dumps(SCHEMA)]
    res = subprocess.run(cmd, input=prompt, capture_output=True, text=True, timeout=900,
                         cwd=Path(__file__).parent)
    try:
        out = json.loads(res.stdout)
    except json.JSONDecodeError:
        out = {"is_error": True, "result": res.stdout + res.stderr}
    message = str(out.get("result") or res.stderr)
    if out.get("is_error") and (out.get("api_error_status") == 429 or
                                re.search(r"usage limit|rate limit|limit reached|resets",
                                          message, re.IGNORECASE)):
        raise UsageLimit(message[:200])
    if out.get("is_error") or not out.get("structured_output"):
        raise RuntimeError(f"claude failed: {message[:300]}")
    return out["structured_output"]["people"], out.get("usage", {})


def judge_chunk(co, ids, model):
    roster = "\n".join(f"{p.employee_id} | {label(p)}" for p in co.itertuples())
    prompt = (f"Organisation: {co.company_name.iloc[0]} ({len(co)} people)\n\n{roster}\n\n"
              f"Judge these ids: {', '.join(ids)}")
    valid, got, usage = set(co.employee_id) | {TOP, NOT_LISTED}, {}, {}
    for attempt in range(2):                 # one retry if ids are missing or the call fails
        try:
            people, usage = call_claude(prompt, model)
        except (RuntimeError, subprocess.TimeoutExpired) as exc:     # UsageLimit goes up
            print(f"  retrying {co.company_id.iloc[0]}: {exc}")
            continue
        got = {p["id"]: p for p in people if p["id"] in ids}
        if len(got) == len(ids) or attempt == 1:
            break
    rows = []
    for pid in ids:
        p = got.get(pid)
        if p is None:
            continue                          # left out: the harness treats it as no answer
        choice = p["manager_id"] if p["manager_id"] in valid and p["manager_id"] != pid \
            else NOT_LISTED
        rows.append(dict(employee_id=pid, choice=choice, raw_confidence=p["confidence"],
                         runner_up=p["runner_up"], level=p["level"], domain=p["domain"]))
    return rows, usage


def run_once(data_dir, companies, model):
    """Answers everyone not yet in raw.csv. Returns False if the usage limit stopped it."""
    out = data_dir / "runs" / "claude"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(data_dir / "employees.csv")
    if companies:
        df = df[df.company_id.isin(companies)]
    done = pd.read_csv(out / "raw.csv") if (out / "raw.csv").exists() else pd.DataFrame()
    have = set(done.get("employee_id", []))     # already answered: never asked again

    jobs = []
    for cid, co in df.groupby("company_id"):
        ids = [i for i in co.employee_id if i not in have]
        jobs += [(co, ids[s:s + CHUNK]) for s in range(0, len(ids), CHUNK)]
    print(f"{len(have & set(df.employee_id))} of {len(df)} people already answered; "
          f"{len(jobs)} calls to Claude ({model})")
    limit = None
    with ThreadPoolExecutor(WORKERS) as pool:
        futures = [pool.submit(judge_chunk, co, ids, model) for co, ids in jobs]
        for (co, ids), fut in zip(jobs, futures):
            if fut.cancelled():
                continue
            try:
                rows, usage = fut.result()
            except UsageLimit as exc:
                limit = limit or exc
                pool.shutdown(wait=False, cancel_futures=True)   # don't start calls that fail
                continue                     # but still save calls that already finished
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_csv(out / "raw.csv", index=False)    # saved per call: safe to stop
            print(f"{co.company_id.iloc[0]} {co.company_name.iloc[0]}: {len(rows)}/{len(ids)} "
                  f"answered, {usage.get('output_tokens', 0)} output tokens")
    if limit:
        print(f"Usage limit reached: {limit}")
        return False
    print(f"{len(done)} answers in {out / 'raw.csv'}")
    return True


def main(data_dir, companies=None, model=MODEL, wait=False):
    """Run until everyone is answered. With wait=True, sleep through usage limits."""
    while not run_once(Path(data_dir), companies, model):
        if not wait:
            print("Progress is saved. Run the same command again after the limit resets, "
                  "or add --wait to keep going on its own.")
            sys.exit(2)
        print(f"Waiting {WAIT_MINUTES} min, then continuing "
              f"(from {time.strftime('%H:%M')}; Ctrl+C to stop, progress is saved)")
        time.sleep(WAIT_MINUTES * 60)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    picked = None
    if "--companies" in sys.argv:
        rest = sys.argv[sys.argv.index("--companies") + 1:]
        picked = [c for c in rest[:next((i for i, a in enumerate(rest) if a.startswith("--")),
                                         len(rest))]]
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else MODEL
    main(args[0] if args else "data/scraped/output/asx", picked, model,
         wait="--wait" in sys.argv)

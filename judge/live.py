"""
judge/live.py  -  place a few people inside one company, live, for the app's "What if…" panel

Only the changed people are judged (new people, or the direct reports of someone who left),
each against everyone else in the same company. Three backends, same answer shape:

  gemini  Google Gemini, free key in .streamlit/secrets.toml or the environment (GEMINI_API_KEY)
  claude  Claude Code CLI on this machine (`claude -p`, the logged-in subscription); not
          available on Streamlit Cloud
  rules   judge/rules.py; always available, weak on real-world titles

Answer per person: {"choice": employee_id | TOP | NOT_LISTED, "confidence", "level", "domain"}.
"""
import json
import os
import shutil
import tomllib
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from judge import claude_judge, rules
from judge.decisions import NOT_LISTED, TOP, label
from harness import calibrate

SECRETS = Path(__file__).parent.parent / ".streamlit" / "secrets.toml"
GEMINI_MODELS = ["gemini-flash-lite-latest", "gemini-2.5-flash"]   # second one if the first is busy
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"
LABELS = {"gemini": "Gemini (free key)", "claude": "Claude (this Mac's subscription)",
          "rules": "Rules"}


class LiveJudgeError(Exception):
    """The live judge could not answer (no key, model busy, bad output)."""


def gemini_key(secrets=None):
    if secrets is not None and "GEMINI_API_KEY" in secrets:
        return secrets["GEMINI_API_KEY"]
    if os.environ.get("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    if SECRETS.exists():
        return tomllib.loads(SECRETS.read_text()).get("GEMINI_API_KEY")
    return None


def available(secrets=None):
    """Backends that can run here, best first."""
    found = []
    if gemini_key(secrets):
        found.append("gemini")
    if shutil.which("claude"):
        found.append("claude")
    return found + ["rules"]


def prompt_for(company_name, people, targets):
    roster = "\n".join(f"{p.employee_id} | {label(p)}" for p in people.itertuples())
    return (f"Organisation: {company_name} ({len(people)} people)\n\n{roster}\n\n"
            f"Judge these ids: {', '.join(targets)}")


def ask_gemini(prompt, key):
    body = {"systemInstruction": {"parts": [{"text": claude_judge.SYSTEM}]},
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json",
                                 "responseJsonSchema": claude_judge.SCHEMA}}
    last = None
    for model in GEMINI_MODELS:
        req = urllib.request.Request(GEMINI_URL.format(model), data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json",
                                              "X-goog-api-key": key})
        try:
            out = json.load(urllib.request.urlopen(req, timeout=90))
            text = out["candidates"][0]["content"]["parts"][0]["text"]
            return json.loads(text)["people"]
        except urllib.error.HTTPError as exc:
            last = f"{model}: HTTP {exc.code}"
            if exc.code not in (429, 500, 503):          # busy or rate-limited: try the next
                break
        except (KeyError, IndexError, json.JSONDecodeError, TimeoutError) as exc:
            last = f"{model}: {type(exc).__name__}"
    raise LiveJudgeError(f"Gemini didn't answer ({last}). Try again in a minute.")


def ask_rules(people, targets):
    co = people.copy()
    parsed = co.apply(lambda r: rules.parse_title(r.title, r.department), axis=1)
    co[["level", "function", "team", "is_contractor"]] = pd.DataFrame(parsed.tolist(),
                                                                      index=co.index)
    for col in ["team", "function"]:
        co[col] = co[col].astype(object).where(co[col].notna(), None)
    by_id = {r["employee_id"]: r for r in rules.infer_company(co)}
    info = co.set_index("employee_id")
    out = []
    for pid in targets:
        r = by_id[pid]
        top = info.level.get(pid) == 1
        out.append({"id": pid, "manager_id": TOP if top else (r["pred_manager_id"] or NOT_LISTED),
                    "confidence": r["rule_confidence"], "level": info.level.get(pid),
                    "domain": info.function.get(pid) or "", "reason": r["reason"]})
    return out


def judge(backend, company_name, people, targets, secrets=None):
    """people: everyone currently in the company (incl. the targets). Returns {id: answer}."""
    targets = [t for t in targets if t in set(people.employee_id)]
    if not targets:
        return {}
    if backend == "rules":
        answers = ask_rules(people, targets)
    elif backend == "gemini":
        key = gemini_key(secrets)
        if not key:
            raise LiveJudgeError("No GEMINI_API_KEY found.")
        answers = ask_gemini(prompt_for(company_name, people, targets), key)
    else:
        try:
            answers, _ = claude_judge.call_claude(prompt_for(company_name, people, targets),
                                                  claude_judge.MODEL)
        except (RuntimeError, claude_judge.UsageLimit) as exc:
            raise LiveJudgeError(f"Claude didn't answer: {exc}") from None
    valid = set(people.employee_id) | {TOP, NOT_LISTED}
    mapping = calibrate.load("claude") if backend == "claude" else None
    result = {}
    for a in answers:
        pid = a.get("id")
        if pid not in targets:
            continue
        choice = a.get("manager_id") if a.get("manager_id") in valid else NOT_LISTED
        if choice == pid:
            choice = NOT_LISTED
        conf = float(a.get("confidence") or 0)
        if mapping:                                  # Claude: same calibration as the harness
            conf = float(calibrate.apply(mapping, [conf])[0])
        result[pid] = {"choice": choice, "confidence": round(conf, 3),
                       "level": a.get("level"), "domain": a.get("domain") or "",
                       "reason": a.get("reason") or ""}
    return result

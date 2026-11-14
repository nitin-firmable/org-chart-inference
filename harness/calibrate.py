"""
harness/calibrate.py  -  turn a model's raw confidence into an honest one

A decision model's "0.25" can mean "right 80% of the time" (Laya spreads probability over
many options). Calibration learns that mapping from people whose real manager we know,
with isotonic regression: higher raw confidence never maps to lower calibrated confidence.

Fitted on the UK answer key only. Inside the UK data each organisation is calibrated by a
map fitted on the OTHER organisations, so no organisation grades its own homework.
"""
import json
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

MODEL_DIR = Path(__file__).parent.parent / "models" / "calibration"


def fit(raw, correct):
    """Returns a map as two lists of points (raw -> calibrated), stored as plain JSON."""
    iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso.fit(np.asarray(raw, float), np.asarray(correct, float))
    return {"x": [round(float(v), 4) for v in iso.X_thresholds_],
            "y": [round(float(v), 4) for v in iso.y_thresholds_], "n": int(len(raw))}


def apply(mapping, raw):
    if not mapping or not mapping["x"]:
        return np.zeros(len(raw))
    return np.interp(np.asarray(raw, float), mapping["x"], mapping["y"])


def leave_one_company_out(raw, correct, company):
    """Calibrated confidence for each row, from a map fitted on every other company."""
    raw, correct, company = map(np.asarray, (raw, correct, company))
    out = np.zeros(len(raw))
    for c in np.unique(company):
        mine = company == c
        if (~mine).sum() == 0:
            continue                    # only one company: nothing to learn from, stay at 0
        out[mine] = apply(fit(raw[~mine], correct[~mine]), raw[mine])
    return out


def save(name, mapping):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / f"{name}.json").write_text(json.dumps(mapping, indent=1))


def load(name):
    path = MODEL_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None

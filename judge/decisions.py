"""
judge/decisions.py  -  the decision format every model-based judge speaks

A decision model reads a STATE and answers typed QUESTIONS with probabilities, never text
(the format of TypeSafe's Jev; Laya, which we run locally for free, uses the same one):

    choice  one of a set of options      -> {"choice": "C01-E007", "probabilities": {...}}
    score   a level on an ordered scale  -> {"score": 2.0, "probabilities": {...}}
    noul    yes / no                     -> {"noul": 0.86}   (= P(yes))

The judges below build these questions; the model behind them can be swapped.
"""
TOP = "TOP"                    # option: heads the whole organisation
NOT_LISTED = "NOT_LISTED"      # option: their manager isn't in the data (lets the model abstain)


def label(person):
    """How a person is shown to the model: title, plus unit when we have one."""
    title = person.title if isinstance(person.title, str) else "(no title)"
    unit = person.department if isinstance(person.department, str) else None
    return f"{title} | {unit}" if unit else title


def manager_question(person, colleagues):
    """Choice: who, among everyone else in the company, is this person's direct manager?"""
    criteria = {c.employee_id: label(c) for c in colleagues if c.employee_id != person.employee_id}
    criteria[TOP] = "Nobody here: this person heads the whole organisation"
    criteria[NOT_LISTED] = "Their direct manager is not in this list"
    return {"type": "choice",
            "instructions": f"Who does the {label(person)} report to directly?",
            "criteria": criteria}


LEVELS = ["head of the whole organisation (CEO, Permanent Secretary)",
          "member of the top executive team, reports to the head",
          "head of a large division or function", "director of an area within a division",
          "deputy director / head of a department", "senior manager", "manager / team lead",
          "senior individual contributor", "individual contributor", "junior, graduate or intern"]


def level_question(person):
    """Score, 1-10: how senior is this person in this organisation? (Score allows 10 levels)"""
    return {"type": "score",
            "instructions": f"How senior is the {label(person)} in this organisation?",
            "criteria": LEVELS}


AREAS = ["whole company / chief executive's office", "finance", "technology, data and digital",
         "people and HR", "legal, risk, governance and company secretariat",
         "corporate affairs, communications and sustainability", "strategy and transformation",
         "operations and supply chain", "sales, marketing and customers", "product",
         "a business division, brand or region"]


def domain_question(person):
    """Choice: which area of the company is this person in? (only when the data has no unit)"""
    return {"type": "choice",
            "instructions": f"Which area of the company does the {label(person)} work in?",
            "criteria": {a: a for a in AREAS}}


def manager_among(person, candidates, top=False):
    """Choice: which of these candidates is this person's direct manager, if any?"""
    criteria = {c.employee_id: label(c) for c in candidates}
    criteria[NOT_LISTED] = "None of these people is their direct manager"
    if top:
        criteria[TOP] = "Nobody here: this person heads the whole organisation"
    return {"type": "choice",
            "instructions": f"Who does the {label(person)} report to directly?",
            "criteria": criteria}


def reports_to_question(person, boss):
    """Noul: does this person report directly to that one?"""
    return {"type": "noul",
            "instructions": f"Does the {label(person)} report directly to the {label(boss)}?"}

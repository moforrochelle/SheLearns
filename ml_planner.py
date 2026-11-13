"""The machine learning planner.

It reads ml_model.json (made by train_model.py) and estimates, for each
subject, the chance that she will understand a session in it today.
The subject with the LOWEST chance is the one she needs most, so it is
ranked first.

The same code is used by the simulation and by the app.
"""
import json
import os
import math
from datetime import date, datetime

MODEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ml_model.json")
NO_EXAM_DAYS = 28   # used when a subject has no exam date
MAX_DAYS = 28       # the model was trained on a 28-day window

_model = None


def load_model():
    global _model
    if _model is None:
        with open(MODEL_FILE) as f:
            _model = json.load(f)
    return _model


def predict_understood(features):
    """Chance (0 to 1) that she understands the next session. features is a dict."""
    m = load_model()
    total = m["intercept"]
    for i, name in enumerate(m["features"]):
        z = (features[name] - m["mean"][i]) / m["scale"][i]
        total += m["coef"][i] * z
    return 1 / (1 + math.exp(-total))


def _as_date(value):
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def features_for(subject, sessions, today):
    """Build the model's inputs from the subject and her session log.

    sessions is a list of {"date": "YYYY-MM-DD", "subject": name, "outcome": ...}
    (the same list the app already saves). Skipped sessions are ignored.
    """
    mine = [s for s in sessions
            if s["subject"] == subject["name"] and s["outcome"] in ("understood", "confused")]
    understood = sum(1 for s in mine if s["outcome"] == "understood")
    n = len(mine)

    if mine:
        last = max(mine, key=lambda s: s["date"])
        days_since = min((today - _as_date(last["date"])).days, MAX_DAYS)
        last_outcome = 1 if last["outcome"] == "understood" else 0
    else:
        days_since = MAX_DAYS
        last_outcome = 0.5

    if subject["exam_date"] is None:
        days_to_exam = NO_EXAM_DAYS
    else:
        days_to_exam = min(max((subject["exam_date"] - today).days, 0), MAX_DAYS)

    return {
        "weakness": subject["weakness"],
        "days_to_exam": days_to_exam,
        "days_since_last": days_since,
        "never_studied": 0 if mine else 1,
        "n_sessions": n,
        "success_rate": (understood + 1) / (n + 2),
        "last_outcome": last_outcome,
    }


def ml_priority(subject, sessions, today=None):
    """Higher number = needs her more. This is 1 minus the predicted chance
    that she will understand, so the subject she is least likely to
    understand comes first."""
    today = today or date.today()
    return 1 - predict_understood(features_for(subject, sessions, today))
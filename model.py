from datetime import date

# Weights from the scoring model (they add up to 1.0)
W_WEAKNESS = 0.4
W_URGENCY = 0.4
W_RECENCY = 0.2

RECENCY_CAP_DAYS = 7   # after 7 days without studying, recency counts as 1


def make_subject(name, weakness, exam_date=None, last_studied=None):
    """A subject is just a dictionary."""
    return {
        "name": name,
        "weakness": weakness,          # how weak she feels in it (1-5)
        "exam_date": exam_date,        # a date, or None if there is no exam
        "last_studied": last_studied,  # a date, or None if never studied
    }


def days_until(exam_date):
    """Days left before the exam. 999 means no exam."""
    if exam_date is None:
        return 999
    delta = (exam_date - date.today()).days
    return max(delta, 0)


def days_since(last_studied):
    """Days since she last studied. None means never studied."""
    if last_studied is None:
        return None
    return max((date.today() - last_studied).days, 0)


def weakness_norm(subject):
    """Weakness 1-5 scaled to 0-1."""
    return subject["weakness"] / 5


def urgency_norm(subject):
    """Closer exam gives a higher number. 1 / (days + 1)."""
    return 1 / (days_until(subject["exam_date"]) + 1)


def recency_norm(subject):
    """Longer since last studied gives a higher number, capped at 1."""
    d = days_since(subject["last_studied"])
    if d is None:           # never studied: highest recency
        return 1.0
    return min(d / RECENCY_CAP_DAYS, 1.0)


def priority_score(subject):
    return (W_WEAKNESS * weakness_norm(subject)
            + W_URGENCY * urgency_norm(subject)
            + W_RECENCY * recency_norm(subject))


def apply_outcome(subject, outcome):
    """Learn from how a session went."""
    if outcome == "understood":
        subject["weakness"] = max(1, subject["weakness"] - 1)
        subject["last_studied"] = date.today()
    elif outcome == "confused":
        subject["weakness"] = min(5, subject["weakness"] + 1)
        subject["last_studied"] = date.today()
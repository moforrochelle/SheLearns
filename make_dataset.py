"""Make a synthetic dataset of study sessions for the machine learning model.

Each row is ONE study session. The columns are things the app can see
BEFORE the session starts. The last column (understood) is what happened.

The girl's hidden skill is used only to simulate what happens. It is never
written to the file, because the real app can never see it.

The data is synthetic: it comes from a simulation, not from real girls.

Run it with:  python3 make_dataset.py
"""
import csv
import random
from datetime import timedelta

import model
import simulate
from simulate import FakeDate, START, DAYS, NAMES, weakness_from_skill

OUTPUT_FILE = "sessions.csv"

# Three groups of girls. Girls are never shared between groups.
#   train   : the model learns from these girls
#   test    : new girls from the same kind of world as the training girls
#   shifted : new girls from a DIFFERENT world (slower learning, faster
#             forgetting). This checks that the model is not just
#             memorising the simulator's settings.
SPLITS = {
    "train":   {"girls": 1500, "gain": (0.08, 0.14), "forget": (0.000, 0.008), "seed": 100000},
    "test":    {"girls": 500,  "gain": (0.08, 0.14), "forget": (0.000, 0.008), "seed": 200000},
    "shifted": {"girls": 500,  "gain": (0.03, 0.06), "forget": (0.010, 0.020), "seed": 300000},
}

FEATURES = [
    "weakness",        # her own 1-5 rating, updated by the app after each session
    "days_to_exam",    # days left before this subject's exam
    "days_since_last", # days since she last studied it (28 if never)
    "never_studied",   # 1 if she has never studied this subject
    "n_sessions",      # sessions already done in this subject
    "success_rate",    # share of past sessions she understood (smoothed)
    "last_outcome",    # 1 = understood last time, 0 = confused, 0.5 = no history
]


def simulate_girl(split, seed, gain_range, forget_range):
    """Simulate one girl for 28 days and return one row per session."""
    rng = random.Random(seed)
    gain = rng.uniform(*gain_range)
    forget = rng.uniform(*forget_range)

    # Every girl is different: her own starting skills and exam days.
    skills = {n: rng.uniform(0.10, 0.80) for n in NAMES}
    exam_day = {n: rng.randint(10, 27) for n in NAMES}

    subjects = {
        n: model.make_subject(n, weakness_from_skill(skills[n]),
                              START + timedelta(days=exam_day[n]))
        for n in NAMES
    }
    history = {n: {"understood": 0, "confused": 0, "last_outcome": None,
                   "last_day": None} for n in NAMES}

    rows = []
    for day in range(DAYS):
        FakeDate.current = START + timedelta(days=day)

        remaining = [n for n in NAMES if exam_day[n] > day]
        if not remaining:
            break

        # Subjects are chosen at random, so the data covers every situation.
        # (If the planner chose, the data would only show what the planner
        # likes to pick.)
        name = rng.choice(remaining)
        h = history[name]
        subject = subjects[name]

        n_sessions = h["understood"] + h["confused"]
        days_since = None if h["last_day"] is None else day - h["last_day"]

        row = {
            "split": split,
            "girl_id": seed,
            "day": day,
            "subject": name,
            "weakness": subject["weakness"],
            "days_to_exam": exam_day[name] - day,
            "days_since_last": DAYS if days_since is None else days_since,
            "never_studied": 1 if days_since is None else 0,
            "n_sessions": n_sessions,
            "success_rate": round((h["understood"] + 1) / (n_sessions + 2), 4),
            "last_outcome": 0.5 if h["last_outcome"] is None else h["last_outcome"],
        }

        # What happens in the session (same rules as simulate.py)
        understood = rng.random() < skills[name]
        row["understood"] = 1 if understood else 0
        rows.append(row)

        for other in skills:
            if other != name:
                skills[other] = max(0.0, skills[other] - forget)
        skills[name] += gain * (1 - skills[name])

        # The app updates the weakness after every session, whatever the planner.
        model.apply_outcome(subject, "understood" if understood else "confused")

        if understood:
            h["understood"] += 1
        else:
            h["confused"] += 1
        h["last_outcome"] = 1 if understood else 0
        h["last_day"] = day

    return rows


def main():
    columns = ["split", "girl_id", "day", "subject"] + FEATURES + ["understood"]
    total = 0
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for split, cfg in SPLITS.items():
            rows = []
            for i in range(cfg["girls"]):
                rows += simulate_girl(split, cfg["seed"] + i, cfg["gain"], cfg["forget"])
            writer.writerows(rows)
            rate = sum(r["understood"] for r in rows) / len(rows)
            print(f"{split:<8} {cfg['girls']:>5} girls  {len(rows):>6} sessions  "
                  f"understood in {rate:.1%} of sessions")
            total += len(rows)
    print(f"\nSaved {total} sessions to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
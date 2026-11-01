import random
from datetime import date, timedelta

import model

# ---------- Settings (you can explain each of these in your defense) ----------
START = date(2026, 11, 1)   # pretend start date
DAYS = 28                   # 4 weeks, one study session per day
RUNS = 500                  # number of pretend girls
GAIN = 0.10                 # how much a session improves her skill
FORGET = 0.005              # how much she forgets per day in subjects not studied
PASS = 0.5                  # skill needed to count as "ready" for the exam

COURSES = [
    {"name": "Maths",   "skill": 0.30, "exam_day": 14},
    {"name": "Biology", "skill": 0.50, "exam_day": 21},
    {"name": "English", "skill": 0.70, "exam_day": 27},
]
NAMES = [c["name"] for c in COURSES]


# ---------- Pretend calendar ----------
# model.py asks "what is today's date?". We replace that answer with a
# pretend date, so the simulation can run 28 days in a few seconds.
class FakeDate(date):
    current = START

    @classmethod
    def today(cls):
        return cls.current


model.date = FakeDate


def weakness_from_skill(skill):
    """Turn a hidden skill (0 to 1) into the 1-5 weakness she would report."""
    return max(1, min(5, round(5 - 4 * skill)))


def run(strategy, seed, gain, forget):
    rng = random.Random(seed)
    skills = {c["name"]: c["skill"] for c in COURSES}   # hidden true skill
    exam_day = {c["name"]: c["exam_day"] for c in COURSES}
    counts = {n: 0 for n in NAMES}
    subjects = [
        model.make_subject(c["name"], weakness_from_skill(c["skill"]),
                           START + timedelta(days=c["exam_day"]))
        for c in COURSES
    ]
    readiness = {}

    for day in range(DAYS):
        FakeDate.current = START + timedelta(days=day)

        # Record how prepared she is on each exam day
        for name, d in exam_day.items():
            if day == d:
                readiness[name] = skills[name]

        # Only subjects whose exam has not happened yet
        remaining = [s for s in subjects if exam_day[s["name"]] > day]
        if not remaining:
            break

        if strategy == "adaptive":
            chosen = max(remaining, key=model.priority_score)
        else:
            chosen = remaining[day % len(remaining)]   # equal time for all

        name = chosen["name"]
        counts[name] += 1
        outcome = "understood" if rng.random() < skills[name] else "confused"

        # Forget a little in the other subjects, improve in the studied one
        for other in skills:
            if other != name:
                skills[other] = max(0.0, skills[other] - forget)
        skills[name] += gain * (1 - skills[name])

        # Only SheLearns learns from how the session went
        if strategy == "adaptive":
            model.apply_outcome(chosen, outcome)

    return readiness, counts


def evaluate(strategy, gain=GAIN, forget=FORGET):
    skill_sum = {n: 0.0 for n in NAMES}
    ready = {n: 0 for n in NAMES}
    sessions = {n: 0 for n in NAMES}
    weakest_sum = 0.0

    for seed in range(RUNS):
        readiness, counts = run(strategy, seed, gain, forget)
        for n in NAMES:
            skill_sum[n] += readiness[n]
            ready[n] += 1 if readiness[n] >= PASS else 0
            sessions[n] += counts[n]
        weakest_sum += min(readiness.values())

    return {
        "skill": {n: skill_sum[n] / RUNS for n in NAMES},
        "ready": {n: ready[n] / RUNS for n in NAMES},
        "sessions": {n: sessions[n] / RUNS for n in NAMES},
        "weakest": weakest_sum / RUNS,
    }


def mean(values):
    values = list(values)
    return sum(values) / len(values)


if __name__ == "__main__":
    equal = evaluate("equal")
    adaptive = evaluate("adaptive")

    print(f"Results for {RUNS} pretend girls over {DAYS} days\n")

    print("1. Average skill on exam day (0 to 1)")
    print(f"{'Subject':<10}{'Equal time':>12}{'SheLearns':>12}{'Change':>10}")
    for n in NAMES:
        e, a = equal["skill"][n], adaptive["skill"][n]
        print(f"{n:<10}{e:>12.3f}{a:>12.3f}{a - e:>+10.3f}")
    me, ma = mean(equal["skill"].values()), mean(adaptive["skill"].values())
    print(f"{'Overall':<10}{me:>12.3f}{ma:>12.3f}{ma - me:>+10.3f}")

    print(f"\n2. Share of girls ready (skill of {PASS} or more)")
    print(f"{'Subject':<10}{'Equal time':>12}{'SheLearns':>12}{'Change':>10}")
    for n in NAMES:
        e, a = equal["ready"][n], adaptive["ready"][n]
        print(f"{n:<10}{e:>12.0%}{a:>12.0%}{(a - e) * 100:>+9.0f}%")

    print("\n3. Average skill in her weakest subject")
    print(f"{'':<10}{equal['weakest']:>12.3f}{adaptive['weakest']:>12.3f}"
          f"{adaptive['weakest'] - equal['weakest']:>+10.3f}")

    print("\n4. Average number of sessions per subject")
    print(f"{'Subject':<10}{'Equal time':>12}{'SheLearns':>12}")
    for n in NAMES:
        print(f"{n:<10}{equal['sessions'][n]:>12.1f}{adaptive['sessions'][n]:>12.1f}")

    print("\n5. Sensitivity: does the result hold with other settings?")
    print(f"{'GAIN':>6}{'FORGET':>9}{'Overall change':>17}{'Weakest change':>17}")
    for g in (0.05, 0.10, 0.15):
        for f in (0.0, 0.005, 0.01):
            e = evaluate("equal", g, f)
            a = evaluate("adaptive", g, f)
            d_all = mean(a["skill"].values()) - mean(e["skill"].values())
            d_weak = a["weakest"] - e["weakest"]
            print(f"{g:>6}{f:>9}{d_all:>+17.3f}{d_weak:>+17.3f}")
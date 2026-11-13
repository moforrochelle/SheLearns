"""Compare three planners on pretend girls:
  equal : equal time for every subject
  rule  : the hand-made priority score in model.py
  ml    : the machine learning planner in ml_planner.py

It also tests girls who sometimes press the wrong button.

Run it with:  python3 simulate_ml.py
"""
import random
from datetime import timedelta

import model
import simulate
from simulate import (FakeDate, START, DAYS, RUNS, GAIN, FORGET, PASS,
                      COURSES, NAMES, weakness_from_skill)
import ml_planner

WRONG_RATE = 0.15
STRATEGIES = ("equal", "rule", "ml")


def run(strategy, seed, gain, forget, wrong_rate):
    rng = random.Random(seed)
    skills = {c["name"]: c["skill"] for c in COURSES}
    exam_day = {c["name"]: c["exam_day"] for c in COURSES}
    counts = {n: 0 for n in NAMES}
    subjects = [
        model.make_subject(c["name"], weakness_from_skill(c["skill"]),
                           START + timedelta(days=c["exam_day"]))
        for c in COURSES
    ]
    log = []
    readiness = {}

    for day in range(DAYS):
        today = START + timedelta(days=day)
        FakeDate.current = today

        for name, d in exam_day.items():
            if day == d:
                readiness[name] = skills[name]

        remaining = [s for s in subjects if exam_day[s["name"]] > day]
        if not remaining:
            break

        if strategy == "equal":
            chosen = remaining[day % len(remaining)]
        elif strategy == "rule":
            chosen = max(remaining, key=model.priority_score)
        else:
            chosen = max(remaining, key=lambda s: ml_planner.ml_priority(s, log, today))

        name = chosen["name"]
        counts[name] += 1

        understood = rng.random() < skills[name]
        reported = understood
        if wrong_rate > 0 and rng.random() < wrong_rate:
            reported = not reported   # she pressed the wrong button

        for other in skills:
            if other != name:
                skills[other] = max(0.0, skills[other] - forget)
        skills[name] += gain * (1 - skills[name])

        if strategy != "equal":
            outcome = "understood" if reported else "confused"
            model.apply_outcome(chosen, outcome)
            log.append({"date": today.isoformat(), "subject": name, "outcome": outcome})

    return readiness, counts


def evaluate(strategy, wrong_rate=0.0, gain=GAIN, forget=FORGET):
    skill_sum = {n: 0.0 for n in NAMES}
    sessions = {n: 0 for n in NAMES}
    weakest_sum = 0.0
    for seed in range(RUNS):
        readiness, counts = run(strategy, seed, gain, forget, wrong_rate)
        for n in NAMES:
            skill_sum[n] += readiness[n]
            sessions[n] += counts[n]
        weakest_sum += min(readiness.values())
    return {
        "skill": {n: skill_sum[n] / RUNS for n in NAMES},
        "sessions": {n: sessions[n] / RUNS for n in NAMES},
        "weakest": weakest_sum / RUNS,
    }


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def main():
    print(f"{RUNS} pretend girls, {DAYS} days. Wrong report = she presses the wrong button.\n")
    print(f"{'Wrong reports':<15}{'Planner':<12}{'Overall':>9}{'Weakest':>9}"
          f"{'Maths':>8}{'Biology':>9}{'English':>9}")
    for wr in (0.0, 0.15, 0.30):
        for s in STRATEGIES:
            r = evaluate(s, wr)
            print(f"{wr:<15.0%}{s:<12}{mean(r['skill'].values()):>9.3f}{r['weakest']:>9.3f}"
                  f"{r['skill']['Maths']:>8.3f}{r['skill']['Biology']:>9.3f}"
                  f"{r['skill']['English']:>9.3f}")
        print()

    print("Sessions per subject (no wrong reports)")
    for s in STRATEGIES:
        r = evaluate(s, 0.0)
        print(f"  {s:<8}" + "  ".join(f"{n} {r['sessions'][n]:.1f}" for n in NAMES))

    print("\nSensitivity: weakest-subject change of the ML planner")
    print(f"{'GAIN':>6}{'FORGET':>9}{'ML vs equal':>13}{'ML vs rule':>12}"
          f"{'  (15% wrong: vs equal':>24}{'vs rule)':>10}")
    for g in (0.05, 0.10, 0.15):
        for f in (0.0, 0.005, 0.01):
            e0, r0, m0 = (evaluate(s, 0.0, g, f)["weakest"] for s in STRATEGIES)
            e1, r1, m1 = (evaluate(s, WRONG_RATE, g, f)["weakest"] for s in STRATEGIES)
            print(f"{g:>6}{f:>9}{m0 - e0:>+13.3f}{m0 - r0:>+12.3f}"
                  f"{m1 - e1:>+24.3f}{m1 - r1:>+10.3f}")


if __name__ == "__main__":
    main()
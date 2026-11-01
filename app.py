import json
import os
import re
import secrets
from datetime import datetime, date

import requests
from flask import Flask, render_template, request, redirect, url_for, session
from google import genai
from werkzeug.security import generate_password_hash, check_password_hash

from model import make_subject, priority_score, apply_outcome

app = Flask(__name__)

BASE_DIR = os.path.dirname(__file__)


# ---------- Secret key (needed to keep users logged in) ----------

def get_secret_key():
    key = os.environ.get("FLASK_SECRET_KEY")
    if key:
        return key
    path = os.path.join(BASE_DIR, "secret_key.txt")
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write(secrets.token_hex(32))
    with open(path) as f:
        return f.read().strip()


app.secret_key = get_secret_key()
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

# ---------- Groups ----------

GROUPS = [
    {"name": "Data Girl Technologies",
     "about": "Trains women and girls in coding and design, based in Cameroon.",
     "link": "", "checked": "", "working": False},
    {"name": "AkiraChix",
     "about": "Supports young women in tech through training, based in Nairobi.",
     "link": "https://akirachix.com", "checked": "", "working": False},
    {"name": "She Code Africa",
     "about": "A community for African women in tech with mentorship and events.",
     "link": "https://shecodeafrica.org", "checked": "", "working": False},
    {"name": "iamtheCODE",
     "about": "Aims to get 1 million women and girls coding by 2030.",
     "link": "https://iamthecode.org", "checked": "", "working": False},
    {"name": "Women Techmakers",
     "about": "Google's program giving women in tech visibility and community.",
     "link": "https://www.womentechmakers.com", "checked": "", "working": False},
]


def check_link(url):
    """True if the page loads, False if not."""
    if not url:
        return False
    try:
        r = requests.get(url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        return r.status_code < 400
    except requests.RequestException:
        return False


# ---------- Gemini helper ----------

GEMINI_MODEL = "gemini-3.6-flash"   # change if you get "model not found"

_client = None   # created once, then kept alive


def get_client():
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _client


def ask_gemini(prompt):
    client = get_client()
    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    text = response.text
    if not text:
        raise ValueError("Gemini returned no text")
    return text.strip()


# ---------- JSON file helpers ----------

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ---------- Accounts ----------

USERS_FILE = os.path.join(BASE_DIR, "users.json")
USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{3,20}$")


def load_users():
    return load_json(USERS_FILE).get("users", {})


def save_users(users):
    save_json(USERS_FILE, {"users": users})


def username_problem(name):
    if not USERNAME_RE.match(name):
        return ("Username must be 3 to 20 characters: letters, numbers, - or _. "
                "No spaces and no email.")
    if sum(ch.isdigit() for ch in name) >= 6:
        return "Use a nickname only. Do not use a phone number."
    return None


def is_admin():
    """True if the logged-in user is the admin set in ADMIN_USERNAME."""
    admin = os.environ.get("ADMIN_USERNAME", "").strip().lower()
    return bool(admin) and session.get("user") == admin


@app.before_request
def require_login():
    if request.endpoint in ("login", "register", "static"):
        return None
    key = session.get("user")
    if not key or key not in load_users():
        session.clear()
        return redirect(url_for("login"))
    return None


@app.context_processor
def inject_user():
    key = session.get("user")
    user = load_users().get(key) if key else None
    return {"username": user["username"] if user else None,
            "is_admin": is_admin()}


@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    if request.method == "POST":
        name = request.form["username"].strip()
        password = request.form["password"]
        confirm = request.form["confirm"]
        users = load_users()
        problem = username_problem(name)

        if problem:
            error = problem
        elif name.lower() in users:
            error = "That username is taken. Try another one."
        elif len(password) < 8:
            error = "Password must be at least 8 characters."
        elif password != confirm:
            error = "The two passwords do not match."
        else:
            users[name.lower()] = {
                "username": name,
                "password_hash": generate_password_hash(password),
                "created": date.today().isoformat(),
            }
            save_users(users)
            session.clear()
            session["user"] = name.lower()
            return redirect(url_for("home"))
    return render_template("register.html", error=error)


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        key = request.form["username"].strip().lower()
        password = request.form["password"]
        user = load_users().get(key)
        if user and check_password_hash(user["password_hash"], password):
            session.clear()
            session["user"] = key
            return redirect(url_for("home"))
        error = "Wrong username or password."
    return render_template("login.html", error=error)


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/whoami")
def whoami():
    """Shows why the Admin link is (or is not) visible. Safe to keep or delete."""
    is_set = "yes" if os.environ.get("ADMIN_USERNAME", "").strip() else "no"
    return (f"Your login name: {session.get('user')}<br>"
            f"ADMIN_USERNAME is set in this terminal: {is_set}<br>"
            f"You are admin: {is_admin()}")


# ---------- Planner data, saved per user ----------

DATA_FILE = os.path.join(BASE_DIR, "data.json")


def _subject_to_json(subject):
    """Dates become text so they can be saved in a JSON file."""
    item = dict(subject)
    for key in ("exam_date", "last_studied"):
        if item.get(key) is not None:
            item[key] = item[key].isoformat()
    return item


def _subject_from_json(item):
    """Text becomes dates again when loading."""
    subject = dict(item)
    for key in ("exam_date", "last_studied"):
        if subject.get(key):
            subject[key] = date.fromisoformat(subject[key])
        else:
            subject[key] = None
    return subject


def load_user_data(user_key):
    all_data = load_json(DATA_FILE).get("users", {})
    mine = all_data.get(user_key, {})
    subjects = [_subject_from_json(s) for s in mine.get("subjects", [])]
    return subjects, mine.get("slots", []), mine.get("sessions", [])


def save_user_data(user_key, subjects, slots, sessions):
    all_data = load_json(DATA_FILE).get("users", {})
    all_data[user_key] = {
        "subjects": [_subject_to_json(s) for s in subjects],
        "slots": slots,
        "sessions": sessions,
    }
    save_json(DATA_FILE, {"users": all_data})


# ---------- Lessons (generated by Gemini, then saved) ----------

LESSONS_FILE = os.path.join(BASE_DIR, "lessons.json")

LESSON_TOPICS = [
    {"id": "ai", "title": "What is AI?",
     "subject": "what artificial intelligence (AI) is, with two everyday examples"},
    {"id": "budgeting", "title": "Budgeting basics",
     "subject": "what a budget is and how to make a simple one with pocket money"},
    {"id": "saving", "title": "Saving money",
     "subject": "why saving matters and three simple ways to start saving small amounts"},
    {"id": "menstrual-hygiene", "title": "Menstrual hygiene",
     "subject": "how to stay clean, comfortable and confident during a period"},
]


def generate_lesson(topic):
    prompt = (
        "You are a kind, accurate tutor writing for girls aged 13 to 16. "
        f"Write a short lesson about {topic['subject']}. "
        "Rules: use simple, clear English; write 150 to 200 words; "
        "plain text only, no markdown, no bullet symbols; "
        "end with one practical tip she can use today. "
        "Do not give medical diagnoses or investment advice. "
        "For anything personal about health, tell her to ask a nurse, "
        "a doctor or a trusted adult."
    )
    return ask_gemini(prompt)


# ---------- Book summaries (generated by Gemini, then saved) ----------

BOOKS_FILE = os.path.join(BASE_DIR, "summaries.json")

# Change these to any books you like.
BOOKS = [
    {"id": "atomic-habits", "title": "Atomic Habits", "author": "James Clear"},
    {"id": "mindset", "title": "Mindset", "author": "Carol S. Dweck"},
    {"id": "psychology-of-money", "title": "The Psychology of Money",
     "author": "Morgan Housel"},
]


def generate_summary(book):
    prompt = (
        "You are a kind, accurate tutor writing for girls aged 13 to 16. "
        f"Write a short summary of the book \"{book['title']}\" by {book['author']}. "
        "Rules: write 120 to 160 words in your own words; use simple, clear English; "
        "plain text only, no markdown; do not quote the book. "
        "Only include ideas you are confident are really in this book. "
        "If you are not sure about the book, say you are not sure instead of guessing. "
        "End with one idea she can try this week."
    )
    return ask_gemini(prompt)


# ---------- Community: study-buddy matching ----------

COMMUNITY_FILE = os.path.join(BASE_DIR, "community.json")
MIN_OVERLAP_MINUTES = 30   # a shared time slot must last at least this long

# Demo girls, shown next to real accounts.
DEMO_PROFILES = [
    {"id": "demo1", "nickname": "Sunshine", "demo": True,
     "subjects": ["Maths", "English"],
     "slots": [{"day": "Mon", "start": "18:00", "end": "19:00"},
               {"day": "Wed", "start": "17:30", "end": "18:30"}]},
    {"id": "demo2", "nickname": "Bright", "demo": True,
     "subjects": ["Biology", "Maths"],
     "slots": [{"day": "Tue", "start": "19:00", "end": "20:00"},
               {"day": "Mon", "start": "18:30", "end": "19:30"}]},
    {"id": "demo3", "nickname": "Star", "demo": True,
     "subjects": ["Chemistry"],
     "slots": [{"day": "Fri", "start": "16:00", "end": "17:00"}]},
]


def load_profiles():
    data = load_json(COMMUNITY_FILE)
    profiles = data.get("profiles")
    if profiles is None:          # first time: start with the demo girls
        profiles = [dict(p) for p in DEMO_PROFILES]
        save_json(COMMUNITY_FILE, {"profiles": profiles})
    return profiles


def save_profiles(profiles):
    save_json(COMMUNITY_FILE, {"profiles": profiles})


def to_minutes(hhmm):
    hours, minutes = hhmm.split(":")
    return int(hours) * 60 + int(minutes)


def overlap_minutes(a, b):
    """Minutes two slots share. Zero if they are on different days."""
    if a["day"] != b["day"]:
        return 0
    start = max(to_minutes(a["start"]), to_minutes(b["start"]))
    end = min(to_minutes(a["end"]), to_minutes(b["end"]))
    return max(0, end - start)


def find_matches(me, others):
    """Other girls who share a subject AND a free time with her."""
    my_subjects = {s.strip().lower(): s.strip() for s in me["subjects"]}
    matches = []
    for other in others:
        shared = [name for name in other["subjects"]
                  if name.strip().lower() in my_subjects]
        common_times = []
        for a in me["slots"]:
            for b in other["slots"]:
                minutes = overlap_minutes(a, b)
                if minutes >= MIN_OVERLAP_MINUTES:
                    start = max(to_minutes(a["start"]), to_minutes(b["start"]))
                    common_times.append({
                        "day": a["day"],
                        "start": f"{start // 60:02d}:{start % 60:02d}",
                        "minutes": minutes,
                    })
        if shared and common_times:
            matches.append({
                "nickname": other["nickname"],
                "demo": other.get("demo", False),
                "shared": shared,
                "times": common_times,
                "score": len(shared) * 100 + sum(t["minutes"] for t in common_times),
            })
    return sorted(matches, key=lambda m: m["score"], reverse=True)


def community_page(error=None):
    key = session["user"]
    profile_id = "user:" + key
    profiles = load_profiles()
    me = next((p for p in profiles if p["id"] == profile_id), None)
    others = [p for p in profiles if p["id"] != profile_id]
    matches = find_matches(me, others) if me else []
    subjects, slots, _ = load_user_data(key)
    return render_template("community.html", me=me, matches=matches, error=error,
                           subjects=subjects, slots=slots)


# ---------- Community: moderated board ----------

BOARD_FILE = os.path.join(BASE_DIR, "board.json")
MAX_POST_LENGTH = 300
MAX_OPEN_POSTS = 5   # posts a girl can have waiting for approval at once


def load_posts():
    return load_json(BOARD_FILE).get("posts", [])


def save_posts(posts):
    save_json(BOARD_FILE, {"posts": posts})


def post_problem(text):
    """Return a reason to reject the post, or None if it is fine."""
    if not text:
        return "Write something first."
    if len(text) > MAX_POST_LENGTH:
        return f"Posts can be at most {MAX_POST_LENGTH} characters."
    lowered = text.lower()
    if sum(ch.isdigit() for ch in text) >= 6:
        return "Please do not share phone numbers or long numbers."
    if "@" in text:
        return "Please do not share emails or social media handles."
    if "http" in lowered or "www." in lowered or ".com" in lowered:
        return "Please do not post links."
    return None


def board_page(error=None):
    key = session["user"]
    posts = load_posts()
    approved = [p for p in posts if p["status"] == "approved"]
    approved.sort(key=lambda p: p["created"], reverse=True)
    mine = [p for p in posts
            if p["author_key"] == key and p["status"] != "approved"]
    return render_template("board.html", approved=approved, mine=mine,
                           error=error, my_key=key, max_length=MAX_POST_LENGTH)


# ---------- Planner pages ----------

@app.route("/")
def home():
    subjects, slots, _ = load_user_data(session["user"])
    return render_template("home.html", subjects=subjects, slots=slots)


@app.route("/add_subject", methods=["POST"])
def add_subject():
    key = session["user"]
    subjects, slots, sessions = load_user_data(key)

    name = request.form["name"]
    weakness = int(request.form["weakness"])
    raw = request.form["exam_date"]
    exam_date = datetime.strptime(raw, "%Y-%m-%d").date() if raw else None

    subject = make_subject(name, weakness, exam_date)
    subject["id"] = len(subjects)
    subjects.append(subject)
    save_user_data(key, subjects, slots, sessions)
    return redirect(url_for("home"))


@app.route("/add_slot", methods=["POST"])
def add_slot():
    key = session["user"]
    subjects, slots, sessions = load_user_data(key)

    slots.append({
        "id": len(slots),
        "day": request.form["day"],
        "start": request.form["start"],
        "end": request.form["end"],
        "skip_count": 0,
    })
    save_user_data(key, subjects, slots, sessions)
    return redirect(url_for("home"))


@app.route("/plan")
def plan():
    subjects, slots, sessions = load_user_data(session["user"])
    ranked = sorted(subjects, key=priority_score, reverse=True)
    # slots she skipped most often are used last
    ordered_slots = sorted(slots, key=lambda s: s["skip_count"])
    plan_rows = []
    for i, slot in enumerate(ordered_slots):
        if not ranked:
            break
        subject = ranked[i % len(ranked)]
        plan_rows.append({"slot": slot, "subject": subject})
    return render_template("plan.html", ranked=ranked, plan_rows=plan_rows,
                           score=priority_score, sessions=sessions)


@app.route("/log", methods=["POST"])
def log():
    key = session["user"]
    subjects, slots, sessions = load_user_data(key)

    subject = next(s for s in subjects if s["id"] == int(request.form["subject_id"]))
    slot = next(t for t in slots if t["id"] == int(request.form["slot_id"]))
    outcome = request.form["outcome"]

    if outcome == "skipped":
        slot["skip_count"] += 1
    else:
        apply_outcome(subject, outcome)

    sessions.append({
        "date": date.today().isoformat(),
        "subject": subject["name"],
        "slot": f'{slot["day"]} {slot["start"]}-{slot["end"]}',
        "outcome": outcome,
    })
    save_user_data(key, subjects, slots, sessions)
    return redirect(url_for("plan"))


# ---------- Groups page ----------

@app.route("/groups")
def groups():
    today = date.today().isoformat()
    for g in GROUPS:
        if g["checked"] != today:        # check once per day
            g["working"] = check_link(g["link"])
            g["checked"] = today
    return render_template("groups.html", groups=GROUPS)


# ---------- Lessons pages ----------

@app.route("/lessons")
def lessons():
    saved = load_json(LESSONS_FILE)
    return render_template("lessons.html", topics=LESSON_TOPICS, saved=saved)


@app.route("/lesson/<topic_id>")
def lesson(topic_id):
    topic = next((t for t in LESSON_TOPICS if t["id"] == topic_id), None)
    if topic is None:
        return redirect(url_for("lessons"))

    saved = load_json(LESSONS_FILE)
    error = None
    if topic_id not in saved:
        try:
            text = generate_lesson(topic)
            saved[topic_id] = {
                "title": topic["title"],
                "text": text,
                "reviewed": False,
                "created": date.today().isoformat(),
            }
            save_json(LESSONS_FILE, saved)
        except Exception as e:
            print("Lesson error:", e)
            error = "Could not create the lesson right now. Please try again in a minute."

    return render_template("lesson.html", topic=topic,
                           lesson=saved.get(topic_id), error=error)


# ---------- Book summary pages ----------

@app.route("/books")
def books():
    saved = load_json(BOOKS_FILE)
    return render_template("books.html", books=BOOKS, saved=saved)


@app.route("/book/<book_id>")
def book(book_id):
    item = next((b for b in BOOKS if b["id"] == book_id), None)
    if item is None:
        return redirect(url_for("books"))

    saved = load_json(BOOKS_FILE)
    error = None
    if book_id not in saved:
        try:
            text = generate_summary(item)
            saved[book_id] = {
                "title": item["title"],
                "author": item["author"],
                "text": text,
                "reviewed": False,
                "created": date.today().isoformat(),
            }
            save_json(BOOKS_FILE, saved)
        except Exception as e:
            print("Summary error:", e)
            error = "Could not create the summary right now. Please try again in a minute."

    return render_template("book.html", book=item,
                           summary=saved.get(book_id), error=error)


# ---------- Community pages ----------

@app.route("/community")
def community():
    return community_page()


@app.route("/community/join", methods=["POST"])
def community_join():
    key = session["user"]
    subjects, slots, _ = load_user_data(key)

    if not subjects or not slots:
        return community_page(
            error="Add at least one subject and one free slot on the home page first.")

    profile_id = "user:" + key
    profiles = [p for p in load_profiles() if p["id"] != profile_id]
    profiles.append({
        "id": profile_id,
        "nickname": load_users()[key]["username"],
        "demo": False,
        "subjects": [s["name"].strip() for s in subjects],
        "slots": [{"day": t["day"], "start": t["start"], "end": t["end"]}
                  for t in slots],
    })
    save_profiles(profiles)
    return redirect(url_for("community"))


@app.route("/community/leave", methods=["POST"])
def community_leave():
    profile_id = "user:" + session["user"]
    profiles = [p for p in load_profiles() if p["id"] != profile_id]
    save_profiles(profiles)
    return redirect(url_for("community"))


# ---------- Board pages ----------

@app.route("/board")
def board():
    return board_page()


@app.route("/board/post", methods=["POST"])
def board_post():
    key = session["user"]
    text = request.form["text"].strip()

    problem = post_problem(text)
    if problem:
        return board_page(error=problem)

    posts = load_posts()
    waiting = [p for p in posts
               if p["author_key"] == key and p["status"] in ("pending", "reported")]
    if len(waiting) >= MAX_OPEN_POSTS:
        return board_page(error="You have several posts waiting. Please wait for them to be reviewed.")

    posts.append({
        "id": secrets.token_hex(4),
        "author_key": key,
        "author": load_users()[key]["username"],
        "text": text,
        "status": "pending",   # pending, approved, rejected or reported
        "created": datetime.now().isoformat(timespec="seconds"),
    })
    save_posts(posts)
    return redirect(url_for("board"))


@app.route("/board/delete/<post_id>", methods=["POST"])
def board_delete(post_id):
    key = session["user"]
    posts = load_posts()
    kept = [p for p in posts
            if not (p["id"] == post_id and (p["author_key"] == key or is_admin()))]
    save_posts(kept)
    return redirect(request.referrer or url_for("board"))


@app.route("/board/report/<post_id>", methods=["POST"])
def board_report(post_id):
    key = session["user"]
    posts = load_posts()
    for p in posts:
        if p["id"] == post_id and p["author_key"] != key and p["status"] == "approved":
            p["status"] = "reported"   # hidden until the admin reviews it again
    save_posts(posts)
    return redirect(url_for("board"))


# ---------- Admin: review posts ----------

@app.route("/admin")
def admin():
    if not is_admin():
        return redirect(url_for("home"))
    posts = load_posts()
    queue = [p for p in posts if p["status"] in ("pending", "reported")]
    queue.sort(key=lambda p: p["created"])
    return render_template("admin.html", queue=queue)


@app.route("/admin/decide/<post_id>/<decision>", methods=["POST"])
def admin_decide(post_id, decision):
    if not is_admin():
        return redirect(url_for("home"))
    if decision not in ("approve", "reject"):
        return redirect(url_for("admin"))
    posts = load_posts()
    for p in posts:
        if p["id"] == post_id:
            p["status"] = "approved" if decision == "approve" else "rejected"
    save_posts(posts)
    return redirect(url_for("admin"))


# ---------- Delete my account ----------

@app.route("/account/delete", methods=["POST"])
def delete_account():
    key = session["user"]

    users = load_users()
    users.pop(key, None)
    save_users(users)

    all_data = load_json(DATA_FILE).get("users", {})
    all_data.pop(key, None)
    save_json(DATA_FILE, {"users": all_data})

    profiles = [p for p in load_profiles() if p["id"] != "user:" + key]
    save_profiles(profiles)

    posts = [p for p in load_posts() if p["author_key"] != key]
    save_posts(posts)

    session.clear()
    return redirect(url_for("register"))


if __name__ == "__main__":
    app.run(debug=True)
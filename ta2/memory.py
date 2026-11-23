"""memory.py - Unit 3 memory layer.

Short-term memory : LangGraph SqliteSaver checkpointer + thread_id (see graph.py / main.py)
Long-term memory  : LangGraph SqliteStore, namespaces (student_id, "history") and
                    (student_id, "profile"). Stored in SQLite, so it survives a restart.
The trend is computed in plain Python (deterministic), never by the LLM.
"""
import re
import sqlite3
from datetime import datetime

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.store.sqlite import SqliteStore


def open_db(path: str):
    """Return (checkpointer, store) backed by the same SQLite file."""
    def conn():
        return sqlite3.connect(path, check_same_thread=False, isolation_level=None)
    saver, store = SqliteSaver(conn()), SqliteStore(conn())
    saver.setup()
    store.setup()
    return saver, store


def save_measurement(store, student: str, weight: float, height: float,
                     bmi: float, category: str, ts: str | None = None) -> None:
    ts = ts or datetime.now().isoformat(timespec="microseconds")
    store.put((student, "history"), ts, {"ts": ts, "weight_kg": weight,
              "height_cm": height, "bmi": bmi, "category": category})
    store.put((student, "profile"), "latest", {"weight_kg": weight, "height_cm": height})


def get_history(store, student: str) -> list[dict]:
    items = store.search((student, "history"), limit=500)
    return [i.value for i in sorted(items, key=lambda i: i.key)]


def get_profile(store, student: str) -> dict:
    item = store.get((student, "profile"), "latest")
    return item.value if item else {}


def trend_report(history: list[dict]) -> str:
    if len(history) < 2:
        return f"Trend: only {len(history)} record(s) stored, need at least 2."
    first, last = history[0], history[-1]
    delta = round(last["bmi"] - first["bmi"], 2)
    direction = "stable" if abs(delta) < 0.3 else ("rising" if delta > 0 else "falling")
    return (f"Trend: {len(history)} records ({first['ts'][:10]} to {last['ts'][:10]}): "
            f"BMI {first['bmi']} -> {last['bmi']} ({delta:+.2f}, {direction}); "
            f"weight {first['weight_kg']} -> {last['weight_kg']} kg.")


# ---------------------------------------------------------------------------
# Improvement 4: ASCII bar chart of BMI history
# No extra dependencies — pure Python and Unicode block characters.
# ---------------------------------------------------------------------------
def trend_chart(history: list[dict]) -> str:
    """Return a formatted ASCII bar chart of BMI over time.

    Returns an empty string if there are fewer than 2 records (nothing to compare).
    The bars are scaled relative to the min/max BMI in the history so small
    changes are still visible.
    """
    if len(history) < 2:
        return ""

    bmis = [h["bmi"] for h in history]
    lo, hi = min(bmis), max(bmis)
    span = hi - lo if hi != lo else 1.0   # avoid division by zero when all readings identical
    BAR_W = 20

    delta = round(bmis[-1] - bmis[0], 2)
    if abs(delta) < 0.3:
        arrow = "➡  stable"
    elif delta > 0:
        arrow = "↑  rising"
    else:
        arrow = "↓  falling"

    header = (
        f"{'─' * 50}\n"
        f" BMI Trend  ({history[0]['ts'][:10]} → {history[-1]['ts'][:10]})\n"
        f"{'─' * 50}\n"
        f"  {'Date':<12} {'BMI':>5}  {'':2}{'Chart (relative)'}\n"
    )
    rows = []
    for h in history:
        filled = round((h["bmi"] - lo) / span * BAR_W)
        bar = "█" * filled + "░" * (BAR_W - filled)
        rows.append(f"  {h['ts'][:10]}  {h['bmi']:>5.2f}  |{bar}|")

    footer = (
        f"{'─' * 50}\n"
        f"  Overall change: {delta:+.2f} BMI points  {arrow}\n"
        f"  Weight: {history[0]['weight_kg']} kg → {history[-1]['weight_kg']} kg\n"
        f"{'─' * 50}"
    )
    return "\n".join([header] + rows + [footer])


# ---------------------------------------------------------------------------
# Semantic Episodic Memory: qualitative user health facts & preferences
# Namespace: (student_id, "facts")
# ---------------------------------------------------------------------------
def save_user_fact(store, student: str, category: str, text: str) -> None:
    """Store qualitative user fact (e.g. dietary preference, injury, lifestyle)."""
    ts = datetime.now().isoformat(timespec="microseconds")
    key = f"{category}_{abs(hash(text)) % 10000}"
    store.put((student, "facts"), key, {"ts": ts, "category": category, "fact": text})


def get_user_facts(store, student: str) -> list[dict]:
    """Retrieve all stored qualitative facts for a student."""
    items = store.search((student, "facts"), limit=100)
    return [i.value for i in sorted(items, key=lambda i: i.key)]


def format_facts(facts: list[dict]) -> str:
    """Format facts into a concise string for the Advisor prompt."""
    if not facts:
        return ""
    cats = {}
    for f in facts:
        cats.setdefault(f.get("category", "general"), []).append(f.get("fact", ""))
    parts = []
    for cat, items in cats.items():
        parts.append(f"{cat.title()}: {', '.join(dict.fromkeys(items))}")
    return "; ".join(parts)


def extract_and_save_facts(store, student: str, text: str) -> list[str]:
    """Extract qualitative facts (diet, limitations, lifestyle, age) and persist them."""
    text_lower = text.lower()
    found = []

    # Diet preferences (check non-vegetarian BEFORE vegetarian to prevent false matches)
    if re.search(r"\b(?:non[- ]vegetarian|non[- ]veg|eats? meat|omnivore)\b", text_lower):
        fact = "Non-vegetarian diet"
        save_user_fact(store, student, "diet", fact)
        found.append(fact)
    else:
        diet_m = re.search(r"\b(vegetarian|vegan|plant-based|pescatarian|keto|ketogenic|halal|kosher|gluten-free)\b", text_lower)
        if diet_m:
            fact = diet_m.group(1).title() + " diet"
            save_user_fact(store, student, "diet", fact)
            found.append(fact)

    # Physical limitations & injuries
    limit_patterns = [
        (r"\b((?:hip|knee|joint|back|leg|ankle|foot|shoulder|neck)\s+pain|arthritis|asthma|injured\s+(?:knee|hip|leg|back|ankle)|bad\s+(?:back|knee|hip)|fracture|recovering from\s+(?:a\s+)?fracture)\b",
         lambda m: f"Reports {m.group(1)} (favors low-impact exercises)"),
        (r"\b(?:cannot|can't|unable to|hard to|difficult (?:for me )?to|struggle to)\s+run\b",
         lambda m: "Cannot run (needs walking/swimming/cycling)"),
    ]
    for p, fn in limit_patterns:
        m = re.search(p, text_lower)
        if m:
            fact = fn(m)
            save_user_fact(store, student, "limitation", fact)
            found.append(fact)

    # Lifestyle
    lifestyle_patterns = [
        (r"\b(desk job|sedentary lifestyle|sit all day)\b", lambda m: "Sedentary / desk job"),
        (r"\b(run daily|go to the gym|lift weights|swim regularly)\b", lambda m: f"Active: {m.group(1)}"),
    ]
    for p, fn in lifestyle_patterns:
        m = re.search(p, text_lower)
        if m:
            fact = fn(m)
            save_user_fact(store, student, "lifestyle", fact)
            found.append(fact)

    # Age
    age_m = re.search(r"\bi (?:am|'m)\s*(\d{1,2})\s*(?:years? old|yo)\b", text_lower)
    if age_m:
        age_val = int(age_m.group(1))
        if 15 <= age_val <= 110:
            fact = f"Age: {age_val} years"
            save_user_fact(store, student, "demographics", fact)
            found.append(fact)

    return found


def clear_user_memory(db_path: str, student: str) -> None:
    """Wipe all long-term memory (measurements, profile, facts) and thread checkpoints for a user."""
    conn = sqlite3.connect(db_path)
    try:
        with conn:
            conn.execute("DELETE FROM store WHERE prefix LIKE ?", (f"{student}.%",))
            conn.execute("DELETE FROM checkpoints WHERE thread_id LIKE ?", (f"{student}%",))
            conn.execute("DELETE FROM writes WHERE thread_id LIKE ?", (f"{student}%",))
    except Exception:
        pass
    finally:
        conn.close()


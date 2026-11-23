"""seed.py - fill the SQLite long-term store with a demo history so the trend report has data.
Run:  python -m ta2.seed
"""
from ta1.bmi_tool import calculate_bmi_category

from .config import DB_PATH
from .memory import open_db, save_measurement

ROWS = [("2026-08-01T09:00:00", 82, 175), ("2026-08-20T09:00:00", 80, 175),
        ("2026-09-10T09:00:00", 78, 175), ("2026-10-01T09:00:00", 76, 175)]

if __name__ == "__main__":
    _, store = open_db(DB_PATH)
    for ts, w, h in ROWS:
        out = calculate_bmi_category.invoke({"weight_kg": w, "height_cm": h})
        bmi = float(out.split("BMI: ")[1].split(" |")[0])
        cat = out.split("Category: ")[1].split(" |")[0]
        save_measurement(store, "demo", w, h, bmi, cat, ts=ts)
    print("Seeded student 'demo' with", len(ROWS), "records.")

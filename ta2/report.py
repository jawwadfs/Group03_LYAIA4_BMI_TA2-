"""report.py - One-click Doctor / Clinical Screening Report generator.

Generates a print-ready, professional HTML/PDF summary report containing:
- Patient / Student ID & report timestamp
- Latest biometric measurements & WHO category
- Ideal weight range calculation (Normal BMI 18.5 - 24.9)
- Historical weigh-in trajectory table
- User-disclosed health context (dietary preferences, physical limitations from episodic memory)
- Doctor discussion prompts and mandatory clinical disclaimer
"""
from datetime import datetime


def generate_doctor_report_html(
    student_id: str,
    profile: dict,
    history: list[dict],
    facts: list[dict],
) -> str:
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")
    latest = history[-1] if history else {}
    first = history[0] if history else {}

    weight_kg = latest.get("weight_kg", profile.get("weight_kg", 0.0))
    height_cm = latest.get("height_cm", profile.get("height_cm", 0.0))
    bmi = latest.get("bmi", 0.0)
    category = latest.get("category", "Not recorded")

    weight_lbs = round(weight_kg * 2.20462, 1) if weight_kg else 0.0
    height_ft = f"{int(height_cm // 30.48)}'{int((height_cm % 30.48) // 2.54)}\"" if height_cm else "N/A"

    # Ideal weight calculation
    if height_cm > 0:
        hm = height_cm / 100.0
        h_sq = hm ** 2
        min_ideal = round(18.5 * h_sq, 1)
        max_ideal = round(24.9 * h_sq, 1)
        ideal_str = f"{min_ideal} – {max_ideal} kg ({round(min_ideal*2.20462, 1)} – {round(max_ideal*2.20462, 1)} lbs)"
    else:
        ideal_str = "Height not recorded"

    # Delta
    if len(history) >= 2:
        delta_bmi = round(latest.get("bmi", 0) - first.get("bmi", 0), 2)
        delta_w = round(latest.get("weight_kg", 0) - first.get("weight_kg", 0), 1)
        direction = "Stable" if abs(delta_bmi) < 0.3 else ("Rising" if delta_bmi > 0 else "Falling")
        trend_summary = f"{len(history)} records ({first.get('ts', '')[:10]} to {latest.get('ts', '')[:10]}): BMI changed by {delta_bmi:+.2f} ({direction}), weight changed by {delta_w:+.1f} kg."
    else:
        trend_summary = "Single record stored; multiple readings required to compute trajectory."

    # Facts grouping
    facts_html = ""
    if facts:
        grouped = {}
        for f in facts:
            grouped.setdefault(f.get("category", "General").title(), []).append(f.get("fact", ""))
        for cat, items in grouped.items():
            facts_html += f"<li><strong>{cat}:</strong> {', '.join(dict.fromkeys(items))}</li>"
    else:
        facts_html = "<li><em>No personal dietary or physical limitations recorded.</em></li>"

    # Table rows
    table_rows = ""
    for h in reversed(history[-10:]):  # last 10 records
        table_rows += (
            f"<tr>"
            f"<td>{h.get('ts', '')[:10]}</td>"
            f"<td>{h.get('weight_kg', '')} kg</td>"
            f"<td>{h.get('height_cm', '')} cm</td>"
            f"<td><strong>{h.get('bmi', '')}</strong></td>"
            f"<td>{h.get('category', '')}</td>"
            f"</tr>"
        )
    if not table_rows:
        table_rows = "<tr><td colspan='5'><em>No historical records found.</em></td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Clinical Health & BMI Screening Summary - {student_id}</title>
<style>
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #1a202c;
    background-color: #f7fafc;
    margin: 0;
    padding: 24px;
    line-height: 1.5;
  }}
  .card {{
    max-width: 820px;
    margin: 0 auto;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 32px;
    box-shadow: 0 4px 6px rgba(0,0,0,0.04);
  }}
  .header {{
    border-bottom: 2px solid #2b6cb0;
    padding-bottom: 16px;
    margin-bottom: 24px;
    display: flex;
    justify-content: space-between;
    align-items: flex-end;
  }}
  h1 {{ margin: 0; color: #2b6cb0; font-size: 24px; }}
  .meta {{ color: #718096; font-size: 13px; text-align: right; }}
  .grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
  }}
  .metric-box {{
    background: #ebf8ff;
    border: 1px solid #bee3f8;
    border-radius: 6px;
    padding: 12px;
    text-align: center;
  }}
  .metric-label {{ font-size: 11px; text-transform: uppercase; color: #4a5568; font-weight: bold; }}
  .metric-val {{ font-size: 20px; font-weight: bold; color: #2b6cb0; margin-top: 4px; }}
  .subtext {{ font-size: 11px; color: #718096; }}
  h2 {{ color: #2d3748; font-size: 16px; border-bottom: 1px solid #edf2f7; padding-bottom: 6px; margin-top: 24px; }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13px; }}
  th, td {{ border: 1px solid #e2e8f0; padding: 8px 12px; text-align: left; }}
  th {{ background: #edf2f7; color: #4a5568; }}
  ul {{ padding-left: 20px; margin: 8px 0; font-size: 14px; }}
  .disclaimer {{
    background: #fffaf0;
    border-left: 4px solid #dd6b20;
    padding: 12px 16px;
    margin-top: 32px;
    font-size: 12px;
    color: #7b341e;
    border-radius: 0 4px 4px 0;
  }}
  @media print {{
    body {{ background: #fff; padding: 0; }}
    .card {{ border: none; box-shadow: none; padding: 0; }}
    .disclaimer {{ border-left: 4px solid #000; }}
  }}
</style>
</head>
<body>
<div class="card">
  <div class="header">
    <div>
      <h1>🏥 Health & BMI Screening Summary</h1>
      <div style="color: #4a5568; font-size: 13px; margin-top: 4px;">Prepared for Medical Discussion & Primary Care Screening</div>
    </div>
    <div class="meta">
      <strong>Subject ID:</strong> {student_id}<br>
      <strong>Date Generated:</strong> {now_str}
    </div>
  </div>

  <h2>Current Biometric Snapshot</h2>
  <div class="grid">
    <div class="metric-box">
      <div class="metric-label">Weight</div>
      <div class="metric-val">{weight_kg} kg</div>
      <div class="subtext">{weight_lbs} lbs</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Height</div>
      <div class="metric-val">{height_cm} cm</div>
      <div class="subtext">{height_ft}</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Body Mass Index</div>
      <div class="metric-val">{bmi}</div>
      <div class="subtext">Screening Index</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">WHO Category</div>
      <div class="metric-val" style="font-size: 15px; margin-top: 8px;">{category}</div>
      <div class="subtext">Adult Classification</div>
    </div>
  </div>

  <div style="background: #f7fafc; border: 1px solid #edf2f7; border-radius: 6px; padding: 12px 16px; margin-bottom: 20px; font-size: 13px;">
    <strong>Recommended Normal Weight Range:</strong> {ideal_str}<br>
    <strong>Trajectory / Historical Trend:</strong> {trend_summary}
  </div>

  <h2>Disclosed Personal Health Context (Episodic Memory)</h2>
  <ul>
    {facts_html}
  </ul>

  <h2>Recent Weigh-in History (Last 10 Records)</h2>
  <table>
    <thead>
      <tr>
        <th>Date</th>
        <th>Weight</th>
        <th>Height</th>
        <th>BMI</th>
        <th>Category</th>
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>

  <h2>Suggested Topics for Discussion with Your Doctor</h2>
  <ul>
    <li>Does my current BMI and weight trajectory align with my overall cardiovascular and metabolic health?</li>
    <li>Given any reported physical limitations (e.g. joint/knee considerations), what low-impact aerobic routines (swimming, brisk walking, cycling) are most suitable?</li>
    <li>Would supplementary screening metrics (e.g., waist-to-height ratio, lipid profile, HbA1c) be beneficial at this stage?</li>
  </ul>

  <div class="disclaimer">
    <strong>Clinical Notice & Disclaimer:</strong> This summary document is produced by an automated screening advisor (Agentic AI TA2) for informational purposes only. BMI is a population screening indicator that does not distinguish muscle mass from body fat, nor does it identify fat distribution or diagnose diseases. Do not alter medication, diet, or treatment plans without consulting a licensed physician or registered dietitian.
  </div>
</div>
</body>
</html>"""
    return html

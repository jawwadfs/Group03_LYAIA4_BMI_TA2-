"""
bmi_tool.py
-----------
Defines the ONE custom tool used by the agent: `calculate_bmi_category`.

This module is intentionally separate from app.py so that:
  1. The tool's logic (validation + calculation) can be unit-tested
     without needing Ollama installed or running.
  2. It's easy to point to a single file during a jury/viva and say
     "this is where all the deterministic business logic lives."

Design decision: all validation and math happens in PLAIN PYTHON inside
this function, not in the LLM / system prompt. The LLM's only job is to
decide WHEN to call this tool and WHAT arguments to pass. The tool is the
single source of truth for correctness.
"""

from langchain.tools import tool

# ---------------------------------------------------------------------------
# Realistic adult height range.
#
# 50 cm is roughly the length of a newborn baby, and this project is aimed
# at adult BMI screening, so anything at or below that is almost certainly
# a data-entry mistake (e.g. typing height in meters instead of cm, or a
# stray "0"). 250 cm (2.5 m) is taller than the tallest verified adult in
# recorded history (~2.72 m for Robert Wadlow, an extreme outlier), so it
# comfortably covers every realistic adult while still catching obvious
# typos like "500" or "1750" (height typed in mm by mistake).
# ---------------------------------------------------------------------------
MIN_HEIGHT_CM = 50.0
MAX_HEIGHT_CM = 250.0


@tool
def calculate_bmi_category(weight_kg: float, height_cm: float) -> str:
    """Calculate BMI, its health category, and short guidance for an adult.

    Use this tool whenever a user provides a weight (in kilograms) and a
    height (in centimeters) and asks about BMI, weight status, or health
    category. Always pass the raw numeric values the user gave you.

    Args:
        weight_kg: The person's weight in kilograms. Must be a positive number.
        height_cm: The person's height in centimeters. Must be a positive,
            realistic adult value (roughly 50-250 cm).

    Returns:
        On success: a single line formatted as
            "BMI: <value> | Category: <category> | Guidance: <text>"
        On failure: a single line starting with "Error: " describing what
        was wrong with the input. This function never raises an exception
        for bad input -- it always returns a string.
    """
    # --- Step 1: safely coerce to float (handles malformed / non-numeric input) ---
    try:
        weight_kg = float(weight_kg)
        height_cm = float(height_cm)
    except (TypeError, ValueError):
        return "Error: weight_kg and height_cm must be numeric values."

    # --- Step 2: guard against non-finite values (NaN, inf) slipping through ---
    if weight_kg != weight_kg or height_cm != height_cm:  # NaN check
        return "Error: weight_kg and height_cm must be valid numbers, not NaN."
    if weight_kg in (float("inf"), float("-inf")) or height_cm in (float("inf"), float("-inf")):
        return "Error: weight_kg and height_cm must be finite numbers."

    # --- Step 3: basic positivity validation ---
    if weight_kg <= 0:
        return "Error: weight_kg must be greater than 0."

    if height_cm <= 0:
        return "Error: height_cm must be greater than 0."

    # --- Step 4: realistic-range validation (prevents nonsense / typo input) ---
    if not (MIN_HEIGHT_CM <= height_cm <= MAX_HEIGHT_CM):
        return (
            f"Error: height_cm must be within a realistic range "
            f"({MIN_HEIGHT_CM:.0f}-{MAX_HEIGHT_CM:.0f} cm)."
        )

    # Optional sanity cap on weight to avoid absurd division results;
    # kept generous since extreme but real medical cases exist.
    if weight_kg > 500:
        return "Error: weight_kg must be within a realistic range (0-500 kg)."

    # --- Step 5: the actual BMI calculation (standard formula) ---
    height_m = height_cm / 100.0
    bmi = weight_kg / (height_m ** 2)
    bmi_rounded = round(bmi, 2)

    # --- Step 6: category lookup using half-open intervals to handle boundaries ---
    # Boundaries: <18.5 Underweight | 18.5-24.9 Normal | 25.0-29.9 Overweight | >=30.0 Obesity
    if bmi < 18.5:
        category = "Underweight"
        guidance = (
            "BMI is below the typical range. Consider speaking with a "
            "healthcare professional about nutrition."
        )
    elif bmi < 25.0:
        category = "Normal weight"
        guidance = (
            "BMI is in the normal range, but it is only one general "
            "health indicator."
        )
    elif bmi < 30.0:
        category = "Overweight"
        guidance = (
            "BMI is above the typical range. A balanced diet and regular "
            "activity are generally recommended."
        )
    else:
        category = "Obesity"
        guidance = (
            "BMI is significantly above the typical range. Consulting a "
            "healthcare professional is advisable."
        )

    return f"BMI: {bmi_rounded} | Category: {category} | Guidance: {guidance}"

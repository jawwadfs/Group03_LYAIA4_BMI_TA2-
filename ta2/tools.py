"""tools.py - TA2 extension tools for the multi-agent advisor.

Extends the frozen TA1 baseline with companion deterministic tools:
- `calculate_ideal_weight_range`: Calculates the healthy weight range (BMI 18.5 - 24.9)
  for a given adult height.
"""
from langchain.tools import tool
from ta1.bmi_tool import calculate_bmi_category  # TA1 baseline tool


@tool
def calculate_ideal_weight_range(height_cm: float) -> str:
    """Calculate the healthy weight range (corresponding to normal adult BMI 18.5 - 24.9) for a given height.

    Use this tool whenever a user asks what their ideal, target, or healthy weight should be,
    or what weight range is recommended for their height.

    Args:
        height_cm: The person's height in centimeters. Must be a positive, realistic adult height (50-250 cm).

    Returns:
        A formatted string with the minimum and maximum recommended weights in kg and lbs.
    """
    try:
        height_cm = float(height_cm)
    except (TypeError, ValueError):
        return "Error: height_cm must be a numeric value."

    if height_cm != height_cm or height_cm in (float("inf"), float("-inf")):
        return "Error: height_cm must be a finite number."

    if not (50.0 <= height_cm <= 250.0):
        return "Error: height_cm must be within a realistic adult range (50-250 cm)."

    height_m = height_cm / 100.0
    h_sq = height_m ** 2
    min_kg = round(18.5 * h_sq, 1)
    max_kg = round(24.9 * h_sq, 1)

    min_lbs = round(min_kg * 2.20462, 1)
    max_lbs = round(max_kg * 2.20462, 1)

    return (
        f"Ideal weight range for {height_cm:.0f} cm (Normal BMI 18.5-24.9): "
        f"{min_kg} to {max_kg} kg ({min_lbs} to {max_lbs} lbs)."
    )

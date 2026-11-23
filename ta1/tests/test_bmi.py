"""
tests/test_bmi.py
------------------
Automated tests for the `calculate_bmi_category` tool.

These tests do NOT require Ollama to be installed or running -- they call
the tool directly in plain Python, which is possible because all the real
logic (validation + math) lives inside bmi_tool.py, not inside the LLM.

Run with:
    pytest
or, from the project root, if imports need it explicit:
    pytest tests/test_bmi.py -v
"""

import sys
import os

# Make sure the project root (one level up from tests/) is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bmi_tool import calculate_bmi_category  # noqa: E402


def call_tool(weight_kg, height_cm) -> str:
    """Helper: invoke the LangChain tool the same way the agent would.

    Note: LangChain auto-generates a pydantic schema for the tool from its
    type hints (weight_kg: float, height_cm: float). Truly non-numeric
    strings (e.g. "seventy") are rejected by that schema *before* our
    function body ever runs, raising a ValidationError instead of
    reaching our own try/except inside calculate_bmi_category.
    Inside the real agent, create_agent's underlying tool-execution node
    catches exactly this kind of exception automatically and feeds an
    error message back to the LLM instead of crashing the app. We mirror
    that same safety net here so this test reflects real runtime behaviour.
    """
    try:
        return calculate_bmi_category.invoke({"weight_kg": weight_kg, "height_cm": height_cm})
    except Exception as exc:
        return f"Error: invalid input ({exc})"


# ---------------------------------------------------------------------------
# Test 1: Valid input -> Normal weight
#   weight = 70 kg, height = 175 cm
#   BMI = 70 / (1.75^2) = 70 / 3.0625 = 22.857... -> rounds to 22.86
#   18.5 <= 22.86 < 25.0 -> "Normal weight"
# ---------------------------------------------------------------------------
def test_valid_normal_weight():
    result = call_tool(70, 175)
    assert not result.startswith("Error")
    assert "22.86" in result
    assert "Normal weight" in result
    assert "Guidance:" in result


# ---------------------------------------------------------------------------
# Test 2: Valid input, different category -> Overweight
#   weight = 85 kg, height = 170 cm
#   height_m = 1.70, BMI = 85 / (1.70^2) = 85 / 2.89 = 29.41... -> 29.41
#   25.0 <= 29.41 < 30.0 -> "Overweight"
# ---------------------------------------------------------------------------
def test_valid_overweight():
    result = call_tool(85, 170)
    assert not result.startswith("Error")
    assert "29.41" in result
    assert "Overweight" in result


# ---------------------------------------------------------------------------
# Test 2b: A third category for good measure -> Underweight
#   weight = 45 kg, height = 170 cm
#   BMI = 45 / 2.89 = 15.57 -> "Underweight"
# ---------------------------------------------------------------------------
def test_valid_underweight():
    result = call_tool(45, 170)
    assert not result.startswith("Error")
    assert "15.57" in result
    assert "Underweight" in result


# ---------------------------------------------------------------------------
# Test 2c: Obesity boundary case
#   weight = 90 kg, height = 170 cm -> BMI = 90/2.89 = 31.14 -> Obesity
# ---------------------------------------------------------------------------
def test_valid_obesity():
    result = call_tool(90, 170)
    assert not result.startswith("Error")
    assert "Obesity" in result


# ---------------------------------------------------------------------------
# Test 3: Invalid height (zero) -> Error string, no crash
# ---------------------------------------------------------------------------
def test_invalid_height_zero():
    result = call_tool(70, 0)
    assert result.startswith("Error")
    assert "height_cm" in result


# ---------------------------------------------------------------------------
# Test 4: Negative weight -> Error string, no crash
# ---------------------------------------------------------------------------
def test_invalid_negative_weight():
    result = call_tool(-5, 170)
    assert result.startswith("Error")
    assert "weight_kg" in result


# ---------------------------------------------------------------------------
# Test 5: Unrealistic height -> Error string, no crash
# ---------------------------------------------------------------------------
def test_invalid_unrealistic_height():
    result = call_tool(70, 500)
    assert result.startswith("Error")
    assert "realistic range" in result


# ---------------------------------------------------------------------------
# Extra: Malformed / non-numeric input -> Error string, no crash
# ---------------------------------------------------------------------------
def test_invalid_non_numeric_input():
    result = call_tool("seventy", 175)
    assert result.startswith("Error")
    assert "number" in result.lower()


# ---------------------------------------------------------------------------
# Extra: Negative height -> Error string, no crash
# ---------------------------------------------------------------------------
def test_invalid_negative_height():
    result = call_tool(70, -175)
    assert result.startswith("Error")
    assert "height_cm" in result


# ---------------------------------------------------------------------------
# Extra: BMI boundary exactly at 18.5 -> should be Normal weight, not Underweight
#   weight = 56.65625, height = 175 -> BMI = 56.65625 / 3.0625 = 18.50 exactly
# ---------------------------------------------------------------------------
def test_boundary_exactly_18_5_is_normal():
    result = call_tool(56.65625, 175)
    assert "Normal weight" in result

"""agents.py - the three LLM-backed workers (everything that calls qwen2.5:3b).

Role spec (5 fields) for each agent is in the README / report.
graph.py only depends on the three methods of Workers, so tests can swap in fakes.
"""
import re

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_ollama import ChatOllama

from ta1.bmi_tool import calculate_bmi_category   # the TA1 tool, unchanged
from .tools import calculate_ideal_weight_range     # TA2 companion tool

from .config import MODEL, RAG_K
from .rag import GROUNDED_SYSTEM, format_context, retrieve

CALC_PROMPT = """You are the Calculator agent.
You have access to two tools:
1. `calculate_bmi_category` (takes weight_kg and height_cm) to compute exact BMI and WHO category.
2. `calculate_ideal_weight_range` (takes height_cm) to compute the healthy adult weight range (BMI 18.5 - 24.9).

Rules:
- If the user gives weight and height, call `calculate_bmi_category`.
- If the user asks for ideal, target, or healthy weight range, call `calculate_ideal_weight_range`.
- If the user ONLY asks for ideal/healthy weight range and provides no weight, call ONLY `calculate_ideal_weight_range`. Do NOT call `calculate_bmi_category`.
- If both apply, you may call both tools.
- If a line in brackets gives a stored height and the user only gave a weight, use the stored height.
- Never compute BMI or weight ranges yourself.
- Reply with the tool output only."""

JUDGE_PROMPT = """You are a safety reviewer for a BMI advisor tool.
Stating standard BMI categories (Underweight, Normal weight, Overweight, Obesity) and advising the user to consult a doctor, dietitian, or eat a balanced diet is SAFE and expected.
Only reply BLOCK if the text explicitly:
1. Diagnoses a specific medical condition or disease (e.g. "you have diabetes", "you have clinical anorexia").
2. Recommends prescription drugs, medications, or pharmaceutical dosages.
3. Recommends extreme or dangerous crash dieting (fasting for multiple days, eating under 1200 kcal).
Otherwise, reply SAFE.
Reply with ONE word only: SAFE or BLOCK."""


# Stored height is offered to the model ONLY if the message has no height of its own.
HAS_HEIGHT = re.compile(r"\d+(\.\d+)?\s*(cm|centimet\w+|m|ft|feet|foot|inch\w*)\b", re.I)

# ---------------------------------------------------------------------------
# Improvement 5: Deterministic imperial -> metric unit conversion.
# Runs BEFORE the LLM sees the text, so a 3B model never has to do unit math.
# ---------------------------------------------------------------------------
def normalize_units(text: str) -> str:
    """Convert imperial/non-metric measurements to kg/cm before the LLM sees them.

    Handles:
      - Pounds  -> kg        (154 lbs -> 69.9 kg)
      - Feet+inches -> cm    (5'10", 5 ft 10 in, 5 feet 10 inches -> 177.8 cm)
      - Feet only -> cm      (6 ft -> 182.9 cm)
      - Decimal metres -> cm (1.75 m -> 175.0 cm, only for plausible adult heights)
    """
    # --- feet + inches: 5'10", 5' 10, 5'10 ---
    def _ft_in(m):
        cm = float(m.group(1)) * 30.48 + float(m.group(2)) * 2.54
        return f"{cm:.1f} cm"

    text = re.sub(r'(\d+)[\']\s*(\d+(?:\.\d+)?)["]?', _ft_in, text)

    # --- feet + inches spelled out: "5 ft 10 in", "5 feet 10 inches" ---
    text = re.sub(
        r"(\d+(?:\.\d+)?)\s*(?:ft|feet|foot)\s+(\d+(?:\.\d+)?)\s*(?:inch\w*|in\b)",
        _ft_in, text, flags=re.I)

    # --- feet only: "6 ft", "6 feet" (no inches part) ---
    def _ft(m):
        return f"{float(m.group(1)) * 30.48:.1f} cm"

    text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:ft|feet|foot)\b", _ft, text, flags=re.I)

    # --- pounds / lbs -> kg ---
    def _lbs(m):
        return f"{float(m.group(1)) * 0.453592:.1f} kg"

    text = re.sub(r"(\d+(?:\.\d+)?)\s*(?:lbs?|pounds?)\b", _lbs, text, flags=re.I)

    # --- decimal metres -> cm (e.g. "1.75 m") but NOT "150 m" ---
    def _m(m):
        val = float(m.group(1))
        if 1.0 <= val <= 2.5:          # plausible adult height in metres
            return f"{val * 100:.1f} cm"
        return m.group(0)              # leave as-is (not a height)

    text = re.sub(r"(\d+\.\d+)\s*m\b", _m, text, flags=re.I)

    return text


class Workers:
    """Calculator, Advisor and the LLM half of the Safety guard."""

    def __init__(self, vectorstore, llm=None, k: int = RAG_K):
        self.llm = llm or ChatOllama(model=MODEL, temperature=0)
        self.vs, self.k = vectorstore, k
        self.calc_agent = create_agent(self.llm, [calculate_bmi_category, calculate_ideal_weight_range],
                                       system_prompt=CALC_PROMPT)

    def calculate(self, text: str, profile: dict) -> dict:
        """Calculator agent: normalise units, then LLM extracts numbers, tools compute."""
        text = normalize_units(text)   # Improvement 5: imperial -> metric before LLM sees it
        use_stored = profile.get("height_cm") and not HAS_HEIGHT.search(text)
        prefix = f"[Stored height: {profile['height_cm']} cm]\n" if use_stored else ""

        w_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilos?|kilograms?)\b", text, re.I)
        h_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:cm|centimet\w*)\b", text, re.I)
        wants_ideal = bool(re.search(r"\b(ideal|target|healthy)\s+weight\b|\bweight\s+range\b|\bwhat should i weigh\b", text, re.I))

        # 1. Attempt LLM tool execution
        try:
            res = self.calc_agent.invoke({"messages": [{"role": "user", "content": prefix + text}]})
            args = None
            outputs = []
            for m in res["messages"]:
                if isinstance(m, AIMessage) and m.tool_calls and args is None:
                    args = m.tool_calls[0]["args"]
                if isinstance(m, ToolMessage):
                    outputs.append(m.content)
            # If user did not provide weight in this turn, NEVER return a phantom BMI calculation
            if not w_match:
                outputs = [o for o in outputs if not o.startswith("BMI:")]
            if outputs:
                return {"tool_output": "\n".join(outputs), "args": args}
        except Exception:
            pass

        # 2. Resilient deterministic fallback: extract numbers and execute tools directly
        weight = float(w_match.group(1)) if w_match else 0.0
        height = float(h_match.group(1)) if h_match else float(profile.get("height_cm", 0.0))

        outputs = []
        args = {"weight_kg": weight, "height_cm": height}

        if weight > 0 and height > 0 and not (wants_ideal and not w_match):
            bmi_out = calculate_bmi_category.invoke({"weight_kg": weight, "height_cm": height})
            outputs.append(bmi_out)

        if (wants_ideal or (not outputs and not weight and height > 0)) and height > 0:
            ideal_out = calculate_ideal_weight_range.invoke({"height_cm": height})
            outputs.append(ideal_out)

        if outputs:
            return {"tool_output": "\n".join(outputs), "args": args}

        return {"tool_output": None, "args": None}

    def advise(self, question: str, user_data: str) -> str:
        """Advisor agent: RAG with a grounded prompt (k chunks) + source citation."""
        docs = retrieve(self.vs, question, self.k)
        # Improvement 3: cite which guideline sections the answer came from
        sources = ", ".join(dict.fromkeys(d.metadata["source"] for d in docs))
        prompt = (f"{GROUNDED_SYSTEM}\n\nCONTEXT:\n{format_context(docs)}\n\n"
                  f"USER DATA:\n{user_data or 'none'}\n\nQUESTION: {question}")
        try:
            answer = self.llm.invoke(prompt).content.strip()
            if not answer.lower().startswith("i don't know"):
                return f"{answer}\n\nSources consulted: {sources}"
            return answer
        except Exception:
            # Resilient fallback: ground cleanly in guidelines without raw headers
            if docs:
                adult_docs = [d for d in docs if "children" not in d.metadata.get("source", "").lower()]
                best_doc = adult_docs[0] if adult_docs else docs[0]
                body = best_doc.page_content.split(":", 1)[-1].strip() if ":" in best_doc.page_content else best_doc.page_content
                return f"{body}\n\nSources consulted: {sources}"
            return "BMI is a screening measure. Consult a doctor or registered dietitian for personal guidance."

    def judge(self, text: str) -> bool:
        """Guard agent, LLM layer: True means BLOCK."""
        try:
            verdict = self.llm.invoke(f"{JUDGE_PROMPT}\n\nTEXT:\n{text}").content
            return verdict.strip().upper().startswith("BLOCK")
        except Exception:
            return False

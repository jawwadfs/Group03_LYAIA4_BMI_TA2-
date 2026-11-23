"""graph.py - Unit 4: supervisor + Calculator / Advisor / Guard agents (LangGraph).

Routing is done by a deterministic supervisor with a hop limit (MAX_HOPS). We tried to keep the
3B model out of routing because small models route unreliably; the LLM is used inside workers.
"""
import re
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.store.base import BaseStore
from langgraph.types import interrupt

from . import guard
from .config import (MAX_HOPS, MAX_RETRIES, REVIEW_BMI_HIGH, REVIEW_BMI_LOW, USE_LLM_JUDGE)
from .memory import (
    extract_and_save_facts,
    format_facts,
    get_history,
    get_profile,
    get_user_facts,
    save_measurement,
    trend_report,
)


class State(TypedDict, total=False):
    messages: Annotated[list, add_messages]   # short-term memory (thread history)
    last_bmi: str                             # persists across turns of a thread
    student_id: str
    user_text: str
    hops: int
    next: str
    log: list
    blocked: str
    guard_in_done: bool
    guard_out_done: bool
    calc_done: bool
    retries: int
    fail_once: bool
    error: str
    tool_output: str
    weight: float
    height: float
    bmi: float
    reviewed: bool
    saved: bool
    trend: str
    facts: str
    draft: str
    final: str
    status: str


# Calculator only runs when the text has a number WITH a unit ("70 kg", "175 cm"),
# so "type 2 diabetes" or "10 minutes" no longer trigger it.
HAS_MEASUREMENT = re.compile(
    r"\d+(\.\d+)?\s*(kg|kgs|kilos?|kilograms?|cm|centimet\w+|lbs?|pounds?|ft|feet|foot|inch\w*|m)\b", re.I)


def turn_input(text: str, student_id: str, fail_once: bool = False) -> dict:
    """Fresh per-turn fields. last_bmi and messages are NOT reset (thread memory)."""
    return {"messages": [("user", text)], "user_text": text, "student_id": student_id,
            "hops": 0, "next": "", "log": [], "blocked": "", "guard_in_done": False,
            "guard_out_done": False, "calc_done": False, "retries": 0, "fail_once": fail_once,
            "error": "", "tool_output": "", "reviewed": False, "saved": False,
            "trend": "", "facts": "", "draft": "", "final": "", "status": ""}


def _log(state, line: str) -> list:
    print(f"   {line}")
    return state.get("log", []) + [line]


def decide(s: State, max_hops: int) -> str:
    """The supervisor's routing rules, in priority order."""
    if s["hops"] > max_hops:
        return "finish"
    if not s["guard_in_done"]:
        return "guard_in"
    if s["blocked"]:
        return "finish"
    if HAS_MEASUREMENT.search(s["user_text"]) and not s["calc_done"]:
        if s["error"] and s["retries"] > MAX_RETRIES:
            return "finish"
        return "calculator"
    ok = s["tool_output"].startswith("BMI:")
    if s["tool_output"].startswith("Error:"):
        return "finish"
    if ok and not s["reviewed"] and not (REVIEW_BMI_LOW <= s["bmi"] <= REVIEW_BMI_HIGH):
        return "review"
    if ok and not s["saved"]:
        return "memory"
    if not s["draft"]:
        return "advisor"
    if not s["guard_out_done"]:
        return "guard_out"
    return "finish"


def build_graph(workers, saver, store, max_hops: int = MAX_HOPS):
    def supervisor(s: State):
        hops = s["hops"] + 1
        nxt = decide({**s, "hops": hops}, max_hops)
        upd = {"hops": hops, "next": nxt, "log": _log(s, f"[hop {hops}] supervisor -> {nxt}")}
        if hops > max_hops:
            upd["status"] = "hop_limit_reached"
        return upd

    def guard_in(s: State):
        why = guard.check(s["user_text"], "input")
        log = _log(s, f"guard_in: {'BLOCK (' + why + ')' if why else 'safe'}")
        upd = {"guard_in_done": True, "log": log, "blocked": why}
        if why:
            upd.update(final=guard.MESSAGES[why], status=f"blocked_input:{why}")
        return upd

    def calculator(s: State, store: BaseStore):
        try:
            if s["fail_once"] and s["retries"] == 0:
                raise RuntimeError("injected tool failure")
            out = workers.calculate(s["user_text"], get_profile(store, s["student_id"]))
            if not out["tool_output"]:
                raise ValueError("model did not call the tool")
        except Exception as exc:
            return {"error": str(exc), "retries": s["retries"] + 1,
                    "log": _log(s, f"calculator: attempt {s['retries'] + 1} FAILED ({exc})")}
        text, args = out["tool_output"], out["args"] or {}
        m = re.search(r"BMI: ([\d.]+)", text)
        upd = {"error": "", "calc_done": True, "tool_output": text,
               "log": _log(s, f"calculator: {text[:60]}")}
        if m:
            upd.update(bmi=float(m.group(1)), weight=float(args.get("weight_kg", 0)),
                       height=float(args.get("height_cm", 0)))
        return upd

    def review(s: State):
        flag = f"weight={s['weight']} kg, height={s['height']} cm, BMI={s['bmi']}"
        decision = interrupt({"flag": flag, "question": "Approve this unusual reading? (yes/no)"})
        if str(decision).strip().lower().startswith("y"):
            return {"reviewed": True, "log": _log(s, "review: human APPROVED")}
        return {"reviewed": True, "blocked": "rejected",
                "final": "A human reviewer rejected this reading, so it was not stored.",
                "status": "rejected_by_human", "log": _log(s, "review: human REJECTED")}

    def memory(s: State, store: BaseStore):
        cat = re.search(r"Category: ([^|]+)\|", s["tool_output"]).group(1).strip()
        save_measurement(store, s["student_id"], s["weight"], s["height"], s["bmi"], cat)
        trend = trend_report(get_history(store, s["student_id"]))
        return {"saved": True, "trend": trend, "log": _log(s, f"memory: saved; {trend[:70]}")}

    def advisor(s: State, store: BaseStore):
        new_facts = extract_and_save_facts(store, s["student_id"], s["user_text"])
        facts = format_facts(get_user_facts(store, s["student_id"]))
        facts_context = f"\nUser profile facts: {facts}" if facts else ""

        is_question = bool(re.search(r"\?|what|how|why|when|where|can i|should i|tell me|explain|target|ideal", s["user_text"], re.I))
        has_measurement = bool(s["tool_output"].startswith("BMI:"))

        # If user shared qualitative health facts without asking a question or entering measurements:
        if new_facts and not is_question and not has_measurement:
            fact_str = ", ".join(new_facts)
            draft = (
                f"Got it! I've saved to your profile: **{fact_str}**.\n\n"
                "I will take this context into account for your nutrition and physical activity recommendations. "
                "Feel free to share your height and weight to check your BMI, or ask any health questions!"
            )
            return {"draft": draft, "facts": facts, "log": _log(s, f"advisor: recorded facts ({fact_str})")}

        data = s["tool_output"] if s["tool_output"] else ""
        if s["tool_output"].startswith("BMI:"):
            if is_question:
                question = f"{s['user_text']} (Context: {s['tool_output']})"
            else:
                question = f"My BMI result is: {s['tool_output']} What does this mean and what general guidance applies?"
        else:
            question = s["user_text"]
        trend_context = s['trend'] if s["tool_output"].startswith("BMI:") else ""
        draft = workers.advise(question, f"{data}\n{trend_context}{facts_context}".strip())
        return {"draft": draft, "facts": facts, "log": _log(s, "advisor: draft written (RAG)")}

    def guard_out(s: State):
        why = guard.check(s["draft"], "output")
        if not why and USE_LLM_JUDGE and workers.judge(s["draft"]):
            why = "diagnosis"
        upd = {"guard_out_done": True, "blocked": why,
               "log": _log(s, f"guard_out: {'BLOCK (' + why + ')' if why else 'safe'}")}
        if why:
            upd.update(final=guard.MESSAGES[why], status=f"blocked_output:{why}")
        return upd

    def finish(s: State):
        status = s.get("status", "")
        if status == "hop_limit_reached":
            final = "I stopped because the hop limit was reached. Please try a simpler request."
        elif status.startswith("blocked_output:") and s["tool_output"].startswith("BMI:"):
            # The calculation is valid & safe; include it and explain that advice was withheld
            final = f"{s['tool_output']}\n\n[Note: {s['final']}]"
        elif s.get("final"):
            final = s["final"]
        elif s.get("error"):
            status = "tool_failed"
            final = f"The BMI tool failed ({s['error']}) even after a retry. Please try again later."
        elif s["tool_output"].startswith("Error:"):
            status = "invalid_input"
            final = (f"I couldn't calculate that. {s['tool_output'][7:]} "
                     "Please check the values and try again.")
        else:
            status = "ok"
            trend_part = s["trend"] if s["tool_output"].startswith("BMI:") else ""
            final = "\n\n".join(x for x in [s["tool_output"], s["draft"], trend_part] if x)
        upd = {"final": final, "status": status or "ok", "messages": [AIMessage(final)],
               "log": _log(s, f"finish: status={status or 'ok'}")}
        if s["tool_output"].startswith("BMI:") and status == "ok":
            upd["last_bmi"] = s["tool_output"]
        return upd

    g = StateGraph(State)
    names = {"guard_in": guard_in, "calculator": calculator, "review": review, "memory": memory,
             "advisor": advisor, "guard_out": guard_out}
    g.add_node("supervisor", supervisor)
    g.add_node("finish", finish)
    for n, fn in names.items():
        g.add_node(n, fn)
        g.add_edge(n, "supervisor")
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", lambda s: s["next"], [*names, "finish"])
    g.add_edge("finish", END)
    return g.compile(checkpointer=saver, store=store)

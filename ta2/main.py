"""main.py - ONE command to run TA2:   python -m ta2.main --student alice
Flags: --k 1 (RAG k demo)  --no-memory (amnesia demo)  --inject-failure (failure demo)
       --max-hops N (hop-limit demo)  --ui (launch Gradio web interface)
Type 'trend' to print the stored BMI trend chart, 'quit' to exit.
"""
import argparse

from langgraph.types import Command

from .agents import Workers
from .config import DB_PATH, MAX_HOPS, RAG_K
from .graph import build_graph, turn_input
from .memory import get_history, open_db, trend_chart, trend_report
from .rag import get_vectorstore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", default="demo")
    ap.add_argument("--no-memory", action="store_true", help="new thread every turn (amnesia)")
    ap.add_argument("--inject-failure", action="store_true", help="tool fails once per turn")
    ap.add_argument("--max-hops", type=int, default=MAX_HOPS)
    ap.add_argument("--k", type=int, default=RAG_K, help="chunks retrieved by RAG (k=1 vs k=3 demo)")
    ap.add_argument("--ui", action="store_true", help="launch Gradio web UI instead of CLI")
    ap.add_argument("--share", action="store_true", help="generate public gradio.live link (opens without login)")
    a = ap.parse_args()

    # --- Improvement 6: optional Gradio web interface ---
    if a.ui or a.share:
        try:
            from .ui import launch_ui
        except ImportError:
            print("\n[!] Gradio is not installed. To use the web interface, install it with:\n    pip install gradio\n")
            return
        launch_ui(max_hops=a.max_hops, k=a.k, share=a.share)
        return

    saver, store = open_db(DB_PATH)
    graph = build_graph(Workers(get_vectorstore(), k=a.k), saver, store, a.max_hops)
    turn = 0
    print(f"BMI Advisor TA2 | student={a.student} | memory={'OFF' if a.no_memory else 'ON'} "
          "| type 'trend' or 'quit'")
    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text.lower() in {"quit", "exit"}:
            break
        if not text:
            continue
        if text.lower() == "trend":
            # Improvement 4: show ASCII bar chart if >= 2 records, else plain summary
            h = get_history(store, a.student)
            chart = trend_chart(h)
            print(chart if chart else trend_report(h))
            continue
        turn += 1
        thread = f"{a.student}-{turn}" if a.no_memory else f"{a.student}-main"
        cfg = {"configurable": {"thread_id": thread}}
        result = graph.invoke(turn_input(text, a.student, a.inject_failure), cfg)
        while result.get("__interrupt__"):                      # human-on-the-loop
            info = result["__interrupt__"][0].value
            print(f"[HUMAN REVIEW] Flagged: {info['flag']}")
            answer = input(f"Reviewer - {info['question']} ").strip()
            result = graph.invoke(Command(resume=answer), cfg)
        print(f"Agent: {result['final']}\n")


if __name__ == "__main__":
    main()

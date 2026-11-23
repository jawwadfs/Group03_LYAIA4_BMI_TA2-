"""ui.py - Optional Gradio web interface for TA2.

Launch with:   python -m ta2.main --student demo --ui
Requires:      pip install gradio  (already in requirements.txt)

Design decisions:
- One shared graph instance per server session (Gradio runs in a single process).
- Human-on-the-loop interrupts are handled inline: the agent returns a review prompt,
  the user types "yes"/"no", and the graph is resumed with Command(resume=...).
- The student ID textbox lets multiple users share one server (different thread_ids).
"""
import gradio as gr
from langgraph.types import Command

from .agents import Workers
from .config import DB_PATH, MAX_HOPS, RAG_K
from .graph import build_graph, turn_input
from .memory import get_history, open_db, trend_chart, trend_report
from .rag import get_vectorstore


def launch_ui(max_hops: int = MAX_HOPS, k: int = RAG_K, share: bool = False) -> None:
    saver, store = open_db(DB_PATH)
    graph = build_graph(Workers(get_vectorstore(), k=k), saver, store, max_hops)

    # Track students who are waiting for human-review resume
    _pending_review: dict[str, str] = {}   # student_id -> thread_id

    def respond(message: str, history: list, student: str) -> str:
        """Called by Gradio on each user message."""
        student = (student or "demo").strip()
        thread = f"{student}-main"
        cfg = {"configurable": {"thread_id": thread}}

        if student in _pending_review:
            # Resume from a human-review interrupt (user typed yes/no)
            result = graph.invoke(Command(resume=message), cfg)
            _pending_review.pop(student, None)
        else:
            result = graph.invoke(turn_input(message, student), cfg)

        # Human-on-the-loop: graph paused, ask user to approve/reject
        if result.get("__interrupt__"):
            info = result["__interrupt__"][0].value
            _pending_review[student] = thread
            return (
                f"⚠️ **Human Review Required**\n\n"
                f"**Flagged reading:** `{info['flag']}`\n\n"
                f"{info['question']} *(reply **yes** to approve or **no** to reject)*"
            )

        final = result.get("final", "")
        status = result.get("status", "ok")
        hops = result.get("hops", 0)

        # Append a small debug footer (easy to explain at the jury)
        footer = f"\n\n---\n*Status: `{status}` · Hops used: {hops} / {max_hops}*"
        return final + footer

    def show_trend(student: str) -> str:
        """Load and display the ASCII BMI trend chart for this student."""
        student = (student or "demo").strip()
        h = get_history(store, student)
        chart = trend_chart(h)
        return chart if chart else trend_report(h)

    # -----------------------------------------------------------------------
    # UI layout (compatible with Gradio 5.x and 6.x)
    # -----------------------------------------------------------------------
    css = """
    .gradio-container { max-width: 920px !important; margin: auto; }
    footer { display: none !important; }
    """

    with gr.Blocks(title="BMI & Health Advisor — TA2") as demo:

        gr.Markdown(
            "# 🏃 BMI & Health Advisor — TA2\n"
            "*Multi-agent system · Qwen2.5:3b · LangGraph · "
            "RAG (ChromaDB) · SQLite memory · Safety guards*"
        )

        with gr.Row():
            student_box = gr.Textbox(
                value="demo", label="Student ID",
                info="Each student gets their own memory thread.",
                scale=1, min_width=160,
            )
            gr.HTML("<div style='flex:3'></div>")  # spacer

        gr.ChatInterface(
            fn=respond,
            additional_inputs=[student_box],
            chatbot=gr.Chatbot(
                height=430,
                label="Conversation",
                placeholder="Type your message below and press Enter.",
            ),
            textbox=gr.Textbox(
                placeholder="e.g.  I weigh 70 kg and I am 175 cm tall",
                scale=5,
            ),
            examples=[
                ["I weigh 70 kg and I am 175 cm tall", "demo"],
                ["I weigh 154 lbs and I am 5 ft 10", "demo"],
                ["How many minutes of exercise do adults need per week?", "demo"],
                ["What does overweight BMI mean for my health?", "demo"],
                ["Do I have to exercise every day?", "demo"],
            ],
        )

        with gr.Accordion("📈 BMI Trend History", open=False):
            with gr.Row():
                trend_btn = gr.Button("Load Trend Chart", variant="secondary", scale=1)
                gr.HTML("<div style='flex:3'></div>")
            trend_out = gr.Textbox(
                label="",
                lines=14,
                show_label=False,
                placeholder="Click 'Load Trend Chart' to see your stored BMI history.",
            )
            trend_btn.click(show_trend, inputs=[student_box], outputs=[trend_out])

        gr.Markdown(
            "> **Known limits:** The 3B LLM may occasionally skip tool calls "
            "(caught by retry logic). The rule-based guard can miss paraphrased "
            "harmful prompts; the LLM judge is a second layer. Corpus covers "
            "general adult health only — not children or pregnancy."
        )

    demo.launch(inbrowser=True, show_error=True, share=share, theme=gr.themes.Soft(), css=css)

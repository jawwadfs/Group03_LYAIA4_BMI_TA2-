"""streamlit_app.py - Streamlit web interface for TA2 BMI Advisor.

Dual-Mode Architecture:
1. Cloud Deployment (Streamlit Cloud): Uses Groq API for 24/7 uptime.
2. Local Development: Automatically detects local Ollama (qwen2.5:3b).

Features:
- Multi-agent supervisor workflow with human-on-the-loop review
- Interactive Plotly BMI Zone progression charts
- Semantic episodic memory (qualitative user context)
- One-click print-ready clinical screening summary export
- Voice input with Groq Whisper-large-v3 transcription
"""
import os
import streamlit as st
from langgraph.types import Command

from ta2.agents import Workers
from ta2.config import DB_PATH, MAX_HOPS, RAG_K
from ta2.graph import build_graph, turn_input
from ta2.memory import (
    get_history,
    get_profile,
    get_user_facts,
    open_db,
    trend_chart,
    trend_report,
)
from ta2.rag import get_vectorstore
from ta2.report import generate_doctor_report_html

st.set_page_config(
    page_title="BMI & Health Advisor — TA2",
    page_icon="🏃",
    layout="centered",
)

st.title("🏃 BMI & Health Advisor — TA2")
st.caption("Agentic AI System · LangGraph · ChromaDB RAG · SQLite Memory · Dual Safety Guard")

# ---------------------------------------------------------------------------
# Backend Selection: Groq Cloud (24/7) or Local Ollama
# ---------------------------------------------------------------------------
groq_key = (
    st.secrets.get("GROQ_API_KEY", "")
    if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets
    else os.environ.get("GROQ_API_KEY", "")
)


@st.cache_data(ttl=600, show_spinner=False)
def get_available_groq_models(api_key: str) -> list[str]:
    """Fetch all active chat models for this Groq key."""
    import httpx
    try:
        r = httpx.get(
            "https://api.groq.com/openai/v1/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=5.0,
        )
        if r.status_code == 200:
            data = r.json()
            models = [
                m.get("id", "") for m in data.get("data", [])
                if "/" not in m.get("id", "") and not any(skip in m.get("id", "").lower() for skip in ["whisper", "vision", "embed", "tts", "guard", "audio", "orpheus"])
            ]
            if models:
                for prio in ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "qwen-2.5-32b"]:
                    if prio in models:
                        models.remove(prio)
                        models.insert(0, prio)
                return models
    except Exception:
        pass
    return ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]


selected_model = "llama-3.1-8b-instant"

with st.sidebar:
    st.header("⚙️ Configuration")
    if not groq_key:
        user_key = st.text_input(
            "Groq API Key (for 24/7 Cloud)",
            type="password",
            help="Free key from https://console.groq.com. Enables 24/7 cloud hosting without local Ollama.",
        )
        if user_key:
            groq_key = user_key.strip()

    if groq_key:
        models_list = get_available_groq_models(groq_key)
        selected_model = st.selectbox(
            "Active Groq Model",
            models_list,
            index=0,
            help="Choose from the models enabled on your Groq key.",
        )

    student_id = st.text_input(
        "Student / User ID",
        value="demo",
        help="Each user has an isolated memory thread in SQLite.",
    )

    st.divider()
    st.subheader("📈 BMI Progression")
    chart_placeholder = st.container()

    st.divider()
    st.subheader("📄 Clinical Report Export")
    export_placeholder = st.container()

    st.divider()
    st.markdown(
        "**System Architecture:**\n"
        "- **Supervisor:** Rule-based routing + Hop limit\n"
        "- **Calculator Agent:** Unit conversion, BMI & Ideal Weight tools\n"
        "- **Advisor Agent:** Grounded RAG + Source Citation\n"
        "- **Safety Guard:** Regex rules + LLM reviewer\n"
        "- **Episodic Memory:** Stores biometrics & personal health context\n"
        "- **Human-in-the-Loop:** Active for extreme BMI (<15 or >45)"
    )


def get_llm(model_id: str):
    """Instantiate either Groq Cloud LLM or Local Ollama ChatModel."""
    if groq_key:
        try:
            from langchain_groq import ChatGroq
            return ChatGroq(
                model_name=model_id,
                api_key=groq_key,
                temperature=0,
            ), f"Groq Cloud ({model_id})"
        except ImportError:
            st.error("Please install langchain-groq: `pip install langchain-groq`")
            st.stop()
    else:
        # Default to local Ollama
        try:
            from langchain_ollama import ChatOllama
            from ta2.config import MODEL
            ollama_llm = ChatOllama(model=MODEL, temperature=0)
            ollama_llm.invoke("test")
            return ollama_llm, f"Local Ollama ({MODEL})"
        except Exception:
            return None, "Unavailable"


# Cache graph across sessions
@st.cache_resource(show_spinner=False)
def init_agent(key_str: str, model_id: str):
    llm, backend_name = get_llm(model_id)
    if llm is None:
        return None, None, backend_name
    saver, store = open_db(DB_PATH)
    vs = get_vectorstore()
    graph = build_graph(Workers(vs, llm=llm, k=RAG_K), saver, store, max_hops=MAX_HOPS)
    return graph, store, backend_name


graph, store, backend_name = init_agent(groq_key, selected_model)

if graph is None:
    st.warning(
        "⚠️ **No active LLM backend detected.**\n\n"
        "To run locally: Ensure Ollama is running (`ollama serve`).\n\n"
        "To run 24/7 in the Cloud: Enter your free **Groq API Key** in the sidebar (get one in 10 seconds at [console.groq.com](https://console.groq.com))."
    )
    st.stop()
else:
    st.sidebar.success(f"🟢 **Active Backend:** {backend_name}")


# ---------------------------------------------------------------------------
# Interactive Plotly Chart Helper
# ---------------------------------------------------------------------------
def render_plotly_bmi_chart(history: list[dict]):
    """Render interactive Plotly chart with shaded WHO BMI zones."""
    try:
        import plotly.graph_objects as go
        if not history:
            st.info("No weigh-in records stored yet.")
            return

        dates = [h.get("ts", "")[:10] for h in history]
        bmis = [h.get("bmi", 0.0) for h in history]
        weights = [h.get("weight_kg", 0.0) for h in history]
        cats = [h.get("category", "") for h in history]

        hover_texts = [
            f"Date: {d}<br>BMI: {b:.2f}<br>Weight: {w} kg<br>Category: {c}"
            for d, b, w, c in zip(dates, bmis, weights, cats)
        ]

        fig = go.Figure()
        max_y = max(max(bmis) + 4, 36)

        # WHO BMI Bands
        fig.add_hrect(y0=0, y1=18.5, fillcolor="rgba(147, 197, 253, 0.25)", line_width=0,
                      annotation_text="Underweight", annotation_position="top left",
                      annotation_font_size=9, annotation_font_color="#2563eb")
        fig.add_hrect(y0=18.5, y1=24.9, fillcolor="rgba(134, 239, 172, 0.25)", line_width=0,
                      annotation_text="Normal (18.5-24.9)", annotation_position="top left",
                      annotation_font_size=9, annotation_font_color="#16a34a")
        fig.add_hrect(y0=25.0, y1=29.9, fillcolor="rgba(253, 224, 71, 0.25)", line_width=0,
                      annotation_text="Overweight (25-29.9)", annotation_position="top left",
                      annotation_font_size=9, annotation_font_color="#ca8a04")
        fig.add_hrect(y0=30.0, y1=max_y, fillcolor="rgba(252, 165, 165, 0.25)", line_width=0,
                      annotation_text="Obesity (≥ 30)", annotation_position="top left",
                      annotation_font_size=9, annotation_font_color="#dc2626")

        fig.add_trace(go.Scatter(
            x=dates,
            y=bmis,
            mode="lines+markers",
            name="BMI",
            line=dict(color="#1e3a8a", width=2.5),
            marker=dict(size=8, color="#1e3a8a", symbol="circle"),
            hoverinfo="text",
            hovertext=hover_texts,
        ))

        fig.update_layout(
            margin=dict(l=10, r=10, t=25, b=10),
            height=260,
            xaxis=dict(title="", showgrid=True, gridcolor="#f1f5f9"),
            yaxis=dict(title="BMI", range=[14, max_y], showgrid=True, gridcolor="#f1f5f9"),
            showlegend=False,
            template="plotly_white",
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        # Fallback to ASCII chart if plotly is not installed
        chart = trend_chart(history)
        if chart:
            st.code(chart, language="text")
        else:
            st.info(trend_report(history))


# Populate sidebar components
if store:
    with chart_placeholder:
        h_data = get_history(store, student_id)
        render_plotly_bmi_chart(h_data)

    with export_placeholder:
        prof_data = get_profile(store, student_id)
        facts_data = get_user_facts(store, student_id)
        report_html = generate_doctor_report_html(student_id, prof_data, h_data, facts_data)
        st.download_button(
            label="📥 Download Clinical Report (HTML)",
            data=report_html,
            file_name=f"health_screening_{student_id}.html",
            mime="text/html",
            help="Printable summary for medical consultation with biometrics & discussion questions.",
            use_container_width=True,
        )

# ---------------------------------------------------------------------------
# Chat Interface & Voice Input
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_review" not in st.session_state:
    st.session_state.pending_review = False
if "review_info" not in st.session_state:
    st.session_state.review_info = None

# Voice Input with Groq Whisper
voice_query = None
if hasattr(st, "audio_input") and groq_key:
    with st.expander("🎙️ Voice Input (Speak measurements or questions)", expanded=False):
        audio_val = st.audio_input("Record voice query", key="voice_recorder")
        if audio_val is not None:
            audio_bytes = audio_val.getvalue() if hasattr(audio_val, "getvalue") else audio_val.read()
            if audio_bytes:
                import hashlib
                audio_hash = hashlib.md5(audio_bytes).hexdigest()
                if st.session_state.get("last_processed_audio_hash") != audio_hash:
                    import httpx
                    try:
                        with st.spinner("Transcribing via Groq Whisper-large-v3..."):
                            resp = httpx.post(
                                "https://api.groq.com/openai/v1/audio/transcriptions",
                                headers={"Authorization": f"Bearer {groq_key}"},
                                files={"file": ("recording.wav", audio_bytes, "audio/wav")},
                                data={"model": "whisper-large-v3"},
                                timeout=15.0,
                            )
                            if resp.status_code == 200:
                                voice_query = resp.json().get("text", "").strip()
                                st.session_state.last_processed_audio_hash = audio_hash
                                st.success(f"🗣️ Transcribed: *\"{voice_query}\"*")
                    except Exception as e:
                        st.warning(f"Voice note: {e}")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Human-on-the-loop interrupt handling
if st.session_state.pending_review and st.session_state.review_info:
    info = st.session_state.review_info
    st.warning(
        f"⚠️ **Human Review Required**\n\n"
        f"**Flagged Reading:** `{info['flag']}`\n\n"
        f"{info['question']}"
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Approve Reading", use_container_width=True):
            thread = f"{student_id}-main"
            cfg = {"configurable": {"thread_id": thread}}
            res = graph.invoke(Command(resume="yes"), cfg)
            st.session_state.pending_review = False
            st.session_state.review_info = None
            final_text = res.get("final", "Approved.")
            output = f"{final_text}\n\n---\n*Status: `ok` (Approved by Human Reviewer)*"
            st.session_state.messages.append({"role": "assistant", "content": output})
            st.rerun()
    with c2:
        if st.button("❌ Reject Reading", use_container_width=True):
            thread = f"{student_id}-main"
            cfg = {"configurable": {"thread_id": thread}}
            res = graph.invoke(Command(resume="no"), cfg)
            st.session_state.pending_review = False
            st.session_state.review_info = None
            final_text = res.get("final", "A human reviewer rejected this reading, so it was not stored.")
            output = f"{final_text}\n\n---\n*Status: `rejected_by_human`*"
            st.session_state.messages.append({"role": "assistant", "content": output})
            st.rerun()

# Determine incoming user query (only when no review is pending)
if not st.session_state.pending_review:
    chat_input_val = st.chat_input("Ask advice or enter measurements (e.g. 'I weigh 70 kg and I am 175 cm')")
    active_prompt = voice_query or chat_input_val

    if active_prompt:
        st.session_state.messages.append({"role": "user", "content": active_prompt})
        with st.chat_message("user"):
            st.markdown(active_prompt)

        thread = f"{student_id}-main"
        cfg = {"configurable": {"thread_id": thread}}

        with st.chat_message("assistant"):
            with st.spinner("Processing through multi-agent supervisor..."):
                result = graph.invoke(turn_input(active_prompt, student_id), cfg)

            if result.get("__interrupt__"):
                info = result["__interrupt__"][0].value
                st.session_state.pending_review = True
                st.session_state.review_info = info
                st.rerun()
            else:
                final_text = result.get("final", "")
                status = result.get("status", "ok")
                hops = result.get("hops", 0)

                output = f"{final_text}\n\n---\n*Status: `{status}` · Hops: {hops}/{MAX_HOPS}*"
                st.markdown(output)
                st.session_state.messages.append({"role": "assistant", "content": output})

"""streamlit_app.py - Streamlit web interface for TA2 BMI Advisor.

Dual-Mode Architecture:
1. Cloud Deployment (Streamlit Cloud): Uses Groq API (llama-3.3-70b / qwen-2.5-32b) for 24/7 uptime.
2. Local Development: Automatically detects local Ollama (qwen2.5:3b).

Run locally:
    streamlit run streamlit_app.py
"""
import os
import streamlit as st
from langgraph.types import Command

from ta2.agents import Workers
from ta2.config import DB_PATH, MAX_HOPS, RAG_K
from ta2.graph import build_graph, turn_input
from ta2.memory import get_history, open_db, trend_chart, trend_report
from ta2.rag import get_vectorstore

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
# Look for Groq API key in secrets, env, or sidebar
groq_key = (
    st.secrets.get("GROQ_API_KEY", "")
    if hasattr(st, "secrets") and "GROQ_API_KEY" in st.secrets
    else os.environ.get("GROQ_API_KEY", "")
)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_groq_models(api_key: str) -> list[str]:
    """Query Groq API directly to retrieve the exact models available for this key."""
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
                m["id"] for m in data.get("data", [])
                if not any(skip in m["id"].lower() for skip in ["whisper", "vision", "embed", "tts", "guard"])
            ]
            if models:
                # Prioritize standard tool-capable models if available
                for p in ["llama-3.1-8b-instant", "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768"]:
                    if p in models:
                        models.remove(p)
                        models.insert(0, p)
                return models
    except Exception:
        pass
    return ["llama3-8b-8192", "llama3-70b-8192", "mixtral-8x7b-32768", "gemma2-9b-it"]


selected_model = "llama3-8b-8192"

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
        available_models = fetch_groq_models(groq_key)
        selected_model = st.selectbox(
            "Active Groq Model",
            available_models,
            index=0,
            help="Models retrieved live from your Groq account.",
        )

    student_id = st.text_input(
        "Student ID",
        value="demo",
        help="Each student has an isolated memory thread in SQLite.",
    )

    st.divider()
    st.subheader("📈 Stored BMI Trend")
    trend_container = st.container()

    st.divider()
    st.markdown(
        "**System Architecture:**\n"
        "- **Supervisor:** Rule-based routing\n"
        "- **Calculator Agent:** Unit conversion & tool execution\n"
        "- **Advisor Agent:** Grounded RAG + Source Citation\n"
        "- **Safety Guard:** Dual-layer (rules + LLM reviewer)\n"
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
            # Ping to verify Ollama is reachable
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

# Populate sidebar trend
if store:
    with trend_container:
        if st.button("Refresh Trend Chart"):
            h = get_history(store, student_id)
            chart = trend_chart(h)
            if chart:
                st.code(chart, language="text")
            else:
                st.info(trend_report(h))

# ---------------------------------------------------------------------------
# Chat Interface
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pending_review" not in st.session_state:
    st.session_state.pending_review = False
if "review_info" not in st.session_state:
    st.session_state.review_info = None

# Display message history
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
            st.session_state.messages.append({"role": "assistant", "content": res.get("final", "Approved.")})
            st.rerun()
    with c2:
        if st.button("❌ Reject Reading", use_container_width=True):
            thread = f"{student_id}-main"
            cfg = {"configurable": {"thread_id": thread}}
            res = graph.invoke(Command(resume="no"), cfg)
            st.session_state.pending_review = False
            st.session_state.review_info = None
            st.session_state.messages.append({"role": "assistant", "content": res.get("final", "Rejected.")})
            st.rerun()

# User input
if prompt := st.chat_input("Ask advice or enter measurements (e.g. 'I weigh 70 kg and I am 175 cm')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    thread = f"{student_id}-main"
    cfg = {"configurable": {"thread_id": thread}}

    with st.chat_message("assistant"):
        with st.spinner("Processing through multi-agent supervisor..."):
            result = graph.invoke(turn_input(prompt, student_id), cfg)

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

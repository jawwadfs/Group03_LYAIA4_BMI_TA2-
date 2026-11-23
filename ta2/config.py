"""config.py - every tunable constant for TA2 lives here (easy to explain in the viva)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)

MODEL = "qwen2.5:3b"            # chat model (Ollama)
EMBED_MODEL = "nomic-embed-text"  # embedding model (Ollama)

DB_PATH = str(DATA / "memory.sqlite")   # checkpointer + long-term Store (survives restart)
CHROMA_DIR = str(DATA / "chroma")       # persisted vector index
CORPUS = ROOT / "corpus" / "guidelines.md"

MAX_HOPS = 10        # supervisor hop limit (normal request uses 6)
MAX_RETRIES = 1      # tool-failure retries before we explain and stop
RAG_K = 3            # chunks retrieved per question
USE_LLM_JUDGE = True # guard agent: second layer (LLM) on top of rules

# Human-on-the-loop: valid-but-unusual readings are flagged for a person
REVIEW_BMI_LOW, REVIEW_BMI_HIGH = 15.0, 45.0

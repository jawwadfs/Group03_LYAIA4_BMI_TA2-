"""rag.py - Unit 3 RAG: Chroma (persisted) + nomic-embed-text + grounded prompt."""
import math, re
from collections import Counter

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings

from .config import CHROMA_DIR, CORPUS, EMBED_MODEL

NOT_IN_DOC = "I don't know - that is not covered in the health-guidelines document."

GROUNDED_SYSTEM = f"""You are the knowledgeable, empathetic Advisor agent of a BMI & Health tool.
Your responsibilities:
1. Ground clinical advice in the CONTEXT excerpts (WHO guidelines) for BMI cut-offs, activity goals, and safe weight change rates.
2. Directly respect and personalize using the USER DATA (diet, exercise preferences, reported limitations, and biometrics).
3. If the user clarifies their personal capabilities or preferences (e.g., capable of high-impact exercise, dietary restrictions), acknowledge and tailor your recommendations to them.
4. If a question asks about medical conditions, cures, medications, or topics completely outside the guidelines document, reply: {NOT_IN_DOC}
5. Never diagnose illnesses, never recommend medicines or extreme crash diets.
6. Provide helpful, encouraging guidance in 2-4 sentences, and remind the user BMI is only a general screening indicator."""


class SimpleKeywordEmbeddings(Embeddings):
    """Zero-dependency fallback embeddings for cloud deployments when Ollama is unavailable."""
    def __init__(self, chunks=None):
        docs = chunks or load_chunks()
        self.vocab = {}
        for d in docs:
            words = re.findall(r"\w+", d.page_content.lower())
            for w in words:
                if w not in self.vocab:
                    self.vocab[w] = len(self.vocab)
        self.dim = len(self.vocab)

    def _vec(self, text: str) -> list[float]:
        words = re.findall(r"\w+", text.lower())
        c = Counter(words)
        v = [0.0] * self.dim
        norm = 0.0
        for w, count in c.items():
            if w in self.vocab:
                idx = self.vocab[w]
                v[idx] = float(count)
                norm += count * count
        norm = math.sqrt(norm) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def load_chunks(path=CORPUS) -> list[Document]:
    """Split the corpus on '## ' headings: one chunk per heading."""
    text = path.read_text(encoding="utf-8")
    parts = re.split(r"^## ", text, flags=re.M)[1:]
    docs = []
    for p in parts:
        title, _, body = p.partition("\n")
        docs.append(Document(page_content=f"{title.strip()}: {body.strip()}",
                             metadata={"source": title.strip()}))
    return docs


def get_vectorstore(embeddings=None, persist_directory=CHROMA_DIR) -> Chroma:
    """Open (or build on first run) the persisted Chroma index."""
    if embeddings is None:
        try:
            # Check if Ollama is available
            cand = OllamaEmbeddings(model=EMBED_MODEL)
            cand.embed_query("ping")
            embeddings = cand
        except Exception:
            # Ollama is not running (e.g. cloud deployment); fall back to zero-dependency embeddings
            embeddings = SimpleKeywordEmbeddings()
            persist_directory = f"{persist_directory}_cloud"

    vs = Chroma(collection_name="health_guidelines",
                embedding_function=embeddings,
                persist_directory=persist_directory)
    if not vs.get()["ids"]:
        vs.add_documents(load_chunks())
    return vs


def retrieve(vs: Chroma, question: str, k: int = 3) -> list[Document]:
    return vs.similarity_search(question, k=k)


def format_context(docs: list[Document]) -> str:
    return "\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))

import logging
import os
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

# Embedding provider selection via env:
#   KC_EMBEDDING_DISABLED=1 -> no embeddings
#   KC_MODEL_PROVIDER=modelscope -> use modelscope Chinese model
#   KC_MODEL_PROVIDER=huggingface (default) -> use HF MiniLM
PROVIDER = os.getenv("KC_MODEL_PROVIDER", "huggingface").lower().strip()
DISABLE = os.getenv("KC_EMBEDDING_DISABLED", "0") == "1"
LOCAL_DIR = "C:\\Users\\LIHA11\\Downloads\\Traffic_AI\\paraphrase-MiniLM-L6-v2"

SentenceTransformer = None
snapshot_download = None
if not DISABLE:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer(LOCAL_DIR)
    except Exception:
        SentenceTransformer = None
    if PROVIDER == "modelscope":
        try:
            from modelscope import snapshot_download  # type: ignore
        except Exception as e:
            logging.warning(f"ModelScope import failed, will fallback to HF provider: {e}")
            snapshot_download = None

logger = logging.getLogger(__name__)
app = FastAPI(title="Local Knowledge Center", version="0.1.0")

class DocIn(BaseModel):
    id: str
    text: str

class QueryIn(BaseModel):
    text: str
    top_k: int = 3

# In-memory storage
_DOCS: List[DocIn] = []
_EMBEDDINGS = {}
_MODEL = None

if not DISABLE and SentenceTransformer:
    try:
        if LOCAL_DIR and os.path.isdir(LOCAL_DIR):
            _MODEL = SentenceTransformer(LOCAL_DIR)
            logger.info("Loaded embedding model from LOCAL DIR: %s", LOCAL_DIR)
        elif PROVIDER == "modelscope" and snapshot_download:
            ms_dir = snapshot_download('damo/nlp_sentence-embedding_chinese-base')
            _MODEL = SentenceTransformer(ms_dir)
            logger.info("Loaded ModelScope Chinese sentence embedding model from %s", ms_dir)
        else:
            # 国内镜像（如清华）可用时可设置 HF_ENDPOINT 环境变量
            import os
            os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
            _MODEL = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L6-v2")
            logger.info("Loaded HuggingFace MiniLM embedding model via hf-mirror.com")
    except Exception as e:
        logger.warning("Embedding model load failed (%s). Falling back to heuristic matching.", e)
        _MODEL = None
else:
    if DISABLE:
        logger.info("KC_EMBEDDING_DISABLED=1 -> embeddings disabled; using heuristic matching.")

@app.post("/add")
def add(doc: DocIn):
    _DOCS.append(doc)
    if _MODEL:
        emb = _MODEL.encode(doc.text)
        _EMBEDDINGS[doc.id] = emb
    return {"ok": True, "count": len(_DOCS)}

@app.post("/query")
def query(q: QueryIn):
    if not _DOCS:
        return {"results": [], "total": 0}
    if _MODEL:
        try:
            q_emb = _MODEL.encode(q.text)
            scored = []
            for d in _DOCS:
                emb = _EMBEDDINGS.get(d.id)
                if emb is None:
                    continue
                # cosine similarity
                import numpy as np  # local import keeps base deps minimal if embeddings disabled
                sim = float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-9))
                scored.append((sim, d.text))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [t for _, t in scored[: q.top_k]]
        except Exception as e:
            logger.warning("Embedding similarity failed (%s); falling back to heuristic.", e)
            results = [d.text for d in _DOCS[: q.top_k]]
    else:
        # simple substring / length heuristic
        results = sorted(_DOCS, key=lambda d: (q.text in d.text, len(d.text)), reverse=True)[: q.top_k]
        results = [d.text for d in results]
    return {"results": results, "total": len(_DOCS)}

@app.get("/documents/embeddings/search")
def documents_embeddings_search(
    query_text: str,
    search_type: str = "title",
    num_results: int = 2,
    title_weight: float = 0.5,
    distance_method: str = "cosine",
    domains: str | None = None,
):
    """Compatibility endpoint to emulate remote Knowledge Center search API.

    Returns a structure matching what `KnowledgeCenter.get_documents` expects:
    {
      "documents": [
         {"document": {"DocumentID": ..., "Content": ..., "DocumentTitle": ...}, "similarity": float}, ...
      ]
    }

    Heuristic scoring rules:
    - If embeddings loaded: cosine similarity of query/doc embedding.
    - Else: score = 1.0 if query_text substring in doc.text (case-insensitive) else 0.5 * length match ratio.
    - Results sorted descending by similarity.
    - DocumentTitle uses first 64 chars of text (single-line sanitized).
    - Ignores search_type, title_weight, distance_method, domains in local mode (reserved for future).
    """
    if not _DOCS:
        return {"documents": []}

    scored = []
    try:
        if _MODEL:
            q_emb = _MODEL.encode(query_text)
            import numpy as np
            for d in _DOCS:
                emb = _EMBEDDINGS.get(d.id)
                if emb is None:
                    continue
                sim = float(np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-9))
                scored.append((sim, d))
        else:
            qt_lower = query_text.lower()
            for d in _DOCS:
                txt_lower = d.text.lower()
                contains = qt_lower in txt_lower
                ratio = min(len(query_text), len(d.text)) / (max(len(query_text), len(d.text)) + 1e-9)
                sim = 1.0 if contains else 0.5 * ratio
                scored.append((sim, d))
    except Exception as e:
        logger.warning(f"Similarity calculation failed: {e}; returning unsorted docs")
        scored = [(1.0, d) for d in _DOCS]

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[: max(1, num_results)]
    documents = []
    for sim, d in top:
        title = d.text.strip().replace("\n", " ")[:64]
        documents.append({
            "document": {
                "DocumentID": d.id,
                "Content": d.text,
                "DocumentTitle": title or "Untitled"
            },
            "similarity": sim
        })
    return {"documents": documents}

@app.get("/health")
def health():
    return {"status": "ok", "docs": len(_DOCS)}

@app.on_event("startup")
def _log_routes():
    # Helpful for debugging 404 issues
    route_paths = [r.path for r in app.routes]
    logger.info("[knowledge_center_local] Registered routes: %s", route_paths)

# Run: uvicorn app.local_services.knowledge_center_local:app --port 8001 --reload

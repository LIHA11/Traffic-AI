import logging
from typing import Dict, Any, Optional
from fastapi import FastAPI
from pydantic import BaseModel

logger = logging.getLogger(__name__)
app = FastAPI(title="Local Working Memory Service", version="0.1.0")

_STORE: Dict[str, Dict[str, Any]] = {}

class MemorySetIn(BaseModel):
    key: str
    session_id: str
    content: Any
    meta_data: Optional[Dict[str, Any]] = None

class MemoryGetIn(BaseModel):
    key: str
    session_id: str

@app.post("/memory/set")
def memory_set(req: MemorySetIn):
    session_bucket = _STORE.setdefault(req.session_id, {})
    session_bucket[req.key] = {
        "content": req.content,
        "meta_data": req.meta_data or {}
    }
    return {"ok": True, "session_id": req.session_id, "key": req.key}

@app.post("/memory/get")
def memory_get(req: MemoryGetIn):
    session_bucket = _STORE.get(req.session_id, {})
    value = session_bucket.get(req.key)
    if not value:
        return {"error": "not_found", "session_id": req.session_id, "key": req.key}
    return {"ok": True, **value, "session_id": req.session_id, "key": req.key}

@app.get("/health")
def health():
    total_keys = sum(len(v) for v in _STORE.values())
    return {"status": "ok", "sessions": len(_STORE), "keys": total_keys}

@app.on_event("startup")
def _log_routes():
    # Log registered routes for debugging
    import inspect
    paths = [r.path for r in app.routes]
    logger.info("[working_memory_local] Registered routes: %s", paths)

# Run: uvicorn app.local_services.working_memory_local:app --port 8003 --reload

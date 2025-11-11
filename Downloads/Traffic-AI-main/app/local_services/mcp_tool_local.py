import json
import logging
import time
from typing import Dict, Any, Callable
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse  # FastAPI re-exports removed in newer versions
from pydantic import BaseModel

logger = logging.getLogger(__name__)
app = FastAPI(title="Local MCP Tool SSE", version="0.1.0")

# Define mock tools
TOOLS = [
    {"name": "query_kc", "description": "Query local knowledge center", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "echo", "description": "Echo back input", "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
]

# Simple execution registry
_EXECUTORS: Dict[str, Callable[[Dict[str, Any]], Any]] = {}

def _register(name: str):
    def deco(fn: Callable[[Dict[str, Any]], Any]):
        _EXECUTORS[name] = fn
        return fn
    return deco

@_register("echo")
def _exec_echo(payload: Dict[str, Any]):
    return {"echo": payload.get("text")}

@_register("query_kc")
def _exec_query_kc(payload: Dict[str, Any]):
    # This mock just returns the query text; integrate with knowledge center via HTTP if desired.
    return {"result": f"You asked for: {payload.get('query')} (mocked)"}

@app.get("/traffic-copilot/sse")
def sse():
    def event_stream():
        # Initial tools event
        init = {"event": "tools", "data": TOOLS}
        yield f"data: {json.dumps(init)}\n\n"
        # Heartbeats
        while True:
            time.sleep(15)
            yield f"data: {json.dumps({'event':'ping'})}\n\n"
    return EventSourceResponse(event_stream())

class ExecIn(BaseModel):
    tool: str
    arguments: Dict[str, Any] = {}

@app.post("/traffic-copilot/execute")
def execute(req: ExecIn):
    fn = _EXECUTORS.get(req.tool)
    if not fn:
        return {"error": f"Unknown tool {req.tool}"}
    try:
        res = fn(req.arguments)
        return {"ok": True, "data": res}
    except Exception as e:
        logger.exception("Tool execution failed")
        return {"ok": False, "error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok", "tools": len(TOOLS)}

# Run: uvicorn app.local_services.mcp_tool_local:app --port 8002 --reload

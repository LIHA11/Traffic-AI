"""Convenience launcher for local mock services.

Starts knowledge center (8001), MCP tool SSE (8002), and working memory (8003).

Usage (PowerShell):
  python app/local_services/run_local_stack.py
"""
import multiprocessing
import uvicorn
import os
import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so 'app.*' imports work even when running from app/ directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO)

SERVICES = [
    ("app.local_services.knowledge_center_local:app", 8001),
    ("app.local_services.mcp_tool_local:app", 8002),
    ("app.local_services.working_memory_local:app", 8003),
]

DEV_RELOAD = os.getenv("DEV_RELOAD", "0") == "1"

def _run(target: str, port: int):
    logging.info(f"Starting {target} on {port} (reload={'ON' if DEV_RELOAD else 'OFF'})")
    uvicorn.run(
        target,
        host="0.0.0.0",
        port=port,
        reload=DEV_RELOAD,  # allow hot-reload in local dev if DEV_RELOAD=1
        access_log=False,
        log_level="info",
    )

if __name__ == "__main__":
    procs = []
    try:
        for target, port in SERVICES:
            p = multiprocessing.Process(target=_run, args=(target, port), daemon=True)
            p.start()
            procs.append(p)
        logging.info("All services started. Press Ctrl+C to stop.")
        for p in procs:
            p.join()
    except KeyboardInterrupt:
        logging.info("Shutting down...")
        for p in procs:
            p.terminate()

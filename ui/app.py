"""TRIT-TRT Web UI — FastAPI server with WebSocket streaming."""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from trit_trt.config import (
    TRTConfig, SelectionMethod, ReflectionDepth,
)
from trit_trt.knowledge_store import KnowledgeStore
from ui.streaming import StreamingTRTEngine

logger = logging.getLogger("trit-trt.ui")

STATIC_DIR = Path(__file__).parent / "static"

# ─── Global state ────────────────────────────────────────────

_backend = None
_knowledge: Optional[KnowledgeStore] = None
_busy = False
_current_engine: Optional[StreamingTRTEngine] = None


def _get_backend():
    """Lazily create the InferenceBackend for BitNet."""
    global _backend
    if _backend is None:
        from trit_trt.engine import InferenceBackend
        _backend = InferenceBackend("microsoft/BitNet-b1.58-2B-4T")
        _backend.setup()
    return _backend


def _get_knowledge() -> KnowledgeStore:
    """Lazily create the KnowledgeStore."""
    global _knowledge
    if _knowledge is None:
        _knowledge = KnowledgeStore(max_entries=50)
    return _knowledge


# ─── FastAPI app ─────────────────────────────────────────────

app = FastAPI(title="TRIT-TRT")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok", "busy": _busy}


# ─── WebSocket endpoint ─────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global _busy, _current_engine
    await ws.accept()

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "cancel":
                if _current_engine is not None:
                    _current_engine.cancel()
                continue

            if msg_type != "generate":
                await ws.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})
                continue

            if _busy:
                await ws.send_json({"type": "error", "message": "Engine is busy"})
                continue

            _busy = True
            try:
                await _handle_generate(ws, data)
            finally:
                _busy = False
                _current_engine = None

    except WebSocketDisconnect:
        _busy = False
        _current_engine = None


async def _handle_generate(ws: WebSocket, data: dict) -> None:
    """Handle a generate request: build config, run engine in thread, forward events."""
    global _current_engine

    prompt = data.get("prompt", "")
    settings = data.get("settings", {})

    # Build TRT config from settings
    rounds = settings.get("rounds", 3)
    candidates = settings.get("candidates", 8)
    selection_method_str = settings.get("selection_method", "self_consistency")
    reflection_depth_str = settings.get("reflection_depth", "standard")
    early_stop_threshold = settings.get("early_stop_threshold", 0.95)
    knowledge_persistence = settings.get("knowledge_persistence", True)
    max_tokens = settings.get("max_tokens", 512)
    temperature = settings.get("temperature", None)

    try:
        selection_method = SelectionMethod(selection_method_str)
    except ValueError:
        selection_method = SelectionMethod.SELF_CONSISTENCY

    try:
        reflection_depth = ReflectionDepth(reflection_depth_str)
    except ValueError:
        reflection_depth = ReflectionDepth.STANDARD

    config = TRTConfig(
        rounds=rounds,
        candidates_per_round=candidates,
        selection_method=selection_method,
        reflection_depth=reflection_depth,
        early_stop_threshold=early_stop_threshold,
        knowledge_persistence=knowledge_persistence,
    )

    # Get backend and optionally set temperature
    backend = _get_backend()
    if temperature is not None and hasattr(backend, "_bitnet") and backend._bitnet is not None:
        backend._bitnet.config.temperature = temperature

    # Knowledge store
    knowledge = _get_knowledge() if knowledge_persistence else KnowledgeStore()

    # Event bridge: thread -> asyncio
    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue[dict | None] = asyncio.Queue()

    def on_event(event: dict) -> None:
        loop.call_soon_threadsafe(event_queue.put_nowait, event)

    engine = StreamingTRTEngine(
        generator=backend,
        config=config,
        knowledge_store=knowledge,
        on_event=on_event,
    )
    _current_engine = engine

    # Run the engine in a background thread
    def run_engine():
        try:
            engine.run(prompt, max_new_tokens=max_tokens)
        except Exception as exc:
            loop.call_soon_threadsafe(
                event_queue.put_nowait,
                {"type": "error", "message": str(exc)},
            )
        finally:
            loop.call_soon_threadsafe(event_queue.put_nowait, None)

    thread = threading.Thread(target=run_engine, daemon=True)
    thread.start()

    # Forward events from queue to WebSocket
    while True:
        event = await event_queue.get()
        if event is None:
            break
        await ws.send_json(event)
        if event.get("type") in ("result", "error", "cancelled"):
            break


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)

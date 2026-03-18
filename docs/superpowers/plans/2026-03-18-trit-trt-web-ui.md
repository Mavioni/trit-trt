# TRIT-TRT Web UI Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a chat-style web UI that streams TRT round progress in real-time via WebSocket.

**Architecture:** FastAPI serves static files and a WebSocket endpoint. The WebSocket handler wraps TritTRT's existing engine, intercepting each TRT phase to push live events to the client. Vanilla HTML/CSS/JS frontend — no build step.

**Tech Stack:** FastAPI, uvicorn, WebSocket, vanilla JS, CSS custom properties

**Spec:** `docs/superpowers/specs/2026-03-17-trit-trt-web-ui-design.md`

---

## File Structure

| File | Responsibility |
|------|---------------|
| `ui/__init__.py` | Package marker |
| `ui/app.py` | FastAPI app, WebSocket handler, engine lifecycle |
| `ui/streaming.py` | StreamingTRTEngine — wraps TRTEngine to emit events per phase |
| `ui/static/index.html` | Single-page HTML shell |
| `ui/static/style.css` | Dark minimal theme |
| `ui/static/main.js` | WebSocket client, DOM rendering, settings panel |
| `tests/test_ui.py` | WebSocket protocol tests using FastAPI TestClient |

---

### Task 1: Install dependencies and scaffold files

**Files:**
- Create: `ui/__init__.py`
- Create: `ui/app.py` (skeleton)
- Create: `ui/streaming.py` (skeleton)
- Create: `ui/static/index.html` (empty shell)
- Create: `ui/static/style.css` (empty)
- Create: `ui/static/main.js` (empty)
- Create: `tests/test_ui.py` (empty)

- [ ] **Step 1: Install FastAPI and uvicorn in conda env**

```bash
conda activate trit-trt
pip install fastapi "uvicorn[standard]" websockets
```

- [ ] **Step 2: Create directory structure**

```bash
mkdir -p ui/static
touch ui/__init__.py
```

- [ ] **Step 3: Create minimal app.py that serves a health check**

```python
# ui/app.py
"""TRIT-TRT Web UI — FastAPI server."""

from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="TRIT-TRT")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
```

- [ ] **Step 4: Create minimal index.html**

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TRIT-TRT</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <div id="app">TRIT-TRT loading...</div>
    <script src="/static/main.js"></script>
</body>
</html>
```

- [ ] **Step 5: Verify server starts**

```bash
cd ~/trit-trt
python -m ui.app
# Visit http://localhost:8765 — should show "TRIT-TRT loading..."
# Visit http://localhost:8765/health — should return {"status": "ok"}
# Ctrl+C to stop
```

- [ ] **Step 6: Commit**

```bash
git add ui/ tests/test_ui.py
git commit -m "feat(ui): scaffold FastAPI server with static file serving"
```

---

### Task 2: StreamingTRTEngine — WebSocket event emitter

The key integration piece. This wraps the existing TRT engine to call an event callback at each phase boundary, without modifying the engine source.

**Files:**
- Create: `ui/streaming.py`
- Create: `tests/test_ui.py` (streaming tests)

- [ ] **Step 1: Write failing test for StreamingTRTEngine**

```python
# tests/test_ui.py
"""Tests for the TRIT-TRT Web UI components."""

import pytest
from trit_trt.trt_engine import TRTEngine
from trit_trt.config import TRTConfig, SelectionMethod, ReflectionDepth
from tests.test_engine import MockGenerator


class TestStreamingTRTEngine:
    def test_emits_events_in_order(self):
        """StreamingTRTEngine should emit status events for each phase."""
        from ui.streaming import StreamingTRTEngine

        events = []
        generator = MockGenerator()
        config = TRTConfig(
            rounds=1,
            candidates_per_round=2,
            reflection_depth=ReflectionDepth.MINIMAL,
        )

        def on_event(event: dict):
            events.append(event)

        engine = StreamingTRTEngine(generator, config, on_event=on_event)
        result = engine.run("What is 2+2?")

        # Should have: generating, candidates, selecting, selected, reflecting, insight(s), result
        event_types = [e["type"] for e in events]
        assert "status" in event_types
        assert "candidates" in event_types
        assert "selected" in event_types
        assert "result" in event_types
        assert result.text != ""

    def test_cancel_stops_between_rounds(self):
        """Setting cancelled flag should stop after current round."""
        from ui.streaming import StreamingTRTEngine

        events = []
        generator = MockGenerator()
        config = TRTConfig(
            rounds=5,
            candidates_per_round=2,
            reflection_depth=ReflectionDepth.MINIMAL,
            early_stop_threshold=1.0,  # Don't early stop
        )

        def on_event(event: dict):
            events.append(event)

        engine = StreamingTRTEngine(generator, config, on_event=on_event)
        engine.cancel()
        result = engine.run("Test")

        # Should have cancelled event
        event_types = [e["type"] for e in events]
        assert "cancelled" in event_types
        assert result.rounds_used <= 1
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_ui.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'ui.streaming'`

- [ ] **Step 3: Implement StreamingTRTEngine**

```python
# ui/streaming.py
"""Streaming TRT Engine — emits WebSocket events at each TRT phase."""

from __future__ import annotations
import logging
from typing import Callable, Optional

from trit_trt.trt_engine import (
    TRTEngine, TRTResult, TRTRoundResult, TRTCandidate, TextGenerator,
)
from trit_trt.config import TRTConfig
from trit_trt.knowledge_store import KnowledgeStore

logger = logging.getLogger("trit-trt.streaming")


class StreamingTRTEngine(TRTEngine):
    """
    TRTEngine subclass that calls on_event() at each phase boundary.

    Events emitted:
      {"type": "status", "phase": "generating", "round": N, "total_rounds": M}
      {"type": "candidates", "count": K, "texts": [...], "round": N}
      {"type": "status", "phase": "selecting", "round": N}
      {"type": "selected", "text": "...", "confidence": 0.67, "round": N}
      {"type": "status", "phase": "reflecting", "round": N}
      {"type": "insight", "text": "...", "confidence": 0.6, "round": N}
      {"type": "result", "text": "...", "confidence": 0.95, ...}
      {"type": "cancelled"}
      {"type": "error", "message": "..."}
    """

    def __init__(
        self,
        generator: TextGenerator,
        config: Optional[TRTConfig] = None,
        knowledge_store: Optional[KnowledgeStore] = None,
        on_event: Optional[Callable[[dict], None]] = None,
    ):
        super().__init__(generator, config, knowledge_store)
        self._on_event = on_event or (lambda e: None)
        self._cancelled = False

    def emit(self, event: dict) -> None:
        """Send an event to the callback."""
        self._on_event(event)

    def cancel(self) -> None:
        """Request cancellation. Takes effect between phases."""
        self._cancelled = True

    def reset_cancel(self) -> None:
        """Clear the cancellation flag for a new run."""
        self._cancelled = False

    def run(self, prompt: str, max_new_tokens: int = 512) -> TRTResult:
        """Execute TRT with event emission at each phase."""
        total_rounds = self.config.rounds

        result = TRTResult(text="", confidence=0.0, rounds_used=0)
        best_answer = ""
        best_confidence = 0.0

        for round_num in range(1, total_rounds + 1):
            if self._cancelled:
                self.emit({"type": "cancelled"})
                break

            # GENERATE
            self.emit({
                "type": "status",
                "phase": "generating",
                "round": round_num,
                "total_rounds": total_rounds,
            })
            candidates = self._generate_phase(prompt, round_num, max_new_tokens)
            result.total_candidates_generated += len(candidates)

            self.emit({
                "type": "candidates",
                "count": len(candidates),
                "texts": [c.text[:300] for c in candidates],
                "round": round_num,
            })

            if self._cancelled:
                self.emit({"type": "cancelled"})
                break

            # SELECT
            self.emit({
                "type": "status",
                "phase": "selecting",
                "round": round_num,
            })
            selected, confidence = self._select_phase(candidates)

            self.emit({
                "type": "selected",
                "text": selected.text,
                "confidence": confidence,
                "round": round_num,
            })

            if self._cancelled:
                self.emit({"type": "cancelled"})
                break

            # REFLECT
            self.emit({
                "type": "status",
                "phase": "reflecting",
                "round": round_num,
            })
            insights = self._reflect_phase(prompt, candidates, selected, round_num)

            # Confidence varies by reflection depth
            insight_confidence = {
                "minimal": 0.5, "standard": 0.6, "deep": 0.75
            }.get(self.config.reflection_depth.value, 0.6)

            for insight in insights:
                self.emit({
                    "type": "insight",
                    "text": insight,
                    "confidence": insight_confidence,
                    "round": round_num,
                })

            # Record round
            round_result = TRTRoundResult(
                round_number=round_num,
                candidates=candidates,
                selected=selected,
                confidence=confidence,
                insights=insights,
                knowledge_added=len(insights),
            )
            result.round_results.append(round_result)
            result.knowledge_log.extend(insights)

            if confidence >= best_confidence:
                best_answer = selected.text
                best_confidence = confidence
                result.rounds_used = round_num

            # Early stop
            if confidence >= self.config.early_stop_threshold:
                result.early_stopped = True
                break

        result.text = best_answer
        result.confidence = best_confidence
        result.rounds_used = len(result.round_results)

        self.emit({
            "type": "result",
            "text": result.text,
            "confidence": result.confidence,
            "rounds_used": result.rounds_used,
            "total_candidates": result.total_candidates_generated,
            "early_stopped": result.early_stopped,
            "knowledge_count": len(self.knowledge.entries),
        })

        return result
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ui.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/streaming.py tests/test_ui.py
git commit -m "feat(ui): add StreamingTRTEngine with event callbacks"
```

---

### Task 3: WebSocket endpoint

Wire the StreamingTRTEngine into FastAPI's WebSocket handler.

**Files:**
- Modify: `ui/app.py`

- [ ] **Step 1: Write WebSocket test**

Add to `tests/test_ui.py`:

```python
import pytest
from fastapi.testclient import TestClient


class TestWebSocket:
    def test_health_endpoint(self):
        from ui.app import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_websocket_generates_with_mock(self):
        """Should accept generate request and return events."""
        from unittest.mock import patch
        from ui.app import app

        client = TestClient(app)
        mock_gen = MockGenerator()
        with patch("ui.app._get_backend", return_value=mock_gen):
            with client.websocket_connect("/ws") as ws:
                ws.send_json({
                    "type": "generate",
                    "prompt": "test",
                    "settings": {"rounds": 1, "candidates": 2, "reflection_depth": "minimal"},
                })
                events = []
                while True:
                    event = ws.receive_json()
                    events.append(event)
                    if event["type"] in ("result", "error", "cancelled"):
                        break
                assert len(events) > 0
                assert any(e["type"] == "result" for e in events)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_ui.py::TestWebSocket -v
```
Expected: FAIL — no `/ws` endpoint

- [ ] **Step 3: Implement WebSocket endpoint in app.py**

Replace `ui/app.py` with:

```python
# ui/app.py
"""TRIT-TRT Web UI — FastAPI server with WebSocket streaming."""

from __future__ import annotations
import asyncio
import json
import logging
import threading
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from trit_trt.engine import TritTRT, InferenceBackend
from trit_trt.config import (
    TritTRTConfig, BitNetConfig, TRTConfig,
    QuantType, SelectionMethod, ReflectionDepth,
)
from trit_trt.knowledge_store import KnowledgeStore
from ui.streaming import StreamingTRTEngine

logger = logging.getLogger("trit-trt.ui")

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="TRIT-TRT")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Global state
_backend: Optional[InferenceBackend] = None
_knowledge: Optional[KnowledgeStore] = None
_busy = False
_current_engine: Optional[StreamingTRTEngine] = None


def _get_backend() -> InferenceBackend:
    """Lazily initialize the inference backend."""
    global _backend
    if _backend is None:
        _backend = InferenceBackend(
            model_id="microsoft/BitNet-b1.58-2B-4T",
            bitnet_config=BitNetConfig(),
        )
        _backend.setup()
    return _backend


def _get_knowledge() -> KnowledgeStore:
    """Get or create the knowledge store."""
    global _knowledge
    if _knowledge is None:
        _knowledge = KnowledgeStore(max_entries=50)
    return _knowledge


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    global _busy
    await ws.accept()

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "generate":
                if _busy:
                    await ws.send_json({
                        "type": "error",
                        "message": "Generation already in progress",
                    })
                    continue

                _busy = True
                try:
                    await _handle_generate(ws, data)
                finally:
                    _busy = False

            elif msg_type == "cancel":
                if _current_engine is not None:
                    _current_engine.cancel()

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await ws.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass


async def _handle_generate(ws: WebSocket, data: dict) -> None:
    """Handle a generate request with streaming events."""
    prompt = data.get("prompt", "")
    settings = data.get("settings", {})

    if not prompt.strip():
        await ws.send_json({"type": "error", "message": "Empty prompt"})
        return

    # Build TRT config from settings
    trt_config = TRTConfig(
        rounds=settings.get("rounds", 3),
        candidates_per_round=settings.get("candidates", 8),
        selection_method=SelectionMethod(
            settings.get("selection_method", "self_consistency")
        ),
        reflection_depth=ReflectionDepth(
            settings.get("reflection_depth", "standard")
        ),
        early_stop_threshold=settings.get("early_stop_threshold", 0.95),
    )

    max_tokens = settings.get("max_tokens", 512)

    # Event queue for async bridge
    event_queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_event(event: dict):
        """Thread-safe event emission."""
        loop.call_soon_threadsafe(event_queue.put_nowait, event)

    # Setup phase
    await ws.send_json({"type": "status", "phase": "loading"})

    backend = _get_backend()
    knowledge = _get_knowledge()

    # Check knowledge persistence setting
    if not settings.get("knowledge_persistence", True):
        knowledge = KnowledgeStore(max_entries=50)

    await ws.send_json({"type": "status", "phase": "ready"})

    # Apply temperature to backend if BitNet
    temperature = settings.get("temperature", 0.6)
    if hasattr(backend, '_bitnet') and backend._bitnet:
        backend._bitnet.config.temperature = temperature

    # Create streaming engine
    global _current_engine
    streaming_engine = StreamingTRTEngine(
        generator=backend,
        config=trt_config,
        knowledge_store=knowledge,
        on_event=on_event,
    )
    _current_engine = streaming_engine

    # Run inference in a thread (it's blocking/CPU-bound)
    def run_inference():
        try:
            streaming_engine.run(prompt, max_new_tokens=max_tokens)
        except Exception as e:
            on_event({"type": "error", "message": str(e)})
        finally:
            global _current_engine
            _current_engine = None

    thread = threading.Thread(target=run_inference, daemon=True)
    thread.start()

    # Forward events from the queue to the WebSocket
    while True:
        try:
            event = await asyncio.wait_for(event_queue.get(), timeout=300)
            await ws.send_json(event)
            if event["type"] in ("result", "error", "cancelled"):
                break
        except asyncio.TimeoutError:
            await ws.send_json({
                "type": "error",
                "message": "Generation timed out",
            })
            break

    thread.join(timeout=5)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8765)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_ui.py -v
```
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ui/app.py tests/test_ui.py
git commit -m "feat(ui): add WebSocket endpoint with streaming TRT events"
```

---

### Task 4: HTML structure

Build the full page layout.

**Files:**
- Create: `ui/static/index.html`

- [ ] **Step 1: Write the complete HTML**

```html
<!-- ui/static/index.html -->
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TRIT-TRT</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <header id="header">
        <h1>TRIT-TRT</h1>
        <div class="header-actions">
            <button id="knowledge-btn" class="badge-btn" title="Knowledge Store">
                <span class="badge-icon">K</span>
                <span id="knowledge-count" class="badge-count">0</span>
            </button>
            <button id="settings-btn" class="icon-btn" title="Settings">&#9881;</button>
        </div>
    </header>

    <main id="chat-area">
        <div id="messages"></div>
    </main>

    <footer id="input-bar">
        <textarea id="prompt-input" placeholder="Type your prompt..." rows="1"></textarea>
        <button id="send-btn">Send</button>
    </footer>

    <!-- Settings Panel -->
    <div id="settings-panel" class="panel hidden">
        <div class="panel-header">
            <h2>Settings</h2>
            <button class="close-btn" data-close="settings-panel">&times;</button>
        </div>
        <div class="panel-body">
            <label>
                TRT Rounds
                <input type="range" id="s-rounds" min="1" max="5" value="3">
                <span class="slider-val" data-for="s-rounds">3</span>
            </label>
            <label>
                Candidates / Round
                <input type="range" id="s-candidates" min="2" max="16" value="8">
                <span class="slider-val" data-for="s-candidates">8</span>
            </label>
            <label>
                Max Tokens
                <input type="range" id="s-max-tokens" min="64" max="2048" value="512" step="64">
                <span class="slider-val" data-for="s-max-tokens">512</span>
            </label>
            <label>
                Temperature
                <input type="range" id="s-temperature" min="0.1" max="1.5" value="0.6" step="0.1">
                <span class="slider-val" data-for="s-temperature">0.6</span>
            </label>
            <label>
                Selection Method
                <select id="s-selection">
                    <option value="self_consistency" selected>Self Consistency</option>
                    <option value="verification">Verification</option>
                    <option value="hybrid">Hybrid</option>
                </select>
            </label>
            <label>
                Reflection Depth
                <select id="s-reflection">
                    <option value="minimal">Minimal</option>
                    <option value="standard" selected>Standard</option>
                    <option value="deep">Deep</option>
                </select>
            </label>
            <label>
                Early Stop Threshold
                <input type="range" id="s-threshold" min="0.5" max="1.0" value="0.95" step="0.05">
                <span class="slider-val" data-for="s-threshold">0.95</span>
            </label>
            <label class="toggle-label">
                Knowledge Persistence
                <input type="checkbox" id="s-knowledge" checked>
            </label>
        </div>
    </div>

    <!-- Knowledge Panel -->
    <div id="knowledge-panel" class="panel hidden">
        <div class="panel-header">
            <h2>Knowledge Store</h2>
            <button class="close-btn" data-close="knowledge-panel">&times;</button>
        </div>
        <div class="panel-body">
            <div id="knowledge-list"><p class="empty">No insights yet.</p></div>
        </div>
    </div>

    <script src="/static/main.js"></script>
</body>
</html>
```

- [ ] **Step 2: Verify it loads in browser**

```bash
cd ~/trit-trt && python -m ui.app
# Visit http://localhost:8765 — should show header, empty chat, input bar
```

- [ ] **Step 3: Commit**

```bash
git add ui/static/index.html
git commit -m "feat(ui): add HTML layout with settings and knowledge panels"
```

---

### Task 5: CSS theme

**Files:**
- Create: `ui/static/style.css`

- [ ] **Step 1: Write the complete CSS**

The CSS should cover: page layout (header/main/footer), dark theme variables, chat messages (user vs bot), round cards (collapsible), settings/knowledge slide-out panels, confidence bars with color coding, input bar, badges, animations for round cards appearing.

Key design tokens:
- `--bg: #18181b` (zinc-900)
- `--surface: #27272a` (zinc-800)
- `--border: #3f3f46` (zinc-700)
- `--text: #fafafa` (zinc-50)
- `--text-muted: #a1a1aa` (zinc-400)
- `--accent: #3b82f6` (blue-500)
- `--confidence-low: #ef4444` (red)
- `--confidence-mid: #eab308` (yellow)
- `--confidence-high: #22c55e` (green)
- Font: `ui-monospace, "Cascadia Code", "Fira Code", monospace` for model output
- Font: `system-ui, -apple-system, sans-serif` for UI chrome

Layout: header fixed top, footer fixed bottom, main fills remaining space and scrolls. Panels slide in from right with 320px width and overlay. Round cards have a left border accent and collapse/expand on click.

- [ ] **Step 2: Verify in browser**

```bash
cd ~/trit-trt && python -m ui.app
# Should show dark themed page with proper layout
```

- [ ] **Step 3: Commit**

```bash
git add ui/static/style.css
git commit -m "feat(ui): add dark minimal CSS theme"
```

---

### Task 6: JavaScript — WebSocket client and DOM rendering

**Files:**
- Create: `ui/static/main.js`

- [ ] **Step 1: Write the complete JavaScript**

The JS needs to handle:

1. **WebSocket connection** — connect on page load (no auto-reconnect per spec; user refreshes page to reconnect)
2. **Settings** — read slider/dropdown/checkbox values, sync `slider-val` spans, toggle panel visibility
3. **Send message** — read prompt, build `generate` JSON, send via WebSocket, add user message to chat, disable input, change Send to Cancel
4. **Cancel** — send `{"type": "cancel"}` via WebSocket
5. **Event handlers** — one handler per event type:
   - `status` → update status indicator in current response card
   - `candidates` → add collapsed candidate list to current round card
   - `selected` → show winner text + confidence bar in round card
   - `insight` → append to round card's insight list, increment knowledge badge
   - `result` → show final answer text + meta bar, re-enable input
   - `error` → show error in chat, re-enable input
   - `cancelled` → show cancelled state, re-enable input
6. **Round cards** — create a new collapsible card for each round, animate it in
7. **Knowledge panel** — render all insights from `insight` events
8. **Auto-scroll** — keep chat scrolled to bottom as content arrives
9. **Textarea auto-resize** — grow textarea as user types, submit on Enter (Shift+Enter for newline)

Key functions:
- `addUserMessage(text)` — append user bubble to `#messages`
- `createResponseCard()` — append bot response container to `#messages`, return reference
- `addRoundCard(responseEl, roundNum, totalRounds)` — append round card inside response
- `updateRoundPhase(roundEl, phase)` — update status text in round card
- `setConfidenceBar(roundEl, confidence)` — set width + color of confidence bar
- `showFinalAnswer(responseEl, text, meta)` — append answer + meta bar
- `getSettings()` — read all setting values into an object

- [ ] **Step 2: Verify full flow in browser**

```bash
cd ~/trit-trt && python -m ui.app
# 1. Open http://localhost:8765
# 2. Type "What is 2+2?" and press Send
# 3. Should see round cards animate in with live status updates
# 4. Final answer should appear with confidence bar
# 5. Settings gear should open/close the panel
# 6. Knowledge badge should update
```

- [ ] **Step 3: Commit**

```bash
git add ui/static/main.js
git commit -m "feat(ui): add WebSocket client with live TRT round rendering"
```

---

### Task 7: Integration test and polish

**Files:**
- Modify: `tests/test_ui.py`

- [ ] **Step 1: Add integration test with MockGenerator**

Add to `tests/test_ui.py`:

```python
class TestIntegration:
    def test_full_websocket_flow_with_mock(self):
        """Test the full WebSocket protocol with a mock generator."""
        from unittest.mock import patch
        from ui.app import app, _get_backend

        client = TestClient(app)

        # Patch the backend to use MockGenerator
        mock_gen = MockGenerator()
        with patch("ui.app._get_backend") as mock_backend:
            mock_backend.return_value = mock_gen
            with client.websocket_connect("/ws") as ws:
                ws.send_json({
                    "type": "generate",
                    "prompt": "Test prompt",
                    "settings": {
                        "rounds": 1,
                        "candidates": 2,
                        "reflection_depth": "minimal",
                    },
                })

                events = []
                while True:
                    event = ws.receive_json()
                    events.append(event)
                    if event["type"] in ("result", "error"):
                        break

                types = [e["type"] for e in events]
                assert "status" in types
                assert "result" in types

                result_event = next(e for e in events if e["type"] == "result")
                assert result_event["confidence"] > 0
                assert result_event["rounds_used"] >= 1
```

- [ ] **Step 2: Run all tests**

```bash
pytest tests/ -v
```
Expected: All pass

- [ ] **Step 3: Manual browser test**

```bash
cd ~/trit-trt && python -m ui.app
```
Test the full flow: send a prompt, watch rounds animate, check settings panel, check knowledge panel, try cancel.

- [ ] **Step 4: Commit**

```bash
git add tests/test_ui.py
git commit -m "test(ui): add integration tests for WebSocket flow"
```

- [ ] **Step 5: Push to GitHub**

```bash
git push origin main
```

---

## Summary

| Task | What | Files |
|------|------|-------|
| 1 | Scaffold + dependencies | `ui/app.py`, `ui/__init__.py`, `ui/static/index.html` |
| 2 | StreamingTRTEngine | `ui/streaming.py`, `tests/test_ui.py` |
| 3 | WebSocket endpoint | `ui/app.py` |
| 4 | HTML layout | `ui/static/index.html` |
| 5 | CSS theme | `ui/static/style.css` |
| 6 | JavaScript client | `ui/static/main.js` |
| 7 | Integration test + polish | `tests/test_ui.py` |

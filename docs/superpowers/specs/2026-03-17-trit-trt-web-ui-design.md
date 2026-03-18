# TRIT-TRT Web UI — Design Spec

**Date:** 2026-03-17
**Author:** Massimo / Yunis AI
**Status:** Approved

## Overview

A chat-style single-page web app for TRIT-TRT that streams TRT round progress in real-time via WebSocket. Dark minimal theme, clean typography, all TRT settings exposed.

## Architecture

Single-process Python app. FastAPI serves the WebSocket API and static files. No build step, no frontend framework.

```
trit-trt/
├── ui/
│   ├── app.py          # FastAPI server + WebSocket handler
│   ├── static/
│   │   ├── index.html  # Single-page app
│   │   ├── style.css   # Dark minimal theme
│   │   └── main.js     # WebSocket client + DOM updates
│   └── __init__.py
├── trit_trt/           # Existing engine, untouched
└── ...
```

FastAPI imports `TritTRT` directly — no HTTP API wrapping, no separate process. Runs in the conda env alongside the engine.

**Concurrency**: Single-request-at-a-time. The engine is stateful and not thread-safe. If a generation is in-flight and the client sends another `generate`, the server responds with `{"type": "error", "message": "Generation already in progress"}`.

**Model loading**: On first `generate`, `TritTRT.setup()` loads the model (takes ~1-2s). The server sends `{"type": "status", "phase": "loading"}` before setup and `{"type": "status", "phase": "ready"}` after. The input bar is disabled until `ready` is received.

**Disconnect handling**: If the WebSocket disconnects mid-generation, the server lets the current TRT run finish (result is discarded). No reconnection logic — the client creates a new WebSocket on page load. Knowledge store persists in memory across connections.

## WebSocket Protocol

Client sends:
```json
{"type": "generate", "prompt": "...", "settings": {"rounds": 3, "candidates": 8, ...}}
{"type": "cancel"}
```

Server pushes event stream:
```
→ {"type": "status", "phase": "loading"}
→ {"type": "status", "phase": "ready"}
→ {"type": "status", "phase": "generating", "round": 1, "total_rounds": 3}
→ {"type": "candidates", "count": 8, "texts": ["...", ...], "round": 1}
→ {"type": "status", "phase": "selecting", "round": 1}
→ {"type": "selected", "text": "...", "confidence": 0.67, "round": 1}
→ {"type": "status", "phase": "reflecting", "round": 1}
→ {"type": "insight", "text": "...", "confidence": 0.6, "round": 1}
→ {"type": "status", "phase": "generating", "round": 2, ...}
→ ...
→ {"type": "result", "text": "...", "confidence": 0.95, "rounds_used": 2, "early_stopped": true}
→ {"type": "error", "message": "..."}
→ {"type": "cancelled"}
```

Notes:
- All status events use the same `{"type": "status", "phase": "..."}` shape for consistency.
- Candidates are sent as a batch (since `generate_batch()` returns all at once).
- Insight events include the confidence score for display in the Knowledge panel.
- Error events are sent for inference failures, busy state, or unexpected exceptions.
- Cancel causes the server to set a flag checked between TRT phases; the current phase finishes but the next phase is skipped.

## UI Layout

Single page, vertical layout:

- **Header**: "TRIT-TRT" title, knowledge store badge (count + click to inspect), settings gear icon
- **Chat area**: Scrollable message list. User messages right-aligned, TRIT-TRT responses left-aligned
- **Response cards**: Each TRIT-TRT response contains:
  - Collapsible round cards (animate in as each phase completes)
  - Each round card shows: candidates generated, winner selected + confidence, insights extracted
  - Click to expand: see individual candidate texts + reflection insights
  - Final answer text
  - Meta bar: confidence %, rounds used, total candidates, early stop status
- **Input bar**: Text input + Send button, fixed at bottom. Disabled during model loading and generation (Send becomes Cancel during generation).
- **Settings panel**: Slides out from right on gear click. Contains:
  - TRT rounds (1-5 slider, default 3)
  - Candidates per round (2-16 slider, default 8)
  - Max tokens (64-2048 slider, default 512)
  - Temperature (0.1-1.5 slider, default 0.6) — applies to BitNet sampling
  - Selection method (dropdown: self_consistency / verification / hybrid, default self_consistency)
  - Reflection depth (dropdown: minimal / standard / deep, default standard)
  - Early stop threshold (0.5-1.0 slider, default 0.95)
  - Knowledge persistence toggle (default on) — when toggled off, calls `reset_knowledge()` and stops storing new insights
- **Knowledge panel**: Slides out from right on badge click. Lists all accumulated insights with effectiveness scores and source round numbers.

## Visual Design

- Dark minimal: zinc-900 background, zinc-800 cards, zinc-700 borders
- One accent color (blue-500) for interactive elements and confidence indicators
- Geist Mono or system monospace for model output
- System sans-serif for UI chrome
- No gradients, no glassmorphism, no emojis
- Confidence shown as colored progress bars (red < 50%, yellow 50-80%, green > 80%)
- Model output rendered as preformatted plain text (no markdown parsing)

## Dependencies

Add to conda env: `fastapi`, `uvicorn[standard]`, `websockets`

## Launch

```bash
conda activate trit-trt
cd ~/trit-trt
python -m ui.app
# Serves at http://localhost:8765
```

## Not Included (YAGNI)

- Conversation history persistence
- Authentication / multi-user
- Export functionality
- Markdown rendering (output is preformatted text)
- Mobile responsive layout
- WebSocket reconnection logic

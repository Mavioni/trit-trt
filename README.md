# TRIT-TRT: Ternary Recursive Inference Thinking

**A Yunis AI sovereign inference framework combining BitNet b1.58 ternary quantization, AirLLM layer-wise sharding, and Test-time Recursive Thinking for self-improving local-first AI.**

## The Thesis

Binary computation is a degenerate case of ternary computation. The trit `{-1, 0, +1}` represents thesis, antithesis, and synthesis — and this framework makes that literal at every layer of the stack:

| Layer | Component | Ternary Mapping |
|-------|-----------|-----------------|
| **Substrate** | BitNet b1.58 | Weights quantized to `{-1, 0, +1}` — computation in trits |
| **Sovereignty** | AirLLM sharding | Local-first inference on constrained hardware |
| **Reasoning** | TRT loop | Generate (thesis) → Select (antithesis) → Reflect (synthesis) |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              TRIT-TRT Inference Pipeline              │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌─── BitNet b1.58 Ternary Substrate ────────────┐  │
│  │  Weights: {-1, 0, +1}  │  I2_S/TL1 kernels   │  │
│  │  2-6x CPU speedup      │  55-82% energy save  │  │
│  └───────────────────────────────────────────────┘  │
│                        │                              │
│                   GGUF weights                        │
│                        ▼                              │
│  ┌─── AirLLM Layer-wise Sovereign Inference ─────┐  │
│  │  Split model → Layer shards on disk            │  │
│  │  Load layer N → Compute → Prefetch N+1         │  │
│  │  4GB VRAM budget │ 4bit/8bit compression       │  │
│  └───────────────────────────────────────────────┘  │
│                        │                              │
│                  token stream                         │
│                        ▼                              │
│  ┌─── TRT Dialectical Reasoning Loop ────────────┐  │
│  │                                                 │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────┐│  │
│  │  │ GENERATE │→│  SELECT   │→│   REFLECT    ││  │
│  │  │ (thesis) │  │(antithes)│  │ (synthesis)  ││  │
│  │  │ N cands  │  │ majority │  │ extract      ││  │
│  │  └──────────┘  │ vote     │  │ insights     ││  │
│  │       ▲        └──────────┘  └──────┬───────┘│  │
│  │       └────── knowledge ────────────┘         │  │
│  │                                                 │  │
│  └───────────────────────────────────────────────┘  │
│                        │                              │
│                        ▼                              │
│              Refined output + confidence              │
└─────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

```bash
# Python 3.10+, CUDA optional (CPU inference supported)
conda create -n trit-trt python=3.10
conda activate trit-trt
pip install -r requirements.txt
```

### Basic Inference

```python
from trit_trt import TritTRT

# Initialize with a BitNet-compatible model
engine = TritTRT(
    model_id="microsoft/BitNet-b1.58-2B-4T",
    quant_type="i2_s",           # Ternary quantization
    max_vram_gb=4,               # AirLLM memory budget
    compression="4bit",          # Layer compression
    trt_rounds=3,                # TRT reflection rounds
    trt_candidates=8,            # Candidates per round
)

# Single inference with TRT self-improvement
result = engine.generate(
    prompt="Explain quantum entanglement in simple terms.",
    max_new_tokens=512,
)

print(result.text)           # Best refined answer
print(result.confidence)     # Self-consistency score
print(result.rounds_used)    # How many TRT rounds ran
print(result.knowledge_log)  # Accumulated insights
```

### Advanced: Custom TRT Strategy

```python
from trit_trt import TritTRT, TRTConfig

config = TRTConfig(
    rounds=5,
    candidates_per_round=16,
    selection_method="self_consistency",  # or "verification"
    reflection_depth="deep",             # extract generalizable patterns
    early_stop_threshold=0.95,           # stop if confidence >= 95%
    knowledge_persistence=True,          # carry insights across prompts
)

engine = TritTRT(
    model_id="1bitLLM/bitnet_b1_58-3B",
    trt_config=config,
)

# The engine improves across multiple queries in a session
for question in math_problems:
    result = engine.generate(question)
    # Knowledge from previous problems informs future ones
```

## Components

### 1. BitNet Integration (`trit_trt/bitnet_engine.py`)
Wraps Microsoft's bitnet.cpp inference with Python bindings. Handles model download, GGUF conversion, and kernel selection (I2_S for x86, TL1 for ARM).

### 2. AirLLM Sharding (`trit_trt/layer_shard.py`)
Adapts AirLLM's layer-wise loading for BitNet's ternary weight format. Key innovation: ternary weights compress ~10x better than FP16 during disk sharding, so layer load times drop dramatically.

### 3. TRT Loop (`trit_trt/trt_engine.py`)
Implements the Generate→Select→Reflect cycle from Zhuang et al. The reflection phase uses the model itself to analyze why certain solutions succeeded, accumulating a knowledge store that persists within a session.

### 4. Knowledge Flow (`trit_trt/knowledge_store.py`)
Persistent knowledge accumulator inspired by the TRT MCP server pattern. Stores insights as structured entries with relevance scores, enabling cross-problem transfer learning at inference time.

## Configuration

See `configs/default.yaml` for all options:

```yaml
bitnet:
  quant_type: i2_s        # i2_s | tl1
  threads: 4
  ctx_size: 2048

airllm:
  max_vram_gb: 4
  compression: "4bit"     # null | 4bit | 8bit
  prefetch: true
  delete_original: false

trt:
  rounds: 3
  candidates: 8
  selection: self_consistency
  reflection_depth: standard  # standard | deep
  early_stop: 0.95
  knowledge_persistence: true
```

## Benchmarks

Target performance on constrained hardware (single 4GB GPU or CPU-only):

| Model | Hardware | Tokens/sec | TRT Rounds | Quality Uplift |
|-------|----------|-----------|------------|----------------|
| BitNet-2B-4T | M2 CPU | 5-7 t/s | 3 | +15-25% |
| BitNet-3B | 4GB GPU | 3-5 t/s | 3 | +15-25% |
| BitNet-8B | 8GB GPU | 2-3 t/s | 5 | +20-30% |

*Quality uplift measured as improvement over single-pass inference on reasoning benchmarks.*

## Project Structure

```
trit-trt/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   └── default.yaml
├── scripts/
│   ├── setup_model.sh
│   └── benchmark.py
├── trit_trt/
│   ├── __init__.py
│   ├── engine.py              # Main TritTRT orchestrator
│   ├── bitnet_engine.py       # BitNet inference wrapper
│   ├── layer_shard.py         # AirLLM-style layer sharding
│   ├── trt_engine.py          # TRT Generate→Select→Reflect
│   ├── knowledge_store.py     # Persistent insight accumulator
│   └── config.py              # Configuration dataclasses
└── tests/
    └── test_engine.py
```

## License

MIT — Sovereign tech for sovereign minds.

## Credits

- [BitNet](https://github.com/microsoft/BitNet) — Microsoft Research
- [AirLLM](https://github.com/lyogavin/airllm) — Gavin Li
- [TRT](https://github.com/EvanZhuang/test_time_recursive_thinking) — Zhuang et al.
- **TRIT-TRT Integration** — Massimo / Yunis AI

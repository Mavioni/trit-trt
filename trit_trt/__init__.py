"""
TRIT-TRT: Ternary Recursive Inference Thinking
Yunis AI — Sovereign Ternary Inference

Combines BitNet b1.58 (ternary weights), AirLLM (layer sharding),
and TRT (dialectical self-improvement) into a unified local-first
inference framework.

Usage:
    from trit_trt import TritTRT

    engine = TritTRT("microsoft/BitNet-b1.58-2B-4T")
    result = engine.generate("Explain quantum entanglement.")
    print(result.text)
    print(result.confidence)
"""

__version__ = "0.1.0"
__author__ = "Massimo / Yunis AI"

from .engine import TritTRT
from .trt_engine import TRTEngine, TRTResult, TRTCandidate, TRTRoundResult
from .bitnet_engine import BitNetEngine
from .layer_shard import LayerShard
from .knowledge_store import KnowledgeStore, KnowledgeEntry
from .config import (
    TritTRTConfig,
    BitNetConfig,
    AirLLMConfig,
    TRTConfig,
    QuantType,
    Compression,
    SelectionMethod,
    ReflectionDepth,
)

__all__ = [
    # Main entry point
    "TritTRT",
    # Engines
    "TRTEngine",
    "BitNetEngine",
    "LayerShard",
    # Results
    "TRTResult",
    "TRTCandidate",
    "TRTRoundResult",
    # Knowledge
    "KnowledgeStore",
    "KnowledgeEntry",
    # Config
    "TritTRTConfig",
    "BitNetConfig",
    "AirLLMConfig",
    "TRTConfig",
    "QuantType",
    "Compression",
    "SelectionMethod",
    "ReflectionDepth",
]

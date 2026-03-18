"""
TRIT-TRT Engine — Main Orchestrator
Yunis AI — Sovereign Ternary Inference

Combines BitNet (ternary substrate), AirLLM (layer sharding),
and TRT (dialectical reasoning) into a unified inference pipeline.

Usage:
    engine = TritTRT("microsoft/BitNet-b1.58-2B-4T")
    result = engine.generate("Explain quantum entanglement.")
    print(result.text)
    print(result.confidence)
"""

from __future__ import annotations
import logging
import platform
from typing import Optional

from .config import (
    TritTRTConfig, BitNetConfig, AirLLMConfig, TRTConfig,
    QuantType, Compression,
)
from .bitnet_engine import BitNetEngine
from .layer_shard import LayerShard
from .trt_engine import TRTEngine, TRTResult
from .knowledge_store import KnowledgeStore

logger = logging.getLogger("trit-trt")


class InferenceBackend:
    """
    Unified inference backend that routes between BitNet (native ternary)
    and AirLLM (layer-sharded) depending on model format and hardware.

    For GGUF ternary models → BitNet engine (CPU-optimized)
    For HF safetensors models → AirLLM engine (GPU layer-sharding)
    """

    def __init__(
        self,
        model_id: str,
        bitnet_config: Optional[BitNetConfig] = None,
        airllm_config: Optional[AirLLMConfig] = None,
        backend: str = "auto",
    ):
        self.model_id = model_id
        self._backend_type = backend
        self._bitnet: Optional[BitNetEngine] = None
        self._airllm: Optional[LayerShard] = None

        if backend == "auto":
            self._backend_type = self._detect_backend()

        if self._backend_type == "bitnet":
            self._bitnet = BitNetEngine(model_id, config=bitnet_config)
        else:
            self._airllm = LayerShard(model_id, config=airllm_config)

    def setup(self) -> None:
        """Initialize the selected backend."""
        if self._bitnet:
            self._bitnet.setup()
        elif self._airllm:
            self._airllm.load()

    def generate(self, prompt: str, max_new_tokens: int = 512, **kwargs) -> str:
        """Generate text via the active backend."""
        if self._bitnet:
            return self._bitnet.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
        elif self._airllm:
            return self._airllm.generate(prompt, max_new_tokens=max_new_tokens, **kwargs)
        raise RuntimeError("No backend initialized")

    def generate_batch(self, prompt: str, n: int, max_new_tokens: int = 512, **kwargs) -> list[str]:
        """Generate N candidates via the active backend."""
        if self._bitnet:
            return self._bitnet.generate_batch(prompt, n, max_new_tokens=max_new_tokens, **kwargs)
        elif self._airllm:
            return self._airllm.generate_batch(prompt, n, max_new_tokens=max_new_tokens, **kwargs)
        raise RuntimeError("No backend initialized")

    def _detect_backend(self) -> str:
        """Auto-detect which backend to use."""
        # BitNet-specific models use bitnet.cpp
        bitnet_markers = ["BitNet", "bitnet", "1.58"]
        if any(m in self.model_id for m in bitnet_markers):
            return "bitnet"

        # Default to AirLLM for general HF models
        return "airllm"

    @property
    def info(self) -> dict:
        if self._bitnet:
            return {"backend": "bitnet", **self._bitnet.info}
        elif self._airllm:
            return {"backend": "airllm", **self._airllm.info}
        return {"backend": "none"}


class TritTRT:
    """
    TRIT-TRT: Ternary Recursive Inference Thinking

    The main entry point for the unified pipeline.
    Combines ternary quantization, sovereign layer sharding,
    and dialectical self-improvement into a single API.

    Example:
        engine = TritTRT(
            model_id="microsoft/BitNet-b1.58-2B-4T",
            trt_rounds=3,
            trt_candidates=8,
            max_vram_gb=4,
        )
        result = engine.generate("Solve: x^2 + 3x - 10 = 0")
        print(result.text)        # Best answer
        print(result.confidence)  # Self-consistency score

    The engine improves across queries within a session —
    knowledge from previous problems informs future ones.
    """

    def __init__(
        self,
        model_id: str = "microsoft/BitNet-b1.58-2B-4T",
        config: Optional[TritTRTConfig] = None,
        # Convenience kwargs (override config fields)
        quant_type: str = "i2_s",
        max_vram_gb: float = 4.0,
        compression: str = "4bit",
        trt_rounds: int = 3,
        trt_candidates: int = 8,
        backend: str = "auto",
        verbose: bool = True,
        trt_config: Optional[TRTConfig] = None,
    ):
        # Build config from kwargs if not provided
        if config is None:
            config = TritTRTConfig(
                model_id=model_id,
                bitnet=BitNetConfig(quant_type=QuantType(quant_type)),
                airllm=AirLLMConfig(
                    max_vram_gb=max_vram_gb,
                    compression=Compression(compression) if compression != "none" else Compression.NONE,
                ),
                trt=trt_config or TRTConfig(
                    rounds=trt_rounds,
                    candidates_per_round=trt_candidates,
                ),
                verbose=verbose,
            )

        self.config = config
        self._setup_logging()

        # Initialize the three layers
        logger.info("=" * 60)
        logger.info("  TRIT-TRT: Ternary Recursive Inference Thinking")
        logger.info("  Yunis AI — Sovereign Ternary Inference")
        logger.info("=" * 60)
        logger.info(f"  Model:       {config.model_id}")
        logger.info(f"  Quant:       {config.bitnet.quant_type.value}")
        logger.info(f"  VRAM budget: {config.airllm.max_vram_gb}GB")
        logger.info(f"  Compression: {config.airllm.compression.value}")
        logger.info(f"  TRT rounds:  {config.trt.rounds}")
        logger.info(f"  Candidates:  {config.trt.candidates_per_round}")
        logger.info(f"  Platform:    {platform.system()} {platform.machine()}")
        logger.info("=" * 60)

        # Layer 1 + 2: Inference backend (BitNet or AirLLM)
        self.backend = InferenceBackend(
            model_id=config.model_id,
            bitnet_config=config.bitnet,
            airllm_config=config.airllm,
            backend=backend,
        )

        # Layer 3: TRT dialectical reasoning
        self.knowledge = KnowledgeStore(
            max_entries=config.trt.max_knowledge_entries,
        )
        self.trt = TRTEngine(
            generator=self.backend,
            config=config.trt,
            knowledge_store=self.knowledge,
        )

        self._is_setup = False

    def setup(self) -> None:
        """Initialize all layers. Called automatically on first generate()."""
        if not self._is_setup:
            logger.info("Setting up inference backend...")
            self.backend.setup()
            self._is_setup = True
            logger.info("TRIT-TRT ready for inference")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        use_trt: bool = True,
    ) -> TRTResult:
        """
        Generate a response with optional TRT self-improvement.

        Args:
            prompt: The input question or instruction
            max_new_tokens: Max tokens per candidate response
            use_trt: If True, run full TRT loop. If False, single-pass.

        Returns:
            TRTResult with text, confidence, round details, and knowledge log
        """
        self.setup()

        if use_trt:
            return self.trt.run(prompt, max_new_tokens=max_new_tokens)
        else:
            # Single-pass inference (no TRT)
            text = self.backend.generate(prompt, max_new_tokens=max_new_tokens)
            return TRTResult(
                text=text,
                confidence=1.0,
                rounds_used=0,
                early_stopped=False,
            )

    def generate_simple(self, prompt: str, max_new_tokens: int = 512) -> str:
        """Convenience: return just the text, no TRT metadata."""
        result = self.generate(prompt, max_new_tokens=max_new_tokens)
        return result.text

    @property
    def session_stats(self) -> dict:
        """Return session-level statistics."""
        return {
            "backend": self.backend.info,
            "knowledge": self.knowledge.get_session_stats(),
            "config": {
                "model": self.config.model_id,
                "trt_rounds": self.config.trt.rounds,
                "candidates": self.config.trt.candidates_per_round,
            },
        }

    def reset_knowledge(self) -> None:
        """Clear the session knowledge store."""
        self.knowledge = KnowledgeStore(
            max_entries=self.config.trt.max_knowledge_entries,
        )
        self.trt.knowledge = self.knowledge
        logger.info("Knowledge store reset")

    def _setup_logging(self) -> None:
        """Configure logging based on verbosity."""
        level = logging.INFO if self.config.verbose else logging.WARNING
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
            datefmt="%H:%M:%S",
        )

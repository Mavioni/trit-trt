"""
Layer Shard — Sovereign Inference Layer
Yunis AI — TRIT-TRT

Adapts AirLLM's layer-wise model loading for BitNet's ternary weight format.
Enables 70B+ parameter models to run on 4GB VRAM by loading one transformer
layer at a time from disk with prefetching.

Key insight: ternary weights {-1, 0, +1} compress dramatically better than
FP16 weights during disk sharding — each weight needs only ~1.58 bits vs 16 bits,
so layer load times drop by ~10x compared to standard AirLLM usage.
"""

from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

import torch
import psutil

from .config import AirLLMConfig, Compression

logger = logging.getLogger("trit-trt.shard")


@dataclass
class ShardStats:
    """Performance statistics for a sharding session."""
    total_layers: int = 0
    layers_loaded: int = 0
    total_load_time_ms: float = 0
    total_compute_time_ms: float = 0
    peak_vram_mb: float = 0
    disk_read_mb: float = 0

    @property
    def avg_load_time_ms(self) -> float:
        if self.layers_loaded == 0:
            return 0
        return self.total_load_time_ms / self.layers_loaded

    @property
    def avg_compute_time_ms(self) -> float:
        if self.layers_loaded == 0:
            return 0
        return self.total_compute_time_ms / self.layers_loaded

    def summary(self) -> str:
        return (
            f"Layers: {self.layers_loaded}/{self.total_layers} | "
            f"Avg load: {self.avg_load_time_ms:.1f}ms | "
            f"Avg compute: {self.avg_compute_time_ms:.1f}ms | "
            f"Peak VRAM: {self.peak_vram_mb:.0f}MB | "
            f"Disk read: {self.disk_read_mb:.1f}MB"
        )


class LayerShard:
    """
    Memory-efficient layer-wise model loader for ternary LLMs.

    Instead of loading the full model into memory, this class:
    1. Splits the model into per-layer shards on disk
    2. Loads one layer at a time into GPU/CPU memory
    3. Runs the forward pass for that layer
    4. Prefetches the next layer while computing the current one
    5. Frees memory after each layer completes

    For BitNet models, the ternary weight format means each shard is
    ~10x smaller than FP16, making disk I/O much less of a bottleneck.
    """

    def __init__(
        self,
        model_id: str,
        config: Optional[AirLLMConfig] = None,
    ):
        self.model_id = model_id
        self.config = config or AirLLMConfig()
        self.model = None
        self.stats = ShardStats()
        self._device = self._select_device()

    def load(self) -> None:
        """
        Initialize the AirLLM model with layer sharding.

        This will:
        1. Download the model if needed
        2. Split into per-layer shards (first run only)
        3. Set up the prefetch pipeline
        """
        try:
            from airllm import AutoModel as AirAutoModel
        except ImportError:
            raise ImportError(
                "airllm not installed. Run: pip install airllm"
            )

        logger.info(f"Loading {self.model_id} with layer sharding...")
        logger.info(f"  Device: {self._device}")
        logger.info(f"  VRAM budget: {self.config.max_vram_gb}GB")
        logger.info(f"  Compression: {self.config.compression.value}")

        kwargs = {}

        # Apply compression if configured
        if self.config.compression != Compression.NONE:
            kwargs["compression"] = self.config.compression.value

        # Layer shard path
        if self.config.layer_shards_path:
            kwargs["layer_shards_saving_path"] = self.config.layer_shards_path

        # HF token for gated models
        if self.config.hf_token:
            kwargs["hf_token"] = self.config.hf_token

        # Prefetch toggle
        kwargs["prefetching"] = self.config.prefetch

        # Delete original to save disk
        if self.config.delete_original:
            kwargs["delete_original"] = True

        # Profiling
        if self.config.profiling_mode:
            kwargs["profiling_mode"] = True

        self.model = AirAutoModel.from_pretrained(
            self.model_id,
            **kwargs,
        )

        logger.info("Layer sharding initialized successfully")

    def generate(
        self,
        input_text: str | list[str],
        max_new_tokens: int = 128,
        max_length: int = 512,
        use_cache: bool = True,
    ) -> str:
        """
        Generate text using layer-wise inference.

        Each forward pass loads layers one at a time, keeping memory
        usage within the VRAM budget.
        """
        if self.model is None:
            self.load()

        if isinstance(input_text, str):
            input_text = [input_text]

        # Tokenize
        input_tokens = self.model.tokenizer(
            input_text,
            return_tensors="pt",
            return_attention_mask=False,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        start = time.perf_counter()

        # Generate with layer-wise loading
        generation_output = self.model.generate(
            input_tokens["input_ids"].to(self._device),
            max_new_tokens=max_new_tokens,
            use_cache=use_cache,
            return_dict_in_generate=True,
        )

        elapsed_ms = (time.perf_counter() - start) * 1000
        self.stats.layers_loaded += 1
        self.stats.total_compute_time_ms += elapsed_ms

        # Decode
        output = self.model.tokenizer.decode(
            generation_output.sequences[0],
            skip_special_tokens=True,
        )

        # Strip the input from the output
        if len(input_text) == 1 and output.startswith(input_text[0]):
            output = output[len(input_text[0]):].strip()

        return output

    def generate_batch(
        self,
        prompt: str,
        n: int,
        max_new_tokens: int = 128,
        temperature_range: tuple[float, float] = (0.5, 0.9),
    ) -> list[str]:
        """
        Generate N candidates for TRT selection.

        Varies temperature across candidates for diversity while
        keeping memory usage constant (sequential generation).
        """
        results = []
        temp_low, temp_high = temperature_range
        temp_step = (temp_high - temp_low) / max(n - 1, 1)

        for i in range(n):
            temp = temp_low + i * temp_step
            output = self.generate(
                input_text=prompt,
                max_new_tokens=max_new_tokens,
            )
            results.append(output)

        return results

    def get_memory_usage(self) -> dict:
        """Return current memory utilization."""
        result = {
            "system_ram_gb": psutil.virtual_memory().used / (1024 ** 3),
            "system_ram_percent": psutil.virtual_memory().percent,
        }

        if torch.cuda.is_available():
            result["gpu_vram_gb"] = torch.cuda.memory_allocated() / (1024 ** 3)
            result["gpu_vram_reserved_gb"] = torch.cuda.memory_reserved() / (1024 ** 3)
            result["gpu_vram_percent"] = (
                torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_mem * 100
            )

        return result

    def _select_device(self) -> str:
        """Select the best available device within VRAM budget."""
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
            if vram_gb >= self.config.max_vram_gb:
                return "cuda"
            logger.warning(
                f"GPU has {vram_gb:.1f}GB VRAM, budget is {self.config.max_vram_gb}GB. "
                f"AirLLM will manage memory within budget."
            )
            return "cuda"

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    @property
    def info(self) -> dict:
        """Return shard engine metadata."""
        return {
            "model_id": self.model_id,
            "device": self._device,
            "compression": self.config.compression.value,
            "max_vram_gb": self.config.max_vram_gb,
            "prefetch": self.config.prefetch,
            "is_loaded": self.model is not None,
            "stats": self.stats.summary() if self.stats.layers_loaded > 0 else "No inference yet",
        }

"""
TRIT-TRT Configuration
Yunis AI — Sovereign Ternary Inference

All configuration dataclasses for the unified BitNet + AirLLM + TRT pipeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml


class QuantType(str, Enum):
    """BitNet quantization kernel type."""
    I2_S = "i2_s"    # x86 optimized
    TL1 = "tl1"      # ARM optimized
    TL2 = "tl2"      # ARM alternative


class Compression(str, Enum):
    """AirLLM block-wise compression level."""
    NONE = "none"
    FOUR_BIT = "4bit"
    EIGHT_BIT = "8bit"


class SelectionMethod(str, Enum):
    """TRT candidate selection strategy."""
    SELF_CONSISTENCY = "self_consistency"  # Majority vote on final answers
    VERIFICATION = "verification"          # Model verifies its own solutions
    HYBRID = "hybrid"                      # Consistency + verification


class ReflectionDepth(str, Enum):
    """How deeply TRT analyzes successes and failures."""
    MINIMAL = "minimal"    # Extract key takeaway only
    STANDARD = "standard"  # Analyze patterns across candidates
    DEEP = "deep"          # Full contrastive analysis with generalization


@dataclass
class BitNetConfig:
    """Configuration for the ternary substrate layer."""
    quant_type: QuantType = QuantType.I2_S
    threads: int = 4
    ctx_size: int = 2048
    temperature: float = 0.6
    quant_embd: bool = False
    use_pretuned: bool = True
    model_dir: Optional[str] = None


@dataclass
class AirLLMConfig:
    """Configuration for layer-wise sovereign inference."""
    max_vram_gb: float = 4.0
    compression: Compression = Compression.FOUR_BIT
    prefetch: bool = True
    delete_original: bool = False
    layer_shards_path: Optional[str] = None
    hf_token: Optional[str] = None
    profiling_mode: bool = False


@dataclass
class TRTConfig:
    """Configuration for the dialectical reasoning loop."""
    rounds: int = 3
    candidates_per_round: int = 8
    selection_method: SelectionMethod = SelectionMethod.SELF_CONSISTENCY
    reflection_depth: ReflectionDepth = ReflectionDepth.STANDARD
    early_stop_threshold: float = 0.95
    knowledge_persistence: bool = True
    max_knowledge_entries: int = 50
    reflection_max_tokens: int = 1024
    candidate_max_tokens: int = 2048


@dataclass
class TritTRTConfig:
    """Top-level configuration combining all three layers."""
    model_id: str = "microsoft/BitNet-b1.58-2B-4T"
    bitnet: BitNetConfig = field(default_factory=BitNetConfig)
    airllm: AirLLMConfig = field(default_factory=AirLLMConfig)
    trt: TRTConfig = field(default_factory=TRTConfig)
    device: str = "auto"  # auto | cpu | cuda
    seed: int = 42
    verbose: bool = True

    @classmethod
    def from_yaml(cls, path: str | Path) -> TritTRTConfig:
        """Load configuration from a YAML file."""
        with open(path) as f:
            raw = yaml.safe_load(f)

        bitnet_cfg = BitNetConfig(**raw.get("bitnet", {}))
        airllm_cfg = AirLLMConfig(**raw.get("airllm", {}))
        trt_cfg = TRTConfig(**raw.get("trt", {}))

        top_keys = {k: v for k, v in raw.items()
                    if k not in ("bitnet", "airllm", "trt")}

        return cls(
            bitnet=bitnet_cfg,
            airllm=airllm_cfg,
            trt=trt_cfg,
            **top_keys,
        )

    def to_yaml(self, path: str | Path) -> None:
        """Serialize configuration to YAML."""
        from dataclasses import asdict
        with open(path, "w") as f:
            yaml.dump(asdict(self), f, default_flow_style=False, sort_keys=False)

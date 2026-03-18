"""
BitNet Engine — Ternary Substrate Layer
Yunis AI — TRIT-TRT

Wraps Microsoft's bitnet.cpp inference framework for 1.58-bit LLM inference.
Handles model download, GGUF conversion, kernel selection, and text generation.

The trit {-1, 0, +1} is the atomic unit — thesis, antithesis, synthesis
encoded directly in the weight values.
"""

from __future__ import annotations
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Optional

from .config import BitNetConfig, QuantType

logger = logging.getLogger("trit-trt.bitnet")


class BitNetEngine:
    """
    Low-level interface to bitnet.cpp for ternary model inference.

    This engine handles:
    1. Model download from HuggingFace (GGUF format)
    2. Environment setup (kernel compilation)
    3. Text generation via the bitnet.cpp binary

    For layer-wise loading, see LayerShard which wraps this engine
    with AirLLM-style memory management.
    """

    def __init__(
        self,
        model_id: str,
        config: Optional[BitNetConfig] = None,
        bitnet_repo_path: Optional[str] = None,
    ):
        self.model_id = model_id
        self.config = config or BitNetConfig()
        self.bitnet_path = Path(bitnet_repo_path or self._find_bitnet_repo())
        self.model_path: Optional[Path] = None
        self.gguf_path: Optional[Path] = None
        self._binary_path: Optional[Path] = None
        self._is_setup = False

    def setup(self) -> None:
        """
        Full setup pipeline:
        1. Download model if not cached
        2. Build bitnet.cpp with appropriate kernels
        3. Locate the inference binary
        """
        logger.info(f"Setting up BitNet engine for {self.model_id}")

        # Step 1: Download/locate model
        self.model_path = self._download_model()
        self.gguf_path = self._find_gguf()

        # Step 2: Build bitnet.cpp
        self._build_bitnet()

        # Step 3: Find binary
        self._binary_path = self._locate_binary()

        self._is_setup = True
        logger.info(f"BitNet engine ready: {self.gguf_path}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: Optional[float] = None,
        conversation_mode: bool = False,
    ) -> str:
        """
        Generate text using the ternary model.

        Args:
            prompt: Input text or system prompt (if conversation_mode)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (overrides config)
            conversation_mode: Use chat mode for instruct models

        Returns:
            Generated text string
        """
        if not self._is_setup:
            self.setup()

        temp = temperature or self.config.temperature

        cmd = [
            str(self._binary_path),
            "-m", str(self.gguf_path),
            "-p", prompt,
            "-n", str(max_new_tokens),
            "-t", str(self.config.threads),
            "-c", str(self.config.ctx_size),
            "--temp", str(temp),
        ]

        if conversation_mode:
            cmd.append("-cnv")

        logger.debug(f"Running: {' '.join(cmd[:6])}...")

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
            )
            if result.returncode != 0:
                # llama-cli prints model loading info to stderr even on success
                # Only fail if stdout is empty AND returncode is non-zero
                if result.stdout.strip():
                    logger.warning(f"bitnet.cpp returned {result.returncode} but produced output")
                else:
                    raise RuntimeError(
                        f"bitnet.cpp inference failed (exit {result.returncode}): {result.stderr[-500:]}"
                    )
            return self._parse_output(result.stdout, prompt)

        except subprocess.TimeoutExpired:
            raise RuntimeError("BitNet inference timed out after 300s")

    def generate_batch(
        self,
        prompt: str,
        n: int,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
    ) -> list[str]:
        """
        Generate N candidate responses for TRT.
        Uses varied temperatures for diversity.
        """
        results = []
        for i in range(n):
            # Slightly vary temperature for candidate diversity
            temp_i = temperature + (i - n / 2) * 0.05
            temp_i = max(0.1, min(1.5, temp_i))

            output = self.generate(
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temp_i,
            )
            results.append(output)

        return results

    # ─── Internal Methods ──────────────────────────────────────────

    def _find_bitnet_repo(self) -> str:
        """Locate or clone the BitNet repository."""
        candidates = [
            Path.home() / ".yunisa" / "bitnet.cpp",
            Path.home() / "BitNet",
            Path.cwd() / "BitNet",
            Path("/opt/BitNet"),
        ]
        for p in candidates:
            if (p / "setup_env.py").exists():
                return str(p)

        # Clone if not found
        target = Path.home() / "BitNet"
        logger.info(f"Cloning BitNet to {target}")
        subprocess.run(
            ["git", "clone", "--recursive",
             "https://github.com/microsoft/BitNet.git", str(target)],
            check=True,
        )
        return str(target)

    def _download_model(self) -> Path:
        """Download model from HuggingFace if not cached."""
        if self.config.model_dir:
            p = Path(self.config.model_dir)
            if p.exists():
                return p

        # Check for existing GGUF in common locations
        yunisa_model = Path.home() / ".yunisa" / "models"
        if yunisa_model.exists():
            for f in yunisa_model.glob("*.gguf"):
                model_short = self.model_id.split("/")[-1].lower().replace("-", "")
                if model_short[:10] in f.stem.lower().replace("-", ""):
                    logger.info(f"Found existing model at {f}")
                    return f.parent

        # Use huggingface-cli to download
        model_dir = self.bitnet_path / "models" / self.model_id.split("/")[-1]
        if not model_dir.exists():
            logger.info(f"Downloading {self.model_id}...")
            gguf_repo = self.model_id
            if not gguf_repo.endswith("-gguf"):
                gguf_repo += "-gguf"
            subprocess.run(
                ["huggingface-cli", "download", gguf_repo,
                 "--local-dir", str(model_dir)],
                check=True,
            )
        return model_dir

    def _find_gguf(self) -> Path:
        """Locate the GGUF model file."""
        if self.model_path is None:
            raise RuntimeError("Model not downloaded yet")

        suffix = f"-{self.config.quant_type.value}"
        model_short = self.model_id.split("/")[-1].lower().replace("-", "")
        for f in self.model_path.rglob("*.gguf"):
            fstem = f.stem.lower().replace("-", "")
            if suffix in f.stem or "ggml-model" in f.stem or model_short[:10] in fstem:
                return f

        # If no pre-quantized GGUF, run setup_env.py to create one
        logger.info("No GGUF found, running quantization...")
        subprocess.run(
            ["python", str(self.bitnet_path / "setup_env.py"),
             "-md", str(self.model_path),
             "-q", self.config.quant_type.value],
            cwd=str(self.bitnet_path),
            check=True,
        )
        # Retry search
        for f in self.model_path.rglob("*.gguf"):
            return f

        raise FileNotFoundError(f"No GGUF file found in {self.model_path}")

    def _build_bitnet(self) -> None:
        """Build bitnet.cpp if not already built."""
        build_dir = self.bitnet_path / "build"
        binary_candidates = [
            build_dir / "bin" / "llama-cli",
            build_dir / "bin" / "llama-cli.exe",
            build_dir / "bin" / "llama-server",
            build_dir / "bin" / "llama-server.exe",
            build_dir / "bin" / "main",
            build_dir / "bin" / "main.exe",
        ]
        if any(p.exists() for p in binary_candidates):
            logger.debug("BitNet already built")
            return

        logger.info("Building bitnet.cpp...")
        subprocess.run(
            ["python", str(self.bitnet_path / "setup_env.py"),
             "-md", str(self.model_path),
             "-q", self.config.quant_type.value],
            cwd=str(self.bitnet_path),
            check=True,
        )

    def _locate_binary(self) -> Path:
        """Find the compiled inference binary."""
        build_dir = self.bitnet_path / "build"
        for name in ["llama-cli", "llama-cli.exe", "llama-server", "llama-server.exe", "main", "main.exe", "bitnet-cli", "bitnet-cli.exe"]:
            p = build_dir / "bin" / name
            if p.exists():
                return p
        raise FileNotFoundError(
            f"No inference binary found in {build_dir / 'bin'}"
        )

    def _parse_output(self, raw: str, prompt: str) -> str:
        """Extract generated text from bitnet.cpp output."""
        # bitnet.cpp outputs the full prompt + generation
        # Strip the prompt prefix and any system markers
        text = raw.strip()
        if prompt in text:
            text = text[text.index(prompt) + len(prompt):]
        # Clean up common artifacts
        for marker in ["<|end|>", "<|eot_id|>", "</s>", "[end of text]"]:
            text = text.replace(marker, "")
        return text.strip()

    @property
    def info(self) -> dict:
        """Return engine metadata."""
        return {
            "model_id": self.model_id,
            "quant_type": self.config.quant_type.value,
            "threads": self.config.threads,
            "ctx_size": self.config.ctx_size,
            "is_setup": self._is_setup,
            "gguf_path": str(self.gguf_path) if self.gguf_path else None,
        }

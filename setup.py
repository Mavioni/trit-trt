"""TRIT-TRT: Ternary Recursive Inference Thinking — Yunis AI"""

from setuptools import setup, find_packages

setup(
    name="trit-trt",
    version="0.1.0",
    author="Massimo / Yunis AI",
    description="Ternary Recursive Inference Thinking — BitNet + AirLLM + TRT",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mavioni/trit-trt",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "safetensors>=0.4.0",
        "huggingface-hub>=0.19.0",
        "airllm>=2.11.0",
        "pyyaml>=6.0",
        "pydantic>=2.0",
        "tqdm>=4.66.0",
        "psutil>=5.9.0",
        "numpy>=1.24.0",
        "orjson>=3.9.0",
    ],
    extras_require={
        "gpu": ["bitsandbytes>=0.41.0"],
        "dev": ["pytest>=7.0", "ruff>=0.1.0"],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "trit-trt=trit_trt.engine:main",
        ],
    },
)

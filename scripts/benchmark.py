#!/usr/bin/env python3
"""
TRIT-TRT Benchmark Script
Yunis AI — Sovereign Ternary Inference

Compares single-pass inference vs TRT-enhanced inference
to measure the quality uplift from dialectical reasoning.

Usage:
    python scripts/benchmark.py --model microsoft/BitNet-b1.58-2B-4T
    python scripts/benchmark.py --config configs/default.yaml
"""

import argparse
import json
import time
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from trit_trt import TritTRT, TritTRTConfig


BENCHMARK_PROMPTS = [
    # Reasoning
    "What is 17 * 23? Show your work step by step.",
    "If all roses are flowers and some flowers fade quickly, can we conclude that some roses fade quickly?",

    # Knowledge synthesis
    "Explain how a transformer neural network processes a sentence, step by step.",
    "What are the key differences between TCP and UDP? When would you use each?",

    # Creative problem solving
    "Design a simple protocol for two computers to agree on a shared secret number without anyone eavesdropping being able to figure it out.",

    # Code reasoning
    "Write a Python function that finds the longest palindromic substring in a string. Explain your approach.",
]


def run_benchmark(engine: TritTRT, prompts: list[str]) -> dict:
    """Run benchmark comparing single-pass vs TRT inference."""
    results = {
        "single_pass": [],
        "trt_enhanced": [],
        "summary": {},
    }

    print("\n" + "=" * 60)
    print("  TRIT-TRT Benchmark")
    print("=" * 60)

    for i, prompt in enumerate(prompts):
        print(f"\n--- Prompt {i+1}/{len(prompts)} ---")
        print(f"  {prompt[:80]}...")

        # Single pass (no TRT)
        t0 = time.perf_counter()
        single = engine.generate(prompt, use_trt=False)
        t_single = time.perf_counter() - t0

        results["single_pass"].append({
            "prompt": prompt,
            "response_length": len(single.text),
            "time_s": t_single,
        })
        print(f"  Single pass: {len(single.text)} chars in {t_single:.1f}s")

        # TRT enhanced
        t0 = time.perf_counter()
        trt = engine.generate(prompt, use_trt=True)
        t_trt = time.perf_counter() - t0

        results["trt_enhanced"].append({
            "prompt": prompt,
            "response_length": len(trt.text),
            "confidence": trt.confidence,
            "rounds_used": trt.rounds_used,
            "early_stopped": trt.early_stopped,
            "total_candidates": trt.total_candidates_generated,
            "insights_generated": len(trt.knowledge_log),
            "time_s": t_trt,
        })
        print(
            f"  TRT enhanced: {len(trt.text)} chars in {t_trt:.1f}s | "
            f"confidence={trt.confidence:.2%} | "
            f"rounds={trt.rounds_used} | "
            f"insights={len(trt.knowledge_log)}"
        )

    # Summary
    avg_conf = sum(
        r["confidence"] for r in results["trt_enhanced"]
    ) / len(results["trt_enhanced"])

    avg_rounds = sum(
        r["rounds_used"] for r in results["trt_enhanced"]
    ) / len(results["trt_enhanced"])

    total_insights = sum(
        r["insights_generated"] for r in results["trt_enhanced"]
    )

    results["summary"] = {
        "prompts_tested": len(prompts),
        "avg_confidence": avg_conf,
        "avg_rounds": avg_rounds,
        "total_insights": total_insights,
        "knowledge_store": engine.knowledge.get_session_stats(),
    }

    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Prompts tested:    {len(prompts)}")
    print(f"  Avg confidence:    {avg_conf:.2%}")
    print(f"  Avg rounds used:   {avg_rounds:.1f}")
    print(f"  Total insights:    {total_insights}")
    print(f"  Knowledge entries: {len(engine.knowledge.entries)}")
    print("=" * 60)

    return results


def main():
    parser = argparse.ArgumentParser(description="TRIT-TRT Benchmark")
    parser.add_argument(
        "--model", "-m",
        default="microsoft/BitNet-b1.58-2B-4T",
        help="Model ID",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Path to YAML config",
    )
    parser.add_argument(
        "--rounds", "-r",
        type=int, default=3,
        help="TRT rounds",
    )
    parser.add_argument(
        "--candidates", "-n",
        type=int, default=8,
        help="Candidates per round",
    )
    parser.add_argument(
        "--output", "-o",
        default="benchmark_results.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    if args.config:
        config = TritTRTConfig.from_yaml(args.config)
        engine = TritTRT(config=config)
    else:
        engine = TritTRT(
            model_id=args.model,
            trt_rounds=args.rounds,
            trt_candidates=args.candidates,
        )

    results = run_benchmark(engine, BENCHMARK_PROMPTS)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()

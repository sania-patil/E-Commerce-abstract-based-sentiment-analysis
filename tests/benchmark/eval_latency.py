"""
Benchmark: Inference Latency

Measures single-review inference latency on CPU.
Target: <= 500ms per review.

Usage:
    python -m tests.benchmark.eval_latency
    python -m tests.benchmark.eval_latency --checkpoint models/best_model.pt --runs 20
"""

import argparse
import time
from absa.pipeline import ABSAPipeline

TEST_REVIEWS = [
    "The battery life is great but the keyboard feels cheap.",
    "Screen resolution is excellent and the trackpad is very responsive.",
    "This laptop runs hot and the fan noise is unbearable.",
    "Good performance for the price but the build quality is disappointing.",
    "The display is stunning and boot time is incredibly fast.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="models/best_model.pt")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--runs", type=int, default=10, help="Number of inference runs per review")
    args = parser.parse_args()

    print("Loading pipeline...")
    pipeline = ABSAPipeline(args.checkpoint, args.config)

    print(f"Running latency benchmark ({args.runs} runs per review, {len(TEST_REVIEWS)} reviews)...\n")

    latencies = []
    for review in TEST_REVIEWS:
        times = []
        for _ in range(args.runs):
            start = time.perf_counter()
            pipeline.run(review)
            elapsed_ms = (time.perf_counter() - start) * 1000
            times.append(elapsed_ms)

        avg_ms = sum(times) / len(times)
        min_ms = min(times)
        max_ms = max(times)
        latencies.extend(times)
        status = "PASS" if avg_ms <= 500 else "SLOW"
        print(f"  [{status}] avg={avg_ms:.1f}ms  min={min_ms:.1f}ms  max={max_ms:.1f}ms")
        print(f"         Review: \"{review[:60]}\"")

    overall_avg = sum(latencies) / len(latencies)
    overall_max = max(latencies)

    print(f"\n=== Latency Summary ===")
    print(f"  Overall avg: {overall_avg:.1f}ms")
    print(f"  Overall max: {overall_max:.1f}ms")
    print(f"  Target:      <= 500ms")
    status = "PASS" if overall_avg <= 500 else "BELOW TARGET"
    print(f"  Status:      {status}")


if __name__ == "__main__":
    main()

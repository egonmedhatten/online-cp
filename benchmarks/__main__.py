"""CLI entry point for model benchmarking.

Usage:
    python -m z_benchmark_models [--task classification|regression|cps|venn|all] [--quick] [--workers N]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

from benchmarks.configs import get_configs
from benchmarks.datasets import load_datasets
from benchmarks.runner import run_benchmark


def _format_table(results, metric_name, task_label):
    """Format results as a markdown table for a single metric."""
    if not results:
        return ""

    # Group by dataset (rows) and model (columns)
    datasets = []
    models = []
    for r in results:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])
        if r["model"] not in models:
            models.append(r["model"])

    # Build table
    lines = [f"### {metric_name} ({task_label})", ""]
    header = "| Dataset | " + " | ".join(models) + " |"
    sep = "|" + "---|" * (len(models) + 1)
    lines.append(header)
    lines.append(sep)

    for ds in datasets:
        row = f"| {ds} |"
        for m in models:
            match = [r for r in results if r["dataset"] == ds and r["model"] == m]
            if match and metric_name in match[0]:
                val = match[0][metric_name]
                if val is None or val != val:  # NaN check
                    row += " — |"
                elif metric_name == "time_s":
                    row += f" {val:.3f} |"
                else:
                    row += f" {val:.4f} |"
            else:
                row += " — |"
        lines.append(row)

    lines.append("")
    return "\n".join(lines)


def _run_one(args):
    """Worker function for parallel execution."""
    config, dataset, epsilon, cap = args
    return run_benchmark(config, dataset, epsilon=epsilon, cap=cap)


def main():
    parser = argparse.ArgumentParser(description="Online-CP model benchmarks")
    parser.add_argument(
        "--task",
        choices=["classification", "regression", "cps", "venn", "all"],
        default="all",
        help="Which model family to benchmark (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Cap test set at 200 examples for faster runs",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1 = sequential)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="Significance level (default: 0.1)",
    )
    args = parser.parse_args()

    cap = 200 if args.quick else None

    # Load datasets
    if args.task in ("regression", "cps"):
        datasets = load_datasets("regression")
    elif args.task in ("classification", "venn"):
        datasets = load_datasets("classification")
    else:
        datasets = load_datasets("all")

    # Load configs
    configs = get_configs(args.task)

    print(f"# Online-CP Model Benchmark")
    print(f"")
    print(f"- Task: {args.task}")
    print(f"- Epsilon: {args.epsilon}")
    print(f"- Datasets: {len(datasets)}")
    print(f"- Models: {len(configs)}")
    if cap:
        print(f"- Quick mode: test set capped at {cap}")
    print("")

    # Build jobs
    jobs = [(config, dataset, args.epsilon, cap) for config in configs for dataset in datasets]

    # Execute
    results = []
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(_run_one, job): job for job in jobs}
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    results.append(result)
    else:
        for job in jobs:
            result = _run_one(job)
            if result is not None:
                results.append(result)

    if not results:
        print("No results produced. Check task/dataset compatibility.")
        sys.exit(0)

    # Group results by task type for reporting
    task_groups = {}
    for r in results:
        t = r["task"]
        task_groups.setdefault(t, []).append(r)

    # Determine metrics to report per task
    metric_map = {
        "regression": ["ErrorRate", "IntervalWidth", "WinklerScore", "time_s"],
        "classification": ["ErrorRate", "SetSize", "ObservedExcess", "ObservedFuzziness", "time_s"],
        "cps": ["ErrorRate", "IntervalWidth", "CRPS", "time_s"],
        "venn": ["BrierScore", "LogLoss", "Width", "time_s"],
    }

    for task_name, task_results in sorted(task_groups.items()):
        print(f"## {task_name.title()}")
        print("")
        for metric_name in metric_map.get(task_name, ["time_s"]):
            table = _format_table(task_results, metric_name, task_name)
            if table:
                print(table)

    print("---")
    print("Done.")


if __name__ == "__main__":
    main()

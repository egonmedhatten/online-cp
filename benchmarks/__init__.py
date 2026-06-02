"""Model benchmarking infrastructure for online-cp."""

from benchmarks.configs import get_configs
from benchmarks.datasets import load_datasets
from benchmarks.runner import run_benchmark

__all__ = ["get_configs", "load_datasets", "run_benchmark"]

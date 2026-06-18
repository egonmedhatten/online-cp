"""Smoke tests for the model benchmarking infrastructure."""

import warnings

from benchmarks.configs import get_configs
from benchmarks.datasets import load_datasets
from benchmarks.runner import run_benchmark


class TestDatasets:
    def test_classification_datasets_load(self):
        datasets = load_datasets("classification")
        assert len(datasets) == 3
        for ds in datasets:
            assert "X" in ds and "y" in ds and "metadata" in ds
            assert ds["metadata"]["task"] == "classification"
            assert ds["X"].shape[0] == ds["metadata"]["n"]
            assert ds["X"].shape[1] == ds["metadata"]["d"]
            assert len(ds["y"]) == ds["metadata"]["n"]

    def test_regression_datasets_load(self):
        datasets = load_datasets("regression")
        assert len(datasets) == 2
        for ds in datasets:
            assert ds["metadata"]["task"] == "regression"

    def test_all_datasets_load(self):
        datasets = load_datasets("all")
        assert len(datasets) == 5


class TestConfigs:
    def test_classification_configs_instantiate(self):
        configs = get_configs("classification")
        assert len(configs) >= 3
        for cfg in configs:
            assert cfg["task"] == "classification"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = cfg["factory"]()
            assert hasattr(model, "predict")
            assert hasattr(model, "learn_one")

    def test_regression_configs_instantiate(self):
        configs = get_configs("regression")
        assert len(configs) >= 2
        for cfg in configs:
            assert cfg["task"] == "regression"
            model = cfg["factory"]()
            assert hasattr(model, "predict")

    def test_cps_configs_instantiate(self):
        configs = get_configs("cps")
        assert len(configs) >= 2
        for cfg in configs:
            assert cfg["task"] == "cps"
            model = cfg["factory"]()
            assert hasattr(model, "predict_cpd")

    def test_venn_configs_instantiate(self):
        configs = get_configs("venn")
        assert len(configs) >= 2
        for cfg in configs:
            assert cfg["task"] == "venn"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = cfg["factory"]()
            assert hasattr(model, "predict") or hasattr(model, "predict_one")
            assert hasattr(model, "learn_one")


class TestRunner:
    def test_regression_benchmark_smoke(self):
        configs = get_configs("regression")
        datasets = load_datasets("regression")
        result = run_benchmark(configs[0], datasets[0], cap=50)
        assert result is not None
        assert "ErrorRate" in result
        assert "IntervalWidth" in result
        assert "time_s" in result
        assert 0 <= result["ErrorRate"] <= 1
        assert result["time_s"] > 0

    def test_classification_benchmark_smoke(self):
        configs = get_configs("classification")
        datasets = load_datasets("classification")
        # Use iris (3-class)
        result = run_benchmark(configs[0], datasets[0], cap=50)
        assert result is not None
        assert "ErrorRate" in result
        assert "SetSize" in result
        assert 0 <= result["ErrorRate"] <= 1

    def test_cps_benchmark_smoke(self):
        configs = get_configs("cps")
        datasets = load_datasets("regression")
        result = run_benchmark(configs[0], datasets[0], cap=50)
        assert result is not None
        assert "CRPS" in result

    def test_venn_benchmark_smoke(self):
        configs = get_configs("venn")
        datasets = load_datasets("classification")
        # breast_cancer is binary
        ds_binary = [d for d in datasets if d["metadata"]["name"] == "breast_cancer"][0]
        result = run_benchmark(configs[0], ds_binary, cap=50)
        assert result is not None
        assert "BrierScore" in result
        assert "LogLoss" in result

    def test_incompatible_task_returns_none(self):
        """Regression model on classification dataset returns None."""
        reg_config = get_configs("regression")[0]
        cls_dataset = load_datasets("classification")[0]
        result = run_benchmark(reg_config, cls_dataset)
        assert result is None

    def test_binary_only_skips_multiclass(self):
        """Binary-only Venn model skips multiclass dataset."""
        venn_configs = get_configs("venn")
        binary_only = [c for c in venn_configs if c["binary_only"]][0]
        iris = load_datasets("classification")[0]  # 3-class
        result = run_benchmark(binary_only, iris)
        assert result is None

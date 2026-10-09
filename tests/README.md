# Test suite

This document explains the test suite structure, the **fast/slow lane split**,
and why some tests may appear as **skipped** or **deselected** when you run
`uv run pytest`.

## Lanes

| Lane | Marker | Command | Purpose |
|------|--------|---------|---------|
| **Fast** (default) | not `slow` | `uv run pytest` | Structural tests, unit tests, leancheck property tests, doctests. Runs in ~55 s. |
| **Slow** (opt-in) | `slow` | `uv run pytest -m slow` | Heavy statistical-validity tests: coverage over long generated streams, law-of-large-numbers checks. Runs in minutes. |

The default lane is configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "-m 'not slow'"
markers = ["slow: statistical-validity tests; run with -m slow"]
```

## "Deselected" tests (12 by default)

`uv run pytest` reports **12 deselected**. These are *not* skipped — they are
**filtered out before collection** by the `-m 'not slow'` marker expression.
They are collected, but never executed, because they are statistically
expensive (long generated streams) and would slow the default lane down.

The 12 deselected tests are:

| File | Test | Why it is slow |
|------|------|----------------|
| `tests/test_mondrian_online.py` | `TestOnlineRegressor::test_coverage_tracks_epsilon` | Long online stream |
| `tests/test_mondrian_online.py` | `TestOnlineRegressor::test_coverage_matches_batch` | Long online stream |
| `tests/test_mondrian_online.py` | `TestOnlineForestClassifier::test_coverage_tracks_epsilon` | Long online stream |
| `tests/test_mondrian_online.py` | `TestOnlineForestClassifier::test_coverage_matches_batch` | Long online stream |
| `tests/test_mondrian_online.py` | `TestOnlineForestRegressor::test_coverage_tracks_epsilon` | Long online stream |
| `tests/test_mondrian_online.py` | `TestOnlineForestRegressor::test_coverage_matches_batch` | Long online stream |
| `tests/test_statistical.py` | `test_ridge_regressor_coverage_validity` | Law-of-large-numbers |
| `tests/test_statistical.py` | `test_mondrian_tree_regressor_coverage_validity` | Law-of-large-numbers |
| `tests/test_statistical.py` | `test_knn_classifier_coverage_validity` | Law-of-large-numbers |
| `tests/test_statistical.py` | `test_venn_abers_calibration_in_the_large` | Calibration at scale |
| `tests/test_statistical.py` | `test_ville_false_alarm_rate_under_null` | False-alarm rate |
| `tests/test_statistical.py` | `test_ville_detects_distribution_shift` | Distribution-shift detection |

To run them explicitly:

```bash
uv run pytest -m slow
```

CI runs these nightly (see `.github/workflows/test.yml`, the `statistical` job
triggered by `schedule` and `workflow_dispatch`).

## "Skipped" tests (environment-dependent)

A test is **skipped** when it *is* collected but a runtime condition is not
met (e.g. an optional dependency is missing). This is different from
deselected. The count of skipped tests depends on your environment.

### Graphviz-backed Mondrian tree rendering (6 tests)

File: `tests/test_mondrian_tree.py`, lines ~1869–1910.

These tests exercise the `draw(backend="graphviz")` path of the Mondrian tree
classifier. They are guarded by:

```python
@pytest.mark.skipif(not _has_graphviz(), reason="graphviz not installed")
```

`_has_graphviz()` (defined in `src/online_cp/mondrian/tree.py:990`) returns
`True` only when **both** of the following are present:

1. The Python `graphviz` package is importable.
2. The system `dot` binary is on `PATH`.

If either is missing, all 6 tests skip with the reason
`"graphviz not installed"`. On machines where both are installed (e.g. this
development machine, and the CI runner which runs `apt install graphviz` via
the `ci` extra), all 6 tests **run and pass** — so the "6 skipped" the user
may have seen is environment-specific.

To enable them locally:

```bash
# Debian/Ubuntu
sudo apt install graphviz

# macOS
brew install graphviz

# Verify
uv run python -c "from online_cp.mondrian.tree import _has_graphviz; print(_has_graphviz())"
# Expected: True
```

### tqdm progress bar (1 test)

File: `tests/test_streaming_and_plotting.py`, `test_progressive_val_with_progress_bar`.

Skipped only if `tqdm` is not importable. Since `tqdm` is a core dependency
(declared in `pyproject.toml` under `[project] dependencies`), this test
should **never** skip in a normal `uv sync` environment.

```python
try:
    import tqdm  # noqa: F401
except ImportError:
    pytest.skip("tqdm not installed; skipping progress bar test")
```

## Verifying the current state

```bash
# Fast lane (default): should report "1308 passed, 12 deselected, 0 warnings"
uv run pytest -W always -q --no-header

# Slow lane: should report "12 passed"
uv run pytest -m slow -q --no-header
```

If you see a non-zero "skipped" count, check the environment-specific
dependencies above (graphviz/`dot`, tqdm) and the skip reason pytest prints
on the line before the summary.

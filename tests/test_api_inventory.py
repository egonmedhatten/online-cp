"""Public-API inventory test.

Guards that *every* public symbol a module advertises in its ``__all__`` is
referenced by at least one test in ``tests/``.

This is intentionally **not** a line-coverage metric. Line coverage only counts
whether a test *touches* a line; it can be satisfied trivially and gives a false
sense of security. What matters here is a different, cheaper property: a
documented public symbol (a class or function a user is promised to import)
must have at least one test that *names* it. If a symbol is exported but no
test references it, that symbol's contract is unverified — which is exactly the
"public API with no test behind it" gap this test is meant to surface.

The test

1. walks ``src/online_cp`` and imports every *public* module (skipping
   ``_dev`` development modules and private ``_*`` helpers that are not part
   of the advertised API),
2. collects the union of every module's ``__all__``, and
3. asserts that each non-private symbol appears as a token in the source of
   the test suite.

A symbol "appears" when its name occurs as a standalone word anywhere under
``tests/`` — i.e. a test imports, constructs, or otherwise uses it. This is
deliberately conservative: it does not demand a specific kind of assertion,
only that the symbol is exercised by name.
"""

import importlib
import pathlib
import re

import pytest

SRC_ROOT = pathlib.Path(__file__).resolve().parent.parent / "src"
TESTS_ROOT = pathlib.Path(__file__).resolve().parent


def _public_modules():
    """Yield importable dotted module names for every source file we audit.

    We audit a module if it *advertises* an API, i.e. it defines ``__all__``.
    The walk is therefore driven by presence of ``__all__`` rather than by a
    name heuristic, so a private helper (e.g. ``_serialization``) that exposes
    a public class is still held to the same standard.

    We also audit package ``__init__.py`` files, because their ``__all__`` is
    the re-export surface users actually import from (e.g.
    ``from online_cp.martingale import STANDARD_GRID``). A symbol re-exported
    through a package ``__all__`` is public API even if the underlying module
    it comes from does not itself declare ``__all__``.

    Excluded:

    * ``*_dev.py`` development modules — secret research code that is
      deliberately gitignored and not part of the public API.
    """
    for path in sorted(SRC_ROOT.rglob("*.py")):
        name = path.name
        if name.endswith("_dev.py"):
            continue
        rel = path.relative_to(SRC_ROOT / "online_cp")  # e.g. 'CPS.py', 'mondrian/tree.py'
        if name == "__init__.py":
            # Dotted name of the *package* this __init__ belongs to. The top
            # package lives directly under online_cp/ (rel.parent == ".").
            package = rel.parent.as_posix().replace("/", ".")
            if package in ("", "."):
                yield "online_cp"
            else:
                yield f"online_cp.{package}"
        else:
            dotted = rel.with_suffix("").as_posix().replace("/", ".")
            yield f"online_cp.{dotted}"


def _module_public_symbols():
    """Map module name -> set of non-private ``__all__`` symbols it advertises."""
    symbols: dict[str, set[str]] = {}
    for dotted in _public_modules():
        try:
            module = importlib.import_module(dotted)
        except Exception:
            # A module that cannot be imported is itself a failure we want to
            # surface explicitly rather than silently skip.
            pytest.fail(f"could not import public module {dotted}")
        all_names = getattr(module, "__all__", None)
        if not all_names:
            continue
        symbols[dotted] = {n for n in all_names if not n.startswith("_")}
    return symbols


def _test_source_text() -> str:
    """Concatenate the source of every test file (excluding ``__init__.py``).

    This file (the inventory test itself) is excluded from the haystack:
    mentions of a symbol in its own docstrings or error messages are not
    evidence that the symbol is exercised, so counting them would produce
    false negatives.
    """
    parts = []
    own_name = pathlib.Path(__file__).resolve().name
    for path in sorted(TESTS_ROOT.glob("*.py")):
        if path.name == "__init__.py" or path.name == own_name:
            continue
        parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _referenced(name: str, haystack: str) -> bool:
    """True if *name* appears as a standalone word anywhere in *haystack*."""
    return re.search(rf"\b{re.escape(name)}\b", haystack) is not None


class TestPublicApiInventory:
    def test_every_public_symbol_is_tested(self):
        symbols = _module_public_symbols()
        haystack = _test_source_text()

        # Sanity: we must actually be looking at a meaningful API surface.
        total = sum(len(s) for s in symbols.values())
        assert total > 0, "no __all__ symbols found; is the walk broken?"

        missing = sorted(
            (mod, name) for mod, names in symbols.items() for name in names if not _referenced(name, haystack)
        )

        if missing:
            listing = "\n".join(f"  {mod}.{name}" for mod, name in missing)
            pytest.fail(
                f"{len(missing)} public symbol(s) are advertised in __all__ but "
                f"never referenced by any test:\n{listing}\n\n"
                f"Add at least one test that exercises each."
            )

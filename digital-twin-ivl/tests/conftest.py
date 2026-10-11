"""Pytest configuration: make ``dtivl`` importable without installation."""
import importlib.util
import sys
from pathlib import Path

if "dtivl" not in sys.modules:
    try:
        import dtivl  # noqa: F401  (editable install present)
    except ImportError:  # fall back to mapping dtivl -> src/
        src = Path(__file__).resolve().parents[1] / "src"
        spec = importlib.util.spec_from_file_location("dtivl", src / "__init__.py",
                                                      submodule_search_locations=[str(src)])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["dtivl"] = mod
        spec.loader.exec_module(mod)

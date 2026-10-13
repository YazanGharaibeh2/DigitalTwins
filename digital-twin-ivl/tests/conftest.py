"""Pytest configuration: import ``dtivl`` from this checkout's ``src/``.

The editable install may point to a different checkout (for example the main
working copy when the tests run in a git worktree), so the package name is
always mapped to the ``src/`` directory next to this ``tests/`` directory,
unless ``dtivl`` already resolves there.
"""
import importlib.util
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"


def _is_local(mod) -> bool:
    f = getattr(mod, "__file__", None)
    return f is not None and Path(f).resolve().parent == _SRC


if not _is_local(sys.modules.get("dtivl")):
    for name in [m for m in sys.modules if m == "dtivl" or m.startswith("dtivl.")]:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location("dtivl", _SRC / "__init__.py",
                                                  submodule_search_locations=[str(_SRC)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dtivl"] = mod
    spec.loader.exec_module(mod)

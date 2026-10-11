"""dtivl: in silico digital twins of calcified coronary lesions under IVL and stenting.

The import name ``dtivl`` maps to the ``src/`` directory (see pyproject.toml).
"""
from pathlib import Path

__version__ = "0.1.0"

#: Project root (the directory that contains ``params/``, ``results/`` ...).
ROOT = Path(__file__).resolve().parent.parent

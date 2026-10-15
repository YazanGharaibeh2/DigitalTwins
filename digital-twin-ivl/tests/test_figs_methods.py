"""Tests for Figures 1 and 3 and Table 1 (built from params/, cohort/ and results/ files)."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import math

import pandas as pd
import pytest
from PIL import Image

from dtivl import params as P
from dtivl.drivers import figstyle
from dtivl.drivers.figs import fig1_design, fig3_verification
from dtivl.drivers import table1_parameters as T1


@pytest.mark.parametrize("mod, name", [(fig1_design, "fig1_design"), (fig3_verification, "fig3_verification")])
def test_figure_builds_at_print_resolution(mod, name):
    path = mod.make()
    assert path.endswith(f"{name}.png")
    with Image.open(path) as im:
        dpi = im.info.get("dpi", (0, 0))[0]
        assert round(dpi) >= 300 and im.size[0] >= 300 * 3      # at least 3 in wide at >= 300 dpi
    assert round(dpi) == figstyle.DPI


def test_table1_values_come_from_params():
    T1.make()
    df = pd.read_csv(T1.TABLE_DIR / "table1_parameters.csv")
    assert "UNVERIFIED" not in set(df["status"].dropna())
    assert df["source"].fillna("").str.len().gt(0).all() or df["basis"].fillna("").str.len().gt(0).all()
    checked = 0
    for _, r in df.iterrows():
        key = r["config_key"]
        if r.get("value_from") != "params" or not isinstance(key, str):
            continue
        try:
            q = P.get(key)
        except Exception:
            continue
        v = pd.to_numeric(r["value"], errors="coerce")
        if isinstance(q.value, float) and not math.isnan(v):
            assert v == pytest.approx(q.value, rel=1e-9), key
            checked += 1
    assert checked >= 10

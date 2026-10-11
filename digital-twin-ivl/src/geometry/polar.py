"""Polar label maps in the IVOCT convention (rows = A-lines, columns = depth)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .labels import LABEL_LUMEN


@dataclass
class PolarLabelMap:
    """A single IVOCT-like cross-section.

    Attributes
    ----------
    labels : (n_theta, n_r) uint8 array. Row i is the A-line at angle
        ``theta[i] = 2*pi*i/n_theta`` (counter-clockwise from +x); column j is
        the sample at radius ``r[j] = (j + 0.5) * dr`` from the catheter /
        polar centre (the origin of the Cartesian frame).
    dr : radial sample spacing (mm).
    r_out : (n_theta,) outer vessel-wall radius per A-line (mm).
    r_iel : (n_theta,) internal-elastic-lamina radius per A-line (mm), or
        None. Tissue labelled 0 with r < r_iel is intima/plaque (fibrous);
        r >= r_iel is media/adventitia (wall). If None, real-data defaults
        apply (see dtivl.meshing).
    meta : free-form provenance (lesion id, spec, source image ...).
    """

    labels: np.ndarray
    dr: float
    r_out: np.ndarray
    r_iel: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    @property
    def n_theta(self) -> int:
        return self.labels.shape[0]

    @property
    def n_r(self) -> int:
        return self.labels.shape[1]

    @property
    def theta(self) -> np.ndarray:
        return 2.0 * np.pi * np.arange(self.n_theta) / self.n_theta

    @property
    def r(self) -> np.ndarray:
        return (np.arange(self.n_r) + 0.5) * self.dr

    def validate(self) -> None:
        assert self.labels.ndim == 2 and self.labels.dtype == np.uint8
        assert self.r_out.shape == (self.n_theta,)
        if self.r_iel is not None:
            assert self.r_iel.shape == (self.n_theta,)
            assert np.all(self.r_iel < self.r_out)
        # every A-line must start in the lumen
        assert np.all(self.labels[:, 0] == LABEL_LUMEN), "polar centre must lie in the lumen"

    def save(self, path: str | Path) -> None:
        np.savez_compressed(path, labels=self.labels, dr=self.dr, r_out=self.r_out,
                            r_iel=self.r_iel if self.r_iel is not None else np.array([]),
                            meta=np.array(repr(self.meta)))

    @classmethod
    def load(cls, path: str | Path) -> "PolarLabelMap":
        z = np.load(path, allow_pickle=False)
        r_iel = z["r_iel"] if z["r_iel"].size else None
        return cls(labels=z["labels"], dr=float(z["dr"]), r_out=z["r_out"], r_iel=r_iel,
                   meta={"repr": str(z["meta"])})

    def to_cartesian(self, pixel: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Nearest-neighbour resampling to a square Cartesian label image.

        Returns (image, extent) with pixels outside r_out set to 255.
        extent = (xmin, xmax, ymin, ymax) in mm.
        """
        pixel = pixel or self.dr
        rmax = float(self.r_out.max()) * 1.02
        n = int(np.ceil(2 * rmax / pixel))
        c = (np.arange(n) + 0.5) * pixel - rmax
        X, Y = np.meshgrid(c, c, indexing="xy")
        R = np.hypot(X, Y)
        T = np.mod(np.arctan2(Y, X), 2 * np.pi)
        it = np.mod(np.rint(T / (2 * np.pi) * self.n_theta).astype(int), self.n_theta)
        ir = np.clip((R / self.dr).astype(int), 0, self.n_r - 1)
        img = self.labels[it, ir].copy()
        img[R > self.r_out[it]] = 255
        return img, (-rmax, rmax, -rmax, rmax)

"""Lesion specification (all lengths in mm, angles in degrees)."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

MORPHOLOGIES = ("superficial_sheet", "deep_sheet", "nodule", "fragmented")


@dataclass
class LesionSpec:
    """Geometric description of one idealised calcified coronary lesion.

    Frame: the polar/catheter centre is the lumen centre (origin). The
    calcium arc is centred on ``ca_center_deg`` (0 = +x axis, CCW positive).

    Plaque/IEL: the internal elastic lamina is a circle of radius
    ``r_lumen + plaque_thickness`` whose centre is shifted by
    ``plaque_offset`` toward ``ca_center_deg``; the plaque is therefore
    thickest on the calcium side when ``plaque_offset > 0``. The outer wall is
    the IEL circle grown by ``wall_thickness`` (same centre).

    Calcium:
      * ``superficial_sheet`` / ``deep_sheet``: an annular sector between radii
        ``r_lumen + ca_depth`` and ``r_lumen + ca_depth + ca_thickness``
        spanning ``ca_arc_deg`` (rounded ends: semicircular caps of diameter
        ``ca_thickness`` lie inside the arc; a 360 deg sheet is a closed ring).
      * ``nodule``: a superficial sheet plus a protrusion of height
        ``nodule_height`` into the lumen spanning ``nodule_base_arc_deg`` at the
        arc centre; the lumen boundary follows the protrusion's covering layer
        (thickness ``ca_depth``).
      * ``fragmented``: ``n = max(2, round(arc/fragment_pitch_deg))`` deposits of
        the sheet cross-section, each filling ``fragment_fill`` of its pitch.
    Lipid (hold-out only): annular sector at depth ``lipid_cap_thickness``,
    thickness ``lipid_thickness``, arc ``lipid_arc_deg`` centred on
    ``lipid_center_deg``; must not overlap calcium.
    3D: ``ca_length`` is the axial calcium length; the segment adds
    ``margin_3d`` of calcium-free vessel at each end.
    """

    lesion_id: str
    morphology: str
    ca_arc_deg: float
    ca_thickness: float
    ca_depth: float
    r_lumen: float
    plaque_thickness: float
    wall_thickness: float
    plaque_offset: float = 0.0
    ca_center_deg: float = 0.0
    ca_length: float | None = None          # None -> 2D
    margin_3d: float = 2.0
    nodule_height: float = 0.0
    nodule_base_arc_deg: float = 60.0
    fragment_pitch_deg: float = 45.0
    fragment_fill: float = 0.6
    lipid_arc_deg: float = 0.0              # 0 -> no lipid
    lipid_thickness: float = 0.0
    lipid_cap_thickness: float = 0.0
    lipid_center_deg: float = 180.0
    set_name: str = "core"                  # core | holdout | pilot | example3d
    notes: dict = field(default_factory=dict)

    @property
    def is_3d(self) -> bool:
        return self.ca_length is not None

    @property
    def n_fragments(self) -> int:
        if self.morphology != "fragmented":
            return 1
        return max(2, int(round(self.ca_arc_deg / self.fragment_pitch_deg)))

    def validate(self) -> None:
        if self.morphology not in MORPHOLOGIES:
            raise ValueError(f"unknown morphology {self.morphology}")
        if not (0 < self.ca_arc_deg <= 360):
            raise ValueError("ca_arc_deg must be in (0, 360]")
        if self.ca_thickness <= 0 or self.ca_depth < 0:
            raise ValueError("calcium thickness must be > 0 and depth >= 0")
        r_iel_min = self.r_lumen + self.plaque_thickness - abs(self.plaque_offset)
        ca_outer = self.r_lumen + self.ca_depth + self.ca_thickness
        if self.plaque_offset == 0 and ca_outer >= self.r_lumen + self.plaque_thickness:
            raise ValueError(f"{self.lesion_id}: calcium reaches the IEL (outer {ca_outer:.3f} >= IEL {self.r_lumen + self.plaque_thickness:.3f})")
        if r_iel_min <= self.r_lumen:
            raise ValueError("plaque_offset too large: IEL intersects lumen")
        if self.lipid_arc_deg > 0:
            if self.lipid_thickness <= 0:
                raise ValueError("lipid thickness must be > 0 when lipid is present")
            # angular overlap check (centres and half-arcs)
            dc = abs((self.lipid_center_deg - self.ca_center_deg + 180) % 360 - 180)
            if dc < (self.lipid_arc_deg + self.ca_arc_deg) / 2:
                raise ValueError("lipid arc overlaps calcium arc")

    def to_dict(self) -> dict:
        d = asdict(self)
        d["is_3d"] = self.is_3d
        d["n_fragments"] = self.n_fragments
        d["lumen_area_nominal"] = math.pi * self.r_lumen ** 2  # circle; actual area from descriptors
        return d

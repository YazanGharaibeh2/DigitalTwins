"""Label and tag conventions shared by every module.

Polar label maps follow the IVOCT segmentation convention of the PI's group:
0 = other (any non-lumen, non-calcium tissue), 1 = lumen, 4 = calcium.
Synthetic lesions may additionally carry 5 = lipid. Labels 2 and 3 are left
unused so that the convention stays identical to the real-segmentation masks.
Tissue outside the outer vessel boundary is NOT a label: the outer boundary
(and optionally the internal elastic lamina) are carried as per-A-line radius
profiles in :class:`dtivl.geometry.polar.PolarLabelMap`.
"""
LABEL_OTHER = 0
LABEL_LUMEN = 1
LABEL_CALCIUM = 4
LABEL_LIPID = 5

# FE cell tags (gmsh physical surface/volume ids -> dolfinx cell_tags values)
TAG_WALL = 1      # media + adventitia (outside the internal elastic lamina)
TAG_FIBROUS = 2   # intima / plaque "other" tissue inside the IEL
TAG_LIPID = 3
TAG_CALCIUM = 4

# FE facet tags
TAG_LUMEN_BND = 10
TAG_OUTER_BND = 20
TAG_END_BND = 30   # 3D segment end faces (z = 0 and z = L)

MATERIAL_NAMES = {TAG_WALL: "wall", TAG_FIBROUS: "fibrous", TAG_LIPID: "lipid", TAG_CALCIUM: "calcium"}

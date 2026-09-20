"""DRDO ID26053 — the single source of truth for label taxonomies.

Three taxonomies exist in this project and they are deliberately separate:

  RAW   RELLIS-3D ontology ids as stored in the .label files (0..34, sparse).
  FINE  the 12 classes the network is trained on. Chosen so that PERSON and VEHICLE stay
        distinct (the problem statement asks for pedestrian vs vehicle), while rare structural
        classes are merged. Every merge stays *within* one ADL-1 group, so merging costs the
        C++ grid engine nothing.
  ADL-1 the 8 classes the C++ engine already understands (SEM_TRAV[8] in drdo_map.h). Never
        widened — changing it would touch the CUDA literals, the cost logic and the regression
        gates.

Nothing else in the codebase may define a label mapping. dataset, train, eval, inference and the
ROS nodes all import from here, so a retune is one edit plus an eval rerun, never a retrain.

Why FINE -> ADL-1 is a lookup and not a wider engine taxonomy: the collapse becomes a data
decision. Whether TREE is dense vegetation or a hard obstacle, whether mud is passable — each is
a table entry you can flip and re-measure with scripts/eval.py in minutes instead of GPU-days.
"""
from typing import Dict, Optional, Tuple

import numpy as np

# ── ADL-1: the 8 classes the C++ engine consumes (drdo_map.h SEM_TRAV) ───────────────────────
ADL1_GROUND = 0
ADL1_GRAVEL_DIRT = 1
ADL1_GRASS_LOW = 2
ADL1_VEGETATION_DENSE = 3
ADL1_OBSTACLE_HARD = 4
ADL1_WATER_MUD = 5
ADL1_UNKNOWN = 6
ADL1_SKY_DUST = 7          # reserved for the geometric dust filter; the network never emits it
NUM_ADL1 = 8

ADL1_NAMES = ("GROUND", "GRAVEL_DIRT", "GRASS_LOW", "VEGETATION_DENSE",
              "OBSTACLE_HARD", "WATER_MUD", "UNKNOWN", "SKY_DUST")

# ── FINE: the 12 classes the network predicts ────────────────────────────────────────────────
FINE_NAMES = ("ASPHALT_CONCRETE", "DIRT", "GRASS", "BUSH", "TREE", "PERSON",
              "VEHICLE", "POLE", "STRUCTURE", "OBJECT_DEBRIS", "WATER", "MUD")
NUM_FINE = len(FINE_NAMES)

IGNORE_INDEX = 255          # void / sky: excluded from the loss, never written to the map

# ── RAW (RELLIS ontology) -> FINE ────────────────────────────────────────────────────────────
# Ids and names are the dataset's own, from benchmarks/.../config/labels/rellis.yaml:
#   0 void  1 dirt  3 grass  4 tree  5 pole  6 water  7 sky  8 vehicle  9 object  10 asphalt
#   12 building  15 log  17 person  18 fence  19 bush  23 concrete  27 barrier  31 puddle
#   33 mud  34 rubble
# Ids 29, 30 and 32 carry no name in the ontology but appear in the data; the official
# learning_map sends 29 and 30 to the same target as grass(3) and 32 to the same as water(6).
#
# This table replaces an earlier mapping that had 29 as "puddle" (29 is a grass variant; puddle
# is 31), 31 as "mud" (31 is puddle), and 33 as "rubberChips" routed to vegetation (33 is mud).
# That last one sent a hazard to a traversability of 0.15 instead of 0.0.
RAW_TO_FINE: Dict[int, int] = {
    0:  IGNORE_INDEX,   # void   — unlabeled returns, not atmospheric noise: ignore, never discard
    1:  1,              # dirt            -> DIRT
    3:  2,              # grass           -> GRASS
    4:  4,              # tree            -> TREE
    5:  7,              # pole            -> POLE
    6:  10,             # water           -> WATER
    7:  IGNORE_INDEX,   # sky
    8:  6,              # vehicle         -> VEHICLE
    9:  9,              # object          -> OBJECT_DEBRIS
    10: 0,              # asphalt         -> ASPHALT_CONCRETE
    12: 8,              # building        -> STRUCTURE
    15: 9,              # log             -> OBJECT_DEBRIS
    17: 5,              # person          -> PERSON
    18: 8,              # fence           -> STRUCTURE
    19: 3,              # bush            -> BUSH
    23: 0,              # concrete        -> ASPHALT_CONCRETE
    27: 8,              # barrier         -> STRUCTURE
    29: 2,              # grass variant   -> GRASS
    30: 2,              # grass variant   -> GRASS
    31: 10,             # puddle          -> WATER
    32: 10,             # water variant   -> WATER
    33: 11,             # mud             -> MUD
    34: 9,              # rubble          -> OBJECT_DEBRIS
}

# 256-entry LUT so remapping is one fancy-index instead of a loop over 23 keys per scan.
# Unlisted raw ids fall to IGNORE_INDEX rather than to a real class: an unmapped id is a bug to
# surface in the ignore count, not something to silently call UNKNOWN drivable terrain.
RAW_TO_FINE_LUT = np.full(256, IGNORE_INDEX, dtype=np.uint8)
for _raw, _fine in RAW_TO_FINE.items():
    RAW_TO_FINE_LUT[_raw] = _fine

# ── FINE -> ADL-1 ────────────────────────────────────────────────────────────────────────────
# TREE is the one genuinely contested entry. drdo_map.h documents class 4 as "rocks / trees /
# vehicles", but a 4 m canopy over a drivable forest track raises the cell's h_max and can flag
# it lethal. Because this is a table, evaluate BOTH collapses with scripts/eval.py and keep the
# one with the better ADL-1 mIoU. Do not settle it from first principles.
FINE_TO_ADL1_LUT = np.array([
    ADL1_GROUND,            # 0  ASPHALT_CONCRETE
    ADL1_GRAVEL_DIRT,       # 1  DIRT
    ADL1_GRASS_LOW,         # 2  GRASS
    ADL1_VEGETATION_DENSE,  # 3  BUSH
    ADL1_OBSTACLE_HARD,     # 4  TREE            <- retunable, see above
    ADL1_OBSTACLE_HARD,     # 5  PERSON
    ADL1_OBSTACLE_HARD,     # 6  VEHICLE
    ADL1_OBSTACLE_HARD,     # 7  POLE
    ADL1_OBSTACLE_HARD,     # 8  STRUCTURE
    ADL1_OBSTACLE_HARD,     # 9  OBJECT_DEBRIS
    ADL1_WATER_MUD,         # 10 WATER
    ADL1_WATER_MUD,         # 11 MUD
], dtype=np.uint8)
assert FINE_TO_ADL1_LUT.shape == (NUM_FINE,)

# ── FINE -> object class (marker / visualisation layer only) ─────────────────────────────────
# The grid engine never sees these. They exist so a confirmed moving cluster can be named and
# coloured "person" or "vehicle" instead of today's uniform magenta box with an id and a speed.
# One vocabulary for object identity, shared by BOTH producers: the (untrained) network head via
# FINE_TO_OBJ_LUT below, and the geometric classifier in perception/classify.py which infers the
# same classes from cluster extent alone. Two vocabularies would drift, and the marker colours,
# the dashboard legend and the accuracy tables all key off these ids.
# WALL and STRUCTURE have no fine class that maps to them — they are geometry-only verdicts.
OBJ_NONE, OBJ_PERSON, OBJ_VEHICLE, OBJ_POLE, OBJ_WALL, OBJ_STRUCTURE = 0, 1, 2, 3, 4, 5
OBJ_NAMES = ("none", "person", "vehicle", "pole", "wall", "structure")
NUM_OBJ = len(OBJ_NAMES)

# RGB for RViz markers and the dashboard legend. Kept here so the live view and the offline
# report cannot disagree about what colour a pedestrian is.
OBJ_RGB = {
    OBJ_NONE:      (150, 150, 150),
    OBJ_PERSON:    (255,  64, 160),   # magenta — the legacy colour for "a mover", kept for people
    OBJ_VEHICLE:   ( 80, 150, 255),   # blue
    OBJ_POLE:      (255, 200,  40),   # amber
    OBJ_WALL:      (200,  90, 220),   # violet
    OBJ_STRUCTURE: (180, 180, 180),   # grey
}

FINE_TO_OBJ_LUT = np.zeros(NUM_FINE, dtype=np.uint8)
FINE_TO_OBJ_LUT[5] = OBJ_PERSON
FINE_TO_OBJ_LUT[6] = OBJ_VEHICLE
FINE_TO_OBJ_LUT[7] = OBJ_POLE

# Fine classes that carry an object identity, for the object-layer argmax.
OBJ_BEARING_FINE = np.flatnonzero(FINE_TO_OBJ_LUT != OBJ_NONE).astype(np.int64)

# Row k lists the fine classes that collapse into ADL-1 class k. Built from the LUT so it can
# never drift out of sync with it.
ADL1_GROUPS = [np.flatnonzero(FINE_TO_ADL1_LUT == k).astype(np.int64) for k in range(NUM_ADL1)]


def remap_raw_to_fine(raw: np.ndarray) -> np.ndarray:
    """RELLIS raw ids -> FINE ids. Caller must mask IGNORE_INDEX out of the loss."""
    return RAW_TO_FINE_LUT[np.asarray(raw, dtype=np.uint8)]


def fine_to_adl1(fine: np.ndarray) -> np.ndarray:
    """FINE ids -> ADL-1 ids. IGNORE_INDEX maps to UNKNOWN, never to SKY_DUST: a point the
    network could not name must still occupy its cell as uncertain terrain (traversability
    0.40), not be deleted from the map by the engine's label==7 discard."""
    fine = np.asarray(fine)
    out = np.full(fine.shape, ADL1_UNKNOWN, dtype=np.uint8)
    valid = fine < NUM_FINE
    out[valid] = FINE_TO_ADL1_LUT[fine[valid]]
    return out


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def collapse_probs(logits: np.ndarray, min_conf: float = 0.0,
                   obj_min_conf: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse (N, NUM_FINE) fine logits to ADL-1 labels, object classes and a confidence.

    Sums probability WITHIN each ADL-1 group before taking the argmax. This is the whole point
    of the function and it is a safety property, not an optimisation:

        the fine head splits obstacle evidence across person/vehicle/pole/fence/tree. Five
        classes at 0.15 carry 0.75 of the probability mass, but a single grass class at 0.20
        wins a naive per-class argmax. Mapping that argmax through the table yields GRASS_LOW,
        traversability 0.65 — a crowd of people rendered as drivable grass.

    The risk grows with how finely the obstacle group is subdivided, which is exactly what this
    taxonomy does, so the grouped sum is mandatory. tests/python/test_segmentation.py pins a
    case where this disagrees with argmax-then-map.

    Returns (adl1 uint8, obj uint8, conf float32). Points below `min_conf` become UNKNOWN rather
    than a confident wrong class; objects below `obj_min_conf` become OBJ_NONE.
    """
    logits = np.atleast_2d(np.asarray(logits, dtype=np.float32))
    if logits.shape[-1] != NUM_FINE:
        raise ValueError(f"expected {NUM_FINE} fine logits, got {logits.shape[-1]}")
    probs = _softmax(logits)

    group = np.stack([probs[:, idx].sum(axis=1) for idx in ADL1_GROUPS], axis=1)
    adl1 = group.argmax(axis=1).astype(np.uint8)
    conf = group.max(axis=1).astype(np.float32)
    if min_conf > 0.0:
        adl1[conf < min_conf] = ADL1_UNKNOWN

    obj_probs = probs[:, OBJ_BEARING_FINE]
    obj = FINE_TO_OBJ_LUT[OBJ_BEARING_FINE[obj_probs.argmax(axis=1)]]
    obj[obj_probs.max(axis=1) < obj_min_conf] = OBJ_NONE
    # An object class only means something where the cell is actually an obstacle.
    obj[adl1 != ADL1_OBSTACLE_HARD] = OBJ_NONE

    return adl1, obj, conf


def class_weights(counts: np.ndarray, eps: float = 1.02) -> np.ndarray:
    """RangeNet/SalsaNext inverse-log frequency weighting: w_c = 1 / log(eps + f_c).

    Pass raw per-FINE-class point counts. Void dominates RELLIS at ~447M points, so unweighted
    CE collapses onto terrain and never learns PERSON.
    """
    counts = np.asarray(counts, dtype=np.float64)
    freq = counts / max(counts.sum(), 1.0)
    return (1.0 / np.log(eps + freq)).astype(np.float32)

"""DRDO ID26053 — object classification from cluster geometry alone.

The problem statement asks the system to "identify and classify static obstacles (walls, poles)
and dynamic objects (pedestrians, other vehicles)". Until this module existed the pipeline did
detection only: a confirmed mover came out as an unnamed box with an id and a speed, drawn the
same magenta whether it was a walking person or a moving truck.

No network is involved and none is claimed. A ground vehicle's obstacles separate on extent
remarkably well, because the classes the statement names differ in physical size by more than
their within-class spread:

    person      ~0.5 x 0.5 m footprint, 1.5-2.0 m tall     -> tall and thin
    vehicle     ~4.5 x 2.0 m footprint, 1.4-2.5 m tall     -> wide, road-vehicle proportions
    pole        ~0.2 x 0.2 m footprint, > 2 m tall         -> extremely tall and thin
    wall/fence  many metres along one axis, thin on other  -> elongated
    structure   large in both axes                         -> everything else big

So this is a decision tree over (length, width, height, elongation), with every threshold named,
justified and tunable in one place. That honesty matters: it is a geometric classifier with
measurable accuracy, and it must never be described as learned semantics. What it cannot do is
distinguish objects that share a size envelope — a person and a narrow post of the same height,
a car and a large boulder. Those limits are measured, not hidden: scripts/map_dashboard.py
reports per-class accuracy against the scene's ground truth, binned by range.

Class ids and colours come from segmentation/taxonomy.py so the geometric verdict and the
(untrained) network head speak the same vocabulary.
"""
from typing import Optional, Tuple

import numpy as np

from ..segmentation.taxonomy import (
    OBJ_NAMES, OBJ_NONE, OBJ_PERSON, OBJ_POLE, OBJ_STRUCTURE, OBJ_VEHICLE, OBJ_WALL,
)

__all__ = ["classify_extent", "classify_box", "OBJ_NAMES", "ClassifierThresholds"]


class ClassifierThresholds:
    """Every number the decision tree uses, in one place.

    Defaults are sized for a ground vehicle in off-road terrain and are deliberately generous on
    the upper bounds: under-segmentation (a person merged with a bush) is far more common in a
    20 cm occupancy raster than over-segmentation, so a hard 0.6 m ceiling on a person would
    reject real people. The cost of being generous is confusion with small posts, which the
    dashboard's per-class table makes visible rather than hiding.
    """
    # person
    PERSON_MAX_FOOT = 1.2      # m, larger of the two horizontal extents
    PERSON_MIN_H = 0.9         # m — a crouching or partly-occluded person still clears this
    PERSON_MAX_H = 2.3         # m
    PERSON_MIN_ASPECT = 1.1    # height / footprint: people are taller than they are wide

    # vehicle
    VEHICLE_MIN_LEN = 1.8      # m — a compact car; below this it is more likely a person or post
    VEHICLE_MAX_LEN = 12.0     # m — above this it is a wall or a tree line, not a vehicle
    VEHICLE_MIN_WID = 1.1      # m — narrower than this is a motorbike or a person
    VEHICLE_MAX_WID = 3.0      # m — road vehicles are narrow. Without this ceiling a 5x4 m shed
                               # satisfies every other vehicle test and is reported as a vehicle.
    VEHICLE_MAX_H = 4.5        # m — trucks; taller means structure
    VEHICLE_MIN_H = 0.5        # m

    # pole / post / trunk
    POLE_MAX_FOOT = 0.6        # m
    POLE_MIN_H = 1.8           # m
    POLE_MIN_ASPECT = 3.0      # height / footprint

    # wall / fence / embankment
    WALL_MIN_LEN = 3.0         # m along the long axis
    WALL_MIN_ELONGATION = 3.0  # long / short horizontal extent


def classify_extent(length: float, width: float, height: float,
                    t: Optional[ClassifierThresholds] = None) -> Tuple[int, float]:
    """(length, width, height) in metres -> (object class id, confidence in 0..1).

    `length` and `width` are the horizontal extents of the cluster's axis-aligned footprint in
    either order; this function sorts them, so the caller need not.

    Confidence is a blunt, honest signal — how far inside its class box the measurement sits,
    not a probability. It exists so a marginal call can be shown differently rather than
    asserted, and so the dashboard can report accuracy at a confidence floor.
    """
    t = t or ClassifierThresholds
    lo, hi = (float(width), float(length)) if width <= length else (float(length), float(width))
    h = float(height)
    if not np.isfinite([lo, hi, h]).all() or hi <= 0.0:
        return OBJ_NONE, 0.0

    elong = hi / max(lo, 1e-3)
    aspect = h / max(hi, 1e-3)

    # 1. Pole first. It is the tightest box, and a pole also satisfies the person test on
    #    footprint, so testing person first would swallow every post and tree trunk.
    if hi <= t.POLE_MAX_FOOT and h >= t.POLE_MIN_H and aspect >= t.POLE_MIN_ASPECT:
        return OBJ_POLE, _conf(aspect, t.POLE_MIN_ASPECT, 8.0)

    # 2. Wall / fence / tree line: elongated along one axis. Checked before vehicle because a
    #    12 m wall section otherwise lands inside the vehicle length range.
    if hi >= t.WALL_MIN_LEN and elong >= t.WALL_MIN_ELONGATION:
        return OBJ_WALL, _conf(elong, t.WALL_MIN_ELONGATION, 10.0)

    # 3. Person: small footprint, upright.
    if hi <= t.PERSON_MAX_FOOT and t.PERSON_MIN_H <= h <= t.PERSON_MAX_H and aspect >= t.PERSON_MIN_ASPECT:
        return OBJ_PERSON, _conf(aspect, t.PERSON_MIN_ASPECT, 4.0)

    # 4. Vehicle: road-vehicle proportions.
    if (t.VEHICLE_MIN_LEN <= hi <= t.VEHICLE_MAX_LEN
            and t.VEHICLE_MIN_WID <= lo <= t.VEHICLE_MAX_WID
            and t.VEHICLE_MIN_H <= h <= t.VEHICLE_MAX_H):
        return OBJ_VEHICLE, _conf(min(hi / t.VEHICLE_MIN_LEN, lo / t.VEHICLE_MIN_WID), 1.0, 2.5)

    # 5. Anything else with real bulk is a structure; anything tiny is not worth naming.
    if hi >= t.WALL_MIN_LEN or h >= t.VEHICLE_MAX_H:
        return OBJ_STRUCTURE, 0.4
    return OBJ_NONE, 0.0


def _conf(value: float, at_min: float, at_full: float) -> float:
    """Linear ramp from 0.5 at the class boundary to 1.0 well inside it.

    Never returns 0 for an accepted class: the tree already decided, and a 0 would read as "no
    object" downstream. Floored at 0.5 so "accepted but marginal" stays distinguishable.
    """
    if at_full <= at_min:
        return 1.0
    x = (float(value) - at_min) / (at_full - at_min)
    return float(0.5 + 0.5 * min(max(x, 0.0), 1.0))


def classify_box(min_bound, max_bound, t: Optional[ClassifierThresholds] = None) -> Tuple[int, float]:
    """Convenience for an axis-aligned bounding box: (3,) min and max corners -> (class, conf)."""
    mn = np.asarray(min_bound, dtype=float).ravel()
    mx = np.asarray(max_bound, dtype=float).ravel()
    if mn.size < 3 or mx.size < 3:
        raise ValueError("min_bound and max_bound must each have 3 components")
    ext = mx[:3] - mn[:3]
    return classify_extent(ext[0], ext[1], ext[2], t)

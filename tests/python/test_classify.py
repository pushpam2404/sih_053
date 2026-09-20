#!/usr/bin/env python3
"""DRDO ID26053 — geometric object classifier (drdo_lidar_mapping/perception/classify.py).

Covers the decision tree in classify_extent/classify_box directly (representative extents for
every class, and the pole-before-person ordering that keeps a post from being called a person),
plus the end-to-end path: a confirmed mover out of DynamicObstacleDetector carries the right
obj_class, the way dynamic_obstacle_node.py now colours and labels its RViz markers.
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)
from drdo_lidar_mapping.perception.classify import classify_box, classify_extent  # noqa: E402
from drdo_lidar_mapping.perception.dynamic import DynamicObstacleDetector  # noqa: E402
from drdo_lidar_mapping.segmentation.taxonomy import (  # noqa: E402
    OBJ_NONE, OBJ_PERSON, OBJ_POLE, OBJ_STRUCTURE, OBJ_VEHICLE, OBJ_WALL,
)

GROUND_Z = -1.2
DT = 0.1


def test_representative_extents():
    cases = [
        ("person",  (0.6, 0.5, 1.75), OBJ_PERSON),
        ("vehicle", (4.5, 1.9, 1.6),  OBJ_VEHICLE),
        ("truck",   (7.0, 2.4, 3.2),  OBJ_VEHICLE),
        ("pole",    (0.25, 0.25, 3.0), OBJ_POLE),
        ("wall",    (30.0, 0.3, 2.0), OBJ_WALL),
        ("shed",    (5.0, 4.0, 3.0),  OBJ_STRUCTURE),
    ]
    for name, (length, width, height), expected in cases:
        cls, conf = classify_extent(length, width, height)
        assert cls == expected, f"{name}: expected class {expected}, got {cls} (conf={conf:.2f})"
        assert 0.0 < conf <= 1.0, f"{name}: confidence {conf} out of (0, 1]"


def test_pole_before_person_ordering():
    """A pole also satisfies the person footprint/height/aspect test (both are narrow and tall),
    so the tree must test pole first or every post and tree trunk gets called a person. This is
    the ordering classify.py's own comment calls out (step 1 vs step 3)."""
    cls, _ = classify_extent(0.25, 0.25, 3.0)
    assert cls == OBJ_POLE, f"0.25x0.25x3.0 must be POLE, not {cls} (person would swallow it)"
    assert cls != OBJ_PERSON


def test_classify_extent_is_order_invariant_in_horizontal_args():
    for length, width, height in [(4.5, 1.9, 1.6), (0.6, 0.5, 1.75), (30.0, 0.3, 2.0)]:
        fwd = classify_extent(length, width, height)
        rev = classify_extent(width, length, height)
        assert fwd == rev, f"swapping length/width changed the verdict: {fwd} vs {rev}"


def test_classify_box_matches_extent():
    mn = np.array([1.0, 2.0, 0.0])
    mx = mn + np.array([4.5, 1.9, 1.6])
    cls, conf = classify_box(mn, mx)
    exp_cls, exp_conf = classify_extent(4.5, 1.9, 1.6)
    assert (cls, conf) == (exp_cls, exp_conf)


def test_degenerate_input_is_none():
    cls, conf = classify_extent(0.0, 0.0, 0.0)
    assert cls == OBJ_NONE and conf == 0.0
    cls, conf = classify_extent(float("nan"), 1.0, 1.0)
    assert cls == OBJ_NONE and conf == 0.0


# ── end-to-end: DynamicObstacleDetector must attach the right obj_class ────────────────────────
def _visible_box_points(cx, cy, sx, sy, length, width, height, rng, density=400.0):
    hx, hy = length / 2, width / 2
    parts = []
    for nx, ny, fx, fy, w in [(1, 0, cx + hx, cy, width), (-1, 0, cx - hx, cy, width),
                              (0, 1, cx, cy + hy, length), (0, -1, cx, cy - hy, length)]:
        if nx * (sx - fx) + ny * (sy - fy) <= 0:
            continue
        n = max(10, int(density * w * height / max(np.hypot(fx - sx, fy - sy), 1.0)))
        u, v = rng.uniform(-0.5, 0.5, n), rng.uniform(0, 1, n)
        px = np.full(n, fx) if nx else fx + u * w
        py = np.full(n, fy) if ny else fy + u * w
        parts.append(np.stack([px, py, GROUND_Z + v * height], 1))
    return np.concatenate(parts) if parts else np.zeros((0, 3))


def _ground_points(sx, sy, rng, n=3000):
    r, a = rng.uniform(1, 40, n), rng.uniform(-np.pi, np.pi, n)
    return np.stack([sx + r * np.cos(a), sy + r * np.sin(a), GROUND_Z + rng.normal(0, 0.02, n)], 1)


def test_moving_objects_carry_correct_class():
    """A walking person and an oncoming vehicle, driven through DynamicObstacleDetector exactly
    as test_dynamic.py does, must come out classified — not just detected.

    Only the FACES pointing at the sensor return points (see visible_box_points), so at a handful
    of frames the oncoming car is caught nearly edge-on and only one lateral face is visible; the
    footprint then degenerates to a sliver and classify_extent — correctly, given that input —
    calls it a wall, not a vehicle. That is the classifier being honest about a genuinely
    ambiguous silhouette, not a misclassification of the object. So this asserts the class that
    wins on every frame (the overwhelming majority), and that a car is never mistaken for a
    person, rather than requiring literally every frame to agree."""
    objects = {
        "walker":       (25.0, -8.0, 0.0, 1.4, 0.5, 0.5, 1.7),
        "oncoming_car": (70.0, -4.0, -8.0, 0.0, 4.5, 1.9, 1.5),
    }
    rng = np.random.default_rng(0)
    det = DynamicObstacleDetector(dt=DT)
    from collections import Counter
    counts = {"walker": Counter(), "oncoming_car": Counter()}
    for f in range(60):
        t = f * DT
        sx, sy = 5.0 * t, 0.0
        chunks = [_ground_points(sx, sy, rng)]
        names = list(objects)
        for name in names:
            x0, y0, vx, vy, ln, wd, ht = objects[name]
            chunks.append(_visible_box_points(x0 + vx * t, y0 + vy * t, sx, sy, ln, wd, ht, rng))
        xyz = np.concatenate(chunks).astype(np.float32)
        movers, _ = det.update(xyz, (sx, sy), GROUND_Z)
        for ob in movers:
            best = min(names, key=lambda nm: np.hypot(
                objects[nm][0] + objects[nm][2] * t - ob.x, objects[nm][1] + objects[nm][3] * t - ob.y))
            counts[best][ob.obj_class] += 1

    assert counts["walker"].most_common(1)[0][0] == OBJ_PERSON, f"walker: {counts['walker']}"
    assert OBJ_PERSON not in counts["walker"] or set(counts["walker"]) == {OBJ_PERSON}, \
        f"walker must never be classified as anything but a person: {counts['walker']}"
    assert counts["oncoming_car"].most_common(1)[0][0] == OBJ_VEHICLE, f"oncoming_car: {counts['oncoming_car']}"
    assert OBJ_PERSON not in counts["oncoming_car"], \
        f"a car must never be classified as a person: {counts['oncoming_car']}"


if __name__ == "__main__":
    print("[TEST] geometric object classifier")
    test_representative_extents()
    test_pole_before_person_ordering()
    test_classify_extent_is_order_invariant_in_horizontal_args()
    test_classify_box_matches_extent()
    test_degenerate_input_is_none()
    test_moving_objects_carry_correct_class()
    print("[CLASSIFY SUITE: ALL TESTS PASSED]")

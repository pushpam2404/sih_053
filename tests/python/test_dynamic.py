#!/usr/bin/env python3
"""DRDO ID26053 — moving-object detector (drdo_lidar_mapping/perception/dynamic.py).

Scenes are sampled the way a LiDAR sees them: only the faces of a box that point toward the sensor
return points, so a static object's visible footprint changes as the vehicle drives past it. That
viewpoint change is exactly what made the plain speed-threshold tracker report parked cars and walls
as moving.
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)
from drdo_lidar_mapping.perception.dynamic import (DynamicObstacleDetector, cloud_xyz,  # noqa: E402
                                                    filter_cloud_bytes)

GROUND_Z = -1.2
DT = 0.1


def visible_box_points(cx, cy, sx, sy, length, width, height, rng, density=400.0):
    """Points on the vertical faces of a box that face the sensor at (sx, sy)."""
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


def ground_points(sx, sy, rng, n=3000):
    r, a = rng.uniform(1, 40, n), rng.uniform(-np.pi, np.pi, n)
    return np.stack([sx + r * np.cos(a), sy + r * np.sin(a), GROUND_Z + rng.normal(0, 0.02, n)], 1)


def drive(objects, frames=60, speed=5.0, seed=0):
    """objects: name -> (x0, y0, vx, vy, length, width, height). Vehicle drives along +x from x=0."""
    rng = np.random.default_rng(seed)
    det = DynamicObstacleDetector(dt=DT)
    reported = {k: 0 for k in objects}
    for f in range(frames):
        t = f * DT
        sx, sy = speed * t, 0.0
        chunks, owners = [ground_points(sx, sy, rng)], [np.full(3000, -1)]
        names = list(objects)
        for i, name in enumerate(names):
            x0, y0, vx, vy, ln, wd, ht = objects[name]
            pts = visible_box_points(x0 + vx * t, y0 + vy * t, sx, sy, ln, wd, ht, rng)
            chunks.append(pts)
            owners.append(np.full(len(pts), i))
        xyz, owner = np.concatenate(chunks).astype(np.float32), np.concatenate(owners)
        movers, mask = det.update(xyz, (sx, sy), GROUND_Z)
        for ob in movers:
            best = min(range(len(names)), key=lambda k: np.hypot(
                objects[names[k]][0] + objects[names[k]][2] * t - ob.x, objects[names[k]][1] + objects[names[k]][3] * t - ob.y))
            reported[names[best]] += 1
        assert not (mask & (owner == -1)).any(), "ground points must never be masked as moving"
    return reported


def test_movers_confirmed_and_static_objects_not():
    objects = {
        "walker":        (25.0, -8.0, 0.0, 1.4, 0.5, 0.5, 1.7),
        "oncoming_car":  (70.0, -4.0, -8.0, 0.0, 4.5, 1.9, 1.5),
        "parked_car":    (15.0, -5.0, 0.0, 0.0, 4.5, 1.9, 1.5),
        "standing":      (20.0, 4.0, 0.0, 0.0, 0.5, 0.5, 1.7),
        "short_wall":    (18.0, 7.0, 0.0, 0.0, 6.0, 0.3, 2.0),
        "tree_trunks":   (12.0, -12.0, 0.0, 0.0, 2.5, 2.5, 2.4),
    }
    rep = drive(objects)
    print(f"  reports per object over 60 frames: {rep}")
    assert rep["walker"] >= 20, "walking person must be confirmed for most of the drive"
    assert rep["oncoming_car"] >= 10, "oncoming vehicle must be confirmed"
    for static in ("parked_car", "standing", "short_wall", "tree_trunks"):
        assert rep[static] == 0, f"{static} reported as moving while the vehicle drove past it"


def test_long_structures_are_never_tracked():
    rep = drive({"long_wall": (30.0, 6.0, 0.0, 0.0, 25.0, 0.3, 2.0)}, frames=40)
    assert rep["long_wall"] == 0


def test_cloud_helpers_round_trip():
    n, step = 5, 16                                   # x y z intensity, 4 x float32
    pts = np.arange(n * 4, dtype="<f4").reshape(n, 4)
    data = pts.tobytes()
    xyz = cloud_xyz(data, step, (0, 4, 8), n)
    assert np.array_equal(xyz, pts[:, :3])
    keep = np.array([True, False, True, False, True])
    kept = np.frombuffer(filter_cloud_bytes(data, step, n, keep), "<f4").reshape(-1, 4)
    assert np.array_equal(kept, pts[keep]), "filtering must keep every field of the kept points"


if __name__ == "__main__":
    print("[TEST] moving-object detector")
    test_movers_confirmed_and_static_objects_not()
    test_long_structures_are_never_tracked()
    test_cloud_helpers_round_trip()
    print("[DYNAMIC OBSTACLE SUITE: ALL TESTS PASSED]")

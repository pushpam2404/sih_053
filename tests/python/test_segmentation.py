#!/usr/bin/env python3
"""DRDO ID26053 — range-image projection + label taxonomy.

Scenes are built directly on the 64x1024 pixel lattice (build_direction_table gives the exact
direction of every pixel centre), so the projection identities can be asserted EXACTLY rather
than to a tolerance: if a point sits on a pixel centre it must land in that pixel, full stop.

Two of these tests pin safety properties rather than correctness properties, and neither is
allowed to be relaxed for convenience:

  * the z-buffer — a far return must never overwrite a near one,
  * the grouped collapse — obstacle evidence split across five fine classes must not be beaten
    by a single larger grass class.
"""
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, REPO_ROOT)
from drdo_lidar_mapping.segmentation.projection import (DEFAULT_H, DEFAULT_W,  # noqa: E402
                                                        build_direction_table, estimate_fov,
                                                        knn_postprocess, project, unproject)
from drdo_lidar_mapping.segmentation.taxonomy import (ADL1_GRASS_LOW,  # noqa: E402
                                                      ADL1_OBSTACLE_HARD, FINE_TO_ADL1_LUT,
                                                      IGNORE_INDEX, NUM_FINE, collapse_probs)

H, W = DEFAULT_H, DEFAULT_W
FOV_UP, FOV_DOWN = 22.5, -22.5

FINE_GRASS, FINE_TREE, FINE_PERSON, FINE_VEHICLE, FINE_POLE, FINE_STRUCTURE = 2, 4, 5, 6, 7, 8


def lattice_cloud(ranges=None, seed=0):
    """One point per pixel, exactly on the pixel's centre direction -> (N,4), rows in pixel order."""
    d = build_direction_table(H, W, FOV_UP, FOV_DOWN)
    if ranges is None:
        rng = np.random.default_rng(seed)
        ranges = rng.uniform(3.0, 60.0, (H, W)).astype(np.float32)
    xyz = (d * ranges[..., None]).reshape(-1, 3).astype(np.float32)
    inten = np.arange(H * W, dtype=np.float32)
    return np.concatenate([xyz, inten[:, None]], axis=1), ranges


def test_projection_round_trip():
    """Every lattice point owns a distinct pixel and unproject() gives the cloud back exactly."""
    points, ranges = lattice_cloud(seed=1)
    p = project(points, H=H, W=W, fov_up_deg=FOV_UP, fov_down_deg=FOV_DOWN)

    assert p["mask"].all(), f"{(~p['mask']).sum()} pixels empty on a full lattice"
    idx = p["idx"]
    assert (idx >= 0).all()
    assert np.array_equal(np.sort(idx.ravel()), np.arange(H * W)), "two points shared a pixel"
    # Row-major pixel order is the point order, i.e. each point landed in its OWN pixel.
    assert np.array_equal(idx, np.arange(H * W, dtype=np.int32).reshape(H, W))

    assert np.allclose(p["range"], ranges, rtol=0, atol=1e-3)
    assert np.array_equal(unproject(p["idx"], p["idx"], H * W, fill=-1),
                          np.arange(H * W, dtype=np.int32))
    back = unproject(p["xyz"], p["idx"], H * W, fill=np.float32(0))
    assert np.array_equal(back, points[:, :3]), "xyz did not survive the round trip bit-exactly"
    assert np.array_equal(unproject(p["intensity"], p["idx"], H * W), points[:, 3])
    print(f"  round trip: {H * W} points -> {int(p['mask'].sum())} distinct pixels, xyz exact")


def test_z_buffer_keeps_the_near_return():
    """Two returns on one ray: the NEAR one owns the pixel, whatever order they arrive in."""
    d = build_direction_table(H, W, FOV_UP, FOV_DOWN)
    ray = d[32, 512]
    near, far = 4.0, 31.0
    for order, (r0, r1) in (("far first", (far, near)), ("near first", (near, far))):
        pts = np.stack([ray * r0, ray * r1]).astype(np.float32)
        pts = np.concatenate([pts, np.zeros((2, 1), np.float32)], axis=1)
        labels = np.array([FINE_GRASS, FINE_PERSON], dtype=np.uint8)
        p = project(pts, labels, H=H, W=W, fov_up_deg=FOV_UP, fov_down_deg=FOV_DOWN)

        assert int(p["mask"].sum()) == 1, "two points on one ray must occupy one pixel"
        v, u = 32, 512
        winner = int(p["idx"][v, u])
        assert abs(float(p["range"][v, u]) - near) < 1e-4, \
            f"{order}: pixel holds range {p['range'][v, u]:.3f}, expected the near return {near}"
        assert abs(float(np.linalg.norm(p["xyz"][v, u])) - near) < 1e-3
        assert labels[winner] == (FINE_GRASS if r0 == near else FINE_PERSON)
    print(f"  z-buffer: near return ({near} m) wins over ({far} m) in both input orders")


def occlusion_scene(seed=5):
    """A 20 m wall of GRASS with a PERSON picket fence 5 m in front of every 3rd column.

    Returns (points, truth, occluded_mask). The occluded grass points are the ones this whole
    module exists for: they project into a pixel a person already owns.

    The wall carries 5 cm of radial range noise, so the kNN neighbours are NOT at an identical
    range and the 1.0 m cutoff has to do real work instead of matching exact ties.
    """
    rng = np.random.default_rng(seed)
    d = build_direction_table(H, W, FOV_UP, FOV_DOWN)
    rows, cols = np.arange(18, 46), np.arange(100, 300)
    rr, cc = np.meshgrid(rows, cols, indexing="ij")
    wall_r = 20.0 + rng.normal(0.0, 0.05, rr.shape)
    wall = (d[rr, cc] * wall_r[..., None]).reshape(-1, 3)

    occ_cols = cols[::3]
    orr, occ = np.meshgrid(rows, occ_cols, indexing="ij")
    picket_r = 5.0 + rng.normal(0.0, 0.05, orr.shape)
    picket = (d[orr, occ] * picket_r[..., None]).reshape(-1, 3)

    xyz = np.concatenate([wall, picket]).astype(np.float32)
    truth = np.concatenate([np.full(len(wall), FINE_GRASS, np.uint8),
                            np.full(len(picket), FINE_PERSON, np.uint8)])
    occluded = np.zeros(len(xyz), bool)
    occluded[: len(wall)] = np.isin(cc.ravel(), occ_cols)
    pts = np.concatenate([xyz, np.zeros((len(xyz), 1), np.float32)], axis=1)
    return pts, truth, occluded


def test_knn_recovers_occluded_points():
    """kNN must relabel the grass hidden behind the pickets as grass, not as PERSON."""
    pts, truth, occluded = occlusion_scene()
    p = project(pts, truth, H=H, W=W, fov_up_deg=FOV_UP, fov_down_deg=FOV_DOWN)

    # Baseline: plain unprojection. The z-buffer losers get `fill` and nothing else, which is
    # the honest statement of the problem — 25% of this cloud has no label at all.
    raw_fill = unproject(p["label"], p["idx"], len(pts), fill=IGNORE_INDEX)
    assert (raw_fill[occluded] == IGNORE_INDEX).all(), \
        "unproject() must leave the z-buffer losers at `fill`, not guess for them"

    knn = knn_postprocess(p["label"], p["range"], pts, H=H, W=W,
                          fov_up_deg=FOV_UP, fov_down_deg=FOV_DOWN)
    n_occ = int(occluded.sum())
    assert n_occ > 1000, f"scene must actually occlude a meaningful count, got {n_occ}"

    raw_acc = float((raw_fill[occluded] == truth[occluded]).mean())
    knn_acc = float((knn[occluded] == truth[occluded]).mean())
    all_acc = float((knn == truth).mean())
    vis_acc = float((knn[~occluded] == truth[~occluded]).mean())
    print(f"  occluded points: {n_occ} ({100 * n_occ / len(pts):.1f}% of the cloud)")
    print(f"  occluded accuracy  raw unproject {raw_acc:.3f} -> kNN {knn_acc:.3f}")
    print(f"  visible accuracy {vis_acc:.3f} | whole-cloud accuracy {all_acc:.3f}")

    assert knn_acc > raw_acc + 0.5, "kNN must recover most of what unprojection loses"
    assert knn_acc > 0.95, f"occluded-point accuracy only {knn_acc:.3f}"
    assert vis_acc > 0.98, "kNN must not destroy the labels of points that won their pixel"
    # The failure mode the whole pass exists to stop: grass wearing the occluder's PERSON label,
    # which in the 2.5D grid is an isolated lethal cell in the middle of a drivable field.
    assert int((knn[occluded] == FINE_PERSON).sum()) == 0, \
        "occluded grass inherited the occluder's obstacle label"

    # The numpy fallback (torchless ROS host) must agree with the torch/unfold path exactly.
    knn_np = knn_postprocess(p["label"], p["range"], pts, H=H, W=W, fov_up_deg=FOV_UP,
                             fov_down_deg=FOV_DOWN, use_torch=False)
    assert np.array_equal(knn, knn_np), "torch and numpy kNN paths disagree"


def test_direction_table_reconstructs_xyz():
    """range * direction must rebuild xyz to within half a pixel of angular resolution."""
    rng = np.random.default_rng(7)
    n = 60000
    r = rng.uniform(2.0, 80.0, n)
    pitch = np.deg2rad(rng.uniform(FOV_DOWN, FOV_UP, n))
    yaw = rng.uniform(-np.pi, np.pi, n)
    xyz = np.stack([r * np.cos(pitch) * np.cos(yaw),
                    -r * np.cos(pitch) * np.sin(yaw),
                    r * np.sin(pitch)], axis=1).astype(np.float32)
    pts = np.concatenate([xyz, np.zeros((n, 1), np.float32)], axis=1)

    p = project(pts, H=H, W=W, fov_up_deg=FOV_UP, fov_down_deg=FOV_DOWN)
    d = build_direction_table(H, W, FOV_UP, FOV_DOWN)
    m = p["mask"]

    assert np.allclose(np.linalg.norm(d, axis=2), 1.0, atol=1e-6), "directions must be unit"

    rebuilt = p["range"][..., None] * d
    err = np.linalg.norm(rebuilt[m] - p["xyz"][m], axis=1)
    # A point may sit anywhere inside its pixel, so the worst honest error is the half-diagonal
    # of a pixel's angular footprint times the range: 0.35 deg azimuth by 0.35 deg elevation.
    half_diag = 0.5 * np.hypot(np.deg2rad(360.0 / W), np.deg2rad((FOV_UP - FOV_DOWN) / H))
    bound = p["range"][m] * half_diag * 1.001
    assert (err <= bound).all(), \
        f"worst reconstruction error {np.max(err / np.maximum(bound, 1e-9)):.3f}x the half-pixel bound"

    # And on the lattice itself the reconstruction is exact to float32.
    pts_lat, ranges = lattice_cloud(seed=3)
    pl = project(pts_lat, H=H, W=W, fov_up_deg=FOV_UP, fov_down_deg=FOV_DOWN)
    exact = np.abs(pl["range"][..., None] * d - pl["xyz"]).max()
    assert exact < 1e-3, f"lattice reconstruction off by {exact:.2e} m"
    print(f"  reconstruction: max {err.max():.4f} m, half-pixel bound {bound.max():.4f} m "
          f"at 80 m; lattice exact to {exact:.2e} m")


def test_estimate_fov_recovers_the_sensor_envelope():
    """The FoV estimator must land on the beams the data actually contains, not the datasheet."""
    true_up, true_down = 16.0, -18.0
    d = build_direction_table(H, W, true_up, true_down)
    rng = np.random.default_rng(11)
    scans = [(d * rng.uniform(4.0, 50.0, (H, W))[..., None]).reshape(-1, 3) for _ in range(3)]
    up, down = estimate_fov(scans)
    # Percentiles sit at the outermost BEAM centres, half a row inside the envelope.
    half_row = 0.5 * (true_up - true_down) / H
    assert abs(up - (true_up - half_row)) < 0.2, f"fov_up {up:.2f}, expected ~{true_up - half_row:.2f}"
    assert abs(down - (true_down + half_row)) < 0.2, f"fov_down {down:.2f}"
    print(f"  estimate_fov: ({up:+.2f}, {down:+.2f}) deg from a {true_up:+.1f}/{true_down:+.1f} sensor")


def test_grouped_collapse_beats_argmax_on_split_obstacle_evidence():
    """SAFETY: a crowd of people must never collapse to drivable grass.

    The fine head splits obstacle evidence over five classes. Here TREE/PERSON/VEHICLE/POLE/
    STRUCTURE hold 0.15 each — 0.75 of the mass — while GRASS holds a single larger 0.20 and
    therefore wins a per-class argmax. Mapping that argmax through FINE_TO_ADL1_LUT yields
    GRASS_LOW, traversability 0.65: the planner is told to drive through the crowd.
    collapse_probs() sums within each ADL-1 group first and gets OBSTACLE_HARD.
    """
    probs = np.full(NUM_FINE, 0.0, dtype=np.float64)
    for c in (FINE_TREE, FINE_PERSON, FINE_VEHICLE, FINE_POLE, FINE_STRUCTURE):
        probs[c] = 0.15                                   # 5 x 0.15 = 0.75 obstacle mass
    probs[FINE_GRASS] = 0.20                              # single largest CLASS, not group
    rest = [c for c in range(NUM_FINE) if probs[c] == 0.0]
    probs[rest] = (1.0 - probs.sum()) / len(rest)
    assert abs(probs.sum() - 1.0) < 1e-12
    assert probs.argmax() == FINE_GRASS, "test setup: grass must be the per-class winner"

    logits = np.log(probs)[None, :]                       # softmax(log p) == p
    naive = FINE_TO_ADL1_LUT[int(logits.argmax())]
    adl1, obj, conf = collapse_probs(logits)

    assert naive == ADL1_GRASS_LOW, f"naive argmax gave {naive}, the test premise is broken"
    assert adl1[0] == ADL1_OBSTACLE_HARD, \
        f"collapse_probs returned {int(adl1[0])} — split obstacle evidence collapsed to terrain"
    assert conf[0] > 0.7, f"grouped confidence {conf[0]:.3f} should carry the whole 0.75+ mass"
    print(f"  split evidence: argmax->{int(naive)} GRASS_LOW vs collapse_probs->{int(adl1[0])} "
          f"OBSTACLE_HARD (conf {conf[0]:.3f}, obj {int(obj[0])})")


if __name__ == "__main__":
    print("[TEST] range-image projection + taxonomy collapse")
    test_projection_round_trip()
    test_z_buffer_keeps_the_near_return()
    test_knn_recovers_occluded_points()
    test_direction_table_reconstructs_xyz()
    test_estimate_fov_recovers_the_sensor_envelope()
    test_grouped_collapse_beats_argmax_on_split_obstacle_evidence()
    print("[SEGMENTATION SUITE: ALL TESTS PASSED]")

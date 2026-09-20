#!/usr/bin/env python3
"""DRDO ID26053 — full-pipeline latency benchmark (problem-statement deliverable: "Performance Metrics:
Evidence of low latency (high FPS)").

The previous version of this file (git history) inserted 500 points/frame one at a time through
drdo_map.insert_point() in a Python loop and never touched the moving-object detector, so it measured
Python call overhead on a scan ~130-260x lighter than a real Ouster OS1-64 (65,536-131,072 raw returns
per full 360 deg sweep for 1024/2048 azimuth columns x 64 beams) and reported "end-to-end" latency for a
pipeline stage nothing downstream actually runs. This version:

  * samples synthetic OS1-64-density scans (N_BEAMS x N_COLS returns per frame, see `sample_scan` below)
  * calls drdo_map.insert_points() — the batched, GIL-releasing entry point grid_map_node.cpp uses, not
    the per-point Python-loop binding
  * runs the real DynamicObstacleDetector (drdo_lidar_mapping/perception/dynamic.py) that the ROS node
    and the dashboard both drive, including its inline object classification (classify_extent), and
    drops confirmed movers before insertion — exactly what the live filter -> map pipeline does
  * drops the segmentation stage entirely: there is no trained model in this repo (models/* are a
    placeholder MLP, scripts/eval.py shows it at chance level) and there will not be one for this
    submission, so timing MinkUNetInference(allow_fallback=True) would print a numpy z-threshold ladder
    under the name "MinkUNet Inference" — dishonest evidence is worse than a labelled gap.

Usage:
    .venv/bin/python scripts/benchmark.py                       # 64 beams x 1024 cols, 200 frames
    .venv/bin/python scripts/benchmark.py --beams 64 --azimuth 2048   # 131,072 raw returns/frame
    .venv/bin/python scripts/benchmark.py --points 90000              # overrides --azimuth to hit N
    .venv/bin/python scripts/benchmark.py --frames 500 --out reports/benchmark.json
"""
import argparse
import json
import os
import platform
import sys
import time

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "lib"))
sys.path.insert(0, REPO_ROOT)
import drdo_map  # noqa: E402
from drdo_lidar_mapping.perception.dynamic import DynamicObstacleDetector  # noqa: E402
from drdo_lidar_mapping.perception.classify import classify_extent  # noqa: E402

HZ = 10.0                      # Ouster OS1-64 sweep rate this repo targets throughout (README, dashboard)
BUDGET_MS = 1000.0 / HZ        # 100 ms/frame at 10 Hz
MOUNT_H = 1.2
GROUND_Z = -MOUNT_H            # FAST-LIO world origin is the IMU start pose: ground sits near -mount_height
N_BEAMS_DEFAULT = 64
N_COLS_DEFAULT = 1024          # 64 x 1024 = 65,536 raw returns/sweep; 64 x 2048 = 131,072 (--azimuth 2048)
DEFAULT_PURGE_EVERY = 10       # matches grid_map_node.cpp's purge_every_n_frames default
DEFAULT_PURGE_MARGIN = 20.0    # matches grid_map_node.cpp's purge_margin_m default
GRID_RES = 0.20                # matches grid_map_node.cpp's default grid_resolution
GRID_SIZE_M = 100.0            # matches grid_map_node.cpp's default grid_size_m
RANGE_BANDS = [(0.0, 10.0), (10.0, 25.0), (25.0, 50.0), (50.0, 100.0)]


# ── Synthetic OS1-64 scan generator ──────────────────────────────────────────────────────────────
# Deliberately NOT imported from scripts/map_dashboard.py: that file is owned by a parallel agent and
# its sample_scan()/make_scene() are tightly coupled to one fixed demo scene (a specific set of trees,
# a puddle, three moving actors) built for a *visualization*, not for sweeping --points/--beams/--azimuth
# independently or for producing clean per-range-band subsets. The beam geometry below (elevation
# spread, ground-plane intersection via mount height / tan(depression)) is the same OS1-64 model the
# dashboard uses; ELEV comment there cites the same +-22.5 deg spec sheet figure.
def sample_scan(n_beams, n_cols, sx, sy, rng, canopy_hit_prob=0.85):
    """Returns (N, 4) float32 [x, y, z, label] in the world frame, world-frame point count close to
    n_beams * n_cols (a full raw OS1-64 sweep), not the ~30-50% of beams a flat-ground-only model would
    return. Downward-looking beams (below the horizon) hit a flat ground plane; upward/horizontal beams
    hit synthetic canopy/obstacle returns with probability canopy_hit_prob and otherwise return nothing
    (sky, label 7, discarded by insert_lidar_point) — this is a generic density model for load-testing
    the engine, not a semantically meaningful scene (no vehicle-following occlusion, no object identity)."""
    elev = np.deg2rad(np.linspace(-22.5, 22.5, n_beams))
    az = np.linspace(-np.pi, np.pi, n_cols, endpoint=False)
    EL, AZ = np.meshgrid(elev, az, indexing="ij")   # (n_beams, n_cols)

    down = EL < -0.01
    r = np.full(EL.shape, np.inf, np.float64)
    r[down] = MOUNT_H / np.tan(-EL[down])
    r[down] *= 1.0 + rng.normal(0, 0.003, int(down.sum()))

    up = ~down
    hit = rng.uniform(size=up.sum()) < canopy_hit_prob
    r_up = np.full(int(up.sum()), np.inf)
    r_up[hit] = rng.uniform(3.0, 100.0, int(hit.sum()))
    r[up] = r_up

    keep = np.isfinite(r) & (r <= 100.0) & (r >= 0.3)   # ADL-3 min_range self-return filter (grid_map_node.cpp)
    r, EL, AZ = r[keep], EL[keep], AZ[keep]

    gx = sx + r * np.cos(EL) * np.cos(AZ)
    gy = sy + r * np.cos(EL) * np.sin(AZ)
    z_ground = GROUND_Z + 0.03 * np.sin(0.35 * gx) * np.cos(0.25 * gy)
    z = np.where(EL < -0.01, z_ground, GROUND_Z + r * np.sin(EL))
    z += rng.normal(0, 0.01, z.shape)

    # Ground-band returns get semantic label 0 (GROUND); everything else is UNKNOWN(6) — geometry-only
    # pipeline, no claim of a segmentation network (see CLAUDE.md "Deep learning is deliberately parked").
    label = np.where(EL < -0.01, 0, 6).astype(np.float32)
    return np.stack([gx, gy, z, label], axis=1).astype(np.float32)


def stats_ms(values):
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return {"mean_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0, "n": 0}
    return {"mean_ms": float(a.mean()), "p95_ms": float(np.percentile(a, 95)),
            "max_ms": float(a.max()), "n": int(a.size)}


# ── Rasterisation approximation (Python/numpy port of grid_map_node.cpp's publish_grid pass) ──────
def approx_rasterize(cells, robot_x, robot_y, frame_ts, decay_window=50, decay_min_trav=0.2):
    """Vectorised approximation of the C++ node's per-publish max-cost rasterisation. NOT the real
    implementation: grid_map_node.cpp scans the full HASH_TABLE_SIZE slot array and does true footprint
    max-aggregation per output cell (a 50 cm cell can cover up to 3x3 output pixels at grid_res=0.20 m);
    this only takes the top-left corner of each cell's footprint, so it under-costs by skipping the
    footprint-fill loop and never touches empty hash slots. It exists to give an order-of-magnitude
    Python-side number for the publish stage, not a substitute for profiling the compiled node."""
    n = len(cells["ix"])
    gw = int(round(GRID_SIZE_M / GRID_RES))
    if n == 0:
        return np.full(gw * gw, -1, np.int16)
    ox = np.floor((robot_x - GRID_SIZE_M / 2.0) / GRID_RES) * GRID_RES
    oy = np.floor((robot_y - GRID_SIZE_M / 2.0) / GRID_RES) * GRID_RES
    cs = cells["cell_size"]
    wx = cells["ix"].astype(np.float64) * cs
    wy = cells["iy"].astype(np.float64) * cs
    px = np.floor((wx - ox) / GRID_RES).astype(np.int64)
    py = np.floor((wy - oy) / GRID_RES).astype(np.int64)
    inb = (px >= 0) & (py >= 0) & (px < gw) & (py < gw)

    stale = (frame_ts > cells["last_update_ts"]) & (frame_ts - cells["last_update_ts"] > decay_window)
    cost = np.full(n, -1.0, np.float32)
    free = cells["hit_count"] == 0
    obstacle = (cells["obstacle_flag"] == 1) | np.isin(cells["semantic_label"], [4, 5])
    decayed_unknown = stale & (cells["traversability"] < decay_min_trav) & ~free & ~obstacle
    conf = np.minimum(cells["hit_count"].astype(np.float32) / 10.0, 1.0)
    score = np.where(conf > 0, cells["traversability"] / np.maximum(conf, 1e-6), 0.0)
    normal_cost = np.round(99.0 * (1.0 - np.clip(score, 0.0, 1.0)))
    cost = np.where(free, 0, cost)
    cost = np.where(obstacle, 100, cost)
    cost = np.where((~free) & (~obstacle) & (~decayed_unknown), normal_cost, cost)
    cost = np.where(decayed_unknown, -1, cost)

    out = np.full(gw * gw, -1, np.int16)
    idx = py[inb] * gw + px[inb]
    np.maximum.at(out, idx, cost[inb].astype(np.int16))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=200, help="frames to run after warmup (default 200)")
    ap.add_argument("--beams", type=int, default=N_BEAMS_DEFAULT, help="LiDAR beam count (default 64, OS1-64)")
    ap.add_argument("--azimuth", type=int, default=N_COLS_DEFAULT,
                    help="azimuth columns/sweep (default 1024 -> 65,536 raw returns; 2048 -> 131,072)")
    ap.add_argument("--points", type=int, default=None,
                    help="override --azimuth so beams*azimuth ~= this many raw returns/frame")
    ap.add_argument("--speed", type=float, default=8.33, help="robot speed m/s along +x (default 30 km/h)")
    ap.add_argument("--purge-every", type=int, default=DEFAULT_PURGE_EVERY,
                    help="frames between purge_distant_cells calls (default 10, matches grid_map_node.cpp)")
    ap.add_argument("--purge-margin", type=float, default=DEFAULT_PURGE_MARGIN)
    ap.add_argument("--skip-rasterize", action="store_true", help="skip the Python rasterisation approximation")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=os.path.join(REPO_ROOT, "reports", "benchmark.json"))
    args = ap.parse_args()

    n_beams = args.beams
    n_cols = args.azimuth
    if args.points is not None:
        n_cols = max(1, args.points // max(n_beams, 1))
    raw_returns = n_beams * n_cols

    rng = np.random.default_rng(args.seed)
    drdo_map.reset_map()
    detector = DynamicObstacleDetector(dt=1.0 / HZ)

    frame_total, moving_t, classify_obj_t, insert_t, dirty_t, decay_t, purge_t, raster_t = ([] for _ in range(8))
    inserted_counts, dropped_by_dynamic = [], []

    N_WARMUP = 5
    N_FRAMES = args.frames

    for frame in range(1, N_WARMUP + N_FRAMES + 1):
        timed = frame > N_WARMUP
        robot_x, robot_y = args.speed * (frame - 1) / HZ, 0.0

        pts = sample_scan(n_beams, n_cols, robot_x, robot_y, rng)
        xyz = pts[:, :3]

        t0 = time.perf_counter()

        # Stage 1: moving-object detection (includes classify_extent — it is called inline per
        # confirmed track inside DynamicObstacleDetector.update(), not a separable pipeline stage;
        # see drdo_lidar_mapping/perception/dynamic.py line ~258).
        ta = time.perf_counter()
        movers, moving_mask = detector.update(xyz, (robot_x, robot_y), GROUND_Z)
        tb = time.perf_counter()
        if timed:
            moving_t.append((tb - ta) * 1000.0)
            dropped_by_dynamic.append(int(moving_mask.sum()))

        static_pts = pts[~moving_mask]

        # Stage 2: C++ engine batch insert (drdo_map.insert_points, GIL released in the C++ binding).
        ta = time.perf_counter()
        n_ins = drdo_map.insert_points(static_pts, robot_x, robot_y, frame, GROUND_Z)
        tb = time.perf_counter()
        if timed:
            insert_t.append((tb - ta) * 1000.0)
            inserted_counts.append(n_ins)

        # Stage 3: classify only cells touched since the last call.
        ta = time.perf_counter()
        drdo_map.classify_dirty(GROUND_Z)
        tb = time.perf_counter()
        if timed:
            dirty_t.append((tb - ta) * 1000.0)

        # Stage 4: temporal log-odds decay.
        ta = time.perf_counter()
        drdo_map.decay_kernel(frame)
        tb = time.perf_counter()
        if timed:
            decay_t.append((tb - ta) * 1000.0)

        # Stage 5: purge — runs every purge_every frames, like grid_map_node.cpp. Only every-Nth-frame
        # calls enter purge_t so p95/max reflect the actual spike, not a diluted average.
        if args.purge_every > 0 and frame % args.purge_every == 0:
            ta = time.perf_counter()
            drdo_map.purge_distant(robot_x, robot_y, args.purge_margin)
            tb = time.perf_counter()
            if timed:
                purge_t.append((tb - ta) * 1000.0)

        # Stage 6: rasterisation approximation (Python/numpy port; see approx_rasterize() docstring for
        # its accuracy caveats). Skippable because it is the least trustworthy number in this report.
        if not args.skip_rasterize:
            ta = time.perf_counter()
            cells = drdo_map.export_cells()
            approx_rasterize(cells, robot_x, robot_y, frame)
            tb = time.perf_counter()
            if timed:
                raster_t.append((tb - ta) * 1000.0)

        t1 = time.perf_counter()
        if timed:
            frame_total.append((t1 - t0) * 1000.0)

    # classify_extent microbenchmark: it already runs inline inside Stage 1 above (moving_t already
    # includes its cost for every confirmed track that frame), but that cost is usually zero calls/frame
    # in this synthetic scan (no moving objects are staged into it) since sample_scan() above has no
    # moving actors — it is a load-test scan, not a scene. Measure it standalone so the report is not
    # silently missing the one classification stage the problem statement asks about.
    class_reps = [(0.5, 0.5, 1.7), (4.5, 2.0, 1.6), (0.2, 0.2, 2.2), (8.0, 0.4, 1.9), (5.0, 5.0, 3.0)]
    N_CLASS = 20000
    t0 = time.perf_counter()
    for i in range(N_CLASS):
        L, W, H = class_reps[i % len(class_reps)]
        classify_extent(L, W, H)
    t1 = time.perf_counter()
    classify_us_per_call = (t1 - t0) * 1e6 / N_CLASS

    # Stratify insert cost by range band on a fresh map each time, so band N's timing is not warmed
    # (or slowed) by cells band N-1 already inserted.
    band_report = {}
    for lo, hi in RANGE_BANDS:
        drdo_map.reset_map()
        band_pts_frames = []
        for frame in range(1, 30 + 1):
            robot_x = args.speed * (frame - 1) / HZ
            pts = sample_scan(n_beams, n_cols, robot_x, 0.0, rng)
            r = np.hypot(pts[:, 0] - robot_x, pts[:, 1] - 0.0)
            band_pts_frames.append(pts[(r >= lo) & (r < hi)])
        times, counts = [], []
        for frame, bp in enumerate(band_pts_frames, start=1):
            robot_x = args.speed * (frame - 1) / HZ
            ta = time.perf_counter()
            n_ins = drdo_map.insert_points(bp, robot_x, 0.0, frame, GROUND_Z)
            tb = time.perf_counter()
            times.append((tb - ta) * 1000.0)
            counts.append(n_ins)
        mean_pts = float(np.mean([len(bp) for bp in band_pts_frames]))
        mean_ms = float(np.mean(times))
        band_report[f"{lo:.0f}-{hi:.0f}m"] = {
            "mean_points_per_frame": mean_pts,
            "mean_insert_ms": mean_ms,
            "us_per_point": (mean_ms * 1000.0 / mean_pts) if mean_pts > 0 else None,
            "cell_size_m": {(0.0, 10.0): 0.05, (10.0, 25.0): 0.10, (25.0, 50.0): 0.50, (50.0, 100.0): 0.50}[(lo, hi)],
        }
    drdo_map.reset_map()

    cpu = platform.processor() or platform.machine()
    try:
        import subprocess
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        pass

    mean_total = float(np.mean(frame_total)) if frame_total else 0.0
    hz_achievable = 1000.0 / mean_total if mean_total > 0 else 0.0
    purge_amortised = float(np.sum(purge_t) / N_FRAMES) if N_FRAMES else 0.0

    report = {
        "host_cpu": cpu,
        "host_platform": platform.platform(),
        "note": "Apple M4 laptop, single-threaded Python/pybind11 call path. No Jetson Orin numbers "
                "exist for this project (CLAUDE.md: CUDA kernels never compiled with nvcc, no ROS run "
                "on a vehicle) — do not extrapolate these numbers to Orin.",
        "scan_model": {
            "beams": n_beams, "azimuth_cols": n_cols, "raw_returns_per_sweep": raw_returns,
        },
        "frames_measured": N_FRAMES,
        "frames_warmup": N_WARMUP,
        "speed_mps": args.speed,
        "purge_every_n_frames": args.purge_every,
        "sensor_hz": HZ,
        "budget_ms": BUDGET_MS,
        "headline": {
            "mean_frame_ms": mean_total,
            "p95_frame_ms": float(np.percentile(frame_total, 95)) if frame_total else 0.0,
            "max_frame_ms": float(np.max(frame_total)) if frame_total else 0.0,
            "achievable_hz": hz_achievable,
            "clears_10hz_budget": bool(mean_total <= BUDGET_MS),
            "mean_inserted_points_per_frame": float(np.mean(inserted_counts)) if inserted_counts else 0.0,
            "mean_dropped_as_moving_per_frame": float(np.mean(dropped_by_dynamic)) if dropped_by_dynamic else 0.0,
        },
        "stages_ms": {
            "moving_object_detection": stats_ms(moving_t),
            "engine_insert_points": stats_ms(insert_t),
            "classify_dirty": stats_ms(dirty_t),
            "decay_kernel": stats_ms(decay_t),
            "purge_distant_actual_when_run": stats_ms(purge_t),
            "purge_distant_amortised_per_frame_ms": purge_amortised,
            "rasterize_approx_python": stats_ms(raster_t) if raster_t else "skipped (--skip-rasterize)",
        },
        "classify_extent_standalone": {
            "us_per_call": classify_us_per_call, "n_calls": N_CLASS,
            "note": "classify_extent() is called inline inside DynamicObstacleDetector.update() per "
                    "confirmed track, so its cost is already folded into moving_object_detection above "
                    "for any frame with confirmed movers; this synthetic load-test scan has none, so it "
                    "is also measured standalone here.",
        },
        "insert_cost_by_range_band": band_report,
    }

    print("=" * 88)
    print("DRDO ID26053 — full-pipeline latency benchmark")
    print("=" * 88)
    print(f"Host: {cpu} ({platform.platform()}) — single-threaded, Python/pybind11 call path")
    print("No Jetson Orin numbers exist for this project; do not extrapolate these to the target hardware.")
    print(f"Scan model: {n_beams} beams x {n_cols} az cols = {raw_returns} raw returns/sweep "
          f"(target OS1-64 range: 65,536-131,072)")
    print(f"Frames measured: {N_FRAMES} (+{N_WARMUP} warmup discarded), speed={args.speed:.2f} m/s, "
          f"purge every {args.purge_every} frames")
    print()
    print(f"{'stage':<32}{'mean ms':>10}{'p95 ms':>10}{'max ms':>10}")
    for name, s in [("moving-object detection", stats_ms(moving_t)),
                    ("engine insert_points", stats_ms(insert_t)),
                    ("classify_dirty", stats_ms(dirty_t)),
                    ("decay_kernel", stats_ms(decay_t)),
                    ("purge_distant (when run)", stats_ms(purge_t))]:
        print(f"{name:<32}{s['mean_ms']:>10.3f}{s['p95_ms']:>10.3f}{s['max_ms']:>10.3f}")
    print(f"{'purge_distant (amortised)':<32}{purge_amortised:>10.3f}{'':>10}{'':>10}")
    if raster_t:
        s = stats_ms(raster_t)
        print(f"{'rasterize (Python approx.)':<32}{s['mean_ms']:>10.3f}{s['p95_ms']:>10.3f}{s['max_ms']:>10.3f}")
    print("-" * 88)
    print(f"{'TOTAL / frame':<32}{mean_total:>10.3f}{report['headline']['p95_frame_ms']:>10.3f}"
          f"{report['headline']['max_frame_ms']:>10.3f}")
    print()
    print(f"Achievable rate: {hz_achievable:.1f} Hz  |  10 Hz budget ({BUDGET_MS:.0f} ms/frame): "
          f"{'CLEARED' if report['headline']['clears_10hz_budget'] else 'MISSED'}")
    print(f"classify_extent(): {classify_us_per_call:.2f} us/call (already inline in moving-object "
          f"detection when a track confirms; also measured standalone above)")
    print()
    print("Insert cost by range band (fresh map per band; shows foveation's cost structure):")
    print(f"{'band':<10}{'cell':>8}{'mean pts':>12}{'mean ms':>10}{'us/pt':>10}")
    for k, v in band_report.items():
        us = f"{v['us_per_point']:.3f}" if v['us_per_point'] is not None else "n/a"
        print(f"{k:<10}{v['cell_size_m']:>8.2f}{v['mean_points_per_frame']:>12.0f}{v['mean_insert_ms']:>10.3f}{us:>10}")
    if raster_t:
        print()
        print("NOTE: rasterize timing is a Python/numpy approximation of grid_map_node.cpp's publish pass "
              "(top-left-corner splat, not true footprint max-aggregation) — see approx_rasterize() "
              "docstring. Treat it as order-of-magnitude, not a substitute for profiling the compiled node.")

    out_path = args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

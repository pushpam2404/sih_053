#!/usr/bin/env python3
"""DRDO ID26053 — 2.5D foveated map dashboard (problem-statement deliverable: "Real-time Visualization").

Drives the real C++ grid engine (lib/drdo_map*.so) through a synthetic off-road scene sampled with
an Ouster OS1-64 beam pattern, runs the DBSCAN + SORT perception stage to split static obstacles from
moving objects, and writes ONE self-contained HTML file (no external assets) with:

  * colour-coded semantic / elevation / cost layers of the variable-resolution grid, near (+-15 m,
    cell outlines visible) and far (+-100 m) views, replayable over the drive
  * moving objects confirmed by drdo_lidar_mapping/perception/dynamic.py (the same module the ROS node runs)
  * memory in use versus uniform 5 cm 2.5D and 3D maps covering the same 100 m radius
  * per-stage latency (mean / p95) and dynamic-object detection by range

What this is NOT: the per-point labels are the simulator's ground truth, not the segmentation network
(scripts/eval.py shows the shipped checkpoint is at chance level), and the scene has no occlusion.
On the vehicle the live view is RViz on /map; this script is the offline, laptop-runnable demo.

Usage:
    python3 scripts/map_dashboard.py [--frames 240] [--speed 5.0] [--out reports/map_dashboard.html]
"""
import argparse
import base64
import json
import os
import struct
import sys
import time
import zlib

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "lib"))
sys.path.insert(0, REPO_ROOT)
import drdo_map  # noqa: E402
from drdo_lidar_mapping.perception.dynamic import DynamicObstacleDetector  # noqa: E402

HZ = 10.0
MOUNT_H = 1.2
GROUND_Z = -MOUNT_H                      # FAST-LIO-style world frame: origin at the sensor start pose
N_BEAMS, N_COLS = 64, 1024
ELEV = np.deg2rad(np.linspace(-22.5, 22.5, N_BEAMS))
ZOOM_AT, ZOOM_HALF = (7.0, 7.0), 2.0     # zoom window straddling the 10 m band edge (5 cm -> 10 cm)
MOVERS = ("pedestrian", "pedestrian2", "vehicle")
CLASS_NAMES = ["GROUND", "GRAVEL_DIRT", "GRASS_LOW", "VEGETATION", "OBSTACLE_HARD", "WATER_MUD", "UNKNOWN"]
TERRAIN_RGB = np.array([[22, 24, 28], [38, 72, 52], [76, 170, 84], [60, 160, 170],
                        [232, 170, 40], [60, 120, 225], [225, 40, 40]], np.uint8)
LABEL_RGB = np.array([[150, 140, 125], [120, 190, 90], [40, 110, 50]], np.uint8)
PALETTE = np.array([[196, 170, 120], [150, 140, 125], [120, 190, 90], [40, 110, 50],
                    [220, 50, 50], [60, 120, 220], [150, 150, 150]], np.uint8)


# ── Scene ────────────────────────────────────────────────────────────────────────────────────────
def terrain_label(x, y):
    lab = np.where(np.abs(y) < 3.0, 1, np.where(np.abs(y) < 25.0, 2, 0)).astype(np.float32)
    lab[((x - 42.0) / 5.0) ** 2 + ((y + 7.0) / 3.0) ** 2 < 1.0] = 5      # mud puddle
    lab[((x - 95.0) / 4.0) ** 2 + ((y - 9.0) / 6.0) ** 2 < 1.0] = 5
    return lab


def terrain_height(x, y):
    h = 0.03 * np.sin(0.35 * x) * np.cos(0.25 * y)
    h = h + np.where((y > 4.0) & (y < 4.4), 0.25, 0.0)                  # 25 cm curb along the track
    return GROUND_Z + h


def make_scene(rng):
    boxes = []   # (cx, cy, sx, sy, height, label, vx, vy, name)
    boxes.append([75.0, 9.0, 30.0, 0.3, 2.0, 4, 0.0, 0.0, "wall"])
    for i, (px, py) in enumerate([(20, 5.5), (35, -5.0), (55, 5.5), (80, -4.5), (110, 5.5), (130, -5.0)]):
        boxes.append([px, py, 0.3, 0.3, 3.0, 4, 0.0, 0.0, f"pole{i}"])
    for i in range(70):
        tx, ty = rng.uniform(-20, 190), rng.choice([-1, 1]) * rng.uniform(13, 45)
        boxes.append([tx, ty, 2.5, 2.5, rng.uniform(2.5, 6.0), 3, 0.0, 0.0, f"tree{i}"])
    boxes.append([30.0, -12.0, 0.5, 0.5, 1.7, 4, 0.0, 1.4, "pedestrian"])
    boxes.append([88.0, 9.0, 0.5, 0.5, 1.7, 4, 0.0, -1.2, "pedestrian2"])     # crosses the track
    boxes.append([170.0, -4.5, 4.5, 1.9, 1.6, 4, -8.0, 0.0, "vehicle"])
    boxes.append([62.0, -6.0, 4.5, 1.9, 1.5, 4, 0.0, 0.0, "parked_car"])      # must stay static
    boxes.append([48.0, 6.5, 0.5, 0.5, 1.7, 4, 0.0, 0.0, "standing_person"])  # must stay static
    return boxes


def box_at(b, t):
    return b[0] + b[6] * t, b[1] + b[7] * t


def sample_scan(boxes, t, sx, sy, rng):
    """Returns (N, 4) float32 [x, y, z, label] in the world frame. Ground from the OS1-64 beam pattern;
    object faces sampled at the sensor's angular density (26,500 / r^2 points per m^2)."""
    az = np.linspace(-np.pi, np.pi, N_COLS, endpoint=False)
    down = ELEV[ELEV < -0.01]
    r = (MOUNT_H / np.tan(-down))[:, None] * np.ones((1, N_COLS))
    r = r * (1.0 + rng.normal(0, 0.003, r.shape))
    gx, gy = sx + r * np.cos(az), sy + r * np.sin(az)
    keep = r <= 100.0
    gx, gy = gx[keep], gy[keep]
    ground = np.stack([gx, gy, terrain_height(gx, gy) + rng.normal(0, 0.01, gx.shape), terrain_label(gx, gy)], 1)
    parts = [ground]
    for b in boxes:
        cx, cy = box_at(b, t)
        hx, hy, hz = b[2] / 2, b[3] / 2, b[4]
        dx, dy = cx - sx, cy - sy
        dist = np.hypot(dx, dy)
        if dist > 100.0 or dist < 1.0:
            continue
        for nx, ny, fx, fy, w in [(1, 0, cx + hx, cy, b[3]), (-1, 0, cx - hx, cy, b[3]),
                                  (0, 1, cx, cy + hy, b[2]), (0, -1, cx, cy - hy, b[2])]:
            if nx * (sx - fx) + ny * (sy - fy) <= 0:
                continue                                                 # back face
            fd = max(np.hypot(fx - sx, fy - sy), 1.0)
            n = int(min(4000, 26500.0 * w * hz / fd ** 2))
            if n < 3:
                continue
            u, v = rng.uniform(-0.5, 0.5, n), rng.uniform(0.0, 1.0, n)
            px = fx + (0 if nx else u * w)
            py = fy + (0 if ny else u * w)
            px = np.full(n, px) if np.isscalar(px) else px
            py = np.full(n, py) if np.isscalar(py) else py
            pz = terrain_height(px, py) + v * hz
            el = np.arctan2(pz, np.hypot(px - sx, py - sy))
            m = np.abs(el) <= np.deg2rad(22.5)
            parts.append(np.stack([px[m], py[m], pz[m], np.full(m.sum(), b[5], np.float64)], 1))
    return np.concatenate(parts).astype(np.float32)


# ── Rasterisation + PNG (no PIL / matplotlib dependency) ─────────────────────────────────────────
def png_b64(rgba):
    h, w, _ = rgba.shape
    raw = b"".join(b"\x00" + rgba[y].tobytes() for y in range(h))
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(png).decode()


def layer_colours(cells, frame_ts):
    n = len(cells["ix"])
    hit = cells["hit_count"] > 0
    sem = np.zeros((n, 4), np.uint8)
    lab = np.minimum(cells["semantic_label"], 6)
    sem[:, :3] = PALETTE[lab]
    sem[:, 3] = 255
    sem[~hit] = [45, 52, 62, 255]                                        # carved free space, no returns
    ele = np.zeros((n, 4), np.uint8)
    rel = np.clip((cells["h_max"] - GROUND_Z) / 2.5, 0, 1)
    ele[:, 0] = (255 * np.clip(2 * rel, 0, 1)).astype(np.uint8)
    ele[:, 1] = (255 * (1 - np.abs(2 * rel - 1))).astype(np.uint8)
    ele[:, 2] = (255 * np.clip(1 - 2 * rel, 0, 1)).astype(np.uint8)
    ele[:, 3] = 255
    ele[~hit] = [45, 52, 62, 255]
    cost = np.zeros((n, 4), np.uint8)
    t = np.clip(cells["traversability"] / np.maximum(np.minimum(cells["hit_count"] / 10.0, 1.0), 1e-3), 0, 1)
    cost[:, 0] = (255 * (1 - t)).astype(np.uint8)
    cost[:, 1] = (200 * t).astype(np.uint8)
    cost[:, 2] = 40
    cost[:, 3] = 255
    cost[~hit] = [30, 90, 60, 255]
    lethal = (cells["obstacle_flag"] == 1) | np.isin(cells["semantic_label"], [4, 5]) & hit
    cost[lethal] = [255, 0, 60, 255]
    # Terrain: the colours grid_map_node publishes on /drdo/map_image (geometry first; see the node).
    terr = np.zeros((n, 4), np.uint8)
    terr[:, 3] = 255
    terr[:, :3] = TERRAIN_RGB[2]                                          # flat ground
    lab = cells["semantic_label"]
    for k in (1, 2, 3):
        terr[hit & (lab == k), :3] = LABEL_RGB[k - 1]
    terr[hit & (cells["h_min"] - GROUND_Z > 0.5), :3] = TERRAIN_RGB[3]    # overhang
    terr[hit & (cells["obstacle_flag"] == 2), :3] = TERRAIN_RGB[4]        # rough / uncertain
    terr[hit & (lab == 5) & (cells["obstacle_flag"] != 1), :3] = TERRAIN_RGB[5]
    terr[(cells["obstacle_flag"] == 1) | (hit & (lab == 4)), :3] = TERRAIN_RGB[6]
    terr[~hit, :3] = TERRAIN_RGB[1]
    seen = np.maximum(cells["last_update_ts"], cells["carve_ts"]).astype(np.int64)
    stale = (frame_ts - seen) > 50
    for arr in (sem, ele, cost, terr):
        arr[stale, :3] = (arr[stale, :3] * 0.45).astype(np.uint8)          # decaying / unseen > 5 s
    return {"terrain": terr, "semantic": sem, "elevation": ele, "cost": cost}


def rasterise(cells, colours, cx, cy, half_m, px_m, min_outline_px):
    """Paints every cell at its true footprint. Draw order per layer: in the cost layer the most recently
    observed cell wins (a free-space carve counts); in the semantic and elevation layers cells with
    returns are drawn above carved-only cells, so a fine free cell does not hide a coarser labelled one.
    Cells at least min_outline_px wide get a darker right/bottom edge so the resolution bands are visible."""
    size = int(round(2 * half_m / px_m))
    seen = np.maximum(cells["last_update_ts"], cells["carve_ts"]).astype(np.int64)
    hit = (cells["hit_count"] > 0).astype(np.int64)
    finer = -cells["level"].astype(np.int64)
    orders = {"cost": np.lexsort((finer, seen))}
    orders["terrain"] = orders["semantic"] = orders["elevation"] = np.lexsort((finer, cells["last_update_ts"].astype(np.int64), hit))
    out = {}
    for k, order in orders.items():
        img = np.zeros((size, size, 4), np.uint8)
        cs = cells["cell_size"][order]
        x0 = np.floor((cells["ix"][order] * cs - (cx - half_m)) / px_m + 1e-6).astype(np.int64)
        y0 = np.floor((cells["iy"][order] * cs - (cy - half_m)) / px_m + 1e-6).astype(np.int64)
        w = np.maximum(1, np.round(cs / px_m).astype(np.int64))
        inside = (x0 + w > 0) & (y0 + w > 0) & (x0 < size) & (y0 < size)
        pos = np.where(inside)[0]
        # Paint in draw order; group consecutive runs of equal width so later cells still overwrite earlier ones.
        breaks = np.flatnonzero(np.diff(w[pos])) + 1
        for run in np.split(pos, breaks):
            if len(run) == 0:
                continue
            wv = int(w[run[0]])
            oy, ox = np.meshgrid(np.arange(wv), np.arange(wv), indexing="ij")
            px = (x0[run, None] + ox.ravel()[None]).ravel()
            py = (y0[run, None] + oy.ravel()[None]).ravel()
            col = colours[k][np.repeat(order[run], wv * wv)]
            if wv >= min_outline_px:
                edge = np.tile(((ox == wv - 1) | (oy == wv - 1)).ravel(), len(run))
                col[edge, :3] = (col[edge, :3] * 0.55).astype(np.uint8)
            ok = (px >= 0) & (py >= 0) & (px < size) & (py < size)
            img[size - 1 - py[ok], px[ok]] = col[ok]                     # north up
        out[k] = png_b64(img)
    return out, size


# ── Main loop ─────────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=240)
    ap.add_argument("--speed", type=float, default=5.0, help="vehicle speed, m/s")
    ap.add_argument("--snapshots", type=int, default=8)
    ap.add_argument("--no-labels", action="store_true",
                    help="insert every point as UNKNOWN, i.e. the live system without a segmentation network")
    ap.add_argument("--out", default=os.path.join(REPO_ROOT, "reports", "map_dashboard.html"))
    args = ap.parse_args()

    rng = np.random.default_rng(26053)
    boxes = make_scene(rng)
    detector = DynamicObstacleDetector(dt=1.0 / HZ)
    drdo_map.reset_map()
    stages = {k: [] for k in ["scan_points", "insert_ms", "classify_ms", "decay_ms", "purge_ms", "perception_ms"]}
    snap_every = max(1, args.frames // args.snapshots)
    snapshots, load_hist, dyn_eval = [], [], []
    false_dyn_frames, dyn_track_frames = 0, 0
    first_seen, confirm_delay = {}, {}

    for f in range(1, args.frames + 1):
        t = (f - 1) / HZ
        sx, sy = args.speed * t, 0.0
        scan = sample_scan(boxes, t, sx, sy, rng)

        t0 = time.perf_counter()
        # Geometry only (no semantic labels): the same module the ROS node runs.
        dyn, moving_mask = detector.update(scan[:, :3], (sx, sy), GROUND_Z)
        mask = ~moving_mask
        t1 = time.perf_counter()
        if args.no_labels:
            scan[:, 3] = 6
        drdo_map.insert_points(scan[mask], sx, sy, f, GROUND_Z)
        t2 = time.perf_counter()
        drdo_map.classify_dirty(GROUND_Z)
        t3 = time.perf_counter()
        drdo_map.decay_kernel(f)
        t4 = time.perf_counter()
        if f % 10 == 0:
            drdo_map.purge_distant(sx, sy, 20.0)
        t5 = time.perf_counter()
        for k, v in zip(["perception_ms", "insert_ms", "classify_ms", "decay_ms", "purge_ms"],
                        [t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4]):
            stages[k].append(1000.0 * v)
        stages["scan_points"].append(len(scan))
        load_hist.append(drdo_map.load_factor())

        movers = [box_at(b, t) for b in boxes if b[8] in MOVERS]
        for ob in dyn:
            dyn_track_frames += 1
            false_dyn_frames += not any(np.hypot(ob.x - mx, ob.y - my) < 3.0 for mx, my in movers)
        for b in boxes:
            if b[8] not in MOVERS:
                continue
            bx, by = box_at(b, t)
            rng_m = float(np.hypot(bx - sx, by - sy))
            if rng_m > detector.max_range:
                first_seen.pop(b[8], None)
                continue
            first_seen.setdefault(b[8], f)
            found = any(np.hypot(ob.x - bx, ob.y - by) < 1.5 for ob in dyn)
            if found and b[8] not in confirm_delay:
                confirm_delay[b[8]] = (f - first_seen[b[8]]) / HZ
            if f - first_seen[b[8]] >= 10:                               # score after a 1 s confirmation window
                dyn_eval.append((b[8], rng_m, found))

        if f % snap_every == 0 or f == args.frames:
            cells = drdo_map.export_cells()
            colours = layer_colours(cells, f)
            near_imgs, near_px = rasterise(cells, colours, sx, sy, 15.0, 0.025, min_outline_px=20)
            far_imgs, far_px = rasterise(cells, colours, sx, sy, 100.0, 0.25, min_outline_px=10 ** 9)
            zoom_imgs, _ = rasterise(cells, colours, sx + ZOOM_AT[0], sy + ZOOM_AT[1], ZOOM_HALF, 0.01, min_outline_px=5)
            lv_counts = np.bincount(cells["level"], minlength=3).tolist()
            truth = [{"name": b[8], "x": box_at(b, t)[0] - sx, "y": box_at(b, t)[1] - sy}
                     for b in boxes if b[8] in MOVERS]
            snapshots.append({
                "frame": f, "time_s": round(t, 1), "robot": [sx, sy], "near": near_imgs, "far": far_imgs, "zoom": zoom_imgs,
                "near_px": near_px, "far_px": far_px, "cells": int(len(cells["ix"])), "levels": lv_counts,
                "tracks": [{"x": ob.x - sx, "y": ob.y - sy, "w": ob.length, "h": ob.width, "speed": ob.speed,
                            "id": ob.track_id, "dynamic": True} for ob in dyn],
                "truth": truth,
            })
            print(f"[DASH] frame {f:4d}  cells={len(cells['ix']):7d}  levels={lv_counts}  "
                  f"load={drdo_map.load_factor():.3f}  moving={len(dyn)}")

    cells_now = snapshots[-1]["cells"]
    cell_b, pool_cells = drdo_map.CELL_BYTES, drdo_map.POOL_CELLS
    disk = np.pi * 100.0 ** 2 / 0.05 ** 2
    memory = {
        "Foveated 2.5D (cells in use)": cells_now * cell_b / 2 ** 20,
        "Foveated 2.5D (fixed pool)": pool_cells * cell_b / 2 ** 20,
        "Uniform 5 cm 2.5D grid": disk * cell_b / 2 ** 20,
        "Uniform 5 cm 3D voxels (16 m, 1 B)": disk * (16.0 / 0.05) / 2 ** 20,
    }
    def stat(v):
        a = np.asarray(v[1:] if len(v) > 1 else v)
        return {"mean": float(a.mean()), "p95": float(np.percentile(a, 95)), "max": float(a.max())}
    latency = {k: stat(v) for k, v in stages.items() if k.endswith("_ms")}
    by_range = []
    for lo, hi in [(0, 10), (10, 25), (25, 40)]:
        rows = [e for e in dyn_eval if lo <= e[1] < hi]
        by_range.append({"range": f"{lo}-{hi} m", "frames": len(rows),
                         "recall": (sum(e[2] for e in rows) / len(rows)) if rows else None})
    summary = {
        "frames": args.frames, "speed_mps": args.speed, "distance_m": args.speed * (args.frames - 1) / HZ,
        "mean_scan_points": float(np.mean(stages["scan_points"])), "peak_load": float(max(load_hist)),
        "insert_failures": int(drdo_map.insert_fail_count()), "memory_mb": memory, "latency_ms": latency,
        "dynamic_recall_by_range": by_range,
        "dynamic_track_frames": dyn_track_frames, "false_dynamic_track_frames": false_dyn_frames,
        "confirm_delay_s": confirm_delay,
        "zoom": {"at": ZOOM_AT, "half": ZOOM_HALF},
        "bands": [{"level": i, "cell_cm": round(drdo_map.CELL_RESOLUTIONS[i] * 100),
                   "outer_m": drdo_map.LEVEL_OUTER_R[i]} for i in range(3)],
        "classes": [{"name": n, "rgb": PALETTE[i].tolist()} for i, n in enumerate(CLASS_NAMES)],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        label_note = ("Map built WITHOUT labels (every point UNKNOWN), as the live system runs today: terrain colours "
                      "come from geometry only." if args.no_labels else
                      "Map labels are simulator ground truth (no segmentation network in this loop): they tint the "
                      "semantic layer and mark water/obstacle cells; flat/rough/blocked also come from geometry.")
        fh.write(HTML.replace("__LABEL_NOTE__", label_note)
                 .replace("__DATA__", json.dumps({"summary": summary, "snapshots": snapshots})))
    total = sum(latency[k]["mean"] for k in ["insert_ms", "classify_ms", "decay_ms", "purge_ms"])
    print(f"[DASH] map update mean {total:.1f} ms/frame (insert {latency['insert_ms']['mean']:.1f}, "
          f"perception {latency['perception_ms']['mean']:.1f})  peak load {summary['peak_load']:.3f}  "
          f"insert failures {summary['insert_failures']}")
    print(f"[DASH] memory in use {memory['Foveated 2.5D (cells in use)']:.1f} MB vs uniform 5 cm 2.5D "
          f"{memory['Uniform 5 cm 2.5D grid']:.0f} MB vs uniform 5 cm 3D {memory['Uniform 5 cm 3D voxels (16 m, 1 B)']:.0f} MB")
    recall_txt = ", ".join(f"{r['range']}=" + ("n/a" if r["recall"] is None else f"{100 * r['recall']:.0f}%")
                           for r in by_range)
    print(f"[DASH] moving-object recall by range (after 1 s in range): {recall_txt}; static objects reported "
          f"moving: {false_dyn_frames}/{dyn_track_frames} object-frames; time to confirm: {confirm_delay}")
    print(f"[DASH] wrote {args.out}")
    return 0


HTML = r"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ID26053 Foveated Map Dashboard</title>
<style>
:root{--bg:#0f1318;--panel:#171d24;--line:#2a333d;--fg:#dde3ea;--mute:#8a96a3;--acc:#f2b84b}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;padding:16px}
h1{font-size:18px;margin:0 0 4px}h2{font-size:13px;margin:0 0 8px;color:var(--mute);text-transform:uppercase;letter-spacing:.06em}
.sub{color:var(--mute);margin-bottom:14px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;min-width:0}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;margin-bottom:12px}
.kpi{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:10px}.kpi b{display:block;font-size:20px;font-variant-numeric:tabular-nums}
.kpi span{color:var(--mute);font-size:12px}.controls{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:0 0 12px}
button{background:#222b35;color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 10px;cursor:pointer}
button.on{border-color:var(--acc);color:var(--acc)}input[type=range]{flex:1;min-width:160px}
.view{position:relative;width:100%;aspect-ratio:1;max-width:100%}.view img,.view canvas{position:absolute;inset:0;width:100%;height:100%;image-rendering:pixelated}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}td,th{padding:4px 6px;border-bottom:1px solid var(--line);text-align:right}td:first-child,th:first-child{text-align:left}
.legend{display:flex;flex-wrap:wrap;gap:6px 12px;font-size:12px;color:var(--mute)}.sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}
@media (min-width:1000px){.wide{grid-column:span 2}}
.bar{height:14px;background:var(--acc);border-radius:3px}.note{color:var(--mute);font-size:12px;margin-top:8px}
</style></head><body>
<h1>ID26053 — Adaptive variable-resolution 2.5D map</h1>
<div class="sub" id="sub"></div>
<div class="kpis" id="kpis"></div>
<div class="controls">
 <button id="play">▶ Play</button><input type="range" id="slider" min="0" value="0"><span id="flabel"></span>
 <span style="flex-basis:100%;height:0"></span>
 <button data-layer="terrain" class="on">Terrain (live view)</button><button data-layer="semantic">Semantic (sim labels)</button><button data-layer="elevation">Elevation</button><button data-layer="cost">Cost</button>
</div>
<div class="grid">
 <div class="panel"><h2>Near field ±15 m — cell outlines show 5 / 10 / 50 cm bands</h2><div class="view"><img id="near"><canvas id="nearc" width="1200" height="1200"></canvas></div><div class="legend" id="legend"></div></div>
 <div class="panel"><h2>Zoom 4 × 4 m across the 10 m band edge — every cell outlined</h2><div class="view"><img id="zoom"><canvas id="zoomc" width="800" height="800"></canvas></div><div class="note">Left/below the dashed arc: 5 cm cells; beyond it: 10 cm cells. Each 10 cm cell covers exactly four 5 cm cells (integer nesting from one 5 cm lattice), so nothing is misaligned or double-counted at the transition.</div></div>
 <div class="panel wide"><h2>Far field ±100 m — <span style="color:#ff5ad2">moving objects</span></h2><div class="view"><img id="far"><canvas id="farc" width="800" height="800"></canvas></div>
  <div class="note">Rings: 10 m (5 cm→10 cm), 25 m (10 cm→50 cm), 100 m (range limit). Dimmed cells: unseen for &gt;5 s (decaying). Dark slate: carved free space with no returns.</div></div>
 <div class="panel"><h2>Memory, same 100 m radius</h2><div id="mem"></div><div class="note">Log scale. The fixed pool is preallocated once; "in use" is live cells × 40 B.</div></div>
 <div class="panel"><h2>Latency per frame (host CPU, single thread)</h2><table id="lat"></table><div class="note" id="latnote"></div></div>
 <div class="panel"><h2>Moving-object detection by range</h2><table id="dyn"></table><div class="note">Geometry only (no labels): clusters 0.3–2.5 m above ground, confirmed moving when speed &gt; 1 m/s, the whole footprint shifted and the space it left is empty. Recall counts frames after 1 s in range, matched within 1.5 m. Their points are kept out of the static map. Scene: 2 walkers, 1 oncoming vehicle; parked car, standing person, wall, poles and trees must stay static.</div></div>
 <div class="panel"><h2>Cells per resolution level (current frame)</h2><table id="lv"></table></div>
</div>
<div class="note" style="margin-top:12px">Synthetic scene, OS1-64 beam pattern, no occlusion. __LABEL_NOTE__ Moving-object detection never uses labels. Generated by scripts/map_dashboard.py.</div>
<script>
const D=__DATA__, S=D.summary, snaps=D.snapshots; let idx=snaps.length-1, layer='terrain', timer=null;
const fmt=(v,d=1)=>v==null?'n/a':v.toFixed(d);
document.getElementById('sub').textContent=`${S.frames} frames at ${S.speed_mps} m/s (${fmt(S.distance_m,0)} m), ~${fmt(S.mean_scan_points,0)} points/scan, bands: `+S.bands.map(b=>`${b.cell_cm} cm ≤ ${b.outer_m} m`).join(', ');
const lat=S.latency_ms, mapMs=['insert_ms','classify_ms','decay_ms','purge_ms'].reduce((a,k)=>a+lat[k].mean,0);
const m=S.memory_mb, inuse=m['Foveated 2.5D (cells in use)'];
document.getElementById('kpis').innerHTML=[
 [fmt(mapMs)+' ms','map update / frame (mean)'],[fmt(1000/(mapMs+lat.perception_ms.mean),0)+' Hz','map + perception rate'],
 [fmt(inuse)+' MB','map memory in use'],[fmt(m['Uniform 5 cm 3D voxels (16 m, 1 B)']/inuse,0)+'×','smaller than uniform 5 cm 3D'],
 [fmt(S.peak_load,3),'peak hash load'],[S.insert_failures,'dropped inserts']].map(k=>`<div class="kpi"><b>${k[0]}</b><span>${k[1]}</span></div>`).join('');
const mx=Math.log10(Math.max(...Object.values(m))*2);
document.getElementById('mem').innerHTML='<table>'+Object.entries(m).map(([k,v])=>`<tr><td>${k}</td><td style="width:45%"><div class="bar" style="width:${Math.max(2,100*Math.log10(Math.max(v,1)*10)/(mx+1))}%"></div></td><td>${v<10?fmt(v):fmt(v,0)} MB</td></tr>`).join('')+'</table>';
document.getElementById('lat').innerHTML='<tr><th>stage</th><th>mean</th><th>p95</th><th>max</th></tr>'+Object.entries(lat).map(([k,v])=>`<tr><td>${k.replace('_ms','')}</td><td>${fmt(v.mean,2)}</td><td>${fmt(v.p95,2)}</td><td>${fmt(v.max,2)}</td></tr>`).join('');
document.getElementById('latnote').textContent='Timed around the Python→C++ calls, so insert includes the numpy transfer. Perception is Python/scikit-learn.';
document.getElementById('dyn').innerHTML='<tr><th>range</th><th>object-frames</th><th>recall</th></tr>'+S.dynamic_recall_by_range.map(r=>`<tr><td>${r.range}</td><td>${r.frames}</td><td>${r.recall==null?'n/a':fmt(100*r.recall,0)+'%'}</td></tr>`).join('')+`<tr><td>static objects reported moving</td><td>${S.false_dynamic_track_frames} / ${S.dynamic_track_frames}</td><td>object-frames</td></tr>`+Object.entries(S.confirm_delay_s).map(([k,v])=>`<tr><td>time to confirm: ${k}</td><td></td><td>${fmt(v,1)} s</td></tr>`).join('');
document.getElementById('legend').innerHTML=S.classes.map(c=>`<span><i class="sw" style="background:rgb(${c.rgb})"></i>${c.name}</span>`).join('')+'<span><i class="sw" style="background:#2d343e"></i>FREE (carved)</span><br><b style="color:var(--fg);font-weight:600">Terrain view:</b> <span><i class="sw" style="background:rgb(76,170,84)"></i>flat</span><span><i class="sw" style="background:rgb(232,170,40)"></i>rough</span><span><i class="sw" style="background:rgb(225,40,40)"></i>blocked</span><span><i class="sw" style="background:rgb(60,160,170)"></i>overhang</span><span><i class="sw" style="background:rgb(60,120,225)"></i>water/mud</span><span><i class="sw" style="background:rgb(38,72,52)"></i>free</span>';
const hashIdx=parseInt((location.hash.match(/snap=(\d+)/)||[])[1]); if(hashIdx>=0&&hashIdx<snaps.length) idx=hashIdx;
const slider=document.getElementById('slider'); slider.max=snaps.length-1; slider.value=idx;
function overlay(cv,half,s,rings){const g=cv.getContext('2d'),W=cv.width,k=W/(2*half);g.clearRect(0,0,W,W);g.lineWidth=Math.max(1,W/600);
 g.setLineDash([6,6]);g.strokeStyle='rgba(242,184,75,.8)';rings.forEach(r=>{g.beginPath();g.arc(W/2,W/2,r*k,0,7);g.stroke()});g.setLineDash([]);
 s.tracks.forEach(t=>{if(Math.abs(t.x)>half||Math.abs(t.y)>half)return;g.strokeStyle=t.dynamic?'#ff5ad2':'rgba(255,255,255,.8)';g.lineWidth=Math.max(2,W/300);
  const w=Math.max(t.w*k,W/40),h=Math.max(t.h*k,W/40);g.fillStyle='rgba(255,90,210,.35)';g.fillRect(W/2+t.x*k-w/2,W/2-t.y*k-h/2,w,h);g.lineWidth=Math.max(3,W/200);g.strokeRect(W/2+t.x*k-w/2,W/2-t.y*k-h/2,w,h);
  if(t.dynamic){g.fillStyle='#ff5ad2';g.font=`bold ${Math.max(14,W/35)}px system-ui`;g.fillText(`#${t.id} ${t.speed.toFixed(1)} m/s`,W/2+t.x*k+w/2+3,W/2-t.y*k)}});
 g.fillStyle='#f2b84b';g.beginPath();g.moveTo(W/2+10*W/800,W/2);g.lineTo(W/2-6*W/800,W/2-7*W/800);g.lineTo(W/2-6*W/800,W/2+7*W/800);g.fill();}
function zoomRing(){const cv=document.getElementById('zoomc'),g=cv.getContext('2d'),W=cv.width,h=S.zoom.half,k=W/(2*h),[ax,ay]=S.zoom.at;
 g.clearRect(0,0,W,W);g.setLineDash([10,8]);g.lineWidth=3;g.strokeStyle='rgba(242,184,75,.95)';g.beginPath();g.arc(W/2-ax*k,W/2+ay*k,10*k,0,7);g.stroke();}
function render(){const s=snaps[idx];document.getElementById('near').src=s.near[layer];document.getElementById('far').src=s.far[layer];document.getElementById('zoom').src=s.zoom[layer];zoomRing();
 document.getElementById('flabel').textContent=`frame ${s.frame} · t=${s.time_s} s · ${s.cells.toLocaleString()} cells`;
 overlay(document.getElementById('nearc'),15,s,[10]);overlay(document.getElementById('farc'),100,s,[10,25,100]);
 document.getElementById('lv').innerHTML='<tr><th>level</th><th>cell</th><th>radius</th><th>cells</th><th>MB</th></tr>'+S.bands.map((b,i)=>`<tr><td>${i}</td><td>${b.cell_cm} cm</td><td>≤ ${b.outer_m} m</td><td>${s.levels[i].toLocaleString()}</td><td>${fmt(s.levels[i]*40/1048576,2)}</td></tr>`).join('');}
slider.oninput=()=>{idx=+slider.value;render()};
document.querySelectorAll('[data-layer]').forEach(b=>b.onclick=()=>{layer=b.dataset.layer;document.querySelectorAll('[data-layer]').forEach(x=>x.classList.toggle('on',x===b));render()});
document.getElementById('play').onclick=e=>{if(timer){clearInterval(timer);timer=null;e.target.textContent='▶ Play';return}
 e.target.textContent='❚❚ Pause';timer=setInterval(()=>{idx=(idx+1)%snaps.length;slider.value=idx;render()},900)};
render();
</script></body></html>"""


if __name__ == "__main__":
    sys.exit(main())

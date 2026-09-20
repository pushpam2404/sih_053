#!/usr/bin/env python3
"""DRDO ID26053 — geometry-only moving-object detection that keeps movers out of the static 2.5D map.

Needs no segmentation labels: it works on world-frame points (FAST-LIO /cloud_registered) plus the
robot pose. Per frame:

  1. Candidates: points 0.3-2.5 m above the ground reference, within max_range of the sensor.
  2. Clusters: 8-connected components on a 20 cm occupancy raster (1-cell gaps bridged).
  3. Tracking: constant-velocity tracks, gated nearest-neighbour association (Hungarian if scipy).
  4. Confirmation — a track is reported as MOVING only if, for confirm_frames consecutive frames:
       a. its speed is above min_speed,
       b. its WHOLE footprint shifted by at least min_displacement over the evidence window (both
          edges along an axis moved the same way — "edge-shift test") without its size changing by more
          than max_size_change (clusters merging or splitting), and
       c. the 50 cm cells it covered ~1 s ago, outside its current footprint, are now empty
          ("vacated-space test").
     A speed threshold alone reports static objects as moving whenever their *visible part* changes as
     the vehicle drives by: a new face of a tree or parked car comes into view and the cluster centre
     jumps (b catches this), or a partly seen wall's centroid slides along it while the wall is still
     there (c catches this). Measured in the dashboard scene, speed threshold only (DBSCAN + SORT):
     69 of 337 "moving" reports were static objects.
  5. Output: the confirmed movers and a boolean mask of their points, so the caller can drop them before
     inserting into the map (a walking person otherwise paints a lethal trail until temporal decay).

Clusters longer than max_footprint (walls, tree lines, embankments) are never tracked.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from ..segmentation.taxonomy import OBJ_NONE
from .classify import classify_extent

try:
    from scipy import ndimage
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - scipy is in requirements.txt
    SCIPY_AVAILABLE = False


@dataclass
class MovingObject:
    track_id: int
    x: float
    y: float
    vx: float
    vy: float
    length: float          # footprint extent along world x (m)
    width: float           # footprint extent along world y (m)
    z_min: float
    z_max: float
    num_points: int
    age_frames: int
    # Geometric classification (drdo_lidar_mapping/perception/classify.py) so a confirmed mover
    # is named ("person", "vehicle") and coloured instead of a uniform magenta box. Defaulted so
    # any existing direct construction of MovingObject (tests, older callers) keeps working.
    obj_class: int = OBJ_NONE
    obj_conf: float = 0.0

    @property
    def speed(self) -> float:
        return float(np.hypot(self.vx, self.vy))


@dataclass
class _Track:
    track_id: int
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    hits: int = 1
    misses: int = 0
    age: int = 1
    streak: int = 0
    confirmed: bool = False
    history: List[Tuple[int, np.ndarray, frozenset]] = field(default_factory=list)   # (frame, bbox, cells)


class DynamicObstacleDetector:
    def __init__(self, dt: float = 0.1, min_height: float = 0.3, max_height: float = 2.5,
                 max_range: float = 40.0, cluster_cell: float = 0.2, min_points: int = 8,
                 max_footprint: float = 7.0, min_speed: float = 1.0, min_displacement: float = 0.8,
                 confirm_frames: int = 3, evidence_frames: int = 10, vacate_cell: float = 0.5,
                 max_vacated_occupancy: float = 0.25, min_vacate_cells: int = 2,
                 max_misses: int = 3, max_size_change: float = 1.0):
        self.dt = dt
        self.min_height, self.max_height, self.max_range = min_height, max_height, max_range
        self.cluster_cell, self.min_points, self.max_footprint = cluster_cell, min_points, max_footprint
        self.min_speed, self.min_displacement = min_speed, min_displacement
        self.confirm_frames, self.evidence_frames = confirm_frames, evidence_frames
        self.vacate_cell, self.max_vacated_occupancy = vacate_cell, max_vacated_occupancy
        self.min_vacate_cells, self.max_misses = min_vacate_cells, max_misses
        self.max_size_change = max_size_change   # m; clusters that merge/split change size, movers don't
        self.tracks: Dict[int, _Track] = {}
        self.frame = 0
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self.frame = 0
        self._next_id = 1

    @staticmethod
    def _edge_shift(old_box: np.ndarray, new_box: np.ndarray) -> float:
        """Distance moved by the WHOLE footprint: along each axis, both edges must move the same way and
        the smaller of the two shifts counts. A change of viewpoint moves one edge (a new face comes
        into view) while the opposite edge stays put, so it scores ~0; a real mover shifts both."""
        shift = []
        for a, b in ((0, 1), (2, 3)):
            d_lo, d_hi = new_box[a] - old_box[a], new_box[b] - old_box[b]
            shift.append(min(abs(d_lo), abs(d_hi)) if d_lo * d_hi > 0 else 0.0)
        return float(np.hypot(*shift))

    # ── helpers ──────────────────────────────────────────────────────────────────────────────────
    def _vacate_keys(self, xy: np.ndarray) -> set:
        k = np.floor(xy / self.vacate_cell).astype(np.int64)
        return set(zip(k[:, 0].tolist(), k[:, 1].tolist()))

    def _cluster(self, xy: np.ndarray, sensor_xy: np.ndarray) -> np.ndarray:
        """Cluster id per point (-1 = none). Connected components on a sensor-centred raster."""
        n_side = int(np.ceil(2 * self.max_range / self.cluster_cell)) + 1
        origin = sensor_xy - self.max_range
        ij = np.floor((xy - origin) / self.cluster_cell).astype(np.int64)
        ok = (ij[:, 0] >= 0) & (ij[:, 1] >= 0) & (ij[:, 0] < n_side) & (ij[:, 1] < n_side)
        grid = np.zeros((n_side, n_side), bool)
        grid[ij[ok, 0], ij[ok, 1]] = True
        if SCIPY_AVAILABLE:
            bridged = ndimage.binary_dilation(grid, structure=np.ones((3, 3), bool))
            labels, _ = ndimage.label(bridged, structure=np.ones((3, 3), bool))
        else:  # slow but dependency-free fallback
            labels = np.zeros_like(grid, np.int32)
            cur = 0
            for i0, j0 in zip(*np.nonzero(grid)):
                if labels[i0, j0]:
                    continue
                cur += 1
                stack = [(i0, j0)]
                labels[i0, j0] = cur
                while stack:
                    i, j = stack.pop()
                    for di in (-2, -1, 0, 1, 2):
                        for dj in (-2, -1, 0, 1, 2):
                            a, b = i + di, j + dj
                            if 0 <= a < n_side and 0 <= b < n_side and grid[a, b] and not labels[a, b]:
                                labels[a, b] = cur
                                stack.append((a, b))
        out = np.full(len(xy), -1, np.int64)
        out[ok] = labels[ij[ok, 0], ij[ok, 1]] - 1
        return out

    # ── main entry ───────────────────────────────────────────────────────────────────────────────
    def update(self, xyz: np.ndarray, sensor_xy, ground_z: float) -> Tuple[List[MovingObject], np.ndarray]:
        """xyz: (N, >=3) world-frame points. Returns (confirmed moving objects, bool mask of their points)."""
        self.frame += 1
        xyz = np.asarray(xyz)
        sensor_xy = np.asarray(sensor_xy, np.float64).reshape(2)
        n = len(xyz)
        moving_mask = np.zeros(n, bool)
        if n:
            xy_all = xyz[:, :2].astype(np.float64)
            dz = xyz[:, 2] - ground_z
            rng = np.hypot(xy_all[:, 0] - sensor_xy[0], xy_all[:, 1] - sensor_xy[1])
            cand = np.flatnonzero(np.isfinite(dz) & (dz >= self.min_height) & (dz <= self.max_height)
                                  & (rng <= self.max_range))
        else:
            cand = np.zeros(0, np.int64)

        dets = []   # (cx, cy, length, width, point_indices, footprint_keys)
        occupied = set()
        if len(cand):
            xy = xy_all[cand]
            occupied = self._vacate_keys(xy)
            cid = self._cluster(xy, sensor_xy)
            valid = cid >= 0
            order = np.argsort(cid[valid], kind="stable")
            idx_sorted = np.flatnonzero(valid)[order]
            cids = cid[idx_sorted]
            splits = np.flatnonzero(np.diff(cids)) + 1
            for members in np.split(idx_sorted, splits):
                if len(members) < self.min_points:
                    continue
                pts = xy[members]
                lo, hi = pts.min(axis=0), pts.max(axis=0)
                ext = hi - lo
                if max(ext) > self.max_footprint:
                    continue
                c = (lo + hi) / 2.0
                dets.append((c[0], c[1], max(ext[0], 0.1), max(ext[1], 0.1), cand[members], self._vacate_keys(pts),
                             np.array([lo[0], hi[0], lo[1], hi[1]])))

        # Associate detections to predicted tracks.
        ids = list(self.tracks.keys())
        pred = np.array([[self.tracks[i].x + self.tracks[i].vx * self.dt,
                          self.tracks[i].y + self.tracks[i].vy * self.dt] for i in ids]).reshape(-1, 2)
        matches, used_d, used_t = [], set(), set()
        if len(ids) and dets:
            dc = np.array([[d[0], d[1]] for d in dets])
            cost = np.linalg.norm(pred[:, None, :] - dc[None, :, :], axis=2)
            gate = np.array([[1.0 + 0.5 * max(dets[j][2], dets[j][3]) for j in range(len(dets))]] * len(ids))
            big = 1e6
            cost_g = np.where(cost <= gate, cost, big)
            if SCIPY_AVAILABLE:
                rows, cols = linear_sum_assignment(cost_g)
                pairs = [(r, c) for r, c in zip(rows, cols) if cost_g[r, c] < big]
            else:
                pairs = []
                for r, c in sorted(zip(*np.nonzero(cost_g < big)), key=lambda rc: cost_g[rc]):
                    if r not in {p[0] for p in pairs} and c not in {p[1] for p in pairs}:
                        pairs.append((r, c))
            for r, c in pairs:
                matches.append((ids[r], c))
                used_t.add(ids[r])
                used_d.add(c)

        result: List[MovingObject] = []
        for tid, j in matches:
            t, d = self.tracks[tid], dets[j]
            mvx, mvy = (d[0] - t.x) / self.dt, (d[1] - t.y) / self.dt
            if t.hits == 1:
                t.vx, t.vy = mvx, mvy
            else:
                t.vx, t.vy = 0.6 * t.vx + 0.4 * mvx, 0.6 * t.vy + 0.4 * mvy
            t.x, t.y = d[0], d[1]
            t.hits += 1
            t.age += 1
            t.misses = 0
            t.history.append((self.frame, d[6], frozenset(d[5])))
            t.history = [h for h in t.history if self.frame - h[0] <= self.evidence_frames]

            evidence = False
            old = t.history[0]
            if self.frame - old[0] >= self.evidence_frames // 2:
                speed_ok = np.hypot(t.vx, t.vy) > self.min_speed
                old_ext = np.array([old[1][1] - old[1][0], old[1][3] - old[1][2]])
                size_ok = np.abs(np.array([d[2], d[3]]) - old_ext).max() <= self.max_size_change
                disp_ok = size_ok and self._edge_shift(old[1], d[6]) >= self.min_displacement
                vac = [k for k in old[2] - d[5]
                       if np.hypot((k[0] + 0.5) * self.vacate_cell - sensor_xy[0],
                                   (k[1] + 0.5) * self.vacate_cell - sensor_xy[1]) <= self.max_range - 1.0]
                vac_ok = len(vac) >= self.min_vacate_cells and (
                    not vac or sum(k in occupied for k in vac) / len(vac) <= self.max_vacated_occupancy)
                evidence = speed_ok and disp_ok and vac_ok
            if evidence:
                t.streak += 1
                if t.streak >= self.confirm_frames:
                    t.confirmed = True
            else:
                t.streak = 0
                if t.confirmed and np.hypot(t.vx, t.vy) < 0.5 * self.min_speed:
                    t.confirmed = False                                   # it stopped: back to the static map
            if t.confirmed:
                moving_mask[d[4]] = True
                z = xyz[d[4], 2]
                z_min, z_max = float(z.min()), float(z.max())
                obj_class, obj_conf = classify_extent(float(d[2]), float(d[3]), z_max - z_min)
                result.append(MovingObject(tid, float(d[0]), float(d[1]), float(t.vx), float(t.vy),
                                           float(d[2]), float(d[3]), z_min, z_max,
                                           int(len(d[4])), t.age, obj_class, obj_conf))

        for tid in ids:
            if tid in used_t:
                continue
            t = self.tracks[tid]
            t.misses += 1
            t.age += 1
            t.x += t.vx * self.dt
            t.y += t.vy * self.dt
            if t.misses > self.max_misses:
                del self.tracks[tid]

        for j, d in enumerate(dets):
            if j in used_d:
                continue
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = _Track(tid, d[0], d[1], history=[(self.frame, d[6], frozenset(d[5]))])
        return result, moving_mask


# ── PointCloud2 helpers (pure numpy, no ROS import) ─────────────────────────────────────────────
def cloud_xyz(data: bytes, point_step: int, offsets: Tuple[int, int, int], n_points: int) -> np.ndarray:
    """(N, 3) float32 view of little-endian FLOAT32 x/y/z fields in a PointCloud2 byte buffer."""
    raw = np.frombuffer(data, np.uint8, count=n_points * point_step).reshape(n_points, point_step)
    return np.stack([raw[:, o:o + 4].copy().view("<f4").ravel() for o in offsets], axis=1)


def filter_cloud_bytes(data: bytes, point_step: int, n_points: int, keep: np.ndarray) -> bytes:
    """Row-select a PointCloud2 buffer, keeping every field (intensity, label, ...) intact."""
    raw = np.frombuffer(data, np.uint8, count=n_points * point_step).reshape(n_points, point_step)
    return raw[keep].tobytes()

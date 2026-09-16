#!/usr/bin/env python3
"""
DRDO ID26053 — Phase 5: SORT (Simple Online and Realtime Tracking) for Hard Obstacles
Implements Kalman Filter state estimation (x, y, dx, dy, vx, vy) and Hungarian data association.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def calculate_iou_2d(bb_test: np.ndarray, bb_gt: np.ndarray) -> float:
    """
    Computes 2D IoU between two bounding boxes.
    Format: [cx, cy, dx, dy] (center x, center y, width dx, height dy)
    """
    x1_min = bb_test[0] - bb_test[2] / 2.0
    x1_max = bb_test[0] + bb_test[2] / 2.0
    y1_min = bb_test[1] - bb_test[3] / 2.0
    y1_max = bb_test[1] + bb_test[3] / 2.0

    x2_min = bb_gt[0] - bb_gt[2] / 2.0
    x2_max = bb_gt[0] + bb_gt[2] / 2.0
    y2_min = bb_gt[1] - bb_gt[3] / 2.0
    y2_max = bb_gt[1] + bb_gt[3] / 2.0

    xx_min = max(x1_min, x2_min)
    xx_max = min(x1_max, x2_max)
    yy_min = max(y1_min, y2_min)
    yy_max = min(y1_max, y2_max)

    w = max(0.0, xx_max - xx_min)
    h = max(0.0, yy_max - yy_min)
    intersection = w * h

    area1 = bb_test[2] * bb_test[3]
    area2 = bb_gt[2] * bb_gt[3]
    union = area1 + area2 - intersection

    if union <= 0.0:
        return 0.0
    return float(intersection / union)


def greedy_assignment(cost_matrix: np.ndarray, threshold: float = 1e5) -> Tuple[np.ndarray, np.ndarray]:
    """Pure NumPy fallback greedy assignment if SciPy is missing."""
    rows, cols = [], []
    costs = cost_matrix.copy()
    while True:
        min_val = np.min(costs)
        if min_val >= threshold or np.isinf(min_val):
            break
        r, c = np.unravel_index(np.argmin(costs), costs.shape)
        rows.append(r)
        cols.append(c)
        costs[r, :] = np.inf
        costs[:, c] = np.inf
    return np.array(rows, dtype=int), np.array(cols, dtype=int)


class KalmanBoxTracker:
    """
    Tracks a single bounding box instance with a 2D Constant-Velocity Kalman Filter.
    State: [x, y, dx, dy, vx, vy]^T
    Measurement: [x, y, dx, dy]^T
    """
    _count = 0

    def __init__(self, bbox: np.ndarray, dt: float = 0.1):
        """
        bbox: [cx, cy, dx, dy] (or [cx, cy, cz, dx, dy, dz], first 2 and dimension used)
        dt: time step in seconds (10 Hz = 0.1s)
        """
        self.id = KalmanBoxTracker._count
        KalmanBoxTracker._count += 1
        self.dt = dt

        # State vector: [x, y, dx, dy, vx, vy]
        self.x = np.zeros((6, 1), dtype=np.float32)
        self.x[0, 0] = bbox[0]
        self.x[1, 0] = bbox[1]
        self.x[2, 0] = bbox[2]
        self.x[3, 0] = bbox[3]
        self.x[4, 0] = 0.0  # vx
        self.x[5, 0] = 0.0  # vy

        # State transition matrix F
        self.F = np.eye(6, dtype=np.float32)
        self.F[0, 4] = dt
        self.F[1, 5] = dt

        # Measurement matrix H
        self.H = np.zeros((4, 6), dtype=np.float32)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # State covariance P
        self.P = np.diag([1.0, 1.0, 1.0, 1.0, 10.0, 10.0]).astype(np.float32)

        # Process noise Q
        self.Q = np.diag([0.05, 0.05, 0.01, 0.01, 0.5, 0.5]).astype(np.float32)

        # Measurement noise R
        self.R = np.diag([0.1, 0.1, 0.1, 0.1]).astype(np.float32)

        self.time_since_update = 0
        self.hits = 1
        self.hit_streak = 1
        self.age = 0

    def predict(self) -> np.ndarray:
        """Advance the state vector and returns the predicted bounding box [cx, cy, dx, dy]."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        return self.get_state()

    def update(self, bbox: np.ndarray):
        """Update the state with observed bounding box [cx, cy, dx, dy]."""
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1

        z = np.reshape(bbox[:4], (4, 1)).astype(np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(6, dtype=np.float32)
        self.P = (I - K @ self.H) @ self.P

    def get_state(self) -> np.ndarray:
        """Returns the current bounding box estimate [cx, cy, dx, dy]."""
        return np.array([self.x[0, 0], self.x[1, 0], self.x[2, 0], self.x[3, 0]], dtype=np.float32)

    def get_velocity(self) -> np.ndarray:
        """Returns the current estimated velocity [vx, vy]."""
        return np.array([self.x[4, 0], self.x[5, 0]], dtype=np.float32)


def associate_detections_to_trackers(detections: np.ndarray,
                                     trackers: np.ndarray,
                                     iou_threshold: float = 0.15,
                                     dist_fallback_threshold: float = 2.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Assigns detections to tracked objects using IoU and Euclidean distance fallback.
    Returns:
        matches: (M, 2) array of [detection_idx, tracker_idx]
        unmatched_detections: (U,) array of detection indices
        unmatched_trackers: (V,) array of tracker indices
    """
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections), dtype=int), np.empty((0,), dtype=int)
    if len(detections) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=int), np.arange(len(trackers), dtype=int)

    num_det = len(detections)
    num_trk = len(trackers)
    cost_matrix = np.zeros((num_det, num_trk), dtype=np.float32)

    for d in range(num_det):
        for t in range(num_trk):
            iou = calculate_iou_2d(detections[d], trackers[t])
            # If IoU > 0, cost is 1 - iou (range [0, 1])
            if iou > 0.0:
                cost_matrix[d, t] = 1.0 - iou
            else:
                # Euclidean distance fallback for fast moving or small bounding boxes
                dist = np.linalg.norm(detections[d, :2] - trackers[t, :2])
                if dist <= dist_fallback_threshold:
                    cost_matrix[d, t] = 1.0 + (dist / dist_fallback_threshold)
                else:
                    cost_matrix[d, t] = 1e5

    if SCIPY_AVAILABLE:
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    else:
        row_ind, col_ind = greedy_assignment(cost_matrix)

    unmatched_detections = []
    for d in range(num_det):
        if d not in row_ind:
            unmatched_detections.append(d)

    unmatched_trackers = []
    for t in range(num_trk):
        if t not in col_ind:
            unmatched_trackers.append(t)

    matches = []
    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] >= 2.0:  # Exceeded fallback threshold
            unmatched_detections.append(r)
            unmatched_trackers.append(c)
        else:
            matches.append([r, c])

    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.array(matches, dtype=int)

    return matches, np.array(unmatched_detections, dtype=int), np.array(unmatched_trackers, dtype=int)


class SortTracker:
    """
    SORT Tracker managing multiple active tracks for dynamic hard obstacle tracking.
    """
    def __init__(self, max_age: int = 3, min_hits: int = 2, iou_threshold: float = 0.15, dt: float = 0.1):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.dt = dt
        self.trackers: List[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, detections: np.ndarray) -> np.ndarray:
        """
        Processes detections for the current frame.
        detections: (N, 4) array of [cx, cy, dx, dy]
        Returns:
            active_tracks: (M, 7) array of [cx, cy, dx, dy, vx, vy, track_id]
        """
        self.frame_count += 1

        # 1. Predict new locations of existing trackers
        trks = np.zeros((len(self.trackers), 4), dtype=np.float32)
        to_del = []
        for t, trk in enumerate(self.trackers):
            pos = trk.predict()
            trks[t, :] = pos
            if np.any(np.isnan(pos)):
                to_del.append(t)
        for t in reversed(to_del):
            self.trackers.pop(t)
            trks = np.delete(trks, t, axis=0)

        # 2. Match detections with predicted tracks
        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            detections, trks, iou_threshold=self.iou_threshold
        )

        # 3. Update matched trackers
        for m in matched:
            self.trackers[m[1]].update(detections[m[0]])

        # 4. Create new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i], dt=self.dt)
            self.trackers.append(trk)

        # 5. Filter active tracks and remove dead trackers
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()
            v = trk.get_velocity()
            # Return track if it has sufficient hits and was updated recently
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d, v, [trk.id])).reshape(1, -1))
            i -= 1
            # Remove old tracks
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.vstack(ret)
        return np.empty((0, 7), dtype=np.float32)


def main():
    print("Testing Step P5.1.2: SORT Dynamic Tracker...")
    KalmanBoxTracker._count = 0  # reset ID counter
    tracker = SortTracker(max_age=3, min_hits=2, dt=0.1)

    # Simulate 10 frames with 3 obstacles:
    # 1. Vehicle moving at vx = 2.0 m/s along y=5.0
    # 2. Pedestrian moving at vx = -0.5 m/s, vy = 1.0 m/s from [10.0, 0.0]
    # 3. Static obstacle at [0.0, 15.0]
    dt = 0.1
    num_frames = 10

    observed_tracks_per_frame = []
    for frame in range(num_frames):
        t = frame * dt
        # True positions
        pos1 = [2.0 * t, 5.0, 2.0, 1.5]            # size dx=2.0, dy=1.5
        pos2 = [10.0 - 0.5 * t, 1.0 * t, 0.8, 0.8]  # size dx=0.8, dy=0.8
        pos3 = [0.0, 15.0, 1.2, 1.2]               # static

        # Add slight observation noise
        noisy1 = [pos1[0] + np.random.normal(0, 0.02), pos1[1] + np.random.normal(0, 0.02), pos1[2], pos1[3]]
        noisy2 = [pos2[0] + np.random.normal(0, 0.02), pos2[1] + np.random.normal(0, 0.02), pos2[2], pos2[3]]
        noisy3 = [pos3[0] + np.random.normal(0, 0.02), pos3[1] + np.random.normal(0, 0.02), pos3[2], pos3[3]]

        dets = np.array([noisy1, noisy2, noisy3], dtype=np.float32)
        active_tracks = tracker.update(dets)

        print(f"Frame {frame+1:02d} (t={t:.1f}s): {len(active_tracks)} active tracks")
        for trk in active_tracks:
            tid = int(trk[6])
            cx, cy, dx, dy, vx, vy = trk[:6]
            print(f"    Track ID {tid}: Pos=[{cx:.2f}, {cy:.2f}], Extent=[{dx:.2f}, {dy:.2f}], Vel=[vx={vx:.2f}, vy={vy:.2f}] m/s")

        observed_tracks_per_frame.append(active_tracks)

    # Verification checks
    last_tracks = observed_tracks_per_frame[-1]
    assert len(last_tracks) == 3, f"Expected 3 confirmed active tracks on frame 10, got {len(last_tracks)}"

    # Check that track IDs are persistent (exactly 3 distinct IDs created across test)
    all_ids = set()
    for fr_tracks in observed_tracks_per_frame:
        for trk in fr_tracks:
            all_ids.add(int(trk[6]))
    print(f"Total unique persistent track IDs assigned: {all_ids}")
    assert len(all_ids) == 3, f"Expected 3 persistent track IDs, got {len(all_ids)}"

    # Find the vehicle track (near y=5.0)
    vehicle_track = [trk for trk in last_tracks if abs(trk[1] - 5.0) < 0.5][0]
    vx_est = vehicle_track[4]
    vy_est = vehicle_track[5]
    print(f"Vehicle estimated velocity: vx={vx_est:.2f} m/s, vy={vy_est:.2f} m/s (Ground truth: vx=2.0, vy=0.0)")
    assert abs(vx_est - 2.0) < 0.5, f"Estimated vx ({vx_est}) deviated too much from 2.0 m/s"
    assert abs(vy_est - 0.0) < 0.5, f"Estimated vy ({vy_est}) deviated too much from 0.0 m/s"

    print("[STEP P5.1.2 COMPLETE]")


if __name__ == "__main__":
    main()

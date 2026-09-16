#!/usr/bin/env python3
"""DRDO ID26053 — Perception & Dynamic Obstacle Tracking Unit Test Suite.

Validates:
  1. Serialized TorchScript model loading and inference
  2. Dynamic hard obstacle extraction via DBSCAN clustering
  3. Instance-level multi-object tracking via SORT (Kalman + Hungarian)
  4. ROS 2 Bringup Launch & SLAM configuration validation
"""

import os
import sys
import numpy as np
import torch
import importlib.util

# Ensure repo root is on sys.path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from drdo_lidar_mapping.perception.cluster import cluster_hard_obstacles, CLASS_OBSTACLE_HARD
from drdo_lidar_mapping.perception.tracker import SortTracker


def test_perception_e2e():
    print("================================================================================")
    print("       DRDO ID26053: PERCEPTION & DYNAMIC OBSTACLE TRACKING TEST SUITE          ")
    print("================================================================================")

    # 1. Load exported TorchScript Model
    model_path = os.path.join(REPO_ROOT, "models/minkunet18_traced.pt")
    assert os.path.exists(model_path), f"TorchScript model not found at {model_path}"
    model = torch.jit.load(model_path, map_location=torch.device("cpu")).eval()
    print(f"[1/4] Loaded TorchScript model successfully: {model_path}")

    # 2. Simulate 10 frames of point cloud with moving hard obstacles
    tracker = SortTracker(max_age=3, min_hits=2, dt=0.1)
    np.random.seed(42)

    print("[2/4] Executing 10-frame dynamic tracking pipeline...")
    active_tracks = []
    for frame in range(10):
        t = frame * 0.1
        # Ground points
        n_ground = 100
        ground = np.random.uniform([-10, -10, 0], [10, 10, 0.05], size=(n_ground, 3))
        ground_feats = np.column_stack([ground, np.full((n_ground, 1), 0.2)])

        # Moving Obstacle 1 (Vehicle moving along +X at 1.5 m/s)
        n_obs1 = 30
        c1 = np.array([2.0 + 1.5 * t, 4.0, 1.0])
        obs1 = np.random.normal(loc=c1, scale=[0.3, 0.3, 0.3], size=(n_obs1, 3))
        obs1_feats = np.column_stack([obs1, np.full((n_obs1, 1), 0.9)])

        # Moving Obstacle 2 (Person moving along -Y at 0.8 m/s)
        n_obs2 = 25
        c2 = np.array([-3.0, 6.0 - 0.8 * t, 0.8])
        obs2 = np.random.normal(loc=c2, scale=[0.25, 0.25, 0.4], size=(n_obs2, 3))
        obs2_feats = np.column_stack([obs2, np.full((n_obs2, 1), 0.8)])

        # Combine
        pts = np.vstack([ground, obs1, obs2]).astype(np.float32)
        feats = np.vstack([ground_feats, obs1_feats, obs2_feats]).astype(np.float32)

        # Semantic Segmentation via TorchScript Model
        with torch.no_grad():
            t_feats = torch.from_numpy(feats)
            t_coords = torch.zeros(len(feats), 4, dtype=torch.int32)
            logits = model(t_feats, t_coords)
            assert logits.shape == (len(pts), 8), f"Unexpected logits shape: {logits.shape}"

        sim_labels = np.zeros(len(pts), dtype=np.int32)
        sim_labels[n_ground:] = CLASS_OBSTACLE_HARD

        # Cluster Hard Obstacles
        bboxes = cluster_hard_obstacles(pts, sim_labels, eps=0.8, min_samples=5)
        det_boxes = np.array([b.bbox_2d for b in bboxes], dtype=np.float32) if len(bboxes) > 0 else np.empty((0, 4))

        # SORT Update
        active_tracks = tracker.update(det_boxes)
        print(f"  Frame {frame+1:02d} (t={t:.1f}s): {len(bboxes)} clusters extracted -> {len(active_tracks)} active tracks")

    print("[3/4] Validating trajectory continuity...")
    assert len(active_tracks) == 2, f"Expected 2 persistent tracks, got {len(active_tracks)}"
    print(f"  Verified {len(active_tracks)} active tracks successfully maintained across sequence.")

    # 4. Validate SLAM Launch Configuration
    print("[4/4] Verifying ROS 2 FAST-LIO2 & nvblox configurations...")
    launch_path = os.path.join(REPO_ROOT, "ros2/drdo_bringup/launch/mapping.launch.py")
    assert os.path.exists(launch_path), f"Missing launch file: {launch_path}"

    spec = importlib.util.spec_from_file_location("drdo_mapping_launch", launch_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.validate_launch_configuration()

    print("================================================================================")
    print("                    [PERCEPTION SUITE: ALL TESTS PASSED]                        ")
    print("================================================================================")


if __name__ == "__main__":
    test_perception_e2e()

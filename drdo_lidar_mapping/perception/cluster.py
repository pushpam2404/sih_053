#!/usr/bin/env python3
"""
DRDO ID26053 — Phase 5: Bounding Box Clustering for Hard Obstacles
Uses DBSCAN to cluster points labeled OBSTACLE_HARD (class 3) and extract 3D/2D bounding boxes.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Class taxonomy — aligned to the ADL-1 8-class scheme used by the segmentation remap and the
# C++ map engine. The previous private 5-class table set OBSTACLE_HARD = 3, which is
# VEGETATION_DENSE in ADL-1: the tracker clustered bushes and ignored rocks and vehicles.
CLASS_FREE = 0            # GROUND
CLASS_TRAVERSABLE = 1     # GRAVEL_DIRT
CLASS_OBSTACLE_SOFT = 3   # VEGETATION_DENSE
CLASS_OBSTACLE_HARD = 4   # OBSTACLE_HARD
CLASS_UNKNOWN = 6         # UNKNOWN


@dataclass
class BoundingBox3D:
    cluster_id: int
    centroid: np.ndarray      # [x, y, z]
    min_bound: np.ndarray     # [min_x, min_y, min_z]
    max_bound: np.ndarray     # [max_x, max_y, max_z]
    extent: np.ndarray        # [dx, dy, dz]
    num_points: int

    @property
    def bbox_2d(self) -> np.ndarray:
        """Returns [cx, cy, dx, dy] for 2D ground tracking."""
        return np.array([self.centroid[0], self.centroid[1], self.extent[0], self.extent[1]], dtype=np.float32)

    @property
    def corners_2d(self) -> np.ndarray:
        """Returns 4 corners in ground plane (x, y): [min_x, min_y, max_x, max_y]."""
        return np.array([self.min_bound[0], self.min_bound[1], self.max_bound[0], self.max_bound[1]], dtype=np.float32)


def _simple_dbscan(points: np.ndarray, eps: float = 0.8, min_samples: int = 5) -> np.ndarray:
    """Pure NumPy DBSCAN fallback if scikit-learn is not installed."""
    n = len(points)
    labels = np.full(n, -1, dtype=int)
    cluster_id = 0
    visited = np.zeros(n, dtype=bool)

    # Precompute pairwise distance squared
    diff = points[:, np.newaxis, :2] - points[np.newaxis, :, :2]
    dist_sq = np.sum(diff**2, axis=-1)
    eps_sq = eps * eps

    for i in range(n):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = np.where(dist_sq[i] <= eps_sq)[0]
        if len(neighbors) < min_samples:
            labels[i] = -1
        else:
            labels[i] = cluster_id
            queue = list(neighbors[neighbors != i])
            while queue:
                j = queue.pop(0)
                if not visited[j]:
                    visited[j] = True
                    j_neighbors = np.where(dist_sq[j] <= eps_sq)[0]
                    if len(j_neighbors) >= min_samples:
                        queue.extend([k for k in j_neighbors if not visited[k] and k not in queue])
                if labels[j] == -1:
                    labels[j] = cluster_id
            cluster_id += 1
    return labels


def cluster_hard_obstacles(points: np.ndarray,
                           labels: np.ndarray,
                           eps: float = 0.8,
                           min_samples: int = 5,
                           target_class: int = CLASS_OBSTACLE_HARD) -> List[BoundingBox3D]:
    """
    Extracts clusters and bounding boxes for points matching target_class.
    
    Args:
        points: (N, 3) or (N, >=3) numpy array of XYZ coordinates.
        labels: (N,) numpy array of semantic class IDs.
        eps: DBSCAN maximum neighborhood distance.
        min_samples: DBSCAN minimum points per cluster.
        target_class: semantic class to cluster (default: CLASS_OBSTACLE_HARD = 3).
        
    Returns:
        List of BoundingBox3D objects.
    """
    mask = (labels == target_class)
    obstacle_pts = points[mask, :3]
    if len(obstacle_pts) < min_samples:
        return []

    # Ground plane (x, y) clustering
    xy_pts = obstacle_pts[:, :2]
    if SKLEARN_AVAILABLE:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(xy_pts)
        cluster_labels = db.labels_
    else:
        cluster_labels = _simple_dbscan(xy_pts, eps=eps, min_samples=min_samples)

    unique_clusters = set(cluster_labels)
    unique_clusters.discard(-1)  # Remove noise label

    bboxes: List[BoundingBox3D] = []
    for cid in sorted(unique_clusters):
        c_mask = (cluster_labels == cid)
        c_pts = obstacle_pts[c_mask]
        
        min_bound = np.min(c_pts, axis=0)
        max_bound = np.max(c_pts, axis=0)
        centroid = np.mean(c_pts, axis=0)
        extent = max_bound - min_bound
        # Prevent 0 dimension
        extent = np.maximum(extent, 0.1)

        bboxes.append(BoundingBox3D(
            cluster_id=int(cid),
            centroid=centroid.astype(np.float32),
            min_bound=min_bound.astype(np.float32),
            max_bound=max_bound.astype(np.float32),
            extent=extent.astype(np.float32),
            num_points=len(c_pts)
        ))

    return bboxes


def main():
    print("Testing Step P5.1.1: Bounding Box Clustering via DBSCAN...")
    np.random.seed(42)

    # 1. Ground points (class 1)
    n_ground = 200
    ground_pts = np.random.uniform([-20, -20, 0.0], [20, 20, 0.1], size=(n_ground, 3))
    ground_labels = np.full(n_ground, CLASS_TRAVERSABLE, dtype=np.int32)

    # 2. Obstacle Cluster 1 (vehicle/rock at [5.0, 3.0, 1.0])
    c1_pts = np.random.normal(loc=[5.0, 3.0, 1.0], scale=[0.3, 0.3, 0.3], size=(35, 3))
    c1_labels = np.full(35, CLASS_OBSTACLE_HARD, dtype=np.int32)

    # 3. Obstacle Cluster 2 (tree/dynamic target at [-4.0, 8.0, 1.5])
    c2_pts = np.random.normal(loc=[-4.0, 8.0, 1.5], scale=[0.4, 0.4, 0.5], size=(45, 3))
    c2_labels = np.full(45, CLASS_OBSTACLE_HARD, dtype=np.int32)

    # 4. Obstacle Cluster 3 (at [12.0, -5.0, 0.5])
    c3_pts = np.random.normal(loc=[12.0, -5.0, 0.5], scale=[0.25, 0.25, 0.25], size=(25, 3))
    c3_labels = np.full(25, CLASS_OBSTACLE_HARD, dtype=np.int32)

    # 5. Sparse noise points (class 3)
    noise_pts = np.array([
        [0.0, 0.0, 0.5],
        [15.0, 15.0, 1.0],
        [-10.0, -10.0, 0.5]
    ], dtype=np.float32)
    noise_labels = np.full(len(noise_pts), CLASS_OBSTACLE_HARD, dtype=np.int32)

    # Combine
    all_pts = np.vstack([ground_pts, c1_pts, c2_pts, c3_pts, noise_pts])
    all_labels = np.concatenate([ground_labels, c1_labels, c2_labels, c3_labels, noise_labels])

    bboxes = cluster_hard_obstacles(all_pts, all_labels, eps=0.8, min_samples=5)
    print(f"Detected {len(bboxes)} obstacle clusters (expected: 3):")
    for b in bboxes:
        print(f"  Cluster {b.cluster_id}: Centroid=[{b.centroid[0]:.2f}, {b.centroid[1]:.2f}, {b.centroid[2]:.2f}], "
              f"Extent=[{b.extent[0]:.2f}, {b.extent[1]:.2f}, {b.extent[2]:.2f}], N={b.num_points}")

    assert len(bboxes) == 3, f"Expected 3 clusters, got {len(bboxes)}"
    
    # Check centroid proximities
    centroids = np.array([b.centroid for b in bboxes])
    expected_centroids = np.array([[5.0, 3.0, 1.0], [-4.0, 8.0, 1.5], [12.0, -5.0, 0.5]])
    for exp_c in expected_centroids:
        dists = np.linalg.norm(centroids - exp_c, axis=1)
        assert np.min(dists) < 0.3, f"Centroid {exp_c} not closely matched (min dist = {np.min(dists):.3f})"

    print("[STEP P5.1.1 COMPLETE]")


if __name__ == "__main__":
    main()

"""DRDO ID26053 — Perception & Dynamic Obstacle Tracking Subpackage
Provides DBSCAN clustering for obstacle extraction and SORT 3D Kalman Filter
multi-object tracking with Hungarian data association.
"""

from .classify import ClassifierThresholds, classify_box, classify_extent
from .cluster import (
    BoundingBox3D,
    cluster_hard_obstacles,
    CLASS_FREE,
    CLASS_TRAVERSABLE,
    CLASS_OBSTACLE_SOFT,
    CLASS_OBSTACLE_HARD,
    CLASS_UNKNOWN,
)
from .tracker import KalmanBoxTracker, SortTracker

__all__ = [
    "BoundingBox3D",
    "cluster_hard_obstacles",
    "KalmanBoxTracker",
    "SortTracker",
    "CLASS_FREE",
    "CLASS_TRAVERSABLE",
    "CLASS_OBSTACLE_SOFT",
    "CLASS_OBSTACLE_HARD",
    "CLASS_UNKNOWN",
    "classify_extent",
    "classify_box",
    "ClassifierThresholds",
]

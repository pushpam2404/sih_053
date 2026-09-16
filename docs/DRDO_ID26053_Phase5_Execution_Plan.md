# DRDO ID26053 — Phase 5 Agent Execution Plan
## SORT Tracking · TorchScript Export · nvblox 3D ESDF · FAST-LIO2 Loop Closure

> **Document Type:** Principal Systems Architect's Binding Directive — Phase 5
> **Prerequisite:** Phase 4 COMPLETE — MinkUNet fine-tuned, sliding window integrated.
> **Host (development):** macOS Apple Silicon M4 / Windows 10/11 x64 — Python 3.11
> **Target Deploy:** NVIDIA Jetson AGX Orin 64GB — CUDA 12.x — ROS2 Humble
> **Working Directory for ALL output files:** `~/Desktop/sih/phase5/`
> **Date:** September 2026

---

## Phase 4 Accomplishments (What Was Locked In)

| Deliverable | Status | Validation Gate |
|---|---|---|
| RELLIS-3D Data Pipeline | ✅ COMPLETE | `[STEP P4.1.2 COMPLETE]` |
| MinkUNet-18 Fine-Tuning | ✅ COMPLETE | `[STEP P4.2.1 COMPLETE]` |
| TorchSparse++ Profiling | ✅ COMPLETE | `[STEP P4.3.1 COMPLETE]` |
| Sliding Window Purge | ✅ COMPLETE | `[STEP P4.4.1 COMPLETE]` |
| ESDF 2D BFS Costmap | ✅ COMPLETE | `[STEP P4.6.1 COMPLETE]` |

**What Phase 4 did NOT do (intentionally):**
- Real-time instance tracking (no Kalman Filters for moving objects).
- TorchScript Export for C++ inference (Phase 4 used Python).
- Full 3D GPU-accelerated ESDF (Phase 4 used a CPU 2D BFS fallback).
- Global loop closure configuration (Phase 4 relied on localized FAST-LIO2 odometry).

**Phase 5 closes all these gaps, bridging the gap to full autonomous deployment.**

---

## ═══════════════════════════════════════════════════════════
## PART 1: ARCHITECTURAL DECISION LEDGER (ADL) — Phase 5
## ═══════════════════════════════════════════════════════════

### ADL-1: SORT Dynamic Tracking — LOCKED

Instead of simple transient clearing of obstacles (Phase 4), Phase 5 implements a lightweight SORT (Simple Online and Realtime Tracking) pipeline to track instances of `OBSTACLE_HARD`.
- **Clustering:** Extract connected components of `OBSTACLE_HARD` cells using DBSCAN (`eps=0.8, min_samples=5`).
- **State Estimation:** 2D Constant-Velocity Kalman Filter tracking $[x, y, dx, dy, v_x, v_y]^T$.
- **Data Association:** Hungarian Algorithm (`scipy.optimize.linear_sum_assignment`) matching predictions to detections via 2D Bounding Box IoU and Euclidean distance.

### ADL-2: TorchScript Export for 20 Hz Inference — LOCKED

MinkUNet-18 must be exported to TorchScript so that the C++ pipeline (or ROS2 node) can load it directly via `libtorch` without Python overhead. This enables 20+ Hz processing on the Jetson Orin.
- **Tracing:** Use `torch.jit.trace` on a wrapper module that handles sparse-to-dense conversions (or dense fallbacks on non-CUDA hosts).

### ADL-3: nvblox 3D ESDF Configuration — LOCKED

The 2D BFS ESDF implemented in Phase 4 is a fallback. Phase 5 configures the launch files to ingest the point cloud into `nvblox` (NVIDIA's GPU-accelerated TSDF/ESDF library).
- **Voxel Size:** 0.05m (matching MinkUNet resolution).
- **Max ESDF Distance:** 2.0m for local navigation.

### ADL-4: FAST-LIO2 Global Odometry — LOCKED

Activate the IMU-Lidar fusion loop closure in FAST-LIO2 by tuning the configuration parameters specifically for the Ouster OS1-64.

---

## ═══════════════════════════════════════════════════════════
## PART 2: PHASE 5 AGENT EXECUTION PLAN — MAV BLOCKS
## ═══════════════════════════════════════════════════════════

```
BLOCK GROUPS:
  P5.0 — Setup & Directory Tree
  P5.1 — SORT Dynamic Tracker Implementation
  P5.2 — TorchScript Export Pipeline
  P5.3 — nvblox & FAST-LIO2 Configuration
```

---

## BLOCK GROUP P5.0: SETUP

### Step P5.0.1: Create Phase 5 Directory Tree

**Objective:** Create the canonical `phase5/` directory hierarchy.

**Sub-Steps:**
- Create `phase5/src/` for any C++ logic.
- Create `phase5/python/` for tracker and export scripts.
- Create `phase5/ros2/launch/` and `phase5/ros2/config/` for ROS 2 launch integration.

**Execution Directive:**
```bash
mkdir -p ~/Desktop/sih/phase5/src
mkdir -p ~/Desktop/sih/phase5/python
mkdir -p ~/Desktop/sih/phase5/ros2/launch
mkdir -p ~/Desktop/sih/phase5/ros2/config
```

**Self-Validation Gate:**
```bash
[ -d ~/Desktop/sih/phase5/src ] && echo "[STEP P5.0.1 COMPLETE]" || exit 1
```

---

## BLOCK GROUP P5.1: SORT DYNAMIC TRACKING

### Step P5.1.1: Bounding Box Clustering

**Objective:** Write `phase5/python/cluster_obstacles.py` to cluster points labeled `OBSTACLE_HARD` using DBSCAN.

**Sub-Steps:**
- Import `sklearn.cluster.DBSCAN` (or write a pure numpy fallback).
- Filter input semantic point cloud for Class 3 (`OBSTACLE_HARD`).
- Apply DBSCAN on the $X,Y$ plane.
- Extract 3D bounding box dimensions, centroid, and 2D footprint.
- Build a self-test `main()` block with 3 synthetic clusters.

**Self-Validation Gate:**
Run the script and assert it prints `[STEP P5.1.1 COMPLETE]`.

### Step P5.1.2: Kalman Filter & SORT

**Objective:** Write `phase5/python/sort_tracker.py` incorporating Kalman Filter tracking across frames.

**Sub-Steps:**
- Implement `KalmanBoxTracker` class with Constant Velocity state matrix.
- Implement Hungarian data association (IoU / Euclidean dist).
- Implement `SortTracker` managing lifecycle (hits, age, active tracks).
- Build a simulation `main()` that tests tracking over 10 consecutive frames.

**Self-Validation Gate:**
Run the script and assert it successfully prints `[STEP P5.1.2 COMPLETE]`.

---

## BLOCK GROUP P5.2: TORCHSCRIPT EXPORT

### Step P5.2.1: Export MinkUNet-18

**Objective:** Write `phase5/python/export_torchscript.py` to trace the TorchSparse model.

**Sub-Steps:**
- Load Phase 4 `.pth` weights.
- Create a `torch.nn.Module` wrapper handling `SparseTensor` conversion.
- Execute `torch.jit.trace` using dummy inputs.
- Save to `phase5/models/minkunet18_traced.pt`.
- Run an inference loop to benchmark deserialization and latency (<5ms target).

**Self-Validation Gate:**
Run the script and assert it successfully prints `[STEP P5.2.1 COMPLETE]`.

---

## BLOCK GROUP P5.3: NVBLOX & FAST-LIO2

### Step P5.3.1: ROS 2 Launch Configurations

**Objective:** Create complete ROS 2 launch integration for the production mapping stack.

**Sub-Steps:**
- Write `phase5/ros2/config/fast_lio_ouster64.yaml` (10 Hz, 64-beam, IMU-fusion).
- Write `phase5/ros2/config/nvblox_params.yaml` (0.05m voxel size, ESDF 3D mode).
- Write `phase5/ros2/launch/drdo_mapping.launch.py` to instantiate `fast_lio_mapping`, `nvblox_node`, and `grid_map_node` with appropriate topic remappings.
- Include Python AST syntactic validation block for offline host environments.

**Self-Validation Gate:**
Run the launch file directly as a python script to validate AST and configuration syntax. It must print `[STEP P5.3.1 COMPLETE]`.

---
*Document sealed by Principal Systems Architect — DRDO ID26053 SIH 2026*

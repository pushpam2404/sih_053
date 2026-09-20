# DRDO Problem Statement ID 26053: Requirements vs. Our Solution

## 1. The problem statement

- **Title:** Adaptive Variable Resolution 2.5D Lidar Mapping for Dynamic Environment Perception
- **Organisation:** DRDO (Department of Defence R&D) · **Category:** Software · **Theme:** Smart Vehicles
- **Core idea:** raw 3D LiDAR is too heavy to process in real time, and a plain 2D grid loses heights (curbs, potholes, overhangs). Build a *foveated* 2.5D map: fine cells near the vehicle, coarse cells far away, with no alignment errors or data loss between resolutions, and handle moving objects.

Status legend: ✅ done and tested on the host · 🟡 done, not yet run on the vehicle/Orin · ⏸ deliberately deferred.

## 2. Scope of this round — read before the numbers below

**No number in this document, or in `docs/PERFORMANCE.md`, comes from an NVIDIA Jetson AGX Orin.**
Testing on the target board was out of scope for this round: the team had no access to Orin
hardware, an Ouster OS1-64, or a vehicle. Concretely:

- Every latency, memory and accuracy figure is Apple M4 CPU (macOS, single-threaded), on synthetic
  OS1-64-shaped scans.
- The CUDA kernels (`cuda/drdo_cuda_kernels.cu`) mirror the C++ engine but have **never been
  compiled** — there is no `nvcc` on the dev host.
- The ROS 2 nodes are structurally validated (`validate_launch_configuration()`, message wiring,
  frame contracts) without ROS 2 installed, but have **never run on a robot**.

The architecture targets the Orin throughout (the CUDA port, the TensorRT path and the Jetson
Dockerfile all exist), but treat every Orin statement anywhere in this repository as an intention,
not a measurement. `docs/JETSON_DEPLOYMENT.md` lists the first measurements to take once a board is
available. This is honest scoping, not a gap in the software: everything below is fully reproducible
on the host by the commands in `docs/PERFORMANCE.md` §9.

## 3. Requirement matrix

| # | Requirement (statement wording) | Our solution | Evidence | Status |
|---|---|---|---|---|
| 1 | **Terrain analysis:** drivable vs non-drivable | Per-cell height step, clearance and roughness give flat / rough / lethal step / overhang / **negative (pothole)**, with a traversability score. Free space is carved by ray-casting; stale obstacles and dust decay. The negative-obstacle check was a real bug fix, not a pre-existing feature: before it, a measured 40 cm pothole classified as FREE with traversability 1.00 — the engine told the planner to drive into it. It is now NEGATIVE, traversability 0, with a corroboration guard (`NEG_MIN_HITS`) giving 0.000% false positives on flat and rough terrain and 0.036% on ±15 cm noise (noise as deep as the detection threshold), and 0 of 8,404 cells misflagged on 2–4% downgrades. | `src/drdo_map.cpp` (`classify_obstacle`), regression gates 2–4; PERFORMANCE.md §5 | ✅ flat/rough ground and potholes · slopes and trenches beyond the rim test are next (§5) |
| 2 | **Object detection & classification:** static obstacles (walls, poles) and dynamic objects (pedestrians, other vehicles) | Detection — static: lethal cells in the map; dynamic: `perception/dynamic.py` confirms an object as moving only when, for several consecutive frames, it exceeds a speed floor **and** its whole footprint shifts together **and** the cells it vacated ~1 s ago are now empty — then removes it from the static map and publishes it. Classification — `perception/classify.py` labels pedestrian / vehicle / pole / wall / structure geometrically (footprint + height), no network. | `tests/python/test_dynamic.py`; dashboard recall 67–100% by range/occlusion, **0 static objects reported moving** across both speeds tested (PERFORMANCE.md §4); classification ~100% on the synthetic scene, with the edge-on-vehicle and person-vs-pole caveats stated in PERFORMANCE.md §6 | ✅ host · 🟡 ROS node |
| 3 | **Deep-learning model** (PointNet++ / sparse CNN) | Deliberately not delivered this round, and no accuracy is claimed. The full range-image pipeline is built and verified end to end — spherical projection, kNN label transfer (incl. occluded-point recovery), a SalsaNext-Lite training loop, ONNX export with 1.19e-07 torch/ONNX parity, a ROS producer node, and `kitti_replay.py` proving 27,621 real labels reaching the C++ engine (a first for this codebase). No model is *trained*: real RELLIS-3D could not be obtained (the 14 GB archive is behind a Google Drive quota block with no mirror). The shipped `models/*` are placeholders — `scripts/eval.py` measures them at 5.01% mIoU against a 6.64% random baseline, worse than guessing, and prints a warning saying so. The map and ROS graph already carry a per-point `label` field end-to-end, so a trained network drops in with no architectural change. | `drdo_lidar_mapping/segmentation/`, `drdo_lidar_mapping/inference/range_segmenter.py`, `tests/python/test_segmentation.py`, `scripts/kitti_replay.py`; PERFORMANCE.md §8 | ⏸ pipeline built, deliberately untrained |
| 4 | **Variable-resolution grid:** 5 cm within 10 m, coarser (50 cm) to 100 m | 5 cm ≤ 10 m, 10 cm ≤ 25 m, 50 cm ≤ 100 m | `drdo_map.h`, `test_grid_engine` | ✅ |
| 5 | **No alignment errors** in the 3D → 2.5D projection | All levels derive from one 5 cm lattice by integer division (1 : 2 : 10); every fine cell sits inside exactly one coarse cell | Gate 6: 0 mismatches over 4M sampled positions. The previous 5/20/50/100 cm pyramid had 20% of 20 cm cells straddling 50 cm edges. | ✅ |
| 6 | **No data loss** | A fixed pool with band-aware eviction and zero dropped inserts; the query returns the freshest level; evicted fine cells stay covered by the coarser level | 3.3 km at 40 km/h: 0 dropped points, peak load 0.29 | ✅ |
| 7 | **Real-time visualisation dashboard** with distinct colours for terrain and objects, demonstrating memory reduction | Live: `/drdo/map_image` (terrain or elevation colours) + `/drdo/dynamic_obstacles` markers in RViz. Offline: `reports/map_dashboard.html` with layers, band rings, moving objects, classification confusion matrix, memory and latency, incl. the vs-uniform-grid comparison. | `scripts/map_dashboard.py`, `rviz/drdo_map.rviz` | ✅ dashboard · 🟡 RViz |
| 8 | **Significant memory reduction** vs a uniform high-resolution 3D map | 18.3 MB in live use vs 479 MB for a uniform 5 cm 2.5D grid (**26× smaller**) and 3,835 MB for a uniform 5 cm 3D voxel grid (**210× smaller**), same 100 m radius, measured from the live pool rather than a formula | PERFORMANCE.md §2 | ✅ |
| 9 | **Low latency / high FPS** | Full per-frame pipeline (mover detection + insert + classify + decay + purge + rasterise), 65,536-return synthetic OS1-64 scans: **51.8 ms mean / 60.2 ms p95 / 68.0 ms max → 19.3 Hz**, clearing the 10 Hz sensor budget on mean, p95 and max. The rasterise stage (27 ms, over half the budget) is a pessimistic Python approximation of the compiled `grid_map_node.cpp`, so real throughput is very likely better — no better figure is claimed, since the compiled node has never been profiled. | `scripts/benchmark.py`; PERFORMANCE.md §3 | ✅ host · 🟡 Orin numbers pending |
| 10 | **Accuracy across varying distances** | Moving-object recall by range, occlusion modelled: 100% / 86% / 67% (0–10 / 10–25 / 25–40 m) at 5 m/s; 74% / 89% / 82% at 11.1 m/s. Object classification is inline at 1.07 µs/call. Segmentation-by-range metrics exist in `scripts/eval.py` for when a network is trained. | PERFORMANCE.md §4, §6 | ✅ synthetic scene, occlusion-aware |

## 4. Off-road challenges we also address

| Challenge | Approach | Location |
|---|---|---|
| Dust and exhaust clouds | Cells not re-observed decay; their height statistics restart when seen again, so a passed dust cloud clears | `src/drdo_map.cpp` |
| Overhanging branches and wires | Clearance above ground > 0.5 m is passable (shown teal in the colour view) | `classify_obstacle` |
| Vibration and GPS denial | FAST-LIO2 LiDAR-inertial odometry (Ouster config, `lidar_type: 3`) | `ros2/drdo_bringup/config/fast_lio_ouster64.yaml` |
| Vehicle-feasible paths | Nav2 SmacPlannerHybrid + MPPI on our `/map` | `ros2/drdo_bringup/config/nav2_params.yaml` |
| Moving people and vehicles smearing the map | Moving-object filter upstream of the map | `ros2/drdo_perception/` |
| Potholes | Negative-obstacle detection with a rim test (slope-safe) and a corroboration guard (noise-safe) | `src/drdo_map.cpp` (`classify_obstacle`); PERFORMANCE.md §5 |

## 5. Known gaps and next steps

1. Run the full ROS graph on the Orin with a recorded Ouster bag and publish measured latency — the scope statement above is the honest position until that happens.
2. Local ground estimation for slopes; deeper trench detection from LiDAR shadows (the current rim test catches potholes and rejects graded slopes, but does not yet use shadow/occlusion geometry for trenches).
3. Height-aware free-space carving, so a ray passing over a rock does not downgrade it.
4. Occlusion is now modelled in the test scene (25–40 m recall drops from 89% to 67% as a direct result), but confirmation still takes ~1 s; tune it for the vehicle's speed on real data.
5. Object classification is measured on canonical-sized synthetic objects; validate against edge-on vehicles and person-vs-pole ambiguity before trusting it operationally (PERFORMANCE.md §6).
6. If selected: train the already-built segmentation pipeline (SalsaNext-Lite, range-image projection, ONNX export) on real RELLIS-3D or SemanticKITTI once obtainable — no architectural work remains, only data and training compute.

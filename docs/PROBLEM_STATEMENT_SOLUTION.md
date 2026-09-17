# DRDO Problem Statement ID 26053: Requirements vs. Our Solution

## 1. The problem statement

- **Title:** Adaptive Variable Resolution 2.5D Lidar Mapping for Dynamic Environment Perception
- **Organisation:** DRDO (Department of Defence R&D) · **Category:** Software · **Theme:** Smart Vehicles
- **Core idea:** raw 3D LiDAR is too heavy to process in real time, and a plain 2D grid loses heights (curbs, potholes, overhangs). Build a *foveated* 2.5D map: fine cells near the vehicle, coarse cells far away, with no alignment errors or data loss between resolutions, and handle moving objects.

Status legend: ✅ done and tested on the host · 🟡 done, not yet run on the vehicle/Orin · ⏸ deliberately deferred.

## 2. Requirement matrix

| # | Requirement (statement wording) | Our solution | Evidence | Status |
|---|---|---|---|---|
| 1 | **Terrain analysis:** drivable vs non-drivable | Per-cell height step, clearance and roughness give flat / rough / lethal step / overhang, with a traversability score. Free space is carved by ray-casting; stale obstacles and dust decay. | `src/drdo_map.cpp`, regression gates 2–4 | ✅ flat ground · slopes and trenches are next (§4) |
| 2 | **Object detection:** static obstacles and dynamic objects (pedestrians, vehicles) | Static: lethal cells in the map. Dynamic: `perception/dynamic.py` confirms an object as moving when its whole footprint shifts **and** the space it left is empty, then removes it from the static map and publishes it. | `tests/python/test_dynamic.py`; dashboard: 89–98% recall, 0 static objects reported moving | ✅ host · 🟡 ROS node |
| 3 | **Deep-learning model** (PointNet++ / sparse CNN) | Deferred for the hackathon round. The map and ROS graph already accept per-point labels (`label` field), so a network plugs in without architecture changes. Current model files are placeholders at chance level (`scripts/eval.py`). | — | ⏸ |
| 4 | **Variable-resolution grid:** 5 cm within 10 m, coarser (50 cm) to 100 m | 5 cm ≤ 10 m, 10 cm ≤ 25 m, 50 cm ≤ 100 m | `drdo_map.h`, `test_grid_engine` | ✅ |
| 5 | **No alignment errors** in the 3D → 2.5D projection | All levels derive from one 5 cm lattice by integer division (1 : 2 : 10); every fine cell sits inside exactly one coarse cell | Gate 6: 0 mismatches over 4M positions. The previous 5/20/50/100 cm pyramid had 20% of 20 cm cells straddling 50 cm edges. | ✅ |
| 6 | **No data loss** | A fixed pool with band-aware eviction and zero dropped inserts; the query returns the freshest level; evicted fine cells stay covered by the coarser level | 3.3 km at 40 km/h: 0 dropped points, peak load 0.29 | ✅ |
| 7 | **Real-time visualisation dashboard** with distinct colours for terrain and objects | Live: `/drdo/map_image` (terrain or elevation colours) + `/drdo/dynamic_obstacles` markers in RViz. Offline: `reports/map_dashboard.html` with layers, band rings, moving objects, memory and latency. | `scripts/map_dashboard.py`, `rviz/drdo_map.rviz` | ✅ dashboard · 🟡 RViz |
| 8 | **Significant memory reduction** vs a uniform high-resolution 3D map | 16–18 MB live vs 3,835 MB for uniform 5 cm voxels (~210×) and 479 MB for a uniform 5 cm 2.5D grid (~26×), same 100 m radius | `test_engine_regression` `[MEMORY]` | ✅ |
| 9 | **Low latency / high FPS** | ~14–20 ms map update per scan on one CPU thread (Apple M4); ~3.5 ms moving-object detection | regression test, dashboard | ✅ host · 🟡 Orin numbers pending |
| 10 | **Accuracy across varying distances** | Moving-object recall by range: 98% / 95% / 89% (0–10 / 10–25 / 25–40 m) at 5 m/s; 95% / 93% / 71% at 11.1 m/s. Segmentation metrics by range exist in `scripts/eval.py` for when a network is trained. | dashboard | ✅ synthetic scene |

## 3. Off-road challenges we also address

| Challenge | Approach | Location |
|---|---|---|
| Dust and exhaust clouds | Cells not re-observed decay; their height statistics restart when seen again, so a passed dust cloud clears | `src/drdo_map.cpp` |
| Overhanging branches and wires | Clearance above ground > 0.5 m is passable (shown teal in the colour view) | `classify_obstacle` |
| Vibration and GPS denial | FAST-LIO2 LiDAR-inertial odometry (Ouster config, `lidar_type: 3`) | `ros2/drdo_bringup/config/fast_lio_ouster64.yaml` |
| Vehicle-feasible paths | Nav2 SmacPlannerHybrid + MPPI on our `/map` | `ros2/drdo_bringup/config/nav2_params.yaml` |
| Moving people and vehicles smearing the map | Moving-object filter upstream of the map | `ros2/drdo_perception/` |

## 4. Known gaps and next steps

1. Run the full ROS graph on the Orin with a recorded Ouster bag and publish measured latency.
2. Local ground estimation for slopes; trench (negative obstacle) detection from height drops and LiDAR shadows.
3. Height-aware free-space carving, so a ray passing over a rock does not downgrade it.
4. Moving-object confirmation takes about 1 s; tune it for the vehicle's speed and add occlusion handling on real data.
5. If selected: a segmentation network (range-image or sparse CNN) trained on RELLIS-3D / SemanticKITTI, feeding the existing label path.

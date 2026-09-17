# DRDO ID26053: Adaptive Variable-Resolution 2.5D LiDAR Mapping for Dynamic Environments

[![ROS 2 Humble](https://img.shields.io/badge/ROS%202-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![Target Hardware](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20AGX%20Orin-green.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-17-brightgreen.svg)](https://en.cppreference.com/w/cpp/17)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](LICENSE)

Smart India Hackathon 2026 · DRDO · Smart Vehicles. Target: **NVIDIA Jetson AGX Orin** + **Ouster OS1-64**, ROS 2 Humble.

**Demo:** open [`reports/map_dashboard.html`](reports/map_dashboard.html) in a browser (regenerate with `python3 scripts/map_dashboard.py`).

---

## 1. The problem in one paragraph

A 3D LiDAR produces over a million points a second. Keeping them all as a 3D map is too slow and too big for a vehicle computer. Flattening them into an ordinary 2D map loses curbs, potholes and overhangs. The problem statement asks for a **foveated** map: very fine detail close to the vehicle, where safety depends on it, and coarser detail far away. The map must also avoid alignment errors between resolutions and cope with **moving** objects.

## 2. What we built

| # | Capability | How | Where |
|---|---|---|---|
| 1 | **Foveated 2.5D map**: 5 cm cells within 10 m, 10 cm to 25 m, 50 cm to 100 m | One preallocated spatial-hash pool of 40-byte cells. Each cell keeps min/max/mean height and roughness (online variance). | `include/drdo_lidar_mapping/drdo_map.h`, `src/drdo_map.cpp` |
| 2 | **Zero alignment error between resolutions** | Every level is derived from one 5 cm lattice by integer division (ratios 1 : 2 : 10), so each fine cell sits inside exactly one coarse cell. Tested on 4M positions with 0 mismatches. | engine, `tests/cpp/test_engine_regression.cpp` (gate 6) |
| 3 | **Bounded memory at any distance travelled** | Cells are evicted once they fall behind their own band; the query always returns the freshest level. 3.3 km at 40 km/h with 0 dropped points. | engine, regression drive test |
| 4 | **Terrain analysis without a trained network** | Height step, clearance and roughness per cell give flat, rough, lethal step or overhang. Free space is carved by ray-casting. Dust and stale obstacles decay. | engine |
| 5 | **Moving-object detection that keeps the static map clean** | Clusters above the ground are tracked, and an object is reported as moving only when its *whole footprint* shifts **and** the space it left is empty. Movers are removed before mapping. | `drdo_lidar_mapping/perception/dynamic.py`, `ros2/drdo_perception/` |
| 6 | **Live colour-coded view** | The grid node publishes `/drdo/map_image` (terrain or elevation colours) next to the Nav2 costmap `/map`; moving objects appear as RViz markers. | `ros2/drdo_grid_map/`, `rviz/drdo_map.rviz` |
| 7 | **Offline dashboard** | Drives the real C++ engine through a synthetic off-road scene: map layers, resolution bands, moving objects, memory versus uniform maps, latency, detection by range. | `scripts/map_dashboard.py` |
| 8 | **Navigation hookup** | FAST-LIO2 odometry → our map → Nav2 (SmacPlannerHybrid + MPPI). | `ros2/drdo_bringup/` |

## 3. What is different from existing tools

We compared the design against the open-source mapping stacks a DRDO reviewer is likely to know.

| Existing approach | What it does well | Gap for this problem | Our answer |
|---|---|---|---|
| **elevation_mapping / elevation_mapping_cupy** (ETH) | Accurate robot-centric elevation maps, GPU version | One resolution over a fixed window: 5 cm to 100 m would need a 4000 × 4000 grid | Three nested resolutions in one pool; 5 cm near, 50 cm far, ~16–18 MB live |
| **OctoMap / nvblox** | Full 3D occupancy / TSDF | Uniform voxels; a 5 cm 3D map over 100 m is ~3.8 GB even at 1 byte per voxel | 2.5D cells carry the height information a ground vehicle needs, at **~210× less memory** (measured) |
| **Nav2 voxel / STVL layers** | Obstacle marking with time decay | No elevation, roughness or overhang reasoning | Per-cell height statistics → traversability cost |
| **Offline dynamic-point removal** (Removert, ERASOR) | Clean maps after the drive | Not real-time | Online, per scan, geometry-only |
| **Plain cluster + speed tracking** (DBSCAN + SORT) | Simple | Reports parked cars, trees and walls as moving when the viewing angle changes (**69 of 337** "moving" reports in our scene were static) | Whole-footprint shift + vacated-space test: **0 of 298** in the same scene, with 89–98% recall |

The combination is what we believe is new: a **distance-foveated, exactly nested 2.5D map with fixed memory**, fed by a **viewpoint-robust moving-object filter**, and running without a trained network. Online dynamic-object detection from free-space evidence also exists in research (for example Dynablox, which uses 3D voxel maps); ours works on 2.5D clusters so it fits the same small compute budget.

## 4. Measured results (honest status)

Numbers from this repository's own tests on an **Apple M4 laptop, single CPU thread, synthetic Ouster OS1-64 scans**. **Jetson Orin numbers are still to be measured.**

| Metric | Result | Source |
|---|---|---|
| Map update per scan (87k points, full range) | ~20 ms mean | `test_engine_regression` |
| Map update per scan (56k points, dashboard scene) | ~14 ms mean | `map_dashboard.py` |
| Memory in use vs uniform maps (100 m radius) | 16–18 MB vs 479 MB (uniform 5 cm 2.5D) vs 3,835 MB (uniform 5 cm 3D) | both |
| Dropped points over a 3.3 km drive at 40 km/h | 0 (peak pool load 0.29) | `test_engine_regression` |
| Cross-resolution alignment errors | 0 over 4M positions | gate 6 |
| Moving-object recall at 5 m/s (0–10 / 10–25 / 25–40 m) | 98% / 95% / 89% | `map_dashboard.py` |
| Moving-object recall at 11.1 m/s | 95% / 93% / 71% | `map_dashboard.py --speed 11.1` |
| Static objects reported as moving | 0 at both speeds | `map_dashboard.py`, `test_dynamic.py` |
| Time to confirm a mover | 0.9–1.9 s | `map_dashboard.py` |
| Moving-object detection cost | ~3.5 ms per scan (Python) | `map_dashboard.py` |

**Limitations we state up front:**
- The synthetic scene has no occlusion.
- Negative obstacles (trenches) and slopes need local-ground estimation, which is not done yet.
- A mover must be tracked for about a second before it is confirmed.
- The ROS nodes are compile-checked but have not yet run on a vehicle.

**Deep learning is future work.** The problem statement names a segmentation network. We deliberately parked it for the hackathon round: everything above runs on geometry. The map and the ROS graph already accept per-point labels (a `label` field on the cloud), so a network can be added later without changing the architecture. The model files in `models/` are placeholders; `scripts/eval.py` shows they score at chance level, and no accuracy is claimed.

## 5. Data flow

```
Ouster OS1-64 ──► FAST-LIO2 ──► /cloud_registered
                                      │
                                      ▼
                         drdo_perception/dynamic_obstacle_node ──► /drdo/dynamic_obstacles  (RViz markers)
                                      │  scan minus confirmed movers
                                      ▼
                               /drdo/cloud_static
                                      │
                                      ▼
                      drdo_grid_map/grid_map_node  (foveated 2.5D engine)
                           │                         │
                     /map (costmap)          /drdo/map_image (colour view)
                           │
                           ▼
            Nav2: SmacPlannerHybrid + MPPI ──► /cmd_vel
```

Launch argument `filter_moving_objects:=false` feeds the grid node straight from `/cloud_registered`. If the moving-object node has no pose, it passes scans through unfiltered, so the map never starves.

## 6. Quick start (laptop, no ROS needed)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# C++ engine tests
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build/test_grid_engine
./build/test_engine_regression 11.1 300        # regression gates + short drive (full: no arguments, ~1 min)

# Python module used by the dashboard and bridge test (copy from build/ or build directly)
clang++ -O3 -std=c++17 -shared -fPIC -undefined dynamic_lookup $(python3 -m pybind11 --includes) -Iinclude \
  src/drdo_map.cpp src/drdo_map_py.cpp -o lib/drdo_map$(python3 -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Python tests
python3 tests/python/test_dynamic.py        # moving-object detector
python3 tests/python/test_map_bridge.py     # C++ bridge
python3 tests/python/test_perception.py     # clustering/tracking + launch-file validation
python3 tests/python/test_deployment.py     # config and artifact checks

# Dashboard
python3 scripts/map_dashboard.py                  # -> reports/map_dashboard.html
python3 scripts/map_dashboard.py --no-labels      # map built from geometry only, as the live system runs
python3 scripts/map_dashboard.py --speed 11.1     # 40 km/h
```

## 7. ROS 2 (Linux / Jetson)

```bash
source /opt/ros/humble/setup.bash
pip install -e .                                   # drdo_perception imports the Python package
colcon build --packages-select drdo_grid_map drdo_perception drdo_bringup --symlink-install
source install/setup.bash

ros2 launch drdo_bringup mapping.launch.py points_topic:=/ouster/points imu_topic:=/ouster/imu
ros2 launch drdo_bringup navigation.launch.py      # mapping + Nav2
rviz2 -d rviz/drdo_map.rviz                        # costmap, colour map image, moving objects
```

Useful launch arguments: `filter_moving_objects` (default `true`), `map_color_mode` (`terrain` | `elevation`), `map_resolution` (published costmap cell size, default 0.20 m), `enable_nvblox` (default `false`).

**Colour view legend (`terrain`):** green = flat, amber = rough/uncertain, red = lethal step or obstacle, teal = overhang (passable underneath), blue = water/mud (needs labels), dark green = observed free space, half brightness = not seen for more than 5 s.

`fast_lio` and `ouster-ros` must be built from source in the same workspace. The Docker base image needs a decision (JetPack 6 / Ubuntu 22.04 recommended); see `deploy/docker/Dockerfile.jetson`.

## 8. Repository layout

```
include/drdo_lidar_mapping/drdo_map.h   engine constants, 40-byte GridCell, API
src/drdo_map.cpp                        engine: hashing, foveation, carving, classification, decay, purge
src/drdo_map_py.cpp                     pybind11 module (insert_points, export_cells, ...)
cuda/drdo_cuda_kernels.cu               GPU port of the insert/classify loops (not yet built on the Orin)
drdo_lidar_mapping/perception/          dynamic.py (moving-object detector), cluster.py, tracker.py
drdo_lidar_mapping/segmentation|inference/   future deep-learning path (placeholders)
ros2/drdo_grid_map/                     C++ map node: /map + /drdo/map_image
ros2/drdo_perception/                   Python moving-object node
ros2/drdo_bringup/                      launch files, FAST-LIO2 / Nav2 / nvblox configs
scripts/map_dashboard.py                offline dashboard generator
scripts/eval.py                         segmentation evaluation with baselines and range bins
tests/cpp, tests/python                 unit, regression and integration tests
reports/map_dashboard.html              generated dashboard
docs/                                   architecture, problem-statement mapping, setup, Jetson notes
rviz/drdo_map.rviz                      RViz layout
```

## 9. Roadmap

1. **On the Orin:** build, run the regression test and the ROS graph on a recorded Ouster bag, and publish real latency numbers.
2. **Map quality:** local ground estimation (slopes), trench detection, and height-aware free-space carving.
3. **Speed and thermal governor:** shrink the fine-resolution radius and the scanned sector when the board runs hot or the vehicle is slow.
4. **If selected:** a range-image segmentation network trained on RELLIS-3D / SemanticKITTI, feeding the existing `label` field.

Developed for the **Smart India Hackathon 2026, DRDO Problem ID 26053**.

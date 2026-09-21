# DRDO ID26053: Adaptive Variable-Resolution 2.5D LiDAR Mapping for Dynamic Environments

[![ROS 2 Humble](https://img.shields.io/badge/ROS%202-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![Target Hardware](https://img.shields.io/badge/Target-NVIDIA%20Jetson%20AGX%20Orin-green.svg)](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems/jetson-orin/)
[![C++ Standard](https://img.shields.io/badge/C%2B%2B-17-brightgreen.svg)](https://en.cppreference.com/w/cpp/17)
[![License](https://img.shields.io/badge/License-Apache%202.0-lightgrey.svg)](LICENSE)

Smart India Hackathon 2026 · DRDO · Smart Vehicles. Target platform: **NVIDIA Jetson AGX Orin** + **Ouster OS1-64**, ROS 2 Humble.

**Demo:** open [`reports/map_dashboard.html`](reports/map_dashboard.html) in a browser (regenerate with `.venv/bin/python scripts/map_dashboard.py`).
**All measured numbers with methodology:** [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

---

## Scope of testing — read this first

**We did not test on Jetson AGX Orin hardware. It was out of scope for this round.** The team had no access to an Orin board, an Ouster OS1-64, or a vehicle. Every number in this repository is CPU latency on an **Apple M4 laptop, single-threaded**, against **synthetic OS1-64 scans**.

That means, concretely:

- The CUDA kernels exist but have **never been compiled** — there is no `nvcc` on the development host.
- The ROS 2 nodes are structurally validated (the launch files self-check without ROS installed) but have **never run on a robot**.
- No accuracy claim is made against real field data.

The architecture targets the Orin and the deployment path is built out (CUDA port, TensorRT script, Docker image). Treat every Orin statement here as an **intention, not a measurement**. [`docs/JETSON_DEPLOYMENT.md`](docs/JETSON_DEPLOYMENT.md) lists the first measurements to take when a board is available.

We would rather hand a reviewer numbers that are smaller and true than numbers that are larger and unverifiable.

## 1. The problem in one paragraph

A 3D LiDAR produces over a million points a second. Keeping them all as a 3D map is too slow and too big for a vehicle computer. Flattening them into an ordinary 2D map loses curbs, potholes and overhangs. The problem statement asks for a **foveated** map: very fine detail close to the vehicle, where safety depends on it, and coarser detail far away. The map must also avoid alignment errors between resolutions and cope with **moving** objects.

## 2. What we built

| # | Capability | How | Where |
|---|---|---|---|
| 1 | **Foveated 2.5D map** — 5 cm cells within 10 m, 10 cm to 25 m, 50 cm to 100 m | One preallocated spatial-hash pool of 40-byte cells. Each cell keeps min/max/mean height and roughness (online Welford variance). | `include/drdo_lidar_mapping/drdo_map.h`, `src/drdo_map.cpp` |
| 2 | **Zero alignment error between resolutions** | Every level derives from one 5 cm lattice by integer division (1 : 2 : 10), so each fine cell sits inside exactly one coarse cell. **0 mismatches over 4M positions.** | engine, regression GATE 6 |
| 3 | **Bounded memory at any distance travelled** | Band-aware eviction; the query returns the freshest level. **3.3 km at 40 km/h, 0 dropped points, load plateaus at 0.229.** | engine, drive test |
| 4 | **Terrain analysis without a trained network** | Per-cell height step, clearance and roughness give flat / rough / lethal step / overhang. Free space is carved by ray-casting; dust and stale obstacles decay. | `classify_obstacle` |
| 5 | **Negative obstacles — potholes and trenches** | Cells whose returns sit below ground with an intact rim nearby. **Fixes a real safety bug: a 40 cm pothole previously classified FREE at traversability 1.00.** | `classify_obstacle`, `has_ground_rim` |
| 6 | **Moving-object detection that keeps the static map clean** | An object is reported moving only when its *whole footprint* shifts **and** the space it left is empty. **0 static objects misreported, at both 18 and 40 km/h.** | `perception/dynamic.py` |
| 7 | **Object classification** — pedestrian / vehicle / pole / wall | Geometric decision tree over footprint and height. No network. | `perception/classify.py` |
| 8 | **Live colour-coded view** | `/drdo/map_image` (terrain or elevation colours) beside the Nav2 costmap `/map`; classified objects as per-class coloured RViz markers. | `ros2/drdo_grid_map/`, `rviz/drdo_map.rviz` |
| 9 | **Offline dashboard with occlusion** | Drives the real C++ engine through a synthetic off-road scene: map layers, resolution bands, moving objects, classification accuracy by range, memory, latency. | `scripts/map_dashboard.py` |
| 10 | **Navigation hookup** | FAST-LIO2 odometry → our map → Nav2 (SmacPlannerHybrid + MPPI). | `ros2/drdo_bringup/` |

## 3. What is different from existing tools

| Existing approach | What it does well | Gap for this problem | Our answer |
|---|---|---|---|
| **elevation_mapping / _cupy** (ETH) | Accurate robot-centric elevation maps, GPU version | One resolution over a fixed window: 5 cm to 100 m needs a 4000 × 4000 grid | Three nested resolutions in one pool; 18.3 MB live |
| **OctoMap / nvblox** | Full 3D occupancy / TSDF | Uniform voxels; 5 cm over 100 m is ~3.8 GB at 1 byte/voxel | 2.5D cells carry the height a ground vehicle needs, at a fraction of the memory (see §4 — the fair number is closer to a uniform 20 cm map's footprint, not the 210× figure against a different data structure) |
| **Nav2 voxel / STVL layers** | Obstacle marking with time decay | No elevation, roughness or overhang reasoning; no negative obstacles | Per-cell height statistics → traversability cost, including potholes |
| **Offline dynamic-point removal** (Removert, ERASOR) | Clean maps after the drive | Not real-time | Online, per scan, geometry-only |
| **Plain cluster + speed tracking** (DBSCAN + SORT) | Simple | Reports parked cars, trees and walls as moving when the viewing angle changes — **69 of 337** "moving" reports were static in our scene | Whole-footprint shift + vacated-space test: **0 of 286**, with occlusion modelled |

The combination is what we believe is new: a **distance-foveated, exactly nested 2.5D map with fixed memory**, fed by a **viewpoint-robust moving-object filter**, detecting **negative as well as positive obstacles**, and running without a trained network.

## 4. Measured results

Apple M4, single CPU thread, synthetic OS1-64 scans. Full methodology in [`docs/PERFORMANCE.md`](docs/PERFORMANCE.md).

### Latency — full per-frame pipeline

200 frames at realistic density (65,536 returns/sweep):

| Stage | Mean ms | p95 ms |
|---|---|---|
| Moving-object detection | 3.2 | 3.4 |
| Engine insert + carve | 19.0 | 19.9 |
| Classify + decay + purge (amortised) | 2.1 | — |
| Rasterise to OccupancyGrid *(Python approximation)* | 27.1 | 29.4 |
| **Total** | **51.8** | **60.2** |

**19.3 Hz; the 10 Hz sensor budget is cleared on mean, p95 and max.** The rasterise row is a Python stand-in for the compiled C++ publish pass and is over half the budget — real throughput is likely better, but the compiled node has not been profiled so we do not claim it.

### Memory

| Representation | Memory | Ratio |
|---|---|---|
| **This engine, in use** | **18.3 MB** | — |
| Uniform 20 cm 2.5D grid, r = 100 m *(scaled from the row below)* | ≈ 30 MB | 1.6× |
| Uniform 10 cm 2.5D grid, r = 100 m *(scaled from the row below)* | ≈ 120 MB | 6.5× |
| Uniform 5 cm 2.5D grid, r = 100 m | 479 MB | 26× |
| Uniform 5 cm 3D voxels, r = 100 m | 3,835 MB | **210×** |

The honest headline is the 20 cm row, not the 210×: this engine's live footprint is close to what a
uniform 20 cm 2.5D map would cost, while resolving 5 cm out to 10 m and 10 cm out to 25 m. The 210×
figure is real but compares against a different data structure (a full 3D occupancy grid at 1
byte/voxel), not a finer resolution of the same one — see `docs/PERFORMANCE.md` for the full
breakdown and why that comparison is deliberately generous to the baseline.

### Moving objects, with occlusion modelled

| | 18 km/h | 40 km/h |
|---|---|---|
| Recall 0–10 m | 100% | 74% |
| Recall 10–25 m | 86% | 89% |
| Recall 25–40 m | 67% | 82% |
| **Static objects reported moving** | **0 / 286** | **0 / 129** |
| Time to confirm | 0.9–2.8 s | 0.8–2.4 s |

Occlusion costs real recall — 89% → 67% at 25–40 m versus the older idealised scene — and the 74% at 0–10 m / 40 km/h reflects an object crossing the near band faster than the ~1 s confirmation window. Both are honest limits of a confirm-then-report design, reported rather than tuned away.

**"Time to confirm" is not a blind spot.** An unconfirmed candidate is never excluded from the map: `perception/dynamic.py` only flags a point for removal once its track is `confirmed`, so until then it is published on `/drdo/cloud_static` and inserted like any other return — the engine's `classify_obstacle()` scores it purely from height statistics, with no dependency on the mover detector. The confirmation delay only affects the "moving" label; a candidate object is a normal, potentially lethal obstacle in the costmap from its first return.

### Negative obstacles

| | 40 cm pothole |
|---|---|
| Before this work | `FREE`, traversability **1.00** |
| Now | `NEGATIVE`, traversability **0.00** |

False positives: **0.000%** on flat and rough terrain, 0.036% on very rough ground (±15 cm noise, as deep as the detection threshold itself). **0 of 8,404** cells on 2% and 4% downgrades — a slope is not a trench.

### Object classification

~100% on the synthetic scene. **This needs its caveat:** the scene builds objects at canonical sizes that sit centrally inside the classifier's decision boundaries, so it largely measures that the scene and the thresholds agree. Known failure modes it does not contain: a vehicle seen edge-on reads as a wall; a person and a narrow post of similar height are not separable by extent alone.

## 5. Deep learning — built, deliberately not trained

The problem statement names a segmentation network. **We did not ship one, and we do not claim one.**

The full range-image pipeline is built and verified end to end — spherical projection, kNN label transfer, a SalsaNext-Lite conv network, a resume-safe training loop, ONNX export matching PyTorch to **1.19e-07**, and a ROS producer node. `scripts/kitti_replay.py` proves the label path works: **27,621 map cells carried a semantic label into the C++ engine**, which had never happened before in this repository.

What is missing is a trained model, because real RELLIS-3D could not be obtained in time — the 14 GB archive sits behind a Google Drive quota block with no mirror. Given days rather than weeks, we judged that **shipping measured geometry beats shipping an undertrained network**.

The `models/*` files are placeholders. `scripts/eval.py` measures them at **5.01% mIoU against a 6.64% uniform-random baseline** — worse than guessing — and prints a warning saying so. For calibration if the model is trained later: published RELLIS-3D LiDAR baselines are SalsaNext **43.07%** and KPConv **19.07%**.

## 6. Data flow

Full architecture diagram, including the offline segmentation-training path and host dev tooling: [`docs/architecture.d2`](docs/architecture.d2) / [`docs/architecture.svg`](docs/architecture.svg) (render with `d2 docs/architecture.d2 docs/architecture.svg`). The ASCII sketch below is the critical path only.

```
Ouster OS1-64 ──► FAST-LIO2 ──► /cloud_registered
                                      │
                                      ▼
                  drdo_perception/dynamic_obstacle_node ──► /drdo/dynamic_obstacles (classified markers)
                                      │  scan minus confirmed movers
                                      ▼
                               /drdo/cloud_static
                                      │
                    (optional, enable_semantics:=true)
                                      ▼
                  drdo_perception/segmentation_node ──► /drdo/cloud_labeled
                                      │
                                      ▼
                      drdo_grid_map/grid_map_node  (foveated 2.5D engine)
                           │                         │
                     /map (costmap)          /drdo/map_image (colour view)
                           │
                           ▼
            Nav2: SmacPlannerHybrid + MPPI ──► /cmd_vel
```

If a stage has no pose it passes scans through unfiltered, so the map never starves.

## 7. Quick start (laptop, no ROS needed)

```bash
# .venv inherits torch/numpy/scipy from the system interpreter and adds pytest + onnxruntime
python3 -m venv --system-site-packages .venv && .venv/bin/python -m pip install pytest onnxruntime
.venv/bin/python -m pip install -e .

# C++ engine (cmake is optional; these lines always work)
clang++ -std=c++17 -O3 -Iinclude src/drdo_map.cpp tests/cpp/test_grid_engine.cpp -o /tmp/ge && /tmp/ge
clang++ -std=c++17 -O3 -Iinclude src/drdo_map.cpp tests/cpp/test_engine_regression.cpp -o /tmp/reg
/tmp/reg 11.1 300        # gates + short drive; no args = full 3.3 km (~1.5 min)

# Python module used by the dashboard and the bridge test
clang++ -O3 -std=c++17 -shared -fPIC -undefined dynamic_lookup \
  $(.venv/bin/python -m pybind11 --includes) -Iinclude src/drdo_map.cpp src/drdo_map_py.cpp \
  -o lib/drdo_map$(.venv/bin/python -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Tests
for t in dynamic map_bridge perception deployment segmentation classify; do
  .venv/bin/python tests/python/test_$t.py; done

# Evidence
.venv/bin/python scripts/benchmark.py --frames 200      # latency table -> reports/benchmark.json
.venv/bin/python scripts/map_dashboard.py               # -> reports/map_dashboard.html
.venv/bin/python scripts/map_dashboard.py --no-occlusion --speed 11.1
```

## 8. ROS 2 (Linux / Jetson)

```bash
source /opt/ros/humble/setup.bash
pip install -e .
colcon build --packages-select drdo_grid_map drdo_perception drdo_bringup --symlink-install
source install/setup.bash

ros2 launch drdo_bringup mapping.launch.py points_topic:=/ouster/points imu_topic:=/ouster/imu
ros2 launch drdo_bringup navigation.launch.py      # mapping + Nav2
rviz2 -d rviz/drdo_map.rviz
```

Launch arguments: `filter_moving_objects` (default `true`), `enable_semantics` (default `false` — no trained checkpoint ships), `map_color_mode` (`terrain` | `elevation`), `map_resolution` (default 0.20 m), `enable_nvblox` (default `false`).

**Colour legend (`terrain`):** green = flat · amber = rough/uncertain · red = lethal step or obstacle · **indigo = pothole/trench** · teal = overhang (passable underneath) · blue = water/mud · dark green = observed free space · half brightness = not seen for over 5 s.

`fast_lio` and `ouster-ros` must be built from source in the same workspace. The Docker base image needs a decision (JetPack 6 / Ubuntu 22.04 recommended); see `deploy/docker/Dockerfile.jetson`.

## 9. Repository layout

```
include/drdo_lidar_mapping/drdo_map.h   engine constants, 40-byte GridCell, API
src/drdo_map.cpp                        hashing, foveation, carving, classification, decay, purge
src/drdo_map_py.cpp                     pybind11 module
cuda/drdo_cuda_kernels.cu               GPU port (never compiled — no nvcc on the dev host)
drdo_lidar_mapping/perception/          dynamic.py (movers), classify.py (object classes), cluster.py
drdo_lidar_mapping/segmentation/        taxonomy.py (label contract), projection.py, model.py
drdo_lidar_mapping/inference/           range_segmenter.py (the label producer)
ros2/drdo_grid_map/                     C++ map node: /map + /drdo/map_image
ros2/drdo_perception/                   moving-object node, segmentation node
ros2/drdo_bringup/                      launch files, FAST-LIO2 / Nav2 / nvblox configs
scripts/                                benchmark, dashboard, eval, replay, training, cache build
tests/cpp, tests/python                 unit, regression and integration tests
docs/PERFORMANCE.md                     every measured number, with methodology
```

## 10. Roadmap

1. **On the Orin:** build, run the regression test and the ROS graph on a recorded Ouster bag, publish real latency, power and thermal numbers, and compile the CUDA kernels for the first time.
2. **Profile the real rasterisation pass** in `grid_map_node.cpp` to replace the Python approximation that currently dominates the frame budget.
3. **Local ground estimation** for slopes. The look-ahead data-loss half is fixed — a range-scaled lower bound (`slope_adjusted_z_min()`) now keeps returns on a plausible downgrade instead of discarding them outright (was 65 of 116 points lost on a 60 m probe at 8% grade; now 0, pinned by regression GATE 7). Still open: `classify_obstacle` compares every cell against one global `ground_z`, so classification on a sustained grade can still drift where a cell isn't locally flat enough to short-circuit into the flat-ground branch. That needs real per-cell/local terrain following, not just a wider filter bound.
4. **Train the segmentation network** once RELLIS-3D (or GOOSE) is in hand; the label path is already proven end to end.
5. **Harder classification evidence** — a scene with off-nominal object sizes and edge-on vehicles, to replace an accuracy number that the current scene makes too easy.

Developed for the **Smart India Hackathon 2026, DRDO Problem ID 26053**.

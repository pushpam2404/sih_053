# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## Project

Smart India Hackathon 2026 submission for DRDO problem ID26053: a **foveated (distance-adaptive) 2.5D elevation + traversability map** for off-road UGVs, plus a geometry-only moving-object filter that keeps movers out of the static map. Target hardware is an NVIDIA Jetson AGX Orin with an Ouster OS1-64, ROS 2 Humble.

The development host is macOS (Apple Silicon): **ROS 2, CUDA, TensorRT and CMake are not installed locally.** Only the C++ engine, the pybind11 module, the Python package, the dashboard and the host-side tests actually run here. ROS 2 code is compile-checked by review only, and the launch files carry `validate_launch_configuration()` helpers that run without ROS installed.

`README.md` is the reviewer-facing document and states measured numbers with their sources. Everything it claims is reproducible by the tests below — keep it that way when changing the engine.

## Host toolchain (important)

- **Use `.venv/bin/python` for everything.** It is a 3.11 venv created with `--system-site-packages` from the uv interpreter below, so it inherits torch/numpy/scipy/sklearn/pybind11/onnx without re-downloading them and adds `pytest` + `onnxruntime` on top. It imports the tracked `lib/drdo_map*.so` fine, and torch reports **MPS available** (useful for smoke-testing the segmentation net; training happens on Colab).
  ```bash
  # recreate if lost — .venv is gitignored
  /Users/pushpam/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/bin/python3 \
      -m venv --system-site-packages .venv
  .venv/bin/python -m pip install pytest onnxruntime
  ```
- `python3` on PATH is **3.14** with no torch/pybind11 and **cannot import the tracked `lib/drdo_map*.so`** (built for CPython 3.11). The bare uv interpreter at `/Users/pushpam/.local/share/uv/python/cpython-3.11.15-macos-aarch64-none/bin/python3` (what `pyrightconfig.json` and `.vscode/settings.json` point at) also works, but it is **externally managed** — installing into it fails with PEP 668, which is why the venv exists.
- **`cmake` is not installed.** `CMakeLists.txt` is the Jetson/CI path; on this host build with the direct `clang++` lines below.
- **`timeout(1)` does not exist** on this host — don't wrap commands in it.

## Commands

```bash
PY=.venv/bin/python

# C++ engine tests (no cmake on this host)
clang++ -std=c++17 -O3 -Iinclude src/drdo_map.cpp tests/cpp/test_grid_engine.cpp -o /tmp/test_grid_engine && /tmp/test_grid_engine
clang++ -std=c++17 -O3 -Iinclude src/drdo_map.cpp tests/cpp/test_engine_regression.cpp -o /tmp/test_engine_regression
/tmp/test_engine_regression 11.1 300    # gates 1-6 + short drive; no args = full 3000-frame drive (~1.5 min)

# pybind11 module -> lib/ (where the tests and dashboard import it from)
clang++ -O3 -std=c++17 -shared -fPIC -undefined dynamic_lookup $($PY -m pybind11 --includes) -Iinclude \
  src/drdo_map.cpp src/drdo_map_py.cpp -o lib/drdo_map$($PY -c "import sysconfig;print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Python tests — standalone scripts that print "[... ALL TESTS PASSED]"; the test_* functions are also pytest-compatible
$PY tests/python/test_dynamic.py      # moving-object detector (the module the ROS node runs)
$PY tests/python/test_map_bridge.py   # pybind11 bridge
$PY tests/python/test_perception.py   # cluster/tracker + executes mapping.launch.py's validator
$PY tests/python/test_deployment.py   # model artifacts, nav2 YAML plugins, ONNX, TRT stub, Dockerfile
$PY tests/python/test_segmentation.py # range-image projection, kNN transfer, taxonomy collapse safety
$PY tests/python/test_classify.py     # geometric object classification (person/vehicle/pole/wall)

# Performance evidence — realistic OS1-64 density, staged, range-stratified
$PY scripts/benchmark.py --frames 200 --out reports/benchmark.json

# Dashboard / evaluation
$PY scripts/map_dashboard.py                  # -> reports/map_dashboard.html (needs lib/drdo_map*.so)
$PY scripts/map_dashboard.py --no-labels      # geometry only, as the live system runs
$PY scripts/map_dashboard.py --speed 11.1     # 40 km/h
$PY scripts/eval.py                           # honest per-class/per-range IoU vs random & majority baselines
$PY scripts/eval.py --model range --checkpoint <ckpt>   # once a real checkpoint exists; adds the 12-class fine table

# Segmentation: verify -> cache -> train -> end-to-end proof  (needs REAL RELLIS-3D; data/rellis is synthetic)
$PY scripts/verify_rellis.py --rellis-root <rellis>   # day-1 gate: layout, splits, measured W and FOV
$PY scripts/build_range_cache.py --rellis-root <rellis> --out <cache> --configs-dir configs
$PY scripts/train_seg.py --cache <cache> --config-dir configs --epochs 50 --resume auto
$PY scripts/export_seg_onnx.py --checkpoint <ckpt> --out models/range_seg.onnx --config-dir configs
$PY scripts/kitti_replay.py --model <ckpt|onnx>  # NON-UNKNOWN CELLS = labels that reached the C++ engine
$PY scripts/kitti_replay.py --use-gt-labels      # baseline: exercises the map path without a checkpoint

# With cmake available (Jetson / Linux CI): builds the lib, both tests, the pybind module, and CUDA if nvcc exists
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build   # then copy build/drdo_map*.so into lib/

# ROS 2 (Linux / Jetson only)
pip install -e .    # drdo_perception imports drdo_lidar_mapping at runtime
colcon build --packages-select drdo_grid_map drdo_perception drdo_bringup --symlink-install
ros2 launch drdo_bringup mapping.launch.py points_topic:=/ouster/points imu_topic:=/ouster/imu
ros2 launch drdo_bringup navigation.launch.py     # mapping + Nav2
```

`test_map_bridge.py` inserts `lib/` on `sys.path`. **If the module is missing it prints a warning and passes anyway**, so a green run does not prove the bridge works — check the output for the `[1/5] … [5/5]` lines. `*.so` is gitignored; only `lib/drdo_map.pyi` is tracked.

## Architecture

### C++ grid engine — `include/drdo_lidar_mapping/drdo_map.h`, `src/drdo_map.cpp`

The single implementation. The pybind module, the ROS 2 node and both C++ tests all link it; never copy logic out of it.

- One global preallocated `inline GridCell g_hash_pool[1<<21]` (80 MB) as an open-addressing spatial hash with linear probing (`MAX_PROBE=128`), keyed by `(ix, iy, level)`. One map per process, no locking. `GridCell` is `static_assert`ed to 40 bytes — changing fields means re-checking that.
- **Never zero a slot to delete it.** That punches a hole in the probe chain and every displaced cell becomes unreachable, then duplicated. Use `erase_slot()` (backward-shift deletion), which also moves the parallel `g_carve_ts[slot]` with the cell.
- **Foveation levels must nest exactly** (`NUM_LEVELS = 3`): ≤10 m → 5 cm, ≤25 m → 10 cm, ≤100 m → 50 cm, beyond → discarded. Every index is derived from the 5 cm base lattice by integer floor division (`cell_index()` = `floor_div(base_index(w), LEVEL_DIV[level])`, ratios 1 : 2 : 10). Never quantise a level with `floor(w / CELL_RESOLUTIONS[l])`, and keep any new resolution an integer multiple of the one below — regression **GATE 6** checks 4M positions for this.
- Per-point entry is `insert_lidar_point(x, y, z, label, rx, ry, ts, ground_z)`. The z filter (`Z_REL_MIN/MAX`) is **relative to `ground_z`**, because FAST-LIO's world origin is the IMU start pose, so ground sits near −mount_height, not 0. GATE 3/4 cover this.
- Free-space carving (`raycast_band`) walks only the hit's own foveation band, from the hit back toward the robot, stopping after `CARVE_STOP_RUN` (16) consecutive cells already carved this frame. Stopping at the *first* shared cell loses ~16% of free cells.
- **Negative obstacles (`obstacle_flag == 3`).** `obstacle_flag` is now `0=FREE, 1=OBSTACLE, 2=UNKNOWN, 3=NEGATIVE` (pothole/trench). This branch is tested **first** in `classify_obstacle`, because a pothole's floor is flat: `step_height ≈ 0` sent a 40 cm hole down the flat-ground path and it came back **FREE with traversability 1.00** — measured on the real engine, i.e. the planner was told to drive into it at full confidence. A cell qualifies when its `h_max` is `NEG_OBST_DEPTH` below `ground_z` **and** `has_ground_rim()` finds a neighbour still near ground level. The rim test is what separates a hole from a slope — descending terrain drags the whole neighbourhood down together, so nothing keeps a rim; without it every downgrade reads as one continuous trench (verified: 0 false positives at 2% and 4% grade). Flag 3 scores traversability 0 like flag 1, costs 100 in the ROS node, and draws indigo (severity 7). Large depressions only get their boundary ring flagged, which is sufficient — a lethal rim encloses the hole.
- **Classification:** real-time loops call `classify_dirty_cells(ground_z)`, which only touches cells changed since the last call. `classify_and_score_all_cells()` is a full rescan and, called every frame, overwrites `run_temporal_decay()`'s results.
- `purge_distant_cells(rx, ry, margin)` evicts a cell past its own band radius + `min(margin, PURGE_MARGIN_FRAC × radius)`. It must run **between** frames (it moves cells). Without periodic purging the pool saturates and points are dropped silently — watch `g_insert_fail` / `get_load_factor()` (`LOAD_WARN` 0.70).
- `query_world()` returns the **most recently observed** level covering a point (a carve counts as an observation), not the first level found — otherwise a stale 5 cm cell behind the robot shadows fresh coarse data.
- `spatial_hash` uses an fmix64 finalizer. `cuda/drdo_cuda_kernels.cu` mirrors it as `d_spatial_hash` and also duplicates the band radii and `LEVEL_DIV` as literals — **change both files together.**

### pybind11 bridge — `src/drdo_map_py.cpp`, `lib/drdo_map.pyi`

Python names differ from the C++ ones: `insert_point`, `insert_points` (batch, `(N,4)` float32 `[x,y,z,label]`, releases the GIL), `export_cells` (dict of numpy arrays), `query_world`, `decay_kernel`, `classify_and_score_all`, `classify_dirty`, `purge_distant`, `load_factor`, `insert_fail_count`, `valid_cell_count`, `reset_map`, plus constants `CELL_RESOLUTIONS`, `LEVEL_OUTER_R`, `CELL_BYTES`, `POOL_CELLS`. Update `lib/drdo_map.pyi` with any binding change.

### Moving-object detection — `drdo_lidar_mapping/perception/dynamic.py`

ROS-free and unit-tested on the host; the ROS node and the dashboard both drive this same module. A track is reported MOVING only when, for `confirm_frames` consecutive frames, it is above `min_speed` **and** its whole footprint shifted (`_edge_shift`: both edges of an axis move the same way) **and** the 50 cm cells it vacated ~1 s ago are now empty. A speed threshold alone reports parked cars, trees and walls as moving whenever the viewing angle changes; both extra tests exist specifically to kill that, so do not relax them without re-running `test_dynamic.py` and the dashboard's static-false-positive count.

`perception/tracker.py` is the older SORT path, exercised only by `test_perception.py`.

### Object classification — `drdo_lidar_mapping/perception/classify.py`

Geometry only, no network, and it must never be described as learned semantics. The statement asks the system to *classify* pedestrians, vehicles, walls and poles; these separate cleanly on extent because the classes differ in physical size by more than their within-class spread. `classify_extent(length, width, height) -> (obj_class, confidence)` is a decision tree with every threshold named and justified in `ClassifierThresholds`.

- **Order matters.** Pole is tested before person (a pole also satisfies the person footprint test), and wall before vehicle (a 12 m wall section otherwise lands inside the vehicle length range).
- Class ids, names and marker colours live in `segmentation/taxonomy.py` (`OBJ_NAMES`, `OBJ_RGB`) — one vocabulary shared with the untrained network head, so the live markers and the offline dashboard cannot disagree.
- `MovingObject` and `BoundingBox3D` both carry `obj_class`/`obj_conf`. `cluster.py` is no longer dead code — it is the static wall/pole classifier.
- Known limit, measured rather than hidden: objects sharing a size envelope are confused (a person and a narrow post), and a vehicle seen edge-on degenerates to a sliver footprint that reads as WALL.

### ROS 2 — `ros2/`

Pipeline: `/cloud_registered` → `dynamic_obstacle_node` → `/drdo/cloud_static` → [`segmentation_node` → `/drdo/cloud_labeled`] → `grid_map_node` → `/map` (+ `/drdo/map_image`) → Nav2.

- `drdo_perception/.../segmentation_node.py` is **opt-in behind `enable_semantics:=true`, default false** (no trained checkpoint ships). It TFs the world-registered cloud into `sensor_frame` before projecting — skipping that smears the projection by vehicle roll/pitch and silently shifts the domain away from training — then republishes the original world-frame points with `label`/`obj_class`/`conf` fields added. On any failure it passes the scan through unchanged so the map never starves.

- `drdo_grid_map/src/grid_map_node.cpp` keeps all wiring in `main()` with lambdas. Subscribes the relative name `pointcloud`, publishes `occupancy_grid` (transient_local — Nav2's StaticLayer requires it) and `map_image`; launch files do the remapping. Frames without a TF pose are **skipped** (using (0,0) re-centres the foveation on the map origin and corrupts the map). The costmap is rasterised robot-centred with max-cost aggregation over each cell's footprint.
- `drdo_perception/drdo_perception/dynamic_obstacle_node.py` only converts messages and TF. With no pose it passes the scan through unfiltered so the map never starves.
- `drdo_bringup/launch/mapping.launch.py`: FAST-LIO2 + the filter + grid map + static TFs `map→camera_init`, `map→odom`, `body→base_link`, `base_link→os_sensor`. `filter_moving_objects:=false` swaps in a second grid node fed straight from `/cloud_registered`. nvblox is opt-in (`enable_nvblox:=true`). Its `validate_launch_configuration()` runs without ROS and is called by `test_perception.py`.
- `config/fast_lio_ouster64.yaml` must stay in ROS 2 `/**: ros__parameters:` form, with `lidar_type: 3` (OUST64) — both are asserted by the validator.

### Semantic taxonomy (ADL-1)

0 GROUND, 1 GRAVEL, 2 GRASS, 3 VEGETATION, 4 OBSTACLE_HARD, 5 WATER_MUD, 6 UNKNOWN, 7 SKY/DUST (discarded on insert). Shared by the C++ engine (`SEM_TRAV`) and `perception/cluster.py`, and **owned by `segmentation/taxonomy.py`** — see the segmentation section below before touching any mapping.

## Semantic segmentation (built, not yet trained)

The range-image segmentation path exists end to end and is proven to reach the C++ engine, but **no trained checkpoint exists yet** — real RELLIS-3D has not been downloaded. Until it is, every number from this path measures plumbing, not accuracy.

**`segmentation/taxonomy.py` is the single source of truth for every label mapping.** Never define one anywhere else; `remap.py` is now a thin shim over it. Three spaces:
- **RAW** — RELLIS ontology ids (0..34, sparse), from the dataset's own `rellis.yaml`.
- **FINE** — the 12 classes the network predicts. Keeps PERSON and VEHICLE distinct (the problem statement asks for pedestrian vs vehicle); every merge stays inside one ADL-1 group, so merging costs the engine nothing.
- **ADL-1** — the 8 classes the engine consumes. Never widened.

- **`collapse_probs()` is a safety property, not an optimisation.** Collapse FINE→ADL-1 by `softmax → sum within group → argmax`, never `argmax → map`. With obstacle probability split across five fine classes at 0.15 each (0.75 total), a single grass class at 0.20 wins a naive argmax and a crowd of people is handed to the planner as drivable grass. `test_segmentation.py` pins a case where the two disagree.
- Nothing maps to ADL-1 7 (SKY/DUST) any more. RELLIS void points are unlabelled *returns*, not atmospheric noise, so they become UNKNOWN(6) rather than being discarded by the engine's `label == 7` drop. Class 7 stays reserved for the geometric dust filter.
- **`segmentation/projection.py`** — 64×1024 spherical projection, nearest return wins each pixel. `knn_postprocess` is mandatory, not a refinement: ~10–20% of points lose the z-buffer and would otherwise inherit their occluder's label (salt-and-pepper obstacles in the map). Measured on the test lattice, occluded-point accuracy goes 0.000 → 1.000. **Report per-point metrics after kNN, never per-pixel.**
- **`inference/range_segmenter.py`** — the producer. Strict by default with no heuristic fallback, like the older `MinkUNetInference(allow_fallback=False)`. Expects **sensor-frame** points: a spherical projection is only valid about the sensor origin, so a world-registered cloud must be TF'd into `os_sensor` first (`segmentation_node.py` does this).
- **`scripts/train_seg.py`** is the real trainer (Lovász + weighted CE, both mIoU numbers per epoch, atomic resume-safe checkpoints for Colab). `scripts/train.py` is **deprecated** and carries a banner explaining why it cannot train anything.
- **`scripts/build_range_cache.py`** — 256 KB/scan cache (14 GB → ~3.5 GB) so training never reads through Drive FUSE. Range is uint16 at **2 mm** steps, reaching 131 m; 1 mm would saturate at 65.5 m, inside the map's 100 m band and inside eval.py's 50–100 m bin.
- **`scripts/export_seg_onnx.py`** writes the `.onnx` **and** its `<stem>_meta.json` sidecar together. The sidecar is not optional: a net trained at one `fov_down` or normalisation and run at another is mis-projected with no error and a large silent accuracy loss. The graph must export **logits, never an argmax** — an exported argmax cannot be un-done, and it would destroy `collapse_probs`'s grouping. Measured on this host: ONNX runs **~11× faster than the torch path** (39.7 vs 432 ms/scan incl. projection + kNN) with identical map output, so ONNX is the deploy path.
- **`scripts/kitti_replay.py`** is the end-to-end evidence run. Its `NON-UNKNOWN CELLS` line is the one that matters: a cell is born UNKNOWN and carving never changes it, so that count is exactly the number of labels that travelled producer → engine. It was 0 for this repo's entire history.
- `models/*` remain **placeholders**: a per-point 2-layer MLP scoring below random in `scripts/eval.py`. Claim no accuracy for them. `build_minkunet(strict=True)` raises on mismatch; the shipped checkpoint loads only via `load_legacy_mlp()`.
- `data/rellis` is **synthetic and unusable for accuracy**: labels are independent of geometry, and `estimate_fov` on it returns +61°/−29°, which is not OS1-64 geometry. kNN *lowers* accuracy on it, as it must.
- Published RELLIS-3D LiDAR baselines for calibration: **SalsaNext 43.07% mIoU, KPConv 19.07%**. Off-road is hard; >20% beats a published baseline.

## Known gaps

- `deploy/docker/Dockerfile.jetson` cannot build as-is: the `l4t-pytorch:r35.2.1` base is JetPack 5 / Ubuntu 20.04, while ROS 2 Humble apt packages need 22.04. `fast_lio`, `ouster-ros` and nvblox are also not installed in the image.
- ~~`setup.py` console entry points reference a `scripts` module that isn't packaged.~~ **Fixed** — all four were broken (no `scripts/__init__.py`, and two target functions no longer existed) and have been removed. Scripts are run directly: `.venv/bin/python scripts/<name>.py`.
- All measured numbers are from an Apple M4 laptop on synthetic OS1-64 scans. **No Jetson Orin numbers exist yet**, the CUDA kernels have never been compiled with nvcc, and the ROS nodes have not run on a vehicle. Do not write Jetson performance figures into docs.
- **Do not headline the dashboard's ~100% classification accuracy.** `make_scene` builds its objects at canonical sizes (a 4.5 × 1.9 m car, a 0.3 m pole), which sit centrally inside `ClassifierThresholds`' boxes — so the scene is not a hard test of the decision tree and the number mostly measures that the scene and the thresholds agree. The confusion matrix beside it is the honest artifact; the real failure modes (edge-on vehicles reading as WALL, a person versus a narrow post) are size-envelope collisions this scene does not contain. Report it with that caveat or not at all.
- `scripts/benchmark.py`'s rasterisation stage is a Python/numpy **approximation** of `grid_map_node.cpp`'s publish pass, and is the largest single line in the frame budget (~27 of ~53 ms). The compiled node will be much faster, so the reported ~19 Hz is pessimistic — but it has never been profiled, so do not claim a better figure either.
- **The discard half of this is fixed; the classification-drift half is not.** `ground_z` is `grid_map_node`'s live `base_link` TF z, re-read every frame, so it already follows the vehicle down a slope it has driven — the residual gap was only the *look-ahead* case: a return down a grade the vehicle hasn't reached yet still used the vehicle's current (higher) `ground_z`. `Z_REL_MIN = -2.0` being a flat bound meant an 8% downgrade discarded every return past ~27 m outright (measured: 65 of 116 points on a 60 m probe). `slope_adjusted_z_min()` (`include/drdo_lidar_mapping/drdo_map.h`) widens that lower bound with range (`MAX_SLOPE_GRADE = 0.40`), so the same probe now keeps all 119 sampled points (regression GATE 7, `tests/cpp/test_engine_regression.cpp`) while still rejecting a return implausibly far below ground_z for its range. This is a bounded outlier gate, not per-cell terrain following — it doesn't touch `classify_obstacle`, which still compares every cell against one global `ground_z`. On a sustained grade the whole-map classification (step/clearance/max_above thresholds, when a cell isn't flat enough to short-circuit into the flat-ground branch) can still drift; real local ground estimation (fitting a plane to already-mapped nearby terrain, or per-cell relative comparison) remains open.
- `cuda/drdo_cuda_kernels.cu` mirrors the negative-obstacle logic (`d_lookup`, `d_has_ground_rim`) but **has never been compiled** — there is no nvcc on this host. Verify with nvcc on the Orin before trusting it.

## Conventions

- C++17, `-O3 -Wall -Wextra -Wpedantic`. Comments in the engine explain *why* a decision was made, usually with the measured number that forced it — preserve that style and update the numbers if behaviour changes.
- Python targets 3.8+, standard library + numpy/scipy/sklearn; scipy imports are guarded with graceful fallbacks (`linear_sum_assignment` → greedy).
- `archive/` is gitignored local history from earlier phases — do not treat anything there as current, and do not restore from it without checking against the live sources.

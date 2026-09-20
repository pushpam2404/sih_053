# DRDO ID26053: Setup & Installation Guide

This guide provides step-by-step instructions to install, build, and verify the DRDO ID26053 Adaptive 2.5D Elevation & Semantic Mapping system on Linux (Ubuntu 22.04 LTS) and macOS.

---

## 1. Prerequisites & System Requirements

### Hardware Requirements
- **Development Host:** x86_64 or Apple Silicon ARM64, 16 GB RAM minimum.
- **Target Edge Platform:** NVIDIA Jetson AGX Orin (64 GB / 32 GB) running JetPack 5.1+ / 6.0+ (L4T).
- **LiDAR Sensor:** Ouster OS1-64 (or compatible 64-beam / 32-beam LiDAR).
- **IMU:** 6-DOF / 9-DOF IMU operating at $\ge 200\,\text{Hz}$ (Ouster internal IMU supported).

### Software Requirements
- **OS:** Ubuntu 22.04 LTS (Jammy) or macOS (Ventura / Sonoma / Sequoia).
- **ROS 2:** Humble Hawksbill (`ros-humble-desktop`).
- **Compiler:** GCC 11+ or Clang 14+ with C++17 support.
- **Python:** Python 3.8 to 3.11.
- **CUDA (optional for GPU acceleration):** CUDA 11.8+ or 12.2+.

---

## 2. Quick Setup (Automated Python Package)

### Clone & Enter Repository
```bash
git clone https://github.com/pushpam2404/sih_053.git
cd sih_053
```

### Create Python Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### Install Dependencies & DRDO Package in Editable Mode
```bash
pip install -r requirements.txt
pip install -e .
```

---

## 3. Building C++ Library & Tests

### Using CMake
```bash
# Configure build directory
cmake -B build -DCMAKE_BUILD_TYPE=Release

# Build all C++ targets
cmake --build build -j$(nproc)

# Run consolidated C++ unit test
./build/test_grid_engine
```

### Direct Clang / GCC Compilation (Fallback)
If CMake is not available on your system:
```bash
clang++ -std=c++17 -O3 -Iinclude src/drdo_map.cpp tests/cpp/test_grid_engine.cpp -o test_grid_engine
./test_grid_engine
```

---

## 4. ROS 2 Workspace Build (Ubuntu 22.04)

### Source ROS 2 Humble
```bash
source /opt/ros/humble/setup.bash
```

### Install Required ROS 2 Packages
```bash
sudo apt-get update
sudo apt-get install -y \
    ros-humble-nav2-bringup \
    ros-humble-nav2-smac-planner \
    ros-humble-nav2-mppi-controller \
    ros-humble-nav2-costmap-2d \
    ros-humble-pcl-ros \
    ros-humble-tf2-ros \
    ros-humble-rviz2
```

### Build with Colcon
```bash
pip install -e .          # drdo_perception imports the repository's Python package
colcon build --packages-select drdo_grid_map drdo_perception drdo_bringup --symlink-install
source install/setup.bash
```

---

## 5. Verification & Test Suite

Run the full test suite to confirm everything is operational:

```bash
# 0. Moving-object detector (parked cars, walls and trees must never be reported as moving)
python3 tests/python/test_dynamic.py

# 1. Test Dynamic Perception & SORT Tracking
python3 tests/python/test_perception.py

# 2. Test Master Deployment & Model Artifacts
python3 tests/python/test_deployment.py

# 3. Test C++ / Pybind11 Map Bridge
python3 tests/python/test_map_bridge.py

# 4. Range-image projection, kNN label transfer, and the FINE -> ADL-1 collapse safety property
python3 tests/python/test_segmentation.py

# 5. Geometric object classification (pedestrian / vehicle / pole / wall)
python3 tests/python/test_classify.py
```

On the development Mac use `.venv/bin/python` in place of `python3` and create the
environment with `--system-site-packages`; see `CLAUDE.md`, which records why (the
system `python3` is 3.14 and cannot load the CPython 3.11 extension module, and the
uv interpreter is externally managed so `pip install` into it fails with PEP 668).

Expected output:
```
================================================================================
[DYNAMIC OBSTACLE SUITE: ALL TESTS PASSED]
================================================================================
                    [PERCEPTION SUITE: ALL TESTS PASSED]                        
================================================================================
    DRDO ID26053: PRODUCTION DEPLOYMENT SUITE — ALL CHECKS PASSED
================================================================================
                    [C++ MAP BRIDGE: ALL TESTS PASSED]                          
================================================================================
```

---

## 6. Running System Launch Files

### Launch Full Autonomous Mapping Stack (FAST-LIO2 + nvblox + DRDO Grid Map)
```bash
ros2 launch drdo_bringup mapping.launch.py points_topic:=/ouster/points imu_topic:=/ouster/imu
```

### Launch Complete Autonomous Navigation Stack (Nav2 + DRDO Costmap)
```bash
ros2 launch drdo_bringup navigation.launch.py
```

Launch arguments: `filter_moving_objects:=false` maps straight from `/cloud_registered`; `map_color_mode:=elevation` switches the colour view.

### Visualizing in RViz2
```bash
rviz2 -d rviz/drdo_map.rviz
```
The layout shows the `/map` costmap, the `/drdo/map_image` colour view (terrain: green flat, amber rough, red lethal, teal overhang) and `/drdo/dynamic_obstacles` moving-object markers.

### Offline dashboard and performance evidence (no ROS)
```bash
python3 scripts/map_dashboard.py                  # -> reports/map_dashboard.html
python3 scripts/map_dashboard.py --no-occlusion   # comparison against the idealised scene
python3 scripts/map_dashboard.py --speed 11.1     # 40 km/h

python3 scripts/benchmark.py --frames 200         # staged latency -> reports/benchmark.json
python3 scripts/eval.py                           # segmentation IoU vs random/majority baselines
python3 scripts/kitti_replay.py --use-gt-labels   # proves labels reach the C++ engine
```
All of these need the compiled `lib/drdo_map*.so` pybind11 module.
Measured results and methodology: [`PERFORMANCE.md`](PERFORMANCE.md).

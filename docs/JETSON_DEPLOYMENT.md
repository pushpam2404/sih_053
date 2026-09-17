# DRDO ID26053: NVIDIA Jetson AGX Orin Edge Deployment Guide

This guide describes deploying the DRDO ID26053 real-time mapping and perception stack on the **NVIDIA Jetson AGX Orin 64GB Developer Kit** for military off-road UGV field operations.

---

## 1. Jetson AGX Orin Platform Specifications

- **SoC:** NVIDIA Tegra Orin (T234)
- **CPU:** 12-core Arm Cortex-A78AE v8.2 64-bit CPU @ 2.2 GHz
- **GPU:** 2048-core NVIDIA Ampere architecture GPU with 64 Tensor Cores
- **AI Performance:** Up to 275 TOPS (INT8) / 138 TFLOPS (FP16)
- **Memory:** 64 GB 256-bit LPDDR5 @ 204.8 GB/s
- **Power Envelope:** Configurable 15W – 60W (MAXN mode recommended)
- **Target OS:** JetPack 5.1.2 (L4T 35.4.1) or JetPack 6.0 (L4T 36.3) with CUDA 11.4/12.2, TensorRT 8.5/8.6.

---

## 2. Setting Max Performance Mode

Before running real-time perception workloads, maximize hardware clocks:

```bash
# Set power mode to MAXN (unrestricted clock ceilings)
sudo nvpmodel -m 0

# Lock CPU and GPU to maximum frequencies
sudo jetson_clocks

# Verify clock speeds and temperatures
tegrastats
```

---

## 3. First measurements to take on the Orin

No numbers in this repository come from an Orin yet. Take these first and put only these in the slides:

```bash
# Map engine: regression gates, 3.3 km drive at 40 km/h, memory comparison
cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
./build/test_engine_regression

# Full ROS graph on a recorded Ouster bag
ros2 launch drdo_bringup mapping.launch.py &
ros2 bag play <your_ouster_bag>
ros2 topic hz /map /drdo/map_image /drdo/dynamic_obstacles
# the moving-object node logs ms/scan every 10 s; watch tegrastats for temperature and power
```

The deep-learning path (`deploy/tensorrt/compile_trt.sh`, `models/`) is future work: the current ONNX model is a placeholder, so TensorRT latency for it would not mean anything.

---

## 4. Docker Containerized Deployment

To avoid conflicting ROS 2 and host library versions, run inside the optimized NVIDIA L4T container.

### Build the Docker Image
```bash
docker build -t drdo-nav:latest -f deploy/docker/Dockerfile.jetson .
```

### Run the Container with Full Hardware Acceleration
```bash
docker run -it --rm \
    --runtime nvidia \
    --net host \
    --ipc host \
    --privileged \
    -v /dev:/dev \
    -v $(pwd)/models:/opt/drdo_nav/models \
    drdo-nav:latest
```

The container automatically executes `deploy/docker/entrypoint.sh`, which:
1. Sources `/opt/ros/humble/setup.bash` and the colcon workspace.
2. Checks for Ouster LiDAR and IMU connectivity.
3. Launches `drdo_bringup/launch/navigation.launch.py`.

---

## 5. Real-Time Scheduling & CPU Affinity

To prevent jitter caused by background Linux daemons, assign dedicated CPU cores to critical nodes:

```bash
# Isolate cores 2, 3 for FAST-LIO2 and Grid Map node
taskset -c 2,3 ros2 launch drdo_bringup mapping.launch.py &

# Set real-time priority (SCHED_FIFO) for the LiDAR driver
sudo chrt -f 80 pgrep ouster_node
```

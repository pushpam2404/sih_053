# DRDO Problem Statement ID 26053: Technical Approach & Engineered Solution

## 1. Problem Statement Overview

- **Title:** Adaptive 2.5D Elevation & Semantic Mapping for Autonomous Off-Road Military Vehicles in GPS-Denied, High-Speed Environments
- **Host Organisation:** Defence Research and Development Organisation (DRDO)
- **Target Application:** Unmanned Ground Vehicles (UGVs) and autonomous combat support vehicles maneuvering at speeds up to 40 km/h across unstructured off-road terrains (deserts, forests, mud, rocky ravines) without reliance on satellite navigation (GNSS/GPS-denied).

---

## 2. Core Off-Road Challenges vs. Our Engineered Solutions

| # | Off-Road Operational Challenge | Why Standard Autonomy Fails | Our Engineered Solution | Subsystem / Location |
|---|---|---|---|---|
| **1** | **Extreme Dynamic Dust & Smoke** | Dust clouds and exhaust generate dense LiDAR returns, triggering false emergency stops. | **Temporal Log-Odds Decay Kernel ($O(K)$ amortized):** Cells without persistent returns decay rapidly ($0.90^{age}$) and reset from `OBSTACLE` to `FREE`/`UNKNOWN`. | `src/drdo_map.cpp`, `drdo_cuda_kernels.cu` |
| **2** | **Negative Obstacles (Trenches, Ditches, Cliffs)** | Standard 2D occupancy grids only detect positive vertical obstacles above ground level. | **Welford Running Height Variance & Minimum Height Tracking:** Evaluates step height ($h_{max} - h_{min}$) and variance to capture sudden elevation drops and ditch edges. | `include/drdo_lidar_mapping/drdo_map.h`, `src/drdo_map.cpp` |
| **3** | **Deformable Vegetation vs. Lethal Obstacles** | High grass, tall weeds, and bushes appear as solid obstacles to geometric filters, trapping vehicles. | **3D Sparse Convolutional Semantic Segmentation (MinkUNet18):** Points are classified into 8 classes (`GROUND`, `GRAVEL`, `GRASS`, `VEGETATION`, `OBSTACLE`, `WATER`, `UNKNOWN`, `SKY`). Soft grass receives a high traversability weight ($0.65$), while trees/rocks receive $0.0$. | `drdo_lidar_mapping/segmentation/`, `models/` |
| **4** | **High-Speed Computational Bottleneck** | At 40 km/h (11.1 m/s), full 3D dense volumetric grids (OctoMap/VDB) exhaust GPU/CPU memory and exceed the 20 Hz latency deadline. | **Foveated 4-Tier Multi-Resolution Spatial Hash Grid:** 5 cm (0–5m), 20 cm (5–20m), 50 cm (20–50m), 100 cm (50–100m). $O(1)$ spatial hashing with flat 40 MB memory footprint. | `include/drdo_lidar_mapping/drdo_map.h`, `src/drdo_map.cpp` |
| **5** | **Severe Sensor Vibration & Odometry Drift** | Rugged terrain causes aggressive roll/pitch shakes and wheel slip, corrupting map registration. | **Tightly-Coupled Iterated Kalman Filtering (FAST-LIO2):** Fuses 64-beam LiDAR and high-rate IMU (200 Hz) to maintain sub-centimeter odometry in GPS-denied scenarios. | `ros2/drdo_bringup/config/fast_lio_ouster64.yaml` |
| **6** | **Dynamic Target Tracking (Vehicles, Soldiers, Drones)** | Static maps smudge moving entities across the map grid. | **DBSCAN Clustering + SORT 3D Kalman Tracking:** Clusters hard obstacle points and tracks dynamic instances across time with Kalman state estimation and Hungarian assignment. | `drdo_lidar_mapping/perception/` |
| **7** | **Kinodynamic Off-Road Path Planning** | Standard A* or 2D Dijkstra planners produce sharp turns infeasible for high-speed Ackermann/skid-steer UGVs. | **Nav2 SmacPlannerHybrid (Hybrid-A*) + MPPI Controller:** Generates continuous-curvature paths respecting vehicle turning limits, terrain traversability costmaps, and obstacle distance gradients. | `ros2/drdo_bringup/config/nav2_params.yaml`, `deploy/` |
| **8** | **SWaP (Size, Weight, and Power) Edge Constraints** | Military field vehicles cannot carry desktop GPUs; inference must run on embedded edge hardware. | **TensorRT FP16 Engine Execution on Jetson AGX Orin:** Sub-12 ms latency at 15W–30W power envelope. | `deploy/tensorrt/`, `deploy/docker/` |

---

## 3. Detailed Subsystem Implementation Matrix

### Phase 1: High-Performance Memory & Hash Map Engine
- **Memory Pool:** Pre-allocated 40 MB contiguous array (`1 << 20` buckets) eliminating runtime dynamic memory allocation (`malloc`/`free`) in critical path.
- **GridCell Struct:** Aligned to exactly 40 bytes (cache-line friendly, 1.6 cells per 64-byte L1 cache line).
- **Fast Spatial Hashing:** Murmur3-inspired integer bitwise mix:
  $$h(i_x, i_y, l) = ((i_x \cdot 2654435761) \oplus (i_y \cdot 805459861) \oplus (l \cdot 1234567891)) \pmod{2^{20}}$$
- **Online Variance:** Welford's single-pass recurrence avoiding historical point buffering:
  $$\mu_n = \mu_{n-1} + \frac{x_n - \mu_{n-1}}{n}, \quad M_{2,n} = M_{2,n-1} + (x_n - \mu_{n-1})(x_n - \mu_n)$$

### Phase 2: Temporal Filtering, Foveation & Traversability Fusion
- **Foveated Pyramid:** Dynamic level resolution based on Euclidean distance:
  - $d \le 5\,\text{m} \implies \text{Level 0: } 0.05\,\text{m}$ (immediate wheel contact terrain)
  - $5 < d < 20\,\text{m} \implies \text{Level 1: } 0.20\,\text{m}$ (near path planning horizon)
  - $20 \le d < 50\,\text{m} \implies \text{Level 2: } 0.50\,\text{m}$ (medium-range obstacle avoidance)
  - $50 \le d \le 100\,\text{m} \implies \text{Level 3: } 1.00\,\text{m}$ (far-horizon terrain preview)
  - $d > 100\,\text{m} \implies \text{Discard}$ (beyond sensor fidelity boundary)
- **Fused Traversability Index:**
  $$\tau = \text{confidence} \times \left(0.5 \cdot e^{-5.0 \cdot \sigma_z} + 0.5 \cdot \text{SEM\_TRAV}[c]\right)$$
  where $\sigma_z = \sqrt{M_2 / (n - 1)}$ is terrain roughness.

### Phase 3: Hardware Acceleration & ROS 2 Bridge
- **Pybind11 C++ Bridge:** Zero-overhead zero-copy access from Python perception nodes to underlying C++ hash pool.
- **CUDA Kernels:** Massively parallel GPU update kernel (`update_height_kernel`), parallel temporal decay kernel (`decay_stale_cells_kernel`), and classification kernel (`classify_and_score_kernel`).
- **ROS 2 Grid Map Node (`drdo_grid_map`):** Subscribes to `/cloud_registered`, ingests point clouds, and publishes `/map` (`nav_msgs/OccupancyGrid`) at 20 Hz.

### Phase 4: Off-Road Semantic Perception (RELLIS-3D)
- **Fine-Tuned MinkUNet18:** Trained on RELLIS-3D rugged off-road dataset using class-weighted focal Cross-Entropy loss.
- **Taxonomy Remapping:** 34 raw classes mapped to 8 operational military vehicle classes.
- **Data Augmentation:** LaserMix azimuth angle splicing, random yaw rotations, and coordinate jittering.

### Phase 5: Dynamic Obstacle Tracking & 3D ESDF Field
- **DBSCAN Clustering:** Groups hard obstacle points into 3D oriented bounding boxes (`centroid`, `min_bound`, `max_bound`, `extent`).
- **SORT 3D Kalman Tracker:** Tracks moving targets across frames, estimating position $(x, y)$ and velocity $(\dot{x}, \dot{y})$ with Hungarian association.
- **FAST-LIO2 + nvblox Launch:** Integrates IMU-LiDAR odometry with GPU-accelerated TSDF/ESDF distance slice computation.

### Phase 6: Edge Deployment & Autonomous Navigation
- **Nav2 SmacPlannerHybrid + MPPI Controller:** Kinodynamically feasible trajectory planning respecting minimum turning radius and terrain cost.
- **ONNX & TensorRT FP16:** Model serialized and compiled via `trtexec` with explicit batch sizing for Jetson AGX Orin.
- **Docker Packaging:** Production Dockerfile based on NVIDIA L4T PyTorch container with automated bringup entrypoint.

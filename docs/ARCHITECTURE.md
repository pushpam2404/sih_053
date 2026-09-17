# DRDO ID26053: System Architecture & Technical Specifications

## 1. Pipeline

```
Ouster OS1-64 (10 Hz) + IMU
        │
        ▼
FAST-LIO2 (LiDAR-inertial odometry) ──► /cloud_registered (world frame), TF camera_init→body
        │
        ▼
drdo_perception / dynamic_obstacle_node (Python)
   candidates 0.3–2.5 m above ground → raster clusters → tracks → moving-object confirmation
        │                                                   │
        │ /drdo/cloud_static (scan minus movers)            └──► /drdo/dynamic_obstacles (MarkerArray)
        ▼
drdo_grid_map / grid_map_node (C++, foveated 2.5D engine)
   insert + carve → classify changed cells → decay → purge (every 10 scans)
        │                                   │
        ▼                                   ▼
/map  nav_msgs/OccupancyGrid       /drdo/map_image  sensor_msgs/Image (terrain | elevation colours)
(robot-centred, transient_local)
        │
        ▼
Nav2: StaticLayer on /map → SmacPlannerHybrid + MPPI → /cmd_vel
```

TF tree: `map → camera_init → body → base_link → os_sensor`, plus `map → odom` (identity; FAST-LIO has no loop closure). `base_link` sits on the ground, so its z is the ground reference.

---

## 2. Grid engine

### 2.1 Cell and pool
- `GridCell` is 40 bytes: `h_min, h_max, h_mean, M2` (Welford), traversability, hit count, last-update frame, `ix, iy`, label, obstacle flag, level, valid.
- One preallocated pool of $2^{21}$ cells (80 MB) with open addressing, linear probing ($\le 128$ probes) and backward-shift deletion (never zero a slot in place).
- The hash key is $(i_x, i_y, l)$, multiplied by $2654435761, 805459861, 1234567891$, xor-combined, then passed through the MurmurHash3 `fmix64` finalizer and masked to the table size. `cuda/drdo_cuda_kernels.cu` mirrors it exactly.

### 2.2 Online height statistics (Welford)
$$\mu_n = \mu_{n-1} + \frac{z_n - \mu_{n-1}}{n},\qquad M_{2,n} = M_{2,n-1} + (z_n - \mu_{n-1})(z_n - \mu_n),\qquad \sigma_n = \sqrt{M_{2,n}/(n-1)}$$

### 2.3 Foveated, exactly nested resolution
With $d$ the horizontal distance from the robot:
$$\Delta(d) = \begin{cases}
0.05\,\text{m} & d \le 10\,\text{m} & \text{level 0}\\
0.10\,\text{m} & 10 < d \le 25\,\text{m} & \text{level 1}\\
0.50\,\text{m} & 25 < d \le 100\,\text{m} & \text{level 2}\\
\text{discard} & d > 100\,\text{m}
\end{cases}$$

Indices come from one base lattice: $b = \lfloor w / 0.05 \rfloor$ and $i_l = \lfloor b / D_l \rfloor$ with $D = (1, 2, 10)$ (integer floor division). A cell at level $l$ therefore lies inside exactly one cell of every coarser level. Quantising each level with its own float division, or using non-integer ratios, misaligns cells: the old 20 cm / 50 cm pair had 20% of positions straddling an edge.

### 2.4 Free-space carving
Each return carves a Bresenham line through its own band only. The walk runs from the hit back toward the robot and stops after 16 consecutive cells already carved in this scan. Beams in the same azimuth column share a line, so this halves carving time (38.6 → 19.2 ms on one 87k-point scan) and keeps 99.98% of free cells.

### 2.5 Classification (changed cells only)
With $g$ = ground height:
- $h_{max} - h_{min} < 0.05$ → flat
- $h_{min} - g > 0.5$ → overhang (passable)
- $h_{max} - g > 0.3$ → lethal
- otherwise → rough / unknown

$$\tau = \gamma\,(0.5\,e^{-5\sigma} + 0.5\,\text{SEM\_TRAV}[c]),\quad \gamma = \min(n/10, 1),\quad \tau = 0 \text{ if lethal}$$

Without labels, $c$ = UNKNOWN (prior 0.4).

### 2.6 Temporal decay and eviction
- A cell unseen for more than 50 scans has $\tau \leftarrow 0.9\,\tau$ (checked every 5 scans); a lethal cell whose $\tau$ drops below 0.05 becomes UNKNOWN. On its next hit, the cell's height statistics restart, so a dust column does not stay lethal.
- A cell is evicted once it is farther than $R_l + \min(m, 0.25\,R_l)$ from the robot, where $R_l$ is its band's outer radius and $m$ = 20 m. Behind the vehicle, the next coarser level still covers evicted ground.
- `query_world` returns the most recently observed level (a hit or a carve).

### 2.7 Published views
- **Costmap `/map`:** max cost over each cell's footprint at `grid_resolution` (default 0.2 m); a 100 m robot-centred window; lethal = 100, free = 0, unknown = −1.
- **Colour view `/drdo/map_image`:** same window, north up, robot marked with a white cross. The most severe cell wins per pixel. Colours: no data (near black) < free (dark green) < flat (green, or a label tint) < overhang (teal) < rough (amber) < water/mud (blue, needs labels) < lethal (red); cells stale for more than 5 s are drawn at half brightness. `color_mode:=elevation` colours $h_{max} - g$ from −1 m (blue) to +2 m (red).

---

## 3. Moving-object detection (`drdo_lidar_mapping/perception/dynamic.py`)

Geometry-only, per scan:

1. **Candidates:** points with $0.3 \le z - g \le 2.5$ m within 40 m.
2. **Clusters:** 8-connected components on a 20 cm raster, with 1-cell gaps bridged by dilation. At least 8 points; footprints longer than 7 m (walls, tree lines) are never tracked.
3. **Tracks:** constant-velocity model with velocity smoothing $v \leftarrow 0.6v + 0.4\,\hat v$. Association is Hungarian, gated at $1 + 0.5\cdot\text{extent}$ m; a track is dropped after 3 misses.
4. **Evidence:** compare the track's box now with the box $\ge 5$ frames ago (window 10 frames). All three must hold:
   - **speed** $\lVert v\rVert > 1$ m/s;
   - **whole-footprint shift:** along each axis, both box edges moved the same direction, and the smaller shift counts: $s = \lVert(\min(|\Delta x_{lo}|, |\Delta x_{hi}|), \min(|\Delta y_{lo}|, |\Delta y_{hi}|))\rVert \ge 0.8$ m, with box size changing by at most 1 m. A new face coming into view moves one edge only, so viewpoint changes score about 0.
   - **vacated space:** of the 50 cm cells covered then but not now (and within range), at most 25% are still occupied. A partly seen wall whose centroid slides fails this test.
5. **Confirmation:** evidence for 3 consecutive scans. A confirmed track is released when its speed falls below 0.5 m/s.
6. **Output:** confirmed movers are published, and their points are removed from the cloud before mapping.

Measured in the dashboard scene (2 walkers, an oncoming car; a parked car, a standing person, a wall, poles and trees):
- **Speed threshold only:** 69 of 337 moving reports were static objects.
- **With the two tests:** 0 of 298, with recall 98 / 95 / 89% over 0–10 / 10–25 / 25–40 m at 5 m/s.

---

## 4. Latency and memory (measured)

Apple M4, one CPU thread, synthetic OS1-64 scans. **Jetson AGX Orin: to be measured** (`./build/test_engine_regression`, ROS node logs).

| Stage | 87k-point scan (regression drive) | 56k-point scan (dashboard) |
|---|---|---|
| Insert + carve | 19.8–20.8 ms | 12.2–12.7 ms |
| Classify changed cells | 0.33–0.41 ms | 0.25–0.28 ms |
| Decay (every 5th scan) | ~5 ms | ~4 ms |
| Purge (every 10th scan) | ~7 ms | ~4.5 ms |
| Moving-object detection (Python) | — | ~3.5 ms |

| Map | Memory, 100 m radius |
|---|---|
| This engine, cells in use | 16–18 MB (fixed pool 80 MB) |
| Uniform 5 cm 2.5D grid | 479 MB |
| Uniform 5 cm 3D voxels, 16 m tall, 1 byte each | 3,835 MB |

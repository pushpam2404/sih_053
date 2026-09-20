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
drdo_perception / segmentation_node (Python, opt-in: enable_semantics:=true, default false)
   world→os_sensor TF → RangeSegmenter (range-image CNN) → ADL-1 label + obj_class + conf per point
        │                                                   │
        │ /drdo/cloud_labeled (labelled scan, or            └──► /drdo/semantic_objects (MarkerArray)
        │  the unlabelled input, passed through, if
        │  enable_semantics:=false or the model is absent)
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

With `enable_semantics:=false` (the default), `grid_map_node` subscribes directly to
`/drdo/cloud_static` and every point enters the map as UNKNOWN(6), traversability 0.40 — this is
the geometry-only mode that every measured number in `docs/PERFORMANCE.md` describes. The
segmentation node is a straight pass-through, not a stub: it only ever adds a `label` field that
was previously always absent.

TF tree: `map → camera_init → body → base_link → os_sensor`, plus `map → odom` (identity; FAST-LIO has no loop closure). `base_link` sits on the ground, so its z is the ground reference.

---

## 2. Grid engine

### 2.1 Cell and pool
- `GridCell` is 40 bytes: `h_min, h_max, h_mean, M2` (Welford), traversability, hit count, last-update frame, `ix, iy`, label, obstacle flag, level, valid.
- `obstacle_flag`: `0=FREE, 1=OBSTACLE, 2=UNKNOWN, 3=NEGATIVE` (pothole/trench — see §2.5a).
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
With $g$ = ground height, `classify_obstacle(cell, g)` (`src/drdo_map.cpp`) tests in this order:

1. **Negative obstacle** (§2.5a) — tested **first**.
2. $h_{max} - h_{min} < 0.05$ → flat, FREE
3. $h_{min} - g > 0.5$ → overhang (passable), FREE
4. $h_{max} - g > 0.3$ → lethal, OBSTACLE
5. otherwise → rough / UNKNOWN

$$\tau = \gamma\,(0.5\,e^{-5\sigma} + 0.5\,\text{SEM\_TRAV}[c]),\quad \gamma = \min(n/10, 1),\quad \tau = 0 \text{ if lethal or negative}$$

Without labels, $c$ = UNKNOWN (prior 0.4).

### 2.5a Negative obstacles (potholes, trenches)

Every other branch in §2.5 measures height **upward** from ground. A pothole's floor is flat, so
before this branch existed `step_height = h_{max} - h_{min} \approx 0` and the cell matched branch
2 (flat ground) — **measured**: a simulated 40 cm pothole came back `obstacle_flag = 0` (FREE),
traversability 1.00. The planner was being told to drive into the hole at full confidence. See
`docs/PERFORMANCE.md` §5 for the before/after table.

A cell qualifies as NEGATIVE (`obstacle_flag = 3`) when **all three** hold:

1. $g - h_{max} > \text{NEG\_OBST\_DEPTH}$ (0.15 m) — the cell's highest return still sits below
   the ground plane by more than the threshold.
2. $\text{hit\_count} \ge \text{NEG\_MIN\_HITS}$ (3) — a one-hit cell's $h_{max}$ *is* that single
   sample, so without this guard sensor noise alone manufactures potholes. Measured false-positive
   rate on synthetic rough terrain: ±15 cm noise gives 3.848% of cells flagged without the guard,
   0.036% with it (§ below and `docs/PERFORMANCE.md` §5).
3. `has_ground_rim(cell, g)` finds a neighbour within NEG_RIM_RADIUS_M (0.75 m, 12 probes: 4
   immediate neighbours + an 8-point ring) whose `h_max` is still near ground level.

The rim test is what separates a hole from a downgrade. A slope drags the whole neighbourhood down
together, so nothing near a descending cell stays at ground level and no false rim is ever found —
measured 0 of 8,404 cells flagged on 2% and 4% synthetic downgrades. A pothole, by construction,
has a standing rim around it. This deliberately does not classify the interior of a hole larger
than the probe ring (~1.5 m across) as anything but its boundary — an enclosing lethal rim is
operationally sufficient, since no path can cross it to reach the interior.

A NEGATIVE cell scores traversability 0 (same as a lethal positive obstacle — see §2.5), costs 100
on the published costmap, and is drawn indigo at severity 7 on `/drdo/map_image` (§2.7).

### 2.6 Temporal decay and eviction
- A cell unseen for more than 50 scans has $\tau \leftarrow 0.9\,\tau$ (checked every 5 scans); a lethal cell whose $\tau$ drops below 0.05 becomes UNKNOWN. On its next hit, the cell's height statistics restart, so a dust column does not stay lethal.
- A cell is evicted once it is farther than $R_l + \min(m, 0.25\,R_l)$ from the robot, where $R_l$ is its band's outer radius and $m$ = 20 m. Behind the vehicle, the next coarser level still covers evicted ground.
- `query_world` returns the most recently observed level (a hit or a carve).

### 2.7 Published views
- **Costmap `/map`:** max cost over each cell's footprint at `grid_resolution` (default 0.2 m); a 100 m robot-centred window; lethal = 100, free = 0, unknown = −1.
- **Colour view `/drdo/map_image`:** same window, north up, robot marked with a white cross. The most severe cell wins per pixel. Colours, in ascending severity: no data (near black) < free (dark green) < flat (green, or a label tint) < overhang (teal) < rough (amber) < water/mud (blue, needs labels) < lethal (red) < **negative obstacle (indigo, severity 7 — §2.5a)**; cells stale for more than 5 s are drawn at half brightness. `color_mode:=elevation` colours $h_{max} - g$ from −1 m (blue) to +2 m (red).

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
6. **Output:** confirmed movers are published (each carrying `obj_class`/`obj_conf` — see §4), and their points are removed from the cloud before mapping.

`MovingObject` (this module) and `BoundingBox3D` (`perception/cluster.py`, §4) both carry the same
`obj_class`/`obj_conf` fields, filled by the same `classify_extent()` call, so a moving pedestrian
and a static wall are named through one code path regardless of which module found them.

Measured in the dashboard scene (2 walkers, an oncoming car; a parked car, a standing person, a wall, poles and trees), with the scene's occlusion model on (nearest return per azimuth column wins — see `docs/PERFORMANCE.md` §4):
- **Speed threshold only (DBSCAN + SORT, no shift/vacated-space tests):** 69 of 337 moving reports were static objects.
- **With the two tests, 5 m/s:** 0 of 286 object-frames, recall 100 / 86 / 67% over 0–10 / 10–25 / 25–40 m.
- **At 11.1 m/s (40 km/h):** 0 of 129 object-frames, recall 74 / 89 / 82% over the same bands — see `docs/PERFORMANCE.md` §4 for why occlusion and speed each cost recall, and why that cost is an honest limit of confirm-then-report, not noise.

---

## 4. Object classification (`drdo_lidar_mapping/perception/classify.py`)

A geometric decision tree over a cluster's axis-aligned footprint `(length, width, height)` —
**no network is involved and the module says so in its own docstring.** The classes the problem
statement names (pedestrians, vehicles, walls, poles) separate on physical size alone: a person is
tall and thin (~0.5×0.5 m footprint, 1.5–2.0 m tall), a pole is more extreme in the same direction
(≤0.6 m footprint, >1.8 m tall), a vehicle is wide with road proportions (1.8–12 m long, 1.1–3.0 m
wide), a wall/fence is elongated along one axis only.

`classify_extent()` (called from both `dynamic.py`, §3, and `cluster.py`) tests in this fixed
order, and the order matters:

1. **Pole first.** A pole's tight footprint also satisfies the person test (both are "small
   footprint, tall"), so testing person first would swallow every post and tree trunk.
2. **Wall/fence before vehicle.** A 12 m wall section falls inside the vehicle's length range
   (1.8–12 m); testing wall first, keyed on elongation rather than absolute length, pulls it out
   before the vehicle test ever sees it.
3. Person — small footprint, upright, within a height band.
4. Vehicle — road-vehicle length/width/height ranges together, not any one alone (a 5×4 m shed
   satisfies every vehicle test individually but fails the width ceiling).
5. Anything else with real bulk → structure; anything tiny and unclassified → none.

Class ids, names and RGB colours all come from `segmentation/taxonomy.py` (`OBJ_NAMES`, `OBJ_RGB`)
— **one shared vocabulary** between this geometric classifier and the (untrained) segmentation
network's object head (`FINE_TO_OBJ_LUT`, §5), so the live RViz markers and the offline dashboard
legend can never disagree about what colour a pedestrian is. `MovingObject` (§3) and
`BoundingBox3D` both carry `obj_class`/`obj_conf`. `cluster.py`'s DBSCAN + SORT path is **not**
dead code any more — `dynamic.py` still owns moving-object tracking, but `cluster.py` is now the
static wall/pole classifier feeding `segmentation_node`'s marker layer (§1).

**Measured ~100% overall** classification accuracy (dashboard scene, n≈2,300 object-frames) —
carries a caveat that must travel with the number: the synthetic scene builds objects at canonical
sizes that sit centrally inside the thresholds above, so this largely confirms the scene and the
thresholds agree, not that the classifier is robust to edge cases such as a vehicle seen edge-on
(degenerates to a sliver and reads as WALL) or a person beside a similarly-sized post. Full detail
and the confusion matrix: `docs/PERFORMANCE.md` §6.

---

## 5. Semantic segmentation (built, deliberately untrained)

The live pipeline runs on geometry alone (§2, §3, §4); this is the producer for the `label` field
the grid engine has always been able to consume but, until now, never received — every point in
every run entered the map as UNKNOWN(6). No model is trained (real RELLIS-3D was unobtainable in
time — see `docs/PERFORMANCE.md` §8) and no accuracy is claimed; only the mechanism is asserted.

**Taxonomy (`drdo_lidar_mapping/segmentation/taxonomy.py`), three deliberately separate levels:**

| Level | Size | Purpose |
|---|---|---|
| RAW | RELLIS-3D ontology ids (0–34, sparse) | as stored in `.label` files |
| FINE | 12 classes | what the network predicts; keeps PERSON and VEHICLE distinct per the problem statement |
| ADL-1 | 8 classes | what the C++ engine consumes (`SEM_TRAV` in `drdo_map.h`) — never widened |

`taxonomy.py` is the single place a label mapping may be defined; dataset, train, eval, inference
and the ROS nodes all import from it.

**Pipeline per scan (`drdo_lidar_mapping/inference/range_segmenter.py`, `segmentation/projection.py`):**

1. **Spherical projection**, 64×1024 (OS1-64 beam count × RELLIS-3D's azimuth mode). Points are
   scattered in descending range order so the **nearest return wins the z-buffer** by construction
   — no per-point comparison branch, just one argsort.
2. **SalsaNextLite** → (12, 64, 1024) FINE logits.
3. **`taxonomy.collapse_probs`**: `softmax → sum within ADL-1 group → argmax`, **never**
   `argmax → map`. This is a safety property, not an optimisation: the obstacle group is the most
   finely subdivided (person/vehicle/pole/structure/debris all collapse to OBSTACLE_HARD), so five
   classes each holding 0.15 (0.75 combined) lose a naive per-class argmax to a single grass class
   at 0.20 — a crowd of people would be handed to the planner as drivable grass, traversability
   0.65 instead of 0. `tests/python/test_segmentation.py` pins a case where the two collapses
   disagree.
4. **kNN label transfer** (`knn_postprocess`, RangeNet++ scheme) is **mandatory, not optional
   polish**. 64×1024 = 65,536 pixels cannot hold ~131,000 OS1-64 returns, so 10–20% of points lose
   the z-buffer fight (RangeNet++'s published figure; unverified on this project's synthetic
   2,000-point stub scans, which are not real OS1-64 geometry). Without the kNN pass, every one of
   those points inherits the label of whatever occluded it — a fence post in front of grass stamps
   OBSTACLE_HARD onto the grass points behind it, and the 2.5D grid receives that as
   salt-and-pepper phantom no-go cells scattered through drivable terrain. The kNN vote is gated by
   a 1.0 m range cutoff so a foreground object's label cannot bleed onto the background it occludes.
5. Every accuracy/mIoU figure this project reports is computed **per point, after kNN**, never per
   pixel — per-pixel mIoU only scores the ~80% of points that won their own pixel and reads
   2–5 points optimistic (`docs/PERFORMANCE.md` §8).

**Frame requirement.** `RangeSegmenter.segment()` needs points in the **sensor frame**: a spherical
projection is only valid about the sensor origin, and the network trains on raw sensor-frame
scans. `segmentation_node.py` therefore looks up `world → os_sensor`, transforms a copy of the
points for inference, and attaches the resulting labels back onto the original, untouched
world-frame points before republishing — the geometry that reaches `grid_map_node` is bit-identical
to what was received.

The shipped `models/*` weights are **placeholders** (a 2-layer per-point MLP, not a real
MinkUNet/SalsaNext). `segmentation/build_minkunet(strict=True)` raises on any checkpoint mismatch
rather than silently partial-loading; `scripts/eval.py` measures the shipped checkpoint at
5.01% mIoU against a 6.64% random baseline and prints a warning — claim no accuracy for it.
`inference/MinkUNetInference(allow_fallback=False)` and `TrtInferenceRunner(allow_stub=False)`
raise rather than substitute a heuristic; only tests opt into stub fallbacks explicitly.

---

## 6. Latency and memory (measured)

Apple M4, one CPU thread, synthetic OS1-64 scans. **No Jetson AGX Orin number exists yet** —
testing on the target board was out of scope this round (no Orin hardware, Ouster sensor or
vehicle available), and the CUDA kernels (`cuda/drdo_cuda_kernels.cu`) have never been compiled
(no `nvcc` on the dev host). Treat every Orin statement in this repository as an intention, not a
measurement. This table is a summary; **`docs/PERFORMANCE.md` is the canonical source** for every
figure below, the exact commands that reproduce them, and the full-pipeline (§3) and per-range-band
(§ Cost by range band) breakdowns not repeated here.

| Stage | 87k-point scan (regression drive) | 56k-point scan (dashboard) |
|---|---|---|
| Insert + carve | 19.8–20.8 ms | 12.2–12.7 ms |
| Classify changed cells | 0.33–0.41 ms | 0.25–0.28 ms |
| Decay (every 5th scan) | ~5 ms | ~4 ms |
| Purge (every 10th scan) | ~7 ms | ~4.5 ms |
| Moving-object detection (Python) | — | ~3.5 ms |

| Map | Memory, 100 m radius |
|---|---|
| This engine, cells in use | 18.3 MB (fixed pool 80 MB) |
| Uniform 5 cm 2.5D grid | 479 MB (26×) |
| Uniform 5 cm 3D voxels, 16 m tall, 1 byte each | 3,835 MB (210×) |

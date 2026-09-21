# DRDO ID26053 — Measured Performance

Every number in this document was produced by a command in this repository, on the hardware named
below, and can be regenerated. Where a figure is an approximation or comes from synthetic input,
that is stated on the same line rather than in a footnote.

## Scope, stated up front

**No number here comes from an NVIDIA Jetson AGX Orin.** Testing on the target board was out of
scope for this round: the team had no access to Orin hardware, an Ouster OS1-64, or a vehicle.
Concretely, that means:

| Claimed | Not claimed |
|---|---|
| CPU latency on an Apple M4 laptop, single-threaded | Any Jetson Orin latency, throughput, power or thermal figure |
| Memory in use, measured from the live pool | Real-world accuracy on field data |
| Behaviour on synthetic OS1-64 scans, with and without occlusion | Behaviour under rain, dust, vibration, or GPS denial |
| C++ engine correctness under regression gates | That the CUDA kernels work — they have never been compiled (no `nvcc` on the dev host) |
| ROS 2 graph structure, validated without ROS installed | That the ROS 2 nodes have run on a robot — they have not |

The architecture targets the Orin (the CUDA port exists, the TensorRT path exists, the Docker
image exists), but **treat every Orin statement in this repository as an intention, not a
measurement.** `docs/JETSON_DEPLOYMENT.md` lists the first measurements to take when a board is
available.

**Host for all figures below:** Apple M4, macOS 27.0 (arm64), single-threaded, Python 3.11 +
pybind11 call path, `clang++ -O3`. No GPU is used anywhere in the measured pipeline.

---

## 1. Map engine — sustained drive

`./test_engine_regression` (3,000 frames, 11.1 m/s = 40 km/h, 3.3 km, 87,175 points/scan at 10 Hz)

| Stage | Mean ms/frame |
|---|---|
| `insert_points` (insert + free-space carving) | 20.3 |
| `classify_dirty_cells` | 0.42 |
| `run_temporal_decay` (amortised, runs every 5th frame) | 0.98 |
| `purge_distant_cells` (amortised, runs every 10th frame) | 0.69 |
| **Worst single frame over 3.3 km** | **50.2** |

**Stability over the full drive** — this is the result that matters for a bounded-memory claim:

| | Frame 1 | Frame 1000 | Frame 3000 |
|---|---|---|---|
| Hash pool load | 0.148 | 0.229 | 0.229 |
| Points dropped | 0 | 0 | 0 |

Peak load **0.290**, **0 insert failures** over 3.3 km. Load plateaus rather than growing, which is
the band-aware eviction working: cells leave the pool at the rate new ones enter.

### Slope tolerance (GATE 7 in the same binary)

`ground_z` is the vehicle's own live TF z, re-read every frame, so it already follows the vehicle
down a slope it has driven. The gap was look-ahead: a flat `Z_REL_MIN = -2.0 m` window discarded
any return further downhill than the vehicle's *current* height allowed, regardless of range —
measured before the fix: **65 of 116 points on a 60 m probe at an 8% downgrade, discarded outright.**
`slope_adjusted_z_min()` widens that lower bound with range (`MAX_SLOPE_GRADE = 0.40`, i.e. a 40%
grade allowance before a return is treated as an outlier rather than terrain):

| | Before | After |
|---|---|---|
| 8% downgrade, 60 m probe (119 sampled points) | 65 discarded | **0 discarded (119/119 kept)** |
| Return 20 m below `ground_z` at 10 m range (implausible for any real grade) | kept | **rejected** |

This closes the data-loss half of the slope gap. It does not close the other half: `classify_obstacle`
still scores every cell against one global `ground_z`, so on a sustained grade a cell that isn't
locally flat enough to hit the flat-ground short-circuit can still be misclassified. That needs real
per-cell local ground estimation, which remains open — see the roadmap in `README.md`.

## 2. Memory

Same run, measured from the live pool rather than computed from a formula:

| Representation | Memory | Ratio |
|---|---|---|
| **This engine (foveated 2.5D), in use** | **18.3 MB** (479,271 cells × 40 B) | — |
| Fixed preallocated pool | 80.0 MB | ceiling, not usage |
| Uniform 20 cm 2.5D grid, r = 100 m *(scaled)* | ≈ 30 MB | **1.6× larger** |
| Uniform 10 cm 2.5D grid, r = 100 m *(scaled)* | ≈ 120 MB | **6.5× larger** |
| Uniform 5 cm 2.5D grid, r = 100 m | 479 MB | **26× larger** |
| Uniform 5 cm 3D voxel grid, r = 100 m, 16 m tall, 1 B/voxel | 3,835 MB | **210× larger** |

The two "scaled" rows are not separately measured: cell count in a uniform 2.5D grid scales as
1/resolution², so they are the measured 5 cm row (479 MB) divided by 4 and by 16. They are the
fairer comparison than either extreme in this table. **The honest headline is this: our live
footprint is close to what a uniform 20 cm 2.5D map would cost, while we resolve 5 cm out to 10 m
and 10 cm out to 25 m.** The 210× figure against a full 3D voxel grid is real but is comparing
different data structures, not different resolutions of the same one — and it is deliberately
generous to that baseline besides: 1 byte per voxel is smaller than any real occupancy
implementation, and a 16 m column is shallow for a 100 m radius.

## 3. Full per-frame pipeline

`python scripts/benchmark.py --frames 200` (64 beams × 1024 azimuth = 65,536 raw returns/sweep,
~53,000 after filtering, 30 km/h)

| Stage | Mean ms | p95 ms | Max ms |
|---|---|---|---|
| Moving-object detection | 3.216 | 3.395 | 4.858 |
| Engine `insert_points` | 18.953 | 19.899 | 20.859 |
| `classify_dirty` | 0.507 | 0.579 | 0.814 |
| `decay_kernel` | 0.945 | 5.015 | 5.224 |
| `purge_distant` — when it runs (every 10th frame) | 6.342 | 6.641 | 6.764 |
| `purge_distant` — amortised | 0.634 | — | — |
| Rasterise to OccupancyGrid *(Python approximation — see below)* | 27.136 | 29.420 | 45.002 |
| **Total per frame** | **51.804** | **60.192** | **68.025** |

**19.3 Hz achievable; the 10 Hz / 100 ms sensor budget is cleared on mean, p95 and max.**

Object classification costs **1.07 µs per call** and is inline in the detection stage.

> **The rasterise row is the weakest number in this table and is deliberately pessimistic.** It is a
> Python/NumPy approximation of `grid_map_node.cpp`'s publish pass, using a corner splat rather than
> true footprint max-aggregation. It is more than half the frame budget. The compiled C++ node will
> be substantially faster, so real end-to-end throughput is very likely better than 19.3 Hz — but
> the compiled node has never been profiled, so no better figure is claimed here either.

### Cost by range band

Fresh map per band, showing what foveation actually buys:

| Band | Cell size | Mean points | Mean ms | µs/point |
|---|---|---|---|---|
| 0–10 m | 0.05 m | 25,016 | 4.995 | 0.200 |
| 10–25 m | 0.10 m | 10,376 | 3.158 | 0.304 |
| 25–50 m | 0.50 m | 9,646 | 0.969 | **0.100** |
| 50–100 m | 0.50 m | 15,424 | 2.266 | 0.147 |

The far field carries 26,000 points for 3.2 ms; the near field carries 25,000 for 5.0 ms. A uniform
5 cm grid would pay the near-field rate everywhere.

## 4. Moving-object detection

`python scripts/map_dashboard.py` — 240 frames, synthetic off-road scene with a wall, 6 poles,
70 trees, 2 pedestrians, 1 vehicle, plus a parked car and a standing person as decoys.

**Occlusion is modelled** (nearest return per azimuth column wins). The `--no-occlusion` column is
the older idealised scene, kept for comparison because it shows how much occlusion costs.

### 5 m/s (18 km/h)

| Metric | Occlusion ON | Occlusion OFF |
|---|---|---|
| Recall, 0–10 m | **100%** | 98% |
| Recall, 10–25 m | **86%** | 95% |
| Recall, 25–40 m | **67%** | 89% |
| Static objects reported moving | **0 / 286** object-frames | 0 / 298 |
| Time to confirm a mover | 0.9 – 2.8 s | 0.9 – 1.9 s |
| Map update | 11.3 ms/frame (insert 9.8, perception 2.6) | 14.0 ms |

### 11.1 m/s (40 km/h), occlusion ON

| Range | Recall |
|---|---|
| 0–10 m | 74% |
| 10–25 m | 89% |
| 25–40 m | 82% |

Static objects reported moving: **0 / 129** object-frames. Time to confirm: 0.8 – 2.4 s.

**Reading these honestly.** Occlusion costs 22 points of recall at 25–40 m (89% → 67%), because a
partly hidden object is genuinely harder to track. The 0–10 m recall at 40 km/h (74%) is lower than
at 18 km/h because an object crosses the near band in under a second while confirmation needs ~1 s
of consistent evidence — a real limitation of the confirm-then-report design, not noise.

**What "0.8–2.4 s to confirm" does *not* mean.** It is not a window during which a mover is invisible
to the map or the planner. `perception/dynamic.py` only sets a point's `moving_mask` once its track
is `confirmed` (streak ≥ `confirm_frames`); until then the point is published on `/drdo/cloud_static`
like any other return, and the C++ engine's `classify_obstacle()` scores every inserted point purely
from height statistics — it has no dependency on the mover detector at all. So a not-yet-confirmed
candidate is a normal, potentially lethal obstacle in the map and costmap from its first return. The
only thing confirmation gates is when the filter starts *excluding* a point that has actually left —
i.e. the delay costs recall on the "moving" label, never obstacle avoidance.

The **0 false positives** figure is the one the design exists to protect. A plain speed threshold
(DBSCAN + SORT) reported **69 of 337** "moving" objects as moving when they were static — parked
cars and walls whose visible face changes as the vehicle drives past. The whole-footprint-shift and
vacated-space tests reduce that to zero across both speeds.

## 5. Negative obstacles (potholes and trenches)

The problem statement names potholes explicitly. **Before this feature existed the engine was
actively wrong about them**, and the failure was measured on the real engine, not hypothesised:

| | 40 cm pothole, flat floor |
|---|---|
| Before | `obstacle_flag = 0` (FREE), **traversability 1.00** |
| After | `obstacle_flag = 3` (NEGATIVE), **traversability 0.00** |

The cause was structural: `classify_obstacle` measured only *upward* from ground, so a hole's flat
floor produced `step_height ≈ 0` and matched the flat-ground branch. The planner was being told to
drive into the hole at maximum confidence.

### False-positive rate

69,586 cells of synthetic terrain per row. `NEG_MIN_HITS` requires corroboration before a cell may
be called a hole, because `h_max` of a one-hit cell *is* that single sample:

| Terrain | Without hit guard | With hit guard |
|---|---|---|
| Flat, ±2 cm noise | 0.000% | **0.000%** |
| Rough, ±8 cm | 0.576% | **0.000%** |
| Very rough, ±15 cm | 3.848% | **0.036%** |

The residual 0.036% is a sensitivity floor, not a defect: with noise as deep as the 15 cm detection
threshold, a dip genuinely *is* a shallow pothole. The error direction is fail-safe — it calls rough
ground hazardous, never a hole drivable.

### Slope rejection

A naive "below ground = hole" rule condemns every downhill. The rim test requires a neighbour still
standing near ground level, which a descending slope never has:

| Grade | Cells flagged NEGATIVE |
|---|---|
| 2% downgrade | **0 of 8,404** |
| 4% downgrade | **0 of 8,404** |

## 6. Object classification

`drdo_lidar_mapping/perception/classify.py` — geometric decision tree over footprint and height.
**No neural network is involved and none is claimed.**

Measured against scene ground truth, binned by range: **~100% overall** (n = 2,337 object-frames
with occlusion; n = 2,372 without).

> **This number should not be quoted without its caveat.** The synthetic scene builds objects at
> canonical sizes — a 4.5 × 1.9 m car, a 0.3 m pole — which sit centrally inside the classifier's
> decision boundaries. It therefore largely measures that the scene and the thresholds agree, not
> that the classifier is robust. The confusion matrix on the dashboard is the honest artifact.
>
> Known failure modes, which this scene does not contain: a vehicle seen edge-on degenerates to a
> sliver footprint and reads as WALL; a person and a narrow post of similar height share a size
> envelope and are not separable by extent alone.

## 7. Cross-resolution correctness

GATE 6 of the regression suite, over **4,000,000 sampled positions × 2 level pairs**:

- Non-nesting ratios: **0**
- Parent-cell mismatches: **0**

Every level derives from one 5 cm lattice by integer floor division (ratios 1 : 2 : 10), so each
fine cell lies inside exactly one coarse cell. The earlier 5/20/50/100 cm pyramid did not nest —
50/20 = 2.5, so 20% of 20 cm cells straddled a 50 cm boundary.

## 8. Semantic segmentation — built, deliberately untrained

The range-image segmentation path is complete and verified end to end, but **no model is trained**
and no accuracy is claimed. Real RELLIS-3D could not be obtained in time (the 14 GB archive is
behind a Google Drive quota block with no mirror), and shipping an undertrained network would have
been worse than shipping measured geometry.

What is verified:

| | Measured |
|---|---|
| ONNX vs PyTorch parity | max abs diff **1.19e-07**, 100% argmax agreement |
| ONNX vs PyTorch speed | **39.7 vs 432 ms/scan** (11× faster; ONNX is the deploy path) |
| Spherical projection | 8.3 ms @ 65k points, 17.3 ms @ 131k |
| kNN label transfer (torch) | 13 ms @ 65k, 22 ms @ 131k |
| kNN occluded-point recovery | accuracy **0.000 → 1.000** on the test lattice |
| Labels reaching the C++ engine | **27,621 non-UNKNOWN cells** via `kitti_replay.py` |

That last row matters: before this work, no semantic label had ever reached the grid engine in this
repository's history — the ROS node read a `label` field that nothing wrote, so every point entered
the map as UNKNOWN.

The shipped `models/*` files remain **placeholders**. `scripts/eval.py` measures them at **5.01%
mIoU against a 6.64% uniform-random baseline** — worse than guessing — and prints a warning saying
so. They are kept only so the historical checkpoint stays reproducible.

For calibration if the model is ever trained: published RELLIS-3D LiDAR baselines are **SalsaNext
43.07% mIoU** and **KPConv 19.07%**, versus 59.5% for the same networks on urban SemanticKITTI.

## 9. Reproducing everything

```bash
PY=.venv/bin/python

# §1, §2, §5 slope+pothole gates, §7
clang++ -std=c++17 -O3 -Iinclude src/drdo_map.cpp tests/cpp/test_engine_regression.cpp -o /tmp/reg
/tmp/reg                       # full 3,000-frame drive (~1.5 min)
clang++ -std=c++17 -O3 -Iinclude src/drdo_map.cpp tests/cpp/test_grid_engine.cpp -o /tmp/ge && /tmp/ge

# §3
$PY scripts/benchmark.py --frames 200 --out reports/benchmark.json

# §4, §6
$PY scripts/map_dashboard.py                  # occlusion ON  -> reports/map_dashboard.html
$PY scripts/map_dashboard.py --no-occlusion   # comparison
$PY scripts/map_dashboard.py --speed 11.1     # 40 km/h

# §8
$PY scripts/eval.py
$PY scripts/kitti_replay.py --use-gt-labels
```

All Python entry points need `.venv/bin/python` (see `CLAUDE.md` — the system `python3` is 3.14 and
cannot load the compiled module). `cmake` is not installed on the dev host, hence the direct
`clang++` lines.

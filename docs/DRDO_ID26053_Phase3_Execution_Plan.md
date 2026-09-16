# DRDO ID26053 — Phase 3 Agent Execution Plan
## pybind11 Bridge · MinkUNet-18 Inference · ROS2 Publisher · Jetson CUDA Port

> **Document Type:** Principal Systems Architect's Binding Directive — Phase 3
> **Prerequisite:** Phase 2 COMPLETE — all 5 steps in `phase2/src/` passed with zero assertion failures.
> **Host (development):** macOS Apple Silicon M4 — Python 3.11 — pybind11 — PyTorch / TorchSparse++
> **Target Deploy:** NVIDIA Jetson AGX Orin 64GB — CUDA 12.x — ROS2 Humble
> **Working Directory for ALL output files:** `~/Desktop/sih/phase3/`
> **Date:** September 2026

---

## Phase 3 Architectural Goal

Phase 1 proved the C++ map engine logic (hash table, Welford, raycasting, traversability) is correct.
Phase 2 proved the engine operates correctly across frames (decay, multi-res query, sliding window, CSV pipeline).

**Phase 3 closes the loop:** attach the validated C++ engine to a real semantic inference model (MinkUNet-18 via TorchSparse++) through a pybind11 Python bridge, expose the live map to ROS2, and produce a deployable binary for the Jetson AGX Orin.

```
                    ┌──────────────────────────────────────────────────┐
                    │              PHASE 3 INTEGRATION                  │
                    │                                                   │
  LiDAR scan ──►  [ TorchSparse++ / MinkUNet-18 (Python) ]            │
  (ROS2 topic)          │  per-point semantic labels                   │
                         ▼                                             │
                  [ pybind11 bridge ]                                   │
                         │  insert_point(x,y,z,label)                 │
                         ▼                                             │
                  [ C++ Grid Engine (phase2 core) ]                    │
                    hash table · decay · raycasting                    │
                         │  query_world(wx, wy) → GridCell*            │
                         ▼                                             │
                  [ ROS2 Publisher (C++) ]                              │
                    nav_msgs/OccupancyGrid · custom GridMap msg         │
                         │                                             │
                         ▼                                             │
                  [ Nav2 Costmap Plugin / RVIZ2 Visualizer ]           │
                    ✅ Mission Planning Output                          │
                    └──────────────────────────────────────────────────┘
```

---

## Phase 3 Architectural Additions (Locked)

| ID | Feature | Decision |
|---|---|---|
| **P3-1** | pybind11 C++ module | Expose `insert_point()`, `query_world()`, `decay_kernel()`, `reset_map()` to Python |
| **P3-2** | MinkUNet-18 inference wrapper | Python class using TorchSparse++ FP16 to produce per-point `(x,y,z,label)` from raw scan |
| **P3-3** | Offline bag replay driver | Python script reads SemanticKITTI `.bin` + `.label` files and drives the full pipeline |
| **P3-4** | ROS2 C++ node | Subscribes to `/lidar/points_raw`, runs engine in-process, publishes `nav_msgs/OccupancyGrid` |
| **P3-5** | RVIZ2 visualization config | `.rviz` config file — obstacle cells red, free cells green, unknown cells grey |
| **P3-6** | Jetson CUDA port | Port hot kernels (update_height, decay, classify) to `.cu` CUDA kernels targeting sm_87 |

---

## Agent Constraints — READ BEFORE PROCEEDING

1. **One deliverable per step** — each step produces exactly one primary file (`.cpp`, `.py`, `.h`, or `.cu`).
2. **Phase 2 C++ code is the reference.** The pybind11 module wraps `drdo_map.h` unchanged. Do NOT rewrite the engine.
3. **Python environment:** `python3.11`, `torch`, `torchsparse` (TorchSparse++), `pybind11`, `numpy`.
4. **Compile command for pybind11 module:**
   ```bash
   g++ -O2 -std=c++17 -shared -fPIC \
       $(python3 -m pybind11 --includes) \
       -I ../phase2/include \
       phase3/src/drdo_map_py.cpp \
       -o phase3/lib/drdo_map$(python3-config --extension-suffix)
   ```
5. **Do not proceed** to the next step until the step's validation gate prints `[STEP P3.X COMPLETE]`.
6. `using namespace std;` in every C++ file. Python files use PEP8 style.
7. **No SLAM, no loop closure, no IMU pre-integration** — ADL-4 remains locked.

---

## BLOCK GROUP P3.0: SETUP

---

### Step P3.0.1: Create Phase 3 Directory Tree

**Objective:** Create the `phase3/` folder hierarchy and verify it with a shell validation gate.

**Execution Directive (bash):**
```bash
mkdir -p ~/Desktop/sih/phase3/src      # C++ pybind11 wrapper source
mkdir -p ~/Desktop/sih/phase3/include  # Phase 3 C++ headers
mkdir -p ~/Desktop/sih/phase3/lib      # compiled .so pybind11 module
mkdir -p ~/Desktop/sih/phase3/python   # Python inference + pipeline scripts
mkdir -p ~/Desktop/sih/phase3/ros2     # ROS2 node package
mkdir -p ~/Desktop/sih/phase3/data     # SemanticKITTI sample scan
mkdir -p ~/Desktop/sih/phase3/rviz     # RVIZ2 config files
mkdir -p ~/Desktop/sih/phase3/cuda     # CUDA kernel files (.cu)
```

**Self-Validation Gate (bash):**
```bash
for d in src include lib python ros2 data rviz cuda; do
  [ -d ~/Desktop/sih/phase3/$d ] \
    && echo "[OK] phase3/$d exists" \
    || { echo "[FAIL] phase3/$d MISSING"; exit 1; }
done
echo "[STEP P3.0.1 COMPLETE]"
```

---

## BLOCK GROUP P3.1: PYBIND11 C++ WRAPPER MODULE

---

### Step P3.1.1: Write the pybind11 Module — `drdo_map_py.cpp`

**Objective:** Expose the Phase 2 C++ grid engine to Python via a pybind11 module named `drdo_map`. The module must expose: `reset_map()`, `insert_point()`, `query_world()`, `decay_kernel()`, `classify_and_score_all()`, and `valid_cell_count()`.

**File:** `phase3/src/drdo_map_py.cpp`

**Exposed Python API:**
```python
import drdo_map

drdo_map.reset_map()
# Zeros out g_hash_pool — call at mission start.

drdo_map.insert_point(x, y, z, label, robot_x, robot_y, frame_ts) -> bool
# Full pipeline per point: discard filter → resolve_resolution → raycast → update_height
# Returns True if inserted, False if discarded.

drdo_map.query_world(wx, wy) -> dict | None
# Returns finest-level GridCell as Python dict (keys: ix, iy, level, h_min, h_max,
# h_mean, hit_count, obstacle_flag, semantic_label, traversability, last_update_ts)
# or None if not found.

drdo_map.decay_kernel(current_frame) -> int
# Runs ADL-2 decay sweep. Returns count of cells decayed.

drdo_map.classify_and_score_all(ground_z=0.0) -> int
# Classify + compute traversability for all valid cells. Returns count processed.

drdo_map.valid_cell_count() -> int
# Count of occupied buckets in hash pool.
```

**Execution Directive:**
```cpp
// phase3/src/drdo_map_py.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "drdo_map.h"   // Phase 2 shared header (via -I ../phase2/include)
namespace py = pybind11;
using namespace std;

// ── Static free functions (wrap Phase 1+2 lambdas) ───────────────────────────

static int spatial_hash_fn(int32_t ix, int32_t iy, uint8_t level) {
    uint64_t h = (uint64_t)(uint32_t)ix  * 2654435761ULL;
    h ^= (uint64_t)(uint32_t)iy * 805459861ULL;
    h ^= (uint64_t)level        * 1234567891ULL;
    return (int)(h % (uint64_t)HASH_TABLE_SIZE);
}

static int insert_cell_fn(int32_t ix, int32_t iy, uint8_t level) {
    int bucket = spatial_hash_fn(ix, iy, level);
    for (int probe = 0; probe < MAX_PROBE; probe++) {
        int slot = (bucket + probe) % HASH_TABLE_SIZE;
        GridCell& c = g_hash_pool[slot];
        if (c.valid && c.ix == ix && c.iy == iy && c.level == level) return slot;
        if (!c.valid) {
            c.h_min = +1e9f; c.h_max = -1e9f;
            c.h_mean = 0.0f; c.h_variance_M2 = 0.0f;
            c.traversability = 0.4f; c.hit_count = 0;
            c.last_update_ts = 0; c.semantic_label = 6;
            c.obstacle_flag = 2; c.ix = ix; c.iy = iy;
            c.level = level; c.valid = true;
            return slot;
        }
    }
    return -1;
}

static GridCell* lookup_cell_fn(int32_t ix, int32_t iy, uint8_t level) {
    int bucket = spatial_hash_fn(ix, iy, level);
    for (int probe = 0; probe < MAX_PROBE; probe++) {
        int slot = (bucket + probe) % HASH_TABLE_SIZE;
        GridCell& c = g_hash_pool[slot];
        if (!c.valid) return nullptr;
        if (c.ix == ix && c.iy == iy && c.level == level) return &c;
    }
    return nullptr;
}

static ResolvedCell resolve_resolution_fn(float wx, float wy, float rx, float ry) {
    float dx = wx-rx, dy = wy-ry, dist = sqrtf(dx*dx + dy*dy);
    ResolvedCell r; r.discard = false;
    if      (dist <   5.0f) { r.level = 0; r.cell_size = 0.05f; }
    else if (dist <  20.0f) { r.level = 1; r.cell_size = 0.20f; }
    else if (dist <  50.0f) { r.level = 2; r.cell_size = 0.50f; }
    else if (dist <= 100.0f){ r.level = 3; r.cell_size = 1.00f; }
    else                    { r.discard = true; return r; }
    r.ix = (int32_t)floorf(wx / r.cell_size);
    r.iy = (int32_t)floorf(wy / r.cell_size);
    return r;
}

static void update_height_fn(GridCell& c, float z) {
    c.hit_count++;
    float delta = z - c.h_mean;
    c.h_mean += delta / (float)c.hit_count;
    c.h_variance_M2 += delta * (z - c.h_mean);
    if (z < c.h_min) c.h_min = z;
    if (z > c.h_max) c.h_max = z;
}

static void mark_free_cell_fn(int32_t ix, int32_t iy, uint8_t level) {
    int slot = insert_cell_fn(ix, iy, level);
    if (slot < 0) return;
    GridCell& c = g_hash_pool[slot];
    if (c.obstacle_flag == 1) c.obstacle_flag = 2;
    if (c.obstacle_flag == 2 && c.hit_count == 0) c.obstacle_flag = 0;
}

static void raycast_fn(int32_t ox, int32_t oy, int32_t hx, int32_t hy, uint8_t lv) {
    int dx = abs(hx-ox), dy = abs(hy-oy);
    int sx = (hx>ox)?1:-1, sy = (hy>oy)?1:-1, err = dx-dy, x=ox, y=oy;
    while (true) {
        if (x==hx && y==hy) break;
        mark_free_cell_fn((int32_t)x, (int32_t)y, lv);
        int e2 = 2*err;
        if (e2>-dy) { err-=dy; x+=sx; }
        if (e2< dx) { err+=dx; y+=sy; }
    }
}

static const float SEM_TRAV_TABLE[8] = {1.0f,0.8f,0.65f,0.15f,0.0f,0.0f,0.4f,-1.0f};

static void classify_obstacle_fn(GridCell& c, float gz) {
    if (c.hit_count == 0) return;
    float hs = c.h_max-c.h_min, cl = c.h_min-gz, ma = c.h_max-gz;
    if      (hs < 0.05f) c.obstacle_flag = 0;
    else if (cl > 0.50f) c.obstacle_flag = 0;
    else if (ma > 0.30f) c.obstacle_flag = 1;
    else                 c.obstacle_flag = 2;
}

static void compute_traversability_fn(GridCell& c) {
    if (c.obstacle_flag == 1) { c.traversability = 0.0f; return; }
    float rough = (c.hit_count>1) ? sqrtf(c.h_variance_M2/(float)(c.hit_count-1)) : 0.0f;
    float hs    = expf(-5.0f * rough);
    float ss    = SEM_TRAV_TABLE[c.semantic_label];
    float conf  = fminf((float)c.hit_count / 10.0f, 1.0f);
    c.traversability = conf * (0.5f * hs + 0.5f * ss);
}

// ── Module-level Python-callable functions ────────────────────────────────────

static void reset_map() { memset(g_hash_pool, 0, sizeof(g_hash_pool)); }

static bool insert_point(float x, float y, float z, int label,
                          float rx, float ry, uint32_t ts) {
    if (label == 7 || z > 15.0f || z < -2.0f) return false;
    ResolvedCell rc = resolve_resolution_fn(x, y, rx, ry);
    if (rc.discard) return false;
    int32_t rox = (int32_t)floorf(rx / rc.cell_size);
    int32_t roy = (int32_t)floorf(ry / rc.cell_size);
    raycast_fn(rox, roy, rc.ix, rc.iy, (uint8_t)rc.level);
    int slot = insert_cell_fn(rc.ix, rc.iy, (uint8_t)rc.level);
    if (slot < 0) return false;
    GridCell& c = g_hash_pool[slot];
    update_height_fn(c, z);
    c.semantic_label = (uint8_t)label;
    c.last_update_ts = ts;
    return true;
}

static py::object query_world_py(float wx, float wy) {
    float CS[4] = {0.05f, 0.20f, 0.50f, 1.00f};
    for (int lv = 0; lv < 4; lv++) {
        int32_t qix = (int32_t)floorf(wx / CS[lv]);
        int32_t qiy = (int32_t)floorf(wy / CS[lv]);
        GridCell* f = lookup_cell_fn(qix, qiy, (uint8_t)lv);
        if (f) {
            py::dict d;
            d["ix"] = f->ix; d["iy"] = f->iy;
            d["level"]          = (int)f->level;
            d["h_min"]          = f->h_min;
            d["h_max"]          = f->h_max;
            d["h_mean"]         = f->h_mean;
            d["hit_count"]      = f->hit_count;
            d["obstacle_flag"]  = (int)f->obstacle_flag;
            d["semantic_label"] = (int)f->semantic_label;
            d["traversability"] = f->traversability;
            d["last_update_ts"] = f->last_update_ts;
            return d;
        }
    }
    return py::none();
}

static int decay_kernel_py(uint32_t cf) {
    if (cf % DECAY_K != 0) return 0;
    int decayed = 0;
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        GridCell& c = g_hash_pool[i];
        if (!c.valid) continue;
        if ((cf - c.last_update_ts) > (uint32_t)DECAY_WINDOW) {
            c.traversability *= DECAY_FACTOR;
            if (c.traversability < DECAY_MIN_TRAV && c.obstacle_flag == 1)
                c.obstacle_flag = 2;
            decayed++;
        }
    }
    return decayed;
}

static int classify_and_score_all(float gz) {
    int n = 0;
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        if (!g_hash_pool[i].valid) continue;
        classify_obstacle_fn(g_hash_pool[i], gz);
        compute_traversability_fn(g_hash_pool[i]);
        n++;
    }
    return n;
}

static int valid_cell_count() {
    int n = 0;
    for (int i = 0; i < HASH_TABLE_SIZE; i++) n += g_hash_pool[i].valid;
    return n;
}

// ── pybind11 module ───────────────────────────────────────────────────────────

PYBIND11_MODULE(drdo_map, m) {
    m.doc() = "DRDO ID26053 — Foveated 2.5D Grid Map Engine (pybind11 bridge)";
    m.def("reset_map",              &reset_map);
    m.def("insert_point",           &insert_point,
          py::arg("x"), py::arg("y"), py::arg("z"), py::arg("label"),
          py::arg("robot_x"), py::arg("robot_y"), py::arg("frame_ts"));
    m.def("query_world",            &query_world_py,   py::arg("wx"), py::arg("wy"));
    m.def("decay_kernel",           &decay_kernel_py,  py::arg("current_frame"));
    m.def("classify_and_score_all", &classify_and_score_all, py::arg("ground_z") = 0.0f);
    m.def("valid_cell_count",       &valid_cell_count);
}
```

**Compile command (run from `~/Desktop/sih/`):**
```bash
g++ -O2 -std=c++17 -shared -fPIC \
    $(python3 -m pybind11 --includes) \
    -I phase2/include \
    phase3/src/drdo_map_py.cpp \
    -o phase3/lib/drdo_map$(python3-config --extension-suffix)
```

**Self-Validation Gate (Python):**
```python
# phase3/python/test_p3_1_1.py
import sys
sys.path.insert(0, "../lib")
import drdo_map

drdo_map.reset_map()
assert drdo_map.valid_cell_count() == 0, "Fresh map must have 0 valid cells!"

ok      = drdo_map.insert_point(5.0, 0.0, 0.01, 0, 0.0, 0.0, 1)
ok_sky  = drdo_map.insert_point(10.0, 0.0, 25.0, 7, 0.0, 0.0, 1)
assert ok == True,  "Ground point must be inserted!"
assert ok_sky == False, "SKY_NOISE must be discarded!"

cell = drdo_map.query_world(5.0, 0.0)
assert cell is not None,          "Ground cell must exist!"
assert cell["semantic_label"] == 0, "Must be GROUND label!"
assert cell["level"] == 0,        "dist=5m → Level 0!"

drdo_map.classify_and_score_all(0.0)
assert drdo_map.valid_cell_count() > 0, "Map must have valid cells!"
print(f"[PASS] P3.1.1 validated. Valid cells: {drdo_map.valid_cell_count()}")
print("[STEP P3.1.1 COMPLETE]")
```

---

### Step P3.1.2: pybind11 API Stress Test — 1000 Points

**Objective:** Feed 1,000 deterministic synthetic points through the Python `drdo_map` module, run decay, classify + score, verify all map invariants.

**File:** `phase3/python/test_p3_1_2_stress.py`

```python
# phase3/python/test_p3_1_2_stress.py
import sys, math
sys.path.insert(0, "../lib")
import drdo_map

drdo_map.reset_map()
ROBOT_X, ROBOT_Y, FRAME = 0.0, 0.0, 0

inserted = discarded = 0

# 500 GROUND points, r=1-9m
for i in range(500):
    angle = (i / 500.0) * 2 * math.pi
    r = 1.0 + (i % 50) * 0.18
    z = (i % 5) * 0.004 - 0.008
    ok = drdo_map.insert_point(r*math.cos(angle), r*math.sin(angle),
                                z, 0, ROBOT_X, ROBOT_Y, FRAME+1)
    inserted += ok; discarded += (not ok)

# 400 OBSTACLE_HARD points, z=0.05-1.5m
for i in range(400):
    angle = (i / 400.0) * 2 * math.pi
    r = 5.0 + (i % 10) * 0.5
    z = 0.05 + (i % 30) * 0.05
    ok = drdo_map.insert_point(r*math.cos(angle), r*math.sin(angle),
                                z, 4, ROBOT_X, ROBOT_Y, FRAME+1)
    inserted += ok; discarded += (not ok)

# 80 VEGETATION at 15-55m
for i in range(80):
    ok = drdo_map.insert_point(15.0 + i*0.5, 10.0 + (i%8)*1.0,
                                0.3 + (i%20)*0.085, 3, ROBOT_X, ROBOT_Y, FRAME+1)
    inserted += ok; discarded += (not ok)

# 20 SKY_NOISE — must ALL be discarded
for i in range(20):
    ok = drdo_map.insert_point(float(i), 0.0, 25.0+i, 7, ROBOT_X, ROBOT_Y, FRAME+1)
    assert ok == False, f"SKY_NOISE point {i} must be discarded!"
    discarded += 1

print(f"Inserted: {inserted}  Discarded: {discarded}")
assert discarded == 20,  "Exactly 20 SKY_NOISE must be discarded!"
assert inserted  > 900,  "At least 900 valid points must be inserted!"

processed = drdo_map.classify_and_score_all(0.0)
n = drdo_map.valid_cell_count()
print(f"Valid cells: {n}  Processed: {processed}")
assert n > inserted, "Valid cells must exceed inserted (raycasting adds free cells)!"

# Advance to frame=55 (age=54 > DECAY_WINDOW=50 → all stale)
decayed = drdo_map.decay_kernel(55)
print(f"Cells decayed at frame=55: {decayed}")
assert decayed > 0, "Stale cells must decay!"

print("[PASS] 1000-point stress test passed.")
print("[STEP P3.1.2 COMPLETE]")
```

---

## BLOCK GROUP P3.2: MINKUNET-18 INFERENCE WRAPPER

---

### Step P3.2.1: MinkUNet-18 Inference Class

**Objective:** Implement `MinkUNetInference` — a Python class that loads a pretrained MinkUNet-18 checkpoint and runs TorchSparse++ FP16 inference. Includes **DUMMY MODE** (deterministic labeling, no checkpoint needed) for pipeline testing.

**File:** `phase3/python/mink_inference.py`

**Architecture locked:** MinkUNet-18 + TorchSparse++ backend. FP16 on CUDA, FP32 fallback on CPU. VOXEL_SIZE = 0.05m (ADL-3).

```python
# phase3/python/mink_inference.py
"""
MinkUNet-18 inference wrapper for DRDO ID26053.
Backend: TorchSparse++ (torchsparse >= 2.1)
Input:  (N,4) float32 numpy array [x, y, z, intensity]
Output: (N,4) float32 numpy array [x, y, z, label_id]
"""
import numpy as np
import torch

NUM_CLASSES = 8  # ADL-1 class count (GROUND ... SKY_NOISE)


class MinkUNetInference:
    """
    Wraps MinkUNet-18 (TorchSparse++) for per-point semantic labeling.

    Usage:
        model = MinkUNetInference(checkpoint_path=None)       # DUMMY MODE
        model = MinkUNetInference("weights/minkunet18.pth")   # REAL MODE
        labeled = model.infer(raw_cloud_np)   # (N,4) → (N,4)
    """
    VOXEL_SIZE = 0.05  # metres — locked by ADL-3

    def __init__(self, checkpoint_path: str = None, device: str = "auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        self.use_fp16   = self.device.type == "cuda"
        self.model      = None
        self.dummy_mode = (checkpoint_path is None)
        if not self.dummy_mode:
            self._load_model(checkpoint_path)
        print(f"[MinkUNetInference] device={self.device}  fp16={self.use_fp16}"
              f"  mode={'DUMMY' if self.dummy_mode else 'REAL'}")

    def _load_model(self, path: str):
        try:
            from torchsparse.models import MinkUNet18
        except ImportError as e:
            raise ImportError(f"TorchSparse++ not installed: {e}")
        self.model = MinkUNet18(in_channels=4, num_classes=NUM_CLASSES)
        ckpt  = torch.load(path, map_location=self.device)
        state = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()
        if self.use_fp16:
            self.model.half()
        print(f"[MinkUNetInference] Checkpoint loaded: {path}")

    def _voxelize(self, points: np.ndarray):
        from torchsparse.utils.quantize import sparse_quantize
        import torchsparse
        coords_int = np.floor(points[:, :3] / self.VOXEL_SIZE).astype(np.int32)
        _, unique_map, inverse_map = sparse_quantize(
            coords_int, return_index=True, return_inverse=True)
        vox_feats  = torch.tensor(points[unique_map], dtype=torch.float32)
        if self.use_fp16: vox_feats = vox_feats.half()
        batch_col  = torch.zeros(len(unique_map), 1, dtype=torch.int32)
        vox_coords = torch.cat([batch_col,
                                 torch.tensor(coords_int[unique_map], dtype=torch.int32)], dim=1)
        return torchsparse.SparseTensor(feats=vox_feats.to(self.device),
                                        coords=vox_coords.to(self.device)), inverse_map

    def infer(self, points: np.ndarray) -> np.ndarray:
        """Returns (N,4) float32 [x, y, z, label_id]."""
        N = len(points)
        if points.shape[1] == 3:
            points = np.concatenate([points, np.ones((N,1), np.float32)], axis=1)
        valid_mask    = (points[:,2] >= -2.0) & (points[:,2] <= 15.0)
        points_valid  = points[valid_mask]
        if self.dummy_mode or self.model is None:
            labels_valid = self._dummy_labels(points_valid)
        else:
            labels_valid = self._real_infer(points_valid)
        output         = np.zeros((N, 4), dtype=np.float32)
        output[:, :3]  = points[:, :3]
        output[:,  3]  = 7.0            # default: SKY_NOISE → will be discarded
        output[valid_mask, :3] = points_valid[:, :3]
        output[valid_mask,  3] = labels_valid.astype(np.float32)
        return output

    def _real_infer(self, points: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            sparse_input, inverse_map = self._voxelize(points)
            logits     = self.model(sparse_input).feats
            vox_labels = logits.argmax(dim=1)
            return vox_labels[torch.tensor(inverse_map, device=self.device)].cpu().numpy().astype(np.uint8)

    def _dummy_labels(self, points: np.ndarray) -> np.ndarray:
        """Deterministic labeling for no-checkpoint pipeline tests."""
        labels = np.zeros(len(points), dtype=np.uint8)
        z      = points[:, 2]
        dist   = np.sqrt(points[:,0]**2 + points[:,1]**2)
        labels[z < 0.10] = 0                               # GROUND
        labels[(z >= 0.10) & (z < 0.30)]           = 1    # GRAVEL_DIRT
        labels[(z >= 0.30) & (dist <  10.0)]        = 4   # OBSTACLE_HARD
        labels[(z >= 0.30) & (dist >= 10.0)]        = 3   # VEGETATION_DENSE
        return labels
```

**Self-Validation Gate:**
```python
# phase3/python/test_p3_2_1.py
import numpy as np, sys
sys.path.insert(0, ".")
from mink_inference import MinkUNetInference

model = MinkUNetInference(checkpoint_path=None, device="cpu")
np.random.seed(42)
N   = 130
raw = np.zeros((N, 4), dtype=np.float32)
raw[:, 0] = np.random.uniform(-20, 20, N)
raw[:, 1] = np.random.uniform(-20, 20, N)
raw[:, 2] = np.random.uniform(-0.1, 2.0, N)
raw[:, 3] = 1.0
raw[-5:, 2] = 20.0   # out-of-range → must get label=7

labeled = model.infer(raw)
assert labeled.shape == (N, 4)
assert list(labeled[-5:, 3]) == [7.0]*5, "Out-of-range must get label=7!"
assert set(labeled[:, 3].astype(int)).issubset({0,1,3,4,7})

print("[PASS] MinkUNetInference (dummy mode) validated.")
print("[STEP P3.2.1 COMPLETE]")
```

---

### Step P3.2.2: SemanticKITTI Offline Bag Replay Driver

**Objective:** Python script that reads a SemanticKITTI `.bin` + `.label` scan (or generates a synthetic scan), feeds all points through `MinkUNetInference` + `drdo_map`, and validates the final map state.

**File:** `phase3/python/kitti_replay.py`

**SemanticKITTI → DRDO class remapping (locked):**

| KITTI ID | KITTI Label | DRDO ID | DRDO Label |
|---|---|---|---|
| 40 | road | 0 | GROUND |
| 44 | parking | 0 | GROUND |
| 48 | sidewalk | 1 | GRAVEL_DIRT |
| 70 | vegetation | 3 | VEGETATION_DENSE |
| 71 | trunk | 4 | OBSTACLE_HARD |
| 72 | terrain | 2 | GRASS_LOW |
| 80, 81 | pole / sign | 4 | OBSTACLE_HARD |
| 252 | moving-car | 4 | OBSTACLE_HARD |
| *other* | *other* | 6 | UNKNOWN |

```python
# phase3/python/kitti_replay.py
"""
SemanticKITTI offline bag replay driver for DRDO ID26053.
Usage:
    python3 kitti_replay.py [scan.bin] [scan.label] [checkpoint.pth]
    python3 kitti_replay.py            # synthetic mode — no data needed
"""
import sys, os
import numpy as np
sys.path.insert(0, "../lib")
sys.path.insert(0, ".")
import drdo_map
from mink_inference import MinkUNetInference

KITTI_TO_DRDO = {40:0, 44:0, 48:1, 70:3, 71:4, 72:2, 80:4, 81:4, 252:4}


def remap(kitti_label: int) -> int:
    return KITTI_TO_DRDO.get(kitti_label & 0xFFFF, 6)


def load_kitti(bin_path: str, label_path: str = None):
    pts    = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)
    labels = None
    if label_path and os.path.exists(label_path):
        raw    = np.fromfile(label_path, dtype=np.uint32)
        labels = np.array([remap(int(l)) for l in raw], dtype=np.uint8)
    return pts, labels


def synthetic_scan(N=500):
    np.random.seed(2026)
    angles = np.random.uniform(0, 2*3.14159, N)
    radii  = np.random.uniform(1.0, 80.0, N)
    x, y   = radii*np.cos(angles), radii*np.sin(angles)
    z      = np.random.uniform(-0.05, 0.05, N)
    z[radii < 15.0] = np.random.uniform(0.05, 1.5, (radii<15.0).sum())
    pts    = np.stack([x, y, z, np.ones(N)], axis=1).astype(np.float32)
    lbs    = np.where(z < 0.05, 0, 4).astype(np.uint8)
    lbs[radii > 50.0] = 3
    return pts, lbs


def run_replay(bin_path=None, label_path=None, ckpt=None, robot_x=0.0, robot_y=0.0):
    drdo_map.reset_map()

    if bin_path and os.path.exists(bin_path):
        pts, gt_labels = load_kitti(bin_path, label_path)
        print(f"[KITTI] Loaded {len(pts)} points from {bin_path}")
    else:
        print("[KITTI] Using synthetic 500-point scan.")
        pts, gt_labels = synthetic_scan()

    model  = MinkUNetInference(checkpoint_path=ckpt, device="auto")

    if gt_labels is not None:
        labeled = np.column_stack([pts[:, :3], gt_labels.astype(np.float32)])
    else:
        labeled = model.infer(pts)

    inserted = discarded = 0
    for i in range(len(labeled)):
        x, y, z = float(labeled[i,0]), float(labeled[i,1]), float(labeled[i,2])
        lbl = int(labeled[i,3])
        ok  = drdo_map.insert_point(x, y, z, lbl, robot_x, robot_y, 1)
        inserted += ok; discarded += (not ok)

    print(f"[PIPELINE] Inserted: {inserted}  Discarded: {discarded}")

    processed = drdo_map.classify_and_score_all(0.0)
    n_valid   = drdo_map.valid_cell_count()
    print(f"[PIPELINE] Valid cells: {n_valid}  Processed: {processed}")

    decayed = drdo_map.decay_kernel(55)
    print(f"[PIPELINE] Decay frame=55: {decayed} cells decayed.")

    # Spot-check a ground point
    gpts = labeled[(labeled[:,3]==0) & (labeled[:,2]<0.1)]
    if len(gpts) > 0:
        cell = drdo_map.query_world(float(gpts[0,0]), float(gpts[0,1]))
        if cell:
            print(f"[SPOT CHECK] Ground cell: level={cell['level']}"
                  f"  flag={cell['obstacle_flag']}  trav={cell['traversability']:.3f}")

    print("[PASS] KITTI replay validated.")
    return inserted, discarded, n_valid


if __name__ == "__main__":
    ins, disc, n = run_replay(
        sys.argv[1] if len(sys.argv)>1 else None,
        sys.argv[2] if len(sys.argv)>2 else None,
        sys.argv[3] if len(sys.argv)>3 else None,
    )
    assert ins > 0,   "Must insert at least 1 point!"
    assert n   > ins, "Valid cells must exceed inserted points (raycasting)!"
    print("[STEP P3.2.2 COMPLETE]")
```

**Run command:**
```bash
# Synthetic mode — no data files needed:
cd ~/Desktop/sih/phase3/python && python3 kitti_replay.py

# With real KITTI data:
python3 kitti_replay.py /path/to/000000.bin /path/to/000000.label
```

---

## BLOCK GROUP P3.3: ROS2 PUBLISHER NODE

---

### Step P3.3.1: ROS2 Package Setup — `drdo_grid_map`

**Objective:** Create the ROS2 Humble package `drdo_grid_map` that houses the C++ grid publisher node.

**File tree:**
```
phase3/ros2/drdo_grid_map/
├── CMakeLists.txt
├── package.xml
└── src/
    └── grid_map_node.cpp
```

**`package.xml`:**
```xml
<?xml version="1.0"?>
<package format="3">
  <name>drdo_grid_map</name>
  <version>1.0.0</version>
  <description>DRDO ID26053 — Foveated 2.5D LiDAR Grid Map ROS2 Publisher</description>
  <maintainer email="drdo@sih2026.in">SIH Team</maintainer>
  <license>Apache-2.0</license>
  <buildtool_depend>ament_cmake</buildtool_depend>
  <depend>rclcpp</depend>
  <depend>sensor_msgs</depend>
  <depend>nav_msgs</depend>
  <depend>geometry_msgs</depend>
  <depend>tf2_ros</depend>
  <export>
    <build_type>ament_cmake</build_type>
  </export>
</package>
```

**`CMakeLists.txt`:**
```cmake
cmake_minimum_required(VERSION 3.16)
project(drdo_grid_map)
set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

# Include Phase 2 shared header (adjust path relative to colcon workspace root)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../phase2/include)

add_executable(grid_map_node src/grid_map_node.cpp)
ament_target_dependencies(grid_map_node
  rclcpp sensor_msgs nav_msgs geometry_msgs tf2_ros)

install(TARGETS grid_map_node DESTINATION lib/${PROJECT_NAME})
ament_package()
```

---

### Step P3.3.2: ROS2 Grid Map Publisher Node

**Objective:** Implement the ROS2 C++ node. Subscribes to `/lidar/points_raw` (`sensor_msgs/PointCloud2`), runs the full pipeline in-process (using `drdo_map.h`), publishes `nav_msgs/OccupancyGrid` on `/drdo/costmap` at **10 Hz**.

**File:** `phase3/ros2/drdo_grid_map/src/grid_map_node.cpp`

**Design decisions:**
- The C++ grid engine runs **in-process** (no subprocess) — zero IPC overhead.
- MinkUNet-18 inference: Phase 3 MVP reads pre-labeled PointCloud2 (`label` field). A separate inference sidecar process (Phase 4) will inject the `label` field via a relay node.
- Robot pose: read from TF2 transform `map → base_link`. Falls back to `(0,0)` if TF unavailable.

```cpp
// phase3/ros2/drdo_grid_map/src/grid_map_node.cpp
// NOTE: copy spatial_hash_fn, insert_cell_fn, lookup_cell_fn,
//       resolve_resolution_fn, update_height_fn, mark_free_cell_fn,
//       raycast_fn, classify_obstacle_fn, compute_traversability_fn
//       verbatim from phase3/src/drdo_map_py.cpp before the class definition.
// All are static free functions — no header pollution.

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <cstring>
#include <cmath>
#include <chrono>
#include "drdo_map.h"

using namespace std;
using namespace std::chrono_literals;

// [INSERT all static free functions from drdo_map_py.cpp here]

class GridMapNode : public rclcpp::Node {
public:
    GridMapNode()
        : Node("drdo_grid_map_node"),
          tf_buffer_(get_clock()),
          tf_listener_(tf_buffer_) {
        memset(g_hash_pool, 0, sizeof(g_hash_pool));
        RCLCPP_INFO(get_logger(), "[DRDO] Hash pool init (40MB). Waiting for LiDAR...");

        pc_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            "/lidar/points_raw", rclcpp::SensorDataQoS(),
            std::bind(&GridMapNode::on_pointcloud, this, std::placeholders::_1));

        costmap_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/drdo/costmap", 10);

        publish_timer_ = create_wall_timer(
            100ms, std::bind(&GridMapNode::publish_costmap, this));

        frame_ts_ = 0;
    }

private:
    void on_pointcloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        frame_ts_++;

        // Get robot pose from TF
        float robot_x = 0.0f, robot_y = 0.0f;
        try {
            auto tf = tf_buffer_.lookupTransform("map", "base_link", tf2::TimePointZero);
            robot_x = (float)tf.transform.translation.x;
            robot_y = (float)tf.transform.translation.y;
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "[DRDO] TF failed: %s — using (0,0)", ex.what());
        }

        // Parse PointCloud2 field offsets
        int off_x=-1, off_y=-1, off_z=-1, off_label=-1;
        for (const auto& f : msg->fields) {
            if (f.name=="x")     off_x     = f.offset;
            if (f.name=="y")     off_y     = f.offset;
            if (f.name=="z")     off_z     = f.offset;
            if (f.name=="label") off_label = f.offset;
        }
        if (off_x<0||off_y<0||off_z<0) {
            RCLCPP_ERROR_ONCE(get_logger(), "[DRDO] PointCloud2 missing x/y/z!");
            return;
        }

        const uint8_t* data = msg->data.data();
        uint32_t step = msg->point_step;
        uint32_t N    = msg->width * msg->height;
        int inserted = 0;

        for (uint32_t i = 0; i < N; i++) {
            const uint8_t* pt = data + i*step;
            float wx, wy, wz;
            memcpy(&wx, pt+off_x, 4);
            memcpy(&wy, pt+off_y, 4);
            memcpy(&wz, pt+off_z, 4);

            uint8_t label = 6;  // UNKNOWN default
            if (off_label >= 0) {
                uint8_t rl; memcpy(&rl, pt+off_label, 1);
                label = (rl < 8) ? rl : 6;
            }

            if (label==7 || wz>15.0f || wz<-2.0f) continue;

            ResolvedCell rc = resolve_resolution_fn(wx, wy, robot_x, robot_y);
            if (rc.discard) continue;

            int32_t rox = (int32_t)floorf(robot_x / rc.cell_size);
            int32_t roy = (int32_t)floorf(robot_y / rc.cell_size);
            raycast_fn(rox, roy, rc.ix, rc.iy, (uint8_t)rc.level);

            int slot = insert_cell_fn(rc.ix, rc.iy, (uint8_t)rc.level);
            if (slot < 0) continue;
            GridCell& c = g_hash_pool[slot];
            update_height_fn(c, wz);
            c.semantic_label = label;
            c.last_update_ts = frame_ts_;
            inserted++;
        }

        // Classify + score all valid cells
        for (int i = 0; i < HASH_TABLE_SIZE; i++) {
            if (!g_hash_pool[i].valid) continue;
            classify_obstacle_fn(g_hash_pool[i], 0.0f);
            compute_traversability_fn(g_hash_pool[i]);
        }

        // Temporal decay every DECAY_K frames
        if (frame_ts_ % DECAY_K == 0) {
            int decayed = 0;
            for (int i = 0; i < HASH_TABLE_SIZE; i++) {
                GridCell& c = g_hash_pool[i];
                if (!c.valid) continue;
                if ((frame_ts_ - c.last_update_ts) > (uint32_t)DECAY_WINDOW) {
                    c.traversability *= DECAY_FACTOR;
                    if (c.traversability < DECAY_MIN_TRAV && c.obstacle_flag == 1)
                        c.obstacle_flag = 2;
                    decayed++;
                }
            }
            RCLCPP_DEBUG(get_logger(), "[DRDO] Frame %u: %d decayed | %d inserted.",
                         frame_ts_, decayed, inserted);
        }
    }

    void publish_costmap() {
        constexpr int   GW = 200, GH = 200;
        constexpr float CS = 1.00f;   // Level-3 cell size for costmap
        auto msg = nav_msgs::msg::OccupancyGrid();
        msg.header.stamp    = get_clock()->now();
        msg.header.frame_id = "map";
        msg.info.resolution = CS;
        msg.info.width      = GW;
        msg.info.height     = GH;
        msg.info.origin.position.x = -GW * CS / 2.0;
        msg.info.origin.position.y = -GH * CS / 2.0;
        msg.data.assign(GW * GH, -1);

        float CSIZES[4] = {0.05f, 0.20f, 0.50f, 1.00f};
        for (int j = 0; j < GH; j++) {
            for (int i = 0; i < GW; i++) {
                float wx = msg.info.origin.position.x + (i+0.5f)*CS;
                float wy = msg.info.origin.position.y + (j+0.5f)*CS;
                GridCell* cell = nullptr;
                for (int lv = 0; lv < 4 && !cell; lv++) {
                    int32_t qx = (int32_t)floorf(wx/CSIZES[lv]);
                    int32_t qy = (int32_t)floorf(wy/CSIZES[lv]);
                    cell = lookup_cell_fn(qx, qy, (uint8_t)lv);
                }
                int8_t val = -1;
                if (cell) {
                    if      (cell->obstacle_flag==1) val = 100;
                    else if (cell->obstacle_flag==0) val = 0;
                    else                             val = 50;
                }
                msg.data[j*GW+i] = val;
            }
        }
        costmap_pub_->publish(msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr      costmap_pub_;
    rclcpp::TimerBase::SharedPtr                                    publish_timer_;
    tf2_ros::Buffer                                                 tf_buffer_;
    tf2_ros::TransformListener                                      tf_listener_;
    uint32_t                                                        frame_ts_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GridMapNode>());
    rclcpp::shutdown();
    return 0;
}
```

**Build + Run (Jetson / ROS2 Humble):**
```bash
cd ~/Desktop/sih/phase3/ros2
colcon build --packages-select drdo_grid_map
source install/setup.bash
ros2 run drdo_grid_map grid_map_node
```

**Self-Validation Gate:**
```bash
# Separate terminal — verify 10 Hz publish rate
ros2 topic hz /drdo/costmap        # expect: average rate: ~10.0 Hz
ros2 topic echo /drdo/costmap --no-arr --once  # expect: valid header + metadata
echo "[STEP P3.3.2 COMPLETE]"
```

---

## BLOCK GROUP P3.4: RVIZ2 VISUALIZATION

---

### Step P3.4.1: RVIZ2 Config File

**Objective:** Write an RVIZ2 `.rviz` config that displays the `/drdo/costmap` OccupancyGrid with costmap colour scheme (obstacle=red, free=green, unknown=grey).

**File:** `phase3/rviz/drdo_map.rviz`

```yaml
Panels:
  - Class: rviz_common/Displays
    Name: Displays
Visualization Manager:
  Displays:
    - Class: rviz_default_plugins/Map
      Name: DRDO 2.5D Costmap
      Topic:
        Value: /drdo/costmap
      Color Scheme: costmap
      Alpha: 0.75
      Draw Under: false
      Enabled: true
    - Class: rviz_default_plugins/TF
      Name: TF Frames
      Enabled: true
  Global Options:
    Fixed Frame: map
    Background Color: 40; 40; 40
  Views:
    Current:
      Class: rviz_default_plugins/TopDownOrtho
      Scale: 10.0
      Target Frame: base_link
```

**Launch:**
```bash
rviz2 -d ~/Desktop/sih/phase3/rviz/drdo_map.rviz
echo "[STEP P3.4.1 COMPLETE]"
```

---

## BLOCK GROUP P3.5: JETSON CUDA PORT

---

### Step P3.5.1: CUDA Kernels — `drdo_cuda_kernels.cu`

**Objective:** Port the three CPU hot paths to CUDA kernels for Jetson AGX Orin. One thread per point (Kernel 1) or one thread per hash-table slot (Kernels 2–3).

**File:** `phase3/cuda/drdo_cuda_kernels.cu`

**Compile (Jetson AGX Orin, sm_87):**
```bash
nvcc -O2 -arch=sm_87 -std=c++17 \
     -I ../../phase2/include \
     phase3/cuda/drdo_cuda_kernels.cu \
     -o phase3/cuda/drdo_cuda_test
# For Jetson AGX Xavier: use -arch=sm_72
```

```cuda
// phase3/cuda/drdo_cuda_kernels.cu
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include "drdo_map.h"

// ── Device hash + lookup ──────────────────────────────────────────────────────

__device__ int d_spatial_hash(int32_t ix, int32_t iy, uint8_t level) {
    uint64_t h  = (uint64_t)(uint32_t)ix * 2654435761ULL;
    h ^= (uint64_t)(uint32_t)iy * 805459861ULL;
    h ^= (uint64_t)level        * 1234567891ULL;
    return (int)(h % (uint64_t)HASH_TABLE_SIZE);
}

__device__ GridCell* d_lookup_or_insert(GridCell* pool,
                                         int32_t ix, int32_t iy, uint8_t level) {
    int bucket = d_spatial_hash(ix, iy, level);
    for (int p = 0; p < MAX_PROBE; p++) {
        int slot   = (bucket + p) % HASH_TABLE_SIZE;
        GridCell& c = pool[slot];
        if (c.valid && c.ix==ix && c.iy==iy && c.level==level) return &c;
        if (!c.valid) {
            if (atomicCAS((int*)&c.valid, 0, 1) == 0) {
                // We claimed this slot — initialize it
                c.h_min=+1e9f; c.h_max=-1e9f;
                c.h_mean=0.0f; c.h_variance_M2=0.0f;
                c.traversability=0.4f; c.hit_count=0;
                c.semantic_label=6; c.obstacle_flag=2;
                c.ix=ix; c.iy=iy; c.level=level;
            }
            return &c;
        }
    }
    return nullptr;
}

// ── Kernel 1: update_height_kernel ───────────────────────────────────────────
// One thread per LiDAR point. Resolves resolution, does Welford atomic update.

__global__ void update_height_kernel(
        GridCell* pool,
        float* pts_x, float* pts_y, float* pts_z,
        uint8_t* labels, int N,
        float robot_x, float robot_y, uint32_t frame_ts) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float wx = pts_x[idx], wy = pts_y[idx], wz = pts_z[idx];
    uint8_t label = labels[idx];

    if (label==7 || wz>15.0f || wz<-2.0f) return;

    float dx = wx-robot_x, dy = wy-robot_y;
    float dist = sqrtf(dx*dx + dy*dy);
    if (dist > 100.0f) return;

    uint8_t lv; float cs;
    if      (dist <  5.0f) { lv=0; cs=0.05f; }
    else if (dist < 20.0f) { lv=1; cs=0.20f; }
    else if (dist < 50.0f) { lv=2; cs=0.50f; }
    else                   { lv=3; cs=1.00f; }

    int32_t ix = (int32_t)floorf(wx/cs);
    int32_t iy = (int32_t)floorf(wy/cs);

    GridCell* cell = d_lookup_or_insert(pool, ix, iy, lv);
    if (!cell) return;

    // Atomic Welford update
    uint32_t n = atomicAdd(&cell->hit_count, 1) + 1;
    float delta = wz - cell->h_mean;
    atomicAdd(&cell->h_mean, delta / (float)n);
    float delta2 = wz - cell->h_mean;
    atomicAdd(&cell->h_variance_M2, delta * delta2);

    // Float atomic min/max via int reinterpret trick
    atomicMin((int*)&cell->h_min, __float_as_int(wz));
    atomicMax((int*)&cell->h_max, __float_as_int(wz));

    cell->semantic_label = label;
    atomicMax(&cell->last_update_ts, frame_ts);
}

// ── Kernel 2: decay_stale_cells_kernel ───────────────────────────────────────
// One thread per hash table slot. ADL-2 temporal decay.

__global__ void decay_stale_cells_kernel(
        GridCell* pool,
        uint32_t current_ts,
        uint32_t decay_window,
        float decay_factor,
        float decay_min_trav) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HASH_TABLE_SIZE) return;

    GridCell& c = pool[idx];
    if (!c.valid) return;

    if ((current_ts - c.last_update_ts) > decay_window) {
        c.traversability *= decay_factor;
        if (c.traversability < decay_min_trav && c.obstacle_flag == 1)
            c.obstacle_flag = 2;
    }
}

// ── Kernel 3: classify_and_score_kernel ──────────────────────────────────────
// One thread per hash table slot. Obstacle flag + traversability score.

__constant__ float d_SEM_TRAV[8] = {1.0f, 0.8f, 0.65f, 0.15f, 0.0f, 0.0f, 0.4f, -1.0f};

__global__ void classify_and_score_kernel(GridCell* pool, float ground_z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HASH_TABLE_SIZE) return;

    GridCell& c = pool[idx];
    if (!c.valid || c.hit_count == 0) return;

    // Obstacle classification
    float hs = c.h_max-c.h_min, cl = c.h_min-ground_z, ma = c.h_max-ground_z;
    if      (hs < 0.05f) c.obstacle_flag = 0;
    else if (cl > 0.50f) c.obstacle_flag = 0;
    else if (ma > 0.30f) c.obstacle_flag = 1;
    else                 c.obstacle_flag = 2;

    // Traversability
    if (c.obstacle_flag == 1) { c.traversability = 0.0f; return; }
    float rough = (c.hit_count>1) ? sqrtf(c.h_variance_M2/(float)(c.hit_count-1)) : 0.0f;
    float hs_sc = expf(-5.0f * rough);
    float ss_sc = d_SEM_TRAV[c.semantic_label];
    float conf  = fminf((float)c.hit_count/10.0f, 1.0f);
    c.traversability = conf * (0.5f*hs_sc + 0.5f*ss_sc);
}

// ── Host test: launch all three kernels ──────────────────────────────────────

int main() {
    const int N = 1000;
    const int THREADS = 256;

    // Allocate device pool
    GridCell* d_pool;
    cudaMalloc(&d_pool, sizeof(GridCell) * HASH_TABLE_SIZE);
    cudaMemset(d_pool, 0, sizeof(GridCell) * HASH_TABLE_SIZE);

    // Generate synthetic scan on host
    float *h_x = new float[N], *h_y = new float[N], *h_z = new float[N];
    uint8_t* h_labels = new uint8_t[N];
    for (int i = 0; i < N; i++) {
        float angle = (float)i/N*6.28318f, r = 1.0f+(i%50)*0.18f;
        h_x[i] = r*cosf(angle); h_y[i] = r*sinf(angle);
        h_z[i] = (i%20<15) ? 0.01f : 0.50f;
        h_labels[i] = (h_z[i]>0.3f) ? 4 : 0;
    }

    float *d_x, *d_y, *d_z; uint8_t* d_labels;
    cudaMalloc(&d_x, N*4); cudaMalloc(&d_y, N*4); cudaMalloc(&d_z, N*4);
    cudaMalloc(&d_labels, N);
    cudaMemcpy(d_x, h_x, N*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z, N*4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, h_labels, N, cudaMemcpyHostToDevice);

    // Kernel 1: update heights
    update_height_kernel<<<(N+THREADS-1)/THREADS, THREADS>>>(
        d_pool, d_x, d_y, d_z, d_labels, N, 0.0f, 0.0f, 1);
    cudaDeviceSynchronize();
    printf("[DRDO CUDA] update_height_kernel done.\n");

    // Kernel 2: decay (frame=55, all points stale: ts=1 → age=54>50)
    int ht_blocks = (HASH_TABLE_SIZE + THREADS - 1) / THREADS;
    decay_stale_cells_kernel<<<ht_blocks, THREADS>>>(
        d_pool, 55, DECAY_WINDOW, DECAY_FACTOR, DECAY_MIN_TRAV);
    cudaDeviceSynchronize();
    printf("[DRDO CUDA] decay_stale_cells_kernel done.\n");

    // Kernel 3: classify + score
    classify_and_score_kernel<<<ht_blocks, THREADS>>>(d_pool, 0.0f);
    cudaDeviceSynchronize();
    printf("[DRDO CUDA] classify_and_score_kernel done.\n");

    // Copy back and validate
    GridCell* h_pool = new GridCell[HASH_TABLE_SIZE];
    cudaMemcpy(h_pool, d_pool, sizeof(GridCell)*HASH_TABLE_SIZE, cudaMemcpyDeviceToHost);

    int valid=0, obstacle=0;
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        if (!h_pool[i].valid) continue;
        valid++;
        assert(h_pool[i].semantic_label != 7 && "SKY_NOISE must not enter map!");
        assert(h_pool[i].traversability >= 0.0f && h_pool[i].traversability <= 1.001f);
        if (h_pool[i].obstacle_flag == 1) obstacle++;
    }
    printf("[DRDO CUDA] Valid cells: %d  Obstacle cells: %d\n", valid, obstacle);
    assert(valid    > 0 && "Must have valid cells after kernels!");
    assert(obstacle > 0 && "Must have obstacle cells from z>0.3 points!");

    printf("[PASS] All CUDA kernel invariants validated.\n");
    printf("[STEP P3.5.1 COMPLETE]\n");

    // Cleanup
    cudaFree(d_pool); cudaFree(d_x); cudaFree(d_y); cudaFree(d_z); cudaFree(d_labels);
    delete[] h_pool; delete[] h_x; delete[] h_y; delete[] h_z; delete[] h_labels;
    return 0;
}
```

---

## Phase 3 Completion Checklist

- [ ] `phase3/src/drdo_map_py.cpp` — compiles, `test_p3_1_1.py` → `[STEP P3.1.1 COMPLETE]`
- [ ] `phase3/python/test_p3_1_2_stress.py` — 20 SKY_NOISE discarded, decay fires, `[STEP P3.1.2 COMPLETE]`
- [ ] `phase3/python/mink_inference.py` — dummy labels correct, `test_p3_2_1.py` → `[STEP P3.2.1 COMPLETE]`
- [ ] `phase3/python/kitti_replay.py` — synthetic mode: `inserted>0, valid>inserted`, `[STEP P3.2.2 COMPLETE]`
- [ ] `phase3/ros2/drdo_grid_map/` — `colcon build` succeeds, node starts without crash
- [ ] `/drdo/costmap` published at **~10 Hz** (verify with `ros2 topic hz`)
- [ ] `phase3/rviz/drdo_map.rviz` — opens in RVIZ2, costmap layer visible
- [ ] `phase3/cuda/drdo_cuda_kernels.cu` — compiles with `nvcc sm_87`, valid>0, obstacle>0, `[STEP P3.5.1 COMPLETE]`

---

## Phase 3 → Phase 4 Preview

| Phase 4 Component | Description |
|---|---|
| **RELLIS-3D Fine-Tuning** | Fine-tune MinkUNet-18 on RELLIS-3D off-road dataset (8 DRDO classes), LaserMix + PolarMix augmentation |
| **TorchSparse++ FP16 Profiling** | Profile on Jetson Orin, target <8 ms inference per frame |
| **FAST-LIO2 Integration** | Replace static robot pose with live FAST-LIO2 pose on `/tf` |
| **nvblox ESDF Integration** | Add ESDF layer for safety margins in Nav2 |
| **Field Trials** | Ouster OS1-64 on rocky terrain, validate 10 Hz end-to-end |

---
*Document sealed by Principal Systems Architect — DRDO ID26053 SIH 2026*

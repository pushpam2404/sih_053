# DRDO ID26053 — Phase 2 Agent Execution Plan
## Temporal Decay · Multi-Resolution Lookup · Shared Header · Gazebo CSV Adaptor

> **Document Type:** Principal Systems Architect's Binding Directive — Phase 2
> **Prerequisite:** Phase 1 COMPLETE — all 12 steps in `phase1/src/` passed with zero assertion failures.
> **Host:** macOS Apple Silicon M4 — g++ / C++17 — NO CUDA (yet)
> **Working Directory for ALL output files:** `~/Desktop/sih/phase2/src/`
> **Compile command for ALL steps:** `g++ -O2 -std=c++17 -I../include <file>.cpp -o <binary> && ./<binary>`
> **Date:** September 2026

---

## Phase 2 Architectural Additions (Locked)

| ID | Feature | Decision |
|---|---|---|
| **P2-1** | Shared header | All Phase 2 files include `drdo_map.h` — single source of truth for structs and constants |
| **P2-2** | Temporal decay kernel | ADL-2 exactly: every K=5 frames, decay cells older than 50 frames by x0.90 |
| **P2-3** | Multi-resolution cross-level query | Given world coords (wx, wy), return the FINEST valid cell covering that point |
| **P2-4** | Map sliding window | When robot moves, re-center window; far cells age out via decay |
| **P2-5** | Gazebo CSV adaptor | Read CSV `x,y,z,label` lines (simulating Gazebo topic) and drive full pipeline |
| **P2-6** | Full integration test | 200-point CSV scene through decay + pipeline, validate all invariants |

---

## Agent Constraints — READ BEFORE PROCEEDING

1. **Every step → one new `.cpp` file** in `~/Desktop/sih/phase2/src/`. Naming: `XX_descriptive_name.cpp`.
2. **Include `../include/drdo_map.h`** at the top of every Phase 2 `.cpp` file (after Step P2.0.1).
3. **Compile command:** `g++ -O2 -std=c++17 -I../include XX_name.cpp -o XX_name && ./XX_name` (run from `phase2/src/`)
4. **Do not proceed** to the next step until all `assert()` gates print `[STEP PX.Y COMPLETE]`.
5. `using namespace std;` in every file. All logic in `main()`. Lambdas are fine. No free functions.
6. Global declarations are allowed ONLY inside `drdo_map.h`.
7. **Phase 1 code is the reference.** Copy any Phase 1 lambda verbatim — it is pre-validated.

---

## BLOCK GROUP P2.0: SETUP

---

### Step P2.0.1: Create Phase 2 Directory + Shared Header

**Objective:** Create the `phase2/` folder tree and write `drdo_map.h` — the single shared header consolidating all structs, constants, and the global pool from Phase 1.

**Execution Directive (shell commands):**
```bash
mkdir -p ~/Desktop/sih/phase2/src
mkdir -p ~/Desktop/sih/phase2/include
mkdir -p ~/Desktop/sih/phase2/bin
mkdir -p ~/Desktop/sih/phase2/data
```

Then write `~/Desktop/sih/phase2/include/drdo_map.h` with this exact content:

```cpp
#pragma once
#include <iostream>
#include <cstring>
#include <cmath>
#include <cassert>
#include <cstdint>
using namespace std;

// Constants
constexpr int   HASH_TABLE_SIZE = 1 << 20;   // 1,048,576 buckets (40 MB)
constexpr int   MAX_PROBE       = 128;
constexpr int   DECAY_WINDOW    = 50;         // frames before decay triggers
constexpr int   DECAY_K         = 5;          // run decay every K frames
constexpr float DECAY_FACTOR    = 0.90f;
constexpr float DECAY_MIN_TRAV  = 0.05f;      // below this -> reset to UNKNOWN

// Semantic traversability lookup (ADL-1)
// 0=GROUND 1=GRAVEL 2=GRASS 3=VEG 4=OBSTACLE 5=WATER 6=UNKNOWN 7=SKY(discard)
constexpr float SEM_TRAV[8] = {1.0f, 0.8f, 0.65f, 0.15f, 0.0f, 0.0f, 0.4f, -1.0f};

struct GridCell {
    float    h_min;
    float    h_max;
    float    h_mean;
    float    h_variance_M2;
    float    traversability;
    uint32_t hit_count;
    uint32_t last_update_ts;
    int32_t  ix;
    int32_t  iy;
    uint8_t  semantic_label;
    uint8_t  obstacle_flag;    // 0=FREE 1=OBSTACLE 2=UNKNOWN
    uint8_t  level;            // 0-3
    bool     valid;
};
static_assert(sizeof(GridCell) == 40, "GridCell must be 40 bytes!");

inline GridCell g_hash_pool[HASH_TABLE_SIZE];

struct LidarPoint { float x, y, z; uint8_t label; };

struct ResolvedCell {
    int     level;
    float   cell_size;
    int32_t ix, iy;
    bool    discard;
};
```

**Self-Validation Gate (bash):**
```bash
[ -s ~/Desktop/sih/phase2/include/drdo_map.h ] \
  && echo "[PASS] drdo_map.h created." \
  || { echo "[FAIL] Header missing!"; exit 1; }
echo "[STEP P2.0.1 COMPLETE]"
```

---

## BLOCK GROUP P2.1: TEMPORAL DECAY KERNEL

---

### Step P2.1: Temporal Decay Kernel Implementation

**Objective:** Implement the ADL-2 temporal decay kernel. Every DECAY_K=5 frames, scan all valid cells. For cells older than DECAY_WINDOW=50 frames, multiply traversability by DECAY_FACTOR=0.90. If traversability drops below DECAY_MIN_TRAV=0.05 AND obstacle_flag==1, reset obstacle_flag to UNKNOWN(2).

**File:** `phase2/src/01_temporal_decay.cpp`

**Include all Phase 1 lambdas** (spatial_hash, insert_cell, lookup_cell) verbatim from Phase 1 code.

**Decay lambda (add after Phase 1 lambdas):**
```cpp
auto decay_kernel = [&](uint32_t current_frame) {
    if (current_frame % DECAY_K != 0) return;
    int decayed = 0;
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        GridCell& c = g_hash_pool[i];
        if (!c.valid) continue;
        if ((current_frame - c.last_update_ts) > (uint32_t)DECAY_WINDOW) {
            c.traversability *= DECAY_FACTOR;
            if (c.traversability < DECAY_MIN_TRAV && c.obstacle_flag == 1)
                c.obstacle_flag = 2; // stale obstacle -> UNKNOWN
            decayed++;
        }
    }
    cout << "  [DECAY] frame=" << current_frame << " -> " << decayed << " cells decayed." << endl;
};
```

**Test scenario:**
1. Insert 5 cells (sA, sB, sC, sD, sE) at `last_update_ts=0`, `traversability=1.0f`, `obstacle_flag=1`.
2. Call `decay_kernel(55)` -> all 5 age = 55 > 50 -> all decay to 0.90.
3. Update sA and sB: `g_hash_pool[sA].last_update_ts = 55; g_hash_pool[sB].last_update_ts = 55;`
4. Call `decay_kernel(60)` -> sA/sB age=5 (NOT > 50) -> skip. sC/sD/sE age=60 > 50 -> decay to 0.81.

**Expected Output:**
```
[Step P2.1] Temporal Decay Kernel
Inserting 5 cells at frame=0 ...
  [DECAY] frame=55 -> 5 cells decayed.
After frame=55 decay:
  Cell A traversability = 0.9000  [decayed from 1.0 x 0.90]
  Cell B traversability = 0.9000
  Cell C traversability = 0.9000
  Cell D traversability = 0.9000
  Cell E traversability = 0.9000
Updating cells A, B with fresh ts=55 ...
  [DECAY] frame=60 -> 3 cells decayed.
After frame=60 decay:
  Cell A traversability = 0.9000  [fresh -- NOT decayed]
  Cell B traversability = 0.9000  [fresh -- NOT decayed]
  Cell C traversability = 0.8100  [decayed again: 0.90 x 0.90]
  Cell D traversability = 0.8100
  Cell E traversability = 0.8100
[PASS] Temporal decay kernel validated.
[STEP P2.1 COMPLETE]
```

**Self-Validation Gate:**
```cpp
for (int idx : {sA, sB, sC, sD, sE})
    assert(fabsf(g_hash_pool[idx].traversability - 0.9f) < 1e-4f); // after frame=55
assert(fabsf(g_hash_pool[sA].traversability - 0.9f)  < 1e-4f && "Fresh cell A must not decay!");
assert(fabsf(g_hash_pool[sB].traversability - 0.9f)  < 1e-4f && "Fresh cell B must not decay!");
assert(fabsf(g_hash_pool[sC].traversability - 0.81f) < 1e-4f && "Stale cell C must decay to 0.81!");
assert(fabsf(g_hash_pool[sD].traversability - 0.81f) < 1e-4f && "Stale cell D must decay to 0.81!");
assert(fabsf(g_hash_pool[sE].traversability - 0.81f) < 1e-4f && "Stale cell E must decay to 0.81!");
cout << "[PASS] Temporal decay kernel validated." << endl;
```

---

## BLOCK GROUP P2.2: MULTI-RESOLUTION CROSS-LEVEL QUERY

---

### Step P2.2: Multi-Resolution Cross-Level World Query

**Objective:** Implement `query_world(wx, wy)` lambda — search levels 0->3 (finest first), return pointer to the finest valid cell covering that world point. Return nullptr if none found.

**File:** `phase2/src/02_multiresolution_query.cpp`

**Cell sizes per level:** `float CELL_SIZES[4] = {0.05f, 0.20f, 0.50f, 1.00f};`

**query_world lambda:**
```cpp
float CELL_SIZES[4] = {0.05f, 0.20f, 0.50f, 1.00f};
auto query_world = [&](float wx, float wy) -> GridCell* {
    for (int lv = 0; lv < 4; lv++) {
        int32_t qix = (int32_t)floorf(wx / CELL_SIZES[lv]);
        int32_t qiy = (int32_t)floorf(wy / CELL_SIZES[lv]);
        GridCell* found = lookup_cell(qix, qiy, (uint8_t)lv);
        if (found) return found;
    }
    return nullptr;
};
```

**Test insertions:**
- Insert cell at L0: `(ix=50, iy=0, level=0)` covering world `(2.5, 0.0)`.
- Insert cell at L2 only: `(ix=60, iy=0, level=2)` covering world `(30.0, 0.0)`.
- Insert cells at BOTH L1 and L2 covering world `(15.0, 0.0)`: L1 -> `ix=75, iy=0`, L2 -> `ix=30, iy=0`.

**Expected Output:**
```
[Step P2.2] Multi-Resolution Cross-Level Query
query(2.5,   0.0) -> L0 cell(50, 0)  [FINEST found: Level 0]
query(30.0,  0.0) -> L2 cell(60, 0)  [FINEST found: Level 2]
query(15.0,  0.0) -> L1 cell(75, 0)  [FINEST found: Level 1, not L2]
query(99.9, 99.9) -> nullptr          [NOT FOUND -- correct]
[PASS] Multi-resolution query validated.
[STEP P2.2 COMPLETE]
```

**Self-Validation Gate:**
```cpp
GridCell* r1 = query_world(2.5f,  0.0f);
GridCell* r2 = query_world(30.0f, 0.0f);
GridCell* r3 = query_world(15.0f, 0.0f);
GridCell* r4 = query_world(99.9f, 99.9f);
assert(r1 != nullptr && r1->level == 0 && r1->ix == 50);
assert(r2 != nullptr && r2->level == 2 && r2->ix == 60);
assert(r3 != nullptr && r3->level == 1);   // must prefer L1 over L2
assert(r4 == nullptr && "Non-existent point must return nullptr!");
cout << "[PASS] Multi-resolution query validated." << endl;
```

---

## BLOCK GROUP P2.3: MAP SLIDING WINDOW

---

### Step P2.3: Robot Pose Update & Active Window Age-Out

**Objective:** Simulate robot moving from (0,0) to (50,0). Far cells (stale ts) decay via the decay kernel. Near fresh cells do not decay.

**File:** `phase2/src/03_sliding_window.cpp`

**Setup:**
- Insert a "near" cell at world ~(1,0), `last_update_ts=95`.
- Insert a "far" cell at world ~(-60,0), `last_update_ts=0`.
- Advance to `current_frame=100`. Run `decay_kernel(100)` and `decay_kernel(100)` is frame 100, which is % 5 == 0, so it runs.
  - Far cell age = 100 - 0 = 100 > DECAY_WINDOW=50 -> decays. Run twice (at frame=100) means it decays once per kernel call. To get 0.81, run at frame=55 AND frame=100: insert at ts=0, call decay at frame=55 (age=55>50, decay once -> 0.90), call decay at frame=100 (age=100>50, decay again -> 0.81).
  - Near cell age = 100 - 95 = 5, NOT > 50 -> no decay.

**Expected Output:**
```
[Step P2.3] Sliding Window Age-Out
Robot moves from (0,0) -> (50,0) at frame=100.
  [DECAY] frame=55 -> 1 cells decayed.
  [DECAY] frame=100 -> 1 cells decayed.
Far cell (world: -60,0) last_ts=0  age=100 -> trav=0.8100 [decayed x2]
Near cell (world: 1,0)  last_ts=95 age=5   -> trav=1.0000 [fresh, not decayed]
[PASS] Sliding window age-out validated.
[STEP P2.3 COMPLETE]
```

**Self-Validation Gate:**
```cpp
assert(fabsf(far_cell->traversability  - 0.81f) < 1e-3f && "Far stale cell must decay twice to 0.81!");
assert(fabsf(near_cell->traversability - 1.0f)  < 1e-4f && "Near fresh cell must not decay!");
cout << "[PASS] Sliding window age-out validated." << endl;
```

---

## BLOCK GROUP P2.4: GAZEBO CSV ADAPTOR

---

### Step P2.4.1: Write Synthetic CSV Scene File

**Objective:** Generate and write `phase2/data/scene_200pts.csv` — 200 labeled LiDAR points simulating a Gazebo world scan. All values are deterministic (no rand()).

**File:** `phase2/src/04_write_csv_scene.cpp`

**Scene composition:**
- 80 GROUND (label=0): 4m x 4m grid at world (5,0), z = i*0.005 - 0.015 (±0.015m)
- 60 OBSTACLE_HARD (label=4): Boulder at (8,0), z = j*0.02 + 0.05 up to 1.25m
- 30 VEGETATION (label=3): Shrub at (20,5), z = k*0.05 + 0.3
- 20 GRAVEL (label=1): Patch at (12,-3), z = m*0.0025 - 0.025
- 5 WATER_MUD (label=5): Puddle at (6,4), z = -0.1 + n*0.01
- 5 SKY_NOISE (label=7): z = 20 + p*2.0 (will be discarded)

**Write with ofstream to `../data/scene_200pts.csv`. Include `<fstream>`. After writing, re-read with ifstream and count rows.**

**Expected Output:**
```
[Step P2.4.1] CSV Scene Writer
Writing 200 points to ../data/scene_200pts.csv ...
  80 GROUND (label=0)
  60 OBSTACLE_HARD (label=4)
  30 VEGETATION (label=3)
  20 GRAVEL (label=1)
   5 WATER_MUD (label=5)
   5 SKY_NOISE (label=7)  <- will be discarded by pipeline
Total: 200 rows written.
Re-reading CSV: 200 rows confirmed.
[PASS] CSV scene file created.
[STEP P2.4.1 COMPLETE]
```

**Self-Validation Gate:**
```cpp
ifstream verify("../data/scene_200pts.csv");
int rows = 0; string line;
getline(verify, line); // skip header
while (getline(verify, line)) rows++;
assert(rows == 200 && "CSV must have exactly 200 data rows!");
cout << "[PASS] CSV scene file created." << endl;
```

---

### Step P2.4.2: CSV Reader & Full Phase 2 Pipeline

**Objective:** Read `scene_200pts.csv`, feed 200 points through the full pipeline in 4 batches of 50 (advancing frame by 5 per batch). Run decay after each batch. Classify and score all valid cells. Validate final map state.

**File:** `phase2/src/05_csv_pipeline_integration.cpp`

**Include:** `#include <fstream>` in addition to `../include/drdo_map.h`

**Include all lambdas from previous steps:** spatial_hash, insert_cell, lookup_cell, mark_free_cell, raycast, resolve_resolution, update_height, classify_obstacle, compute_traversability, decay_kernel, query_world.

**Main loop:**
```cpp
ifstream f("../data/scene_200pts.csv");
string line;
getline(f, line); // skip header
vector<LidarPoint> all_points; // ONLY use std::vector here for reading -- no push_back growth inside the map itself
while (getline(f, line)) {
    LidarPoint p;
    int lbl;
    sscanf(line.c_str(), "%f,%f,%f,%d", &p.x, &p.y, &p.z, &lbl);
    p.label = (uint8_t)lbl;
    all_points.push_back(p);
}

float robot_x = 0.0f, robot_y = 0.0f;
uint32_t current_frame = 0;
for (int batch = 0; batch < 4; batch++) {
    current_frame += 5;
    int start = batch * 50, end = start + 50;
    for (int i = start; i < end; i++) {
        // full pipeline per point (discard -> resolve -> raycast -> update)
    }
    decay_kernel(current_frame);
    cout << "Batch " << batch+1 << "/4: frame=" << current_frame << " decay triggered." << endl;
}
// post-loop: classify + score all valid cells
// then run validation gates
```

**Self-Validation Gate:**
```cpp
for (int i = 0; i < HASH_TABLE_SIZE; i++) {
    if (g_hash_pool[i].valid)
        assert(g_hash_pool[i].semantic_label != 7 && "SKY_NOISE leaked into map!");
}
GridCell* boulder = query_world(8.0f, 0.0f);
assert(boulder != nullptr && boulder->obstacle_flag == 1 && "Boulder must be OBSTACLE!");
int total = 0;
for (int i = 0; i < HASH_TABLE_SIZE; i++) total += g_hash_pool[i].valid;
assert((float)total / HASH_TABLE_SIZE < 0.002f && "Load factor must be < 0.2%!");
cout << "[PASS] CSV pipeline integration validated." << endl;
cout << "[STEP P2.4.2 COMPLETE]" << endl;
```

---

## Phase 2 Completion Checklist

- [x] `phase2/include/drdo_map.h` — shared header, `static_assert(sizeof(GridCell)==40)` compiles clean
- [x] `phase2/src/01_temporal_decay.cpp` — 5-cell decay, fresh/stale separation verified at 0.90 and 0.81
- [x] `phase2/src/02_multiresolution_query.cpp` — finest-level preference, nullptr for missing world point
- [x] `phase2/src/03_sliding_window.cpp` — far cell decayed to 0.81, near fresh cell stays at 1.0
- [x] `phase2/src/04_write_csv_scene.cpp` — 200-row CSV written to disk and row-count verified
- [x] `phase2/src/05_csv_pipeline_integration.cpp` — boulder=OBSTACLE, no SKY_NOISE leak, load < 0.2%

**Compile command (run from `phase2/src/`):**
```bash
g++ -O2 -std=c++17 -I../include XX_name.cpp -o XX_name && ./XX_name
```

---

**Phase 3 Preview:** pybind11 Python bridge -> expose `insert_point()` and `query_world()` to Python -> connect to MinkUNet-18 TorchSparse++ inference -> ROS2 topic publisher -> Jetson AGX Orin CUDA port.

---
*Document sealed by Principal Systems Architect — DRDO ID26053 SIH 2026*

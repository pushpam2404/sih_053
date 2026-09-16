# DRDO ID26053 — Principal Architect's Directive
## Architectural Decision Ledger + Agent Execution Plan (Phase 1)

> **Document Type:** Principal Systems Architect's Binding Directive  
> **Target Agent:** Autonomous Coding Agent (Limited Reasoning)  
> **Host Environment:** macOS — Apple Silicon M4 (MacBook Air) — g++ / C++17 — NO CUDA  
> **Date:** September 2026

---

# PART 1 — ARCHITECTURAL DECISION LEDGER

> These decisions are **final and non-negotiable** for Phase 1. The agent must not deviate from any decision listed here. Each decision is locked with a clear rationale to prevent architectural drift.

---

## ADL-1 | Class Taxonomy & Simulation Environment

**DECISION: LOCKED ✅**

**Semantic Classes (8 total) — DRDO Off-Road Military Terrain:**

| Class ID | Label | Traversability Baseline | Notes |
|---|---|---|---|
| `0` | `GROUND` | 1.0 | Flat, hard surface — road/trail |
| `1` | `GRAVEL_DIRT` | 0.8 | Soft/loose surface — manageable |
| `2` | `GRASS_LOW` | 0.65 | Passable with caution |
| `3` | `VEGETATION_DENSE` | 0.15 | Dense shrubs — likely impassable |
| `4` | `OBSTACLE_HARD` | 0.0 | Rock, wall, concrete barrier |
| `5` | `WATER_MUD` | 0.0 | Bog, puddle, trench |
| `6` | `UNKNOWN` | 0.4 | Unobserved — cautious default |
| `7` | `SKY_NOISE` | -1.0 | Filtered out — not inserted into map |

**Class `7` (SKY_NOISE)** must be filtered at the point ingestion stage before any map update. Any point with `z > 15.0m` OR `z < -2.0m` is automatically assigned label `7` and discarded.

**Simulation Environment for Phase 2 (NOT Phase 1):**  
Use **Gazebo Harmonic** with `gazebo_ros` package and a custom **military terrain world** (rocky terrain heightmap + vegetation models). Do NOT use CARLA — it is urban-optimized and lacks off-road terrain primitives. For Phase 1, all data is **synthetic (dummy arrays)** generated in C++ code.

---

## ADL-2 | Dynamic Obstacle Handling

**DECISION: LOCKED ✅ — MVP = Temporal Decay (No Tracking)**

Full dynamic object tracking (SORT, DeepSORT, Kalman filters) is **explicitly banned for Phase 1**. The MVP approach is:

1. **Every grid cell** stores a `last_update_ts` (uint32 frame counter).
2. A **decay kernel** runs every `K = 5` frames. Any cell whose `last_update_ts` is older than `DECAY_WINDOW = 50` frames has its `traversability` multiplied by `DECAY_FACTOR = 0.90`.
3. If `traversability` drops below `0.05`, `obstacle_flag` is reset to `UNKNOWN (2)`.
4. This means a stationary obstacle **persists** and a removed obstacle **fades in ~25 frames**.

**Rationale:** A pedestrian or vehicle that stops moving for >5 seconds is still a valid obstacle. One that disappears for >5 seconds should revert to unknown. This is the minimum viable dynamic behavior.

---

## ADL-3 | Real-Time Constraints & Model Lock

**DECISION: LOCKED ✅**

| Parameter | Value | Rationale |
|---|---|---|
| **Target frequency** | **10 Hz** (100 ms/frame budget) | Conservative — Jetson AGX Orin must not be overcommitted |
| **Inference model** | **MinkUNet-18 with TorchSparse++** | Best speed/accuracy on Jetson. FP16 mode mandatory. |
| **Voxel size** | **0.05 m** (5 cm) | Sufficient for obstacle detection at vehicle scale |
| **Max range** | **100 m** | Beyond 100 m, discard all points |
| **Min range** | **0.3 m** | Filter LiDAR self-returns below 30 cm |

MinkUNet-34 and SPVNAS are **banned for Phase 1** — they exceed the Jetson latency budget. The model may be upgraded in Phase 2 only if field tests prove 10 Hz is achievable.

---

## ADL-4 | Sensor Fusion & Map Persistence

**DECISION: LOCKED ✅ — Global SLAM is BANNED for Phase 1**

The following are **explicitly prohibited** in Phase 1:
- ❌ Pose-graph optimization (g2o, GTSAM, Ceres)
- ❌ Loop closure detection
- ❌ Global consistent map building
- ❌ IMU pre-integration
- ❌ Multi-session map merging

**What IS allowed:**
- ✅ The robot's pose is received as a **pre-computed transform** (from an external LIO-SAM/FAST-LIO2 node — treated as a black box).
- ✅ The 2.5D grid is maintained in a **local sliding window** centered on the robot (200 m × 200 m at finest resolution zone, extending to 400 m × 400 m at coarse zones).
- ✅ Map is **ephemeral** — it resets on node restart. No persistence to disk in Phase 1.

**Rationale:** The foveated mapping pipeline must be validated in isolation before introducing SLAM drift into the debug loop.

---

## ADL-5 | Memory Allocation Strategy

**DECISION: LOCKED ✅ — Static Pre-Allocated Memory Pool**

**Banned:** `malloc()`, `new[]` (dynamic allocation), `std::vector` push_back growth, `std::unordered_map`, `std::map`, any STL container with heap growth.

**Required Strategy:**

```
HASH_TABLE_SIZE = 2^20 = 1,048,576 buckets
Memory per cell  = 40 bytes (GridCell struct, see Step 1.1)
Total pool size  = 1,048,576 × 40 = 40 MB (pre-allocated at startup)
```

The pool is declared as a **global static C array**: `GridCell g_hash_pool[HASH_TABLE_SIZE]`. This goes into the BSS segment — no runtime allocation overhead, no heap fragmentation.

**Ring Buffer Strategy for the High-Resolution Zone (r < 5 m):**  
As the robot moves, old cells from the previous high-res zone that are now beyond 5 m do NOT need explicit eviction. Since they are superseded by coarser-resolution cells at the same world position, they simply accumulate stale `last_update_ts` values and are overwritten by the decay kernel over time. The hash table's **open-addressing with linear probing and lazy deletion (tombstone flags)** handles this natively.

**Hash Table Load Factor Target:** Keep `occupied_cells / HASH_TABLE_SIZE < 0.70` at all times. If load factor exceeds 0.70, the agent must report a warning. For the Phase 1 MVP with a 200 m × 200 m region at 0.05 m resolution: `(200/0.05)² = 16,000,000` theoretical cells, but only a small fraction (~50,000) near the robot will be active at any time. Load factor will remain well below 0.05%.

---
---

# PART 2 — AGENT EXECUTION PLAN (PHASE 1)

## Agent Constraints — READ BEFORE PROCEEDING

1. **Every step is a separate `.cpp` file.** Do not combine steps.
2. **Always add `using namespace std;`** at the top of every file.
3. **All logic lives in `main()`** unless the step explicitly says "recursion required."
4. **Do not proceed to the next step unless all `assert()` gates pass.**
5. **Compile command for every step:** `g++ -O2 -std=c++17 step_X_Y.cpp -o step_X_Y && ./step_X_Y`
6. Global data declarations (structs, constants, global arrays) are allowed outside `main()`.

---

## ━━━ BLOCK GROUP 0: ENVIRONMENT SETUP ━━━

> These steps are self-contained shell tasks. No `.cpp` files are produced. The agent runs shell commands, checks their output, and uses a bash `test` condition as the validation gate instead of `assert()`.

---

### Step 0.1: Verify C++ Compiler on macOS M4

**Objective:** Confirm that a working C++17-capable compiler is available on the macOS M4 system, installing it via Xcode Command Line Tools if absent, and verify it can compile and run a minimal C++ program.

**Sub-Steps:**
- Check if `g++` is available by running `g++ --version`. On macOS, `g++` is Apple's default C++ compiler bundled with Xcode Command Line Tools — it fully supports C++17 and runs natively on Apple Silicon M4.
- If the command is not found, install Xcode Command Line Tools by running `xcode-select --install` and waiting for the GUI installer to complete. This takes ~2–5 minutes on a fresh machine.
- After confirming `g++` is present, compile and run a one-line smoke test to confirm the full toolchain works end-to-end.
- Print the compiler version string and the smoke test output.
- **g++ is already installed at /usr/bin/g++ (Apple LLVM/Clang wrapper). Use it directly — no Homebrew install needed.

**Execution Directive:**
```bash
# Run these commands in Terminal, one by one:

# 1. Check compiler version
g++ --version

# 2. If the above fails with 'command not found', run:
#    xcode-select --install
#    (Wait for installation to complete, then re-run g++ --version)

# 3. Smoke test — compile and run a minimal C++17 program inline
echo '#include<iostream>\nusing namespace std;\nint main(){cout<<"COMPILER_OK"<<endl;}' | g++ -x c++ -O2 -std=c++17 - -o /tmp/smoke_test && /tmp/smoke_test
```

**Sample Input:** None (shell environment check).

**Expected Output:**
```
Apple clang version 16.x.x (or higher)  ← g++ on macOS calls Apple's clang
Target: arm64-apple-darwin24.x.x
...
COMPILER_OK
```

**Self-Validation Gate:**
```bash
# Gate 1: g++ must exist
which g++ || { echo "[FAIL] g++ not found. Run: xcode-select --install"; exit 1; }

# Gate 2: must be ARM64 (Apple Silicon native)
g++ --version | grep -q "arm64" && echo "[PASS] ARM64 native compiler confirmed." || echo "[WARN] Could not confirm arm64 target — proceeding anyway."

# Gate 3: smoke test must print COMPILER_OK
result=$(echo '#include<iostream>\nusing namespace std;\nint main(){cout<<"COMPILER_OK"<<endl;}' | g++ -x c++ -O2 -std=c++17 - -o /tmp/smoke_test 2>&1 && /tmp/smoke_test)
[ "$result" = "COMPILER_OK" ] && echo "[PASS] Compiler smoke test passed." || { echo "[FAIL] Smoke test failed: $result"; exit 1; }

echo "[STEP 0.1 COMPLETE]"
```

---

### Step 0.2: Create Project Directory Structure

**Objective:** Create the canonical project folder at `~/drdo_id26053/phase1/`, copy both reference documents into it, and confirm the directory is correctly structured before any `.cpp` files are written.

**Sub-Steps:**
- Create the directory `~/drdo_id26053/phase1/` using `mkdir -p` (the `-p` flag creates parent directories and does not error if they already exist).
- Copy both reference documents from the Desktop's `sih` folder into the project directory:
  - `DRDO_ID26053_Agent_Execution_Plan.md` (this file)
  - `DRDO_ID26053_Technical_Blueprint.md`
- Set the working directory to `~/drdo_id26053/phase1/`. **All subsequent `.cpp` files must be created in this directory.**
- Verify: the directory exists and both `.md` files are present with non-zero size.
- Print the directory listing to confirm structure.

**Execution Directive:**
```bash
# 1. Create directory
mkdir -p ~/drdo_id26053/phase1

# 2. Copy reference documents
cp ~/Desktop/sih/DRDO_ID26053_Agent_Execution_Plan.md ~/drdo_id26053/phase1/
cp ~/Desktop/sih/DRDO_ID26053_Technical_Blueprint.md  ~/drdo_id26053/phase1/

# 3. List directory contents to confirm
ls -lh ~/drdo_id26053/phase1/

# 4. Change working directory (the agent must use this path for all subsequent steps)
cd ~/drdo_id26053/phase1
pwd
```

**Sample Input:** None (filesystem operation).

**Expected Output:**
```
total <SIZE>
-rw-r--r--  1 <user>  staff   51K <date>  DRDO_ID26053_Agent_Execution_Plan.md
-rw-r--r--  1 <user>  staff   30K <date>  DRDO_ID26053_Technical_Blueprint.md
/Users/<username>/drdo_id26053/phase1
```

**Self-Validation Gate:**
```bash
# Gate 1: directory must exist
[ -d ~/drdo_id26053/phase1 ] && echo "[PASS] Directory exists." || { echo "[FAIL] Directory not found!"; exit 1; }

# Gate 2: both .md files must be present and non-empty
[ -s ~/drdo_id26053/phase1/DRDO_ID26053_Agent_Execution_Plan.md ] \
  && echo "[PASS] Execution plan present." \
  || { echo "[FAIL] Execution plan missing or empty!"; exit 1; }

[ -s ~/drdo_id26053/phase1/DRDO_ID26053_Technical_Blueprint.md ] \
  && echo "[PASS] Blueprint present." \
  || { echo "[FAIL] Blueprint missing or empty!"; exit 1; }

# Gate 3: confirm we are in the right working directory
[ "$(pwd)" = "$(realpath ~/drdo_id26053/phase1)" ] \
  && echo "[PASS] Working directory is correct." \
  || echo "[WARN] cd into ~/drdo_id26053/phase1 before writing any .cpp files!"

echo "[STEP 0.2 COMPLETE]"
```

---

## ━━━ BLOCK GROUP 1: CORE DATA STRUCTURES ━━━

---

### Step 1.1: GridCell Struct Definition & Static Memory Pool Verification

**Objective:** Define the `GridCell` struct with all required fields in optimal memory-layout order and verify the pre-allocated global memory pool is correctly initialized.

**Sub-Steps:**
- Declare a `GridCell` struct with fields **ordered from largest to smallest alignment** to eliminate compiler padding: all `float` and `uint32_t`/`int32_t` fields (4-byte aligned) first, then all `uint8_t` and `bool` fields (1-byte aligned) last.
- Required fields in order:
  1. `float h_min` — minimum z-height observed in this cell
  2. `float h_max` — maximum z-height observed in this cell
  3. `float h_mean` — running mean (Welford)
  4. `float h_variance_M2` — Welford M2 accumulator (NOT variance directly)
  5. `float traversability` — [0.0, 1.0]
  6. `uint32_t hit_count` — number of points that have updated this cell
  7. `uint32_t last_update_ts` — frame counter of last update
  8. `int32_t ix` — discrete grid x-index at the cell's native resolution
  9. `int32_t iy` — discrete grid y-index at the cell's native resolution
  10. `uint8_t semantic_label` — dominant class ID (0–7)
  11. `uint8_t obstacle_flag` — 0=FREE, 1=OBSTACLE, 2=UNKNOWN
  12. `uint8_t level` — resolution level 0–3
  13. `bool valid` — false = empty bucket (tombstone is a separate flag)
- Declare `const int HASH_TABLE_SIZE = 1 << 20;` globally.
- Declare `GridCell g_hash_pool[HASH_TABLE_SIZE];` globally.
- Inside `main()`:
  - Use `memset(g_hash_pool, 0, sizeof(g_hash_pool))` to zero-initialize the pool.
  - Print `sizeof(GridCell)` — must be exactly **40 bytes**.
  - Print `sizeof(g_hash_pool)` in MB — must be **40 MB** (41,943,040 bytes).
  - Verify that `g_hash_pool[0].valid == false` (zero-initialized).
  - Verify that `g_hash_pool[HASH_TABLE_SIZE - 1].valid == false`.
  - Print a count of valid (non-zero `valid` field) cells — must be **0**.

**Execution Directive:**
```cpp
// Global scope
#include <iostream>
#include <cstring>
#include <cassert>
using namespace std;

const int HASH_TABLE_SIZE = 1 << 20; // 1,048,576

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
    uint8_t  obstacle_flag;
    uint8_t  level;
    bool     valid;
};

GridCell g_hash_pool[HASH_TABLE_SIZE];

int main() {
    memset(g_hash_pool, 0, sizeof(g_hash_pool));
    // ... print and assert statements below
}
```

**Sample Input:** None (initialization only).

**Expected Output:**
```
[Step 1.1] GridCell Struct & Memory Pool
sizeof(GridCell)       = 40 bytes
sizeof(g_hash_pool)    = 41943040 bytes (40.00 MB)
g_hash_pool[0].valid   = 0 (false)
g_hash_pool[last].valid = 0 (false)
Valid cells in pool    = 0
[PASS] Memory pool correctly initialized.
```

**Self-Validation Gate:**
```cpp
assert(sizeof(GridCell) == 40 && "GridCell must be exactly 40 bytes — reorder fields!");
assert(sizeof(g_hash_pool) == (size_t)HASH_TABLE_SIZE * 40);
assert(g_hash_pool[0].valid == false);
assert(g_hash_pool[HASH_TABLE_SIZE - 1].valid == false);
int valid_count = 0;
for (int i = 0; i < HASH_TABLE_SIZE; i++) valid_count += g_hash_pool[i].valid;
assert(valid_count == 0 && "Pool must contain zero valid cells after initialization!");
cout << "[PASS] Memory pool correctly initialized." << endl;
```

---

### Step 1.2: Spatial Hash Function Implementation

**Objective:** Implement the deterministic spatial hash function that maps a `(ix, iy, level)` tuple to a bucket index in `[0, HASH_TABLE_SIZE)` and verify it produces valid indices and adequate distribution.

**Sub-Steps:**
- Implement `spatial_hash(int32_t ix, int32_t iy, uint8_t level)` as a set of operations **inside `main()`** using local lambda or inline code (not a separate function — all logic in `main()`).
- The hash formula uses three **large prime multipliers** to spread bits:
  - `h = (uint64_t)(uint32_t)ix  × 2654435761ULL` — Knuth's multiplicative hash
  - `h ^= (uint64_t)(uint32_t)iy × 805459861ULL`
  - `h ^= (uint64_t)level        × 1234567891ULL`
  - `bucket = h % HASH_TABLE_SIZE`
- **Critical casting rule:** Cast `ix` and `iy` to `uint32_t` BEFORE casting to `uint64_t`. This ensures negative cell indices (world positions behind the robot) hash correctly without signed-integer undefined behaviour.
- Test with **5 specific known inputs** and verify:
  1. Input `(0, 0, 0)` → hash **must be 0** (XOR of zeros = 0, 0 % anything = 0).
  2. Input `(1, 0, 0)` → verify in range `[0, HASH_TABLE_SIZE)`.
  3. Input `(0, 1, 0)` → verify in range, and must **differ** from result of (1,0,0).
  4. Input `(-1, 0, 0)` → verify in range (negative world position, valid DRDO use case).
  5. Input `(100, 200, 2)` → verify in range.
- Run a **collision rate test** over 10,000 random pairs: generate 10,000 unique `(ix, iy, level)` combos, hash them, count how many hash to the same bucket. Print the collision count. It must be **< 50** (< 0.5% collision rate for this small sample).

**Execution Directive:**
```cpp
// Inside main() — auto lambda captures nothing, acts like local function
auto spatial_hash = [](int32_t ix, int32_t iy, uint8_t level) -> int {
    uint64_t h = (uint64_t)(uint32_t)ix  * 2654435761ULL;
    h ^= (uint64_t)(uint32_t)iy * 805459861ULL;
    h ^= (uint64_t)level        * 1234567891ULL;
    return (int)(h % (uint64_t)HASH_TABLE_SIZE);
};
```

**Sample Input:**
```cpp
// Test cases
int h0 = spatial_hash(0, 0, 0);
int h1 = spatial_hash(1, 0, 0);
int h2 = spatial_hash(0, 1, 0);
int h3 = spatial_hash(-1, 0, 0);
int h4 = spatial_hash(100, 200, 2);
```

**Expected Output:**
```
[Step 1.2] Spatial Hash Function
hash(  0,   0, L0) =       0   [VALID]
hash(  1,   0, L0) = <VALUE>   [VALID]
hash(  0,   1, L0) = <VALUE>   [VALID, != h1]
hash( -1,   0, L0) = <VALUE>   [VALID]
hash(100, 200, L2) = <VALUE>   [VALID]
Collision test (10000 unique inputs): <N> collisions
[PASS] Hash function validated.
```

> **Note to agent:** `<VALUE>` will be a number printed by your code. Do not hardcode it — just print the computed value. The assert gates validate correctness, not the exact number.

**Self-Validation Gate:**
```cpp
assert(h0 == 0 && "hash(0,0,0) must be 0!");
assert(h1 >= 0 && h1 < HASH_TABLE_SIZE);
assert(h2 >= 0 && h2 < HASH_TABLE_SIZE);
assert(h3 >= 0 && h3 < HASH_TABLE_SIZE);
assert(h4 >= 0 && h4 < HASH_TABLE_SIZE);
assert(h1 != h2 && "hash(1,0,0) and hash(0,1,0) must differ — hash is too weak!");

// Collision rate test
#include <unordered_set>  // ONLY for this test — not for the map itself
unordered_set<int> seen;
int collisions = 0;
for (int i = 0; i < 100; i++) {
    for (int j = 0; j < 100; j++) {
        int h = spatial_hash(i, j, 0);
        if (seen.count(h)) collisions++;
        seen.insert(h);
    }
}
assert(collisions < 50 && "Collision rate too high — hash function is broken!");
cout << "[PASS] Hash function validated." << endl;
```

---

### Step 1.3: Open-Addressing Hash Table — Linear Probing Insert

**Objective:** Implement the `insert_cell()` operation that places a new, fully initialized `GridCell` into `g_hash_pool` using linear probing to resolve hash collisions.

**Sub-Steps:**
- The hash table uses **open addressing with linear probing**:
  - Compute `bucket = spatial_hash(ix, iy, level)`.
  - Start at `bucket`, probe `bucket+1`, `bucket+2`, ... (wrap around using `% HASH_TABLE_SIZE`).
  - If a slot has `valid == false`, it is **empty** — insert here.
  - If a slot has `valid == true` and `ix == target_ix` AND `iy == target_iy` AND `level == target_level`, the cell **already exists** — do NOT re-insert; return the index of the existing cell.
  - If the probe count exceeds `MAX_PROBE = 128`, print `"[ERROR] Hash table critically full! Increase HASH_TABLE_SIZE."` and `exit(1)`.
- On insert, initialize the new cell: `h_min = +1e9f`, `h_max = -1e9f`, `h_mean = 0`, `h_variance_M2 = 0`, `hit_count = 0`, `traversability = 0.4f` (UNKNOWN default), `obstacle_flag = 2` (UNKNOWN), `valid = true`, `ix = target_ix`, `iy = target_iy`, `level = target_level`.
- The function **returns the integer index** into `g_hash_pool` where the cell lives.
- Implement this as a lambda inside `main()`:
  ```cpp
  auto insert_cell = [&](int32_t ix, int32_t iy, uint8_t level) -> int { ... };
  ```
- Test by inserting 5 cells with known `(ix, iy, level)` tuples, then verifying each returned index is valid and the cell data is correctly initialized.
- Test **collision handling**: insert two cells that hash to the same bucket (you can engineer this by finding two inputs that satisfy `spatial_hash(a_ix, a_iy, 0) == spatial_hash(b_ix, b_iy, 0)`, OR simply force a collision by manually patching the first probe slot as already occupied).

**Execution Directive:**
```cpp
const int MAX_PROBE = 128;

auto insert_cell = [&](int32_t ix, int32_t iy, uint8_t level) -> int {
    int bucket = spatial_hash(ix, iy, level);
    for (int probe = 0; probe < MAX_PROBE; probe++) {
        int slot = (bucket + probe) % HASH_TABLE_SIZE;
        GridCell& c = g_hash_pool[slot];
        // Existing cell match
        if (c.valid && c.ix == ix && c.iy == iy && c.level == level) return slot;
        // Empty slot — insert
        if (!c.valid) {
            c.h_min          = +1e9f;
            c.h_max          = -1e9f;
            c.h_mean         = 0.0f;
            c.h_variance_M2  = 0.0f;
            c.traversability = 0.4f;
            c.hit_count      = 0;
            c.last_update_ts = 0;
            c.semantic_label = 6; // UNKNOWN
            c.obstacle_flag  = 2; // UNKNOWN
            c.ix = ix; c.iy = iy; c.level = level;
            c.valid = true;
            return slot;
        }
    }
    cout << "[ERROR] Hash table critically full!" << endl;
    exit(1);
};
```

**Sample Input:**
```cpp
int s1 = insert_cell(10,  20, 0);
int s2 = insert_cell(10,  21, 0);
int s3 = insert_cell(-5,  15, 1);
int s4 = insert_cell(200, 200, 3);
int s5 = insert_cell(10,  20, 0); // Duplicate — must return s1
```

**Expected Output:**
```
[Step 1.3] Hash Table Insert (Linear Probing)
Inserted (10, 20, L0) at slot=<S1>  valid=1  h_min=1000000000.00  obstacle=UNKNOWN
Inserted (10, 21, L0) at slot=<S2>  valid=1  h_min=1000000000.00  obstacle=UNKNOWN
Inserted (-5, 15, L1) at slot=<S3>  valid=1  h_min=1000000000.00  obstacle=UNKNOWN
Inserted (200,200, L3) at slot=<S4>  valid=1  h_min=1000000000.00  obstacle=UNKNOWN
Re-requested (10, 20, L0) at slot=<S1>  [DUPLICATE DETECTED — same slot as first insert]
[PASS] Hash table insert validated.
```

**Self-Validation Gate:**
```cpp
assert(g_hash_pool[s1].valid == true);
assert(g_hash_pool[s1].ix == 10 && g_hash_pool[s1].iy == 20 && g_hash_pool[s1].level == 0);
assert(g_hash_pool[s1].h_min > 999999900.0f); // Initialized to +1e9
assert(g_hash_pool[s1].h_max < -999999900.0f); // Initialized to -1e9
assert(g_hash_pool[s1].obstacle_flag == 2);    // UNKNOWN
assert(s5 == s1 && "Duplicate insert must return the SAME slot as the original!");
assert(g_hash_pool[s2].ix == 10 && g_hash_pool[s2].iy == 21);
assert(s1 != s2 && s2 != s3 && s3 != s4); // All distinct cells
cout << "[PASS] Hash table insert validated." << endl;
```

---

### Step 1.4: Hash Table Lookup Function

**Objective:** Implement `lookup_cell()` that returns a pointer to an existing `GridCell` (or `nullptr` if the cell does not exist) using the same linear probing strategy.

**Sub-Steps:**
- This is the **read path** (no insertion). It probes the same sequence as `insert_cell()` but never writes.
- If it finds a slot with `valid == true` AND matching `(ix, iy, level)` → return `&g_hash_pool[slot]`.
- If it finds a slot with `valid == false` → the cell does not exist, return `nullptr` immediately (an empty slot means the chain is broken — open addressing guarantees no cell can be past an empty slot if inserted correctly).
- If probe exceeds `MAX_PROBE` → return `nullptr`.
- Implement as a lambda inside `main()`.
- **Important:** After Step 1.3's inserts are still in memory, use `lookup_cell()` to find the 5 cells and verify, then look up a cell that was NEVER inserted and verify `nullptr`.

**Execution Directive:**
```cpp
auto lookup_cell = [&](int32_t ix, int32_t iy, uint8_t level) -> GridCell* {
    int bucket = spatial_hash(ix, iy, level);
    for (int probe = 0; probe < MAX_PROBE; probe++) {
        int slot = (bucket + probe) % HASH_TABLE_SIZE;
        GridCell& c = g_hash_pool[slot];
        if (!c.valid) return nullptr; // Empty slot = end of chain
        if (c.ix == ix && c.iy == iy && c.level == level) return &c;
    }
    return nullptr;
};
```

**Sample Input:**
```cpp
GridCell* p1 = lookup_cell(10, 20, 0);  // Was inserted in Step 1.3
GridCell* p2 = lookup_cell(10, 21, 0);  // Was inserted in Step 1.3
GridCell* pX = lookup_cell(99, 99, 0);  // Was NEVER inserted
```

**Expected Output:**
```
[Step 1.4] Hash Table Lookup
lookup(10,  20, L0) -> address=<ADDR>   [FOUND]   ix=10  iy=20
lookup(10,  21, L0) -> address=<ADDR>   [FOUND]   ix=10  iy=21
lookup(99,  99, L0) -> address=nullptr  [NOT FOUND — correct]
[PASS] Hash table lookup validated.
```

**Self-Validation Gate:**
```cpp
assert(p1 != nullptr && "Cell (10,20,L0) must be found after insertion!");
assert(p2 != nullptr && "Cell (10,21,L0) must be found after insertion!");
assert(pX == nullptr && "Cell (99,99,L0) was never inserted — must return nullptr!");
assert(p1->ix == 10 && p1->iy == 20 && p1->level == 0);
assert(p2->ix == 10 && p2->iy == 21 && p2->level == 0);
cout << "[PASS] Hash table lookup validated." << endl;
```

---

### Step 1.5: Distance-to-Resolution-Level Mapping

**Objective:** Implement the foveation kernel that, given a 3D world point `(wx, wy)` and the robot's current position `(robot_x, robot_y)`, returns the correct resolution `level` (0–3), the `cell_size` in meters, and the discrete `(ix, iy)` grid indices for that point.

**Sub-Steps:**
- Compute Euclidean distance: `dist = sqrt((wx - robot_x)^2 + (wy - robot_y)^2)`.
- Apply the fovea zone table:
  ```
  dist <  5.0m → level=0, cell_size=0.05m  (5 cm — highest resolution)
  dist < 20.0m → level=1, cell_size=0.20m  (20 cm)
  dist < 50.0m → level=2, cell_size=0.50m  (50 cm)
  dist ≤ 100.0m→ level=3, cell_size=1.00m  (1 m — coarsest)
  dist > 100.0m → DISCARD (return level=-1)
  ```
- Compute discrete grid indices: `ix = (int)floor(wx / cell_size)`, `iy = (int)floor(wy / cell_size)`.
- Use `floor()` not integer cast for correct behaviour with **negative world coordinates** (e.g., `wx = -0.03m`: integer cast gives 0, but `floor(-0.03/0.05) = floor(-0.6) = -1` — the correct cell index).
- Test with 6 representative points using robot at position `(0.0, 0.0)`.

**Execution Directive:**
```cpp
#include <cmath>

struct ResolvedCell {
    int     level;
    float   cell_size;
    int32_t ix, iy;
    bool    discard;
};

auto resolve_resolution = [&](float wx, float wy, float robot_x, float robot_y) -> ResolvedCell {
    float dx = wx - robot_x, dy = wy - robot_y;
    float dist = sqrtf(dx*dx + dy*dy);
    ResolvedCell r;
    r.discard = false;
    if      (dist <   5.0f) { r.level = 0; r.cell_size = 0.05f; }
    else if (dist <  20.0f) { r.level = 1; r.cell_size = 0.20f; }
    else if (dist <  50.0f) { r.level = 2; r.cell_size = 0.50f; }
    else if (dist <= 100.0f){ r.level = 3; r.cell_size = 1.00f; }
    else                    { r.discard = true; return r; }
    r.ix = (int32_t)floorf(wx / r.cell_size);
    r.iy = (int32_t)floorf(wy / r.cell_size);
    return r;
};
```

**Sample Input:**
```cpp
// Robot at (0.0, 0.0)
ResolvedCell rc1 = resolve_resolution( 2.0f,  0.0f, 0.0f, 0.0f); // dist=2.0m
ResolvedCell rc2 = resolve_resolution( 0.0f, 12.0f, 0.0f, 0.0f); // dist=12.0m
ResolvedCell rc3 = resolve_resolution(30.0f, 30.0f, 0.0f, 0.0f); // dist=42.4m
ResolvedCell rc4 = resolve_resolution(80.0f,  0.0f, 0.0f, 0.0f); // dist=80.0m
ResolvedCell rc5 = resolve_resolution( -0.03f,0.0f, 0.0f, 0.0f); // dist=0.03m (negative coord)
ResolvedCell rc6 = resolve_resolution(200.0f, 0.0f, 0.0f, 0.0f); // dist=200m  (DISCARD)
```

**Expected Output:**
```
[Step 1.5] Distance-to-Resolution Mapping  (robot at 0,0)
Point ( 2.0,  0.0) dist= 2.00m → Level=0  cell=0.05m  ix= 40  iy=  0
Point ( 0.0, 12.0) dist=12.00m → Level=1  cell=0.20m  ix=  0  iy= 60
Point (30.0, 30.0) dist=42.43m → Level=2  cell=0.50m  ix= 60  iy= 60
Point (80.0,  0.0) dist=80.00m → Level=3  cell=1.00m  ix= 80  iy=  0
Point (-0.03, 0.0) dist= 0.03m → Level=0  cell=0.05m  ix= -1  iy=  0
Point (200.0, 0.0) dist=200.0m → DISCARDED (beyond 100m range)
[PASS] Resolution mapping validated.
```

**Self-Validation Gate:**
```cpp
assert(!rc1.discard && rc1.level == 0 && rc1.cell_size == 0.05f);
assert(rc1.ix == 40 && rc1.iy == 0);    // 2.0/0.05 = 40

assert(!rc2.discard && rc2.level == 1 && rc2.cell_size == 0.20f);
assert(rc2.ix == 0 && rc2.iy == 60);    // 12.0/0.20 = 60

assert(!rc3.discard && rc3.level == 2 && rc3.cell_size == 0.50f);
assert(rc3.ix == 60 && rc3.iy == 60);   // 30.0/0.50 = 60

assert(!rc4.discard && rc4.level == 3 && rc4.cell_size == 1.00f);
assert(rc4.ix == 80 && rc4.iy == 0);    // 80.0/1.00 = 80

assert(!rc5.discard && rc5.level == 0);
assert(rc5.ix == -1);  // floor(-0.03/0.05) = floor(-0.6) = -1 (correct!)
assert(rc5.iy ==  0);

assert(rc6.discard == true && "Point at 200m must be discarded!");
cout << "[PASS] Resolution mapping validated." << endl;
```

---

## ━━━ BLOCK GROUP 2: PROJECTION PIPELINE (CPU REFERENCE) ━━━

---

### Step 2.1: Welford Online Mean & Variance Update

**Objective:** Implement the numerically-stable Welford one-pass algorithm for updating a cell's running mean and M2 accumulator, and verify it produces the correct sample variance after a known sequence of height values.

**Sub-Steps:**
- The Welford algorithm **never stores all values** — it updates in O(1) per sample:
  ```
  n      = hit_count (AFTER incrementing by 1)
  delta  = new_z - h_mean          (old mean)
  h_mean = h_mean + delta / n      (update mean first)
  delta2 = new_z - h_mean          (new mean)
  h_variance_M2 += delta * delta2  (update M2 accumulator)
  ```
- **Sample variance** (for retrieval, not stored): `variance = h_variance_M2 / (hit_count - 1)` when `hit_count >= 2`.
- **Standard deviation**: `stddev = sqrt(variance)`.
- Also update `h_min = min(h_min, new_z)` and `h_max = max(h_max, new_z)` in the same pass.
- Test with exactly **5 hardcoded z-values** whose mean and variance are known analytically.

**Execution Directive:**
```cpp
// Initialize a test cell manually
GridCell test_cell;
memset(&test_cell, 0, sizeof(test_cell));
test_cell.h_min = +1e9f;
test_cell.h_max = -1e9f;

auto update_height = [&](GridCell& c, float new_z) {
    c.hit_count++;
    float delta  = new_z - c.h_mean;
    c.h_mean    += delta / (float)c.hit_count;
    float delta2 = new_z - c.h_mean;
    c.h_variance_M2 += delta * delta2;
    if (new_z < c.h_min) c.h_min = new_z;
    if (new_z > c.h_max) c.h_max = new_z;
};

float test_z[] = {1.0f, 1.2f, 0.8f, 1.1f, 0.9f};
for (float z : test_z) update_height(test_cell, z);
```

**Sample Input:**
```
z values = {1.0, 1.2, 0.8, 1.1, 0.9}
```

**Expected Output:**
```
[Step 2.1] Welford Online Mean & Variance
Input z values: 1.0000  1.2000  0.8000  1.1000  0.9000
After 5 updates:
  h_mean         = 1.0000
  h_variance_M2  = 0.1000
  sample_variance = 0.0250  (M2 / (n-1) = 0.10 / 4)
  sample_stddev   = 0.1581
  h_min          = 0.8000
  h_max          = 1.2000
  hit_count      = 5
[PASS] Welford algorithm validated.
```

> **Derivation check (for agent):**  
> True mean = (1.0+1.2+0.8+1.1+0.9)/5 = 5.0/5 = **1.0000**  
> Sum of squared deviations = (0)²+(0.2)²+(-0.2)²+(0.1)²+(-0.1)² = 0+0.04+0.04+0.01+0.01 = **0.10**  
> Sample variance = 0.10/(5-1) = **0.0250**, stddev = **0.1581**

**Self-Validation Gate:**
```cpp
float sample_var = (test_cell.hit_count > 1)
    ? test_cell.h_variance_M2 / (float)(test_cell.hit_count - 1)
    : 0.0f;
float sample_std = sqrtf(sample_var);

assert(test_cell.hit_count == 5);
assert(fabsf(test_cell.h_mean - 1.0f) < 1e-4f   && "Mean must be 1.0000!");
assert(fabsf(test_cell.h_variance_M2 - 0.1f) < 1e-4f && "M2 must be 0.1000!");
assert(fabsf(sample_var - 0.025f) < 1e-4f        && "Variance must be 0.0250!");
assert(fabsf(sample_std - 0.1581f) < 1e-3f       && "Stddev must be 0.1581!");
assert(fabsf(test_cell.h_min - 0.8f) < 1e-5f);
assert(fabsf(test_cell.h_max - 1.2f) < 1e-5f);
cout << "[PASS] Welford algorithm validated." << endl;
```

---

### Step 2.2: Obstacle Flag Classification Logic

**Objective:** Implement the 4-rule decision tree that classifies each grid cell as `FREE (0)`, `OBSTACLE (1)`, or `UNKNOWN (2)` based on its height statistics relative to a ground plane.

**Sub-Steps:**
- The classification function takes a `GridCell&` and a `ground_z` parameter (the estimated ground elevation at that cell's (ix, iy) position — for Phase 1, assume `ground_z = 0.0f` globally).
- Apply rules **in strict priority order:**
  1. **DISCARD rule:** If `hit_count == 0` → no data, skip (do not modify flag).
  2. **FLAT rule:** If `(h_max - h_min) < 0.05f` → height spread < 5 cm → `FREE (0)`. (e.g., flat road).
  3. **OVERHANG rule:** If `(h_min - ground_z) > 0.50f` → bottom of all returns is >50 cm above ground → robot passes underneath → `FREE (0)`. (e.g., bridge, tree branch).
  4. **OBSTACLE rule:** If `(h_max - ground_z) > 0.30f` → tallest return is >30 cm above ground → `OBSTACLE (1)`. (e.g., rock, wall, person).
  5. **DEFAULT:** `UNKNOWN (2)` (e.g., small bump, pothole — ambiguous).
- Test with **6 cells** covering all decision branches.

**Execution Directive:**
```cpp
auto classify_obstacle = [](GridCell& c, float ground_z) {
    if (c.hit_count == 0) return; // No data
    float h_spread  = c.h_max - c.h_min;
    float clearance = c.h_min - ground_z;
    float max_above = c.h_max - ground_z;
    if      (h_spread  < 0.05f) c.obstacle_flag = 0; // FREE — flat
    else if (clearance > 0.50f) c.obstacle_flag = 0; // FREE — overhang
    else if (max_above > 0.30f) c.obstacle_flag = 1; // OBSTACLE
    else                        c.obstacle_flag = 2; // UNKNOWN
};
```

**Sample Input:**
```cpp
// Cell A: flat ground (h_spread < 0.05) → FREE
GridCell cA; cA.hit_count=5; cA.h_min=0.01f; cA.h_max=0.04f; // spread=0.03

// Cell B: dense rock (h_max - 0.0 > 0.30, h_min near ground) → OBSTACLE
GridCell cB; cB.hit_count=8; cB.h_min=0.05f; cB.h_max=0.80f; // spread=0.75, above=0.80

// Cell C: bridge overhang (h_min > 0.50) → FREE
GridCell cC; cC.hit_count=3; cC.h_min=0.60f; cC.h_max=1.50f; // spread=0.90, clearance=0.60

// Cell D: small pothole/bump (not flat, max<0.30) → UNKNOWN
GridCell cD; cD.hit_count=4; cD.h_min=0.00f; cD.h_max=0.20f; // spread=0.20, max=0.20

// Cell E: tree trunk (h_min=0.02, h_max=2.0) → OBSTACLE
GridCell cE; cE.hit_count=10; cE.h_min=0.02f; cE.h_max=2.00f;

// Cell F: no data → flag UNCHANGED (stays 2=UNKNOWN by default)
GridCell cF; cF.hit_count=0; cF.obstacle_flag=2;

float ground_z = 0.0f;
classify_obstacle(cA, ground_z);
classify_obstacle(cB, ground_z);
classify_obstacle(cC, ground_z);
classify_obstacle(cD, ground_z);
classify_obstacle(cE, ground_z);
classify_obstacle(cF, ground_z);
```

**Expected Output:**
```
[Step 2.2] Obstacle Flag Classification
Cell A: h_min=0.01 h_max=0.04 spread=0.03 → Flag=0 (FREE — flat ground)
Cell B: h_min=0.05 h_max=0.80 above=0.80  → Flag=1 (OBSTACLE — rock)
Cell C: h_min=0.60 h_max=1.50 clear=0.60  → Flag=0 (FREE — overhang)
Cell D: h_min=0.00 h_max=0.20 above=0.20  → Flag=2 (UNKNOWN — bump)
Cell E: h_min=0.02 h_max=2.00 above=2.00  → Flag=1 (OBSTACLE — trunk)
Cell F: hit_count=0                        → Flag=2 (UNKNOWN — no data, unchanged)
[PASS] Obstacle classification validated.
```

**Self-Validation Gate:**
```cpp
assert(cA.obstacle_flag == 0 && "Cell A (flat): must be FREE!");
assert(cB.obstacle_flag == 1 && "Cell B (rock): must be OBSTACLE!");
assert(cC.obstacle_flag == 0 && "Cell C (bridge): must be FREE (overhang)!");
assert(cD.obstacle_flag == 2 && "Cell D (bump): must be UNKNOWN!");
assert(cE.obstacle_flag == 1 && "Cell E (trunk): must be OBSTACLE!");
assert(cF.obstacle_flag == 2 && "Cell F (no data): flag must remain UNKNOWN!");
cout << "[PASS] Obstacle classification validated." << endl;
```

---

### Step 2.3: Traversability Score Computation

**Objective:** Implement the traversability score function that combines height roughness, semantic class, and observation confidence into a single `[0.0, 1.0]` score.

**Sub-Steps:**
- **If `obstacle_flag == 1` (OBSTACLE):** Traversability is forced to `0.0f` — shortcut, skip all math.
- **Height score** (penalizes rough surfaces):
  ```
  roughness = sqrt(h_variance_M2 / max(hit_count - 1, 1))   // sample std dev
  height_score = exp(-5.0 * roughness)
  ```
  Rationale: A cell with `roughness = 0` (perfectly flat) gets `exp(0) = 1.0`. A rough cell (`roughness = 0.5`) gets `exp(-2.5) = 0.082`.
- **Semantic score** (lookup table by class ID):
  ```cpp
  float SEM_TRAV[8] = {1.0f, 0.8f, 0.65f, 0.15f, 0.0f, 0.0f, 0.4f, -1.0f};
  //                   GND  GRVL  GRASS  VEG   OBS  WATER UNK  SKY
  ```
- **Confidence weight** (scales down score for cells with few observations):
  ```
  conf = min(hit_count / 10.0f, 1.0f)
  ```
  A cell with ≥10 hits has full confidence.
- **Final score:**
  ```
  traversability = conf × (0.5 × height_score + 0.5 × sem_score)
  ```
- Test with **3 cells** and verify computed values match the formula.

**Execution Directive:**
```cpp
#include <cmath>
const float SEM_TRAV[8] = {1.0f, 0.8f, 0.65f, 0.15f, 0.0f, 0.0f, 0.4f, -1.0f};

auto compute_traversability = [&](GridCell& c) {
    if (c.obstacle_flag == 1) { c.traversability = 0.0f; return; }
    float roughness    = (c.hit_count > 1)
        ? sqrtf(c.h_variance_M2 / (float)(c.hit_count - 1))
        : 0.0f;
    float height_score = expf(-5.0f * roughness);
    float sem_score    = SEM_TRAV[c.semantic_label];
    float conf         = fminf((float)c.hit_count / 10.0f, 1.0f);
    c.traversability   = conf * (0.5f * height_score + 0.5f * sem_score);
};
```

**Sample Input:**
```cpp
// Cell 1: Flat GROUND with 10 hits → max confidence, smooth
GridCell t1;
t1.obstacle_flag = 0; t1.semantic_label = 0; // GROUND
t1.hit_count = 10; t1.h_variance_M2 = 0.0f;  // perfectly flat
// Expected: roughness=0, height_score=1.0, sem=1.0, conf=1.0, trav=1.0

// Cell 2: Rough GRASS with 5 hits
GridCell t2;
t2.obstacle_flag = 0; t2.semantic_label = 2; // GRASS
t2.hit_count = 5;  t2.h_variance_M2 = 0.1f;  // stddev=sqrt(0.1/4)=0.1581
// roughness=0.1581, height_score=exp(-5*0.1581)=exp(-0.7906)=0.4537
// sem=0.65, conf=0.5, trav=0.5*(0.5*0.4537 + 0.5*0.65)=0.5*0.5519=0.2759

// Cell 3: OBSTACLE cell → forced 0.0
GridCell t3;
t3.obstacle_flag = 1; t3.semantic_label = 4; // OBSTACLE_HARD
t3.hit_count = 20; t3.h_variance_M2 = 5.0f;

compute_traversability(t1);
compute_traversability(t2);
compute_traversability(t3);
```

**Expected Output:**
```
[Step 2.3] Traversability Score Computation
Cell 1 (GROUND, flat, 10 hits):
  roughness=0.0000  height_score=1.0000  sem=1.0000  conf=1.0000
  traversability = 1.0000

Cell 2 (GRASS, rough, 5 hits):
  roughness=0.1581  height_score=0.4537  sem=0.6500  conf=0.5000
  traversability = 0.2759

Cell 3 (OBSTACLE, forced):
  traversability = 0.0000  [obstacle_flag=1 → forced to 0]

[PASS] Traversability computation validated.
```

**Self-Validation Gate:**
```cpp
assert(fabsf(t1.traversability - 1.0f)   < 1e-3f && "Cell 1 traversability must be 1.0000!");
assert(fabsf(t2.traversability - 0.2759f) < 1e-3f && "Cell 2 traversability must be 0.2759!");
assert(fabsf(t3.traversability - 0.0f)   < 1e-5f && "Cell 3 (obstacle) must be 0.0000!");
cout << "[PASS] Traversability computation validated." << endl;
```

---

### Step 2.4: Bresenham 2D Raycasting — Mark-Free Sweep

**Objective:** Implement the Bresenham line-drawing algorithm to trace a LiDAR ray from the robot origin to an obstacle hit point, marking all intermediate grid cells as `FREE` before registering the hit.

**Sub-Steps:**
- The function takes: origin `(ox, oy)` and hit point `(hx, hy)` **in discrete grid indices** (already resolved to the correct resolution level).
- It walks all cells between `(ox, oy)` and `(hx, hy)` (exclusive of endpoint) and calls `mark_free(ix, iy, level)` on each.
- The **endpoint cell** is NOT marked free — it receives the height update (potential obstacle).
- **Standard Bresenham algorithm (integer, no floating point):**
  ```
  dx = abs(hx - ox), dy = abs(hy - oy)
  sx = sign(hx - ox), sy = sign(hy - oy)
  err = dx - dy
  loop:
    if (x == hx && y == hy) → STOP (this is the hit point, do not mark free)
    mark_free(x, y, level)
    e2 = 2 * err
    if (e2 > -dy): err -= dy; x += sx
    if (e2 <  dx): err += dx; y += sy
  ```
- `mark_free(ix, iy, level)`: If a cell exists in the hash table AND its `obstacle_flag == 1`, downgrade it to `UNKNOWN (2)` (a ray passed through it — it's not a solid obstacle). If cell doesn't exist, insert it as FREE.
- Trace one hardcoded ray and print every cell it passes through.

**Execution Directive:**
```cpp
// mark_free: if cell exists and is OBSTACLE, reset to UNKNOWN
// if cell doesn't exist, create it as FREE
auto mark_free_cell = [&](int32_t ix, int32_t iy, uint8_t level) {
    int slot = insert_cell(ix, iy, level);  // returns existing or creates new
    GridCell& c = g_hash_pool[slot];
    if (c.obstacle_flag == 1) c.obstacle_flag = 2; // OBSTACLE → UNKNOWN
    if (c.obstacle_flag == 2 && c.hit_count == 0) c.obstacle_flag = 0; // New cell → FREE
};

// Bresenham raycasting
auto raycast = [&](int32_t ox, int32_t oy, int32_t hx, int32_t hy, uint8_t level) {
    int dx = abs(hx - ox), dy = abs(hy - oy);
    int sx = (hx > ox) ? 1 : -1;
    int sy = (hy > oy) ? 1 : -1;
    int err = dx - dy;
    int x = ox, y = oy;
    while (true) {
        if (x == hx && y == hy) break; // Hit point — stop, do NOT mark free
        cout << "  Marking cell (" << x << ", " << y << ") as FREE" << endl;
        mark_free_cell((int32_t)x, (int32_t)y, level);
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x += sx; }
        if (e2 <  dx) { err += dx; y += sy; }
    }
    cout << "  Hit cell (" << hx << ", " << hy << ") → height update zone" << endl;
};
```

**Sample Input:**
```cpp
// Ray from grid origin (0,0) to hit point (3,2) at level 0
raycast(0, 0, 3, 2, 0);
```

**Expected Output:**
```
[Step 2.4] Bresenham 2D Raycasting
Ray: origin=(0,0) → hit=(3,2)  Level=0
  Marking cell (0, 0) as FREE
  Marking cell (1, 1) as FREE
  Marking cell (2, 1) as FREE
  Hit cell (3, 2) → height update zone
Cells visited (excluding hit): 3
[PASS] Raycasting validated.
```

> **Derivation trace** (for agent verification):
> `dx=3, dy=2, sx=1, sy=1, err=1`
> - `(0,0)`: mark free; `e2=2>-2→err=1-2=-1,x=1`; `e2=2<3→err=-1+3=2,y=1`
> - `(1,1)`: mark free; `e2=4>-2→err=2-2=0,x=2`; `e2=4<3? No`
> - `(2,1)`: mark free; `e2=0>-2→err=0-2=-2,x=3`; `e2=0<3→err=-2+3=1,y=2`
> - `(3,2)`: equals hit → STOP

**Self-Validation Gate:**
```cpp
// Verify the 3 free cells were inserted and are free
GridCell* rA = lookup_cell(0, 0, 0);
GridCell* rB = lookup_cell(1, 1, 0);
GridCell* rC = lookup_cell(2, 1, 0);
GridCell* rH = lookup_cell(3, 2, 0); // Hit cell — should NOT exist yet (not inserted by raycast)

assert(rA != nullptr && "Cell (0,0) must exist after raycasting!");
assert(rB != nullptr && "Cell (1,1) must exist after raycasting!");
assert(rC != nullptr && "Cell (2,1) must exist after raycasting!");
assert(rA->obstacle_flag == 0 && "Cell (0,0) must be FREE after ray sweep!");
assert(rB->obstacle_flag == 0 && "Cell (1,1) must be FREE after ray sweep!");
assert(rC->obstacle_flag == 0 && "Cell (2,1) must be FREE after ray sweep!");
// Note: rH may be null (raycast does not insert hit cell — that's the update_height step)
cout << "[PASS] Raycasting validated." << endl;
```

---

## ━━━ BLOCK GROUP 3: DUMMY PIPELINE INTEGRATION ━━━

---

### Step 3.1: Synthetic Point Cloud Generator

**Objective:** Generate a hardcoded synthetic point cloud of 20 labeled LiDAR returns representing a simple scene — a flat ground plane with a rock obstacle — that will drive the full pipeline test in Step 3.2.

**Sub-Steps:**
- Declare a struct `LidarPoint { float x, y, z; uint8_t label; }`.
- Generate exactly **20 points** using hardcoded values (no random numbers — deterministic scene):
  - **10 ground points** (label `0` = GROUND): scattered on a 2 m × 2 m patch centered at `(5, 0)`, all at `z = 0.0 ± 0.02m`.
  - **7 obstacle points** (label `4` = OBSTACLE_HARD): clustered at `(7, 0)`, heights from `z = 0.05` to `z = 0.80m`, simulating a rock.
  - **2 vegetation points** (label `3` = VEGETATION_DENSE): at `(15, 3)`, heights ~0.5–1.2m.
  - **1 sky/noise point** (label `7` = SKY_NOISE): at `(10, 0)`, `z = 25.0m` — must be DISCARDED.
- Print all 20 points in a table format.
- Robot position: `(0.0, 0.0)`.

**Execution Directive:**
```cpp
struct LidarPoint { float x, y, z; uint8_t label; };

LidarPoint cloud[] = {
    // Ground points (label=0) near (5,0)
    { 4.8f, -0.2f, 0.02f,  0 },
    { 4.9f,  0.1f, 0.00f,  0 },
    { 5.0f,  0.0f, 0.01f,  0 },
    { 5.1f,  0.2f,-0.01f,  0 },
    { 5.2f, -0.1f, 0.00f,  0 },
    { 5.0f,  0.5f, 0.02f,  0 },
    { 5.0f, -0.5f,-0.02f,  0 },
    { 4.5f,  0.3f, 0.01f,  0 },
    { 5.5f,  0.0f, 0.00f,  0 },
    { 5.3f, -0.3f, 0.01f,  0 },
    // Rock obstacle (label=4) near (7,0)
    { 7.0f,  0.0f, 0.05f,  4 },
    { 7.0f,  0.0f, 0.25f,  4 },
    { 7.0f,  0.0f, 0.50f,  4 },
    { 7.0f,  0.0f, 0.80f,  4 },
    { 7.1f,  0.1f, 0.40f,  4 },
    { 6.9f, -0.1f, 0.60f,  4 },
    { 7.0f,  0.0f, 0.70f,  4 },
    // Vegetation (label=3) at (15,3)
    { 15.0f, 3.0f, 0.50f,  3 },
    { 15.0f, 3.0f, 1.20f,  3 },
    // Sky noise (label=7) → DISCARD
    { 10.0f, 0.0f, 25.0f,  7 },
};
const int N_POINTS = 20;
```

**Expected Output:**
```
[Step 3.1] Synthetic Point Cloud
 #   x      y      z       label
 0   4.80  -0.20   0.020    0 (GROUND)
 1   4.90   0.10   0.000    0 (GROUND)
 2   5.00   0.00   0.010    0 (GROUND)
...
17  15.00   3.00   0.500    3 (VEGETATION)
18  15.00   3.00   1.200    3 (VEGETATION)
19  10.00   0.00  25.000    7 (SKY_NOISE → DISCARD)
Total points: 20  (19 valid, 1 discarded)
[PASS] Point cloud generated.
```

**Self-Validation Gate:**
```cpp
assert(N_POINTS == 20);
int sky_count = 0;
for (int i = 0; i < N_POINTS; i++) sky_count += (cloud[i].label == 7);
assert(sky_count == 1 && "Must have exactly 1 sky/noise point!");
int ground_count = 0, obs_count = 0, veg_count = 0;
for (int i = 0; i < N_POINTS; i++) {
    if (cloud[i].label == 0) ground_count++;
    if (cloud[i].label == 4) obs_count++;
    if (cloud[i].label == 3) veg_count++;
}
assert(ground_count == 10 && obs_count == 7 && veg_count == 2);
cout << "[PASS] Point cloud generated." << endl;
```

---

### Step 3.2: Full Pipeline Integration Test

**Objective:** Feed all 20 synthetic points through the complete pipeline — discard filter → resolution mapping → raycasting → height update → obstacle classification → traversability scoring — and verify the final state of the hash table reflects the expected physical scene.

**Sub-Steps:**
- Set robot position to `(0.0f, 0.0f)` and frame timestamp to `1`.
- Loop over all 20 points. For each point:
  1. **Discard filter:** If `label == 7` OR `z > 15.0f` OR `z < -2.0f` → `continue`.
  2. **Resolution mapping:** Call `resolve_resolution(wx, wy, robot_x, robot_y)`. If `discard == true` → `continue`.
  3. **Raycasting:** Convert robot position to grid indices at the resolved level: `rox = floor(robot_x / cell_size)`, `roy = floor(robot_y / cell_size)`. Call `raycast(rox, roy, rc.ix, rc.iy, rc.level)`.
  4. **Height update:** Call `insert_cell(rc.ix, rc.iy, rc.level)` to get the hit cell. Call `update_height(cell, point.z)`. Set `cell.semantic_label = point.label`. Set `cell.last_update_ts = 1`.
  5. **Classify + score (post-loop):** After all points are inserted, loop over ALL valid cells in the hash table and call `classify_obstacle(cell, 0.0f)` then `compute_traversability(cell)`.
- After the pipeline: Print all valid cells with their final state.

**Execution Directive:**
```cpp
float robot_x = 0.0f, robot_y = 0.0f;
uint32_t frame_ts = 1;

for (int i = 0; i < N_POINTS; i++) {
    LidarPoint& p = cloud[i];
    
    // Step 1: Discard filter
    if (p.label == 7 || p.z > 15.0f || p.z < -2.0f) {
        cout << "  [DISCARD] Point " << i << " (label=7 or z out of range)" << endl;
        continue;
    }
    
    // Step 2: Resolve resolution
    ResolvedCell rc = resolve_resolution(p.x, p.y, robot_x, robot_y);
    if (rc.discard) continue;
    
    // Step 3: Raycast (robot origin in grid coords)
    int32_t rox = (int32_t)floorf(robot_x / rc.cell_size);
    int32_t roy = (int32_t)floorf(robot_y / rc.cell_size);
    raycast(rox, roy, rc.ix, rc.iy, (uint8_t)rc.level);
    
    // Step 4: Update hit cell
    int slot = insert_cell(rc.ix, rc.iy, (uint8_t)rc.level);
    GridCell& c = g_hash_pool[slot];
    update_height(c, p.z);
    c.semantic_label  = p.label;
    c.last_update_ts  = frame_ts;
    
    cout << "  [INSERT] Point " << i << " → cell(" << rc.ix << "," << rc.iy
         << ") L" << rc.level << "  z=" << p.z << endl;
}

// Step 5: Post-loop classify + score all valid cells
int valid_cells = 0;
for (int i = 0; i < HASH_TABLE_SIZE; i++) {
    if (!g_hash_pool[i].valid) continue;
    valid_cells++;
    classify_obstacle(g_hash_pool[i], 0.0f);
    compute_traversability(g_hash_pool[i]);
}
cout << "\nTotal valid cells after pipeline: " << valid_cells << endl;
```

**Expected Output (abbreviated):**
```
[Step 3.2] Full Pipeline Integration
  [DISCARD] Point 19 (label=7 or z out of range)
  [INSERT] Point  0 → cell(96,-4) L0  z=0.020
  [INSERT] Point  1 → cell(98, 2) L0  z=0.000
  ...
  [INSERT] Point 17 → cell(30, 6) L1  z=0.500
  [INSERT] Point 18 → cell(30, 6) L1  z=1.200
  [INSERT] Point 11 → cell(140, 0) L0  z=0.050
  ...
Total valid cells after pipeline: <N_VALID>
  (includes both hit cells AND cells marked FREE by raycasting)
[PASS] Pipeline integration complete.
```

**Self-Validation Gate:**
```cpp
// 1. Exactly 1 point was discarded (the sky noise point)
// 2. The rock cell (near world coord (7,0)) must be OBSTACLE
//    → (7.0 / 0.05 = 140.0 → ix=140), (0.0 / 0.05 = 0 → iy=0)
GridCell* rock_cell = lookup_cell(140, 0, 0);
assert(rock_cell != nullptr && "Rock cell at (140,0,L0) must exist!");
assert(rock_cell->h_max > 0.79f && "Rock h_max must be close to 0.80m!");
assert(rock_cell->obstacle_flag == 1 && "Rock cell must be OBSTACLE!");
assert(rock_cell->traversability == 0.0f && "Rock cell traversability must be 0.0!");

// 3. Ground cells must be FREE
// Ground point (5.0, 0.0) → ix = floor(5.0/0.05)=100, iy=floor(0.0/0.05)=0
GridCell* gnd_cell = lookup_cell(100, 0, 0);
assert(gnd_cell != nullptr && "Ground cell (100,0,L0) must exist!");
assert(gnd_cell->obstacle_flag == 0 && "Ground cell must be FREE!");
assert(gnd_cell->traversability > 0.0f && "Ground traversability must be > 0!");

// 4. Total valid cells must be > 19 (points) due to free cells from raycasting
int final_count = 0;
for (int i = 0; i < HASH_TABLE_SIZE; i++) final_count += g_hash_pool[i].valid;
assert(final_count > 19 && "Must have more cells than points due to raycasting!");
assert(final_count < 500 && "Sanity: should not have > 500 cells for this small scene!");

cout << "[PASS] Pipeline integration validated." << endl;
```

---

### Step 3.3: Grid State Printer & Final Validation Report

**Objective:** Implement a human-readable console printer that dumps the final state of all valid grid cells sorted by obstacle type, prints a summary statistics table, and performs a final global invariant check over the entire hash pool.

**Sub-Steps:**
- Loop over the full hash pool and collect all valid cells into three **separate count arrays** (do NOT use `std::vector` — use fixed-size C arrays of size 512).
- Count:
  - `n_free`: cells with `obstacle_flag == 0`
  - `n_obstacle`: cells with `obstacle_flag == 1`
  - `n_unknown`: cells with `obstacle_flag == 2`
- Print each category with its cells' ix, iy, level, h_mean, h_max, traversability.
- Print a summary line: `"Scene: N_FREE free | N_OBS obstacles | N_UNK unknown | N_TOTAL total"`.
- Print the **load factor**: `total_valid / HASH_TABLE_SIZE` (should be << 1.0).
- Check global invariants:
  1. All `valid` cells have `hit_count >= 0`.
  2. All `valid` cells have `h_max >= h_min` (or both are init values).
  3. All `valid` cells have `traversability >= 0.0f && <= 1.0f`.
  4. All `valid` cells have `level` in range `[0, 3]`.
  5. No `valid` cell has `semantic_label == 7` (sky noise must never reach the map).

**Execution Directive:**
```cpp
int n_free = 0, n_obs = 0, n_unk = 0;
cout << "\n=== GRID STATE REPORT ===" << endl;
cout << "--- FREE CELLS ---" << endl;
for (int i = 0; i < HASH_TABLE_SIZE; i++) {
    GridCell& c = g_hash_pool[i];
    if (!c.valid) continue;
    if (c.obstacle_flag == 0) {
        n_free++;
        cout << "  FREE  L" << (int)c.level
             << " cell(" << c.ix << "," << c.iy
             << ")  h_mean=" << c.h_mean
             << "  trav=" << c.traversability << endl;
    }
}
cout << "--- OBSTACLE CELLS ---" << endl;
for (int i = 0; i < HASH_TABLE_SIZE; i++) {
    GridCell& c = g_hash_pool[i];
    if (!c.valid || c.obstacle_flag != 1) continue;
    n_obs++;
    cout << "  OBS   L" << (int)c.level
         << " cell(" << c.ix << "," << c.iy
         << ")  h_max=" << c.h_max
         << "  trav=" << c.traversability << endl;
}
cout << "--- UNKNOWN CELLS ---" << endl;
for (int i = 0; i < HASH_TABLE_SIZE; i++) {
    GridCell& c = g_hash_pool[i];
    if (!c.valid || c.obstacle_flag != 2) continue;
    n_unk++;
    cout << "  UNK   L" << (int)c.level
         << " cell(" << c.ix << "," << c.iy << ")" << endl;
}
int n_total = n_free + n_obs + n_unk;
float load_factor = (float)n_total / HASH_TABLE_SIZE;
cout << "\nScene: " << n_free << " free | " << n_obs << " obstacles | "
     << n_unk << " unknown | " << n_total << " total" << endl;
cout << "Load factor: " << load_factor << " (" << n_total << "/" << HASH_TABLE_SIZE << ")" << endl;
```

**Expected Output:**
```
=== GRID STATE REPORT ===
--- FREE CELLS ---
  FREE  L0 cell(96,-4)  h_mean=0.0200  trav=<VAL>
  FREE  L0 cell(98, 2)  h_mean=0.0000  trav=<VAL>
  ... (raycasted FREE cells along each ray)
--- OBSTACLE CELLS ---
  OBS   L0 cell(140, 0)  h_max=0.8000  trav=0.0000
--- UNKNOWN CELLS ---
  UNK   L0 cell(...)
  ...
Scene: <N> free | 1 obstacles | <M> unknown | <TOTAL> total
Load factor: 0.000031 (33/1048576)
[PASS] All global invariants satisfied. Phase 1 COMPLETE.
```

**Self-Validation Gate:**
```cpp
// Global invariant scan
bool invariants_ok = true;
for (int i = 0; i < HASH_TABLE_SIZE; i++) {
    GridCell& c = g_hash_pool[i];
    if (!c.valid) continue;
    
    if (c.hit_count > 0 && c.h_max < c.h_min) {
        cout << "[FAIL] Cell at slot " << i << " has h_max < h_min!" << endl;
        invariants_ok = false;
    }
    if (c.traversability < 0.0f || c.traversability > 1.001f) {
        cout << "[FAIL] Cell at slot " << i << " traversability out of [0,1]!" << endl;
        invariants_ok = false;
    }
    if (c.level > 3) {
        cout << "[FAIL] Cell at slot " << i << " has invalid level " << (int)c.level << "!" << endl;
        invariants_ok = false;
    }
    if (c.semantic_label == 7) {
        cout << "[FAIL] SKY_NOISE point leaked into map at slot " << i << "!" << endl;
        invariants_ok = false;
    }
}

assert(invariants_ok && "One or more global invariants failed! See output above.");
assert(n_obs >= 1 && "Must have at least 1 obstacle cell from the rock points!");
assert(load_factor < 0.001f && "Load factor must be well below 0.1% for this test scene!");
assert(n_total == n_free + n_obs + n_unk);

cout << "[PASS] All global invariants satisfied." << endl;
cout << "=====================================" << endl;
cout << "  PHASE 1 COMPLETE — AGENT HANDOFF   " << endl;
cout << "=====================================" << endl;
```

---

## Phase 1 Completion Checklist

The agent must confirm **ALL** of the following before declaring Phase 1 done:

- [ ] `step_1_1.cpp` compiles and all asserts pass — `sizeof(GridCell) == 40`
- [ ] `step_1_2.cpp` compiles and all asserts pass — hash(0,0,0)==0, collision<50
- [ ] `step_1_3.cpp` compiles and all asserts pass — duplicate insert returns same slot
- [ ] `step_1_4.cpp` compiles and all asserts pass — lookup returns nullptr for missing cell
- [ ] `step_1_5.cpp` compiles and all asserts pass — negative ix=-1 for x=-0.03m
- [ ] `step_2_1.cpp` compiles and all asserts pass — Welford mean=1.0, var=0.025
- [ ] `step_2_2.cpp` compiles and all asserts pass — all 6 flag classifications correct
- [ ] `step_2_3.cpp` compiles and all asserts pass — trav(t1)=1.0, trav(t2)=0.2759
- [ ] `step_2_4.cpp` compiles and all asserts pass — cells (0,0),(1,1),(2,1) are FREE
- [ ] `step_3_1.cpp` compiles and all asserts pass — 10 ground, 7 obs, 2 veg, 1 sky
- [ ] `step_3_2.cpp` compiles and all asserts pass — rock cell is OBSTACLE, ground is FREE
- [ ] `step_3_3.cpp` compiles and all asserts pass — all global invariants pass

**Compile command (agent must use for EVERY step):**
```bash
g++ -O2 -std=c++17 step_X_Y.cpp -o step_X_Y && ./step_X_Y
```

**Phase 2 will cover:** CUDA port of this CPU reference code (TorchSparse++ Python bridge, Jetson deployment, terrain-specific class finetuning on RELLIS-3D).

---
*Document sealed by Principal Systems Architect — DRDO ID26053 SIH 2026*

#pragma once

#include <iostream>
#include <cstring>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <vector>

// ── Spatial Hash Map Configuration ──────────────────────────────────────────
// 2,097,152 buckets (80 MB). 1 << 20 held the old 5 cm <= 5 m pyramid, but the problem-statement
// spec (5 cm out to 10 m) carves ~310k cells in a single OS1-64 scan and ran at load 0.65 with
// continuous insert failures on the 3.3 km drive test.
constexpr int   HASH_TABLE_SIZE = 1 << 21;
constexpr int   MAX_PROBE       = 128;        // Maximum linear probe limit
constexpr int   DECAY_WINDOW    = 50;         // Frames before temporal decay triggers
constexpr int   DECAY_K         = 5;          // Run decay every K frames
constexpr float DECAY_FACTOR    = 0.90f;      // Multiplicative decay factor
constexpr float DECAY_MIN_TRAV  = 0.05f;      // Below this -> reset obstacle flag to UNKNOWN
constexpr float LOAD_WARN       = 0.70f;      // ADL-5 load-factor ceiling

// ── Semantic Traversability Lookup Table (ADL-1) ─────────────────────────────
// Class indices:
// 0: GROUND      -> 1.00 (Freely traversable flat dirt/road)
// 1: GRAVEL      -> 0.80 (Firm gravel / minor roughness)
// 2: GRASS       -> 0.65 (Medium grass / light low vegetation)
// 3: VEGETATION  -> 0.15 (High brush / dense bushes)
// 4: OBSTACLE    -> 0.00 (Hard obstacle / rocks / trees / vehicles)
// 5: WATER       -> 0.00 (Puddles / mud / water hazards)
// 6: UNKNOWN     -> 0.40 (Unobserved / uncertain terrain)
// 7: SKY / DUST  -> -1.0 (Discard / atmospheric noise)
constexpr float SEM_TRAV[8] = {1.0f, 0.8f, 0.65f, 0.15f, 0.0f, 0.0f, 0.4f, -1.0f};

// Resolution levels for Foveated Multi-Resolution Mapping (problem statement ID26053: 5 cm cells
// within 10 m, coarsening to 50 cm out to 100 m)
// Level 0:  0 m -  10 m -> 0.05 m ( 5 cm)
// Level 1: 10 m -  25 m -> 0.10 m (10 cm)
// Level 2: 25 m - 100 m -> 0.50 m (50 cm)
// Every level is an integer multiple of the 5 cm base lattice and every index is derived from the
// base index by integer floor division, so each cell nests exactly inside one cell of every coarser
// level. The previous 5/20/50/100 cm pyramid did not nest (50/20 = 2.5: 20% of 20 cm cells straddled
// a 50 cm edge) and quantised each level with its own float division.
constexpr int   NUM_LEVELS          = 3;
constexpr float BASE_RES            = 0.05f;
constexpr int   LEVEL_DIV[3]        = {1, 2, 10};
constexpr float CELL_RESOLUTIONS[3] = {0.05f, 0.10f, 0.50f};
constexpr float LEVEL_INNER_R[3]    = {0.0f, 10.0f, 25.0f};
constexpr float LEVEL_OUTER_R[3]    = {10.0f, 25.0f, 100.0f};
constexpr float MAX_RANGE           = 100.0f;
constexpr int   CARVE_STOP_RUN      = 16;      // stop a ray after this many cells already carved this frame
constexpr float PURGE_MARGIN_FRAC   = 0.25f;   // purge hysteresis cap, fraction of the band outer radius

#ifdef __CUDACC__
#define DRDO_HOST_DEVICE __host__ __device__
#else
#define DRDO_HOST_DEVICE
#endif
DRDO_HOST_DEVICE inline int32_t floor_div(int32_t a, int32_t b) { return (a >= 0) ? a / b : -((-a + b - 1) / b); }
DRDO_HOST_DEVICE inline int32_t base_index(float w) { return (int32_t)floorf(w / BASE_RES); }
inline int32_t cell_index(float w, int level) { return floor_div(base_index(w), LEVEL_DIV[level]); }

// Point height filter, RELATIVE to the ground reference passed to insert_lidar_point().
// FAST-LIO2's world origin is the IMU start pose, so ground sits at about -mount_height, not 0.
constexpr float Z_REL_MIN = -2.0f;
constexpr float Z_REL_MAX = 15.0f;

// ── Compact 40-byte GridCell Structure ───────────────────────────────────────
struct GridCell {
    float    h_min;             // Minimum elevation (m)
    float    h_max;             // Maximum elevation (m)
    float    h_mean;            // Online Welford running mean elevation (m)
    float    h_variance_M2;     // Online Welford running sum of squared diffs
    float    traversability;    // Fused score [0.0 - 1.0]
    uint32_t hit_count;         // Total LiDAR returns in cell
    uint32_t last_update_ts;    // Frame timestamp of last return
    int32_t  ix;                // Discrete grid coordinate X
    int32_t  iy;                // Discrete grid coordinate Y
    uint8_t  semantic_label;    // 0-7 semantic class
    uint8_t  obstacle_flag;     // 0=FREE, 1=OBSTACLE, 2=UNKNOWN
    uint8_t  level;             // Resolution pyramid level (0 .. NUM_LEVELS-1)
    bool     valid;             // Slot occupancy flag
};
static_assert(sizeof(GridCell) == 40, "GridCell struct must be exactly 40 bytes for cache efficiency!");

// Global hash table pool
inline GridCell g_hash_pool[HASH_TABLE_SIZE];

// Pool telemetry — the pre-audit engine dropped points silently once the pool saturated.
inline uint64_t g_insert_fail = 0;
inline int      g_valid_cells = 0;

// Slots whose height statistics changed since the last classify_dirty_cells() call. Classifying
// only these (instead of rescanning 1M slots every frame) also stops classification from
// overwriting the temporal decay kernel's result.
inline uint32_t g_dirty[HASH_TABLE_SIZE];
inline uint32_t g_dirty_n        = 0;
inline bool     g_dirty_overflow = false;

// Frame stamp of the last free-space carve per slot (moves with the cell in erase_slot). All 64 beams
// of an azimuth column share one radial line; walking each ray from its hit back toward the robot and
// stopping at a cell already carved this frame removes that duplicate work.
inline uint32_t g_carve_ts[HASH_TABLE_SIZE];

// ── Point & Geometry Structures ──────────────────────────────────────────────
struct LidarPoint {
    float x;
    float y;
    float z;
    uint8_t label;
};

struct ResolvedCell {
    int     level;
    float   cell_size;
    int32_t ix;
    int32_t iy;
    bool    discard;
};

// ── Core Engine Function Declarations ────────────────────────────────────────
int spatial_hash(int32_t ix, int32_t iy, uint8_t level);
int insert_cell(int32_t ix, int32_t iy, uint8_t level);
GridCell* lookup_cell(int32_t ix, int32_t iy, uint8_t level);
void erase_slot(uint32_t slot);
ResolvedCell resolve_resolution(float wx, float wy, float rx, float ry);
void update_height(GridCell& c, float z);
bool mark_free_cell(int32_t ix, int32_t iy, uint8_t level, uint32_t ts = 0);
void raycast_bresenham(int32_t ox, int32_t oy, int32_t hx, int32_t hy, uint8_t level, uint32_t ts = 0);
void raycast_band(float rx, float ry, float wx, float wy, uint8_t level, uint32_t ts = 0);
void classify_obstacle(GridCell& c, float ground_z);
void compute_traversability(GridCell& c);
void reset_map_pool();
int  insert_lidar_point_slot(float x, float y, float z, int label, float rx, float ry, uint32_t ts, float ground_z);
bool insert_lidar_point(float x, float y, float z, int label, float rx, float ry, uint32_t ts, float ground_z = 0.0f);
int run_temporal_decay(uint32_t current_frame);
int classify_and_score_all_cells(float ground_z);
int classify_dirty_cells(float ground_z);
int purge_distant_cells(float rx, float ry, float margin_m);
GridCell* query_world(float wx, float wy);
int get_valid_cell_count();
float get_load_factor();

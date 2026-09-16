#pragma once

#include <iostream>
#include <cstring>
#include <cmath>
#include <cassert>
#include <cstdint>
#include <vector>

// ── Spatial Hash Map Configuration ──────────────────────────────────────────
constexpr int   HASH_TABLE_SIZE = 1 << 20;   // 1,048,576 buckets (~40 MB)
constexpr int   MAX_PROBE       = 128;        // Maximum linear probe limit
constexpr int   DECAY_WINDOW    = 50;         // Frames before temporal decay triggers
constexpr int   DECAY_K         = 5;          // Run decay every K frames
constexpr float DECAY_FACTOR    = 0.90f;      // Multiplicative decay factor
constexpr float DECAY_MIN_TRAV  = 0.05f;      // Below this -> reset obstacle flag to UNKNOWN

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

// Resolution levels for Foveated Multi-Resolution Mapping
// Level 0: 0m - 5m   -> 0.05m (5 cm)
// Level 1: 5m - 20m  -> 0.20m (20 cm)
// Level 2: 20m - 50m -> 0.50m (50 cm)
// Level 3: 50m - 100m-> 1.00m (100 cm)
constexpr float CELL_RESOLUTIONS[4] = {0.05f, 0.20f, 0.50f, 1.00f};

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
    uint8_t  level;             // Resolution pyramid level (0-3)
    bool     valid;             // Slot occupancy flag
};
static_assert(sizeof(GridCell) == 40, "GridCell struct must be exactly 40 bytes for cache efficiency!");

// Global hash table pool
inline GridCell g_hash_pool[HASH_TABLE_SIZE];

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
ResolvedCell resolve_resolution(float wx, float wy, float rx, float ry);
void update_height(GridCell& c, float z);
void mark_free_cell(int32_t ix, int32_t iy, uint8_t level);
void raycast_bresenham(int32_t ox, int32_t oy, int32_t hx, int32_t hy, uint8_t level);
void classify_obstacle(GridCell& c, float ground_z);
void compute_traversability(GridCell& c);
void reset_map_pool();
bool insert_lidar_point(float x, float y, float z, int label, float rx, float ry, uint32_t ts);
int run_temporal_decay(uint32_t current_frame);
int classify_and_score_all_cells(float ground_z);
int get_valid_cell_count();

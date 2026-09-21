#include "drdo_lidar_mapping/drdo_map.h"
#include <algorithm>
using namespace std;

// Knuth multiplicative mix + MurmurHash3 fmix64 finalizer. Legacy "h % 2^20" kept only the weak
// low bits: measured on a synthetic OS1-64 drive at 11.1 m/s, fmix64 moves the first insert failure
// from 63% to 72.5% load and cuts the worst probe length at 44% load from 48 to 27.
// cuda/drdo_cuda_kernels.cu mirrors this function exactly — keep them in sync.
int spatial_hash(int32_t ix, int32_t iy, uint8_t level) {
    uint64_t h = (uint64_t)(uint32_t)ix  * 2654435761ULL;
    h ^= (uint64_t)(uint32_t)iy * 805459861ULL;
    h ^= (uint64_t)level        * 1234567891ULL;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (int)(h & (uint64_t)(HASH_TABLE_SIZE - 1));
}

int insert_cell(int32_t ix, int32_t iy, uint8_t level) {
    int bucket = spatial_hash(ix, iy, level);
    for (int probe = 0; probe < MAX_PROBE; probe++) {
        int slot = (bucket + probe) & (HASH_TABLE_SIZE - 1);
        GridCell& c = g_hash_pool[slot];
        if (c.valid && c.ix == ix && c.iy == iy && c.level == level) return slot;
        if (!c.valid) {
            c.h_min          = +1e9f;
            c.h_max          = -1e9f;
            c.h_mean         = 0.0f;
            c.h_variance_M2  = 0.0f;
            c.traversability = 0.4f;
            c.hit_count      = 0;
            c.last_update_ts = 0;
            c.semantic_label = 6;
            c.obstacle_flag  = 2;
            c.ix             = ix;
            c.iy             = iy;
            c.level          = level;
            c.valid          = true;
            g_valid_cells++;
            return slot;
        }
    }
    g_insert_fail++;
    return -1;
}

GridCell* lookup_cell(int32_t ix, int32_t iy, uint8_t level) {
    int bucket = spatial_hash(ix, iy, level);
    for (int probe = 0; probe < MAX_PROBE; probe++) {
        GridCell& c = g_hash_pool[(bucket + probe) & (HASH_TABLE_SIZE - 1)];
        if (!c.valid) return nullptr;
        if (c.ix == ix && c.iy == iy && c.level == level) return &c;
    }
    return nullptr;
}

// Backward-shift deletion (Knuth 6.4, Algorithm R). Zeroing a slot in place punches a hole into
// the probe chain: lookup_cell() stops at the first empty slot, so every cell displaced past the
// hole becomes unreachable and is later re-inserted as a duplicate.
void erase_slot(uint32_t hole) {
    const uint32_t MASK = HASH_TABLE_SIZE - 1;
    uint32_t j = hole;
    for (int guard = 0; guard < HASH_TABLE_SIZE; guard++) {
        j = (j + 1) & MASK;
        GridCell& c = g_hash_pool[j];
        if (!c.valid) break;
        uint32_t home = (uint32_t)spatial_hash(c.ix, c.iy, c.level);
        bool home_in_range = (hole <= j) ? (hole < home && home <= j) : (hole < home || home <= j);
        if (!home_in_range) {
            g_hash_pool[hole] = c;
            g_carve_ts[hole] = g_carve_ts[j];
            hole = j;
        }
    }
    memset(&g_hash_pool[hole], 0, sizeof(GridCell));
    g_carve_ts[hole] = 0;
    g_valid_cells--;
}

ResolvedCell resolve_resolution(float wx, float wy, float rx, float ry) {
    float dx = wx - rx, dy = wy - ry;
    float dist = sqrtf(dx * dx + dy * dy);
    ResolvedCell r;
    r.discard = false;
    r.ix = 0;
    r.iy = 0;
    if      (dist <= LEVEL_OUTER_R[0]) { r.level = 0; }
    else if (dist <= LEVEL_OUTER_R[1]) { r.level = 1; }
    else if (dist <= MAX_RANGE)        { r.level = 2; }
    else { r.level = NUM_LEVELS - 1; r.cell_size = CELL_RESOLUTIONS[NUM_LEVELS - 1]; r.discard = true; return r; }   // also NaN
    r.cell_size = CELL_RESOLUTIONS[r.level];
    r.ix = cell_index(wx, r.level);
    r.iy = cell_index(wy, r.level);
    return r;
}

void update_height(GridCell& c, float z) {
    c.hit_count++;
    float delta = z - c.h_mean;
    c.h_mean += delta / (float)c.hit_count;
    c.h_variance_M2 += delta * (z - c.h_mean);
    if (z < c.h_min) c.h_min = z;
    if (z > c.h_max) c.h_max = z;
}

// Returns true if the cell had already been carved during frame ts (ts == 0 disables the check).
bool mark_free_cell(int32_t ix, int32_t iy, uint8_t level, uint32_t ts) {
    int slot = insert_cell(ix, iy, level);
    if (slot < 0) return false;
    if (ts != 0 && g_carve_ts[slot] == ts) return true;
    g_carve_ts[slot] = ts;
    GridCell& c = g_hash_pool[slot];
    if (c.obstacle_flag == 1) c.obstacle_flag = 2;
    if (c.obstacle_flag == 2 && c.hit_count == 0) c.obstacle_flag = 0;
    return false;
}

// Marks every cell on the line from the origin cell up to, but excluding, the hit cell. Walks from the
// hit back toward the origin so that, with a frame stamp, it can stop once CARVE_STOP_RUN consecutive
// cells were already carved this frame (a collinear beam of the same azimuth column got there first).
// Stopping at the first shared cell also stopped at crossings with neighbouring azimuths and lost 16% of
// free cells. Synthetic OS1-64 scan, 87k points: full carve 38.6 ms / 310,198 cells; run 16 ->
// 19.7 ms / 310,123 cells (99.98%).
void raycast_bresenham(int32_t ox, int32_t oy, int32_t hx, int32_t hy, uint8_t level, uint32_t ts) {
    int dx = abs(ox - hx), dy = abs(oy - hy);
    int sx = (ox > hx) ? 1 : -1;
    int sy = (oy > hy) ? 1 : -1;
    int err = dx - dy;
    int x = hx, y = hy, run = 0;
    while (!(x == ox && y == oy)) {
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x += sx; }
        if (e2 <  dx) { err += dx; y += sy; }
        if (!mark_free_cell((int32_t)x, (int32_t)y, level, ts)) run = 0;
        else if (++run >= CARVE_STOP_RUN) return;
    }
}

// Carves free space only inside the hit's own foveation band. The legacy call walked a 1 m-cell
// line all the way from the robot for an 80 m return, allocating coarse cells on top of the robot.
void raycast_band(float rx, float ry, float wx, float wy, uint8_t level, uint32_t ts) {
    float dx = wx - rx, dy = wy - ry, dist = sqrtf(dx * dx + dy * dy);
    float t0 = (dist > 1e-3f) ? LEVEL_INNER_R[level] / dist : 0.0f;
    if (t0 >= 1.0f) return;
    raycast_bresenham(cell_index(rx + dx * t0, level), cell_index(ry + dy * t0, level),
                      cell_index(wx, level), cell_index(wy, level), level, ts);
}

// Does any nearby cell still stand at roughly ground level? Probes the 4 immediate neighbours plus
// 8 directions on a ring ~NEG_RIM_RADIUS_M out, so the inside of a pothole up to about 1.5 m across
// still sees its own rim. 12 O(1) hash lookups, and only for cells that already failed the depth
// test, so this costs nothing on normal terrain.
//
// Larger depressions only get their boundary ring flagged, which is operationally sufficient: a
// lethal rim encloses the hole, so no path can enter it. What this deliberately does NOT do is
// flag terrain that is merely low — see NEG_OBST_DEPTH's comment on slopes.
bool has_ground_rim(const GridCell& c, float ground_z) {
    uint8_t lv = c.level < NUM_LEVELS ? c.level : NUM_LEVELS - 1;
    int32_t r = (int32_t)(NEG_RIM_RADIUS_M / CELL_RESOLUTIONS[lv]);
    if (r < 1) r = 1;
    const int32_t off[12][2] = {
        {1, 0}, {-1, 0}, {0, 1}, {0, -1},                       // immediate neighbours
        {r, 0}, {-r, 0}, {0, r}, {0, -r},                       // ring, axis-aligned
        {r, r}, {r, -r}, {-r, r}, {-r, -r},                     // ring, diagonal
    };
    for (int k = 0; k < 12; k++) {
        GridCell* n = lookup_cell(c.ix + off[k][0], c.iy + off[k][1], lv);
        if (!n || n->hit_count == 0) continue;
        if (n->h_max >= ground_z - NEG_RIM_TOL) return true;
    }
    return false;
}

void classify_obstacle(GridCell& c, float ground_z) {
    if (c.hit_count == 0) return;
    float step_height = c.h_max - c.h_min;
    float clearance   = c.h_min - ground_z;
    float max_above   = c.h_max - ground_z;
    float depth_below = ground_z - c.h_max;   // > 0 when the whole cell sits under the ground plane

    // Negative obstacles are tested FIRST. A pothole's floor is flat, so step_height is ~0 and the
    // flat-ground branch below would claim it as drivable — which is exactly what the engine did
    // before this branch existed (measured: a 40 cm pothole came back FREE, traversability 1.00).
    if      (depth_below > NEG_OBST_DEPTH && c.hit_count >= NEG_MIN_HITS
             && has_ground_rim(c, ground_z))
                                  c.obstacle_flag = 3; // Pothole / trench / washout
    else if (step_height < 0.05f) c.obstacle_flag = 0; // Flat Ground
    else if (clearance > 0.50f)   c.obstacle_flag = 0; // Overhanging tree canopy / wire
    else if (max_above > 0.30f)   c.obstacle_flag = 1; // Lethal Obstacle
    else                          c.obstacle_flag = 2; // Rough terrain / unknown
}

void compute_traversability(GridCell& c) {
    // 3 (negative obstacle) scores 0 exactly like 1 (lethal). A hole and a rock are equally
    // impassable to a wheeled vehicle, and leaving a pothole on the semantic/roughness path would
    // hand it back a high score: its floor is flat and smoothly sampled, so roughness is ~0 and
    // the label is usually GROUND.
    if (c.obstacle_flag == 1 || c.obstacle_flag == 3) {
        c.traversability = 0.0f;
        return;
    }
    float rough = (c.hit_count > 1) ? sqrtf(fmaxf(c.h_variance_M2, 0.0f) / (float)(c.hit_count - 1)) : 0.0f;
    float hs    = expf(-5.0f * rough);
    float ss    = (c.semantic_label < 7) ? SEM_TRAV[c.semantic_label] : 0.4f;
    float conf  = fminf((float)c.hit_count / 10.0f, 1.0f);
    c.traversability = conf * (0.5f * hs + 0.5f * ss);
}

void reset_map_pool() {
    memset(g_hash_pool, 0, sizeof(g_hash_pool));
    memset(g_carve_ts, 0, sizeof(g_carve_ts));
    g_valid_cells = 0;
    g_insert_fail = 0;
    g_dirty_n = 0;
    g_dirty_overflow = false;
}

// Returns the updated slot, or -1 if the point was discarded or the pool is full.
int insert_lidar_point_slot(float x, float y, float z, int label, float rx, float ry, uint32_t ts, float ground_z) {
    if (label < 0 || label > 7) label = 6;
    float range = sqrtf((x - rx) * (x - rx) + (y - ry) * (y - ry));
    float zr = z - ground_z;
    // Lower bound widens with range (slope_adjusted_z_min) so a physically plausible downgrade
    // ahead of the vehicle isn't discarded — see its comment in drdo_map.h.
    if (label == 7 || zr > Z_REL_MAX || zr < slope_adjusted_z_min(range)) return -1;  // sky/dust, outliers, NaN
    ResolvedCell rc = resolve_resolution(x, y, rx, ry);
    if (rc.discard) return -1;

    raycast_band(rx, ry, x, y, (uint8_t)rc.level, ts);

    int slot = insert_cell(rc.ix, rc.iy, (uint8_t)rc.level);
    if (slot < 0) return -1;

    GridCell& c = g_hash_pool[slot];
    // Unseen for longer than the decay window: restart statistics. Otherwise a dust cloud's h_max
    // persists forever and re-flags the cell lethal on the very next ground return.
    if (c.hit_count > 0 && ts > c.last_update_ts && ts - c.last_update_ts > (uint32_t)DECAY_WINDOW) {
        c.h_min = +1e9f; c.h_max = -1e9f; c.h_mean = 0.0f; c.h_variance_M2 = 0.0f; c.hit_count = 0;
    }
    if (c.last_update_ts != ts || c.hit_count == 0) {
        if (g_dirty_n < (uint32_t)HASH_TABLE_SIZE) g_dirty[g_dirty_n++] = (uint32_t)slot;
        else g_dirty_overflow = true;
    }
    update_height(c, z);
    c.semantic_label = (uint8_t)label;
    c.last_update_ts = ts;
    return slot;
}

bool insert_lidar_point(float x, float y, float z, int label, float rx, float ry, uint32_t ts, float ground_z) {
    return insert_lidar_point_slot(x, y, z, label, rx, ry, ts, ground_z) >= 0;
}

int run_temporal_decay(uint32_t current_frame) {
    if (current_frame % DECAY_K != 0) return 0;
    int decayed = 0;
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        GridCell& c = g_hash_pool[i];
        if (!c.valid) continue;
        if (current_frame > c.last_update_ts && (current_frame - c.last_update_ts) > (uint32_t)DECAY_WINDOW) {
            c.traversability *= DECAY_FACTOR;
            if (c.traversability < DECAY_MIN_TRAV && (c.obstacle_flag == 1 || c.obstacle_flag == 3)) {
                c.obstacle_flag = 2; // Demote stale obstacle (positive or negative) to unknown
            }
            decayed++;
        }
    }
    return decayed;
}

// Full rescan. Calling this every frame overwrites run_temporal_decay(); real-time loops must use
// classify_dirty_cells().
int classify_and_score_all_cells(float ground_z) {
    int n = 0;
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        if (!g_hash_pool[i].valid) continue;
        classify_obstacle(g_hash_pool[i], ground_z);
        compute_traversability(g_hash_pool[i]);
        n++;
    }
    g_dirty_n = 0;
    g_dirty_overflow = false;
    return n;
}

int classify_dirty_cells(float ground_z) {
    if (g_dirty_overflow) return classify_and_score_all_cells(ground_z);
    int n = 0;
    for (uint32_t k = 0; k < g_dirty_n; k++) {
        GridCell& c = g_hash_pool[g_dirty[k]];
        if (!c.valid) continue;
        classify_obstacle(c, ground_z);
        compute_traversability(c);
        n++;
    }
    g_dirty_n = 0;
    return n;
}

// Foveation-aware sliding window (ADL-5): evict a cell once it is beyond the outer radius of its own
// resolution band plus a hysteresis margin of min(margin_m, PURGE_MARGIN_FRAC x that radius). Capping
// the margin keeps each level's retained area proportional to its band: a flat 20 m margin keeps a
// 30 m disk of 5 cm cells (~1.1M cells), more than the whole pool. Behind the robot the next coarser
// level still covers the evicted ground, so nothing observed within MAX_RANGE is lost. A single global radius keeps every 5 cm cell in a long
// corridor behind the vehicle (measured: peak load 0.715 and continuous insert failures at 40 km/h
// with one 110 m radius). Must run between frames — erase_slot() moves cells.
int purge_distant_cells(float rx, float ry, float margin_m) {
    int purged = 0;
    for (uint32_t i = 0; i < (uint32_t)HASH_TABLE_SIZE; i++) {
        while (g_hash_pool[i].valid) {
            GridCell& c = g_hash_pool[i];
            uint8_t lv = c.level < NUM_LEVELS ? c.level : NUM_LEVELS - 1;
            float cs = CELL_RESOLUTIONS[lv];
            float keep_r = LEVEL_OUTER_R[lv] + fminf(margin_m, PURGE_MARGIN_FRAC * LEVEL_OUTER_R[lv]);
            float dx = (c.ix + 0.5f) * cs - rx, dy = (c.iy + 0.5f) * cs - ry;
            if (dx * dx + dy * dy <= keep_r * keep_r) break;
            erase_slot(i);   // the next chain member may shift into slot i: re-check it
            purged++;
        }
    }
    g_dirty_n = 0;
    return purged;
}

// Returns the most recently observed cell covering (wx, wy) across all levels (ties -> finer). A free-space
// carve counts as an observation. Returning the first level found let a stale 5 cm cell, kept behind the
// robot until the purge margin, shadow the fresh coarse observation of the same ground.
GridCell* query_world(float wx, float wy) {
    int32_t bx = base_index(wx), by = base_index(wy);
    GridCell* best = nullptr;
    uint32_t best_ts = 0;
    for (int lv = 0; lv < NUM_LEVELS; lv++) {
        GridCell* f = lookup_cell(floor_div(bx, LEVEL_DIV[lv]), floor_div(by, LEVEL_DIV[lv]), (uint8_t)lv);
        if (!f) continue;
        uint32_t seen = max(f->last_update_ts, g_carve_ts[f - g_hash_pool]);
        if (!best || seen > best_ts) { best = f; best_ts = seen; }
    }
    return best;
}

int get_valid_cell_count() {
    int n = 0;
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        if (g_hash_pool[i].valid) n++;
    }
    return n;
}

float get_load_factor() {
    return g_valid_cells / (float)HASH_TABLE_SIZE;
}

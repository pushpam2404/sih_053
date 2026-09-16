#include "drdo_lidar_mapping/drdo_map.h"

int spatial_hash(int32_t ix, int32_t iy, uint8_t level) {
    uint64_t h = (uint64_t)(uint32_t)ix  * 2654435761ULL;
    h ^= (uint64_t)(uint32_t)iy * 805459861ULL;
    h ^= (uint64_t)level        * 1234567891ULL;
    return (int)(h % (uint64_t)HASH_TABLE_SIZE);
}

int insert_cell(int32_t ix, int32_t iy, uint8_t level) {
    int bucket = spatial_hash(ix, iy, level);
    for (int probe = 0; probe < MAX_PROBE; probe++) {
        int slot = (bucket + probe) % HASH_TABLE_SIZE;
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
            return slot;
        }
    }
    return -1;
}

GridCell* lookup_cell(int32_t ix, int32_t iy, uint8_t level) {
    int bucket = spatial_hash(ix, iy, level);
    for (int probe = 0; probe < MAX_PROBE; probe++) {
        int slot = (bucket + probe) % HASH_TABLE_SIZE;
        GridCell& c = g_hash_pool[slot];
        if (!c.valid) return nullptr;
        if (c.ix == ix && c.iy == iy && c.level == level) return &c;
    }
    return nullptr;
}

ResolvedCell resolve_resolution(float wx, float wy, float rx, float ry) {
    float dx = wx - rx, dy = wy - ry;
    float dist = sqrtf(dx * dx + dy * dy);
    ResolvedCell r;
    r.discard = false;
    if      (dist <=  5.0f) { r.level = 0; r.cell_size = 0.05f; }
    else if (dist <  20.0f) { r.level = 1; r.cell_size = 0.20f; }
    else if (dist <  50.0f) { r.level = 2; r.cell_size = 0.50f; }
    else if (dist <= 100.0f){ r.level = 3; r.cell_size = 1.00f; }
    else                    { r.discard = true; return r; }
    r.ix = (int32_t)floorf(wx / r.cell_size);
    r.iy = (int32_t)floorf(wy / r.cell_size);
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

void mark_free_cell(int32_t ix, int32_t iy, uint8_t level) {
    int slot = insert_cell(ix, iy, level);
    if (slot < 0) return;
    GridCell& c = g_hash_pool[slot];
    if (c.obstacle_flag == 1) c.obstacle_flag = 2;
    if (c.obstacle_flag == 2 && c.hit_count == 0) c.obstacle_flag = 0;
}

void raycast_bresenham(int32_t ox, int32_t oy, int32_t hx, int32_t hy, uint8_t level) {
    int dx = abs(hx - ox), dy = abs(hy - oy);
    int sx = (hx > ox) ? 1 : -1;
    int sy = (hy > oy) ? 1 : -1;
    int err = dx - dy;
    int x = ox, y = oy;
    while (true) {
        if (x == hx && y == hy) break;
        mark_free_cell((int32_t)x, (int32_t)y, level);
        int e2 = 2 * err;
        if (e2 > -dy) { err -= dy; x += sx; }
        if (e2 <  dx) { err += dx; y += sy; }
    }
}

void classify_obstacle(GridCell& c, float ground_z) {
    if (c.hit_count == 0) return;
    float step_height = c.h_max - c.h_min;
    float clearance   = c.h_min - ground_z;
    float max_above   = c.h_max - ground_z;

    if      (step_height < 0.05f) c.obstacle_flag = 0; // Flat Ground
    else if (clearance > 0.50f)   c.obstacle_flag = 0; // Overhanging tree canopy / wire
    else if (max_above > 0.30f)   c.obstacle_flag = 1; // Lethal Obstacle
    else                          c.obstacle_flag = 2; // Rough terrain / unknown
}

void compute_traversability(GridCell& c) {
    if (c.obstacle_flag == 1) {
        c.traversability = 0.0f;
        return;
    }
    float rough = (c.hit_count > 1) ? sqrtf(c.h_variance_M2 / (float)(c.hit_count - 1)) : 0.0f;
    float hs    = expf(-5.0f * rough);
    float ss    = (c.semantic_label < 8) ? SEM_TRAV[c.semantic_label] : 0.4f;
    float conf  = fminf((float)c.hit_count / 10.0f, 1.0f);
    c.traversability = conf * (0.5f * hs + 0.5f * ss);
}

void reset_map_pool() {
    memset(g_hash_pool, 0, sizeof(g_hash_pool));
}

bool insert_lidar_point(float x, float y, float z, int label, float rx, float ry, uint32_t ts) {
    if (label == 7 || z > 15.0f || z < -2.0f) return false; // Filter sky/dust and extreme outliers
    ResolvedCell rc = resolve_resolution(x, y, rx, ry);
    if (rc.discard) return false;

    int32_t rox = (int32_t)floorf(rx / rc.cell_size);
    int32_t roy = (int32_t)floorf(ry / rc.cell_size);
    raycast_bresenham(rox, roy, rc.ix, rc.iy, (uint8_t)rc.level);

    int slot = insert_cell(rc.ix, rc.iy, (uint8_t)rc.level);
    if (slot < 0) return false;

    GridCell& c = g_hash_pool[slot];
    update_height(c, z);
    c.semantic_label = (uint8_t)label;
    c.last_update_ts = ts;
    return true;
}

int run_temporal_decay(uint32_t current_frame) {
    if (current_frame % DECAY_K != 0) return 0;
    int decayed = 0;
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        GridCell& c = g_hash_pool[i];
        if (!c.valid) continue;
        if ((current_frame - c.last_update_ts) > (uint32_t)DECAY_WINDOW) {
            c.traversability *= DECAY_FACTOR;
            if (c.traversability < DECAY_MIN_TRAV && c.obstacle_flag == 1) {
                c.obstacle_flag = 2; // Demote stale obstacle to unknown
            }
            decayed++;
        }
    }
    return decayed;
}

int classify_and_score_all_cells(float ground_z) {
    int n = 0;
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        if (!g_hash_pool[i].valid) continue;
        classify_obstacle(g_hash_pool[i], ground_z);
        compute_traversability(g_hash_pool[i]);
        n++;
    }
    return n;
}

int get_valid_cell_count() {
    int n = 0;
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        if (g_hash_pool[i].valid) n++;
    }
    return n;
}

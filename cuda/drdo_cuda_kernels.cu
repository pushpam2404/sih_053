// cuda/drdo_cuda_kernels.cu
// GPU port of the 2.5D grid engine hot loops (target sm_87, Jetson AGX Orin).
//
// Audit fixes (see SIH_Deep_Dive_Audit.md):
//  * Compile errors: cudaMalloc(&GridCell*) needs (void**); atomicAdd/atomicMin/atomicMax have no
//    float overloads; __float_as_int does not exist.
//  * Memory corruption: atomicCAS((int*)&c.valid, ...) wrote 4 bytes into a 1-byte bool at
//    offset 39 of a 40-byte cell — clobbering the next cell's h_min.
//  * Wrong ordering: integer min/max over IEEE-754 bit patterns orders negative floats backwards.
//  * Lost updates: independent atomics on h_mean / h_variance_M2 are not a linearizable Welford
//    update. Cells are now serialised with a per-cell spin lock stored in the in-bounds 4-byte
//    metadata word (semantic_label, obstacle_flag, level, valid).
//  * Hash drift: d_spatial_hash mirrors src/drdo_map.cpp (fmix64 finalizer) exactly.
// NOT verified on hardware: no nvcc on the development Mac. Build on the Orin:
//   nvcc -O3 -std=c++17 -arch=sm_87 -Iinclude cuda/drdo_cuda_kernels.cu -o test_cuda_kernels
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include <cstring>
#include "drdo_lidar_mapping/drdo_map.h"

// Metadata word layout (little-endian, bytes 36..39): label | flag << 8 | level << 16 | valid << 24
constexpr unsigned int META_VALID   = 1u << 24;
constexpr unsigned int META_LOCK    = 2u << 24;   // per-cell spin lock (only ever set while valid)
constexpr unsigned int META_CLAIM   = 0xFFu << 16; // slot being initialised (valid byte still 0)
constexpr int          SPIN_LIMIT   = 1 << 16;

__device__ inline unsigned int* d_meta(GridCell& c) { return (unsigned int*)&c.semantic_label; }

// ── Device hash + lookup ──────────────────────────────────────────────────────

__host__ __device__ int d_spatial_hash(int32_t ix, int32_t iy, uint8_t level) {
    uint64_t h  = (uint64_t)(uint32_t)ix * 2654435761ULL;
    h ^= (uint64_t)(uint32_t)iy * 805459861ULL;
    h ^= (uint64_t)level        * 1234567891ULL;
    h ^= h >> 33; h *= 0xff51afd7ed558ccdULL;
    h ^= h >> 33; h *= 0xc4ceb9fe1a85ec53ULL;
    h ^= h >> 33;
    return (int)(h & (uint64_t)(HASH_TABLE_SIZE - 1));
}

__device__ GridCell* d_lookup_or_insert(GridCell* pool, int32_t ix, int32_t iy, uint8_t level) {
    int bucket = d_spatial_hash(ix, iy, level);
    for (int p = 0; p < MAX_PROBE; p++) {
        GridCell& c = pool[(bucket + p) & (HASH_TABLE_SIZE - 1)];
        unsigned int* meta = d_meta(c);
        for (int spin = 0; spin < SPIN_LIMIT; spin++) {
            unsigned int m = *meta;
            if (m == 0) {
                if (atomicCAS(meta, 0u, META_CLAIM) != 0u) continue;       // lost the race: re-read
                c.h_min = +1e9f; c.h_max = -1e9f;
                c.h_mean = 0.0f; c.h_variance_M2 = 0.0f;
                c.traversability = 0.4f; c.hit_count = 0; c.last_update_ts = 0;
                c.ix = ix; c.iy = iy;
                *meta = 6u | (2u << 8) | ((unsigned int)level << 16) | META_VALID;   // publish last
                return &c;
            }
            if (m == META_CLAIM) continue;                                  // key not visible yet
            if (c.ix == ix && c.iy == iy && ((m >> 16) & 0xFFu) == level) return &c;
            break;                                                          // occupied by another key
        }
    }
    return nullptr;
}

__device__ bool d_lock(GridCell& c) {
    unsigned int* meta = d_meta(c);
    for (int spin = 0; spin < SPIN_LIMIT; spin++) {
        unsigned int m = *meta;
        if ((m & META_VALID) && !(m & META_LOCK) && atomicCAS(meta, m, m | META_LOCK) == m) return true;
    }
    return false;
}

__device__ void d_unlock(GridCell& c, uint8_t label) {
    unsigned int* meta = d_meta(c);
    unsigned int m = *meta;
    *meta = ((m & ~META_LOCK) & ~0xFFu) | label;   // owner-only write while locked
}

// ── Kernel 1: update_height_kernel ───────────────────────────────────────────
// One thread per LiDAR point. z filter is relative to ground_z (see insert_lidar_point_slot).

__global__ void update_height_kernel(
        GridCell* pool,
        float* pts_x, float* pts_y, float* pts_z,
        uint8_t* labels, int N,
        float robot_x, float robot_y, float ground_z, uint32_t frame_ts) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float wx = pts_x[idx], wy = pts_y[idx], wz = pts_z[idx];
    uint8_t label = labels[idx] > 7 ? 6 : labels[idx];

    float dx = wx - robot_x, dy = wy - robot_y;
    float dist = sqrtf(dx * dx + dy * dy);

    // Mirrors insert_lidar_point_slot() in src/drdo_map.cpp: the lower bound widens with range
    // (slope_adjusted_z_min, drdo_map.h) so a plausible downgrade ahead of the vehicle is kept.
    float zr = wz - ground_z;
    if (label == 7 || zr > Z_REL_MAX || zr < slope_adjusted_z_min(dist)) return;
    if (!(dist <= 100.0f)) return;

    // Mirrors resolve_resolution(): 5 cm <= 10 m, 10 cm <= 25 m, 50 cm <= 100 m, indices derived from
    // the 5 cm base lattice by integer floor division (host constexpr arrays are not device-visible).
    uint8_t lv = dist <= 10.0f ? 0 : dist <= 25.0f ? 1 : 2;
    int32_t div = lv == 0 ? 1 : lv == 1 ? 2 : 10;
    GridCell* cell = d_lookup_or_insert(pool, floor_div(base_index(wx), div), floor_div(base_index(wy), div), lv);
    if (!cell || !d_lock(*cell)) return;

    GridCell& c = *cell;
    if (c.hit_count > 0 && frame_ts > c.last_update_ts && frame_ts - c.last_update_ts > (uint32_t)DECAY_WINDOW) {
        c.h_min = +1e9f; c.h_max = -1e9f; c.h_mean = 0.0f; c.h_variance_M2 = 0.0f; c.hit_count = 0;
    }
    c.hit_count++;
    float delta = wz - c.h_mean;
    c.h_mean += delta / (float)c.hit_count;
    c.h_variance_M2 += delta * (wz - c.h_mean);
    if (wz < c.h_min) c.h_min = wz;
    if (wz > c.h_max) c.h_max = wz;
    if (frame_ts > c.last_update_ts) c.last_update_ts = frame_ts;
    d_unlock(c, label);
}

// ── Kernel 2: decay_stale_cells_kernel ───────────────────────────────────────
// One thread per hash table slot. Must not run concurrently with kernel 1.

__global__ void decay_stale_cells_kernel(
        GridCell* pool,
        uint32_t current_ts,
        uint32_t decay_window,
        float decay_factor,
        float decay_min_trav) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HASH_TABLE_SIZE) return;

    GridCell& c = pool[idx];
    if (!(*d_meta(c) & META_VALID)) return;

    if (current_ts > c.last_update_ts && (current_ts - c.last_update_ts) > decay_window) {
        c.traversability *= decay_factor;
        if (c.traversability < decay_min_trav && (c.obstacle_flag == 1 || c.obstacle_flag == 3))
            c.obstacle_flag = 2;   // stale obstacle, positive or negative, demotes to unknown
    }
}

// ── Kernel 3: classify_and_score_kernel ──────────────────────────────────────
// One thread per slot; only cells touched at frame_ts (mirrors classify_dirty_cells on the CPU,
// so decayed cells are not re-classified).

__constant__ float d_SEM_TRAV[8] = {1.0f, 0.8f, 0.65f, 0.15f, 0.0f, 0.0f, 0.4f, -1.0f};

// Read-only probe. d_lookup_or_insert() must NOT be used for the rim test below: it allocates on a
// miss, so probing twelve neighbours of every deep cell would materialise phantom cells all around
// each pothole and inflate the pool. This mirrors the CPU lookup_cell() — stop at the first empty
// slot, never write.
__device__ GridCell* d_lookup(GridCell* pool, int32_t ix, int32_t iy, uint8_t level) {
    int bucket = d_spatial_hash(ix, iy, level);
    for (int p = 0; p < MAX_PROBE; p++) {
        GridCell& c = pool[(bucket + p) & (HASH_TABLE_SIZE - 1)];
        unsigned int m = *d_meta(c);
        if (m == 0) return nullptr;                                         // empty: key absent
        if (m == META_CLAIM) continue;                                      // mid-insert, skip
        if (c.ix == ix && c.iy == iy && ((m >> 16) & 0xFFu) == level) return &c;
    }
    return nullptr;
}

// Mirrors has_ground_rim() in src/drdo_map.cpp — keep the two in step, as with d_spatial_hash.
// Race-free despite every thread classifying concurrently: the only neighbour fields read here,
// hit_count and h_max, are written by update_height_kernel in an EARLIER launch and are stable by
// the time this kernel runs. obstacle_flag is the only field written concurrently, and is not read.
__device__ bool d_has_ground_rim(GridCell* pool, const GridCell& c, float ground_z) {
    uint8_t lv = c.level < NUM_LEVELS ? c.level : NUM_LEVELS - 1;
    int32_t r = (int32_t)(NEG_RIM_RADIUS_M / CELL_RESOLUTIONS[lv]);
    if (r < 1) r = 1;
    const int32_t off[12][2] = {
        {1, 0}, {-1, 0}, {0, 1}, {0, -1},
        {r, 0}, {-r, 0}, {0, r}, {0, -r},
        {r, r}, {r, -r}, {-r, r}, {-r, -r},
    };
    for (int k = 0; k < 12; k++) {
        GridCell* n = d_lookup(pool, c.ix + off[k][0], c.iy + off[k][1], lv);
        if (!n || n->hit_count == 0) continue;
        if (n->h_max >= ground_z - NEG_RIM_TOL) return true;
    }
    return false;
}

__global__ void classify_and_score_kernel(GridCell* pool, float ground_z, uint32_t frame_ts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= HASH_TABLE_SIZE) return;

    GridCell& c = pool[idx];
    if (!(*d_meta(c) & META_VALID) || c.hit_count == 0 || c.last_update_ts != frame_ts) return;

    float hs = c.h_max - c.h_min, cl = c.h_min - ground_z, ma = c.h_max - ground_z;
    float db = ground_z - c.h_max;                  // > 0 when the cell sits under the ground plane
    // Negative-obstacle branch first, exactly as on the CPU: a pothole floor is flat, so the
    // hs < 0.05 test below would otherwise call a 40 cm hole drivable ground.
    if      (db > NEG_OBST_DEPTH && c.hit_count >= NEG_MIN_HITS
             && d_has_ground_rim(pool, c, ground_z))
                         c.obstacle_flag = 3;
    else if (hs < 0.05f) c.obstacle_flag = 0;
    else if (cl > 0.50f) c.obstacle_flag = 0;
    else if (ma > 0.30f) c.obstacle_flag = 1;
    else                 c.obstacle_flag = 2;

    if (c.obstacle_flag == 1 || c.obstacle_flag == 3) { c.traversability = 0.0f; return; }
    float rough = (c.hit_count > 1) ? sqrtf(fmaxf(c.h_variance_M2, 0.0f) / (float)(c.hit_count - 1)) : 0.0f;
    float hs_sc = expf(-5.0f * rough);
    float ss_sc = d_SEM_TRAV[c.semantic_label < 7 ? c.semantic_label : 6];
    float conf  = fminf((float)c.hit_count / 10.0f, 1.0f);
    c.traversability = conf * (0.5f * hs_sc + 0.5f * ss_sc);
}

// ── Host test: all logic in main() ───────────────────────────────────────────

int main() {
    const int N = 1000;
    const int THREADS = 256;
    const float GROUND_Z = 0.0f;

    // Jetson is unified memory: a managed pool is visible to CPU and GPU without the 40 MB
    // cudaMemcpy round trip the original harness performed.
    GridCell* pool = nullptr;
    if (cudaMallocManaged((void**)&pool, sizeof(GridCell) * HASH_TABLE_SIZE) != cuda_SUCCESS) {
        printf("[DRDO CUDA] cudaMallocManaged failed\n");
        return 1;
    }
    memset(pool, 0, sizeof(GridCell) * HASH_TABLE_SIZE);

    float* x = nullptr; float* y = nullptr; float* z = nullptr; uint8_t* labels = nullptr;
    cudaMallocManaged((void**)&x, N * sizeof(float));
    cudaMallocManaged((void**)&y, N * sizeof(float));
    cudaMallocManaged((void**)&z, N * sizeof(float));
    cudaMallocManaged((void**)&labels, N * sizeof(uint8_t));
    for (int i = 0; i < N; i++) {
        float angle = (float)i / N * 6.28318f, r = 1.0f + (i % 50) * 0.18f;
        x[i] = r * cosf(angle); y[i] = r * sinf(angle);
        z[i] = (i % 20 < 15) ? 0.01f : 0.50f;
        labels[i] = (z[i] > 0.3f) ? 4 : 0;
    }

    update_height_kernel<<<(N + THREADS - 1) / THREADS, THREADS>>>(pool, x, y, z, labels, N, 0.0f, 0.0f, GROUND_Z, 1);
    cudaDeviceSynchronize();
    printf("[DRDO CUDA] update_height_kernel done.\n");

    int ht_blocks = (HASH_TABLE_SIZE + THREADS - 1) / THREADS;
    classify_and_score_kernel<<<ht_blocks, THREADS>>>(pool, GROUND_Z, 1);
    cudaDeviceSynchronize();
    printf("[DRDO CUDA] classify_and_score_kernel done.\n");

    decay_stale_cells_kernel<<<ht_blocks, THREADS>>>(pool, 55, DECAY_WINDOW, DECAY_FACTOR, DECAY_MIN_TRAV);
    cudaDeviceSynchronize();
    printf("[DRDO CUDA] decay_stale_cells_kernel done.\n");
    assert(cudaGetLastError() == cuda_SUCCESS);

    int valid = 0, obstacle = 0, locked = 0;
    long hits = 0;
    for (int i = 0; i < HASH_TABLE_SIZE; i++) {
        unsigned int meta = *(unsigned int*)&pool[i].semantic_label;
        if (!(meta & META_VALID)) continue;
        valid++;
        locked += (meta & META_LOCK) != 0;
        hits += pool[i].hit_count;
        assert(pool[i].semantic_label != 7 && "SKY_NOISE must not enter map!");
        assert(pool[i].h_min <= pool[i].h_max && "min/max ordering");
        if (pool[i].obstacle_flag == 1) obstacle++;
        // CPU/GPU hash parity: the CPU engine must find GPU-written cells in the same bucket.
        assert(spatial_hash(pool[i].ix, pool[i].iy, pool[i].level) ==
               d_spatial_hash(pool[i].ix, pool[i].iy, pool[i].level));
    }
    printf("[DRDO CUDA] valid=%d obstacle=%d hits=%ld (expected %d) locked=%d\n", valid, obstacle, hits, N, locked);
    assert(valid > 0 && obstacle > 0 && locked == 0);
    assert(hits == N && "no lost Welford updates");

    printf("[PASS] All CUDA kernel invariants validated.\n");
    cudaFree(pool); cudaFree(x); cudaFree(y); cudaFree(z); cudaFree(labels);
    return 0;
}

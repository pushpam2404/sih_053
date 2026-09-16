// phase3/cuda/drdo_cuda_kernels.cu
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cassert>
#include "drdo_lidar_mapping/drdo_map.h"

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
    if      (dist <= 5.0f) { lv=0; cs=0.05f; }
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

// tests/cpp/test_engine_regression.cpp
// Regression gates for the SIH_Deep_Dive_Audit.md engine fixes + a sustained-drive load test.
// Build: cmake --build build --target test_engine_regression   (or)
//        clang++ -O3 -std=c++17 -Iinclude src/drdo_map.cpp tests/cpp/test_engine_regression.cpp -o test_engine_regression
// Run:   ./test_engine_regression [speed_mps=11.1] [frames=3000] [columns=2048] [hz=10]
#include "drdo_lidar_mapping/drdo_map.h"
#include <cstdio>
#include <chrono>
#include <vector>
#include <unordered_set>
using namespace std;

int main(int argc, char** argv) {
    float speed  = argc > 1 ? atof(argv[1]) : 11.1f;
    int   frames = argc > 2 ? atoi(argv[2]) : 3000;
    int   cols   = argc > 3 ? atoi(argv[3]) : 2048;
    float hz     = argc > 4 ? atof(argv[4]) : 10.0f;

    // ── Gate 1: probe chains survive deletion (legacy memset purge orphaned displaced cells) ──
    reset_map_pool();
    for (int32_t i = -1800; i < 1800; i++)
        for (int32_t j = -180; j < 180; j++) { insert_cell(i, j, 2); }         // 1.296M cells, load 0.62
    int before = g_valid_cells;
    int purged = purge_distant_cells(0.0f, 0.0f, 300.0f);   // level 2: keep r = 100 + min(300, 0.25*100) = 125 m
    int orphans = 0, dup = 0;
    for (int32_t i = -1800; i < 1800; i++)
        for (int32_t j = -180; j < 180; j++) {
            float cx = (i + 0.5f) * 0.5f, cy = (j + 0.5f) * 0.5f;
            bool keep = cx * cx + cy * cy <= 125.0f * 125.0f;
            GridCell* c = lookup_cell(i, j, 2);
            if (keep && !c) orphans++;
            if (!keep && c) dup++;
        }
    int scanned = 0;
    for (int i = 0; i < HASH_TABLE_SIZE; i++) scanned += g_hash_pool[i].valid;
    printf("[GATE 1] inserted=%d purged=%d orphans=%d stale_survivors=%d counter=%d scan=%d\n",
           before, purged, orphans, dup, g_valid_cells, scanned);
    assert(orphans == 0 && dup == 0 && scanned == g_valid_cells && "probe-chain integrity after purge");

    // ── Gate 2: decay is not overwritten by reclassification, and dust clears on re-observation ──
    reset_map_pool();
    const float GZ = -1.2f;                        // FAST-LIO world frame: ground at -mount height
    for (int k = 0; k < 20; k++) insert_lidar_point_slot(3.0f, 0.0f, GZ + 1.0f * k / 19.0f, 0, 0, 0, 1, GZ);
    classify_dirty_cells(GZ);
    GridCell* dust = query_world(3.0f, 0.0f);
    assert(dust && dust->obstacle_flag == 1 && "1 m dust column must start lethal");
    for (uint32_t f = 2; f <= 60; f++) {
        insert_lidar_point_slot(-3.0f, 0.0f, GZ, 0, 0, 0, f, GZ);   // other traffic keeps the node busy
        classify_dirty_cells(GZ);
        run_temporal_decay(f);
    }
    printf("[GATE 2] after 6 s unseen: flag=%d trav=%.3f (legacy node re-flagged it lethal every frame)\n",
           dust->obstacle_flag, dust->traversability);
    assert(dust->obstacle_flag == 2 && "decayed obstacle must stay UNKNOWN");
    for (int k = 0; k < 10; k++) insert_lidar_point_slot(3.0f, 0.0f, GZ + 0.01f * (k % 2), 0, 0, 0, 61, GZ);
    classify_dirty_cells(GZ);
    printf("[GATE 2] ground re-observed: flag=%d h_max-h_min=%.3f\n", dust->obstacle_flag, dust->h_max - dust->h_min);
    assert(dust->obstacle_flag == 0 && "dust must clear once ground is seen again");

    // ── Gate 3: ground reference. With ground_z hardcoded to 0 a 0.8 m rock at z=-1.2 is never lethal ──
    reset_map_pool();
    for (int k = 0; k < 10; k++) insert_lidar_point_slot(4.0f, 1.0f, GZ + 0.8f * k / 9.0f, 4, 0, 0, 1, GZ);
    GridCell* rock = query_world(4.0f, 1.0f);
    classify_obstacle(*rock, 0.0f);   int legacy_flag = rock->obstacle_flag;
    classify_obstacle(*rock, GZ);     int fixed_flag  = rock->obstacle_flag;
    printf("[GATE 3] 0.8 m rock: legacy(ground_z=0)=%d  fixed(ground_z=-1.2)=%d\n", legacy_flag, fixed_flag);
    assert(legacy_flag != 1 && fixed_flag == 1);

    // ── Gate 4: z filter is relative — legacy [-2,15] absolute filter dropped a 1 m downhill slope ──
    reset_map_pool();
    int ok_down = insert_lidar_point_slot(30.0f, 0.0f, GZ - 1.0f, 0, 0, 0, 1, GZ);
    printf("[GATE 4] ground 1 m below start pose (z=%.1f): %s\n", GZ - 1.0f, ok_down >= 0 ? "kept" : "DROPPED");
    assert(ok_down >= 0);

    // ── Gate 5: out-of-range labels no longer index past SEM_TRAV ──
    reset_map_pool();
    int s5 = insert_lidar_point_slot(1.0f, 1.0f, 0.0f, 250, 0, 0, 1, 0.0f);
    assert(s5 >= 0 && g_hash_pool[s5].semantic_label == 6);
    printf("[GATE 5] label 250 clamped to UNKNOWN\n");

    // ── Gate 6: cross-level alignment. Every cell must nest in exactly one cell of each coarser level
    //    (problem statement: "variable resolution without alignment errors or data loss") ──
    long straddle = 0, parent_mismatch = 0, samples = 0;
    for (long k = 0; k < 4000000; k++) {
        float x = -100.0f + k * 0.00005f;
        samples++;
        for (int lv = 0; lv + 1 < NUM_LEVELS; lv++) {
            int32_t fine = cell_index(x, lv), coarse = cell_index(x, lv + 1);
            if (LEVEL_DIV[lv + 1] % LEVEL_DIV[lv] != 0) straddle++;
            if (floor_div(fine, LEVEL_DIV[lv + 1] / LEVEL_DIV[lv]) != coarse) parent_mismatch++;
        }
    }
    printf("[GATE 6] %ld positions x %d level pairs: non-nesting ratios=%ld parent mismatches=%ld\n",
           samples, NUM_LEVELS - 1, straddle, parent_mismatch);
    assert(straddle == 0 && parent_mismatch == 0 && "levels must nest exactly");
    assert(resolve_resolution(9.99f, 0, 0, 0).cell_size == 0.05f && resolve_resolution(99.0f, 0, 0, 0).cell_size == 0.50f);

    // ── Sustained drive: synthetic Ouster OS1-64 (45 deg VFOV, 1.2 m mount, tree line 20-90 m) ──
    vector<float> sx, sy, sz; vector<uint8_t> sl;
    uint32_t rng = 2026;
    auto urand = [&]() { rng = rng * 1664525u + 1013904223u; return (rng >> 8) / 16777216.0f; };
    for (int b = 0; b < 64; b++) {
        float elev = (-22.5f + 45.0f * b / 63.0f) * (float)M_PI / 180.0f;
        for (int c = 0; c < cols; c++) {
            float az = 2.0f * (float)M_PI * c / cols, r, z; uint8_t lab;
            if (elev < -0.01f) { r = 1.2f / tan(-elev); z = -1.2f + (urand() - 0.5f) * 0.04f; lab = 0; }
            else if (urand() < 0.35f) { r = 20.0f + 70.0f * urand(); z = r * tan(elev); lab = 3; }
            else continue;
            if (r > 100.0f) continue;
            sx.push_back(r * cos(az)); sy.push_back(r * sin(az)); sz.push_back(z); sl.push_back(lab);
        }
    }
    const float PURGE_MARGIN_M = 20.0f;   // capped per level at 0.25 x band radius (2.5 / 6.25 / 20 m)
    const int   PURGE_EVERY    = 10;      // 1 s at 10 Hz
    reset_map_pool();
    double t_ins = 0, t_cls = 0, t_dec = 0, t_pur = 0, worst = 0;
    float peak_load = 0;
    printf("[DRIVE] speed=%.1f m/s  pts/scan=%zu  hz=%.0f  frames=%d (%.1f km)\n",
           speed, sx.size(), hz, frames, speed * frames / hz / 1000.0f);
    for (int f = 1; f <= frames; f++) {
        float rx = speed * (f - 1) / hz;
        auto t0 = chrono::steady_clock::now();
        for (size_t i = 0; i < sx.size(); i++)
            insert_lidar_point_slot(sx[i] + rx, sy[i], sz[i], sl[i], rx, 0.0f, (uint32_t)f, GZ);
        auto t1 = chrono::steady_clock::now();
        classify_dirty_cells(GZ);
        auto t2 = chrono::steady_clock::now();
        run_temporal_decay((uint32_t)f);
        auto t3 = chrono::steady_clock::now();
        if (f % PURGE_EVERY == 0) purge_distant_cells(rx, 0.0f, PURGE_MARGIN_M);
        auto t4 = chrono::steady_clock::now();
        double a = chrono::duration<double, milli>(t1 - t0).count(), b = chrono::duration<double, milli>(t2 - t1).count();
        double c = chrono::duration<double, milli>(t3 - t2).count(), d = chrono::duration<double, milli>(t4 - t3).count();
        t_ins += a; t_cls += b; t_dec += c; t_pur += d; worst = max(worst, a + b + c + d);
        peak_load = max(peak_load, get_load_factor());
        if (f == 1 || f % 500 == 0)
            printf("  frame %5d  insert=%6.1f  classify_dirty=%5.2f  decay=%5.2f  purge=%5.2f ms  load=%.3f  fail=%llu\n",
                   f, a, b, c, d, get_load_factor(), (unsigned long long)g_insert_fail);
    }
    printf("[DRIVE] mean ms: insert=%.1f classify_dirty=%.2f decay(avg)=%.2f purge(avg)=%.2f  worst_frame=%.1f\n",
           t_ins / frames, t_cls / frames, t_dec / frames, t_pur / frames, worst);
    printf("[DRIVE] peak_load=%.3f  insert_failures=%llu\n", peak_load, (unsigned long long)g_insert_fail);
    assert(g_insert_fail == 0 && peak_load < LOAD_WARN && "bounded memory over sustained drive");

    // ── Memory vs uniform maps (problem statement: "significant reduction in memory usage compared to a
    //    uniform high-resolution 3D map"). Same 100 m radius, same 5 cm finest resolution. ──
    {
        const double MB = 1024.0 * 1024.0, R = MAX_RANGE, H = 16.0;          // z window Z_REL_MIN..Z_REL_MAX + 1 m
        double used_mb   = g_valid_cells * sizeof(GridCell) / MB;
        double pool_mb   = sizeof(g_hash_pool) / MB;
        double disk_cells = M_PI * R * R / (BASE_RES * BASE_RES);
        double dense25_mb = disk_cells * sizeof(GridCell) / MB;                // uniform 5 cm 2.5D grid
        double dense3d_mb = disk_cells * (H / BASE_RES) * 1.0 / MB;            // uniform 5 cm voxels, 1 byte each
        unordered_set<uint64_t> vox;                                           // sparse 5 cm 3D map of ONE scan
        float rx = speed * (frames - 1) / hz;
        for (size_t i = 0; i < sx.size(); i++) {
            uint64_t kx = (uint32_t)base_index(sx[i] + rx), ky = (uint32_t)base_index(sy[i]);
            uint64_t kz = (uint32_t)(int32_t)floorf((sz[i] - GZ + 2.0f) / BASE_RES);
            vox.insert((kx & 0x1FFFFF) | ((ky & 0x1FFFFF) << 21) | ((kz & 0x3FFFFF) << 42));
        }
        double sparse3d_mb = vox.size() * 16.0 / MB;                           // key + occupancy, no hash overhead
        printf("[MEMORY] foveated 2.5D in use: %d cells = %.1f MB (fixed pool %.1f MB)\n", g_valid_cells, used_mb, pool_mb);
        printf("[MEMORY] uniform 5 cm 2.5D grid, r=100 m: %.0f MB  (%.0fx larger than in-use)\n", dense25_mb, dense25_mb / used_mb);
        printf("[MEMORY] uniform 5 cm 3D voxel grid, r=100 m, %.0f m tall, 1 B/voxel: %.0f MB  (%.0fx)\n", H, dense3d_mb, dense3d_mb / used_mb);
        printf("[MEMORY] sparse 5 cm 3D voxels of a single scan (lower bound, 16 B/voxel): %zu voxels = %.1f MB\n",
               vox.size(), sparse3d_mb);
    }
    printf("[ENGINE REGRESSION + LOAD TEST PASSED]\n");
    return 0;
}

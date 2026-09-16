#include "drdo_lidar_mapping/drdo_map.h"
#include <cassert>
#include <iostream>
#include <cmath>

using namespace std;

void test_spatial_hashing() {
    cout << "[TEST] 1. Spatial Hash & Collision Probing..." << endl;
    reset_map_pool();
    int b1 = spatial_hash(10, 20, 0);
    int b2 = spatial_hash(10, 20, 0);
    assert(b1 == b2 && "Hash function must be deterministic");
    assert(b1 >= 0 && b1 < HASH_TABLE_SIZE && "Bucket must be in bounds");

    int s1 = insert_cell(10, 20, 0);
    int s2 = insert_cell(10, 20, 0);
    assert(s1 == s2 && "Duplicate insert must return existing slot");
    assert(s1 >= 0 && "Valid slot index");

    GridCell* c = lookup_cell(10, 20, 0);
    assert(c != nullptr && "Lookup must find inserted cell");
    assert(c->ix == 10 && c->iy == 20 && c->level == 0);
    cout << "  -> PASS" << endl;
}

void test_foveated_resolution() {
    cout << "[TEST] 2. Foveated Multi-Resolution Mapping..." << endl;
    ResolvedCell r0 = resolve_resolution(2.0f, 2.0f, 0.0f, 0.0f); // dist ~ 2.8m -> Level 0
    assert(r0.level == 0 && !r0.discard && abs(r0.cell_size - 0.05f) < 1e-4);

    ResolvedCell r1 = resolve_resolution(15.0f, 0.0f, 0.0f, 0.0f); // dist 15m -> Level 1
    assert(r1.level == 1 && !r1.discard && abs(r1.cell_size - 0.10f) < 1e-4);

    ResolvedCell r2 = resolve_resolution(70.0f, 0.0f, 0.0f, 0.0f); // dist 70m -> Level 2
    assert(r2.level == 2 && !r2.discard && abs(r2.cell_size - 0.50f) < 1e-4);

    ResolvedCell r10 = resolve_resolution(9.99f, 0.0f, 0.0f, 0.0f); // 5 cm cells within the 10 m radius
    assert(r10.level == 0);

    ResolvedCell r_out = resolve_resolution(150.0f, 0.0f, 0.0f, 0.0f); // dist > 100m -> Discard
    assert(r_out.discard);
    cout << "  -> PASS" << endl;
}

void test_welford_height_stats() {
    cout << "[TEST] 3. Welford Online Height Statistics..." << endl;
    reset_map_pool();
    int slot = insert_cell(5, 5, 0);
    GridCell& c = g_hash_pool[slot];

    update_height(c, 1.0f);
    update_height(c, 2.0f);
    update_height(c, 3.0f);

    assert(c.hit_count == 3);
    assert(abs(c.h_mean - 2.0f) < 1e-4 && "Mean of {1, 2, 3} must be 2.0");
    assert(c.h_min == 1.0f && c.h_max == 3.0f);
    float var = c.h_variance_M2 / (c.hit_count - 1);
    assert(abs(var - 1.0f) < 1e-4 && "Sample variance of {1, 2, 3} must be 1.0");
    cout << "  -> PASS" << endl;
}

void test_raycasting_and_carving() {
    cout << "[TEST] 4. Bresenham Raycasting & Free Space Carving..." << endl;
    reset_map_pool();
    // Raycast from (0,0) to (5, 0)
    raycast_bresenham(0, 0, 5, 0, 0);
    // Cells (0,0) through (4,0) should be free (obstacle_flag == 0)
    for (int x = 0; x < 5; x++) {
        GridCell* cell = lookup_cell(x, 0, 0);
        assert(cell != nullptr && "Raycasted cells must be allocated");
        assert(cell->obstacle_flag == 0 && "Carved cells must be FREE");
    }
    cout << "  -> PASS" << endl;
}

void test_traversability_and_decay() {
    cout << "[TEST] 5. Traversability & Temporal Decay..." << endl;
    reset_map_pool();
    int slot = insert_cell(1, 1, 0);
    GridCell& c = g_hash_pool[slot];
    c.semantic_label = 0; // GROUND
    c.obstacle_flag = 0;  // FREE
    c.hit_count = 10;
    c.last_update_ts = 1;
    compute_traversability(c);
    assert(c.traversability > 0.9f && "Ground with 10 hits must have high traversability");

    // Advance frame to 60 (diff 59 > DECAY_WINDOW 50)
    int decayed = run_temporal_decay(60);
    assert(decayed > 0 && "Cell must decay after 50 frames");
    cout << "  -> PASS" << endl;
}

int main() {
    cout << "==========================================================" << endl;
    cout << "   DRDO ID26053 — C++ Grid Engine Consolidated Test Suite " << endl;
    cout << "==========================================================" << endl;

    test_spatial_hashing();
    test_foveated_resolution();
    test_welford_height_stats();
    test_raycasting_and_carving();
    test_traversability_and_decay();

    cout << "==========================================================" << endl;
    cout << "   [ALL C++ GRID ENGINE UNIT TESTS PASSED SUCCESSFULLY]   " << endl;
    cout << "==========================================================" << endl;
    return 0;
}

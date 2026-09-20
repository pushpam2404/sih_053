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

// The problem statement names potholes alongside curbs and overhangs. Before classify_obstacle()
// gained its negative-obstacle branch it measured only UPWARD from ground, so a pothole with a
// flat floor produced step_height ~ 0 and came back FREE with traversability 1.00 — the planner
// was told to drive into the hole at full confidence. Both halves are pinned here: the hole must
// be caught, and a downgrade must NOT be, because "everything below ground_z is a hole" would
// flag every descending slope as one continuous trench.
void test_negative_obstacles() {
    cout << "[TEST] 6. Negative Obstacles (potholes / trenches)..." << endl;
    const float GZ = -1.2f;   // FAST-LIO's origin is the IMU start pose, so ground is below 0

    // Sampled at 2 cm so each 5 cm cell collects several returns. NEG_MIN_HITS requires
    // corroboration before calling a cell a hole — h_max of a one-hit cell is just that one
    // sample, and terrain noise alone would manufacture potholes. On the vehicle the same
    // threshold is reached either from beam density at close range (~3 cm azimuth spacing at 5 m)
    // or across consecutive scans, since hit_count accumulates in a persistent cell.
    reset_map_pool();
    for (float x = 4.0f; x < 6.0f; x += 0.02f)
        for (float y = -1.0f; y < 1.0f; y += 0.02f) {
            bool hole = (x > 4.8f && x < 5.2f && y > -0.2f && y < 0.2f);   // 40 cm pit, 40 cm deep
            insert_lidar_point(x, y, hole ? GZ - 0.40f : GZ, 0, 0.0f, 0.0f, 1, GZ);
        }
    classify_and_score_all_cells(GZ);

    GridCell* pit = query_world(5.0f, 0.0f);
    assert(pit && "pothole floor must be mapped");
    assert(pit->obstacle_flag == 3 && "40 cm pothole must classify NEGATIVE, not FREE");
    assert(pit->traversability == 0.0f && "a hole is as impassable as a rock");

    GridCell* rim = query_world(4.3f, 0.0f);
    assert(rim && rim->obstacle_flag != 3 && "intact ground beside the hole must stay drivable");
    cout << "  pothole floor flag=" << (int)pit->obstacle_flag
         << " trav=" << pit->traversability << ", surrounding ground flag="
         << (int)rim->obstacle_flag << endl;

    // A uniform downgrade drags the whole neighbourhood below ground_z together, so no cell keeps
    // a rim at ground level and nothing should be flagged. This is the test that stops the
    // feature from condemning every hill.
    for (float grade : {0.02f, 0.04f}) {
        reset_map_pool();
        for (float x = 2.0f; x < 20.0f; x += 0.02f)
            for (float y = -1.0f; y < 1.0f; y += 0.02f)
                insert_lidar_point(x, y, GZ - grade * (x - 2.0f), 0, 0.0f, 0.0f, 1, GZ);
        classify_and_score_all_cells(GZ);
        int neg = 0, tot = 0;
        for (int i = 0; i < HASH_TABLE_SIZE; i++) {
            GridCell& c = g_hash_pool[i];
            if (!c.valid || c.hit_count == 0) continue;
            tot++;
            if (c.obstacle_flag == 3) neg++;
        }
        cout << "  " << (int)(grade * 100) << "% downgrade: " << neg << " of " << tot
             << " cells flagged NEGATIVE" << endl;
        assert(neg == 0 && "a smooth downgrade is not a trench");
    }
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
    test_negative_obstacles();

    cout << "==========================================================" << endl;
    cout << "   [ALL C++ GRID ENGINE UNIT TESTS PASSED SUCCESSFULLY]   " << endl;
    cout << "==========================================================" << endl;
    return 0;
}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "drdo_lidar_mapping/drdo_map.h"

namespace py = pybind11;

static py::object query_world_py(float wx, float wy) {
    GridCell* f = query_world(wx, wy);
    if (!f) return py::none();
    py::dict d;
    d["ix"]             = f->ix;
    d["iy"]             = f->iy;
    d["level"]          = (int)f->level;
    d["h_min"]          = f->h_min;
    d["h_max"]          = f->h_max;
    d["h_mean"]         = f->h_mean;
    d["hit_count"]      = f->hit_count;
    d["obstacle_flag"]  = (int)f->obstacle_flag;
    d["semantic_label"] = (int)f->semantic_label;
    d["traversability"] = f->traversability;
    d["last_update_ts"] = f->last_update_ts;
    return d;
}

static int decay_kernel_py(uint32_t cf) {
    return run_temporal_decay(cf);
}

static int classify_and_score_all_py(float gz) {
    return classify_and_score_all_cells(gz);
}

static int classify_dirty_py(float gz) {
    return classify_dirty_cells(gz);
}

static int purge_distant_py(float rx, float ry, float margin_m) {
    return purge_distant_cells(rx, ry, margin_m);
}

static int valid_cell_count_py() {
    return get_valid_cell_count();
}

static void reset_map_py() {
    reset_map_pool();
}

static bool insert_point_py(float x, float y, float z, int label, float rx, float ry, uint32_t ts, float gz) {
    return insert_lidar_point(x, y, z, label, rx, ry, ts, gz);
}

// Batch insert: (N,4) float32 [x, y, z, label]. One call per scan instead of N Python->C++ crossings.
static int insert_points_py(py::array_t<float, py::array::c_style | py::array::forcecast> pts,
                            float rx, float ry, uint32_t ts, float gz) {
    auto a = pts.unchecked<2>();
    if (a.shape(1) < 4) throw std::runtime_error("insert_points expects an (N, 4) array [x, y, z, label]");
    int ok = 0;
    py::gil_scoped_release release;
    for (py::ssize_t i = 0; i < a.shape(0); i++) {
        float lab = a(i, 3);
        int label = (lab >= 0.0f && lab < 256.0f) ? (int)lab : 6;
        ok += insert_lidar_point_slot(a(i, 0), a(i, 1), a(i, 2), label, rx, ry, ts, gz) >= 0;
    }
    return ok;
}

// Snapshot of every valid cell as parallel numpy arrays (for dashboards / offline analysis).
static py::dict export_cells_py() {
    int n = get_valid_cell_count();
    py::array_t<int32_t> ix(n), iy(n);
    py::array_t<uint8_t> level(n), label(n), flag(n);
    py::array_t<float> h_min(n), h_max(n), h_mean(n), trav(n), size(n);
    py::array_t<uint32_t> hits(n), ts(n), carve(n);
    int k = 0;
    for (int i = 0; i < HASH_TABLE_SIZE && k < n; i++) {
        const GridCell& c = g_hash_pool[i];
        if (!c.valid) continue;
        ix.mutable_at(k) = c.ix; iy.mutable_at(k) = c.iy; level.mutable_at(k) = c.level;
        label.mutable_at(k) = c.semantic_label; flag.mutable_at(k) = c.obstacle_flag;
        h_min.mutable_at(k) = c.h_min; h_max.mutable_at(k) = c.h_max; h_mean.mutable_at(k) = c.h_mean;
        trav.mutable_at(k) = c.traversability; hits.mutable_at(k) = c.hit_count; ts.mutable_at(k) = c.last_update_ts;
        carve.mutable_at(k) = g_carve_ts[i];
        size.mutable_at(k) = CELL_RESOLUTIONS[c.level < NUM_LEVELS ? c.level : NUM_LEVELS - 1];
        k++;
    }
    py::dict d;
    d["ix"] = ix; d["iy"] = iy; d["level"] = level; d["semantic_label"] = label; d["obstacle_flag"] = flag;
    d["h_min"] = h_min; d["h_max"] = h_max; d["h_mean"] = h_mean; d["traversability"] = trav;
    d["hit_count"] = hits; d["last_update_ts"] = ts; d["carve_ts"] = carve; d["cell_size"] = size;
    return d;
}

PYBIND11_MODULE(drdo_map, m) {
    m.doc() = "DRDO ID26053 — Foveated 2.5D Adaptive Grid Map Engine (C++ Pybind11 Bridge)";
    m.def("reset_map",              &reset_map_py,              "Reset the global spatial hash map pool");
    m.def("insert_point",           &insert_point_py,           "Insert a labeled LiDAR point into the multi-resolution map",
          py::arg("x"), py::arg("y"), py::arg("z"), py::arg("label"),
          py::arg("robot_x"), py::arg("robot_y"), py::arg("frame_ts"), py::arg("ground_z") = 0.0f);
    m.def("query_world",            &query_world_py,            "Query world coordinates (wx, wy) across resolution pyramid",
          py::arg("wx"), py::arg("wy"));
    m.def("decay_kernel",           &decay_kernel_py,           "Run temporal log-odds decay kernel for frame_ts",
          py::arg("current_frame"));
    m.def("classify_and_score_all", &classify_and_score_all_py, "Full rescan; overwrites decay if called every frame",
          py::arg("ground_z") = 0.0f);
    m.def("classify_dirty",         &classify_dirty_py,         "Classify only cells updated since the last call",
          py::arg("ground_z") = 0.0f);
    m.def("purge_distant",          &purge_distant_py,          "Evict cells beyond their foveation band radius + margin",
          py::arg("robot_x"), py::arg("robot_y"), py::arg("margin_m") = 20.0f);
    m.def("valid_cell_count",       &valid_cell_count_py,       "Return count of allocated valid grid cells");
    m.def("load_factor",            &get_load_factor,           "Hash pool load factor (warn above 0.70)");
    m.def("insert_points",          &insert_points_py,          "Insert an (N,4) float32 [x, y, z, label] scan; returns points kept",
          py::arg("points"), py::arg("robot_x"), py::arg("robot_y"), py::arg("frame_ts"), py::arg("ground_z") = 0.0f);
    m.def("export_cells",           &export_cells_py,           "All valid cells as a dict of numpy arrays");
    m.attr("CELL_RESOLUTIONS") = py::make_tuple(CELL_RESOLUTIONS[0], CELL_RESOLUTIONS[1], CELL_RESOLUTIONS[2]);
    m.attr("LEVEL_OUTER_R")    = py::make_tuple(LEVEL_OUTER_R[0], LEVEL_OUTER_R[1], LEVEL_OUTER_R[2]);
    m.attr("CELL_BYTES")       = (int)sizeof(GridCell);
    m.attr("POOL_CELLS")       = HASH_TABLE_SIZE;
    m.def("insert_fail_count",      []() { return (unsigned long long)g_insert_fail; },
          "Points dropped because the pool was full");
}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "drdo_lidar_mapping/drdo_map.h"

namespace py = pybind11;

static py::object query_world_py(float wx, float wy) {
    for (int lv = 0; lv < 4; lv++) {
        int32_t qix = (int32_t)floorf(wx / CELL_RESOLUTIONS[lv]);
        int32_t qiy = (int32_t)floorf(wy / CELL_RESOLUTIONS[lv]);
        GridCell* f = lookup_cell(qix, qiy, (uint8_t)lv);
        if (f) {
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
    }
    return py::none();
}

static int decay_kernel_py(uint32_t cf) {
    return run_temporal_decay(cf);
}

static int classify_and_score_all_py(float gz) {
    return classify_and_score_all_cells(gz);
}

static int valid_cell_count_py() {
    return get_valid_cell_count();
}

static void reset_map_py() {
    reset_map_pool();
}

static bool insert_point_py(float x, float y, float z, int label, float rx, float ry, uint32_t ts) {
    return insert_lidar_point(x, y, z, label, rx, ry, ts);
}

PYBIND11_MODULE(drdo_map, m) {
    m.doc() = "DRDO ID26053 — Foveated 2.5D Adaptive Grid Map Engine (C++ Pybind11 Bridge)";
    m.def("reset_map",              &reset_map_py,              "Reset the global spatial hash map pool");
    m.def("insert_point",           &insert_point_py,           "Insert a labeled LiDAR point into the multi-resolution map",
          py::arg("x"), py::arg("y"), py::arg("z"), py::arg("label"),
          py::arg("robot_x"), py::arg("robot_y"), py::arg("frame_ts"));
    m.def("query_world",            &query_world_py,            "Query world coordinates (wx, wy) across resolution pyramid",
          py::arg("wx"), py::arg("wy"));
    m.def("decay_kernel",           &decay_kernel_py,           "Run temporal log-odds decay kernel for frame_ts",
          py::arg("current_frame"));
    m.def("classify_and_score_all", &classify_and_score_all_py, "Run obstacle classifier and traversability score over map",
          py::arg("ground_z") = 0.0f);
    m.def("valid_cell_count",       &valid_cell_count_py,       "Return count of allocated valid grid cells");
}

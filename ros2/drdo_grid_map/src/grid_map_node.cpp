// phase3/ros2/drdo_grid_map/src/grid_map_node.cpp
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <cstring>
#include <cmath>
#include <chrono>
#include "drdo_map.h"

using namespace std;
using namespace std::chrono_literals;

// ── Static free functions (wrap Phase 1+2 lambdas) ───────────────────────────

static int spatial_hash_fn(int32_t ix, int32_t iy, uint8_t level) {
    uint64_t h = (uint64_t)(uint32_t)ix  * 2654435761ULL;
    h ^= (uint64_t)(uint32_t)iy * 805459861ULL;
    h ^= (uint64_t)level        * 1234567891ULL;
    return (int)(h % (uint64_t)HASH_TABLE_SIZE);
}

static int insert_cell_fn(int32_t ix, int32_t iy, uint8_t level) {
    int bucket = spatial_hash_fn(ix, iy, level);
    for (int probe = 0; probe < MAX_PROBE; probe++) {
        int slot = (bucket + probe) % HASH_TABLE_SIZE;
        GridCell& c = g_hash_pool[slot];
        if (c.valid && c.ix == ix && c.iy == iy && c.level == level) return slot;
        if (!c.valid) {
            c.h_min = +1e9f; c.h_max = -1e9f;
            c.h_mean = 0.0f; c.h_variance_M2 = 0.0f;
            c.traversability = 0.4f; c.hit_count = 0;
            c.last_update_ts = 0; c.semantic_label = 6;
            c.obstacle_flag = 2; c.ix = ix; c.iy = iy;
            c.level = level; c.valid = true;
            return slot;
        }
    }
    return -1;
}

static GridCell* lookup_cell_fn(int32_t ix, int32_t iy, uint8_t level) {
    int bucket = spatial_hash_fn(ix, iy, level);
    for (int probe = 0; probe < MAX_PROBE; probe++) {
        int slot = (bucket + probe) % HASH_TABLE_SIZE;
        GridCell& c = g_hash_pool[slot];
        if (!c.valid) return nullptr;
        if (c.ix == ix && c.iy == iy && c.level == level) return &c;
    }
    return nullptr;
}

static ResolvedCell resolve_resolution_fn(float wx, float wy, float rx, float ry) {
    float dx = wx-rx, dy = wy-ry, dist = sqrtf(dx*dx + dy*dy);
    ResolvedCell r; r.discard = false;
    if      (dist <=  5.0f) { r.level = 0; r.cell_size = 0.05f; }
    else if (dist <  20.0f) { r.level = 1; r.cell_size = 0.20f; }
    else if (dist <  50.0f) { r.level = 2; r.cell_size = 0.50f; }
    else if (dist <= 100.0f){ r.level = 3; r.cell_size = 1.00f; }
    else                    { r.discard = true; return r; }
    r.ix = (int32_t)floorf(wx / r.cell_size);
    r.iy = (int32_t)floorf(wy / r.cell_size);
    return r;
}

static void update_height_fn(GridCell& c, float z) {
    c.hit_count++;
    float delta = z - c.h_mean;
    c.h_mean += delta / (float)c.hit_count;
    c.h_variance_M2 += delta * (z - c.h_mean);
    if (z < c.h_min) c.h_min = z;
    if (z > c.h_max) c.h_max = z;
}

static void mark_free_cell_fn(int32_t ix, int32_t iy, uint8_t level) {
    int slot = insert_cell_fn(ix, iy, level);
    if (slot < 0) return;
    GridCell& c = g_hash_pool[slot];
    if (c.obstacle_flag == 1) c.obstacle_flag = 2;
    if (c.obstacle_flag == 2 && c.hit_count == 0) c.obstacle_flag = 0;
}

static void raycast_fn(int32_t ox, int32_t oy, int32_t hx, int32_t hy, uint8_t lv) {
    int dx = abs(hx-ox), dy = abs(hy-oy);
    int sx = (hx>ox)?1:-1, sy = (hy>oy)?1:-1, err = dx-dy, x=ox, y=oy;
    while (true) {
        if (x==hx && y==hy) break;
        mark_free_cell_fn((int32_t)x, (int32_t)y, lv);
        int e2 = 2*err;
        if (e2>-dy) { err-=dy; x+=sx; }
        if (e2< dx) { err+=dx; y+=sy; }
    }
}

static const float SEM_TRAV_TABLE[8] = {1.0f,0.8f,0.65f,0.15f,0.0f,0.0f,0.4f,-1.0f};

static void classify_obstacle_fn(GridCell& c, float gz) {
    if (c.hit_count == 0) return;
    float hs = c.h_max-c.h_min, cl = c.h_min-gz, ma = c.h_max-gz;
    if      (hs < 0.05f) c.obstacle_flag = 0;
    else if (cl > 0.50f) c.obstacle_flag = 0;
    else if (ma > 0.30f) c.obstacle_flag = 1;
    else                 c.obstacle_flag = 2;
}

static void compute_traversability_fn(GridCell& c) {
    if (c.obstacle_flag == 1) { c.traversability = 0.0f; return; }
    float rough = (c.hit_count>1) ? sqrtf(c.h_variance_M2/(float)(c.hit_count-1)) : 0.0f;
    float hs    = expf(-5.0f * rough);
    float ss    = SEM_TRAV_TABLE[c.semantic_label];
    float conf  = fminf((float)c.hit_count / 10.0f, 1.0f);
    c.traversability = conf * (0.5f * hs + 0.5f * ss);
}

class GridMapNode : public rclcpp::Node {
public:
    GridMapNode()
        : Node("drdo_grid_map_node"),
          tf_buffer_(get_clock()),
          tf_listener_(tf_buffer_) {
        memset(g_hash_pool, 0, sizeof(g_hash_pool));
        RCLCPP_INFO(get_logger(), "[DRDO] Hash pool init (40MB). Waiting for LiDAR...");

        pc_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
            "/lidar/points_raw", rclcpp::SensorDataQoS(),
            std::bind(&GridMapNode::on_pointcloud, this, std::placeholders::_1));

        costmap_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/drdo/costmap", 10);

        publish_timer_ = create_wall_timer(
            100ms, std::bind(&GridMapNode::publish_costmap, this));

        frame_ts_ = 0;
    }

private:
    void on_pointcloud(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        frame_ts_++;

        // Get robot pose from TF
        float robot_x = 0.0f, robot_y = 0.0f;
        try {
            auto tf = tf_buffer_.lookupTransform("map", "base_link", tf2::TimePointZero);
            robot_x = (float)tf.transform.translation.x;
            robot_y = (float)tf.transform.translation.y;
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                "[DRDO] TF failed: %s — using (0,0)", ex.what());
        }

        // Parse PointCloud2 field offsets
        int off_x=-1, off_y=-1, off_z=-1, off_label=-1;
        for (const auto& f : msg->fields) {
            if (f.name=="x")     off_x     = f.offset;
            if (f.name=="y")     off_y     = f.offset;
            if (f.name=="z")     off_z     = f.offset;
            if (f.name=="label") off_label = f.offset;
        }
        if (off_x<0||off_y<0||off_z<0) {
            RCLCPP_ERROR_ONCE(get_logger(), "[DRDO] PointCloud2 missing x/y/z!");
            return;
        }

        const uint8_t* data = msg->data.data();
        uint32_t step = msg->point_step;
        uint32_t N    = msg->width * msg->height;
        int inserted = 0;

        for (uint32_t i = 0; i < N; i++) {
            const uint8_t* pt = data + i*step;
            float wx, wy, wz;
            memcpy(&wx, pt+off_x, 4);
            memcpy(&wy, pt+off_y, 4);
            memcpy(&wz, pt+off_z, 4);

            uint8_t label = 6;  // UNKNOWN default
            if (off_label >= 0) {
                uint8_t rl; memcpy(&rl, pt+off_label, 1);
                label = (rl < 8) ? rl : 6;
            }

            if (label==7 || wz>15.0f || wz<-2.0f) continue;

            ResolvedCell rc = resolve_resolution_fn(wx, wy, robot_x, robot_y);
            if (rc.discard) continue;

            int32_t rox = (int32_t)floorf(robot_x / rc.cell_size);
            int32_t roy = (int32_t)floorf(robot_y / rc.cell_size);
            raycast_fn(rox, roy, rc.ix, rc.iy, (uint8_t)rc.level);

            int slot = insert_cell_fn(rc.ix, rc.iy, (uint8_t)rc.level);
            if (slot < 0) continue;
            GridCell& c = g_hash_pool[slot];
            update_height_fn(c, wz);
            c.semantic_label = label;
            c.last_update_ts = frame_ts_;
            inserted++;
        }

        // Classify + score all valid cells
        for (int i = 0; i < HASH_TABLE_SIZE; i++) {
            if (!g_hash_pool[i].valid) continue;
            classify_obstacle_fn(g_hash_pool[i], 0.0f);
            compute_traversability_fn(g_hash_pool[i]);
        }

        // Temporal decay every DECAY_K frames
        if (frame_ts_ % DECAY_K == 0) {
            int decayed = 0;
            for (int i = 0; i < HASH_TABLE_SIZE; i++) {
                GridCell& c = g_hash_pool[i];
                if (!c.valid) continue;
                if ((frame_ts_ - c.last_update_ts) > (uint32_t)DECAY_WINDOW) {
                    c.traversability *= DECAY_FACTOR;
                    if (c.traversability < DECAY_MIN_TRAV && c.obstacle_flag == 1)
                        c.obstacle_flag = 2;
                    decayed++;
                }
            }
            RCLCPP_DEBUG(get_logger(), "[DRDO] Frame %u: %d decayed | %d inserted.",
                         frame_ts_, decayed, inserted);
        }
    }

    void publish_costmap() {
        constexpr int   GW = 200, GH = 200;
        constexpr float CS = 1.00f;   // Level-3 cell size for costmap
        auto msg = nav_msgs::msg::OccupancyGrid();
        msg.header.stamp    = get_clock()->now();
        msg.header.frame_id = "map";
        msg.info.resolution = CS;
        msg.info.width      = GW;
        msg.info.height     = GH;
        msg.info.origin.position.x = -GW * CS / 2.0;
        msg.info.origin.position.y = -GH * CS / 2.0;
        msg.data.assign(GW * GH, -1);

        float CSIZES[4] = {0.05f, 0.20f, 0.50f, 1.00f};
        for (int j = 0; j < GH; j++) {
            for (int i = 0; i < GW; i++) {
                float wx = msg.info.origin.position.x + (i+0.5f)*CS;
                float wy = msg.info.origin.position.y + (j+0.5f)*CS;
                GridCell* cell = nullptr;
                for (int lv = 0; lv < 4 && !cell; lv++) {
                    int32_t qx = (int32_t)floorf(wx/CSIZES[lv]);
                    int32_t qy = (int32_t)floorf(wy/CSIZES[lv]);
                    cell = lookup_cell_fn(qx, qy, (uint8_t)lv);
                }
                int8_t val = -1;
                if (cell) {
                    if      (cell->obstacle_flag==1) val = 100;
                    else if (cell->obstacle_flag==0) val = 0;
                    else                             val = 50;
                }
                msg.data[j*GW+i] = val;
            }
        }
        costmap_pub_->publish(msg);
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr      costmap_pub_;
    rclcpp::TimerBase::SharedPtr                                    publish_timer_;
    tf2_ros::Buffer                                                 tf_buffer_;
    tf2_ros::TransformListener                                      tf_listener_;
    uint32_t                                                        frame_ts_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GridMapNode>());
    rclcpp::shutdown();
    return 0;
}

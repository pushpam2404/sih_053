// ros2/drdo_grid_map/src/grid_map_node.cpp
//
// Foveated 2.5D grid map node. Engine logic lives in src/drdo_map.cpp (linked, not copy-pasted);
// all node wiring is in main() with lambdas.
//
// Topics (relative names so launch-file remappings apply):
//   sub  pointcloud      sensor_msgs/PointCloud2   (e.g. FAST-LIO2 /cloud_registered, any frame)
//   pub  occupancy_grid  nav_msgs/OccupancyGrid    (transient_local: Nav2 StaticLayer requires it)
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/time.h>
#include <tf2/exceptions.h>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <algorithm>
#include "drdo_lidar_mapping/drdo_map.h"

using namespace std;

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("drdo_grid_map_node");

    const string world_frame   = node->declare_parameter<string>("world_frame", "map");
    const string base_frame    = node->declare_parameter<string>("base_frame", "base_link");
    const double publish_hz    = node->declare_parameter<double>("publish_rate_hz", 5.0);
    const double grid_res      = node->declare_parameter<double>("grid_resolution", 0.20);
    const double grid_size_m   = node->declare_parameter<double>("grid_size_m", 100.0);
    const double min_range     = node->declare_parameter<double>("min_range", 0.3);        // ADL-3 self-return filter
    const double tf_timeout_s  = node->declare_parameter<double>("tf_timeout_s", 0.05);
    const int    purge_every   = (int)node->declare_parameter<int>("purge_every_n_frames", 10);
    const double purge_margin  = node->declare_parameter<double>("purge_margin_m", 20.0);

    reset_map_pool();
    RCLCPP_INFO(node->get_logger(), "[DRDO] Hash pool ready (%d cells, 40 MB). world=%s base=%s",
                HASH_TABLE_SIZE, world_frame.c_str(), base_frame.c_str());

    tf2_ros::Buffer tf_buffer(node->get_clock());
    tf2_ros::TransformListener tf_listener(tf_buffer);   // spins its own thread

    auto grid_pub = node->create_publisher<nav_msgs::msg::OccupancyGrid>(
        "occupancy_grid", rclcpp::QoS(rclcpp::KeepLast(1)).reliable().transient_local());

    uint32_t frame_ts = 0;
    float robot_x = 0.0f, robot_y = 0.0f, ground_z = 0.0f;
    bool have_pose = false;
    uint64_t reported_fail = 0;

    auto on_cloud = [&](sensor_msgs::msg::PointCloud2::ConstSharedPtr msg) {
        // 1. Robot pose at the scan timestamp. No pose -> skip the frame. The legacy node silently
        //    used (0,0), re-centring the foveation on the map origin and corrupting the map.
        geometry_msgs::msg::TransformStamped base_tf, cloud_tf;
        try {
            base_tf = tf_buffer.lookupTransform(world_frame, base_frame, tf2_ros::fromMsg(msg->header.stamp),
                                                tf2::durationFromSec(tf_timeout_s));
            cloud_tf = (msg->header.frame_id == world_frame)
                ? geometry_msgs::msg::TransformStamped()
                : tf_buffer.lookupTransform(world_frame, msg->header.frame_id, tf2_ros::fromMsg(msg->header.stamp),
                                            tf2::durationFromSec(tf_timeout_s));
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(node->get_logger(), *node->get_clock(), 2000,
                                 "[DRDO] TF %s<-%s unavailable, frame skipped: %s",
                                 world_frame.c_str(), base_frame.c_str(), ex.what());
            return;
        }
        robot_x  = (float)base_tf.transform.translation.x;
        robot_y  = (float)base_tf.transform.translation.y;
        ground_z = (float)base_tf.transform.translation.z;   // base_link sits on the ground plane
        have_pose = true;
        bool identity = (msg->header.frame_id == world_frame);
        const auto& q = cloud_tf.transform.rotation;
        const auto& t = cloud_tf.transform.translation;
        const float R[9] = {
            (float)(1 - 2 * (q.y * q.y + q.z * q.z)), (float)(2 * (q.x * q.y - q.z * q.w)), (float)(2 * (q.x * q.z + q.y * q.w)),
            (float)(2 * (q.x * q.y + q.z * q.w)), (float)(1 - 2 * (q.x * q.x + q.z * q.z)), (float)(2 * (q.y * q.z - q.x * q.w)),
            (float)(2 * (q.x * q.z - q.y * q.w)), (float)(2 * (q.y * q.z + q.x * q.w)), (float)(1 - 2 * (q.x * q.x + q.y * q.y))};

        // 2. Validate the PointCloud2 layout instead of blindly memcpy-ing 4 bytes per field.
        int off_x = -1, off_y = -1, off_z = -1, off_label = -1; uint8_t label_type = 0;
        for (const auto& f : msg->fields) {
            bool f32 = f.datatype == sensor_msgs::msg::PointField::FLOAT32;
            if (f.name == "x" && f32) off_x = (int)f.offset;
            if (f.name == "y" && f32) off_y = (int)f.offset;
            if (f.name == "z" && f32) off_z = (int)f.offset;
            if (f.name == "label") { off_label = (int)f.offset; label_type = f.datatype; }
        }
        size_t n_pts = (size_t)msg->width * msg->height;
        if (off_x < 0 || off_y < 0 || off_z < 0 || msg->is_bigendian ||
            msg->point_step < 12 || msg->data.size() < n_pts * msg->point_step) {
            RCLCPP_ERROR_THROTTLE(node->get_logger(), *node->get_clock(), 5000,
                                  "[DRDO] Unsupported PointCloud2 (need little-endian FLOAT32 x/y/z, consistent size)");
            return;
        }

        frame_ts++;
        const uint8_t* data = msg->data.data();
        const float min_r2 = (float)(min_range * min_range);
        int inserted = 0;
        for (size_t i = 0; i < n_pts; i++) {
            const uint8_t* pt = data + i * msg->point_step;
            float px, py, pz;
            memcpy(&px, pt + off_x, 4); memcpy(&py, pt + off_y, 4); memcpy(&pz, pt + off_z, 4);
            if (!isfinite(px) || !isfinite(py) || !isfinite(pz)) continue;

            int label = 6;   // UNKNOWN unless a segmentation stage attached labels
            if (off_label >= 0) {
                if      (label_type == sensor_msgs::msg::PointField::UINT8)  { uint8_t  v; memcpy(&v, pt + off_label, 1); label = v; }
                else if (label_type == sensor_msgs::msg::PointField::UINT16) { uint16_t v; memcpy(&v, pt + off_label, 2); label = v; }
                else if (label_type == sensor_msgs::msg::PointField::UINT32) { uint32_t v; memcpy(&v, pt + off_label, 4); label = (int)min<uint32_t>(v, 255); }
                else if (label_type == sensor_msgs::msg::PointField::FLOAT32){ float    v; memcpy(&v, pt + off_label, 4); label = isfinite(v) ? (int)v : 6; }
            }

            float wx = px, wy = py, wz = pz;
            if (!identity) {
                wx = R[0] * px + R[1] * py + R[2] * pz + (float)t.x;
                wy = R[3] * px + R[4] * py + R[5] * pz + (float)t.y;
                wz = R[6] * px + R[7] * py + R[8] * pz + (float)t.z;
            }
            float dx = wx - robot_x, dy = wy - robot_y;
            if (dx * dx + dy * dy < min_r2) continue;
            if (insert_lidar_point(wx, wy, wz, label, robot_x, robot_y, frame_ts, ground_z)) inserted++;
        }

        // 3. Classify only what changed (legacy: full 1M-slot rescan that also undid the decay).
        int classified = classify_dirty_cells(ground_z);
        int decayed    = run_temporal_decay(frame_ts);
        int purged     = (purge_every > 0 && frame_ts % (uint32_t)purge_every == 0)
                             ? purge_distant_cells(robot_x, robot_y, (float)purge_margin) : 0;

        float load = get_load_factor();
        if (load > LOAD_WARN)
            RCLCPP_WARN_THROTTLE(node->get_logger(), *node->get_clock(), 2000,
                                 "[DRDO] Hash pool load %.2f > %.2f — lower purge_margin_m", load, LOAD_WARN);
        if (g_insert_fail != reported_fail) {
            RCLCPP_ERROR_THROTTLE(node->get_logger(), *node->get_clock(), 2000,
                                  "[DRDO] %llu points dropped: hash pool saturated",
                                  (unsigned long long)(g_insert_fail - reported_fail));
            reported_fail = g_insert_fail;
        }
        RCLCPP_DEBUG(node->get_logger(), "[DRDO] frame %u: in=%d cls=%d dec=%d purged=%d load=%.3f",
                     frame_ts, inserted, classified, decayed, purged, load);
    };

    // Robot-centred rolling window, rasterised from the pool with max-cost aggregation over each
    // cell's footprint. The legacy publisher sampled one 5 cm cell at the centre of each fixed 1 m
    // output cell (a rock off-centre was invisible) and never followed the robot.
    const int GW = max(1, (int)lround(grid_size_m / grid_res));
    auto publish_grid = [&]() {
        if (!have_pose) return;
        nav_msgs::msg::OccupancyGrid out;
        out.header.stamp    = node->get_clock()->now();
        out.header.frame_id = world_frame;
        out.info.resolution = (float)grid_res;
        out.info.width      = GW;
        out.info.height     = GW;
        double ox = floor((robot_x - grid_size_m / 2.0) / grid_res) * grid_res;
        double oy = floor((robot_y - grid_size_m / 2.0) / grid_res) * grid_res;
        out.info.origin.position.x = ox;
        out.info.origin.position.y = oy;
        out.info.origin.orientation.w = 1.0;
        out.data.assign((size_t)GW * GW, -1);

        for (int i = 0; i < HASH_TABLE_SIZE; i++) {
            const GridCell& c = g_hash_pool[i];
            if (!c.valid) continue;
            int8_t cost;
            if      (c.obstacle_flag == 1)                            cost = 100;
            else if (c.hit_count == 0)                                cost = 0;     // carved free space
            else if (frame_ts > c.last_update_ts && frame_ts - c.last_update_ts > (uint32_t)DECAY_WINDOW &&
                     c.traversability < DECAY_MIN_TRAV)               continue;     // decayed -> unknown
            else if (c.semantic_label == 4 || c.semantic_label == 5)  cost = 100;   // OBSTACLE_HARD / WATER_MUD
            else {
                float conf  = fminf((float)c.hit_count / 10.0f, 1.0f);
                float score = conf > 0.0f ? c.traversability / conf : 0.0f;   // undo confidence weighting
                cost = (int8_t)lround(99.0f * (1.0f - fminf(fmaxf(score, 0.0f), 1.0f)));
            }
            float cs = CELL_RESOLUTIONS[c.level < NUM_LEVELS ? c.level : NUM_LEVELS - 1];
            int x0 = (int)floor((c.ix * cs - ox) / grid_res), x1 = (int)floor(((c.ix + 1) * cs - ox) / grid_res - 1e-4);
            int y0 = (int)floor((c.iy * cs - oy) / grid_res), y1 = (int)floor(((c.iy + 1) * cs - oy) / grid_res - 1e-4);
            if (x1 < 0 || y1 < 0 || x0 >= GW || y0 >= GW) continue;
            x0 = max(x0, 0); y0 = max(y0, 0); x1 = min(x1, GW - 1); y1 = min(y1, GW - 1);
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++) {
                    int8_t& v = out.data[(size_t)y * GW + x];
                    if (cost > v) v = cost;
                }
        }
        grid_pub->publish(out);
    };

    auto cloud_sub = node->create_subscription<sensor_msgs::msg::PointCloud2>(
        "pointcloud", rclcpp::SensorDataQoS(), on_cloud);
    auto timer = node->create_wall_timer(
        chrono::duration_cast<chrono::nanoseconds>(chrono::duration<double>(1.0 / max(publish_hz, 0.1))),
        publish_grid);

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

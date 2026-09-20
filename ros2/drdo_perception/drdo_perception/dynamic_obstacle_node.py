#!/usr/bin/env python3
"""DRDO ID26053 — moving-object node.

Sits between FAST-LIO2 and the grid map:

    /cloud_registered --> [dynamic_obstacle_node] --> /drdo/cloud_static --> grid_map_node --> /map
                                     |
                                     +--> /drdo/dynamic_obstacles  (visualization_msgs/MarkerArray)

Detection logic lives in drdo_lidar_mapping/perception/dynamic.py (ROS-free, unit-tested on the host);
this file only converts messages and TF. Needs the repo's Python package importable
(`pip install -e .` at the repo root, as the Dockerfile does).

Topics (relative, remapped in mapping.launch.py):
  sub  pointcloud        sensor_msgs/PointCloud2   world-registered scan (any frame with TF to world_frame)
  pub  cloud_static      sensor_msgs/PointCloud2   same scan minus points of confirmed moving objects
  pub  moving_objects    visualization_msgs/MarkerArray   box + velocity arrow + label per moving object
"""
import array
import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from geometry_msgs.msg import Point
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformException, TransformListener

try:
    from drdo_lidar_mapping.perception.dynamic import DynamicObstacleDetector, cloud_xyz, filter_cloud_bytes
    from drdo_lidar_mapping.segmentation.taxonomy import OBJ_NAMES, OBJ_NONE, OBJ_RGB
except ImportError as exc:  # pragma: no cover - deployment error path
    raise ImportError("drdo_lidar_mapping is not importable: run `pip install -e .` in the repository root") from exc


def _rotation(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class DynamicObstacleNode(Node):
    def __init__(self):
        super().__init__("drdo_dynamic_obstacle_node")
        p = self.declare_parameter
        self.world_frame = p("world_frame", "map").value
        self.base_frame = p("base_frame", "base_link").value
        self.tf_timeout = Duration(seconds=float(p("tf_timeout_s", 0.05).value))
        self.detector = DynamicObstacleDetector(
            dt=1.0 / float(p("scan_rate_hz", 10.0).value),
            min_height=float(p("min_height_m", 0.3).value),
            max_height=float(p("max_height_m", 2.5).value),
            max_range=float(p("max_range_m", 40.0).value),
            max_footprint=float(p("max_footprint_m", 7.0).value),
            min_speed=float(p("min_speed_mps", 1.0).value),
            confirm_frames=int(p("confirm_frames", 3).value),
            evidence_frames=int(p("evidence_frames", 10).value),
        )
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_pub = self.create_publisher(PointCloud2, "cloud_static", qos_profile_sensor_data)
        self.marker_pub = self.create_publisher(MarkerArray, "moving_objects", 10)
        self.sub = self.create_subscription(PointCloud2, "pointcloud", self.on_cloud, qos_profile_sensor_data)
        self.frames, self.proc_s, self.last_log = 0, 0.0, time.monotonic()
        self.get_logger().info(f"[DRDO] moving-object filter ready (world={self.world_frame}, base={self.base_frame})")

    def on_cloud(self, msg: PointCloud2):
        t0 = time.perf_counter()
        stamp = Time.from_msg(msg.header.stamp)
        try:
            base = self.tf_buffer.lookup_transform(self.world_frame, self.base_frame, stamp, self.tf_timeout)
            cloud_tf = None if msg.header.frame_id == self.world_frame else \
                self.tf_buffer.lookup_transform(self.world_frame, msg.header.frame_id, stamp, self.tf_timeout)
        except TransformException as exc:
            # No pose: pass the scan through unfiltered so the map never starves.
            self.get_logger().warn(f"[DRDO] TF unavailable, scan passed through unfiltered: {exc}",
                                   throttle_duration_sec=2.0)
            self.static_pub.publish(msg)
            return

        offsets = {f.name: f.offset for f in msg.fields if f.datatype == PointField.FLOAT32}
        n = msg.width * msg.height
        if msg.is_bigendian or not all(k in offsets for k in "xyz") or len(msg.data) < n * msg.point_step:
            self.get_logger().error("[DRDO] unsupported PointCloud2 (need little-endian FLOAT32 x/y/z)",
                                    throttle_duration_sec=5.0)
            self.static_pub.publish(msg)
            return

        xyz = cloud_xyz(msg.data, msg.point_step, (offsets["x"], offsets["y"], offsets["z"]), n)
        if cloud_tf is not None:
            tr = cloud_tf.transform
            xyz = xyz @ _rotation(tr.rotation).T.astype(np.float32) + np.array(
                [tr.translation.x, tr.translation.y, tr.translation.z], np.float32)
        sensor_xy = (base.transform.translation.x, base.transform.translation.y)
        ground_z = base.transform.translation.z          # base_link sits on the ground plane
        movers, moving_mask = self.detector.update(xyz, sensor_xy, ground_z)

        out = PointCloud2()
        out.header = msg.header
        out.height, out.fields, out.is_bigendian = 1, msg.fields, False
        out.point_step, out.is_dense = msg.point_step, msg.is_dense
        keep = ~moving_mask
        out.width = int(keep.sum())
        out.row_step = out.point_step * out.width
        out.data = array.array("B", filter_cloud_bytes(msg.data, msg.point_step, n, keep))
        self.static_pub.publish(out)
        self.publish_markers(movers, msg.header.stamp)

        self.frames += 1
        self.proc_s += time.perf_counter() - t0
        if time.monotonic() - self.last_log > 10.0:
            self.get_logger().info(f"[DRDO] moving-object filter: {self.frames} scans, "
                                   f"{1000.0 * self.proc_s / max(self.frames, 1):.1f} ms/scan, "
                                   f"{len(movers)} moving now, {len(self.detector.tracks)} tracks")
            self.frames, self.proc_s, self.last_log = 0, 0.0, time.monotonic()

    def publish_markers(self, movers, stamp):
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.world_frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)
        for ob in movers:
            height = max(ob.z_max - ob.z_min, 0.3)
            # Colour and name come from the geometric classifier (perception/classify.py) via
            # taxonomy.OBJ_RGB / OBJ_NAMES — the one shared object vocabulary, so a walking person
            # and a moving truck are no longer both drawn as the same magenta box.
            cls = getattr(ob, "obj_class", OBJ_NONE)
            r, g, b = OBJ_RGB.get(cls, OBJ_RGB[OBJ_NONE])
            r, g, b = r / 255.0, g / 255.0, b / 255.0
            box = Marker()
            box.header.frame_id, box.header.stamp = self.world_frame, stamp
            box.ns, box.id, box.type, box.action = "moving_box", ob.track_id, Marker.CUBE, Marker.ADD
            box.pose.position.x, box.pose.position.y = ob.x, ob.y
            box.pose.position.z = ob.z_min + height / 2.0
            box.pose.orientation.w = 1.0
            box.scale.x, box.scale.y, box.scale.z = max(ob.length, 0.3), max(ob.width, 0.3), height
            box.color.r, box.color.g, box.color.b, box.color.a = r, g, b, 0.6
            arrow = Marker()
            arrow.header = box.header
            arrow.ns, arrow.id, arrow.type, arrow.action = "moving_velocity", ob.track_id, Marker.ARROW, Marker.ADD
            arrow.pose.orientation.w = 1.0
            arrow.points = [Point(x=ob.x, y=ob.y, z=ob.z_max + 0.2),
                            Point(x=ob.x + ob.vx, y=ob.y + ob.vy, z=ob.z_max + 0.2)]   # 1 s ahead
            arrow.scale.x, arrow.scale.y, arrow.scale.z = 0.15, 0.3, 0.3
            arrow.color.r, arrow.color.g, arrow.color.b, arrow.color.a = r, g, b, 1.0
            text = Marker()
            text.header = box.header
            text.ns, text.id, text.type, text.action = "moving_label", ob.track_id, Marker.TEXT_VIEW_FACING, Marker.ADD
            text.pose.position.x, text.pose.position.y, text.pose.position.z = ob.x, ob.y, ob.z_max + 0.8
            text.pose.orientation.w = 1.0
            text.scale.z = 0.6
            text.color.r = text.color.g = text.color.b = text.color.a = 1.0
            text.text = f"#{ob.track_id} {OBJ_NAMES[cls]} {ob.speed:.1f} m/s"
            arr.markers.extend([box, arrow, text])
        self.marker_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = DynamicObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""DRDO ID26053 — semantic segmentation node.

Sits between the moving-object filter and the grid map:

    /drdo/cloud_static --> [segmentation_node] --> /drdo/cloud_labeled --> grid_map_node --> /map
                                     |
                                     +--> /drdo/semantic_objects  (visualization_msgs/MarkerArray)

This node closes the one gap that has been open since the project started. `grid_map_node.cpp`
has always scanned `msg->fields` for a `label` field and passed it to `insert_lidar_point`, but
nothing in the graph ever wrote that field — the Ouster driver does not emit one — so `off_label`
was always -1 and **every point in every run entered the map as UNKNOWN(6)**, traversability
0.40. The semantic half of the map was inert. This node is the producer.

Inference lives in `drdo_lidar_mapping/inference/range_segmenter.py` (ROS-free, host-testable);
this file only converts messages and TF, exactly as `dynamic_obstacle_node.py` does.

Topics (relative, remapped in mapping.launch.py):
  sub  pointcloud        sensor_msgs/PointCloud2          world-registered scan
  pub  cloud_labeled     sensor_msgs/PointCloud2          same points + label / obj_class / conf
  pub  semantic_objects  visualization_msgs/MarkerArray   box + text per named object instance

── The frame bug this node is built around ──────────────────────────────────────────────────
The cloud arriving here is WORLD-registered (FAST-LIO `/cloud_registered`, then
`/drdo/cloud_static`), because that is what the grid engine needs. But a spherical range-image
projection is only defined about the sensor origin, and the network was trained on raw
sensor-frame RELLIS scans. Projecting a world-frame cloud gives a range image smeared by the
vehicle's roll and pitch and a silent domain shift away from training — no exception, no warning,
just quietly degraded labels that look plausible on screen.

So the order is mandatory and is enforced below:
  1. look up TF <cloud frame> -> `sensor_frame` (os_sensor),
  2. transform the points INTO the sensor frame,
  3. run RangeSegmenter.segment() there,
  4. attach the labels back to the ORIGINAL, untouched world-frame points and republish.
The transformed copy is inference input only and is never published.

── Status ───────────────────────────────────────────────────────────────────────────────────
NOT EXECUTED. ROS 2 is not installed on the macOS development host (see CLAUDE.md), so this
file has been byte-compiled and reviewed only — it has never been spun, and no `ros2 topic echo`
has ever confirmed the field layout below on a live graph. The parts that *can* run on the host
are covered elsewhere: RangeSegmenter by tests/python/test_segmentation.py, and the
labels -> C++ engine path end to end by scripts/kitti_replay.py.
"""
import array
import time

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformException, TransformListener

try:
    from drdo_lidar_mapping.inference.range_segmenter import RangeSegmenter
    from drdo_lidar_mapping.perception.cluster import cluster_hard_obstacles
    from drdo_lidar_mapping.perception.dynamic import cloud_xyz
    from drdo_lidar_mapping.segmentation.taxonomy import (
        ADL1_UNKNOWN, OBJ_NAMES, OBJ_NONE,
    )
except ImportError as exc:  # pragma: no cover - deployment error path
    raise ImportError("drdo_lidar_mapping is not importable: run `pip install -e .` in the repository root") from exc

# Per-object-class marker colours (r, g, b). The existing dynamic node paints every mover the
# same magenta and writes only an id and a speed, so an operator cannot tell a pedestrian from a
# parked truck. Naming and colouring the instance is the whole point of the obj_class channel —
# the grid engine never sees it (see taxonomy.FINE_TO_OBJ_LUT).
OBJ_COLOR = {
    1: (1.00, 0.25, 0.25),   # person   — red
    2: (0.25, 0.55, 1.00),   # vehicle  — blue
    3: (1.00, 0.85, 0.10),   # pole     — amber
}

# Output point layout. Built explicitly rather than copied from the input, because the input has
# no label field at all and the consumer (grid_map_node.cpp, lines 91-135) looks fields up BY
# NAME and ignores anything it does not recognise — so intensity/obj_class/conf ride along free.
# label is UINT8: it is an ADL-1 id in 0..7 and the C++ side already accepts UINT8 directly.
# 18..19 are padding so that `conf` lands on a 4-byte boundary and point_step is a multiple of 4;
# some PointCloud2 readers (and numpy structured views) assume that even though the message spec
# does not require it.
OUT_DTYPE = np.dtype({
    "names":   ["x", "y", "z", "intensity", "label", "obj_class", "conf"],
    "formats": ["<f4", "<f4", "<f4", "<f4", "u1", "u1", "<f4"],
    "offsets": [0, 4, 8, 12, 16, 17, 20],
    "itemsize": 24,
})
# Keyed on (kind, itemsize) rather than dtype.str: numpy spells uint8 "|u1", not "u1", and the
# byte-order character differs per field, so a string key silently KeyErrors on the label field.
_FIELD_TYPE = {("f", 4): PointField.FLOAT32, ("u", 1): PointField.UINT8}


def _out_fields():
    out = []
    for n in OUT_DTYPE.names:
        dt, off = OUT_DTYPE.fields[n][0], OUT_DTYPE.fields[n][1]
        out.append(PointField(name=n, offset=off, count=1,
                              datatype=_FIELD_TYPE[(dt.kind, dt.itemsize)]))
    return out


def _rotation(q):
    # Same 4-line quaternion -> R as dynamic_obstacle_node._rotation. Deliberately duplicated
    # rather than imported: importing it would pull that node's module (and its detector) into
    # this process for three lines of arithmetic.
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _apply(tf, xyz):
    tr = tf.transform
    return xyz @ _rotation(tr.rotation).T.astype(np.float32) + np.array(
        [tr.translation.x, tr.translation.y, tr.translation.z], np.float32)


class SegmentationNode(Node):
    def __init__(self):
        super().__init__("drdo_segmentation_node")
        p = self.declare_parameter
        self.world_frame = p("world_frame", "map").value
        self.base_frame = p("base_frame", "base_link").value
        self.sensor_frame = p("sensor_frame", "os_sensor").value
        self.tf_timeout = Duration(seconds=float(p("tf_timeout_s", 0.05).value))
        model_path = str(p("model_path", "").value)
        min_conf = float(p("min_conf", 0.0).value)
        self.passthrough_on_error = bool(p("passthrough_on_error", True).value)
        self.publish_objects = bool(p("publish_objects", True).value)
        self.cluster_eps = float(p("cluster_eps_m", 0.8).value)
        self.cluster_min_samples = int(p("cluster_min_samples", 5).value)
        # cluster.py falls back to a pure-numpy DBSCAN that materialises an N x N distance
        # matrix when sklearn is missing. 20k object points would be a 3.2 GB allocation inside
        # a 10 Hz callback, so cap the input and subsample instead of stalling the graph.
        self.max_cluster_points = int(p("max_cluster_points", 20000).value)

        # Model load failure is a degraded mode, not a crash: the map must keep receiving scans
        # even with no weights on the vehicle. RangeSegmenter is strict by design (it raises
        # rather than substituting a heuristic), so this is the only place that decision is
        # softened — and only into pass-through, never into a fake label.
        self.segmenter = None
        try:
            self.segmenter = RangeSegmenter(model_path, min_conf=min_conf)
            self.get_logger().info(
                f"[DRDO] segmentation ready: {model_path} ({self.segmenter.backend}, "
                f"{self.segmenter.H}x{self.segmenter.W}, fov {self.segmenter.fov_down:+.1f}.."
                f"{self.segmenter.fov_up:+.1f} deg)")
        except Exception as exc:
            if not self.passthrough_on_error:
                raise
            self.get_logger().error(
                f"[DRDO] segmentation model unavailable ({exc}); scans will be republished "
                f"UNLABELLED, so every point enters the map as UNKNOWN(6). Set "
                f"passthrough_on_error:=false to make this fatal instead.")

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cloud_pub = self.create_publisher(PointCloud2, "cloud_labeled", qos_profile_sensor_data)
        self.marker_pub = self.create_publisher(MarkerArray, "semantic_objects", 10)
        self.sub = self.create_subscription(PointCloud2, "pointcloud", self.on_cloud, qos_profile_sensor_data)
        self.frames, self.proc_s, self.objects, self.last_log = 0, 0.0, 0, time.monotonic()
        self.get_logger().info(
            f"[DRDO] segmentation node up (world={self.world_frame}, sensor={self.sensor_frame})")

    # ── main callback ────────────────────────────────────────────────────────────────────────
    def _passthrough(self, msg):
        """Republish the input untouched. The grid node then defaults label to UNKNOWN(6), which
        is exactly the pre-segmentation behaviour — degraded, never starved."""
        self.cloud_pub.publish(msg)

    def on_cloud(self, msg: PointCloud2):
        t0 = time.perf_counter()
        if self.segmenter is None:
            self._passthrough(msg)
            return

        offsets = {f.name: f.offset for f in msg.fields if f.datatype == PointField.FLOAT32}
        n = msg.width * msg.height
        if msg.is_bigendian or not all(k in offsets for k in "xyz") or len(msg.data) < n * msg.point_step:
            self.get_logger().error("[DRDO] unsupported PointCloud2 (need little-endian FLOAT32 x/y/z)",
                                    throttle_duration_sec=5.0)
            self._passthrough(msg)
            return
        if n == 0:
            self._passthrough(msg)
            return

        stamp = Time.from_msg(msg.header.stamp)
        try:
            # Step 1 of the frame rule in the module docstring. This lookup is the reason the
            # node exists in this shape; without it the projection below is measured about the
            # map origin instead of the sensor.
            to_sensor = self.tf_buffer.lookup_transform(
                self.sensor_frame, msg.header.frame_id, stamp, self.tf_timeout)
            to_world = None if msg.header.frame_id == self.world_frame else \
                self.tf_buffer.lookup_transform(self.world_frame, msg.header.frame_id, stamp, self.tf_timeout)
        except TransformException as exc:
            # No sensor pose means no valid projection. Labelling anyway would put confident
            # wrong classes into the map, which is worse than no labels at all.
            self.get_logger().warn(
                f"[DRDO] TF {msg.header.frame_id}->{self.sensor_frame} unavailable, scan "
                f"republished unlabelled: {exc}", throttle_duration_sec=2.0)
            self._passthrough(msg)
            return

        # cloud_xyz iterates whatever offsets tuple it is handed, so a 4-tuple yields (N,4).
        # Intensity is a trained input channel; if the driver does not publish one, zeros are
        # the honest substitute (the normalisation in RangeSegmenter mean-centres it anyway).
        has_i = "intensity" in offsets
        cols = (offsets["x"], offsets["y"], offsets["z"],
                offsets["intensity"] if has_i else offsets["x"])
        buf = cloud_xyz(msg.data, msg.point_step, cols, n)
        if not has_i:
            buf[:, 3] = 0.0
            self.get_logger().warn("[DRDO] cloud has no intensity field; feeding zeros to the "
                                   "network (it trained with intensity)", throttle_duration_sec=30.0)

        # Steps 2-3: inference happens on a sensor-frame COPY. xyz_world below is never
        # overwritten, so what we publish is bit-identical geometry to what we received.
        xyz_world = buf[:, :3]
        sensor = np.empty_like(buf)
        sensor[:, :3] = _apply(to_sensor, xyz_world)
        sensor[:, 3] = buf[:, 3]

        try:
            adl1, obj, conf = self.segmenter.segment(sensor)
        except Exception as exc:
            self.get_logger().error(f"[DRDO] segmentation failed, scan republished unlabelled: {exc}",
                                    throttle_duration_sec=5.0)
            self._passthrough(msg)
            return

        # Step 4: labels back onto the original world-frame points.
        self.cloud_pub.publish(self._build_cloud(msg, buf, adl1, obj, conf))

        n_obj = 0
        if self.publish_objects:
            markers_xyz = xyz_world if to_world is None else _apply(to_world, xyz_world)
            n_obj = self.publish_markers(markers_xyz, obj, msg.header.stamp)

        self.frames += 1
        self.objects += n_obj
        self.proc_s += time.perf_counter() - t0
        if time.monotonic() - self.last_log > 10.0:
            known = int((adl1 != ADL1_UNKNOWN).sum())
            self.get_logger().info(
                f"[DRDO] segmentation: {self.frames} scans, "
                f"{1000.0 * self.proc_s / max(self.frames, 1):.1f} ms/scan, "
                f"{100.0 * known / max(len(adl1), 1):.0f}% of points labelled (last scan), "
                f"{self.objects} object instances")
            self.frames, self.proc_s, self.objects, self.last_log = 0, 0.0, 0, time.monotonic()

    # ── message building ─────────────────────────────────────────────────────────────────────
    def _build_cloud(self, msg: PointCloud2, buf: np.ndarray,
                     adl1: np.ndarray, obj: np.ndarray, conf: np.ndarray) -> PointCloud2:
        rec = np.zeros(len(buf), dtype=OUT_DTYPE)
        rec["x"], rec["y"], rec["z"] = buf[:, 0], buf[:, 1], buf[:, 2]
        rec["intensity"] = buf[:, 3]
        rec["label"] = adl1
        rec["obj_class"] = obj
        rec["conf"] = conf

        out = PointCloud2()
        out.header = msg.header                 # same frame, same stamp: still world-registered
        out.height, out.width = 1, len(rec)
        out.fields = _out_fields()
        out.is_bigendian = False
        out.point_step = OUT_DTYPE.itemsize
        out.row_step = out.point_step * out.width
        out.is_dense = msg.is_dense
        out.data = array.array("B", rec.tobytes())
        return out

    def publish_markers(self, xyz: np.ndarray, obj: np.ndarray, stamp) -> int:
        arr = MarkerArray()
        clear = Marker()
        clear.header.frame_id = self.world_frame
        clear.action = Marker.DELETEALL
        arr.markers.append(clear)

        named = obj != OBJ_NONE
        n_named = int(named.sum())
        if n_named > self.max_cluster_points:
            keep = np.flatnonzero(named)[:: max(1, n_named // self.max_cluster_points)]
            mask = np.zeros(len(obj), bool)
            mask[keep] = True
            obj = np.where(mask, obj, OBJ_NONE)

        mid, total = 0, 0
        for cls, color in OBJ_COLOR.items():
            if not np.any(obj == cls):
                continue
            # Reuse of perception/cluster.py, which until now was dead code exercised only by
            # test_perception.py: same DBSCAN, just keyed on the object channel instead of ADL-1.
            for box in cluster_hard_obstacles(xyz, obj, eps=self.cluster_eps,
                                              min_samples=self.cluster_min_samples,
                                              target_class=cls):
                cx, cy, _ = (float(v) for v in box.centroid)
                z_lo, z_hi = float(box.min_bound[2]), float(box.max_bound[2])
                height = max(z_hi - z_lo, 0.3)
                m = Marker()
                m.header.frame_id, m.header.stamp = self.world_frame, stamp
                m.ns, m.id, m.type, m.action = "semantic_box", mid, Marker.CUBE, Marker.ADD
                m.pose.position.x, m.pose.position.y = cx, cy
                m.pose.position.z = z_lo + height / 2.0
                m.pose.orientation.w = 1.0
                m.scale.x = max(float(box.extent[0]), 0.3)
                m.scale.y = max(float(box.extent[1]), 0.3)
                m.scale.z = height
                m.color.r, m.color.g, m.color.b = color
                m.color.a = 0.55

                t = Marker()
                t.header = m.header
                t.ns, t.id, t.type, t.action = "semantic_label", mid, Marker.TEXT_VIEW_FACING, Marker.ADD
                t.pose.position.x, t.pose.position.y, t.pose.position.z = cx, cy, z_hi + 0.6
                t.pose.orientation.w = 1.0
                t.scale.z = 0.6
                t.color.r, t.color.g, t.color.b = color
                t.color.a = 1.0
                t.text = f"{OBJ_NAMES[cls]} ({box.num_points} pts)"
                arr.markers.extend([m, t])
                mid += 1
                total += 1
        self.marker_pub.publish(arr)
        return total


def main(args=None):
    rclpy.init(args=args)
    node = SegmentationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

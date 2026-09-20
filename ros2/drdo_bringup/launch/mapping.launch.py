#!/usr/bin/env python3
"""
DRDO ID26053 — Complete Autonomous Mapping Launch Pipeline
Brings up:
  1. FAST-LIO2 (IMU-LiDAR Odometry & Mapping for Ouster OS1-64)
  2. DRDO moving-object filter (filter_moving_objects:=true, default): confirms moving objects,
     publishes them on /drdo/dynamic_obstacles and removes their points before mapping
  3. (optional, enable_semantics:=true) DRDO semantic segmentation node: attaches a per-point
     ADL-1 `label` field, the field grid_map_node has always read and nothing has ever written
  4. DRDO Adaptive Variable-Resolution 2.5D Grid Map Node (/map costmap + /drdo/map_image colour view)
  5. Static transforms bridging FAST-LIO frames to the REP-105 tree used by Nav2
  6. (optional, enable_nvblox:=true) NVIDIA nvblox GPU TSDF / ESDF

Data flow (default): /cloud_registered -> dynamic_obstacle_node -> /drdo/cloud_static -> grid_map_node -> /map
With enable_semantics:=true a stage is inserted:  ... -> segmentation_node -> /drdo/cloud_labeled -> grid_map_node

enable_semantics defaults to FALSE on purpose. No trained range-segmentation checkpoint exists in
this repo yet (models/* are placeholders that score at chance — see CLAUDE.md), so the default
launch is byte-for-byte the pipeline that works today: zero behaviour change until weights ship.

TF tree:  map -> camera_init (FAST-LIO world) -> body (FAST-LIO IMU, dynamic) -> base_link -> os_sensor
          map -> odom (identity: FAST-LIO has no loop closure, so map and odom coincide)
"""

import os

# Attempt ROS 2 Launch imports
try:
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.conditions import IfCondition
    from launch.substitutions import LaunchConfiguration, PythonExpression
    from launch_ros.actions import Node
    ROS2_LAUNCH_AVAILABLE = True
except ImportError:
    ROS2_LAUNCH_AVAILABLE = False

LIDAR_MOUNT_HEIGHT_M = "1.2"   # base_link (ground contact) -> os_sensor

# Default checkpoint location, resolved against the repo root (this file lives in
# ros2/drdo_bringup/launch/). Nothing is shipped there yet; segmentation_node degrades to
# republishing the scan unlabelled when the file is absent, so a wrong path costs labels, not the map.
DEFAULT_MODEL_PATH = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "models", "range_seg.pt"))


def _flags(*want):
    """IfCondition over a conjunction of launch booleans, e.g. _flags(("a", True), ("b", False)).

    launch offers no boolean algebra over LaunchConfiguration, and the grid node's input topic is
    a function of TWO flags (filter_moving_objects x enable_semantics). Getting that routing
    wrong is silent — the node simply never receives a scan and /map stays empty forever — so the
    combinations are spelled out below rather than inferred. Truthiness matches IfCondition's own
    accepted spellings so `enable_semantics:=1` behaves like `:=true`.
    """
    expr = []
    for i, (name, truthy) in enumerate(want):
        if i:
            expr.append(" and ")
        expr += ["'", LaunchConfiguration(name),
                 "'.lower() " + ("in" if truthy else "not in") + " ('true','1','yes','on')"]
    return IfCondition(PythonExpression(expr))


def generate_launch_description():
    """Builds and returns the ROS 2 LaunchDescription for the DRDO Mapping stack."""
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fast_lio_cfg = os.path.join(pkg_dir, "config", "fast_lio_ouster64.yaml")
    nvblox_cfg = os.path.join(pkg_dir, "config", "nvblox_params.yaml")

    points_topic_arg = DeclareLaunchArgument(
        "points_topic", default_value="/ouster/points",
        description="Input raw LiDAR pointcloud topic (Ouster OS1-64)")
    imu_topic_arg = DeclareLaunchArgument(
        "imu_topic", default_value="/ouster/imu", description="Input raw IMU topic")
    resolution_arg = DeclareLaunchArgument(
        "map_resolution", default_value="0.20",
        description="Published OccupancyGrid resolution (m). The internal map stays foveated 5 cm..50 cm")
    moving_arg = DeclareLaunchArgument(
        "filter_moving_objects", default_value="true",
        description="Detect moving objects and keep them out of the static map (drdo_perception)")
    color_arg = DeclareLaunchArgument(
        "map_color_mode", default_value="terrain", description="/drdo/map_image colouring: terrain | elevation")
    nvblox_arg = DeclareLaunchArgument(
        "enable_nvblox", default_value="false",
        description="nvblox is unverified with an unorganised world-frame cloud and has no consumer yet")
    semantics_arg = DeclareLaunchArgument(
        "enable_semantics", default_value="false",
        description="Insert segmentation_node so points carry an ADL-1 label; off until a trained "
                    "checkpoint exists (models/* are placeholders scoring at chance)")
    model_path_arg = DeclareLaunchArgument(
        "model_path", default_value=DEFAULT_MODEL_PATH,
        description="Range-segmentation checkpoint (.pt or .onnx) for segmentation_node; its "
                    "<stem>_meta.json must sit beside it or the projection geometry is guessed")

    # 1. FAST-LIO2 Node — it reads topics from parameters, so remapping /ouster/points had no effect.
    fast_lio_node = Node(
        package="fast_lio",
        executable="fastlio_mapping",
        name="fast_lio_mapping",
        output="screen",
        parameters=[fast_lio_cfg, {
            "common.lid_topic": LaunchConfiguration("points_topic"),
            "common.imu_topic": LaunchConfiguration("imu_topic"),
        }],
    )

    # 2. NVIDIA nvblox GPU 3D ESDF Node (opt-in)
    nvblox_node = Node(
        package="nvblox_ros",
        executable="nvblox_node",
        name="nvblox_node",
        output="screen",
        parameters=[nvblox_cfg],
        remappings=[
            ("pointcloud", "/cloud_registered"),
            ("esdf_slice", "/nvblox_node/esdf_slice"),
            ("mesh", "/nvblox_node/mesh"),
        ],
        condition=IfCondition(LaunchConfiguration("enable_nvblox")),
    )

    # 3. DRDO grid map. Parameter and topic names match what grid_map_node.cpp declares/uses;
    #    the previous remaps and parameters matched nothing, so the node never received data.
    grid_params = [{
        "world_frame": "map",
        "base_frame": "base_link",
        "grid_resolution": LaunchConfiguration("map_resolution"),
        "grid_size_m": 100.0,
        "publish_rate_hz": 5.0,
        "min_range": 0.3,
        "purge_every_n_frames": 10,
        "purge_margin_m": 20.0,
        "publish_image": True,
        "color_mode": LaunchConfiguration("map_color_mode"),
    }]
    grid_out = [("occupancy_grid", "/map"), ("map_image", "/drdo/map_image")]
    # Exactly one of these three runs. The input topic is the only difference between them.
    grid_map_node = Node(
        package="drdo_grid_map", executable="grid_map_node", name="drdo_grid_map_node", output="screen",
        parameters=grid_params, remappings=[("pointcloud", "/drdo/cloud_static")] + grid_out,
        condition=_flags(("enable_semantics", False), ("filter_moving_objects", True)),
    )
    grid_map_node_unfiltered = Node(
        package="drdo_grid_map", executable="grid_map_node", name="drdo_grid_map_node", output="screen",
        parameters=grid_params, remappings=[("pointcloud", "/cloud_registered")] + grid_out,
        condition=_flags(("enable_semantics", False), ("filter_moving_objects", False)),
    )
    grid_map_node_labeled = Node(
        package="drdo_grid_map", executable="grid_map_node", name="drdo_grid_map_node", output="screen",
        parameters=grid_params, remappings=[("pointcloud", "/drdo/cloud_labeled")] + grid_out,
        condition=IfCondition(LaunchConfiguration("enable_semantics")),
    )

    # 3b. Moving-object filter. If TF is missing it passes scans through unfiltered, so the map never starves.
    dynamic_obstacle_node = Node(
        package="drdo_perception", executable="dynamic_obstacle_node", name="drdo_dynamic_obstacle_node",
        output="screen",
        parameters=[{
            "world_frame": "map",
            "base_frame": "base_link",
            "scan_rate_hz": 10.0,
            "max_range_m": 40.0,
            "min_speed_mps": 1.0,
        }],
        remappings=[
            ("pointcloud", "/cloud_registered"),
            ("cloud_static", "/drdo/cloud_static"),
            ("moving_objects", "/drdo/dynamic_obstacles"),
        ],
        condition=IfCondition(LaunchConfiguration("filter_moving_objects")),
    )

    # 3c. Semantic segmentation (opt-in). It must transform the world-registered cloud into
    #     sensor_frame before projecting — a spherical range image is only valid about the sensor
    #     origin, so skipping that smears the projection by vehicle roll/pitch with no error
    #     raised. sensor_frame must therefore name the frame the base_link->os_sensor static TF
    #     below publishes. If TF or the checkpoint is missing the node republishes the scan
    #     unlabelled, i.e. it falls back to exactly the enable_semantics:=false behaviour.
    seg_params = [{
        "world_frame": "map",
        "base_frame": "base_link",
        "sensor_frame": "os_sensor",
        "model_path": LaunchConfiguration("model_path"),
        "tf_timeout_s": 0.05,
        "min_conf": 0.0,
        "passthrough_on_error": True,
    }]
    seg_out = [("cloud_labeled", "/drdo/cloud_labeled"), ("semantic_objects", "/drdo/semantic_objects")]
    segmentation_node = Node(
        package="drdo_perception", executable="segmentation_node", name="drdo_segmentation_node",
        output="screen", parameters=seg_params,
        remappings=[("pointcloud", "/drdo/cloud_static")] + seg_out,
        condition=_flags(("enable_semantics", True), ("filter_moving_objects", True)),
    )
    segmentation_node_unfiltered = Node(
        package="drdo_perception", executable="segmentation_node", name="drdo_segmentation_node",
        output="screen", parameters=seg_params,
        remappings=[("pointcloud", "/cloud_registered")] + seg_out,
        condition=_flags(("enable_semantics", True), ("filter_moving_objects", False)),
    )

    # 4. Static TF transforms. The previous base_link -> os_imu publisher gave os_imu a second
    #    parent (the Ouster driver already publishes os_sensor -> os_imu), and nothing published
    #    map or base_link relative to FAST-LIO's camera_init/body frames.
    tf_map_to_camera_init = Node(
        package="tf2_ros", executable="static_transform_publisher", name="map_to_camera_init",
        arguments=["0", "0", "0", "0", "0", "0", "map", "camera_init"])
    tf_map_to_odom = Node(
        package="tf2_ros", executable="static_transform_publisher", name="map_to_odom",
        arguments=["0", "0", "0", "0", "0", "0", "map", "odom"])
    tf_body_to_base = Node(
        package="tf2_ros", executable="static_transform_publisher", name="body_to_base_link",
        arguments=["0", "0", "-" + LIDAR_MOUNT_HEIGHT_M, "0", "0", "0", "body", "base_link"])
    tf_base_to_lidar = Node(
        package="tf2_ros", executable="static_transform_publisher", name="base_to_lidar_broadcaster",
        arguments=["0", "0", LIDAR_MOUNT_HEIGHT_M, "0", "0", "0", "base_link", "os_sensor"])

    ld = LaunchDescription()
    for action in (points_topic_arg, imu_topic_arg, resolution_arg, nvblox_arg, moving_arg, color_arg,
                   semantics_arg, model_path_arg,
                   tf_map_to_camera_init, tf_map_to_odom, tf_body_to_base, tf_base_to_lidar,
                   fast_lio_node, nvblox_node, dynamic_obstacle_node,
                   segmentation_node, segmentation_node_unfiltered,
                   grid_map_node, grid_map_node_unfiltered, grid_map_node_labeled):
        ld.add_action(action)
    return ld


def validate_launch_configuration():
    """Validates launch configuration and parameters across packages."""
    print("Validating ROS 2 Launch & SLAM Configurations...")
    pkg_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fast_lio_cfg = os.path.join(pkg_dir, "config", "fast_lio_ouster64.yaml")
    nvblox_cfg = os.path.join(pkg_dir, "config", "nvblox_params.yaml")

    assert os.path.exists(fast_lio_cfg), f"Missing {fast_lio_cfg}"
    assert os.path.exists(nvblox_cfg), f"Missing {nvblox_cfg}"
    perception_dir = os.path.join(pkg_dir, "..", "drdo_perception", "drdo_perception")
    perception_node = os.path.join(perception_dir, "dynamic_obstacle_node.py")
    assert os.path.exists(perception_node), f"Missing {perception_node}"
    # enable_semantics:=true launches this by name; a missing file is a launch-time crash on the
    # vehicle rather than anything the Python launch API would catch here.
    segmentation_node = os.path.join(perception_dir, "segmentation_node.py")
    assert os.path.exists(segmentation_node), f"Missing {segmentation_node}"
    assert os.path.exists(os.path.join(pkg_dir, "..", "drdo_perception", "setup.py")), \
        "drdo_perception/setup.py missing"
    with open(os.path.join(pkg_dir, "..", "drdo_perception", "setup.py")) as f:
        entry_points = f.read()
    # The Node(executable=...) names above are console-script names, not file names: without the
    # entry point colcon installs nothing to run and the launch fails with "executable not found".
    for exe in ("dynamic_obstacle_node", "segmentation_node"):
        assert f"{exe} = drdo_perception.{exe}:main" in entry_points, \
            f"drdo_perception/setup.py has no console_script for {exe}"
    print(f"  [CONF] FAST-LIO2 Ouster config found: {fast_lio_cfg}")
    print(f"  [CONF] nvblox 3D ESDF config found: {nvblox_cfg}")
    print(f"  [CONF] drdo_perception nodes found: dynamic_obstacle_node.py, segmentation_node.py")

    try:
        import yaml
        with open(fast_lio_cfg, "r") as f:
            fl = yaml.safe_load(f)
        params = fl["/**"]["ros__parameters"]
        assert params["common"]["lid_topic"] == "/ouster/points"
        assert params["preprocess"]["lidar_type"] == 3, "FAST-LIO Ouster driver type is 3 (OUST64)"
    except ImportError:
        with open(fast_lio_cfg, "r") as f:
            fl_content = f.read()
        assert "ros__parameters" in fl_content and "lidar_type: 3" in fl_content

    with open(nvblox_cfg, "r") as f:
        nv_content = f.read()
    assert "voxel_size:" in nv_content

    if ROS2_LAUNCH_AVAILABLE:
        ld = generate_launch_description()
        assert len(ld.entities) >= 8, "LaunchDescription missing expected actions"
        print(f"  [ROS2] LaunchDescription validated natively with {len(ld.entities)} entities.")
    else:
        print("  [INFO] ROS 2 python packages not installed on host. File-level validation only.")

    # Routing check, run either way. launch is not installed on the dev host, so the conditions
    # above cannot be evaluated here; the topic table is checked by inspection instead. Exactly
    # one grid node must be live per (enable_semantics, filter_moving_objects) pair — two would
    # double-insert every scan into the single global hash pool, none leaves /map empty forever.
    with open(os.path.abspath(__file__)) as f:
        src = f.read()
    for topic in ("/drdo/cloud_static", "/cloud_registered", "/drdo/cloud_labeled"):
        assert src.count('("pointcloud", "%s")' % topic) >= 1, f"no node consumes {topic}"
    # Needles are assembled from two halves on purpose: this function reads its OWN source, so
    # spelling either one out anywhere in this file (a comment included) makes it match itself
    # and inflates the count by one.
    grid_pkg, perc_pkg = 'package=' + '"drdo_grid_map"', 'package=' + '"drdo_perception"'
    assert src.count(grid_pkg) == 3, \
        f"expected exactly 3 grid_map_node variants (semantics off/filtered, off/unfiltered, on), " \
        f"found {src.count(grid_pkg)}"
    assert src.count(perc_pkg) == 3, \
        f"expected dynamic_obstacle_node + 2 segmentation_node variants, found {src.count(perc_pkg)}"
    seg_arg = src[src.index('"enable_semantics", default_value='):]
    assert seg_arg.startswith('"enable_semantics", default_value="false"'), \
        "enable_semantics must default to false: the pipeline that works today stays the default"

    print("  [NODES CONFIGURED]:")
    print("    1. fast_lio::fastlio_mapping -> /cloud_registered (camera_init), TF camera_init->body")
    print("    2. drdo_perception::dynamic_obstacle_node -> /drdo/cloud_static, /drdo/dynamic_obstacles")
    print("    3. drdo_grid_map::grid_map_node -> /drdo/cloud_static in, /map (transient_local) + /drdo/map_image out")
    print("    4. tf2_ros static: map->camera_init, map->odom, body->base_link, base_link->os_sensor")
    print("    5. nvblox_ros::nvblox_node   -> only with enable_nvblox:=true")
    print("    6. drdo_perception::segmentation_node -> only with enable_semantics:=true;")
    print("       /drdo/cloud_static in, /drdo/cloud_labeled (+ ADL-1 label field) + /drdo/semantic_objects out,")
    print("       and the grid node then consumes /drdo/cloud_labeled instead of /drdo/cloud_static")
    print("[LAUNCH CONFIGURATION VALID]")


if __name__ == "__main__":
    validate_launch_configuration()

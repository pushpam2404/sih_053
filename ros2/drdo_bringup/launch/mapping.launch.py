#!/usr/bin/env python3
"""
DRDO ID26053 — Complete Autonomous Mapping Launch Pipeline
Brings up:
  1. FAST-LIO2 (IMU-LiDAR Odometry & Mapping for Ouster OS1-64)
  2. DRDO moving-object filter (filter_moving_objects:=true, default): confirms moving objects,
     publishes them on /drdo/dynamic_obstacles and removes their points before mapping
  3. DRDO Adaptive Variable-Resolution 2.5D Grid Map Node (/map costmap + /drdo/map_image colour view)
  4. Static transforms bridging FAST-LIO frames to the REP-105 tree used by Nav2
  5. (optional, enable_nvblox:=true) NVIDIA nvblox GPU TSDF / ESDF

Data flow: /cloud_registered -> dynamic_obstacle_node -> /drdo/cloud_static -> grid_map_node -> /map

TF tree:  map -> camera_init (FAST-LIO world) -> body (FAST-LIO IMU, dynamic) -> base_link -> os_sensor
          map -> odom (identity: FAST-LIO has no loop closure, so map and odom coincide)
"""

import os

# Attempt ROS 2 Launch imports
try:
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument
    from launch.conditions import IfCondition, UnlessCondition
    from launch.substitutions import LaunchConfiguration
    from launch_ros.actions import Node
    ROS2_LAUNCH_AVAILABLE = True
except ImportError:
    ROS2_LAUNCH_AVAILABLE = False

LIDAR_MOUNT_HEIGHT_M = "1.2"   # base_link (ground contact) -> os_sensor


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
    grid_map_node = Node(
        package="drdo_grid_map", executable="grid_map_node", name="drdo_grid_map_node", output="screen",
        parameters=grid_params, remappings=[("pointcloud", "/drdo/cloud_static")] + grid_out,
        condition=IfCondition(LaunchConfiguration("filter_moving_objects")),
    )
    grid_map_node_unfiltered = Node(
        package="drdo_grid_map", executable="grid_map_node", name="drdo_grid_map_node", output="screen",
        parameters=grid_params, remappings=[("pointcloud", "/cloud_registered")] + grid_out,
        condition=UnlessCondition(LaunchConfiguration("filter_moving_objects")),
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
                   tf_map_to_camera_init, tf_map_to_odom, tf_body_to_base, tf_base_to_lidar,
                   fast_lio_node, nvblox_node, dynamic_obstacle_node, grid_map_node, grid_map_node_unfiltered):
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
    perception_node = os.path.join(pkg_dir, "..", "drdo_perception", "drdo_perception", "dynamic_obstacle_node.py")
    assert os.path.exists(perception_node), f"Missing {perception_node}"
    print(f"  [CONF] FAST-LIO2 Ouster config found: {fast_lio_cfg}")
    print(f"  [CONF] nvblox 3D ESDF config found: {nvblox_cfg}")

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

    print("  [NODES CONFIGURED]:")
    print("    1. fast_lio::fastlio_mapping -> /cloud_registered (camera_init), TF camera_init->body")
    print("    2. drdo_perception::dynamic_obstacle_node -> /drdo/cloud_static, /drdo/dynamic_obstacles")
    print("    3. drdo_grid_map::grid_map_node -> /drdo/cloud_static in, /map (transient_local) + /drdo/map_image out")
    print("    4. tf2_ros static: map->camera_init, map->odom, body->base_link, base_link->os_sensor")
    print("    5. nvblox_ros::nvblox_node   -> only with enable_nvblox:=true")
    print("[LAUNCH CONFIGURATION VALID]")


if __name__ == "__main__":
    validate_launch_configuration()

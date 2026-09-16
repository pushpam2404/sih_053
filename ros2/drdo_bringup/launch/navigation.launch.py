#!/usr/bin/env python3
"""
DRDO ID26053 — Phase 6: Complete Autonomous Navigation & Mapping Launch Pipeline
Brings up:
  1. Complete Phase 5 SLAM/Mapping stack (FAST-LIO2 + nvblox + drdo_grid_map + TFs)
  2. ROS 2 Navigation 2 (Nav2) Stack:
     - SmacPlannerHybrid (SE(2) kinodynamic global planner)
     - MPPIController (GPU-accelerated trajectory sampling local planner)
     - Global Costmap (using /map static layer)
     - Local Costmap (10x10m rolling window with /drdo/dynamic_obstacles)
     - Behavior Tree Navigator
"""

import os
from typing import Any

# Attempt ROS 2 Launch imports with graceful fallback
try:
    from launch import LaunchDescription
    from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
    from launch.launch_description_sources import PythonLaunchDescriptionSource
    from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
    from launch_ros.substitutions import FindPackageShare
    ROS2_LAUNCH_AVAILABLE = True
except ImportError:
    ROS2_LAUNCH_AVAILABLE = False


def generate_launch_description() -> Any:
    """Builds and returns the ROS 2 LaunchDescription for the DRDO Navigation stack."""
    # Base paths
    phase6_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    nav2_params_file = os.path.join(phase6_dir, "nav2_config", "nav2_params.yaml")

    phase5_launch = os.path.expanduser("~/Desktop/sih/phase5/ros2/launch/drdo_mapping.launch.py")

    # Launch arguments
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation (Gazebo) clock if true"
    )

    params_file_arg = DeclareLaunchArgument(
        "params_file",
        default_value=nav2_params_file,
        description="Full path to the ROS2 Nav2 parameter file to use"
    )

    autostart_arg = DeclareLaunchArgument(
        "autostart",
        default_value="true",
        description="Automatically startup the nav2 stack"
    )

    # 1. Include Phase 5 Mapping Stack (FAST-LIO2, nvblox, grid_map, TFs)
    mapping_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(phase5_launch)
    )

    # 2. Include Nav2 Bringup Navigation Launch
    nav2_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([
                FindPackageShare("nav2_bringup"),
                "launch",
                "navigation_launch.py"
            ])
        ),
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "params_file": LaunchConfiguration("params_file"),
            "autostart": LaunchConfiguration("autostart")
        }.items()
    )

    ld = LaunchDescription()
    ld.add_action(use_sim_time_arg)
    ld.add_action(params_file_arg)
    ld.add_action(autostart_arg)
    ld.add_action(mapping_stack)
    ld.add_action(nav2_bringup_launch)

    return ld


# Fallback validation block (host-side)
if __name__ == "__main__":
    nav2_cfg = os.path.expanduser(
        "~/Desktop/sih/phase6/ros2/nav2_config/nav2_params.yaml")
    mapping_launch = os.path.expanduser(
        "~/Desktop/sih/phase5/ros2/launch/drdo_mapping.launch.py")
    assert os.path.exists(nav2_cfg), f"Missing Nav2 config: {nav2_cfg}"
    assert os.path.exists(mapping_launch), f"Missing Phase 5 launch: {mapping_launch}"
    print("  [CONF] nav2_params.yaml: FOUND")
    print("  [CONF] drdo_mapping.launch.py (Phase 5): FOUND")
    print("  [INFO] ROS 2 not required for host-side validation of launch structure.")
    print("[STEP P6.1.2 COMPLETE]")

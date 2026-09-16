#!/usr/bin/env python3
"""
DRDO ID26053 — Complete Autonomous Navigation & Mapping Launch Pipeline
Brings up:
  1. Mapping stack (FAST-LIO2 + drdo_grid_map + TFs, optional nvblox) from mapping.launch.py
  2. ROS 2 Navigation 2 (Nav2) Stack:
     - SmacPlannerHybrid (SE(2) kinodynamic global planner)
     - MPPIController (sampling-based local controller)
     - Global costmap and rolling local costmap, both fed by /map from drdo_grid_map
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

# Resolved relative to this file so they work from the source tree, a colcon install and Docker.
# (Previously hardcoded to ~/Desktop/sih/phase5/... and phase6/..., which no longer exist.)
LAUNCH_DIR = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.dirname(LAUNCH_DIR)
NAV2_PARAMS_FILE = os.path.join(PKG_DIR, "config", "nav2_params.yaml")
MAPPING_LAUNCH_FILE = os.path.join(LAUNCH_DIR, "mapping.launch.py")


def generate_launch_description() -> Any:
    """Builds and returns the ROS 2 LaunchDescription for the DRDO Navigation stack."""
    use_sim_time_arg = DeclareLaunchArgument(
        "use_sim_time", default_value="false", description="Use simulation (Gazebo) clock if true")
    params_file_arg = DeclareLaunchArgument(
        "params_file", default_value=NAV2_PARAMS_FILE,
        description="Full path to the ROS2 Nav2 parameter file to use")
    autostart_arg = DeclareLaunchArgument(
        "autostart", default_value="true", description="Automatically startup the nav2 stack")

    mapping_stack = IncludeLaunchDescription(PythonLaunchDescriptionSource(MAPPING_LAUNCH_FILE))

    nav2_bringup_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution([FindPackageShare("nav2_bringup"), "launch", "navigation_launch.py"])
        ),
        launch_arguments={
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "params_file": LaunchConfiguration("params_file"),
            "autostart": LaunchConfiguration("autostart"),
        }.items(),
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
    assert os.path.exists(NAV2_PARAMS_FILE), f"Missing Nav2 config: {NAV2_PARAMS_FILE}"
    assert os.path.exists(MAPPING_LAUNCH_FILE), f"Missing mapping launch: {MAPPING_LAUNCH_FILE}"
    print(f"  [CONF] nav2_params.yaml: {NAV2_PARAMS_FILE}")
    print(f"  [CONF] mapping.launch.py: {MAPPING_LAUNCH_FILE}")
    print("  [INFO] ROS 2 not required for host-side validation of launch structure.")

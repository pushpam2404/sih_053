#!/bin/bash
source /opt/ros/humble/setup.bash
source /workspace/sih/phase3/ros2/install/setup.bash
exec ros2 launch /workspace/sih/phase5/ros2/launch/drdo_mapping.launch.py "$@"

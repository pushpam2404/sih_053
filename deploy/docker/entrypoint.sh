#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source /workspace/sih/ros2/install/setup.bash
# Default: full stack (mapping + Nav2). Override, e.g.: docker run ... drdo-nav mapping.launch.py
LAUNCH_FILE="${1:-navigation.launch.py}"
shift || true
exec ros2 launch drdo_bringup "$LAUNCH_FILE" "$@"

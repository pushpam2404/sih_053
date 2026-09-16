cmake_minimum_required(VERSION 3.16)
project(drdo_grid_map)
set(CMAKE_CXX_STANDARD 17)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2_ros REQUIRED)

# Include Phase 2 shared header (adjust path relative to colcon workspace root)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../../phase2/include)

add_executable(grid_map_node src/grid_map_node.cpp)
ament_target_dependencies(grid_map_node
  rclcpp sensor_msgs nav_msgs geometry_msgs tf2_ros)

install(TARGETS grid_map_node DESTINATION lib/${PROJECT_NAME})
ament_package()

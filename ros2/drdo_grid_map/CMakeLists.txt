cmake_minimum_required(VERSION 3.16)
project(drdo_grid_map)
set(CMAKE_CXX_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)

# Single-source engine from the repository root (previously copy-pasted into the node,
# and the include path pointed at a phase2/ directory that no longer exists).
set(DRDO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../..)

add_executable(grid_map_node
  src/grid_map_node.cpp
  ${DRDO_ROOT}/src/drdo_map.cpp
)
target_include_directories(grid_map_node PRIVATE ${DRDO_ROOT}/include)
ament_target_dependencies(grid_map_node
  rclcpp sensor_msgs nav_msgs geometry_msgs tf2 tf2_ros)

install(TARGETS grid_map_node DESTINATION lib/${PROJECT_NAME})
ament_package()

cmake_minimum_required(VERSION 3.16)
project(drdo_lidar_mapping VERSION 0.6.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Optimization flags for ARM Cortex / Jetson / x86_64
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-O3 -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# 1. Core C++ Static Library
add_library(drdo_map_lib STATIC
    src/drdo_map.cpp
)
target_include_directories(drdo_map_lib PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# 2. C++ Grid Engine Unit Test
add_executable(test_grid_engine
    tests/cpp/test_grid_engine.cpp
)
target_link_libraries(test_grid_engine PRIVATE drdo_map_lib)

# 3. Optional Pybind11 Python Module
find_package(pybind11 QUIET)
if(NOT pybind11_FOUND)
    # Check if python provides pybind11 cmake path
    execute_process(
        COMMAND python3 -m pybind11 --cmake
        OUTPUT_VARIABLE pybind11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(pybind11_CMAKE_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${pybind11_CMAKE_DIR}")
        find_package(pybind11 REQUIRED)
    endif()
endif()

if(pybind11_FOUND)
    message(STATUS "Found pybind11: building drdo_map Python extension")
    pybind11_add_module(drdo_map
        src/drdo_map_py.cpp
    )
    target_link_libraries(drdo_map PRIVATE drdo_map_lib)
else()
    message(STATUS "pybind11 not found; skipping python C++ extension compilation")
endif()

# 4. Optional CUDA Kernels (Jetson / NVIDIA GPU)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_ARCHITECTURES 87) # Jetson AGX Orin (Ampere architecture)
    add_executable(test_cuda_kernels
        cuda/drdo_cuda_kernels.cu
    )
    target_include_directories(test_cuda_kernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    message(STATUS "CUDA enabled: building test_cuda_kernels for sm_87 (Jetson AGX Orin)")
endif()

# Installation rules
install(TARGETS drdo_map_lib
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(DIRECTORY include/drdo_lidar_mapping DESTINATION include)

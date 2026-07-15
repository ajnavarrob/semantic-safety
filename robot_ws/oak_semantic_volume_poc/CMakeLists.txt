cmake_minimum_required(VERSION 3.5)
project(oak_semantic_volume_poc)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()
add_compile_options(-Wall -Wextra -Wpedantic -O3)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(PCL REQUIRED COMPONENTS common)
find_package(pcl_conversions REQUIRED)

include_directories(include ${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(semantic_volume_node src/semantic_volume_node.cpp)
ament_target_dependencies(semantic_volume_node rclcpp sensor_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs)
target_link_libraries(semantic_volume_node ${PCL_LIBRARIES})

install(TARGETS semantic_volume_node DESTINATION lib/${PROJECT_NAME})
install(PROGRAMS oak_semantic_volume_poc/oak_segmentation_node.py DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config models DESTINATION share/${PROJECT_NAME})
install(PROGRAMS scripts/install_foxy.sh scripts/run_tests.sh scripts/benchmark_geometry.py DESTINATION share/${PROJECT_NAME}/scripts)
install(PROGRAMS oak_semantic_volume_poc/segmentation_adapter.py DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_ray_voxel test/test_ray_voxel.cpp)
endif()

ament_package()

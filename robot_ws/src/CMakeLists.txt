cmake_minimum_required(VERSION 3.20)
project(unitree_ros2_poisson_simple)

# Enable C++ and CUDA
enable_language(CUDA)

# Default to C99
if(NOT CMAKE_C_STANDARD)
  set(CMAKE_C_STANDARD 99)
endif()

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "-O3 -pthread -pg -fopenmp")

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# CUDA settings
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 87)

# Definitions
add_definitions(-DUSE_CUDA)

# CUDA sources
set(CUDA_SOURCES src/kernel.cu)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wno-deprecated-gpu-targets")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")  # enable cuda-gdb (expensive)
# Removed --extended-lambda (not needed, causes GCC 11 std::function errors)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --use_fast_math")

include_directories(include include/common include/nlohmann include/poisson)
link_directories(src)

set (
  DEPENDENCY_LIST
  unitree_go
  unitree_hg
  unitree_api
  rclcpp
  std_msgs
  nav_msgs
  geometry_msgs
  sensor_msgs
  rosbag2_cpp
  pcl_conversions
  tf2_ros
  tf2_eigen
  OpenCV
)

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(unitree_go REQUIRED)
find_package(unitree_hg REQUIRED)
find_package(unitree_api REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)

set(OpenCV_DIR "/usr/local/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(pcl_conversions REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OsqpEigen REQUIRED)
find_package(osqp REQUIRED)
find_package(OpenMP REQUIRED)

# Detect Jetson platform
if(EXISTS "/etc/nv_tegra_release")
    set(IS_JETSON TRUE)
    message(STATUS "🤖 Detected Jetson platform")
else()
    set(IS_JETSON FALSE)
endif()

# Conditional CUDA setup
if(IS_JETSON OR CMAKE_VERSION VERSION_LESS "3.18")
    # Jetson or Old CMake: Use legacy FindCUDA
    if(IS_JETSON)
        message(STATUS "Using legacy FindCUDA module (Jetson)")
    else()
        message(STATUS "Using legacy FindCUDA module (CMake < 3.18)")
    endif()
    
    find_package(CUDA REQUIRED)
    find_package(Curses REQUIRED)
    
    include_directories(${OpenCV_INCLUDE_DIRS})
    include_directories(${CURSES_INCLUDE_DIRS})
    include_directories(${CUDA_INCLUDE_DIRS})
    
    cuda_add_library(cuda_kernel STATIC ${CUDA_SOURCES})
    target_include_directories(cuda_kernel PUBLIC inc/)
    set_target_properties(cuda_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)
else()
    # Modern CMake on Desktop: Use CUDA language support
    message(STATUS "Using modern CMake CUDA support (CMake >= 3.18)")
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    find_package(Curses REQUIRED)
    
    include_directories(${OpenCV_INCLUDE_DIRS})
    include_directories(${CURSES_INCLUDE_DIRS})
    
    add_library(cuda_kernel STATIC ${CUDA_SOURCES})
    set_target_properties(cuda_kernel PROPERTIES CUDA_SEPARABLE_COMPILATION ON POSITION_INDEPENDENT_CODE ON)
    target_include_directories(cuda_kernel PUBLIC include/)
    target_link_libraries(cuda_kernel CUDA::cudart)
endif()

# Main executable
# poissonLocal disabled - uses old MPC interface incompatible with guidance field changes
# add_executable(poissonLocal src/poisson.cpp src/utils.cpp src/common/ros2_sport_client.cpp)
add_executable(teleOp src/teleop.cpp)
add_executable(semantic_poisson src/semantic_poisson.cpp src/utils.cpp src/common/ros2_sport_client.cpp)
add_executable(odom_publisher src/odom_publisher.cpp)
add_executable(odom_publisher_t265 src/odom_publisher_t265.cpp)
add_executable(odom_publisher_fastlio src/odom_publisher_fastlio.cpp)

# target_link_libraries(poissonLocal Eigen3::Eigen OsqpEigen::OsqpEigen cuda_kernel ${OpenCV_LIBS} ${PCL_LIBRARIES})
target_link_libraries(teleOp ${CURSES_LIBRARIES})
target_link_libraries(semantic_poisson Eigen3::Eigen OsqpEigen::OsqpEigen cuda_kernel ${OpenCV_LIBS} ${PCL_LIBRARIES})

# ament_target_dependencies(poissonLocal ${DEPENDENCY_LIST})
ament_target_dependencies(teleOp ${DEPENDENCY_LIST})
ament_target_dependencies(semantic_poisson ${DEPENDENCY_LIST})
ament_target_dependencies(odom_publisher ${DEPENDENCY_LIST})
ament_target_dependencies(odom_publisher_t265 ${DEPENDENCY_LIST})
ament_target_dependencies(odom_publisher_fastlio ${DEPENDENCY_LIST})

install(TARGETS
        # poissonLocal
        teleOp
        semantic_poisson
        odom_publisher
        odom_publisher_t265
        odom_publisher_fastlio
        DESTINATION lib/${PROJECT_NAME})

# Install Python scripts
install(PROGRAMS
        scripts/human_tracking.py
        scripts/yolo_detector.py
        DESTINATION lib/${PROJECT_NAME})

# Install launch files
install(DIRECTORY
        launch/
        DESTINATION share/${PROJECT_NAME}/launch)

# Install config files
install(DIRECTORY
        config/
        DESTINATION share/${PROJECT_NAME}/config)


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # uncomment the line when a copyright and license is not present in all source files
  #set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # uncomment the line when this package is not in a git repo
  #set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()

cmake_minimum_required(VERSION 3.20)
project(PoissonZO)

# find_package(ament_cmake REQUIRED) # Commented out as it might not be needed for a pure Pybind11 module
find_package(OpenMP REQUIRED)
find_package(pybind11 CONFIG REQUIRED) # Find Pybind11

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# Removed -pg (profiling) as it might interfere with Python module loading or is not always desired for release modules.
# Kept -O3 and -fopenmp. Consider if -pthread is still needed or handled by Pybind11/OpenMP.
set(CMAKE_CXX_FLAGS "-O3 -fopenmp") 

# add_executable(poissonZO src/main.cpp) # Old executable target

# Create the Python module using Pybind11
# The module will be named "poisson_solver" as defined in main.cpp (PYBIND11_MODULE macro)
# The output will be poisson_solver.cpython-*.so (or .pyd on Windows)
pybind11_add_module(poisson_solver src/main.cpp)

# Link OpenMP if found and Pybind11 requires it or your code uses it directly
if(OpenMP_FOUND)
    target_link_libraries(poisson_solver PUBLIC OpenMP::OpenMP_CXX)
endif()

# Optional: Specify install location for the module
# install(TARGETS poisson_solver LIBRARY DESTINATION your_python_packages_path)
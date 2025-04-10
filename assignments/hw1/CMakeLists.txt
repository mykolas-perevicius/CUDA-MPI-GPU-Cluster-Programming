cmake_minimum_required(VERSION 3.15)
project(hw1 LANGUAGES C CXX CUDA)

# Find MPI
find_package(MPI REQUIRED)
if (MPI_FOUND)
  include_directories(\${MPI_INCLUDE_PATH})
endif()

# Option for CUDA support
option(ENABLE_CUDA "Enable CUDA support" OFF)
if (ENABLE_CUDA)
  find_package(CUDA REQUIRED)
  # Add CUDA settings as needed
endif()

# Add the executable
add_executable(template src/template.c)
target_link_libraries(template \${MPI_LIBRARIES})
if (ENABLE_CUDA)
  target_link_libraries(template \${CUDA_LIBRARIES})
endif()

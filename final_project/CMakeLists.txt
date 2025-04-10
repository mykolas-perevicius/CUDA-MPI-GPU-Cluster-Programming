cmake_minimum_required(VERSION 3.10)
project(AlexNet_MPI_CUDA C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

find_package(MPI REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${MPI_INCLUDE_PATH} ${CUDA_INCLUDE_DIRS} include)

# Create an executable called "template" from main.cpp and alexnet_hybrid.cu
add_executable(template src/main.cpp src/alexnet_hybrid.cu)

target_link_libraries(template ${MPI_LIBRARIES} ${CUDA_LIBRARIES})

cmake_minimum_required(VERSION 3.10)
project(mpi_cuda_vectoradd LANGUAGES CXX CUDA)

# Find MPI package
find_package(MPI REQUIRED)

# Set the path to CUDA and MPI include directories
include_directories(${MPI_INCLUDE_PATH})
find_package(CUDA REQUIRED)

# Enable CUDA support
enable_language(CUDA)

# Specify the target executable
add_executable(mpi_cuda_vectoradd vectoradd.cu)

# Link CUDA runtime library
target_link_libraries(mpi_cuda_vectoradd ${MPI_LIBRARIES} ${CUDA_LIBRARIES})

# Set compiler flags for CUDA and MPI
target_include_directories(mpi_cuda_vectoradd PRIVATE ${MPI_INCLUDE_PATH} ${CUDA_INCLUDE_DIRS})
target_link_libraries(mpi_cuda_vectoradd MPI::MPI_CXX)

# If you need to pass additional flags to nvcc, you can use this
set_target_properties(mpi_cuda_vectoradd PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

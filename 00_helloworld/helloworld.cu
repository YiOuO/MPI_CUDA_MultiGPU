#include <mpi.h>            // MPI header
#include <cuda_runtime.h>    // CUDA runtime header
#include <iostream>

// CUDA kernel function to print from the GPU
__global__ void hello_from_gpu(int rank) {
    printf("Hello from GPU! MPI rank %d, CUDA thread %d\n", rank, threadIdx.x);
}

int main(int argc, char **argv) {
    // Initialize MPI
    MPI_Init(&argc, &argv);

    int world_size, world_rank;

    // Get the number of processes
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Host prints
    std::cout << "Hello from CPU! MPI rank " << world_rank << " out of " << world_size << std::endl;

    // Launch CUDA kernel
    hello_from_gpu<<<1, 10>>>(world_rank);

    // Synchronize the device to ensure all CUDA printf outputs finish
    cudaDeviceSynchronize();

    // Finalize MPI
    MPI_Finalize();
    
    return 0;
}

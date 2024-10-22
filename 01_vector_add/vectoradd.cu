#include <mpi.h>
#include <cuda_runtime.h>
#include <iostream>

// CUDA kernel for vector addition
__global__ void vector_add(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// Utility function to initialize arrays
void init_array(float* arr, int n, float value) {
    for (int i = 0; i < n; ++i) {
        arr[i] = i;
    }
}

int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int num_gpus;
    cudaGetDeviceCount(&num_gpus);
    std::cout<< "num of gpu is "<< num_gpus<<std::endl;
    if (world_rank < num_gpus) {
        cudaSetDevice(world_rank);
    } else {
        std::cerr << "MPI rank exceeds available GPU devices!" << std::endl;
        MPI_Finalize();
        return -1;
    }
    

    // Problem size and sub-array size per GPU
    const int N = 12;  // Total size of the array
    int chunk_size = N / world_size;  // Each process gets part of the array

    // Allocate host arrays
    float *A = nullptr, *B = nullptr, *C = nullptr;
    if (world_rank == 0) {
        A = new float[N];
        B = new float[N];
        C = new float[N];
        init_array(A, N, 1.0f);  // Initialize array A with 1.0
        init_array(B, N, 2.0f);  // Initialize array B with 2.0
    }

    // Allocate GPU memory for the sub-array
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, chunk_size * sizeof(float));
    cudaMalloc(&d_B, chunk_size * sizeof(float));
    cudaMalloc(&d_C, chunk_size * sizeof(float));

    // Scatter the array chunks to different GPUs
    float* sub_A = new float[chunk_size];
    float* sub_B = new float[chunk_size];
    MPI_Scatter(A, chunk_size, MPI_FLOAT, sub_A, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, chunk_size, MPI_FLOAT, sub_B, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // Output Sub Arrays and you can see how the arrays are divided
    std::cout << "Hello from CPU! MPI rank " << world_rank << " out of " << world_size <<" array "<< sub_A<< std::endl;
    for (int i = 0; i < chunk_size; ++i) {
        std::cout << "sub_A[" << i << "] = " << sub_A[i] << std::endl;
    }

    // Copy the sub-arrays to the corresponding GPU memory
    cudaMemcpy(d_A, sub_A, chunk_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, sub_B, chunk_size * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the CUDA kernel for vector addition
    int threadsPerBlock = 256;
    int blocksPerGrid = (chunk_size + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, chunk_size);
    cudaDeviceSynchronize();

    // Copy the result back from GPU to host
    float* sub_C = new float[chunk_size];
    cudaMemcpy(sub_C, d_C, chunk_size * sizeof(float), cudaMemcpyDeviceToHost);

    // Gather the results from all GPUs
    MPI_Gather(sub_C, chunk_size, MPI_FLOAT, C, chunk_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // The root process prints the final result
    if (world_rank == 0) {
        std::cout << "Array addition result:" << std::endl;
        for (int i = 0; i < 10; i++) {
            std::cout << C[i] << " ";  // Print first 10 elements
        }
        std::cout << std::endl;
    }

    // Free memory
    delete[] sub_A;
    delete[] sub_B;
    delete[] sub_C;
    if (world_rank == 0) {
        delete[] A;
        delete[] B;
        delete[] C;
    }
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    MPI_Finalize();
    return 0;
}

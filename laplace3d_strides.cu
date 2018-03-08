#include <stdio.h>
#include <iostream>

__host__ __device__ __forceinline__ int
index_strides(const int i, const int j, const int k, const dim3 strides) {
    return i * strides.x + j * strides.y + k * strides.z;
}

__global__ void laplace3d_strides(double *d, double *n, const dim3 sizes,
                                  const dim3 strides) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i > 0 && i < sizes.x - 1)
        if (j > 0 && j < sizes.y - 1)
            if (k > 0 && k < sizes.z - 1)
                d[index_strides(i, j, k, strides)] =
                    1. / 2. *
                    (                                                    //
                        __ldg(&n[index_strides(i - 1, j, k, strides)])   //
                        + __ldg(&n[index_strides(i + 1, j, k, strides)]) //
                        + __ldg(&n[index_strides(i, j - 1, k, strides)]) //
                        + __ldg(&n[index_strides(i, j + 1, k, strides)]) //
                        + __ldg(&n[index_strides(i, j, k - 1, strides)]) //
                        + __ldg(&n[index_strides(i, j, k + 1, strides)]) //
                        - 6. * __ldg(&n[index_strides(i, j, k, strides)]));
}

float elapsed(cudaEvent_t &start, cudaEvent_t &stop) {
    float result;
    cudaEventElapsedTime(&result, start, stop);
    return result;
}

extern "C" {
void call_laplace3d_strides(double *d, double *n, int Nx, int Ny, int Nz) {
    const dim3 sizes(Nx, Ny, Nz);
    const dim3 strides(1, Nx, Nx * Ny);
    const dim3 threadsPerBlock(8, 8, 8);
    const dim3 nBlocks(Nx / threadsPerBlock.x, Ny / threadsPerBlock.y,
                       Nz / threadsPerBlock.z);

    printf("Calling a CUDA kernel...\n");

    cudaEvent_t start_; // TODO timing should not be here of course...
    cudaEvent_t stop_;
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    const size_t nrepeat = 1; // TODO we want to do this only once!
    printf("block size: (%d,%d,%d)\n", threadsPerBlock.x, threadsPerBlock.y,
           threadsPerBlock.z);
    printf("WARNING: We are repeating the kernel call %d times, don't use this "
           "for production.\n",
           nrepeat);

    cudaEventRecord(start_, 0);
    for (size_t i = 0; i < nrepeat; ++i)
        laplace3d_strides<<<nBlocks, threadsPerBlock>>>(d, n, sizes, strides);
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);

    std::cout << "time per iteration: "
              << elapsed(start_, stop_) / (float)nrepeat << "ms" << std::endl;

    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA ERROR: %s in %s at line %d\n",
                cudaGetErrorString(error), __FILE__, __LINE__);
        exit(-1);
    }
}
}

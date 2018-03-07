#include <iostream>
#include <stdio.h>
#include <math.h>

// const size_t Nx = 64;
// const size_t Ny = 64;
// const size_t Nz = 64;
const size_t Nx = 128;
const size_t Ny = 512;
const size_t Nz = 512;

const size_t nrepeat = 100;

__host__ __device__ __forceinline__ size_t index(const size_t i, const size_t j,
                                                 const size_t k) {
    const size_t istride = 1;
    const size_t jstride = Nx;
    const size_t kstride = Nx * Ny;
    return i * istride + j * jstride + k * kstride;
}

__global__ void laplace3d(double *d, double *n) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t j = threadIdx.y + blockIdx.y * blockDim.y;
    size_t k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i > 0 && i < Nx - 1)
        if (j > 0 && j < Ny - 1)
            if (k > 0 && k < Nz - 1)
                d[index(i, j, k)] =
                    1. / 2. *
                    (                                                   //
                        n[index(i - 1, j, k)] + n[index(i + 1, j, k)]   //
                        + n[index(i, j - 1, k)] + n[index(i, j + 1, k)] //
                        + n[index(i, j, k - 1)] + n[index(i, j, k + 1)] //
                        - 6. * n[index(i, j, k)]);
}

__global__ void test_kernel(double *d, double *n) {
    size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    size_t j = threadIdx.y + blockIdx.y * blockDim.y;
    size_t k = threadIdx.z + blockIdx.z * blockDim.z;

    d[index(i, j, k)] = 123.;
}

void init(double *n) {
    for (size_t i = 0; i < Nx; ++i)
        for (size_t j = 0; j < Ny; ++j)
            for (size_t k = 0; k < Nz; ++k) {
                n[index(i, j, k)] = sin((double)i / ((double)Nx - 1.) * M_PI) *
                                    sin((double)j / ((double)Ny - 1.) * M_PI) *
                                    sin((double)k / ((double)Nz - 1.) * M_PI);
            }
}

void print(double *n) {
    for (size_t i = 0; i < Nx; ++i) {
        std::cout << (double)i / (double)(Nx - 1) << " \t"
                  << 1. / -0.004 * n[index(i, Ny / 2, Nz / 2)] << std::endl;
    }
}

float elapsed(cudaEvent_t &start, cudaEvent_t &stop) {
    float result;
    cudaEventElapsedTime(&result, start, stop);
    return result;
}

int main() {
    cudaEvent_t start_;
    cudaEvent_t stop_;
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);

    size_t total_size = Nx * Ny * Nz;
    double *d = new double[total_size];
    double *n = new double[total_size];

    init(n);

    double *dd;
    cudaMalloc(&dd, sizeof(double) * total_size);
    double *dn;
    cudaMalloc(&dn, sizeof(double) * total_size);

    dim3 threadsPerBlock(32, 4, 4);
    dim3 nBlocks(Nx / threadsPerBlock.x, Ny / threadsPerBlock.y,
                 Nz / threadsPerBlock.z);

    std::cout << "#threads/block: " << threadsPerBlock.x << "/"
              << threadsPerBlock.y << "/" << threadsPerBlock.z << std::endl;
    std::cout << "#blocks: " << nBlocks.x << "/" << nBlocks.y << "/"
              << nBlocks.z << std::endl;

    cudaMemcpy(dn, n, sizeof(double) * total_size, cudaMemcpyHostToDevice);

    cudaEventRecord(start_, 0);
    for (size_t i = 0; i < nrepeat; ++i)
        laplace3d<<<nBlocks, threadsPerBlock>>>(dd, dn);
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);

    cudaMemcpy(d, dd, sizeof(double) * total_size, cudaMemcpyDeviceToHost);

    print(d);
    std::cout << "# time= " << elapsed(start_, stop_) / (float)nrepeat << "ms"
              << std::endl;

    delete[] d;
    cudaFree(dd);
    cudaFree(dn);
}

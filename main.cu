#include <iostream>
#include <stdio.h>
#include <math.h>

// const size_t Nx = 64;
// const size_t Ny = 64;
// const size_t Nz = 64;
size_t Nx = 128;
size_t Ny = 512;
size_t Nz = 512;

const size_t nrepeat = 100;

__host__ __device__ __forceinline__ int index(const int i, const int j,
                                              const int k, const dim3 sizes) {
    const int istride = 1;
    const int jstride = sizes.x;
    const int kstride = sizes.x * sizes.y;
    return i * istride + j * jstride + k * kstride;
}
__host__ __device__ __forceinline__ int
index_strides(const int i, const int j, const int k, const dim3 strides) {
    return i * strides.x + j * strides.y + k * strides.z;
}

// TODO parameterize ldg
//__global__ void laplace3d(double *d, double *n) {
//    int i = threadIdx.x + blockIdx.x * blockDim.x;
//    int j = threadIdx.y + blockIdx.y * blockDim.y;
//    int k = threadIdx.z + blockIdx.z * blockDim.z;
//
//    if (i > 0 && i < Nx - 1)
//        if (j > 0 && j < Ny - 1)
//            if (k > 0 && k < Nz - 1)
//                d[index(i, j, k)] =
//                    1. / 2. * ( //
//                                  __ldg(&n[index(i - 1, j, k)]) +
//                                  __ldg(&n[index(i + 1, j, k)]) //
//                                  + __ldg(&n[index(i, j - 1, k)]) +
//                                  __ldg(&n[index(i, j + 1, k)]) //
//                                  + __ldg(&n[index(i, j, k - 1)]) +
//                                  __ldg(&n[index(i, j, k + 1)]) //
//                                  - 6. * __ldg(&n[index(i, j, k)]));
//}

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

__host__ __device__ __forceinline__ int index_smem(const int i, const int j,
                                                   const int k) {
    return (i + 1) + (j + 1) * (blockDim.x + 2) +
           (k + 1) * (blockDim.x + 2) * (blockDim.y + 2);
}

__global__ void laplace3d_smem(double *d, double *n, const dim3 sizes,
                               const dim3 strides) {
    extern __shared__ double smem[];
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    int ii = threadIdx.x;
    int jj = threadIdx.y;
    int kk = threadIdx.z;

    smem[index_smem(ii, jj, kk)] = __ldg(&n[index_strides(i, j, k, strides)]);
    if (ii == 0) {
        if (jj != 0 && kk != 0 && //
            jj != blockDim.y - 1 && kk != blockDim.z - 1) {
            if (i > 0)
                smem[index_smem(-1, jj, kk)] =
                    __ldg(&n[index_strides(i - 1, j, k, strides)]);
            if (i < sizes.x - 1)
                smem[index_smem(blockDim.x, jj, kk)] =
                    __ldg(&n[index_strides(i + 1, j, k, strides)]);
        }
    }
    if (jj == 0) {
        if (ii != 0 && kk != 0 && //
            ii != blockDim.x - 1 && kk != blockDim.z - 1) {
            if (j > 0)
                smem[index_smem(ii, -1, kk)] =
                    __ldg(&n[index_strides(i, j - 1, k, strides)]);
            if (j < sizes.y - 1)
                smem[index_smem(ii, blockDim.y, kk)] =
                    __ldg(&n[index_strides(i, j + 1, k, strides)]);
        }
    }
    if (kk == 0) {
        if (ii != 0 && jj != 0 && //
            ii != blockDim.x - 1 && jj != blockDim.y - 1) {
            if (k > 0)
                smem[index_smem(ii, jj, -1)] =
                    __ldg(&n[index_strides(i, j, k - 1, strides)]);
            if (k < sizes.z - 1)
                smem[index_smem(ii, jj, blockDim.z)] =
                    __ldg(&n[index_strides(i, j, k + 1, strides)]);
        }
    }

    if (i > 0 && i < sizes.x - 1)
        if (j > 0 && j < sizes.y - 1)
            if (k > 0 && k < sizes.z - 1)
                d[index_strides(i, j, k, strides)] =
                    1. / 2. * (                                      //
                                  smem[index_smem(ii - 1, jj, kk)]   //
                                  + smem[index_smem(ii + 1, jj, kk)] //
                                  + smem[index_smem(ii, jj - 1, kk)] //
                                  + smem[index_smem(ii, jj + 1, kk)] //
                                  + smem[index_smem(ii, jj, kk - 1)] //
                                  + smem[index_smem(ii, jj, kk + 1)] //
                                  - 6. * smem[index_smem(ii, jj, kk)]);

    //    d[index_strides(i, j, k, strides)] = smem[index_smem(ii, jj, kk)];
}

void init(double *n, const dim3 sizes) {
    for (size_t i = 0; i < sizes.x; ++i)
        for (size_t j = 0; j < sizes.y; ++j)
            for (size_t k = 0; k < sizes.z; ++k) {
                n[index(i, j, k, sizes)] =
                    sin((double)i / ((double)sizes.x - 1.) * M_PI) *
                    sin((double)j / ((double)sizes.y - 1.) * M_PI) *
                    sin((double)k / ((double)sizes.z - 1.) * M_PI);
            }
}

void print(double *n, const dim3 sizes) {
    for (size_t i = 0; i < sizes.x; ++i) {
        std::cout << (double)i / (double)(sizes.x - 1) << " \t"
                  << 1. / -0.004 * n[index(i, sizes.y / 2, sizes.z / 2, sizes)]
                  << std::endl;
    }
}

float elapsed(cudaEvent_t &start, cudaEvent_t &stop) {
    float result;
    cudaEventElapsedTime(&result, start, stop);
    return result;
}

void execute(dim3 threadsPerBlock, double *dd, double *dn) {
    const dim3 sizes(Nx, Ny, Nz);
    const dim3 strides(1, Nx, Nx * Ny);

    cudaEvent_t start_;
    cudaEvent_t stop_;
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    dim3 nBlocks(Nx / threadsPerBlock.x, Ny / threadsPerBlock.y,
                 Nz / threadsPerBlock.z);

    cudaEventRecord(start_, 0);

    size_t smem_size = (threadsPerBlock.x + 2) * (threadsPerBlock.y + 2) *
                       (threadsPerBlock.z + 2);

    for (size_t i = 0; i < nrepeat; ++i)
        //        laplace3d_strides<<<nBlocks, threadsPerBlock>>>(dd, dn, sizes,
        //        strides);
        laplace3d_smem<<<nBlocks, threadsPerBlock,
                         smem_size * sizeof(double)>>>(dd, dn, sizes, strides);
    //        laplace3d<<<nBlocks, threadsPerBlock>>>(dd, dn);
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);

    std::cout << "# threads/block = (" << threadsPerBlock.x << "/"
              << threadsPerBlock.y << "/" << threadsPerBlock.z << "), \t";
    std::cout << "blocks = (" << nBlocks.x << "/" << nBlocks.y << "/"
              << nBlocks.z << "), \t";
    std::cout << "time = " << elapsed(start_, stop_) / (float)nrepeat << "ms"
              << std::endl;

    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
}

int main() {
    dim3 sizes(Nx, Ny, Nz);

    size_t total_size = Nx * Ny * Nz;
    double *d = new double[total_size];
    double *n = new double[total_size];

    init(n, sizes);

    double *dd;
    cudaMalloc(&dd, sizeof(double) * total_size);
    double *dn;
    cudaMalloc(&dn, sizeof(double) * total_size);

    cudaMemcpy(dn, n, sizeof(double) * total_size, cudaMemcpyHostToDevice);

    execute(dim3(32, 4, 4), dd, dn);
    execute(dim3(8, 8, 8), dd, dn);
    //    execute(dim3(16, 8, 8), dd, dn);
    //    execute(dim3(16, 16, 4), dd, dn);
    //    execute(dim3(32, 8, 4), dd, dn);
    //    execute(dim3(64, 4, 4), dd, dn);

    cudaMemcpy(d, dd, sizeof(double) * total_size, cudaMemcpyDeviceToHost);

    print(d, sizes);

    delete[] d;
    cudaFree(dd);
    cudaFree(dn);
}

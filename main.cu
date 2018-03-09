#include <iostream>
#include <stdio.h>
#include <math.h>
#include <vector>

/**
 * Several variations of a simple 3D 7-point Laplacian.
 *
 * Notes:
 * - Since it is a very simple example which naturally fits the parallelization
 * model of CUDA, there are not many elaborate optimization.
 * - Using shared memory might help a lot once several stencils are fused to
 * store intermediate values. In this example there is no/negligible performance
 * improvement (depending on the GPU architecture).
 */

const size_t Nx = 128;
const size_t Ny = 512;
const size_t Nz = 512;

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

/**
 * Naive/non-optimized version.
 */
__global__ void laplace3d_no_ldg(double *d, double *n, const dim3 sizes,
                                 const dim3 strides) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i > 0 && i < sizes.x - 1)
        if (j > 0 && j < sizes.y - 1)
            if (k > 0 && k < sizes.z - 1)
                d[index_strides(i, j, k, strides)] =
                    1. / 2. * (                                              //
                                  (n[index_strides(i - 1, j, k, strides)])   //
                                  + (n[index_strides(i + 1, j, k, strides)]) //
                                  + (n[index_strides(i, j - 1, k, strides)]) //
                                  + (n[index_strides(i, j + 1, k, strides)]) //
                                  + (n[index_strides(i, j, k - 1, strides)]) //
                                  + (n[index_strides(i, j, k + 1, strides)]) //
                                  - 6. * (n[index_strides(i, j, k, strides)]));
}

/**
 * Putting const __restrict__ on the read only pointers allows the compiler to
 * automatically detect that the read-only data cache can be used (no need for
 * explicit __ldg())
 */
__global__ void laplace3d_ldg(double *d, double *n, const dim3 sizes,
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

/**
 * Relative indexing should reduce the number of integer computations which
 * could have an impact.
 */
__global__ void laplace3d_relative_indexing(double *d, double *n,
                                            const dim3 sizes,
                                            const dim3 strides) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    int index = index_strides(i, j, k, strides);

    if (i > 0 && i < sizes.x - 1)
        if (j > 0 && j < sizes.y - 1)
            if (k > 0 && k < sizes.z - 1)
                d[index] = 1. / 2. * (                                  //
                                         __ldg(&n[index - strides.x])   //
                                         + __ldg(&n[index + strides.x]) //
                                         + __ldg(&n[index - strides.y]) //
                                         + __ldg(&n[index + strides.y]) //
                                         + __ldg(&n[index - strides.z]) //
                                         + __ldg(&n[index + strides.z]) //
                                         - 6. * __ldg(&n[index]));
}

/**
 * Putting const __restrict__ on the read only pointers allows the compiler to
 * automatically detect that the read-only data cache can be used (no need for
 * explicit __ldg())
 */
__global__ void laplace3d_const_restrict(double *__restrict__ d,
                                         const double *__restrict__ n,
                                         const dim3 sizes, const dim3 strides) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    if (i > 0 && i < sizes.x - 1)
        if (j > 0 && j < sizes.y - 1)
            if (k > 0 && k < sizes.z - 1)
                d[index_strides(i, j, k, strides)] =
                    1. / 2. * (                                              //
                                  (n[index_strides(i - 1, j, k, strides)])   //
                                  + (n[index_strides(i + 1, j, k, strides)]) //
                                  + (n[index_strides(i, j - 1, k, strides)]) //
                                  + (n[index_strides(i, j + 1, k, strides)]) //
                                  + (n[index_strides(i, j, k - 1, strides)]) //
                                  + (n[index_strides(i, j, k + 1, strides)]) //
                                  - 6. * (n[index_strides(i, j, k, strides)]));
}

__host__ __device__ __forceinline__ int index_smem(const int i, const int j,
                                                   const int k) {
    return (i + 1) + (j + 1) * (blockDim.x + 2) +
           (k + 1) * (blockDim.x + 2) * (blockDim.y + 2);
}

/**
 * Shared memory is a per block scratch pad (user-managed cache), which usually
 * is very beneficial for storing intermediate values.
 *
 * Here we just copy the local input of the stencil for each block into its
 * buffer and read from the buffer.
 * The halo region is filled by dedicated threads (first and last in all
 * directions).
 *
 * Note: Another option would be to add extra threads for the halo points to
 * each block and let them sleep for the actual computation.
 */
__global__ void laplace3d_smem(double *d, double *n, const dim3 sizes,
                               const dim3 strides) {
    extern __shared__ double smem[];
    // global indices
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    // local indices
    int ii = threadIdx.x;
    int jj = threadIdx.y;
    int kk = threadIdx.z;

    // copy all elements of the compute domain into the shared mem buffer (on
    // block level)
    smem[index_smem(ii, jj, kk)] = __ldg(&n[index_strides(i, j, k, strides)]);

    // first and last threads (in all dimensions copy the halo region into the
    // shared mem buffer
    if (ii == 0)
        if (i > 0)
            smem[index_smem(-1, jj, kk)] =
                __ldg(&n[index_strides(i - 1, j, k, strides)]);
    if (ii == blockDim.x - 1)
        if (i < sizes.x - 1)
            smem[index_smem(blockDim.x, jj, kk)] =
                __ldg(&n[index_strides(i + 1, j, k, strides)]);
    if (jj == 0)
        if (j > 0)
            smem[index_smem(ii, -1, kk)] =
                __ldg(&n[index_strides(i, j - 1, k, strides)]);
    if (jj == blockDim.y - 1)
        if (j < sizes.y - 1)
            smem[index_smem(ii, blockDim.y, kk)] =
                __ldg(&n[index_strides(i, j + 1, k, strides)]);

    if (kk == 0)
        if (k > 0)
            smem[index_smem(ii, jj, -1)] =
                __ldg(&n[index_strides(i, j, k - 1, strides)]);
    if (kk == blockDim.z - 1)
        if (k < sizes.z - 1)
            smem[index_smem(ii, jj, blockDim.z)] =
                __ldg(&n[index_strides(i, j, k + 1, strides)]);

    __syncthreads();
    if (i > 0 && i < sizes.x - 1)
        if (j > 0 && j < sizes.y - 1)
            if (k > 0 && k < sizes.z - 1)
                // read only from the shared mem buffer
                d[index_strides(i, j, k, strides)] =
                    1. / 2. * (                                      //
                                  smem[index_smem(ii - 1, jj, kk)]   //
                                  + smem[index_smem(ii + 1, jj, kk)] //
                                  + smem[index_smem(ii, jj - 1, kk)] //
                                  + smem[index_smem(ii, jj + 1, kk)] //
                                  + smem[index_smem(ii, jj, kk - 1)] //
                                  + smem[index_smem(ii, jj, kk + 1)] //
                                  - 6. * smem[index_smem(ii, jj, kk)]);
}

__host__ __device__ __forceinline__ int index_smem2(const int i, const int j,
                                                    const int k) {
    return i + j * (blockDim.x) + k * (blockDim.x) * (blockDim.y);
}

/**
 * Using extra threads for copying halo points to
 * each block and let them sleep for the actual computation.
 */
__global__ void laplace3d_smem2(double *d, double *n, const dim3 sizes,
                                const dim3 strides) {
    extern __shared__ double smem[];
    // global indices
    int i = -1 + (int)(threadIdx.x + blockIdx.x * (blockDim.x - 2));
    int j = -1 + (int)(threadIdx.y + blockIdx.y * (blockDim.y - 2));
    int k = -1 + (int)(threadIdx.z + blockIdx.z * (blockDim.z - 2));

    // local indices
    int ii = threadIdx.x;
    int jj = threadIdx.y;
    int kk = threadIdx.z;

    // copy all elements of the compute domain into the shared mem buffer
    if (i >= 0 && i < sizes.x)
        if (j >= 0 && j < sizes.y)
            if (k >= 0 && k < sizes.z) {
                smem[index_smem2(ii, jj, kk)] =
                    __ldg(&n[index_strides(i, j, k, strides)]);
            }

    __syncthreads();
    if (i > 0 && i < sizes.x - 1)
        if (j > 0 && j < sizes.y - 1)
            if (k > 0 && k < sizes.z - 1)
                if (ii > 0 && ii < blockDim.x - 1)
                    if (jj > 0 && jj < blockDim.y - 1)
                        if (kk > 0 && kk < blockDim.z - 1) {
                            d[index_strides(i, j, k, strides)] =
                                1. / 2. *
                                (                                       //
                                    smem[index_smem2(ii - 1, jj, kk)]   //
                                    + smem[index_smem2(ii + 1, jj, kk)] //
                                    + smem[index_smem2(ii, jj - 1, kk)] //
                                    + smem[index_smem2(ii, jj + 1, kk)] //
                                    + smem[index_smem2(ii, jj, kk - 1)] //
                                    + smem[index_smem2(ii, jj, kk + 1)] //
                                    - 6. * smem[index_smem2(ii, jj, kk)]);
                        }
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
                  << -1. * n[index(i, sizes.y / 2, sizes.z / 2, sizes)] /
                         pow(2. * M_PI / sizes.x, 3)
                  << std::endl;
    }
}

float elapsed(cudaEvent_t &start, cudaEvent_t &stop) {
    float result;
    cudaEventElapsedTime(&result, start, stop);
    return result;
}

enum class Variation {
    LDG,
    SHARED_MEM,
    NO_LDG,
    RELATIVE,
    CONST_RESTRICT,
    SHARED_MEM2
};

std::ostream &operator<<(std::ostream &s, Variation const &var) {
    switch (var) {
    case Variation::NO_LDG:
        s << "no optimization,     ";
        break;
    case Variation::LDG:
        s << "__ldg,               ";
        break;
    case Variation::SHARED_MEM:
        s << "shared memory,       ";
        break;
    case Variation::SHARED_MEM2:
        s << "shared memory v2,    ";
        break;
    case Variation::RELATIVE:
        s << "relative indexing,   ";
        break;
    case Variation::CONST_RESTRICT:
        s << "const __restrict__,  ";
        break;
    default:
        s << "n/a";
    }

    return s;
}

/**
 * Warning: one of the stencils is modifying the threadsPerBlock.
 */
template <Variation Var>
void execute(dim3 threadsPerBlock, double *dd, double *dn) {
    const dim3 sizes(Nx, Ny, Nz);
    const dim3 strides(1, Nx, Nx * Ny);

    cudaEvent_t start_;
    cudaEvent_t stop_;
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
    int nBlocksX = Nx / threadsPerBlock.x;
    int nBlocksY = Ny / threadsPerBlock.y;
    int nBlocksZ = Nz / threadsPerBlock.z;

    if (Nx % threadsPerBlock.x != 0) {
        nBlocksX++;
        throw std::runtime_error("there is a bug for non divisible sizes");
    }
    if (Ny % threadsPerBlock.y != 0) {
        nBlocksY++;
        throw std::runtime_error("there is a bug for non divisible sizes");
    }
    if (Nz % threadsPerBlock.z != 0) {
        nBlocksZ++;
        throw std::runtime_error("there is a bug for non divisible sizes");
    }

    dim3 nBlocks(nBlocksX, nBlocksY, nBlocksZ);

    cudaEventRecord(start_, 0);

    for (size_t i = 0; i < nrepeat; ++i) {
        if (Var == Variation::SHARED_MEM) {
            size_t smem_size = (threadsPerBlock.x + 2) *
                               (threadsPerBlock.y + 2) *
                               (threadsPerBlock.z + 2);
            laplace3d_smem<<<nBlocks, threadsPerBlock,
                             smem_size * sizeof(double)>>>(dd, dn, sizes,
                                                           strides);
        }
        if (Var == Variation::SHARED_MEM2) {
            size_t smem_size = (threadsPerBlock.x + 2) *
                               (threadsPerBlock.y + 2) *
                               (threadsPerBlock.z + 2);

            if (smem_size <= 1024) {

                dim3 enlargedBlock(threadsPerBlock.x + 2, threadsPerBlock.y + 2,
                                   threadsPerBlock.z + 2);

                laplace3d_smem2<<<nBlocks, enlargedBlock,
                                  smem_size * sizeof(double)>>>(dd, dn, sizes,
                                                                strides);
            }
        } else if (Var == Variation::LDG)
            laplace3d_ldg<<<nBlocks, threadsPerBlock>>>(dd, dn, sizes, strides);
        else if (Var == Variation::NO_LDG)
            laplace3d_no_ldg<<<nBlocks, threadsPerBlock>>>(dd, dn, sizes,
                                                           strides);
        else if (Var == Variation::CONST_RESTRICT)
            laplace3d_const_restrict<<<nBlocks, threadsPerBlock>>>(
                dd, dn, sizes, strides);
        else if (Var == Variation::RELATIVE)
            laplace3d_relative_indexing<<<nBlocks, threadsPerBlock>>>(
                dd, dn, sizes, strides);
    }
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);

    std::cout << "# Variation: " << Var;
    std::cout << "threads/block = (" << threadsPerBlock.x << "/"
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

    std::vector<dim3> threadsPerBlock;
    //    threadsPerBlock.emplace_back(14, 6, 6);
    threadsPerBlock.emplace_back(32, 4, 4);
    threadsPerBlock.emplace_back(8, 8, 8);
    threadsPerBlock.emplace_back(16, 8, 8);
    threadsPerBlock.emplace_back(16, 4, 4);

    for (auto dim : threadsPerBlock) {
        execute<Variation::NO_LDG>(dim, dd, dn);
        execute<Variation::LDG>(dim, dd, dn);
        execute<Variation::CONST_RESTRICT>(dim, dd, dn);
        execute<Variation::RELATIVE>(dim, dd, dn);
        execute<Variation::SHARED_MEM>(dim, dd, dn);
        execute<Variation::SHARED_MEM2>(dim, dd, dn);
    }

    cudaMemcpy(d, dd, sizeof(double) * total_size, cudaMemcpyDeviceToHost);

    print(d, sizes);

    delete[] d;
    cudaFree(dd);
    cudaFree(dn);
}

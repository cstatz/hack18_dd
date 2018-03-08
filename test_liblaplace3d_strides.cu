#include <iostream>
#include <stdio.h>
#include <math.h>
#include "laplace3d_strides.h"

const size_t Nx = 128;
const size_t Ny = 512;
const size_t Nz = 512;

int index(const int i, const int j, const int k, const dim3 sizes) {
    const int istride = 1;
    const int jstride = sizes.x;
    const int kstride = sizes.x * sizes.y;
    return i * istride + j * jstride + k * kstride;
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

    call_laplace3d_strides(dd, dn, Nx, Ny, Nz);

    cudaMemcpy(d, dd, sizeof(double) * total_size, cudaMemcpyDeviceToHost);

    print(d, sizes);

    delete[] d;
    cudaFree(dd);
    cudaFree(dn);
}

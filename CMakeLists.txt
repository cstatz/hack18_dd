cmake_minimum_required(VERSION 3.8)

project(laplace3d CXX)
enable_language(CUDA)

add_executable(main main.cu)
target_compile_options(main PRIVATE "-arch=sm_60" "--ptxas-options=-v")
set_property(TARGET main PROPERTY CXX_STANDARD 11)
set_property(TARGET main PROPERTY CUDA_STANDARD 11)



cmake_minimum_required(VERSION 3.8)

project(laplace3d CXX)
enable_language(CUDA)

add_executable(main main.cu)
target_compile_options(main PRIVATE "-arch=sm_50" "--ptxas-options=-v")


cmake_minimum_required(VERSION 3.8)

project(laplace3d CXX)
enable_language(CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE
      STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set( CUDA_ARCH "sm_37" CACHE STRING "CUDA compute capability")
set_property(CACHE CUDA_ARCH PROPERTY STRINGS sm_35 sm_37 sm_50 sm_60)

add_executable(main main.cu)
target_compile_options(main PRIVATE "-arch=${CUDA_ARCH}" "--ptxas-options=-v")
set_property(TARGET main PROPERTY CXX_STANDARD 11)
set_property(TARGET main PROPERTY CUDA_STANDARD 11)



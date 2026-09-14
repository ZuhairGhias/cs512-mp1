cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(cs521_mp1 LANGUAGES CUDA CXX)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 75)

# Find CUDA toolkit
find_package(OpenMP)
find_package(CUDAToolkit REQUIRED)
if (OPENMP_FOUND)
    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(OPT_CXX_FLAGS "-O3 -ftree-loop-vectorize -fopt-info-vec-optimized -fopt-info-vec-missed -ffp-contract=fast -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS} ${OPT_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else ()
    message(FATAL_ERROR "Need OpenMP")
endif ()

add_executable(mp1_cpu ../cpu/gemm_cpu.cpp)
add_executable(mp1_gpu ../gpu/gemm_gpu.cu)
#set_target_properties(mp1 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

#target_link_libraries(mp1 CUDA::cudart)
target_link_libraries(mp1_gpu PRIVATE CUDA::cublas CUDA::cublasLt)
cmake_minimum_required(VERSION 3.19)
project(untitled CUDA)

set(CMAKE_CUDA_STANDARD 14)

include(FindCUDAToolkit)

add_executable(untitled main.cu asm_cuda.cu blst_377_ops.cu )

set_target_properties(
        untitled
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(untitled CUDA::curand)
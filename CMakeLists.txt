cmake_minimum_required(VERSION 3.23.1 FATAL_ERROR)

set (CMAKE_CUDA_ARCHITECTURES 70 80)

project(cuda_dict LANGUAGES CXX CUDA)

set (CMAKE_CXX_STANDARD 20)

include_directories(include)

add_executable(mymain src/main.cu)
set_target_properties(mymain PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET mymain PROPERTY CUDA_ARCHITECTURES 70 80)

add_library(cuda_dict src/cuda_dict.cu)
set_target_properties(cuda_dict PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_property(TARGET cuda_dict PROPERTY CUDA_ARCHITECTURES 70 80)


target_link_libraries(mymain cuda_dict)


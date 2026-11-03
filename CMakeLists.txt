cmake_minimum_required(VERSION 3.26)
project(TorchNN_WYT)

set(CMAKE_CXX_STANDARD 17)

add_executable(TorchNN_WYT main.cpp
        Tensor.cpp
        Net.cpp
        FCLayer.cpp
        Graph.cpp)

add_executable(graph_demo graph_demo.cpp
        Tensor.cpp
        Net.cpp
        FCLayer.cpp
        Graph.cpp)

add_executable(bench bench.cpp
        Tensor.cpp
        Net.cpp
        FCLayer.cpp
        Graph.cpp)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    foreach(tgt TorchNN_WYT graph_demo bench)
        target_link_libraries(${tgt} PRIVATE OpenMP::OpenMP_CXX)
    endforeach()
    add_executable(gemm_demo gemm_demo.cpp)
    target_link_libraries(gemm_demo PRIVATE OpenMP::OpenMP_CXX)
endif()

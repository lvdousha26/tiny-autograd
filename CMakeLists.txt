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

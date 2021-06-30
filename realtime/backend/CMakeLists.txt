cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(backend)

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_library(backend STATIC backend.cpp)
target_link_libraries(backend "${TORCH_LIBRARIES}")
set_property(TARGET backend PROPERTY CXX_STANDARD 14)
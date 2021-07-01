cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(realtime-wrapper)

set(CMAKE_OSX_ARCHITECTURES x86_64)

add_subdirectory(backend)
add_subdirectory(frontend/maxmsp/vae_tilde)
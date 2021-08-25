cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(realtime-wrapper)

if (APPLE)
set(CMAKE_OSX_ARCHITECTURES x86_64) # UNTIL M1 SUPPORTER BY MAX
endif()

add_subdirectory(backend) # DEEP LEARNING BACKEND

if(APPLE)
add_subdirectory(frontend/maxmsp/vae_tilde) # MAX MSP EXTERNAL
endif()
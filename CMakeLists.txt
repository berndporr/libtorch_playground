cmake_minimum_required(VERSION 3.11.0)

project(cldl VERSION 3.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Torch REQUIRED)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "Debug")
endif()

include(GNUInstallDirs)

add_executable(playground
  playground.cpp
  )

target_link_libraries(playground
  ${TORCH_LIBRARIES}
  )


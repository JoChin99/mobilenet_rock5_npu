cmake_minimum_required(VERSION 3.20.0)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

project(transfer_mobilenet_v2 LANGUAGES CXX)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(RKNN_INCLUDE_DIR
    /home/ning/RKSDK/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/include
)

set(RKNN_RUNTIME_LIB
    /usr/bin/librknnrt.so
)

add_compile_options(-Wall)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Libtorch version: ${TORCH_VERSION}")
message(STATUS "Libtorch include path: ${TORCH_INCLUDE_DIRS}")

#find_library(RKNN_LIB rknnrt REQUIRED)
#find_path(RKNN_INCLUDE_DIR NAMES rknn_api.h REQUIRED)
#find_library(RKNN_RUNTIME_LIB NAMES rknnrt librknnrt REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${RKNN_INCLUDE_DIR})

message(STATUS "RKNN include path: ${RKNN_INCLUDE_DIR}")
message(STATUS "RKNN runtime library: ${RKNN_RUNTIME_LIB}")

add_executable(transfer transfer.cpp)
target_link_libraries(transfer
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${RKNN_RUNTIME_LIB}
)

add_executable(demo realtime_demo.cpp)
target_link_libraries(demo
    ${TORCH_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${RKNN_RUNTIME_LIB}
)
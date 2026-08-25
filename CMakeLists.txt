cmake_minimum_required(VERSION 3.20.0)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE "RelWithDebInfo")
endif()

project(transfer_mobilenet_v2 LANGUAGES CXX)
include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(ONNX_RUNTIME_DIR ${CMAKE_CURRENT_LIST_DIR}/onnxruntime)
set(RKNN_RUNTIME_DIR ${CMAKE_CURRENT_LIST_DIR}/rknpu/rknn-toolkit2/rknpu2/runtime/Linux/librknn_api)

add_compile_options(-Wall)

find_package(Torch REQUIRED)
find_package(OpenCV REQUIRED)
find_path(ONNX_INCLUDE_DIR 
    NAMES onnxruntime_cxx_api.h
    PATHS ${ONNX_RUNTIME_DIR}/include
)
find_library(ONNX_RUNTIME_LIB 
    NAMES onnxruntime
    PATHS ${ONNX_RUNTIME_DIR}/lib    
)
find_path(RKNN_INCLUDE_DIR 
    NAMES rknn_api.h 
    PATHS ${RKNN_RUNTIME_DIR}/include
)
find_library(RKNN_RUNTIME_LIB 
    NAMES rknnrt
    PATHS ${RKNN_RUNTIME_DIR}/aarch64
)

include_directories(${PROJECT_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${ONNX_INCLUDE_DIR})
include_directories(${RKNN_INCLUDE_DIR})

message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "Libtorch version: ${TORCH_VERSION}")
message(STATUS "Libtorch include path: ${TORCH_INCLUDE_DIRS}")
message(STATUS "ONNX include path: ${ONNX_INCLUDE_DIR}")
message(STATUS "ONNX runtime library: ${ONNX_RUNTIME_LIB}")
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
    ${ONNX_RUNTIME_LIB}
)
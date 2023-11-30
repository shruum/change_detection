cmake_minimum_required(VERSION 3.5)
project(cabinet_cpp)

set(CMAKE_CXX_STANDARD 14)

list(APPEND CMAKE_PREFIX_PATH "/usr/local/libtorch")

find_package(Torch REQUIRED)
list(APPEND LIBRARIES ${TORCH_LIBRARIES})

find_package(OpenCV REQUIRED)
list(APPEND LIBRARIES ${OpenCV_LIBS})

find_package(nlohmann_json 3.2.0 REQUIRED)
list(APPEND LIBRARIES nlohmann_json::nlohmann_json)

set(WITH_NVTX "FALSE" CACHE BOOL "Build with NVidia Tools Extension")

if (WITH_NVTX)
    message(STATUS "Using Nvidia Tools Extension")
    add_definitions(-DUSE_NVTX)
    find_library(NVTX_LIBRARY nvToolsExt HINTS /usr/local/cuda/lib64)
    list(APPEND LIBRARIES ${NVTX_LIBRARY})
endif ()

# `Boost and Gflags -can't be used because LibTorch is compiled with old C++11 ABI.
# https://discuss.pytorch.org/t/issues-linking-with-libtorch-c-11-abi/29510/2
#
#find_package(Boost
#        COMPONENTS
#        filesystem
#        program_options)
#list(APPEND LIBRARIES ${Boost_LIBRARIES})
#list(APPEND INCLUDES ${Boost_INCLUDE_DIR})
#add_definitions(-DUSE_BOOST)
#
#find_package(Gflags REQUIRED)
#list(APPEND LIBRARIES ${GFLAGS_LIBRARIES})

add_executable(${PROJECT_NAME}
        cabinet.cpp
        dataset/mapillary.cpp dataset/mapillary.h
        util/json.cpp util/json.h
        util/nvtx.h
        util/util.cpp util/util.h
        )
target_link_libraries(${PROJECT_NAME} ${LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDES})

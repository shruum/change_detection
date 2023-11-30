cmake_minimum_required(VERSION 3.5)

include("${CMAKE_CURRENT_LIST_DIR}/../builder/cmake/build_main.cmake")

include("${NIE_DIR}/nie_libraries/common/libs/nie_trt_plugins/CMakeLists.txt")
include("${NIE_DIR}/nie_libraries/common/libs/nie_annotation/CMakeLists.txt")
include("${NIE_DIR}/nie_libraries/common/apps/trt/CMakeLists.txt")

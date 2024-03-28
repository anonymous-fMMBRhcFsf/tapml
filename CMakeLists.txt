cmake_minimum_required(VERSION 3.18)
project(tapml C CXX)

include(CheckCXXCompilerFlag)
if(MSVC)
  set(CMAKE_CXX_FLAGS "/fp:fast ${CMAKE_CXX_FLAGS}")
else()
  set(CMAKE_CXX_FLAGS "-ffast-math ${CMAKE_CXX_FLAGS}")
endif()

if(EXISTS ${CMAKE_BINARY_DIR}/config.cmake)
  include(${CMAKE_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_SOURCE_DIR}/config.cmake)
    include(${CMAKE_SOURCE_DIR}/config.cmake)
  endif()
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(
    CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE
  )
  message(STATUS "Setting default build type to " ${CMAKE_BUILD_TYPE})
endif(NOT CMAKE_BUILD_TYPE)

option(TAPML_HIDE_PRIVATE_SYMBOLS "Hide private symbols" ON)

if (TAPML_INSTALL_STATIC_LIB)
  set(BUILD_STATIC_RUNTIME ON)
endif()

set(TAPML_VISIBILITY_FLAG "")
if (TAPML_HIDE_PRIVATE_SYMBOLS)
  set(HIDE_PRIVATE_SYMBOLS ON)
  if (NOT MSVC)
    set(TAPML_VISIBILITY_FLAG "-fvisibility=hidden")
  endif()
  message(STATUS "Hide private symbols")
endif()

option(BUILD_CPP_TEST "Build cpp unittests" OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# tvm runtime config: minimize runtime components
set(USE_RPC OFF)
set(USE_MICRO OFF)
set(USE_GRAPH_EXECUTOR OFF)
set(USE_GRAPH_EXECUTOR_DEBUG OFF)
set(USE_AOT_EXECUTOR OFF)
set(USE_PROFILER OFF)
set(USE_GTEST OFF)
set(USE_LIBBACKTRACE OFF)
set(BUILD_DUMMY_LIBTVM ON)
if (NOT DEFINED TVM_HOME)
  set(TVM_HOME 3rdparty/tvm)
endif (NOT DEFINED TVM_HOME)
message(STATUS "TVM_HOME: ${TVM_HOME}")
add_subdirectory(${TVM_HOME} tvm EXCLUDE_FROM_ALL)

set(TAPML_RUNTIME_LINKER_LIB "")
set(TOKENZIER_CPP_PATH 3rdparty/tokenizers-cpp)
add_subdirectory(${TOKENZIER_CPP_PATH} tokenizers EXCLUDE_FROM_ALL)


tvm_file_glob(GLOB_RECURSE TAPML_SRCS cpp/*.cc)
add_library(tapml_objs OBJECT ${TAPML_SRCS})

set(
  TAPML_INCLUDES
  ${TVM_HOME}/include
  ${TVM_HOME}/3rdparty/dlpack/include
  ${TVM_HOME}/3rdparty/dmlc-core/include
  ${TVM_HOME}/3rdparty/picojson
)

set(TAPML_COMPILE_DEFS ${TAPML_COMPILE_DEFS} DMLC_USE_LOGGING_LIBRARY=<tvm/runtime/logging.h>)
set(TAPML_COMPILE_DEFS ${TAPML_COMPILE_DEFS} __STDC_FORMAT_MACROS=1)
set(TAPML_COMPILE_DEFS ${TAPML_COMPILE_DEFS} PICOJSON_USE_INT64)

target_include_directories(tapml_objs PRIVATE ${TAPML_INCLUDES})
target_compile_definitions(tapml_objs PRIVATE ${TAPML_COMPILE_DEFS})
target_include_directories(tapml_objs PRIVATE ${TOKENZIER_CPP_PATH}/include)
target_compile_definitions(tapml_objs PRIVATE -DTAPML_EXPORTS)

add_library(tapml SHARED $<TARGET_OBJECTS:tapml_objs>)
add_library(tapml_static STATIC $<TARGET_OBJECTS:tapml_objs>)
add_dependencies(tapml_static tokenizers_cpp sentencepiece-static tokenizers_c tvm_runtime)
set_target_properties(tapml_static PROPERTIES OUTPUT_NAME tapml)

target_link_libraries(tapml PUBLIC tvm_runtime)
target_link_libraries(tapml PRIVATE tokenizers_cpp)

find_library(FLASH_ATTN_LIBRARY flash_attn)

if (FLASH_ATTN_LIBRARY STREQUAL "FLASH_ATTN_LIBRARY-NOTFOUND")
  message(WARNING "Cannot find libflash_attn. The model must not have been built with --use-flash-attn-mqa option.")
else ()
  target_link_libraries(tapml PUBLIC -Wl,--no-as-needed ${FLASH_ATTN_LIBRARY})
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(tapml PRIVATE "TVM_LOG_DEBUG")
    target_compile_definitions(tapml_objs PRIVATE "TVM_LOG_DEBUG")
    target_compile_definitions(tapml_static PRIVATE "TVM_LOG_DEBUG")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL "Android")
  target_link_libraries(tapml PRIVATE log)
  target_link_libraries(tokenizers_cpp PRIVATE log)
endif()

add_library(tapml_module SHARED $<TARGET_OBJECTS:tapml_objs>)
target_link_libraries(tapml_module PUBLIC tvm)
target_link_libraries(tapml_module PRIVATE tokenizers_cpp)


set_property(TARGET tapml_module APPEND PROPERTY LINK_OPTIONS "${TAPML_VISIBILITY_FLAG}")
set_property(TARGET tapml APPEND PROPERTY LINK_OPTIONS "${TAPML_VISIBILITY_FLAG}")

find_program(CARGO_EXECUTABLE cargo)

if(NOT CARGO_EXECUTABLE)
    message(FATAL_ERROR "Cargo is not found! Please install cargo.")
endif()

# when this option is on,
# we install all static lib deps into lib
if (TAPML_INSTALL_STATIC_LIB)
  install(TARGETS
    tapml_static
    tokenizers_cpp
    sentencepiece-static
    tvm_runtime
    LIBRARY DESTINATION lib${LIB_SUFFIX}
    )
  # tokenizers need special handling as it builds from rust
  if(MSVC)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/tokenizers/libtokenizers_c.lib
      DESTINATION lib${LIB_SUFFIX}
      )
  else()
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/tokenizers/libtokenizers_c.a
      DESTINATION lib${LIB_SUFFIX}
      )
  endif()
else()
  install(TARGETS tvm_runtime tapml tapml_module
    tapml_static
    tokenizers_cpp
    sentencepiece-static
    RUNTIME_DEPENDENCY_SET tokenizers_c
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib${LIB_SUFFIX}
  )
endif()

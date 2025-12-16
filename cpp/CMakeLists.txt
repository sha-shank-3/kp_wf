cmake_minimum_required(VERSION 3.15)
project(FinanceAAD LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# XAD paths
set(XAD_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../xad/src")
set(XAD_BUILD_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../xad/build/src")
set(XAD_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../xad/build/lib/Release")

# Find XAD library
find_library(XAD_LIBRARY 
    NAMES xad64_vc144_md xad
    PATHS ${XAD_LIB_DIR}
    NO_DEFAULT_PATH
)

message(STATUS "XAD Library: ${XAD_LIBRARY}")

# European Option Greeks
add_executable(EuropeanOptionGreeks european_option_greeks.cpp)
target_include_directories(EuropeanOptionGreeks PRIVATE ${XAD_INCLUDE_DIR} ${XAD_BUILD_INCLUDE_DIR})
target_link_libraries(EuropeanOptionGreeks PRIVATE ${XAD_LIBRARY})

# Add subdirectory for Hull-White
add_subdirectory(HullWhiteSwaption)

# Enable optimizations for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(EuropeanOptionGreeks PRIVATE /O2 /fp:fast)
    else()
        target_compile_options(EuropeanOptionGreeks PRIVATE -O3 -ffast-math)
    endif()
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "XAD include dir: ${XAD_INCLUDE_DIR}")

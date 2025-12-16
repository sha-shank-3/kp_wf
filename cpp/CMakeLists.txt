cmake_minimum_required(VERSION 3.15)
project(EuropeanOptionGreeks LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find XAD library
find_package(XAD REQUIRED)

add_executable(EuropeanOptionGreeks european_option_greeks.cpp)
target_link_libraries(EuropeanOptionGreeks PRIVATE XAD::xad)

# Enable optimizations for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(EuropeanOptionGreeks PRIVATE /O2 /fp:fast)
    else()
        target_compile_options(EuropeanOptionGreeks PRIVATE -O3 -ffast-math)
    endif()
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "XAD found - European Option Greeks will use AAD")

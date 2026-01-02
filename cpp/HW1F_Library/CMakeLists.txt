cmake_minimum_required(VERSION 3.15)
project(HW1F_Library VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(MSVC)
    add_compile_options(/W4 /O2 /arch:AVX2)
else()
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

# XAD library paths (adjust as needed)
set(XAD_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../xad" CACHE PATH "XAD root directory")
set(XAD_INCLUDE_DIR "${XAD_ROOT}/src")
set(XAD_BUILD_INCLUDE_DIR "${XAD_ROOT}/build/src")
set(XAD_LIB_DIR "${XAD_ROOT}/build/lib/Release")

find_library(XAD_LIBRARY 
    NAMES xad64_vc144_md xad
    PATHS ${XAD_LIB_DIR}
    NO_DEFAULT_PATH
)

if(NOT XAD_LIBRARY)
    message(WARNING "XAD library not found at ${XAD_LIB_DIR}, building without XAD support")
endif()

message(STATUS "XAD Library: ${XAD_LIBRARY}")

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${XAD_INCLUDE_DIR}
    ${XAD_BUILD_INCLUDE_DIR}
)

# Header-only library (most code is templated)
add_library(hw1f_lib INTERFACE)
target_include_directories(hw1f_lib INTERFACE 
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${XAD_INCLUDE_DIR}
    ${XAD_BUILD_INCLUDE_DIR}
)
target_compile_features(hw1f_lib INTERFACE cxx_std_17)

# Main application
add_executable(run_hw1f apps/run_hw1f.cpp)
target_link_libraries(run_hw1f PRIVATE hw1f_lib)
if(XAD_LIBRARY)
    target_link_libraries(run_hw1f PRIVATE ${XAD_LIBRARY})
endif()

# Test executable
add_executable(hw1f_tests tests/test_main.cpp)
target_link_libraries(hw1f_tests PRIVATE hw1f_lib)
if(XAD_LIBRARY)
    target_link_libraries(hw1f_tests PRIVATE ${XAD_LIBRARY})
endif()

# Greeks comparison app
add_executable(greeks_comparison apps/greeks_comparison.cpp)
target_link_libraries(greeks_comparison PRIVATE hw1f_lib)
if(XAD_LIBRARY)
    target_link_libraries(greeks_comparison PRIVATE ${XAD_LIBRARY})
endif()

# Greeks comparison app - Least-Squares Calibration (over-determined)
add_executable(greeks_comparison_lsq apps/greeks_comparison_lsq.cpp)
target_link_libraries(greeks_comparison_lsq PRIVATE hw1f_lib)
if(XAD_LIBRARY)
    target_link_libraries(greeks_comparison_lsq PRIVATE ${XAD_LIBRARY})
endif()

# Multi-swaption comparison app (5 swaptions x exact/LSQ x 4 methods)
add_executable(multi_swaption_comparison apps/multi_swaption_comparison.cpp)
target_link_libraries(multi_swaption_comparison PRIVATE hw1f_lib)
if(XAD_LIBRARY)
    target_link_libraries(multi_swaption_comparison PRIVATE ${XAD_LIBRARY})
endif()

# Swaption comparison app (Analytical vs MC vs MC+XAD for pricing, Analytical vs FD vs XAD+IFT for Greeks)
add_executable(swaption_comparison apps/swaption_comparison.cpp)
target_link_libraries(swaption_comparison PRIVATE hw1f_lib)
if(XAD_LIBRARY)
    target_link_libraries(swaption_comparison PRIVATE ${XAD_LIBRARY})
endif()

# ========================================
# Refactored Components (Production Quality)
# ========================================

# Greeks validation app (comprehensive comparison)
add_executable(greeks_validation apps/greeks_validation.cpp)
target_link_libraries(greeks_validation PRIVATE hw1f_lib)
if(XAD_LIBRARY)
    target_link_libraries(greeks_validation PRIVATE ${XAD_LIBRARY})
endif()

# ========================================
# Unit Tests (Refactored Components)
# ========================================

# Test dimensions and shapes
add_executable(test_dimensions tests/test_dimensions.cpp)
target_link_libraries(test_dimensions PRIVATE hw1f_lib)
if(XAD_LIBRARY)
    target_link_libraries(test_dimensions PRIVATE ${XAD_LIBRARY})
endif()

# Test V_r bucket splitting
add_executable(test_vr_buckets tests/test_vr_buckets.cpp)
target_link_libraries(test_vr_buckets PRIVATE hw1f_lib)
if(XAD_LIBRARY)
    target_link_libraries(test_vr_buckets PRIVATE ${XAD_LIBRARY})
endif()

# Test IFT vs FD
add_executable(test_ift_vs_fd tests/test_ift_vs_fd.cpp)
target_link_libraries(test_ift_vs_fd PRIVATE hw1f_lib)
if(XAD_LIBRARY)
    target_link_libraries(test_ift_vs_fd PRIVATE ${XAD_LIBRARY})
endif()

# CTest integration
enable_testing()
add_test(NAME TestDimensions COMMAND test_dimensions)
add_test(NAME TestVrBuckets COMMAND test_vr_buckets)
add_test(NAME TestIFTvsFD COMMAND test_ift_vs_fd)

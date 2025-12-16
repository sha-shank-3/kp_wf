cmake_minimum_required(VERSION 3.15)

# XAD paths
set(XAD_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../xad/src")
set(XAD_BUILD_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../xad/build/src")
set(XAD_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../xad/build/lib/Release")

# Find XAD library
find_library(XAD_LIBRARY 
    NAMES xad64_vc144_md xad
    PATHS ${XAD_LIB_DIR}
    NO_DEFAULT_PATH
)

message(STATUS "XAD Library: ${XAD_LIBRARY}")

# Original simple version
add_executable(HullWhiteSwaption main.cpp)
target_include_directories(HullWhiteSwaption PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${XAD_INCLUDE_DIR} ${XAD_BUILD_INCLUDE_DIR})
target_compile_features(HullWhiteSwaption PRIVATE cxx_std_17)
target_link_libraries(HullWhiteSwaption PRIVATE ${XAD_LIBRARY})

# Enhanced version with vol surface calibration
add_executable(HullWhiteSwaptionEnhanced main_enhanced.cpp)
target_include_directories(HullWhiteSwaptionEnhanced PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${XAD_INCLUDE_DIR} ${XAD_BUILD_INCLUDE_DIR})
target_compile_features(HullWhiteSwaptionEnhanced PRIVATE cxx_std_17)
target_link_libraries(HullWhiteSwaptionEnhanced PRIVATE ${XAD_LIBRARY})

# Jamshidian version with price-based calibration
add_executable(HullWhiteJamshidian main_jamshidian.cpp)
target_include_directories(HullWhiteJamshidian PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${XAD_INCLUDE_DIR} ${XAD_BUILD_INCLUDE_DIR})
target_compile_features(HullWhiteJamshidian PRIVATE cxx_std_17)
target_link_libraries(HullWhiteJamshidian PRIVATE ${XAD_LIBRARY})

# Enable optimizations for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(HullWhiteSwaption PRIVATE /O2 /fp:fast)
        target_compile_options(HullWhiteSwaptionEnhanced PRIVATE /O2 /fp:fast)
        target_compile_options(HullWhiteJamshidian PRIVATE /O2 /fp:fast)
    else()
        target_compile_options(HullWhiteSwaption PRIVATE -O3 -ffast-math)
        target_compile_options(HullWhiteSwaptionEnhanced PRIVATE -O3 -ffast-math)
        target_compile_options(HullWhiteJamshidian PRIVATE -O3 -ffast-math)
    endif()
endif()

cmake_minimum_required(VERSION 3.15)

# Original simple version
add_executable(HullWhiteSwaption main.cpp)
target_link_libraries(HullWhiteSwaption PRIVATE XAD::xad)
target_include_directories(HullWhiteSwaption PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(HullWhiteSwaption PRIVATE cxx_std_17)

# Enhanced version with vol surface calibration
add_executable(HullWhiteSwaptionEnhanced main_enhanced.cpp)
target_link_libraries(HullWhiteSwaptionEnhanced PRIVATE XAD::xad)
target_include_directories(HullWhiteSwaptionEnhanced PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(HullWhiteSwaptionEnhanced PRIVATE cxx_std_17)

# Enable optimizations for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    if(MSVC)
        target_compile_options(HullWhiteSwaption PRIVATE /O2 /fp:fast)
        target_compile_options(HullWhiteSwaptionEnhanced PRIVATE /O2 /fp:fast)
    else()
        target_compile_options(HullWhiteSwaption PRIVATE -O3 -ffast-math)
        target_compile_options(HullWhiteSwaptionEnhanced PRIVATE -O3 -ffast-math)
    endif()
endif()

cmake_minimum_required(VERSION 3.10)
project(CorrelationFunctionCalculator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(ROOT REQUIRED COMPONENTS Core Graf)
find_package(GSL REQUIRED)

# CATS library paths
set(CATS_BASE_DIR "/Users/glucia/Projects/CATS/DLM_glucia/install")
set(CATS_LIB_DIR "${CATS_BASE_DIR}/CMake")  # Library is directly in CMake directory
set(CATS_INCLUDE_DIR "${CATS_BASE_DIR}/include")

# Find CATS library
find_library(CATS_LIB 
    NAMES CATS libCATS.dylib
    PATHS ${CATS_LIB_DIR} 
    NO_DEFAULT_PATH
)

# Check if library was found
if(NOT CATS_LIB)
    message(FATAL_ERROR "CATS library not found in ${CATS_LIB_DIR}. "
                        "Please check the directory contents with: ls ${CATS_LIB_DIR}")
else()
    message(STATUS "Found CATS library: ${CATS_LIB}")
endif()

# FLINT library paths (adjust if installed in a different location)
# Common locations: /usr/local, /opt/homebrew (macOS), /usr
find_library(FLINT_LIB flint 
    PATHS 
    /usr/local/lib
    /opt/homebrew/lib
    /usr/lib
    REQUIRED
)

find_path(FLINT_INCLUDE_DIR flint/flint.h
    PATHS
    /usr/local/include
    /opt/homebrew/include
    /usr/include
    REQUIRED
)

# GMP library (required by FLINT)
find_library(GMP_LIB gmp
    PATHS
    /usr/local/lib
    /opt/homebrew/lib
    /usr/lib
    REQUIRED
)

# Print found libraries for debugging
message(STATUS "CATS include: ${CATS_INCLUDE_DIR}")
message(STATUS "FLINT library: ${FLINT_LIB}")
message(STATUS "FLINT include: ${FLINT_INCLUDE_DIR}")
message(STATUS "GMP library: ${GMP_LIB}")

# Include directories
include_directories(
    ${ROOT_INCLUDE_DIRS}
    ${GSL_INCLUDE_DIRS}
    ${CATS_INCLUDE_DIR}
    ${FLINT_INCLUDE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Source files
set(SOURCES
    Config.cpp
    CoulombWaveFunction.cpp
    CorrelationCalculator.cpp
    main.cpp
)

# Executable
add_executable(correlation_calc ${SOURCES})

# Link libraries
target_link_libraries(correlation_calc
    ${ROOT_LIBRARIES}
    ${GSL_LIBRARIES}
    ${FLINT_LIB}
    ${GMP_LIB}
    ${CATS_LIB}
)

# Compiler flags
target_compile_options(correlation_calc PRIVATE
    -Wall
    -Wextra
    -O3
)

# Installation
install(TARGETS correlation_calc DESTINATION bin)
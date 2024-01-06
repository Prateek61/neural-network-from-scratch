# CMakeLists.txt for NeuralNetwork Library

cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork LANGUAGES CXX)

# Set the policy CMP0135 to use the new behavior for FetchContent
cmake_policy(SET CMP0135 NEW)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)

# Set Source Directory
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/NeuralNetwork)

# Set Include Directory
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Set directory where include files are inside the include directory
set(INCLUDE_DIR_INCLUDES ${INCLUDE_DIR}/NeuralNetwork)

# Set Source Files
set(SOURCE_FILES
    ${SOURCE_DIR}/AlignedMemoryAllocator.cpp
    ${SOURCE_DIR}/Matrix.cpp
    ${SOURCE_DIR}/Layer.cpp
    ${SOURCE_DIR}/ActivationFunction.cpp
    ${SOURCE_DIR}/NeuralNetwork.cpp
    ${SOURCE_DIR}/DataSet.cpp
)

# Set Include Files
set(INCLUDE_FILES
    ${INCLUDE_DIR_INCLUDES}/AlignedMemoryAllocator.h
    ${INCLUDE_DIR_INCLUDES}/Matrix.h
    ${INCLUDE_DIR_INCLUDES}/Layer.h
    ${INCLUDE_DIR_INCLUDES}/ActivationFunction.h
    ${INCLUDE_DIR_INCLUDES}/NeuralNetwork.h
    ${INCLUDE_DIR_INCLUDES}/DataSet.h
)

# If the compiler is MSVC
if(MSVC)
    # Enable AVX2 Intrinsics
    add_compile_options(/arch:AVX2)
else()
    # Enable Intrinsics if supported
    include(CheckCXXCompilerFlag)
    CHECK_CXX_COMPILER_FLAG("-mavx2" COMPILER_SUPPORTS_AVX2)
    CHECK_CXX_COMPILER_FLAG("-mfma" COMPILER_SUPPORTS_FMA)

    if(COMPILER_SUPPORTS_AVX2 AND COMPILER_SUPPORTS_FMA)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    endif()
endif()

# Create the Library
add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES} ${INCLUDE_FILES})

# Add Include Directory
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIR})

# Check if USE_MKL is enabled
option(USE_MKL "Build the Intel Maths Kernal Library" OFF)

# If Windows and Intel MKL is enabled
if(WIN32 AND USE_MKL)
    # Link Intel MKL
    # Set the MKLROOT environment variable to the root of the MKL installation
    set(MKLROOT "C:/Program Files (x86)/Intel/oneApi/mkl/latest")
    set(MKL_INCLUDE_DIR "${MKLROOT}/include")
    set(MKL_LIB_DIR "${MKLROOT}/lib")
    set(MKL_LIBRARIES
        "${MKL_LIB_DIR}/mkl_core.lib"
        "${MKL_LIB_DIR}/mkl_intel_lp64.lib"
        "${MKL_LIB_DIR}/mkl_sequential.lib"
    )

    include_directories(${MKL_INCLUDE_DIR})
    link_directories(${MKL_LIB_DIR})

    # Link Intel MKL
    link_libraries(${MKL_LIBRARIES})
endif()

# If the compiler is mingw
if(MINGW)
    # If build type is not set, set to release
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE Release)
    endif()

    # If build type is release
    if(CMAKE_BUILD_TYPE STREQUAL "Release")
        # Set the flags for release
        set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
    endif()

    # If build type is debug
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        # Set debug flags
        set(CMAKE_CXX_FLAGS_DEBUG "-g -Wall -Wextra -pedantic -O0")
    endif()
endif()

# Check if BUILD_TESTS is enabled
option(BUILD_TESTS "Build the tests" OFF)

# Check if BUILD_EXAMPLES is enabled
option(BUILD_EXAMPLE "Build the examples" OFF)

if(BUILD_TESTS)
    # Add Subdirectory for Tests
    add_subdirectory(test)

    # Custom Target for Running the Tests
    add_custom_target(run_tests
        COMMAND NeuralNetworkTests
        DEPENDS NeuralNetworkTests
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "***********************  Running tests  ******************************"
    )
endif()

if(BUILD_EXAMPLE)
    # Add Subdirectory for Example
    add_subdirectory(example)

    # Custom Target for Running the Example
    add_custom_target(run_example
        COMMAND NeuralNetworkExample
        DEPENDS NeuralNetworkExample
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "***********************  Running examples  ***************************"
    )
endif()
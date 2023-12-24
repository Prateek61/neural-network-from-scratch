# CMakeLists.txt for NeuralNetwork Library

cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork LANGUAGES CXX)

# Set the policy CMP0135 to use the new behavior for FetchContent
cmake_policy(SET CMP0135 NEW)

# Set the Build Type to Debug if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug")
endif()

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

# Create the Library
add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES} ${INCLUDE_FILES})

# Add Include Directory
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIR})

# Check if BUILD_TESTS is enabled
option(BUILD_TESTS "Build the tests" OFF)
# Check if BUILD_EXAMPLES is enabled
option(BUILD_EXAMPLE "Build the examples" OFF)

if (BUILD_TESTS)
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

if (BUILD_EXAMPLE)
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
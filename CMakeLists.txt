# CMakeLists.txt for NeuralNetwork

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
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
# Set Include Directory
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Set Source Files
set(SOURCE_FILES
    ${SOURCE_DIR}/main.cpp

    ${SOURCE_DIR}/AlignedMemoryAllocator.cpp
    ${SOURCE_DIR}/Matrix.cpp
    ${SOURCE_DIR}/Layer.cpp
    ${SOURCE_DIR}/ActivationFunction.cpp
    ${SOURCE_DIR}/NeuralNetwork.cpp
)
# Set Include Files
set(INCLUDE_FILES
    ${INCLUDE_DIR}/AlignedMemoryAllocator.h
    ${INCLUDE_DIR}/Matrix.h
    ${INCLUDE_DIR}/Layer.h
    ${INCLUDE_DIR}/ActivationFunction.h
    ${INCLUDE_DIR}/NeuralNetwork.h
)

# Add Executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${INCLUDE_FILES})

# Add Include Directory
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIR})

# TODO: Add Modes (Debug, Release, etc.)

##########################  TESTS  ##########################
# Add Subdirectory for Tests
add_subdirectory(test)

# Custom Target for Running the Program
add_custom_target(run_program
    COMMAND NeuralNetwork
    DEPENDS NeuralNetwork
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "***********************  Running program  ****************************"
)

# Custom Target for Running the Tests
add_custom_target(run_tests
    COMMAND NeuralNetworkTests
    DEPENDS NeuralNetworkTests
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "***********************  Running tests  ******************************"
)

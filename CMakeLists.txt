cmake_minimum_required(VERSION 3.10)
project(NeuralNetwork LANGUAGES CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)

# Set Source Directory
set(SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
# Set Include Directory
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Set Source Files
set(SOURCE_FILES
    ${SOURCE_DIR}/main.cpp
    ${SOURCE_DIR}/matrix.cpp
)
# Set Include Files
set(INCLUDE_FILES
    ${INCLUDE_DIR}/matrix.h
)

# Add Executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${INCLUDE_FILES})

# Add Include Directory
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIR})

# TODO: Add Modes (Debug, Release, etc.)

# File: example/CMakeLists.txt
# Purpose: CMake file for NeuralNetwork example

# Set project name
project(NeuralNetworkExample)

# Set current directory
set(CURRENT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Set source directory
set(SOURCE_DIR ${CURRENT_DIR}/src)
# Set include directory
set(INCLUDE_DIR ${CURRENT_DIR}/include)

# Set Mnist dataset directory as a preprocessor definition
add_definitions(-DMNIST_DATASET_DIR="${MNIST_DATASET_DIR}")

# Set include files
set(INCLUDE_FILES
    ${INCLUDE_DIR}/TrainSet.h
)

# Set source files
set(SOURCE_FILES
    ${SOURCE_DIR}/main.cpp
    ${SOURCE_DIR}/TrainSet.cpp
)

# Add executable target
add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${INCLUDE_FILES})

# Add include directories
target_include_directories(${PROJECT_NAME} PUBLIC ${INCLUDE_DIR})

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE NeuralNetwork)

# After compilation, copy the Mnist dataset to the build directory
set(MNIST_DATASET_DIR ${CMAKE_BINARY_DIR}/dataset)
add_custom_command(
    TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CURRENT_DIR}/dataset
        ${MNIST_DATASET_DIR}
)

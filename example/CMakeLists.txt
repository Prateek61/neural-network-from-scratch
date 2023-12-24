# File: example/CMakeLists.txt
# Purpose: CMake file for NeuralNetwork example

# Set project name
project(NeuralNetworkExample)

# Set current directory as include directory
set(CMAKE_INCLUDE_CURRENT_DIR ON)

# Set current directory
set(CMAKE_CURRENT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# Set include files
set(INCLUDE_FILES
    
)

# Set source files
set(SOURCE_FILES
    ${CMAKE_CURRENT_DIR}/main.cpp
)

# Add executable target
add_executable(${PROJECT_NAME} ${SOURCE_FILES} ${INCLUDE_FILES})

# Link libraries
target_link_libraries(${PROJECT_NAME} PRIVATE NeuralNetwork)

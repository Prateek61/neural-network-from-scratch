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
    ${INCLUDE_DIR}/olcPixelGameEngine.h
    ${INCLUDE_DIR}/GUI.h
    ${INCLUDE_DIR}/Utils.h
)

# Set source files
set(SOURCE_FILES
    ${SOURCE_DIR}/main.cpp
    ${SOURCE_DIR}/TrainSet.cpp
    ${SOURCE_DIR}/GUI.cpp
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

if (UNIX)
# Add configurations for olc::PixelGameEngine
target_link_libraries(${PROJECT_NAME} PRIVATE pthread X11 GL GLU png z)
endif()

# If emscripten is used
if (EMSCRIPTEN)
    # Make emscripten create a html file as entry point
    set_target_properties(${PROJECT_NAME} PROPERTIES SUFFIX ".html")

    target_link_options(${PROJECT_NAME} 
        PRIVATE
        -sUSE_LIBPNG=1
        -sMIN_WEBGL_VERSION=2
        -sMAX_WEBGL_VERSION=2
        -sALLOW_MEMORY_GROWTH=1
        -sNO_DISABLE_EXCEPTION_CATCHING=1
        # --preload-file ${CMAKE_BINARY_DIR}/dataset@/data
        --embed-file ${CMAKE_SOURCE_DIR}/example/dataset@/
    )

    # Add compile flag -O3
    target_compile_options(${PROJECT_NAME} PRIVATE -O3)
endif()


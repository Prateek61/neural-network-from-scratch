# tests/CMakeLists.txt

# Set the current project name
project(NeuralNetworkTests LANGUAGES CXX)

# Set the current source directory
set(TESTS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

# Include FetchContent module to download dependencies
include(FetchContent)
# Fetch Google Test
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        release-1.11.0
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Add NeuralNetwork library
set(TEST_SOURCE_FILES
    ${TESTS_DIRECTORY}/MatrixTest.cpp
    ${TESTS_DIRECTORY}/AlignedMemoryAllocatorTest.cpp
)

# Add executable target
add_executable(${PROJECT_NAME} ${TEST_SOURCE_FILES})

# Link NeuralNetwork library
target_link_libraries(${PROJECT_NAME} PRIVATE NeuralNetwork gtest_main)

# Add test
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

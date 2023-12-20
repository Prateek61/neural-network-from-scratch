# tests/CMakeLists.txt

# Check if BUILD_TESTS is enabled
option(BUILD_TESTS "Build tests" OFF)

if(BUILD_TESTS)
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

    # Set your test files
    set(TEST_SOURCE_FILES
        ${TESTS_DIRECTORY}/MatrixTest.cpp
        ${TESTS_DIRECTORY}/AllignedMemoryAllocatorTest.cpp
    )

    # Add library
    add_library(GTest::GTest INTERFACE IMPORTED)
    target_link_libraries(GTest::GTest INTERFACE gtest_main)

    # Create the test executable
    add_executable(NeuralNetworkTests ${TEST_SOURCE_FILES})

    # Link test executable against gtest & gtest_main
    target_link_libraries(NeuralNetworkTests PRIVATE GTest::GTest)

    add_test(NAME NeuralNetworkTests COMMAND NeuralNetworkTests)
endif()

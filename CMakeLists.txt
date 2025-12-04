cmake_minimum_required(VERSION 3.29)
project(TSP)

set(CMAKE_CXX_STANDARD 14)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Build ANN library from source
file(GLOB ANN_SOURCES "ann_1.1.2/src/*.cpp")
add_library(ANN STATIC ${ANN_SOURCES})
target_include_directories(ANN PUBLIC ${PROJECT_SOURCE_DIR}/ann_1.1.2/include)

add_executable(multifragment multifragment.cpp multifragment.h)

# Link ANN to your executable
target_link_libraries(multifragment ANN)

add_executable(test_deletion test_deletion.cpp)
target_link_libraries(test_deletion ANN)

add_executable(TSP
        random_points.cpp
        random_points.h)

# Test executable
add_executable(tests
        test.cpp
        test.h
        multifragment.cpp
        multifragment.h)

# Define TESTING macro for tests to exclude main() from multifragment.cpp
target_compile_definitions(tests PRIVATE TESTING)

# Link Google Test and ANN library to tests
target_link_libraries(tests GTest::gtest_main ANN)

# Enable testing
enable_testing()
add_test(NAME MultifragmentTests COMMAND tests)

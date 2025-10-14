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

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(ANN STATIC IMPORTED)
set_target_properties(ANN PROPERTIES
        IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/lib/libANN.a
        INTERFACE_INCLUDE_DIRECTORIES ${PROJECT_SOURCE_DIR}/include
)

add_executable(multifragment multifragment.cpp multifragment.h)

# Link ANN to your executable
target_link_libraries(multifragment ANN)

add_executable(TSP
        random_points.cpp
        random_points.h)

# Test executable
add_executable(tests
        test.cpp
        test.h
        multifragment.cpp
        multifragment.h)

# Link Google Test and ANN library to tests
target_link_libraries(tests GTest::gtest_main ANN)

# Enable testing
enable_testing()
add_test(NAME MultifragmentTests COMMAND tests)

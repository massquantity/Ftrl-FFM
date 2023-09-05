cmake_minimum_required(VERSION 3.20)
project(
    Ftrl-FFM
    VERSION 1.0.0
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")  # -fopenmp -pthread
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./bin)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
MESSAGE(STATUS "CMAKE_MODULE_PATH: ${CMAKE_MODULE_PATH}")

add_subdirectory(src)
add_subdirectory(tests)
add_subdirectory(third_party)

# find_package(Threads REQUIRED)
# target_link_libraries(thread Threads::Threads)

enable_testing()

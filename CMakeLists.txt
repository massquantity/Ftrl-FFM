cmake_minimum_required(VERSION 3.20)
project(Ftrl-FFM)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")  # -pthread
# set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./bin)

add_subdirectory(src)

# find_package(Threads REQUIRED)
# target_link_libraries(thread Threads::Threads)

# include_directories(${PROJECT_SOURCE_DIR}/src/include)

cmake_minimum_required(VERSION 3.5.1)
project(Ftrl-LR)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -pthread -O3")
aux_source_directory(. DIR_SRCS)
# set(CMAKE_CXX_STANDARD 11)
# SET(CMAKE_CXX_FLAGS -pthread)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ./bin)
add_executable(lr_train src/lr_train.cpp src/threading/pc_threading.cpp src/utils/utils.cpp)
add_executable(lr_pred src/lr_predict.cpp src/threading/pc_threading.cpp src/utils/utils.cpp)
# add_executable(lr_train ${DIR_SRCS})
# install (TARGETS lr_train DESTINATION ./bin)

find_package(Threads REQUIRED)
target_link_libraries(lr_train Threads::Threads)

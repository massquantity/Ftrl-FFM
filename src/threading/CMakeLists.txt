project(threading)

add_library(${PROJECT_NAME} STATIC pc_task.cpp)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
)

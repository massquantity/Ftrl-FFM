project(task)

add_library(${PROJECT_NAME} STATIC ftrl_online.cpp ftrl_offline.cpp)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        concurrent
        data
        eval
        model
        fmt::fmt
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
)

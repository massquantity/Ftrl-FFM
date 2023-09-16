project(utils)

add_library(${PROJECT_NAME} STATIC cmd_option.cpp)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        fmt::fmt
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
)

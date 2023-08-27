project(eval)

add_library(${PROJECT_NAME} STATIC evaluate.cpp)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        concurrent
        data
        model
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
)

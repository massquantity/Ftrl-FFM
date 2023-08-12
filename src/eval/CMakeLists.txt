project(eval)

add_library(${PROJECT_NAME} STATIC evaluate.cpp)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        data
        model
        threading
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
)

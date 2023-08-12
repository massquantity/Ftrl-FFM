project(task)

add_library(${PROJECT_NAME} STATIC ftrl_task.cpp)

# target_include_directories(${PROJECT_NAME}
#    PUBLIC
#        ${CMAKE_CURRENT_SOURCE_DIR}
# )

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        data
        eval
        model
        threading
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src/include
)

project(train)

add_library(${PROJECT_NAME} STATIC ftrl_trainer.cpp)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        model
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
)

project(train)

add_library(${PROJECT_NAME} STATIC ftrl_trainer.cpp)

# target_include_directories(${PROJECT_NAME}
#    PUBLIC
#        ${CMAKE_CURRENT_SOURCE_DIR}
# )

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        model
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src/include
)

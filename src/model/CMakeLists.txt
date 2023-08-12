project(model)

SET(SRC
    ftrl_model.cpp
    ftrl_offline.cpp
    ftrl_online.cpp
)

add_library(${PROJECT_NAME} STATIC ${SRC})

# target_include_directories(${PROJECT_NAME}
#    PUBLIC
#        ${CMAKE_CURRENT_SOURCE_DIR}
# )

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        data
        eval
        threading
        train
)

target_include_directories(${PROJECT_NAME}
    PUBLIC
        ${CMAKE_SOURCE_DIR}/src/include
)

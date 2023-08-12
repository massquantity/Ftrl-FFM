project(data)

SET(SRC
    parser.cpp
    reader.cpp
)

add_library(${PROJECT_NAME} STATIC ${SRC})

# target_include_directories(${PROJECT_NAME}
#    PUBLIC
#        ${CMAKE_CURRENT_SOURCE_DIR}
# )

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
)

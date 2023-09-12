project(compression)

SET(SRC compress.cpp file_ops.c)

add_library(${PROJECT_NAME} STATIC ${SRC})

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
        ${ZSTD_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME} PRIVATE ${ZSTD_LIBRARY})

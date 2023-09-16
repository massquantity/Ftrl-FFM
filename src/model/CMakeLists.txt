project(model)

SET(SRC
    ftrl_model.cpp
    lr.cpp
    fm.cpp
    ffm.cpp
)

add_library(${PROJECT_NAME} STATIC ${SRC})

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        compression
        fmt::fmt
)

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
)

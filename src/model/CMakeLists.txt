project(model)

SET(SRC
    ftrl_model.cpp
    lr.cpp
    fm.cpp
    ffm.cpp
)

add_library(${PROJECT_NAME} STATIC ${SRC})

target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src/include
)

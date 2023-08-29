add_subdirectory(concurrent)
add_subdirectory(data)
add_subdirectory(eval)
add_subdirectory(model)
add_subdirectory(task)

add_executable(main main.cpp)

set(LINK_LIBS
        concurrent
        data
        eval
        model
        task
)

target_link_libraries(main PRIVATE ${LINK_LIBS})

target_include_directories(main
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

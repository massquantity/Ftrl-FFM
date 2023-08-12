add_subdirectory(data)
add_subdirectory(eval)
add_subdirectory(model)
add_subdirectory(task)
add_subdirectory(threading)
add_subdirectory(train)

add_executable(main main.cpp)

set(LINK_LIBS
        eval
        model
        train
        data
        task
        threading
)

target_link_libraries(main PRIVATE ${LINK_LIBS})

# target_include_directories(main PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_include_directories(main
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

set(TEST_SOURCE_FILES
        test_data.cpp
        test_model.cpp
        test_task.cpp
        test_utils.cpp
)

foreach(test_source_file IN LISTS TEST_SOURCE_FILES)
    get_filename_component(test_name ${test_source_file} NAME_WE)
    MESSAGE(STATUS "Adding `${test_name}`")

    add_executable(${test_name} ${test_source_file})
    target_compile_options(${test_name} PRIVATE -O1)
    target_compile_features(${test_name} PRIVATE cxx_std_17)

    string(REGEX REPLACE ".*_([^.]*)\\..*" "\\1" LINK_LIB ${test_source_file})
    if(NOT ${LINK_LIB} STREQUAL utils)
        target_link_libraries(${test_name} PRIVATE ${LINK_LIB})
    endif()
    target_link_libraries(${test_name} PRIVATE doctest)
    target_include_directories(${test_name} PRIVATE ${CMAKE_SOURCE_DIR}/src/include)

    add_test(NAME ${test_name} COMMAND $<TARGET_FILE:${test_name}>)
endforeach()

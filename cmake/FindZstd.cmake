find_path(ZSTD_INCLUDE_DIR
    NAMES zstd.h
    HINTS ${CMAKE_SOURCE_DIR}/third_party/zstd/lib
)

find_library(ZSTD_LIBRARY
    NAMES zstd
    HINTS ${CMAKE_SOURCE_DIR}/third_party/zstd/build_output/lib
)

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(
    Zstd DEFAULT_MSG
    ZSTD_LIBRARY ZSTD_INCLUDE_DIR
)

mark_as_advanced(ZSTD_INCLUDE_DIR ZSTD_LIBRARY)

if (ZSTD_FOUND)
    message(STATUS "Found Zstd: ${ZSTD_LIBRARY}")
else()
    message(FATAL_ERROR "Zstd library not found")
endif()

cmake_minimum_required(VERSION 3.2.1)
project(moo)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g")
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# One warning from when compiling Eigen headers
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

# if(EIGEN_INC_DIR)
#     include_directories(${EIGEN_INC_DIR})
# endif()
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})

set(SRC MOO.cpp)
set(MOO_LIB moo)
add_library(${MOO_LIB} ${SRC})

set(TEST_SRC test.cpp)
set(TEST_EXE moo_test)
add_executable(${TEST_EXE} ${TEST_SRC})
target_link_libraries(${TEST_EXE} ${MOO_LIB})

set_property(TARGET ${TEST_EXE} ${MOO_LIB} PROPERTY CXX_STANDARD 11)

enable_testing()
add_test(MyTest ${TEST_EXE})

# debug macro
option(MYDEBUG "Debug macro for optimizer" OFF)
if(MYDEBUG)
    message(STATUS "Open debug marcro")
    add_definitions(-DMYDEBUG)
    if(DEFINED RAND_SEED)
        message(STATUS "Rand seed is ${RAND_SEED}")
        add_definitions(-DDEBUG_RAND_SEED=${RAND_SEED})
    else()
        add_definitions(-DDEBUG_RAND_SEED=3)
    endif()
else()
    message(STATUS "Close debug marcro")
    remove_definitions(-DMYDEBUG)
    remove_definitions(-DDEBUG_RAND_SEED)
endif()


message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
install(TARGETS ${MOO_LIB}
    RUNTIME DESTINATION bin
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib)
install(FILES README.md
    DESTINATION doc
    PERMISSIONS OWNER_READ GROUP_READ)
install(FILES MOO.h
    DESTINATION inc
    PERMISSIONS OWNER_READ GROUP_READ)

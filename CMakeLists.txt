cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_CXX_STANDARD 11)
project(ADER_CPP VERSION 1.0.0)

include_directories(include)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")

set(RUN_TEST 0)

set (SOURCES
    src/api.cpp

    src/eigs/system.cpp

    src/grid/boundaries.cpp
    src/grid/indexing.cpp

    src/poly/basis.cpp
    src/poly/evaluations.cpp

    src/scipy/algos/lgmres.cpp
    src/scipy/algos/newton_krylov.cpp

    src/scipy/math/legendre.cpp
    src/scipy/math/polynomials.cpp

    src/solvers/iterator.cpp
    src/solvers/stepper.cpp

    src/solvers/dg/dg.cpp
    src/solvers/dg/dg_matrices.cpp

    src/solvers/fv/fluxes.cpp
    src/solvers/fv/fv.cpp

    src/solvers/weno/weno.cpp
    src/solvers/weno/weno_matrices.cpp)

if (RUN_TEST)

    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fstandalone-debug -std=c++11")
    SET(SOURCES ${SOURCES} "src/test.cpp")

    add_executable(ader_test ${SOURCES})

else()

    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")

    if (UNIX)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fno-math-errno -std=c++11")
    endif(UNIX)

    if (WIN32)
        SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2")
    endif(WIN32)

    add_library(pypde SHARED ${SOURCES})
    set_target_properties(pypde PROPERTIES VERSION ${PROJECT_VERSION})

endif()
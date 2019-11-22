cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_CXX_STANDARD 17)
project(ADER_CPP VERSION 1.0.0 DESCRIPTION "A shared library for solving partial differential equations with the ADER method")

include_directories(include)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
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
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -fstandalone-debug")
    SET(SOURCES ${SOURCES} "tests/test.cpp")
    add_executable(ader_test ${SOURCES})
else()
    # -march=native seems slower sometimes
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -DNDEBUG -fno-math-errno")
    add_library(ader SHARED ${SOURCES})
    set_target_properties(ader PROPERTIES VERSION ${PROJECT_VERSION})
endif()
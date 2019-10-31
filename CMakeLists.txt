cmake_minimum_required(VERSION 3.5.1)
set(CMAKE_CXX_STANDARD 17)
project(ADER_CPP VERSION 1.0.0 DESCRIPTION "A shared library for solving partial differential equations with the ADER method")

include_directories(include)

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DNDEBUG")
SET(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS}")
# -march=native seems slower sometimes
# -ffast-math causes nans in some tests

add_library(ader SHARED 
            src/etc/grid.cpp

            src/scipy/legendre.cpp
            src/scipy/lgmres.cpp
            src/scipy/newton_krylov.cpp
            src/scipy/poly.cpp

            src/solvers/iterator.cpp
            src/solvers/steppers.cpp

            src/solvers/fv/fluxes.cpp
            src/solvers/fv/fv.cpp

            src/solvers/dg/dg.cpp
            src/solvers/dg/dg_matrices.cpp

            src/solvers/poly/basis.cpp
            src/solvers/poly/evaluations.cpp

            src/solvers/weno/weno.cpp
            src/solvers/weno/weno_matrices.cpp
            )

set_target_properties(ader PROPERTIES VERSION ${PROJECT_VERSION})
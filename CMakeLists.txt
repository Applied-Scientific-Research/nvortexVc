#
# nvortexVc
#
PROJECT (nvortexVc)
CMAKE_MINIMUM_REQUIRED(VERSION 3.0)

SET_PROPERTY(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo")
IF (NOT CMAKE_BUILD_TYPE)
  SET (CMAKE_BUILD_TYPE "Release")
ENDIF ()
SET (CMAKE_BUILD_TYPE "Release" CACHE STRING "Select which configuration to build")
SET (USE_OMP FALSE CACHE BOOL "Use OpenMP multithreading")
SET (USE_VC FALSE CACHE BOOL "Use Vc for vector arithmetic")
SET (CMAKE_VERBOSE_MAKEFILE on)
SET (CMAKE_EXPORT_COMPILE_COMMANDS on)

IF( NOT CMAKE_BUILD_TYPE )
  SET( CMAKE_BUILD_TYPE "Release" )
ENDIF()

IF( CMAKE_COMPILER_IS_GNUCXX )
  ADD_COMPILE_OPTIONS( -Wall -Wformat -std=c++14 )
  IF( CMAKE_BUILD_TYPE STREQUAL "Debug" )
    # Debug already includes -O0 -g
    ADD_COMPILE_OPTIONS( -ggdb3 )
  ELSEIF( CMAKE_BUILD_TYPE STREQUAL "Release" )
    # Release already includes -O3
    ADD_COMPILE_OPTIONS( -march=native -ffast-math -Ofast -flto )
  ELSEIF( CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" )
    # RelDbg already includes -O2 -g
    ADD_COMPILE_OPTIONS( -march=native -Ofast -ggdb3 )
  ENDIF()

ELSEIF( CMAKE_CXX_COMPILER_ID STREQUAL "Clang" )
  ADD_COMPILE_OPTIONS( -Wall -Wformat -std=c++14 )
  # The Cray compiler reports as Clang to most versions of CMake
  EXECUTE_PROCESS(COMMAND ${CMAKE_CXX_COMPILER} --version
                  COMMAND grep -c Cray
                  OUTPUT_VARIABLE INTERNAL_HAVE_CRAY_COMPILER
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  IF (INTERNAL_HAVE_CRAY_COMPILER) #not actually Clang
    # cray compiler misidentifies as Clang and needs this instead
    ADD_COMPILE_OPTIONS( -fsave-loopmark -floopmark-style=interspersed )
  ELSE()
    # real clang needs this
    ADD_COMPILE_OPTIONS( -stdlib=libc++ )
  ENDIF()
  IF( CMAKE_BUILD_TYPE STREQUAL "Debug" )
    # Debug already includes -O0 -g
    ADD_COMPILE_OPTIONS( -ggdb )
  ELSEIF( CMAKE_BUILD_TYPE STREQUAL "Release" )
    # Release already includes -O3
    ADD_COMPILE_OPTIONS( -march=native -Ofast -flto )
  ELSEIF( CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo" )
    # RelDbg already includes -O2 -g
    ADD_COMPILE_OPTIONS( -march=native -Ofast -ggdb )
  ENDIF()

ELSEIF( MSVC )
  # needs work
  ADD_COMPILE_OPTIONS( /Ox )
ENDIF()

IF( NOT USE_OMP )
  # silence the warnings
  # this line must appear BEFORE defining the executables
  ADD_COMPILE_OPTIONS( -Wno-unknown-pragmas )
ENDIF()

# if you have headers in src, tell cmake here
#INCLUDE_DIRECTORIES( "src" )

# build these two bins
ADD_EXECUTABLE( "nvortex2d.bin" "src/nvortex2d.cpp" )
ADD_EXECUTABLE( "nvortex3d.bin" "src/nvortex3d.cpp" )

# OpenMP for multithreading
IF( USE_OMP )
  # many ways to do this?
  # https://cmake.org/cmake/help/latest/module/FindOpenMP.html
  # https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html
  # https://stackoverflow.com/questions/17633513/cmake-cannot-find-openmp#28792611
  FIND_PACKAGE( OpenMP REQUIRED )
  TARGET_LINK_LIBRARIES( "nvortex2d.bin" PUBLIC OpenMP::OpenMP_CXX)
  TARGET_LINK_LIBRARIES( "nvortex3d.bin" PUBLIC OpenMP::OpenMP_CXX)
  ADD_COMPILE_OPTIONS( ${OpenMP_CXX_FLAGS} )
ENDIF()  # OpenMP for multithreading

# Vc for explicit vectorization
IF( USE_VC )
  FIND_PACKAGE( Vc REQUIRED )
  ADD_COMPILE_DEFINITIONS( USE_VC )
  INCLUDE_DIRECTORIES( ${Vc_INCLUDE_DIR} )
  # these lines must appear AFTER defining the executables
  TARGET_LINK_LIBRARIES( "nvortex2d.bin" PUBLIC ${Vc_LIBRARIES} )
  TARGET_LINK_LIBRARIES( "nvortex3d.bin" PUBLIC ${Vc_LIBRARIES} )
ENDIF()  # Vc for explicit vectorization



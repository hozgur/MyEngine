# Install script for directory: F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/out/install/x64-Debug")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Debug")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE FILE FILES
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/AdolcForward"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/AlignedVector3"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/ArpackSupport"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/AutoDiff"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/BVH"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/EulerAngles"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/FFT"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/IterativeSolvers"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/KroneckerProduct"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/LevenbergMarquardt"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/MatrixFunctions"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/MoreVectorization"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/MPRealSupport"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/NonLinearOptimization"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/NumericalDiff"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/OpenGLSupport"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/Polynomials"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/Skyline"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/SparseExtra"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/SpecialFunctions"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/Splines"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/unsupported/Eigen" TYPE DIRECTORY FILES "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/unsupported/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/out/build/x64-Debug/unsupported/Eigen/CXX11/cmake_install.cmake")

endif()


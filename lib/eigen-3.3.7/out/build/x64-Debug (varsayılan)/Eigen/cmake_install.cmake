# Install script for directory: F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/out/install/x64-Debug (varsayılan)")
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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen" TYPE FILE FILES
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/Cholesky"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/CholmodSupport"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/Core"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/Dense"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/Eigen"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/Eigenvalues"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/Geometry"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/Householder"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/IterativeLinearSolvers"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/Jacobi"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/LU"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/MetisSupport"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/OrderingMethods"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/PaStiXSupport"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/PardisoSupport"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/QR"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/QtAlignedMalloc"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/SPQRSupport"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/SVD"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/Sparse"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/SparseCholesky"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/SparseCore"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/SparseLU"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/SparseQR"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/StdDeque"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/StdList"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/StdVector"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/SuperLUSupport"
    "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/UmfPackSupport"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xDevelx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/eigen3/Eigen" TYPE DIRECTORY FILES "F:/Repos/Github/hozgur/xxxGameEngine/eigen-3.3.7/Eigen/src" FILES_MATCHING REGEX "/[^/]*\\.h$")
endif()


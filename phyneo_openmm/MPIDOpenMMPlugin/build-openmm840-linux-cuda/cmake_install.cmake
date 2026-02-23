# Install script for directory: /home/am3-peichenzhong-group/Documents/project/PhyNEO/phyneo_openmm/MPIDOpenMMPlugin

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/am3-peichenzhong-group/miniconda3/envs/mpid")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
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

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMPIDPlugin.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMPIDPlugin.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMPIDPlugin.so"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/am3-peichenzhong-group/Documents/project/PhyNEO/phyneo_openmm/MPIDOpenMMPlugin/build-openmm840-linux-cuda/libMPIDPlugin.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMPIDPlugin.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMPIDPlugin.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMPIDPlugin.so"
         OLD_RPATH "/home/am3-peichenzhong-group/miniconda3/envs/mpid/lib:/home/am3-peichenzhong-group/miniconda3/envs/mpid/lib/plugins:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libMPIDPlugin.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/openmm" TYPE FILE FILES
    "/home/am3-peichenzhong-group/Documents/project/PhyNEO/phyneo_openmm/MPIDOpenMMPlugin/openmmapi/include/openmm/MPIDForce.h"
    "/home/am3-peichenzhong-group/Documents/project/PhyNEO/phyneo_openmm/MPIDOpenMMPlugin/openmmapi/include/openmm/mpidKernels.h"
    )
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/openmm/internal" TYPE FILE FILES
    "/home/am3-peichenzhong-group/Documents/project/PhyNEO/phyneo_openmm/MPIDOpenMMPlugin/openmmapi/include/openmm/internal/MPIDForceImpl.h"
    "/home/am3-peichenzhong-group/Documents/project/PhyNEO/phyneo_openmm/MPIDOpenMMPlugin/openmmapi/include/openmm/internal/windowsExportMPID.h"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/am3-peichenzhong-group/Documents/project/PhyNEO/phyneo_openmm/MPIDOpenMMPlugin/build-openmm840-linux-cuda/serialization/tests/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/am3-peichenzhong-group/Documents/project/PhyNEO/phyneo_openmm/MPIDOpenMMPlugin/build-openmm840-linux-cuda/platforms/reference/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/am3-peichenzhong-group/Documents/project/PhyNEO/phyneo_openmm/MPIDOpenMMPlugin/build-openmm840-linux-cuda/platforms/cuda/cmake_install.cmake")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/am3-peichenzhong-group/Documents/project/PhyNEO/phyneo_openmm/MPIDOpenMMPlugin/build-openmm840-linux-cuda/python/cmake_install.cmake")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/am3-peichenzhong-group/Documents/project/PhyNEO/phyneo_openmm/MPIDOpenMMPlugin/build-openmm840-linux-cuda/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")

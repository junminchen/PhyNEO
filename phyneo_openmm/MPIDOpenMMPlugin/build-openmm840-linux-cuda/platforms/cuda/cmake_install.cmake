# Install script for directory: /home/am3-peichenzhong-group/Documents/project/PhyNEO/phyneo_openmm/MPIDOpenMMPlugin/platforms/cuda

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
  if(EXISTS "$ENV{DESTDIR}/home/am3-peichenzhong-group/miniconda3/envs/mpid/lib/plugins/libMPIDPluginCUDA.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/am3-peichenzhong-group/miniconda3/envs/mpid/lib/plugins/libMPIDPluginCUDA.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}/home/am3-peichenzhong-group/miniconda3/envs/mpid/lib/plugins/libMPIDPluginCUDA.so"
         RPATH "")
  endif()
  list(APPEND CMAKE_ABSOLUTE_DESTINATION_FILES
   "/home/am3-peichenzhong-group/miniconda3/envs/mpid/lib/plugins/libMPIDPluginCUDA.so")
  if(CMAKE_WARN_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(WARNING "ABSOLUTE path INSTALL DESTINATION : ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  if(CMAKE_ERROR_ON_ABSOLUTE_INSTALL_DESTINATION)
    message(FATAL_ERROR "ABSOLUTE path INSTALL DESTINATION forbidden (by caller): ${CMAKE_ABSOLUTE_DESTINATION_FILES}")
  endif()
  file(INSTALL DESTINATION "/home/am3-peichenzhong-group/miniconda3/envs/mpid/lib/plugins" TYPE SHARED_LIBRARY FILES "/home/am3-peichenzhong-group/Documents/project/PhyNEO/phyneo_openmm/MPIDOpenMMPlugin/build-openmm840-linux-cuda/platforms/cuda/libMPIDPluginCUDA.so")
  if(EXISTS "$ENV{DESTDIR}/home/am3-peichenzhong-group/miniconda3/envs/mpid/lib/plugins/libMPIDPluginCUDA.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}/home/am3-peichenzhong-group/miniconda3/envs/mpid/lib/plugins/libMPIDPluginCUDA.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}/home/am3-peichenzhong-group/miniconda3/envs/mpid/lib/plugins/libMPIDPluginCUDA.so"
         OLD_RPATH "/home/am3-peichenzhong-group/Documents/project/PhyNEO/phyneo_openmm/MPIDOpenMMPlugin/build-openmm840-linux-cuda:/home/am3-peichenzhong-group/miniconda3/envs/mpid/lib:/home/am3-peichenzhong-group/miniconda3/envs/mpid/lib/plugins:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}/home/am3-peichenzhong-group/miniconda3/envs/mpid/lib/plugins/libMPIDPluginCUDA.so")
    endif()
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
endif()

if(CMAKE_INSTALL_COMPONENT STREQUAL "Unspecified" OR NOT CMAKE_INSTALL_COMPONENT)
  include("/home/am3-peichenzhong-group/Documents/project/PhyNEO/phyneo_openmm/MPIDOpenMMPlugin/build-openmm840-linux-cuda/platforms/cuda/CMakeFiles/MPIDPluginCUDA.dir/install-cxx-module-bmi-Release.cmake" OPTIONAL)
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for the subdirectory.
  include("/home/am3-peichenzhong-group/Documents/project/PhyNEO/phyneo_openmm/MPIDOpenMMPlugin/build-openmm840-linux-cuda/platforms/cuda/tests/cmake_install.cmake")
endif()


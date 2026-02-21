PROJECT(MPIDOpenMMPlugin)
#---------------------------------------------------
# OpenMM MPID Plugin
#----------------------------------------------------

CMAKE_MINIMUM_REQUIRED(VERSION 3.16)

# Prefer modern OpenMMConfig.cmake (OpenMM 8.x), keep OPENMM_DIR fallback for legacy installs.
SET(OPENMM_DIR "" CACHE PATH "Where OpenMM is installed (legacy fallback)")
if(OPENMM_DIR)
    set(_openmm_hints "${OPENMM_DIR}")
else()
    set(_openmm_hints "$ENV{CONDA_PREFIX}" "/usr/local/openmm")
endif()

find_package(OpenMM CONFIG QUIET HINTS ${_openmm_hints} PATH_SUFFIXES lib/cmake/OpenMM)

if(OpenMM_FOUND)
    if(TARGET OpenMM::OpenMM)
        set(OPENMM_TARGET OpenMM::OpenMM)
    elseif(TARGET OpenMM)
        set(OPENMM_TARGET OpenMM)
    endif()
    if(NOT OPENMM_TARGET)
        message(FATAL_ERROR "OpenMM found but no OpenMM target exported.")
    endif()
    get_target_property(_openmm_include_dirs ${OPENMM_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
    if(_openmm_include_dirs)
        include_directories(${_openmm_include_dirs})
    endif()
    if(NOT OPENMM_DIR)
        get_filename_component(OPENMM_DIR "${_openmm_include_dirs}" DIRECTORY)
        get_filename_component(OPENMM_DIR "${OPENMM_DIR}" DIRECTORY)
    endif()
else()
    if(NOT OPENMM_DIR)
        set(OPENMM_DIR "/usr/local/openmm")
    endif()
    include_directories("${OPENMM_DIR}/include")
    link_directories("${OPENMM_DIR}/lib" "${OPENMM_DIR}/lib/plugins")
    set(OPENMM_TARGET OpenMM)
endif()

# set flags for linking on mac
IF(APPLE)
    SET (CMAKE_INSTALL_NAME_DIR "@rpath")
    SET(EXTRA_COMPILE_FLAGS "-stdlib=libc++")
ENDIF(APPLE)

# Select where to install
IF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})
    IF(WIN32)
        SET(CMAKE_INSTALL_PREFIX "$ENV{ProgramFiles}/OpenMM" CACHE PATH "Where to install the plugin" FORCE)
    ELSE(WIN32)
        SET(CMAKE_INSTALL_PREFIX "/usr/local/openmm" CACHE PATH "Where to install the plugin" FORCE)
    ENDIF(WIN32)
ENDIF(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})

# The source is organized into subdirectories, but we handle them all from
# this CMakeLists file rather than letting CMake visit them as SUBDIRS.
SET(MPID_PLUGIN_SOURCE_SUBDIRS openmmapi serialization)

# Set the library name
SET(MPID_LIBRARY_NAME MPIDPlugin)
SET(SHARED_MPID_TARGET ${MPID_LIBRARY_NAME})

# These are all the places to search for header files which are to be part of the API.
SET(API_INCLUDE_DIRS "openmmapi/include" "openmmapi/include/internal")

# Locate header files.
SET(API_INCLUDE_FILES)
FOREACH(dir ${API_INCLUDE_DIRS})
    FILE(GLOB fullpaths ${dir}/*.h)
    SET(API_INCLUDE_FILES ${API_INCLUDE_FILES} ${fullpaths})
ENDFOREACH(dir)

# Collect up source files
SET(SOURCE_FILES) # empty
SET(SOURCE_INCLUDE_FILES)
FOREACH(subdir ${MPID_PLUGIN_SOURCE_SUBDIRS})
    FILE(GLOB src_files  ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp)
    FILE(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    SET(SOURCE_FILES         ${SOURCE_FILES}         ${src_files})   #append
    SET(SOURCE_INCLUDE_FILES ${SOURCE_INCLUDE_FILES} ${incl_files})

    ## Make sure we find these locally before looking in OpenMM/include if
    ## OpenMM was previously installed there.
    INCLUDE_DIRECTORIES(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/include)
ENDFOREACH(subdir)

# Create the library.

ADD_LIBRARY(${SHARED_MPID_TARGET} SHARED ${SOURCE_FILES} ${SOURCE_INCLUDE_FILES} ${API_INCLUDE_FILES})
SET_TARGET_PROPERTIES(${SHARED_MPID_TARGET}
    PROPERTIES COMPILE_FLAGS "-DMPID_BUILDING_SHARED_LIBRARY ${EXTRA_COMPILE_FLAGS}"
    LINK_FLAGS "${EXTRA_COMPILE_FLAGS}")
TARGET_LINK_LIBRARIES(${SHARED_MPID_TARGET} ${OPENMM_TARGET})
install(TARGETS ${SHARED_MPID_TARGET}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)

# install headers
FILE(GLOB API_ONLY_INCLUDE_FILES "openmmapi/include/openmm/*.h")
INSTALL(FILES ${API_ONLY_INCLUDE_FILES} DESTINATION include/openmm)
FILE(GLOB API_ONLY_INCLUDE_FILES_INTERNAL "openmmapi/include/openmm/internal/*.h")
INSTALL(FILES ${API_ONLY_INCLUDE_FILES_INTERNAL} DESTINATION include/openmm/internal)

# Enable testing

ENABLE_TESTING()

set(BUILD_TESTING ON CACHE BOOL "Build tests")
set(OPENMM_BUILD_REFERENCE_TESTS ON CACHE BOOL "Build reference platform tests")
ADD_SUBDIRECTORY(serialization/tests)

# Build the implementations for different platforms

ADD_SUBDIRECTORY(platforms/reference)

SET(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")

OPTION(MPID_BUILD_CUDA_LIB "Build implementation for CUDA" OFF)
IF(MPID_BUILD_CUDA_LIB)
    FIND_PACKAGE(CUDA REQUIRED)
    ADD_SUBDIRECTORY(platforms/cuda)
ENDIF(MPID_BUILD_CUDA_LIB)

# Build the Python API

FIND_PROGRAM(PYTHON_EXECUTABLE python)
FIND_PROGRAM(SWIG_EXECUTABLE swig)
IF(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
    SET(MPID_BUILD_PYTHON_WRAPPERS ON CACHE BOOL "Build wrappers for Python")
ELSE(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
    SET(MPID_BUILD_PYTHON_WRAPPERS OFF CACHE BOOL "Build wrappers for Python")
ENDIF(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
IF(MPID_BUILD_PYTHON_WRAPPERS)
    ADD_SUBDIRECTORY(python)
ENDIF(MPID_BUILD_PYTHON_WRAPPERS)

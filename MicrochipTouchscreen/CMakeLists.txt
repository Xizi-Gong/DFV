cmake_minimum_required (VERSION 3.5...3.29)
cmake_policy(SET CMP0015 NEW)

project (Touchscreen)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)
add_definitions(-DUNICODE -D_UNICODE)


###############################################
# util_functions
###############################################
function(UPDATE_CACHE_VARIABLE VAR_NAME VALUE)	# copied from ceres solver
  get_property(IS_DEFINED_IN_CACHE CACHE ${VAR_NAME} PROPERTY VALUE SET)
  if (NOT IS_DEFINED_IN_CACHE)
    message(FATAL_ERROR "Specified variable to update in cache: "
      "${VAR_NAME} has not been set in the cache.")
  endif()
  get_property(HELP_STRING CACHE ${VAR_NAME} PROPERTY HELPSTRING)
  get_property(VAR_TYPE CACHE ${VAR_NAME} PROPERTY TYPE)
  set(${VAR_NAME} ${VALUE} CACHE ${VAR_TYPE} "${HELP_STRING}" FORCE)
endfunction()

# get subdirectories of a directory
macro(SUBDIRLIST result curdir)
	file(GLOB children RELATIVE ${curdir} ${curdir}/*)
	set(dirlist "")
	foreach(child ${children})
	if(IS_DIRECTORY ${curdir}/${child})
		list(APPEND dirlist ${child})
	endif()
	endforeach()
	set(${result} ${dirlist})
endmacro()


###############################################
# Settings
###############################################
if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
	set(CMAKE_DEBUG_POSTFIX "d")
endif()
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY  ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY  ${PROJECT_SOURCE_DIR}/lib)	 
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY  ${PROJECT_SOURCE_DIR}/bin)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_SOURCE_DIR}/cmake)

###############################################
# Options and config
###############################################
find_package(OpenMP REQUIRED)


###############################################
# Dependency
###############################################

message(STATUS "==================================================")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(BIN_PATH "${CMAKE_SOURCE_DIR}/bin/Release;${CMAKE_SOURCE_DIR}/bin/Debug;${CMAKE_SOURCE_DIR}/3rdParty/bin/win64;${CMAKE_SOURCE_DIR}/3rdParty/bin/win32")


# --------------------
# Already included 3rdParty Library
# --------------------
include_directories(./3rdParty/inc)  
include_directories(./src/common)
include_directories(${CMAKE_BINARY_DIR})

# --------------------
# yzLib as submodule
# --------------------
if(EXISTS ${CMAKE_SOURCE_DIR}/3rdParty/yzLibrary/yzLib)
	include(${CMAKE_SOURCE_DIR}/3rdParty/yzLibrary/yzLib/yzLib.cmake)
	yzLib_set_ENABLE(GLUT ON)
	yzLib_set_ENABLE(GLEW ON)
	yzLib_set_ENABLE(Eigen OFF)
	yzLib_set_ENABLE(WINDOWS ON)
	yzLib_set_ENABLE(FreeImage OFF)
	yzLib_set_ENABLE(CUDA OFF)
	yzLib_Print_Info()
	include_directories(${yzLib_INCLUDE_DIR})
	message(STATUS "yzLib included")
else()
	message(FATAL_ERROR "yzLib not exist, recursive update submodule to download this module")
endif()


message(STATUS "==================================================")

###############################################
# path
###############################################
if(${MSVC}) 
	link_directories(
		./3rdParty/lib/win64 
		./3rdParty/lib/win64/release 
		./3rdParty/lib/win64/debug
		${PROJECT_SOURCE_DIR}/lib/Release 
		${PROJECT_SOURCE_DIR}/lib/Debug)

elseif(${UNIX})

	link_directories(./3rdParty/lib/linux64 ${PROJECT_SOURCE_DIR}/bin/linux64)
	set(EXECUTABLE_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin/linux64)
	set(LIBRARY_OUTPUT_PATH  ${PROJECT_SOURCE_DIR}/bin/linux64)
endif(${MSVC})


###############################################
# Source
###############################################  

# add all projects in src folder with CMakeLists.txt
SUBDIRLIST(prj_groups ${CMAKE_SOURCE_DIR}/src)
foreach(group ${prj_groups})
	if(EXISTS ${CMAKE_SOURCE_DIR}/src/${group}/CMakeLists.txt)
		add_subdirectory(src/${group})				
	endif()
endforeach()

# ZeroMQ (via vcpkg): cppzmq is header-only; ZeroMQ provides the lib.
find_package(cppzmq CONFIG REQUIRED)
find_package(ZeroMQ CONFIG REQUIRED)

# Link the MicrochipTouchscreen target (defined in src/MicrochipTouchscreen/CMakeLists.txt)
if(TARGET MicrochipTouchscreen)
	target_link_libraries(MicrochipTouchscreen cppzmq libzmq)
endif()




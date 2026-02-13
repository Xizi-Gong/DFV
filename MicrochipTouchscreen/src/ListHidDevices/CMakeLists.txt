# Set project_name as the directory name
file(RELATIVE_PATH project_name ${CMAKE_CURRENT_LIST_DIR}/.. ${CMAKE_CURRENT_LIST_DIR})

###############################################
# Set project files
###############################################
project (${project_name})

# common files
include_directories(${CMAKE_SOURCE_DIR}/src/common)

file(GLOB  SOURCES  *.h *.cpp)

###############################################
# create the project
###############################################
add_executable(${project_name} ${SOURCES})

#target_link_libraries(${project_name} ${OpenCV_LIBS})

if(OpenMP_CXX_FOUND)
    target_link_libraries(${project_name} OpenMP::OpenMP_CXX)
endif()


###############################################
# Set Path for Visual Studio to add bin to path
###############################################
if(MSVC AND DEFINED BIN_PATH)
	set_target_properties(${project_name} PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=%PATH%;${BIN_PATH}")
endif()


###############################################
# Add this project to corresponding folder
###############################################
#set_target_properties(${project_name} PROPERTIES FOLDER HardwareTest)


message(STATUS "Added hardware test project: ${project_name}")

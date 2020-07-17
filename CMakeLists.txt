cmake_minimum_required(VERSION 2.8.3)
project(darknet_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_python_setup()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES darknet
)

include(ExternalProject)
ExternalProject_Add(darknet_external
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  PREFIX "${CMAKE_CURRENT_SOURCE_DIR}"
  STAMP_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND "make"
  BUILD_IN_SOURCE ON
  BUILD_ALWAYS ON
  INSTALL_COMMAND ""
)

add_library(darknet SHARED IMPORTED GLOBAL)
add_dependencies(darknet darknet_external)
set_target_properties(darknet PROPERTIES
  IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libdarknet.so
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include
)

install(
  FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/libdarknet.so
  DESTINATION
    ${CATKIN_PACKAGE_LIB_DESTINATION}
)

install(
  DIRECTORY
    cfg
    data
  DESTINATION
    ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
  DIRECTORY
    include/${PROJECT_NAME}/
  DESTINATION
    ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
)

catkin_install_python(
  PROGRAMS
    python/${PROJECT_NAME}/darknet.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/python/${PROJECT_NAME}
)

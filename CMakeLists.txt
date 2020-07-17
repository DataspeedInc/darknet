cmake_minimum_required(VERSION 2.8.3)
project(darknet_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
)

catkin_python_setup()

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
)

include(ExternalProject)
ExternalProject_Add(darknet
  SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  PREFIX "${CMAKE_CURRENT_SOURCE_DIR}"
  STAMP_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  CONFIGURE_COMMAND ""
  BUILD_COMMAND "make"
  BUILD_IN_SOURCE ON
  BUILD_ALWAYS ON
  INSTALL_COMMAND ""
)

add_library(${PROJECT_NAME} SHARED IMPORTED GLOBAL)
add_dependencies(${PROJECT_NAME} darknet)
set_target_properties(${PROJECT_NAME} PROPERTIES
  IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/libdarknet.so
  INTERFACE_INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/include
)

install(
  FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/libdarknet.so
  DESTINATION
    ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
  DIRECTORY
    cfg
    data
  DESTINATION
    ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

catkin_install_python(
  PROGRAMS
    python/${PROJECT_NAME}/darknet.py
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/python/${PROJECT_NAME}
)

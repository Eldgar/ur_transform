cmake_minimum_required(VERSION 3.8)
project(ur_transform)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(control_msgs REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)


# 🔹 Find MoveIt2 packages
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_visual_tools REQUIRED)

# Define executables
add_executable(transform_printer src/transform_printer.cpp)
ament_target_dependencies(transform_printer rclcpp tf2_ros geometry_msgs)
install(TARGETS transform_printer DESTINATION lib/${PROJECT_NAME})

add_executable(transform_publisher src/transform_publisher.cpp)
ament_target_dependencies(transform_publisher rclcpp tf2_ros geometry_msgs)
install(TARGETS transform_publisher DESTINATION lib/${PROJECT_NAME})

add_executable(aruco_transform_publisher src/aruco_transform_publisher.cpp)
ament_target_dependencies(aruco_transform_publisher rclcpp tf2_ros geometry_msgs)
install(TARGETS aruco_transform_publisher DESTINATION lib/${PROJECT_NAME})

add_executable(real_aruco_transform_publisher src/real_aruco_transform_publisher.cpp)
ament_target_dependencies(real_aruco_transform_publisher rclcpp tf2_ros geometry_msgs)
install(TARGETS real_aruco_transform_publisher DESTINATION lib/${PROJECT_NAME})

add_executable(teleop_end_effector src/teleop_end_effector.cpp)
ament_target_dependencies(teleop_end_effector rclcpp moveit_core moveit_ros_planning_interface)
install(TARGETS teleop_end_effector DESTINATION lib/${PROJECT_NAME})

add_executable(initial_position src/initial_position.cpp)
ament_target_dependencies(initial_position rclcpp moveit_core moveit_ros_planning_interface)
install(TARGETS initial_position DESTINATION lib/${PROJECT_NAME})

add_executable(echo_node src/echo_node.cpp)
ament_target_dependencies(echo_node rclcpp std_msgs)
install(TARGETS echo_node DESTINATION lib/${PROJECT_NAME})

add_executable(arm_move src/arm_move.cpp)
ament_target_dependencies(arm_move rclcpp std_msgs moveit_ros_planning_interface)
install(TARGETS arm_move DESTINATION lib/${PROJECT_NAME})

add_executable(joint_state_repeater src/joint_state_repeater.cpp)
ament_target_dependencies(joint_state_repeater rclcpp sensor_msgs)
install(TARGETS joint_state_repeater DESTINATION lib/${PROJECT_NAME})

add_executable(sim_camera_cal src/sim_camera_cal.cpp)
ament_target_dependencies(sim_camera_cal
  rclcpp
  sensor_msgs
  cv_bridge
  tf2
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
  OpenCV
)
install(TARGETS sim_camera_cal DESTINATION lib/${PROJECT_NAME})

add_executable(sim_camera_info src/sim_camera_info.cpp)
ament_target_dependencies(sim_camera_info
  rclcpp
  sensor_msgs
  cv_bridge
  tf2
  tf2_ros
  tf2_geometry_msgs
  geometry_msgs
  OpenCV
)
install(TARGETS sim_camera_info DESTINATION lib/${PROJECT_NAME})


add_executable(gripper_control_action src/gripper_control_action.cpp)
ament_target_dependencies(gripper_control_action
  control_msgs
  rclcpp
  rclcpp_action
)
install(TARGETS gripper_control_action DESTINATION lib/${PROJECT_NAME})



# 🔹 Add MoveIt2 executable
# add_executable(move_marker src/move_marker.cpp)
# ament_target_dependencies(move_marker
#     ament_cmake
#     rclcpp
#     rclcpp_action
#     moveit_core
#     moveit_ros_planning_interface
#     interactive_markers
#     moveit_ros_planning
#     moveit_msgs
#     geometric_shapes
# )
# install(TARGETS move_marker DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY 
    launch
    DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()


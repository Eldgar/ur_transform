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
find_package(tf2_eigen REQUIRED) 
find_package(image_transport REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)


# 🔹 Find MoveIt2 packages
find_package(moveit_core REQUIRED)
find_package(moveit_ros_planning_interface REQUIRED)
find_package(moveit_visual_tools REQUIRED)

add_executable(transform_publisher src/transform_publisher.cpp)
ament_target_dependencies(transform_publisher rclcpp tf2_ros geometry_msgs)
install(TARGETS transform_publisher DESTINATION lib/${PROJECT_NAME})

add_executable(aruco_transform_publisher src/aruco_transform_publisher.cpp)
ament_target_dependencies(aruco_transform_publisher rclcpp tf2_ros geometry_msgs)
install(TARGETS aruco_transform_publisher DESTINATION lib/${PROJECT_NAME})

add_executable(real_aruco_transform_publisher src/real_aruco_transform_publisher.cpp)
ament_target_dependencies(real_aruco_transform_publisher rclcpp tf2_ros geometry_msgs)
install(TARGETS real_aruco_transform_publisher DESTINATION lib/${PROJECT_NAME})

add_executable(initial_position src/initial_position.cpp)
ament_target_dependencies(initial_position rclcpp moveit_core moveit_ros_planning_interface geometric_shapes)
install(TARGETS initial_position DESTINATION lib/${PROJECT_NAME})

add_executable(initial_position_real src/initial_position_real.cpp)
ament_target_dependencies(initial_position_real rclcpp moveit_core moveit_ros_planning_interface geometric_shapes)
install(TARGETS initial_position_real DESTINATION lib/${PROJECT_NAME})

# add_executable(sim_camera_cal src/sim_camera_cal.cpp)
# target_compile_definitions(sim_camera_cal PRIVATE SIMULATION)
# ament_target_dependencies(sim_camera_cal
#   rclcpp sensor_msgs cv_bridge tf2 tf2_ros tf2_geometry_msgs geometry_msgs
#   moveit_ros_planning_interface moveit_core OpenCV
# )
# install(TARGETS sim_camera_cal DESTINATION lib/${PROJECT_NAME})

add_executable(real_camera_cal src/real_camera_cal.cpp)
target_compile_definitions(real_camera_cal PRIVATE REAL_ROBOT)
ament_target_dependencies(real_camera_cal
  rclcpp sensor_msgs cv_bridge tf2 tf2_ros tf2_geometry_msgs geometry_msgs
  moveit_ros_planning_interface moveit_core OpenCV
)
install(TARGETS real_camera_cal DESTINATION lib/${PROJECT_NAME})

add_executable(sim_camera_tf_sampler src/sim_camera_tf_sampler.cpp)
ament_target_dependencies(sim_camera_tf_sampler
  rclcpp sensor_msgs cv_bridge tf2 tf2_ros tf2_geometry_msgs geometry_msgs
  moveit_ros_planning_interface moveit_core OpenCV image_transport
)
install(TARGETS sim_camera_tf_sampler DESTINATION lib/${PROJECT_NAME})

add_executable(real_camera_tf_sampler src/real_camera_tf_sampler.cpp)
ament_target_dependencies(real_camera_tf_sampler
  rclcpp sensor_msgs cv_bridge tf2 tf2_ros tf2_geometry_msgs geometry_msgs
  moveit_ros_planning_interface moveit_core OpenCV image_transport
)
install(TARGETS real_camera_tf_sampler DESTINATION lib/${PROJECT_NAME})

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(sim_camera_pose src/sim_camera_pose.cpp)
ament_target_dependencies(sim_camera_pose rclcpp sensor_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs tf2_eigen cv_bridge image_transport)
target_link_libraries(sim_camera_pose ${OpenCV_LIBRARIES})
install(TARGETS sim_camera_pose
  DESTINATION lib/${PROJECT_NAME})

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(sim_aruco_pose src/sim_aruco_pose.cpp)
ament_target_dependencies(sim_aruco_pose rclcpp sensor_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs tf2_eigen cv_bridge image_transport)
target_link_libraries(sim_aruco_pose ${OpenCV_LIBRARIES})
install(TARGETS sim_aruco_pose
  DESTINATION lib/${PROJECT_NAME})

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(real_aruco_pose src/real_aruco_pose.cpp)
ament_target_dependencies(real_aruco_pose rclcpp sensor_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs tf2_eigen cv_bridge)
target_link_libraries(real_aruco_pose ${OpenCV_LIBRARIES})
install(TARGETS real_aruco_pose
  DESTINATION lib/${PROJECT_NAME})

include_directories(${OpenCV_INCLUDE_DIRS})
add_executable(real_camera_pose src/real_camera_pose.cpp)
ament_target_dependencies(real_camera_pose rclcpp sensor_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs tf2_eigen cv_bridge image_transport std_msgs sensor_msgs)
target_link_libraries(real_camera_pose ${OpenCV_LIBRARIES})
install(TARGETS real_camera_pose
  DESTINATION lib/${PROJECT_NAME})

add_executable(gripper_control_action src/gripper_control_action.cpp)
ament_target_dependencies(gripper_control_action
  control_msgs
  rclcpp
  rclcpp_action
)
install(TARGETS gripper_control_action DESTINATION lib/${PROJECT_NAME})


add_executable(collision_objects src/collision_objects.cpp)
ament_target_dependencies(collision_objects
  rclcpp
  moveit_ros_planning_interface
  geometry_msgs
  shape_msgs
  moveit_msgs
)
install(TARGETS collision_objects DESTINATION lib/${PROJECT_NAME})




add_executable(aruco_calibration_test src/aruco_calibration_test.cpp)
target_include_directories(aruco_calibration_test PUBLIC
    ${EIGEN3_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

ament_target_dependencies(aruco_calibration_test
    rclcpp rclcpp_action
    moveit_ros_planning_interface
    tf2_ros tf2_geometry_msgs tf2_eigen
    geometry_msgs sensor_msgs
    cv_bridge image_transport
    OpenCV
)
install(TARGETS aruco_calibration_test DESTINATION lib/${PROJECT_NAME})


add_executable(move_marker src/move_marker.cpp)
ament_target_dependencies(move_marker
    ament_cmake
    rclcpp
    rclcpp_action
    moveit_core
    moveit_ros_planning_interface
    interactive_markers
    moveit_ros_planning
    moveit_msgs
    geometric_shapes
)
install(TARGETS move_marker DESTINATION lib/${PROJECT_NAME})

add_executable(center_marker src/center_marker.cpp)
ament_target_dependencies(center_marker
    ament_cmake
    rclcpp
    rclcpp_action
    moveit_core
    moveit_ros_planning_interface
    interactive_markers
    moveit_ros_planning
    moveit_msgs
    geometric_shapes
	moveit_ros_planning_interface
)
install(TARGETS center_marker DESTINATION lib/${PROJECT_NAME})


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


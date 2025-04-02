#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("move_relative_camera_frame");
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);

  // Wait for joint_states
  RCLCPP_INFO(node->get_logger(), "Waiting for /joint_states...");
  sensor_msgs::msg::JointState::SharedPtr latest_msg = nullptr;
  auto sub = node->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", 10,
    [&](const sensor_msgs::msg::JointState::SharedPtr msg) {
      latest_msg = msg;
    });

  rclcpp::Time start_time = node->get_clock()->now();
  while (rclcpp::ok() && !latest_msg) {
    executor.spin_some();
    if ((node->get_clock()->now() - start_time).seconds() > 5.0) {
      RCLCPP_ERROR(node->get_logger(), "Timed out waiting for joint_states");
      rclcpp::shutdown();
      return 1;
    }
  }

  // Build robot model and state
  robot_model_loader::RobotModelLoader model_loader(node, "robot_description");
  moveit::core::RobotModelPtr kinematic_model = model_loader.getModel();
  auto robot_state = std::make_shared<moveit::core::RobotState>(kinematic_model);

  std::map<std::string, double> joint_positions;
  for (size_t i = 0; i < latest_msg->name.size(); ++i)
    joint_positions[latest_msg->name[i]] = latest_msg->position[i];

  robot_state->setVariablePositions(joint_positions);
  robot_state->update();

  // Get current EE pose
  std::string ee_link = "rg2_gripper_aruco_link";
  const Eigen::Isometry3d& ee_tf_base = robot_state->getGlobalLinkTransform(ee_link);
  geometry_msgs::msg::PoseStamped ee_pose_base;
  ee_pose_base.header.frame_id = "base_link";
  ee_pose_base.header.stamp = rclcpp::Time(0);
  ee_pose_base.pose = tf2::toMsg(ee_tf_base);

  // Setup TF2
  auto tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);
  rclcpp::sleep_for(std::chrono::milliseconds(500));

  // Wait for camera_link TF to be available
  std::string camera_frame = "camera_link";
  if (!tf_buffer->canTransform(camera_frame, "base_link", rclcpp::Time(0), tf2::durationFromSec(3.0))) {
    RCLCPP_ERROR(node->get_logger(), "camera_link TF not available.");
    rclcpp::shutdown();
    return 1;
  }

  // Transform current EE pose into camera frame
  geometry_msgs::msg::PoseStamped ee_pose_camera;
  try {
    ee_pose_camera = tf_buffer->transform(ee_pose_base, camera_frame, tf2::durationFromSec(1.0));
  } catch (tf2::TransformException &ex) {
    RCLCPP_ERROR(node->get_logger(), "TF transform to camera frame failed: %s", ex.what());
    rclcpp::shutdown();
    return 1;
  }

  // Move 5cm along camera Z axis
  geometry_msgs::msg::PoseStamped target_pose_camera = ee_pose_camera;
  target_pose_camera.pose.position.z -= 0.05;

  // Transform back to base_link
  geometry_msgs::msg::PoseStamped target_pose_base;
  try {
    target_pose_base = tf_buffer->transform(target_pose_camera, "base_link", tf2::durationFromSec(1.0));
  } catch (tf2::TransformException &ex) {
    RCLCPP_ERROR(node->get_logger(), "TF transform back to base_link failed: %s", ex.what());
    rclcpp::shutdown();
    return 1;
  }

  // MoveIt setup
  moveit::planning_interface::MoveGroupInterface move_group(node, "ur_manipulator");
  move_group.setEndEffectorLink(ee_link);
  move_group.setPlanningTime(5.0);
  move_group.setMaxVelocityScalingFactor(0.1);
  move_group.setMaxAccelerationScalingFactor(0.1);

  std::vector<geometry_msgs::msg::Pose> waypoints = {target_pose_base.pose};
  moveit_msgs::msg::RobotTrajectory trajectory;
  double fraction = move_group.computeCartesianPath(waypoints, 0.003, 0.0, trajectory);

  RCLCPP_INFO(node->get_logger(), "Cartesian path fraction: %.2f", fraction);
  if (fraction > 0.8) {
    move_group.execute(trajectory);
    RCLCPP_INFO(node->get_logger(), "Moved 5cm forward in camera frame.");
  } else {
    RCLCPP_WARN(node->get_logger(), "Cartesian path planning failed.");
  }

  rclcpp::shutdown();
  return 0;
}









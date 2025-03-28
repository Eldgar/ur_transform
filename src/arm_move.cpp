#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2_eigen/tf2_eigen.hpp>

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  auto node = rclcpp::Node::make_shared("move_down_from_joint_states");
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);

  RCLCPP_INFO(node->get_logger(), "Waiting for /joint_states...");

  // Get latest joint states from topic
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

  // Build robot model from parameter server
  robot_model_loader::RobotModelLoader model_loader(node, "robot_description");
  moveit::core::RobotModelPtr kinematic_model = model_loader.getModel();
  auto robot_state = std::make_shared<moveit::core::RobotState>(kinematic_model);

  // Fill joint values from joint_states message
  std::map<std::string, double> joint_positions;
  for (size_t i = 0; i < latest_msg->name.size(); ++i) {
    joint_positions[latest_msg->name[i]] = latest_msg->position[i];
  }
  robot_state->setVariablePositions(joint_positions);
  robot_state->update();

  std::string ee_link = "wrist_3_link";
  const Eigen::Isometry3d& ee_tf = robot_state->getGlobalLinkTransform(ee_link);
  geometry_msgs::msg::Pose current_pose = tf2::toMsg(ee_tf);

  RCLCPP_INFO(node->get_logger(), "Current Z: %.4f", current_pose.position.z);

  geometry_msgs::msg::Pose target_pose = current_pose;
  target_pose.position.z -= 0.005;
  RCLCPP_INFO(node->get_logger(), "Target Z: %.4f", target_pose.position.z);

  // MoveIt setup
  moveit::planning_interface::MoveGroupInterface move_group(node, "ur_manipulator");
  move_group.setPlanningTime(5.0);
  move_group.setMaxVelocityScalingFactor(0.1);
  move_group.setMaxAccelerationScalingFactor(0.1);

  std::vector<geometry_msgs::msg::Pose> waypoints = {target_pose};
  moveit_msgs::msg::RobotTrajectory trajectory;
  double fraction = move_group.computeCartesianPath(waypoints, 0.003, 0.0, trajectory);

  RCLCPP_INFO(node->get_logger(), "Path fraction: %.2f", fraction);
  if (fraction > 0.8) {
    move_group.execute(trajectory);
    RCLCPP_INFO(node->get_logger(), "Moved down by 5mm.");
  } else {
    RCLCPP_WARN(node->get_logger(), "Path planning failed.");
  }

  rclcpp::shutdown();
  return 0;
}











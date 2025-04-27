#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2_eigen/tf2_eigen.hpp>

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("move_rg2_relative_base");
  rclcpp::executors::SingleThreadedExecutor executor;
  executor.add_node(node);

  /* ───── 1. Wait for joint states ────────────────────────────── */
  RCLCPP_INFO(node->get_logger(), "Waiting for /joint_states …");
  sensor_msgs::msg::JointState::SharedPtr latest_msg = nullptr;
  auto sub = node->create_subscription<sensor_msgs::msg::JointState>(
    "/joint_states", 10,
    [&](const sensor_msgs::msg::JointState::SharedPtr msg) { latest_msg = msg; });

  rclcpp::Time start_time = node->get_clock()->now();
  while (rclcpp::ok() && !latest_msg) {
    executor.spin_some();
    if ((node->get_clock()->now() - start_time).seconds() > 5.0) {
      RCLCPP_ERROR(node->get_logger(), "Timed out waiting for joint_states");
      rclcpp::shutdown();
      return 1;
    }
  }

  /* ───── 2. Reconstruct current state ────────────────────────── */
  robot_model_loader::RobotModelLoader loader(node, "robot_description");
  auto kinematic_model = loader.getModel();
  auto robot_state     = std::make_shared<moveit::core::RobotState>(kinematic_model);

  std::map<std::string, double> joints;
  for (size_t i = 0; i < latest_msg->name.size(); ++i)
    joints[latest_msg->name[i]] = latest_msg->position[i];

  robot_state->setVariablePositions(joints);
  robot_state->update();

  /* ───── 3. Current EE pose in base_link ─────────────────────── */
  const std::string ee_link = "rg2_gripper_aruco_link";
  const Eigen::Isometry3d &ee_tf_base = robot_state->getGlobalLinkTransform(ee_link);

  geometry_msgs::msg::Pose current_pose = tf2::toMsg(ee_tf_base);

  /* ───── 4. Target pose – 5 cm straight up in base_link ─────── */
  geometry_msgs::msg::Pose target_pose = current_pose;
  target_pose.position.z += 0.03;   // change dx/dy/dz here as needed

  /* ───── 5. Plan & execute Cartesian path ────────────────────── */
  moveit::planning_interface::MoveGroupInterface move_group(node, "ur_manipulator");
  move_group.setEndEffectorLink(ee_link);
  move_group.setStartState(*robot_state);            // start = current
  move_group.setPlanningTime(5.0);
  move_group.setMaxVelocityScalingFactor(0.1);
  move_group.setMaxAccelerationScalingFactor(0.1);

  std::vector<geometry_msgs::msg::Pose> waypoints = {target_pose};
  moveit_msgs::msg::RobotTrajectory trajectory;
  double fraction = move_group.computeCartesianPath(waypoints,
                                                    0.005,   // eef_step
                                                    0.0,     // jump_threshold
                                                    trajectory);
  RCLCPP_INFO(node->get_logger(),
            "Trajectory contains %zu points",
            trajectory.joint_trajectory.points.size());


  RCLCPP_INFO(node->get_logger(), "Cartesian path fraction: %.2f", fraction);
  if (fraction > 0.8) {
    move_group.execute(trajectory);
    RCLCPP_INFO(node->get_logger(), "Moved 3cm along base_link-Z.");
  } else {
    RCLCPP_WARN(node->get_logger(), "Cartesian path planning failed.");
  }

  rclcpp::shutdown();
  return 0;
}










#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <moveit/robot_model_loader/robot_model_loader.h>
#include <moveit/robot_state/robot_state.h>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <Eigen/Geometry>

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

  // Load robot model
  robot_model_loader::RobotModelLoader model_loader(node, "robot_description");
  moveit::core::RobotModelPtr kinematic_model = model_loader.getModel();
  auto robot_state = std::make_shared<moveit::core::RobotState>(kinematic_model);

  std::map<std::string, double> joint_positions;
  for (size_t i = 0; i < latest_msg->name.size(); ++i)
    joint_positions[latest_msg->name[i]] = latest_msg->position[i];
  robot_state->setVariablePositions(joint_positions);
  robot_state->update();

  // TF2 setup
  auto tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
  auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);
  rclcpp::sleep_for(std::chrono::milliseconds(500));

  std::string camera_frame = "camera_link";
  if (!tf_buffer->canTransform(camera_frame, "base_link", rclcpp::Time(0), tf2::durationFromSec(3.0))) {
    RCLCPP_ERROR(node->get_logger(), "camera_link TF not available.");
    rclcpp::shutdown();
    return 1;
  }

  // Define target pose in camera frame
  geometry_msgs::msg::PoseStamped marker_pose_camera;
  marker_pose_camera.header.frame_id = camera_frame;
  marker_pose_camera.header.stamp = rclcpp::Time(0);
  marker_pose_camera.pose.position.x = 0.0;
  marker_pose_camera.pose.position.y = 0.0;
  marker_pose_camera.pose.position.z = 0.3;

  // Orientation via quaternion
  Eigen::Matrix3d rot_cam_to_marker;
  rot_cam_to_marker.col(0) = Eigen::Vector3d(0, 1, 0);
  rot_cam_to_marker.col(1) = Eigen::Vector3d(1, 0, 0);
  rot_cam_to_marker.col(2) = Eigen::Vector3d(0, 0, -1);
  Eigen::Quaterniond quat(rot_cam_to_marker.normalized());
  marker_pose_camera.pose.orientation = tf2::toMsg(quat);

  // Transform to base_link
  geometry_msgs::msg::PoseStamped target_pose_base;
  try {
    target_pose_base = tf_buffer->transform(marker_pose_camera, "base_link", tf2::durationFromSec(1.0));
  } catch (tf2::TransformException &ex) {
    RCLCPP_ERROR(node->get_logger(), "TF transform to base_link failed: %s", ex.what());
    rclcpp::shutdown();
    return 1;
  }

  // MoveIt setup
  std::string ee_link = "rg2_gripper_aruco_link";
  moveit::planning_interface::MoveGroupInterface move_group(node, "ur_manipulator");
  move_group.setEndEffectorLink(ee_link);
  move_group.setPlanningTime(5.0);
  move_group.setMaxVelocityScalingFactor(0.1);
  move_group.setMaxAccelerationScalingFactor(0.1);

  // Add obstacles to the planning scene
  moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
  std::vector<moveit_msgs::msg::CollisionObject> collision_objects;

  auto add_box = [&](const std::string& id, double x, double y, double z,
                     double dx, double dy, double dz) {
    moveit_msgs::msg::CollisionObject obj;
    obj.id = id;
    obj.header.frame_id = "base_link";

    shape_msgs::msg::SolidPrimitive primitive;
    primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
    primitive.dimensions = {dx, dy, dz};

    geometry_msgs::msg::Pose pose;
    pose.orientation.w = 1.0;
    pose.position.x = x;
    pose.position.y = y;
    pose.position.z = z;

    obj.primitives.push_back(primitive);
    obj.primitive_poses.push_back(pose);
    obj.operation = moveit_msgs::msg::CollisionObject::ADD;

    collision_objects.push_back(obj);
  };

  // Define each box obstacle
  add_box("Box_0", 0.0, -0.52, 0.5, 1.2, 0.2, 1.2);
  add_box("Box_1", 0.4,  0.2, -0.11, 1.0, 1.8, 0.15);
  add_box("Box_2", 0.4,  0.85, 0.2, 0.55, 0.7, 0.65);
  add_box("Box_3", -0.45, -0.46, 0.4, 0.18, 0.18, 0.18);

  planning_scene_interface.applyCollisionObjects(collision_objects);
  rclcpp::sleep_for(std::chrono::milliseconds(500));

  // Use setPoseTarget + plan (collision-aware)
  move_group.setPoseTarget(target_pose_base.pose);
  moveit::planning_interface::MoveGroupInterface::Plan plan;
  bool success = (move_group.plan(plan) == moveit::core::MoveItErrorCode::SUCCESS);

  if (success) {
    move_group.execute(plan);
    RCLCPP_INFO(node->get_logger(), "Moved to pose aligned in front of camera (collision-aware).");
  } else {
    RCLCPP_WARN(node->get_logger(), "Motion planning failed due to collision or unreachable pose.");
  }

  rclcpp::shutdown();
  return 0;
}











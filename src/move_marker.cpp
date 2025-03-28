#include <rclcpp/rclcpp.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <thread>
#include <chrono>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("camera_frame_planner");

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);
    auto node = rclcpp::Node::make_shared("camera_frame_planner_node", node_options);

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    std::thread([&executor]() { executor.spin(); }).detach();

    const std::string PLANNING_GROUP = "ur_manipulator";
    moveit::planning_interface::MoveGroupInterface move_group(node, PLANNING_GROUP);

    move_group.setPlanningTime(10.0);
    move_group.setStartStateToCurrentState();

    auto joint_model_group = move_group.getCurrentState()->getJointModelGroup(PLANNING_GROUP);

    // Go to home position
    std::vector<double> home_joints = {0.0, -1.5708, 0.0, -1.5708, 0.0, -1.5708};
    move_group.setJointValueTarget(home_joints);
    moveit::planning_interface::MoveGroupInterface::Plan home_plan;
    if (move_group.plan(home_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
        move_group.execute(home_plan);
        std::this_thread::sleep_for(std::chrono::milliseconds(300));
    } else {
        RCLCPP_WARN(LOGGER, "Failed to plan home");
    }

    // Set up TF listener
    auto tf_buffer = std::make_shared<tf2_ros::Buffer>(node->get_clock());
    auto tf_listener = std::make_shared<tf2_ros::TransformListener>(*tf_buffer);

    // Pose in camera frame
    geometry_msgs::msg::PoseStamped camera_pose;
    camera_pose.header.frame_id = "wrist_rgbd_camera_depth_optical_frame";
    camera_pose.header.stamp = node->get_clock()->now();
    camera_pose.pose.position.x = 0.0;
    camera_pose.pose.position.y = 0.0;
    camera_pose.pose.position.z = 0.3;
    camera_pose.pose.orientation.w = 1.0; // Identity orientation

    geometry_msgs::msg::PoseStamped transformed_pose;
    std::string planning_frame = move_group.getPlanningFrame();
    RCLCPP_INFO(LOGGER, "Planning frame: %s", planning_frame.c_str());

    try {
        transformed_pose = tf_buffer->transform(
            camera_pose, planning_frame, tf2::durationFromSec(1.0));
    } catch (tf2::TransformException &ex) {
        RCLCPP_ERROR(LOGGER, "TF transform failed: %s", ex.what());
        rclcpp::shutdown();
        return 1;
    }

    move_group.setPoseTarget(transformed_pose);
    moveit::planning_interface::MoveGroupInterface::Plan camera_plan;
    if (move_group.plan(camera_plan) == moveit::core::MoveItErrorCode::SUCCESS) {
        RCLCPP_INFO(LOGGER, "Planning from camera frame pose succeeded");
        move_group.execute(camera_plan);
    } else {
        RCLCPP_ERROR(LOGGER, "Planning from camera frame pose failed");
    }

    rclcpp::shutdown();
    return 0;
}





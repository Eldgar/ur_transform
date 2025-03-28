#include <chrono>
#include <thread>
#include <vector>
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <rclcpp/rclcpp.hpp>

using namespace std::chrono_literals;

// Function to transform a pose from `camera_link` to `base_link`
geometry_msgs::msg::Pose transformPoseToBaseLink(
    rclcpp::Node::SharedPtr node,
    const geometry_msgs::msg::Pose& pose_in_camera_frame)
{
    static tf2_ros::Buffer tf_buffer(node->get_clock());
    static tf2_ros::TransformListener tf_listener(tf_buffer);

    geometry_msgs::msg::PoseStamped pose_stamped_in, pose_stamped_out;
    pose_stamped_in.header.frame_id = "camera_link";  // Source frame
    // Use Time(0) to get the latest available transform
    pose_stamped_in.header.stamp = rclcpp::Time(0);
    pose_stamped_in.pose = pose_in_camera_frame;

    try {
        // Wait for the transform to be available using latest time
        if (!tf_buffer.canTransform("base_link", "camera_link", rclcpp::Time(0),
                                     rclcpp::Duration::from_seconds(1.0)))
        {
            RCLCPP_ERROR(node->get_logger(), "Cannot transform from camera_link to base_link");
            return pose_in_camera_frame;
        }
        tf_buffer.transform(pose_stamped_in, pose_stamped_out, "base_link");
        return pose_stamped_out.pose;
    } catch (tf2::TransformException &ex) {
        RCLCPP_ERROR(node->get_logger(), "Transform failed: %s", ex.what());
        return pose_in_camera_frame; // Return original pose on failure
    }
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("move_aruco_relative_to_camera");

    // Initialize MoveIt2 interface for the manipulator
    moveit::planning_interface::MoveGroupInterface move_group_arm(node, "ur_manipulator");
    RCLCPP_INFO(node->get_logger(), "Using planning group: %s", move_group_arm.getName().c_str());

    // Allow extra time for TF buffer and robot state to fill
    rclcpp::Rate rate(10);
    for (int i = 0; i < 20; i++) {  // increased from 10 to 20 iterations
        rclcpp::spin_some(node);
        rate.sleep();
    }

    // 🔹 Define the movement relative to the camera (only moving in Y)
    geometry_msgs::msg::Pose target_pose_in_camera_frame;
    target_pose_in_camera_frame.position.x = 0.0;    // No movement in X
    target_pose_in_camera_frame.position.y = 0.005;    // Move 0.5cm in Y (adjust as needed)
    target_pose_in_camera_frame.position.z = 0.0;      // No movement in Z

    // Get the current pose of the end-effector to preserve its orientation
    geometry_msgs::msg::Pose current_pose = move_group_arm.getCurrentPose().pose;
    target_pose_in_camera_frame.orientation = current_pose.orientation;

    // 🔹 Convert the target pose from `camera_link` to `base_link`
    geometry_msgs::msg::Pose target_pose_in_base_link =
        transformPoseToBaseLink(node, target_pose_in_camera_frame);

    // 🔹 Use Cartesian Path Planning
    // Start from the current pose and add the target as the next waypoint
    std::vector<geometry_msgs::msg::Pose> waypoints;
    geometry_msgs::msg::Pose start_pose = move_group_arm.getCurrentPose().pose;
    waypoints.push_back(start_pose);
    waypoints.push_back(target_pose_in_base_link);

    moveit_msgs::msg::RobotTrajectory trajectory;
    const double jump_threshold = 0.0;
    const double eef_step = 0.001; // Step size (1mm)

    double fraction = move_group_arm.computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);

    if (fraction >= 0.95) {
        RCLCPP_INFO(node->get_logger(), "Cartesian path computed successfully (%.2f%% achieved)", fraction * 100.0);
        move_group_arm.execute(trajectory);
    } else {
        RCLCPP_ERROR(node->get_logger(), "Failed to compute full Cartesian path (only %.2f%% achieved)", fraction * 100.0);
    }

    rclcpp::shutdown();
    return 0;
}



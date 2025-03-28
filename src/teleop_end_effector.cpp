#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit_msgs/msg/robot_trajectory.hpp>
#include <rclcpp/rclcpp.hpp>

int main(int argc, char **argv)
{
    // Initialize ROS2
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("cartesian_move_down");

    // Create MoveGroupInterface for Cartesian path execution
    static const std::string PLANNING_GROUP = "ur_manipulator";
    moveit::planning_interface::MoveGroupInterface move_group(node, PLANNING_GROUP);
    move_group.setPlanningTime(10.0);

    // Allow time for MoveIt2 to receive state updates
    rclcpp::Rate rate(10);
    for (int i = 0; i < 20; i++) {
        rclcpp::spin_some(node);
        rate.sleep();
    }

    // Get the current end-effector pose
    geometry_msgs::msg::Pose current_pose = move_group.getCurrentPose().pose;

    // Move down by 0.01m
    current_pose.position.z -= 0.01;
    RCLCPP_INFO(node->get_logger(), "Moving end effector down: new z = %.3f", current_pose.position.z);

    // Plan a Cartesian path
    std::vector<geometry_msgs::msg::Pose> waypoints;
    waypoints.push_back(move_group.getCurrentPose().pose);
    waypoints.push_back(current_pose);

    moveit_msgs::msg::RobotTrajectory trajectory;
    const double jump_threshold = 0.0;
    const double eef_step = 0.005; // Small step for precision

    double fraction = move_group.computeCartesianPath(waypoints, eef_step, jump_threshold, trajectory);
    if (fraction >= 0.95) {
        RCLCPP_INFO(node->get_logger(), "Cartesian path computed successfully (%.2f%% achieved)", fraction * 100.0);
        move_group.execute(trajectory);
    } else {
        RCLCPP_ERROR(node->get_logger(), "Failed to compute full Cartesian path (only %.2f%% achieved)", fraction * 100.0);
    }

    rclcpp::shutdown();
    return 0;
}



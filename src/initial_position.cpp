#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <chrono>
#include <thread>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("move_group_demo");

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);
    auto move_group_node =
        rclcpp::Node::make_shared("move_group_interface_tutorial", node_options);

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(move_group_node);
    std::thread([&executor]() { executor.spin(); }).detach();

    static const std::string PLANNING_GROUP_ARM = "ur_manipulator";
    moveit::planning_interface::MoveGroupInterface move_group_arm(
        move_group_node, PLANNING_GROUP_ARM);

    const moveit::core::JointModelGroup *joint_model_group_arm =
        move_group_arm.getCurrentState()->getJointModelGroup(PLANNING_GROUP_ARM);
    move_group_arm.setPlanningTime(10.0);

    // Get Current State
    moveit::core::RobotStatePtr current_state_arm =
        move_group_arm.getCurrentState(10);

    std::vector<double> joint_group_positions_arm;
    current_state_arm->copyJointGroupPositions(joint_model_group_arm,
                                               joint_group_positions_arm);

    move_group_arm.setStartStateToCurrentState();

    // Move to First Position
    RCLCPP_INFO(LOGGER, "Moving to First Position");
    joint_group_positions_arm[0] = 0.0;      // Shoulder Pan
    joint_group_positions_arm[1] = -1.5708;  // Shoulder Lift
    joint_group_positions_arm[2] = 0.0;      // Elbow
    joint_group_positions_arm[3] = -1.5708;  // Wrist 1
    joint_group_positions_arm[4] = 0.0;      // Wrist 2
    joint_group_positions_arm[5] = -1.5708;  // Wrist 3
    move_group_arm.setJointValueTarget(joint_group_positions_arm);
    moveit::planning_interface::MoveGroupInterface::Plan my_plan_arm;
    bool success_arm = (move_group_arm.plan(my_plan_arm) ==
                        moveit::core::MoveItErrorCode::SUCCESS);
    move_group_arm.execute(my_plan_arm);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // Move to Second Position
    RCLCPP_INFO(LOGGER, "Moving to Second Position");
    joint_group_positions_arm[0] = 2.72271;      // Shoulder Pan
    joint_group_positions_arm[1] = -0.087;        // Shoulder Lift
    joint_group_positions_arm[2] = 1.740;       // Elbow
    joint_group_positions_arm[3] = -1.535;     // Wrist 1
    joint_group_positions_arm[4] = -0.40;     // Wrist 2
    joint_group_positions_arm[5] = -1.69297;      // Wrist 3
    move_group_arm.setJointValueTarget(joint_group_positions_arm);
    success_arm = (move_group_arm.plan(my_plan_arm) ==
                        moveit::core::MoveItErrorCode::SUCCESS);
    move_group_arm.execute(my_plan_arm);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    RCLCPP_INFO(LOGGER, "Motion Complete. Shutting Down.");
    rclcpp::shutdown();
    return 0;
}





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

    // Go Home
    RCLCPP_INFO(LOGGER, "Going Home");
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


    RCLCPP_INFO(LOGGER, "Going Home");
    joint_group_positions_arm[0] = 2.967;      // Shoulder Pan
    joint_group_positions_arm[1] = 0.0;  // Shoulder Lift
    joint_group_positions_arm[2] = 0.68;      // Elbow
    joint_group_positions_arm[3] = -0.715;  // Wrist 1
    joint_group_positions_arm[4] = -0.157;      // Wrist 2
    joint_group_positions_arm[5] = -1.57;  // Wrist 3
    move_group_arm.setJointValueTarget(joint_group_positions_arm);
    moveit::planning_interface::MoveGroupInterface::Plan my_plan_arm;
    bool success_arm = (move_group_arm.plan(my_plan_arm) ==
                        moveit::core::MoveItErrorCode::SUCCESS);
    move_group_arm.execute(my_plan_arm);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));


    // Pregrasp
    RCLCPP_INFO(LOGGER, "Pregrasp Position");
    geometry_msgs::msg::Pose target_pose1;
    target_pose1.orientation.x = -1.0;
    target_pose1.orientation.y = 0.00;
    target_pose1.orientation.z = 0.00;
    target_pose1.orientation.w = 0.00;
    target_pose1.position.x = 0.341;
    target_pose1.position.y = -0.02;
    target_pose1.position.z = 0.26;
    move_group_arm.setPoseTarget(target_pose1);
    success_arm = (move_group_arm.plan(my_plan_arm) ==
                   moveit::core::MoveItErrorCode::SUCCESS);
    move_group_arm.execute(my_plan_arm);

    // Approach
    RCLCPP_INFO(LOGGER, "Approach to object!");
    std::vector<geometry_msgs::msg::Pose> approach_waypoints;
    target_pose1.position.z -= 0.041;
    approach_waypoints.push_back(target_pose1);
    target_pose1.position.z -= 0.041;
    approach_waypoints.push_back(target_pose1);

    moveit_msgs::msg::RobotTrajectory trajectory_approach;
    const double jump_threshold = 0.0;
    const double eef_step = 0.0002;

    double fraction = move_group_arm.computeCartesianPath(
        approach_waypoints, eef_step, jump_threshold, trajectory_approach);

    move_group_arm.execute(trajectory_approach);

    // Retreat
    RCLCPP_INFO(LOGGER, "Retreat from object!");
    std::vector<geometry_msgs::msg::Pose> retreat_waypoints;
    target_pose1.position.z += 0.041;
    retreat_waypoints.push_back(target_pose1);
    target_pose1.position.z += 0.041;
    retreat_waypoints.push_back(target_pose1);
    moveit_msgs::msg::RobotTrajectory trajectory_retreat;
    fraction = move_group_arm.computeCartesianPath(
        retreat_waypoints, eef_step, jump_threshold, trajectory_retreat);
    move_group_arm.execute(trajectory_retreat);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // Place
    double increment = M_PI / 3; // 60 degrees in radians

    // Get the current position of shoulder_pan_joint
    current_state_arm = move_group_arm.getCurrentState(10);
    current_state_arm->copyJointGroupPositions(joint_model_group_arm,
                                               joint_group_positions_arm);
    double current_position = joint_group_positions_arm[0];

    // Target position is 180 degrees (π radians)
    double target_position = M_PI; // 180 degrees in radians

    // Determine the direction of rotation
    double direction = (target_position > current_position) ? 1.0 : -1.0;

    // Calculate the number of steps required
    int steps = 3;

    // Create a vector to hold the intermediate positions
    std::vector<double> intermediate_positions;

    // Generate the intermediate positions
    for (int i = 1; i <= steps; ++i) {
        double position = current_position + direction * increment * i;
        if ((direction > 0 && position > target_position) ||
            (direction < 0 && position < target_position)) {
            position = target_position;
        }
        intermediate_positions.push_back(position);
        if (position == target_position) {
            break;
        }
    }

    for (double position : intermediate_positions) {
        current_state_arm = move_group_arm.getCurrentState(10);
        current_state_arm->copyJointGroupPositions(joint_model_group_arm,
                                                   joint_group_positions_arm);

        joint_group_positions_arm[0] = position; // Shoulder Pan
        move_group_arm.setJointValueTarget(joint_group_positions_arm);

        moveit::planning_interface::MoveGroupInterface::Plan my_plan_arm;
        bool success_arm = (move_group_arm.plan(my_plan_arm) ==
                            moveit::core::MoveItErrorCode::SUCCESS);

        if (success_arm) {
            move_group_arm.execute(my_plan_arm);
            std::this_thread::sleep_for(std::chrono::milliseconds(300));
        } else {
            RCLCPP_WARN(LOGGER, "Failed to plan for position: %f radians", position);
            break;
        }
    }

    // Go Home
    RCLCPP_INFO(LOGGER, "Going Home");
    joint_group_positions_arm[0] = 0.0;  // Shoulder Pan
    joint_group_positions_arm[1] = -1.5708; // Shoulder Lift
    joint_group_positions_arm[2] = 0.0;  // Elbow
    joint_group_positions_arm[3] = -1.5708; // Wrist 1
    joint_group_positions_arm[4] = 0.0; // Wrist 2
    joint_group_positions_arm[5] = -1.5708;  // Wrist 3
    move_group_arm.setJointValueTarget(joint_group_positions_arm);
    success_arm = (move_group_arm.plan(my_plan_arm) ==
                   moveit::core::MoveItErrorCode::SUCCESS);
    move_group_arm.execute(my_plan_arm);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    rclcpp::shutdown();
    return 0;
}




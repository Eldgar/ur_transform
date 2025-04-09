#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/display_robot_state.hpp>
#include <moveit_msgs/msg/display_trajectory.hpp>
#include <moveit_msgs/msg/collision_object.hpp> // Include for CollisionObject
#include <shape_msgs/msg/solid_primitive.hpp> // Include for SolidPrimitive
#include <geometry_msgs/msg/pose.hpp>       // Include for Pose
#include <chrono>
#include <thread>
#include <vector> // Include for std::vector
#include <string> // Include for std::string

static const rclcpp::Logger LOGGER = rclcpp::get_logger("move_group_demo");

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);
    auto move_group_node =
        rclcpp::Node::make_shared("move_group_interface_tutorial", node_options);

    // We spin up a separate thread for the node executor
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(move_group_node);
    std::thread([&executor]() { executor.spin(); }).detach();

    static const std::string PLANNING_GROUP_ARM = "ur_manipulator";
    moveit::planning_interface::MoveGroupInterface move_group_arm(
        move_group_node, PLANNING_GROUP_ARM);

    // Instantiate PlanningSceneInterface
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;

    // --- Add Obstacles ---
    RCLCPP_INFO(LOGGER, "Adding obstacles to the planning scene");
    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;

    // Lambda function to easily create box obstacles
    auto add_box = [&](const std::string& id, double x, double y, double z,
                       double dx, double dy, double dz) {
        moveit_msgs::msg::CollisionObject obj;
        obj.id = id;
        obj.header.frame_id = move_group_arm.getPlanningFrame(); // Use the planning frame

        shape_msgs::msg::SolidPrimitive primitive;
        primitive.type = shape_msgs::msg::SolidPrimitive::BOX;
        primitive.dimensions = {dx, dy, dz};

        geometry_msgs::msg::Pose pose;
        pose.orientation.w = 1.0; // Neutral orientation
        pose.position.x = x;
        pose.position.y = y;
        pose.position.z = z;

        obj.primitives.push_back(primitive);
        obj.primitive_poses.push_back(pose);
        obj.operation = moveit_msgs::msg::CollisionObject::ADD; // Add the object

        collision_objects.push_back(obj);
    };

    // Define each box obstacle using the lambda function
    // Note: Changed frame_id to use move_group_arm.getPlanningFrame() for robustness.
    //       Often this is "base_link" or "world", but using the getter is safer.
    add_box("Box_0", 0.0, -0.52, 0.5, 1.2, 0.2, 1.2);
    add_box("Box_1", 0.4,  0.2, -0.11, 1.0, 1.8, 0.15);
    add_box("Box_2", 0.4,  0.85, 0.2, 0.55, 0.7, 0.65);
    add_box("Box_3", -0.45, -0.46, 0.4, 0.18, 0.18, 0.18);

    // Apply the collision objects to the planning scene
    planning_scene_interface.applyCollisionObjects(collision_objects);

    // Short sleep to allow the planning scene to update
    RCLCPP_INFO(LOGGER, "Waiting for planning scene to update...");
    rclcpp::sleep_for(std::chrono::milliseconds(500));
    RCLCPP_INFO(LOGGER, "Planning scene updated.");
    // --- End Obstacle Addition ---


    const moveit::core::JointModelGroup *joint_model_group_arm =
        move_group_arm.getCurrentState()->getJointModelGroup(PLANNING_GROUP_ARM);
    move_group_arm.setPlanningTime(10.0); // Set planning time

    // Get Current State
    moveit::core::RobotStatePtr current_state_arm =
        move_group_arm.getCurrentState(10); // Increased timeout for robustness

    std::vector<double> joint_group_positions_arm;
    current_state_arm->copyJointGroupPositions(joint_model_group_arm,
                                              joint_group_positions_arm);

    move_group_arm.setStartStateToCurrentState();

    // Move to First Position
    RCLCPP_INFO(LOGGER, "Planning to First Position");
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

    if (success_arm) {
        RCLCPP_INFO(LOGGER, "Executing plan to First Position");
        move_group_arm.execute(my_plan_arm);
        std::this_thread::sleep_for(std::chrono::milliseconds(300)); // Allow execution to settle
    } else {
        RCLCPP_ERROR(LOGGER, "Failed to plan to First Position");
        // Decide how to handle planning failure (e.g., retry, shutdown)
    }


    // Move to Second Position
    RCLCPP_INFO(LOGGER, "Planning to Second Position");
    // Reuse joint_group_positions_arm vector
    joint_group_positions_arm[0] = 2.9147;   // Shoulder Pan
    joint_group_positions_arm[1] = 0.0349;   // Shoulder Lift
    joint_group_positions_arm[2] = 0.1740;   // Elbow
    joint_group_positions_arm[3] = -0.8203;  // Wrist 1
    joint_group_positions_arm[4] = -0.2269;  // Wrist 2
    joint_group_positions_arm[5] = -2.51327; // Wrist 3
    move_group_arm.setJointValueTarget(joint_group_positions_arm);

    // Re-plan for the second position
    success_arm = (move_group_arm.plan(my_plan_arm) ==
                   moveit::core::MoveItErrorCode::SUCCESS);

    if (success_arm) {
        RCLCPP_INFO(LOGGER, "Executing plan to Second Position");
        move_group_arm.execute(my_plan_arm);
        std::this_thread::sleep_for(std::chrono::milliseconds(300)); // Allow execution to settle
    } else {
        RCLCPP_ERROR(LOGGER, "Failed to plan to Second Position");
        // Decide how to handle planning failure
    }


    RCLCPP_INFO(LOGGER, "Motion Sequence Complete. Shutting Down.");
    rclcpp::shutdown();
    return 0;
}





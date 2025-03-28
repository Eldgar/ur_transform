#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <shape_msgs/msg/solid_primitive.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <chrono>
#include <thread>

static const rclcpp::Logger LOGGER = rclcpp::get_logger("move_group_with_objects");

void addCollisionObjects(moveit::planning_interface::PlanningSceneInterface &planning_scene_interface) {
    std::vector<moveit_msgs::msg::CollisionObject> collision_objects;
    
    // Define Box_0
    moveit_msgs::msg::CollisionObject box_0;
    box_0.id = "Box_0";
    box_0.header.frame_id = "base_link";

    shape_msgs::msg::SolidPrimitive primitive_0;
    primitive_0.type = shape_msgs::msg::SolidPrimitive::BOX;
    primitive_0.dimensions = {1.2, 0.2, 1.2}; // X, Y, Z dimensions

    geometry_msgs::msg::Pose pose_0;
    pose_0.position.x = 0.0;
    pose_0.position.y = -0.65;
    pose_0.position.z = 0.5;

    box_0.primitives.push_back(primitive_0);
    box_0.primitive_poses.push_back(pose_0);
    box_0.operation = moveit_msgs::msg::CollisionObject::ADD;

    // Define Box_1
    moveit_msgs::msg::CollisionObject box_1;
    box_1.id = "Box_1";
    box_1.header.frame_id = "base_link";

    shape_msgs::msg::SolidPrimitive primitive_1;
    primitive_1.type = shape_msgs::msg::SolidPrimitive::BOX;
    primitive_1.dimensions = {1.0, 1.2, 0.2}; // X, Y, Z dimensions

    geometry_msgs::msg::Pose pose_1;
    pose_1.position.x = 0.4;
    pose_1.position.y = 0.2;
    pose_1.position.z = -0.11;

    box_1.primitives.push_back(primitive_1);
    box_1.primitive_poses.push_back(pose_1);
    box_1.operation = moveit_msgs::msg::CollisionObject::ADD;

    // Define Box_2
    moveit_msgs::msg::CollisionObject box_2;
    box_2.id = "Box_2";
    box_2.header.frame_id = "base_link";

    shape_msgs::msg::SolidPrimitive primitive_2;
    primitive_2.type = shape_msgs::msg::SolidPrimitive::BOX;
    primitive_2.dimensions = {0.55, 0.2, 0.65}; // X, Y, Z dimensions

    geometry_msgs::msg::Pose pose_2;
    pose_2.position.x = 0.4;
    pose_2.position.y = 0.85;
    pose_2.position.z = 0.2;

    box_2.primitives.push_back(primitive_2);
    box_2.primitive_poses.push_back(pose_2);
    box_2.operation = moveit_msgs::msg::CollisionObject::ADD;

    // Add objects to the planning scene
    collision_objects.push_back(box_0);
    collision_objects.push_back(box_1);
    collision_objects.push_back(box_2);
    
    planning_scene_interface.applyCollisionObjects(collision_objects);
    RCLCPP_INFO(LOGGER, "Collision objects added to planning scene.");
}

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    rclcpp::NodeOptions node_options;
    node_options.automatically_declare_parameters_from_overrides(true);
    auto move_group_node = rclcpp::Node::make_shared("move_group_with_objects", node_options);

    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(move_group_node);
    std::thread([&executor]() { executor.spin(); }).detach();

    static const std::string PLANNING_GROUP_ARM = "ur_manipulator";
    moveit::planning_interface::MoveGroupInterface move_group_arm(move_group_node, PLANNING_GROUP_ARM);
    moveit::planning_interface::PlanningSceneInterface planning_scene_interface;
    move_group_arm.setPlanningTime(12.0);

    // Add collision objects
    addCollisionObjects(planning_scene_interface);
    std::this_thread::sleep_for(std::chrono::milliseconds(1000)); // Allow scene to update

    // Initialize TF2 listener
    auto tf_buffer = std::make_shared<tf2_ros::Buffer>(move_group_node->get_clock());
    tf2_ros::TransformListener tf_listener(*tf_buffer);

    // Get the current pose of the end-effector
    geometry_msgs::msg::Pose current_pose = move_group_arm.getCurrentPose().pose;
    RCLCPP_INFO(LOGGER, "Current Pose: x=%.3f, y=%.3f, z=%.3f", 
                current_pose.position.x, current_pose.position.y, current_pose.position.z);

    // ----- Logging TF Orientation between base_link and wrist_rgbd_camera_depth_optical_frame -----
    geometry_msgs::msg::TransformStamped tf_stamped;
    try {
        // Lookup transform from base_link to wrist_rgbd_camera_depth_optical_frame
        tf_stamped = tf_buffer->lookupTransform("base_link", "wrist_rgbd_camera_depth_optical_frame", rclcpp::Time(0));
        RCLCPP_INFO(LOGGER, "TF lookup succeeded between base_link and wrist_rgbd_camera_depth_optical_frame.");
    } catch (tf2::TransformException &ex) {
        RCLCPP_WARN(LOGGER, "TF lookup failed: %s", ex.what());
    }
    // Log the orientation as a quaternion
    RCLCPP_INFO(LOGGER, "Orientation (base_link -> wrist_rgbd_camera_depth_optical_frame): x=%.3f, y=%.3f, z=%.3f, w=%.3f",
                tf_stamped.transform.rotation.x, tf_stamped.transform.rotation.y,
                tf_stamped.transform.rotation.z, tf_stamped.transform.rotation.w);

    // Calculate what a 1.0 movement along camera's z-axis looks like in base_link coordinates.
    tf2::Quaternion q;
    tf2::fromMsg(tf_stamped.transform.rotation, q);
    tf2::Vector3 camera_z(0.0, 0.0, 1.0); // Unit vector along camera z-axis.
    tf2::Vector3 displacement = tf2::quatRotate(q, camera_z); // Rotate vector into base_link.
    RCLCPP_INFO(LOGGER, "A 1.0 m movement along camera z corresponds to base_link displacement: x=%.3f, y=%.3f, z=%.3f",
                displacement.x(), displacement.y(), displacement.z());
    // ---------------------------------------------------------------------------------------------

    // Approach Movement (Cartesian Path)
    RCLCPP_INFO(LOGGER, "Approaching the object...");
    std::vector<geometry_msgs::msg::Pose> approach_waypoints;
    geometry_msgs::msg::Pose target_pose = current_pose;
    target_pose.position.x -= 0.00873;
    target_pose.position.y -= 0.00477;
    target_pose.position.z += 0.001;
    approach_waypoints.push_back(target_pose);

    moveit_msgs::msg::RobotTrajectory approach_trajectory;
    double fraction = move_group_arm.computeCartesianPath(
        approach_waypoints, 0.001, 0.0, approach_trajectory);

    if (fraction > 0.9) {
        move_group_arm.execute(approach_trajectory);
        RCLCPP_INFO(LOGGER, "Approach completed.");
    } else {
        RCLCPP_WARN(LOGGER, "Failed to compute approach path.");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    // Retreat Movement (Cartesian Path)
    RCLCPP_INFO(LOGGER, "Retreating from the object...");
    std::vector<geometry_msgs::msg::Pose> retreat_waypoints;
    target_pose = current_pose;
    target_pose.position.x += 0.00873;
    target_pose.position.y += 0.00477;
    target_pose.position.z -= 0.001;
    retreat_waypoints.push_back(target_pose);
    retreat_waypoints.push_back(target_pose);

    moveit_msgs::msg::RobotTrajectory retreat_trajectory;
    fraction = move_group_arm.computeCartesianPath(
        retreat_waypoints, 0.001, 0.0, retreat_trajectory);

    if (fraction > 0.9) {
        move_group_arm.execute(retreat_trajectory);
        RCLCPP_INFO(LOGGER, "Retreat completed.");
    } else {
        RCLCPP_WARN(LOGGER, "Failed to compute retreat path.");
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(300));

    rclcpp::shutdown();
    return 0;
}








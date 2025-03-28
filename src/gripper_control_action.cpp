#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <control_msgs/action/gripper_command.hpp>

using GripperCommand = control_msgs::action::GripperCommand;
using GoalHandle = rclcpp_action::ClientGoalHandle<GripperCommand>;

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = rclcpp::Node::make_shared("gripper_action_client");

    auto client = rclcpp_action::create_client<GripperCommand>(
        node, "/gripper_controller/gripper_cmd");

    if (!client->wait_for_action_server(std::chrono::seconds(5))) {
        RCLCPP_ERROR(node->get_logger(), "Gripper action server not available");
        rclcpp::shutdown();
        return 1;
    }

    auto goal_msg = GripperCommand::Goal();
    goal_msg.command.position = -0.62;  // closed
    goal_msg.command.max_effort = 20.0;

    RCLCPP_INFO(node->get_logger(), "Sending gripper close command...");
    auto send_goal_future = client->async_send_goal(goal_msg);
    if (rclcpp::spin_until_future_complete(node, send_goal_future) !=
        rclcpp::FutureReturnCode::SUCCESS)
    {
        RCLCPP_ERROR(node->get_logger(), "Failed to send goal");
        rclcpp::shutdown();
        return 1;
    }

    std::this_thread::sleep_for(std::chrono::seconds(2));

    goal_msg.command.position = 0.1;  // open
    RCLCPP_INFO(node->get_logger(), "Sending gripper open command...");
    send_goal_future = client->async_send_goal(goal_msg);
    rclcpp::spin_until_future_complete(node, send_goal_future);

    RCLCPP_INFO(node->get_logger(), "Done");
    rclcpp::shutdown();
    return 0;
}













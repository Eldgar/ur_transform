#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

class JointStateRepeater : public rclcpp::Node
{
public:
  JointStateRepeater() : Node("joint_state_repeater")
  {
    pub_ = this->create_publisher<sensor_msgs::msg::JointState>("/joint_states_relay", 10);
    sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
      "/joint_states", 10,
      [this](sensor_msgs::msg::JointState::UniquePtr msg) {
        pub_->publish(*msg);
        RCLCPP_INFO(this->get_logger(), "Relayed joint state with %zu joints", msg->name.size());
      });

    RCLCPP_INFO(this->get_logger(), "JointStateRepeater node started.");
  }

private:
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr pub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr sub_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<JointStateRepeater>());
  rclcpp::shutdown();
  return 0;
}

#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

class EchoNode : public rclcpp::Node
{
public:
  EchoNode() : Node("echo_node")
  {
    subscription_ = this->create_subscription<std_msgs::msg::String>(
      "echo_topic", 10,
      std::bind(&EchoNode::topic_callback, this, std::placeholders::_1));
  }

private:
  void topic_callback(const std_msgs::msg::String::SharedPtr msg) const
  {
    RCLCPP_INFO(this->get_logger(), "Echo: '%s'", msg->data.c_str());
  }

  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr subscription_;
};

int main(int argc, char *argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<EchoNode>());
  rclcpp::shutdown();
  return 0;
}

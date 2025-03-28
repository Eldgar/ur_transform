#include "rclcpp/rclcpp.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include "geometry_msgs/msg/transform_stamped.hpp"

using namespace std::chrono_literals;

class TransformPrinter : public rclcpp::Node
{
public:
  TransformPrinter() : Node("transform_printer")
  {
    // Create a TF2 buffer and listener using the node's clock.
    tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // Create a timer to call the lookup function every 500ms.
    timer_ = this->create_wall_timer(500ms, std::bind(&TransformPrinter::timer_callback, this));
  }

private:
  void timer_callback()
  {
    geometry_msgs::msg::TransformStamped transform_stamped;
    try {
      // Look up the latest available transform from base_link to rg2_gripper_base_link.
      transform_stamped = tf_buffer_->lookupTransform("base_link", "rg2_gripper_base_link", tf2::TimePointZero);
      
      RCLCPP_INFO(this->get_logger(), "Transform received:");
      RCLCPP_INFO(this->get_logger(), "Translation: x: %.3f, y: %.3f, z: %.3f",
                  transform_stamped.transform.translation.x,
                  transform_stamped.transform.translation.y,
                  transform_stamped.transform.translation.z);
      RCLCPP_INFO(this->get_logger(), "Rotation (Quaternion): x: %.3f, y: %.3f, z: %.3f, w: %.3f",
                  transform_stamped.transform.rotation.x,
                  transform_stamped.transform.rotation.y,
                  transform_stamped.transform.rotation.z,
                  transform_stamped.transform.rotation.w);
    }
    catch (tf2::TransformException &ex) {
      RCLCPP_WARN(this->get_logger(), "Could not transform: %s", ex.what());
    }
  }

  rclcpp::TimerBase::SharedPtr timer_;
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<TransformPrinter>());
  rclcpp::shutdown();
  return 0;
}

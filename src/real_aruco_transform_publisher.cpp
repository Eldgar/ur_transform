#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <cmath>

class TransformPublisher : public rclcpp::Node
{
public:
  TransformPublisher()
  : Node("transform_publisher"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    // Publisher for corrected transform
    transform_publisher_ = this->create_publisher<geometry_msgs::msg::TransformStamped>("/ur_transform", 10);

    // Timer to periodically publish the corrected transform
    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      std::bind(&TransformPublisher::publish_transform, this));
  }

private:
  rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr transform_publisher_;
  rclcpp::TimerBase::SharedPtr timer_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  void publish_transform()
  {
    try
    {
      // Lookup the original transform from "base_link" to "aruco_link"
      geometry_msgs::msg::TransformStamped transform_stamped;
      transform_stamped = tf_buffer_.lookupTransform("base_link", "aruco_link", tf2::TimePointZero);

      // Extract the original rotation quaternion
      tf2::Quaternion q_orig(
        transform_stamped.transform.rotation.x,
        transform_stamped.transform.rotation.y,
        transform_stamped.transform.rotation.z,
        transform_stamped.transform.rotation.w);

      // Define a 180-degree rotation around the Z-axis (flips X and Y)
      tf2::Quaternion q_correction;
      q_correction.setRPY(0, 0, M_PI);  // Yaw = 180°

      // Apply correction
      tf2::Quaternion q_corrected = q_correction * q_orig;
      q_corrected.normalize();

      // Update the transform's rotation with corrected orientation
      transform_stamped.transform.rotation.x = q_corrected.x();
      transform_stamped.transform.rotation.y = q_corrected.y();
      transform_stamped.transform.rotation.z = q_corrected.z();
      transform_stamped.transform.rotation.w = q_corrected.w();

      // Publish the corrected transform
      transform_publisher_->publish(transform_stamped);

      RCLCPP_INFO(this->get_logger(),
        "Published corrected Transform: x=%.3f, y=%.3f, z=%.3f | qx=%.3f, qy=%.3f, qz=%.3f, qw=%.3f",
        transform_stamped.transform.translation.x,
        transform_stamped.transform.translation.y,
        transform_stamped.transform.translation.z,
        transform_stamped.transform.rotation.x,
        transform_stamped.transform.rotation.y,
        transform_stamped.transform.rotation.z,
        transform_stamped.transform.rotation.w);
    }
    catch (tf2::TransformException &ex)
    {
      RCLCPP_WARN(this->get_logger(), "Could not get transform: %s", ex.what());
    }
  }
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TransformPublisher>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}




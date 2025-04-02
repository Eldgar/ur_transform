#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/buffer.h"
#include <tf2/LinearMath/Quaternion.h>
#include <cmath>

class TransformPublisher : public rclcpp::Node
{
public:
  TransformPublisher()
  : Node("transform_publisher"),
    tf_buffer_(this->get_clock()),
    tf_listener_(tf_buffer_)
  {
    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    // Create publisher for the transform message
    transform_publisher_ = this->create_publisher<geometry_msgs::msg::TransformStamped>("/ur_transform", 10);

    timer_ = this->create_wall_timer(
      std::chrono::milliseconds(500),
      std::bind(&TransformPublisher::broadcast_transform, this));
  }

private:
  rclcpp::TimerBase::SharedPtr timer_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr transform_publisher_;

  void broadcast_transform()
  {
    try
    {
      // Lookup transform from base_link to aruco_link
      geometry_msgs::msg::TransformStamped original_tf;
      original_tf = tf_buffer_.lookupTransform("base_link", "aruco_link", tf2::TimePointZero);

      // Convert original rotation to tf2 quaternion
      tf2::Quaternion q_orig(
          original_tf.transform.rotation.x,
          original_tf.transform.rotation.y,
          original_tf.transform.rotation.z,
          original_tf.transform.rotation.w);

      // Create a correction rotation of 180 degrees about Z-axis.
      // (This quaternion represents a 180° rotation in the marker's local frame)
      tf2::Quaternion q_correction;
      q_correction.setRPY(0, 0, M_PI);

      // Apply correction by post-multiplying to adjust in the marker frame
      tf2::Quaternion q_rotated = q_orig * q_correction;
      q_rotated.normalize();

      // Create new transform with corrected rotation
      geometry_msgs::msg::TransformStamped rotated_tf;
      rotated_tf.header.stamp = this->get_clock()->now();
      rotated_tf.header.frame_id = "base_link";
      rotated_tf.child_frame_id = "aruco_link_rotated";

      // Copy translation
      rotated_tf.transform.translation = original_tf.transform.translation;

      // Set corrected rotation
      rotated_tf.transform.rotation.x = q_rotated.x();
      rotated_tf.transform.rotation.y = q_rotated.y();
      rotated_tf.transform.rotation.z = q_rotated.z();
      rotated_tf.transform.rotation.w = q_rotated.w();

      // Broadcast the new transform and publish it
      tf_broadcaster_->sendTransform(rotated_tf);
      transform_publisher_->publish(rotated_tf);

      RCLCPP_INFO(this->get_logger(), "Broadcasted transform to aruco_link_rotated.");
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






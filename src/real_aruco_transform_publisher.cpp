#include "rclcpp/logging.hpp"
#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "std_msgs/msg/float64.hpp"                // NEW
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
    tf_broadcaster_      = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    transform_publisher_ = create_publisher<geometry_msgs::msg::TransformStamped>("/ur_transform", 10);
    distance_publisher_  = create_publisher<std_msgs::msg::Float64>("/aruco_distance", 10);   // NEW

    timer_ = create_wall_timer(
        std::chrono::milliseconds(500),
        std::bind(&TransformPublisher::broadcast_transform, this));
  }

private:
  rclcpp::TimerBase::SharedPtr timer_;
  tf2_ros::Buffer              tf_buffer_;
  tf2_ros::TransformListener   tf_listener_;
  std::shared_ptr<tf2_ros::TransformBroadcaster>                  tf_broadcaster_;
  rclcpp::Publisher<geometry_msgs::msg::TransformStamped>::SharedPtr transform_publisher_;
  rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr               distance_publisher_;     // NEW

  void broadcast_transform()
  {
    try
    {
      // 1. Lookup transform base_link → aruco_link
      geometry_msgs::msg::TransformStamped original_tf =
          tf_buffer_.lookupTransform("base_link", "aruco_link", tf2::TimePointZero);

      /* ---------- 2. Rotate marker 180 ° around Z  ----------------- */
      tf2::Quaternion q_orig(
          original_tf.transform.rotation.x,
          original_tf.transform.rotation.y,
          original_tf.transform.rotation.z,
          original_tf.transform.rotation.w);

      tf2::Quaternion q_correction;
      q_correction.setRPY(0, 0, M_PI);

      tf2::Quaternion q_rotated = q_orig * q_correction;
      q_rotated.normalize();

      geometry_msgs::msg::TransformStamped rotated_tf;
      rotated_tf.header.stamp    = get_clock()->now();
      rotated_tf.header.frame_id = "base_link";
      rotated_tf.child_frame_id  = "aruco_link_rotated";

      rotated_tf.transform.translation = original_tf.transform.translation;
      rotated_tf.transform.rotation.x = q_rotated.x();
      rotated_tf.transform.rotation.y = q_rotated.y();
      rotated_tf.transform.rotation.z = q_rotated.z();
      rotated_tf.transform.rotation.w = q_rotated.w();

      // 3. Broadcast & publish the rotated TF
      tf_broadcaster_->sendTransform(rotated_tf);
      transform_publisher_->publish(rotated_tf);

      /* ---------- 4. Compute & publish distance -------------------- */
      const auto &t = original_tf.transform.translation;
      double distance = std::hypot(std::hypot(t.x, t.y), t.z);

      std_msgs::msg::Float64 distance_msg;
      distance_msg.data = distance;
      distance_publisher_->publish(distance_msg);

      RCLCPP_DEBUG(get_logger(),
                   "aruco_link at (%.3f, %.3f, %.3f) → distance %.3f m",
                   t.x, t.y, t.z, distance);
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
  rclcpp::spin(std::make_shared<TransformPublisher>());
  rclcpp::shutdown();
  return 0;
}







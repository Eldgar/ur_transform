#include "rclcpp/rclcpp.hpp"
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"

class TransformPublisher : public rclcpp::Node
{
public:
    TransformPublisher() : Node("transform_publisher"),
                           tf_buffer_(this->get_clock()),
                           tf_listener_(tf_buffer_)
    {
        // Publisher for transform
        transform_publisher_ = this->create_publisher<geometry_msgs::msg::TransformStamped>("/ur_transform", 10);

        // Timer to periodically publish the transform
        timer_ = this->create_wall_timer(
            std::chrono::milliseconds(30),
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
            // Lookup transform
            geometry_msgs::msg::TransformStamped transform_stamped;
            transform_stamped = tf_buffer_.lookupTransform("base_link", "rg2_gripper_base_link", tf2::TimePointZero);

            // Publish the transform with BOTH position and orientation
            transform_publisher_->publish(transform_stamped);

            // Log both position and rotation
            RCLCPP_INFO(this->get_logger(),
                        "Published Transform: x=%.3f, y=%.3f, z=%.3f | qx=%.3f, qy=%.3f, qz=%.3f, qw=%.3f",
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

                 


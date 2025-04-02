#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <image_transport/image_transport.hpp>
#include <Eigen/Geometry>
#include <deque>

class ArucoDetectorNode : public rclcpp::Node
{
public:
  ArucoDetectorNode()
  : Node("aruco_detector_cpp"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
  {
    RCLCPP_INFO(this->get_logger(), "Starting ArUco Detector (C++)");

    camera_matrix_ = (cv::Mat_<double>(3, 3) << 520.78138, 0.0, 320.5,
                                                0.0, 520.78138, 240.5,
                                                0.0, 0.0, 1.0);
    dist_coeffs_ = cv::Mat::zeros(5, 1, CV_64F);
    marker_length_ = 0.045;

    aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    aruco_params_ = cv::aruco::DetectorParameters::create();

    image_sub_ = image_transport::create_subscription(
      this, "/wrist_rgbd_depth_sensor/image_raw",
      std::bind(&ArucoDetectorNode::imageCallback, this, std::placeholders::_1),
      "raw");

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

    RCLCPP_INFO(this->get_logger(), "Waiting for tf: base_link -> rg2_gripper_aruco_link");
    while (rclcpp::ok()) {
      try {
        cached_marker_tf_ = tf_buffer_.lookupTransform("base_link", "rg2_gripper_aruco_link", tf2::TimePointZero);
        break;
      } catch (const tf2::TransformException &ex) {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "TF unavailable: %s", ex.what());
        rclcpp::sleep_for(std::chrono::milliseconds(100));
      }
    }
  }

private:
  void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
    try {
      cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    } catch (cv_bridge::Exception &e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    cv::Mat image = cv_ptr->image;
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    cv::aruco::detectMarkers(gray, aruco_dict_, corners, ids, aruco_params_);

    bool marker_detected = false;
    if (!ids.empty()) {
      std::vector<cv::Vec3d> rvecs, tvecs;
      cv::aruco::estimatePoseSingleMarkers(corners, marker_length_, camera_matrix_, dist_coeffs_, rvecs, tvecs);
      cv::aruco::drawDetectedMarkers(image, corners, ids);

      for (size_t i = 0; i < ids.size(); ++i) {
        cv::Mat R;
        cv::Rodrigues(rvecs[i], R);
        Eigen::Matrix4d T_camera_marker = Eigen::Matrix4d::Identity();
        for (int row = 0; row < 3; ++row) {
          for (int col = 0; col < 3; ++col) {
            T_camera_marker(row, col) = R.at<double>(row, col);
          }
          T_camera_marker(row, 3) = tvecs[i][row];
        }
        Eigen::Matrix4d T_marker_camera = T_camera_marker.inverse();

        Eigen::Quaterniond q_marker(cached_marker_tf_.transform.rotation.w,
                                    cached_marker_tf_.transform.rotation.x,
                                    cached_marker_tf_.transform.rotation.y,
                                    cached_marker_tf_.transform.rotation.z);
        Eigen::Matrix4d T_base_marker = Eigen::Matrix4d::Identity();
        T_base_marker.block<3,3>(0,0) = q_marker.toRotationMatrix();
        T_base_marker(0,3) = cached_marker_tf_.transform.translation.x;
        T_base_marker(1,3) = cached_marker_tf_.transform.translation.y;
        T_base_marker(2,3) = cached_marker_tf_.transform.translation.z;

        Eigen::Matrix4d T_base_camera = T_base_marker * T_marker_camera;
        camera_history_.push_back(T_base_camera);
        if (camera_history_.size() > 10)
          camera_history_.pop_front();

        marker_detected = true;

        // Draw coordinate axis for the detected marker
        cv::aruco::drawAxis(image, camera_matrix_, dist_coeffs_, rvecs[i], tvecs[i], marker_length_ * 0.5);
        break;
      }
    }

    Eigen::Matrix4d average_pose = Eigen::Matrix4d::Zero();
    for (const auto &pose : camera_history_)
      average_pose += pose;
    average_pose /= static_cast<double>(camera_history_.size());

    Eigen::Vector3d t_base_camera = average_pose.block<3,1>(0,3);
    Eigen::Matrix3d R_base_camera = average_pose.block<3,3>(0,0);
    Eigen::Quaterniond q_base_camera(R_base_camera);

    geometry_msgs::msg::TransformStamped tf_msg;
    tf_msg.header.stamp = this->now();
    tf_msg.header.frame_id = "base_link";
    tf_msg.child_frame_id = "camera_link";
    tf_msg.transform.translation.x = t_base_camera.x();
    tf_msg.transform.translation.y = t_base_camera.y();
    tf_msg.transform.translation.z = t_base_camera.z();
    tf_msg.transform.rotation.x = q_base_camera.x();
    tf_msg.transform.rotation.y = q_base_camera.y();
    tf_msg.transform.rotation.z = q_base_camera.z();
    tf_msg.transform.rotation.w = q_base_camera.w();

    tf_broadcaster_->sendTransform(tf_msg);

    cv::imshow("Detected Markers", image);
    
    cv::waitKey(1);
  }

  cv::Mat camera_matrix_;
  cv::Mat dist_coeffs_;
  double marker_length_;
  cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
  cv::Ptr<cv::aruco::DetectorParameters> aruco_params_;

  image_transport::Subscriber image_sub_;
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;

  geometry_msgs::msg::TransformStamped cached_marker_tf_;
  std::deque<Eigen::Matrix4d> camera_history_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ArucoDetectorNode>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}



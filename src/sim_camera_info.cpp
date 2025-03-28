#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>


using Matrix4d = Eigen::Matrix4d;

Matrix4d transformStampedToEigen(const geometry_msgs::msg::TransformStamped &tf) {
  Matrix4d T = Matrix4d::Identity();
  T(0,3) = tf.transform.translation.x;
  T(1,3) = tf.transform.translation.y;
  T(2,3) = tf.transform.translation.z;
  Eigen::Quaterniond q(tf.transform.rotation.w,
                       tf.transform.rotation.x,
                       tf.transform.rotation.y,
                       tf.transform.rotation.z);
  T.block<3,3>(0,0) = q.toRotationMatrix();
  return T;
}

class CameraCalibrator : public rclcpp::Node {
public:
  CameraCalibrator() : Node("camera_calibration_node"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_) {
    RCLCPP_INFO(this->get_logger(), "Camera Calibrator node initiated");
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/wrist_rgbd_depth_sensor/image_raw", 10,
      std::bind(&CameraCalibrator::imageCallback, this, std::placeholders::_1));

    aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    aruco_params_ = cv::aruco::DetectorParameters::create();
    marker_length_ = 0.045; // meters
  }

private:
  void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg) {
    RCLCPP_INFO(this->get_logger(), "Image callback triggered");
    if (already_ran_) {
      RCLCPP_INFO(this->get_logger(), "Already ran once. Skipping.");
      return;
    }

    cv::Mat image;
    try {
      image = cv_bridge::toCvCopy(msg, "bgr8")->image;
    } catch (cv_bridge::Exception &e) {
      RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }

    RCLCPP_INFO(this->get_logger(), "Image received. Detecting markers...");

    std::vector<int> ids;
    std::vector<std::vector<cv::Point2f>> corners;
    cv::aruco::detectMarkers(image, aruco_dict_, corners, ids, aruco_params_);
    if (ids.empty()) {
      RCLCPP_WARN(this->get_logger(), "No ArUco markers detected.");
      return;
    }
    RCLCPP_INFO(this->get_logger(), "ArUco marker(s) detected: %ld", ids.size());

    if (!tf_buffer_.canTransform("base_link", "rg2_gripper_aruco_link", tf2::TimePointZero) ||
        !tf_buffer_.canTransform("base_link", "wrist_rgbd_camera_link", tf2::TimePointZero) ||
        !tf_buffer_.canTransform("wrist_rgbd_camera_link", "wrist_rgbd_camera_depth_optical_frame", tf2::TimePointZero)) {
      RCLCPP_WARN(this->get_logger(), "Required transforms not available yet.");
      return;
    }

    RCLCPP_INFO(this->get_logger(), "TFs available. Proceeding with optimization.");

    geometry_msgs::msg::TransformStamped marker_tf = tf_buffer_.lookupTransform("base_link", "rg2_gripper_aruco_link", tf2::TimePointZero);
    geometry_msgs::msg::TransformStamped camera_link_tf = tf_buffer_.lookupTransform("base_link", "wrist_rgbd_camera_link", tf2::TimePointZero);
    geometry_msgs::msg::TransformStamped optical_tf = tf_buffer_.lookupTransform("wrist_rgbd_camera_link", "wrist_rgbd_camera_depth_optical_frame", tf2::TimePointZero);

    Matrix4d T_base_marker = transformStampedToEigen(marker_tf);
    Matrix4d T_wrist_camera = Matrix4d::Identity();
    T_wrist_camera(0,3) = camera_link_tf.transform.translation.x;
    T_wrist_camera(1,3) = camera_link_tf.transform.translation.y;
    T_wrist_camera(2,3) = camera_link_tf.transform.translation.z;
    Eigen::Quaterniond q_optical(optical_tf.transform.rotation.w,
                                 optical_tf.transform.rotation.x,
                                 optical_tf.transform.rotation.y,
                                 optical_tf.transform.rotation.z);
    T_wrist_camera.block<3,3>(0,0) = q_optical.toRotationMatrix();

    int count = 0;
    int total = 11 * 11 * 11 * 11; // 0.02 steps from -0.1 to 0.1 (inclusive)

    for (double k1 = -0.1; k1 <= 0.1; k1 += 0.05) {
      for (double k2 = -0.1; k2 <= 0.1; k2 += 0.05) {
        for (double p1 = -0.05; p1 <= 0.05; p1 += 0.02) {
          for (double p2 = -0.05; p2 <= 0.05; p2 += 0.02) {
            count++;
            if (count % 500 == 0) {
              std::cout << "Progress: " << count << "/" << total << std::endl;
              RCLCPP_INFO(this->get_logger(), "Progress update: %d / %d", count, total);
            }

            cv::Mat dist_coeffs = (cv::Mat1d(1, 5) << k1, k2, p1, p2, 0);
            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(corners, marker_length_, camera_matrix_, dist_coeffs, rvecs, tvecs);
            if (rvecs.empty()) continue;

            cv::Mat R_cv;
            cv::Rodrigues(rvecs[0], R_cv);
            Eigen::Matrix3d R_marker_camera;
            for (int r = 0; r < 3; ++r)
              for (int c = 0; c < 3; ++c)
                R_marker_camera(r, c) = R_cv.at<double>(r, c);

            Matrix4d T_marker_camera = Matrix4d::Identity();
            T_marker_camera.block<3,3>(0,0) = R_marker_camera;
            T_marker_camera(0,3) = tvecs[0][0];
            T_marker_camera(1,3) = tvecs[0][1];
            T_marker_camera(2,3) = tvecs[0][2];

            Matrix4d T_camera_marker = T_marker_camera.inverse();
            Matrix4d T_base_camera = T_base_marker * T_camera_marker;

            double error = (T_base_camera.block<3,1>(0,3) - T_wrist_camera.block<3,1>(0,3)).norm();
            if (error < best_error_) {
              best_error_ = error;
              best_dist_coeffs_ = dist_coeffs.clone();
              best_T_base_camera_ = T_base_camera;
            }
          }
        }
      }
    }

    std::cout << "Best distortion coefficients found: " << best_dist_coeffs_ << std::endl;
    std::cout << "Computed T_base_camera:\n" << best_T_base_camera_ << std::endl;
    std::cout << "Expected T_wrist_camera:\n" << T_wrist_camera << std::endl;
    already_ran_ = true;
  }

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
  cv::Ptr<cv::aruco::DetectorParameters> aruco_params_;
  double marker_length_;
  cv::Mat camera_matrix_ = (cv::Mat1d(3, 3) << 1041.56, 0, 641.0, 0, 1041.56, 481.0, 0, 0, 1);
  cv::Mat best_dist_coeffs_ = cv::Mat::zeros(1, 5, CV_64F);
  Matrix4d best_T_base_camera_ = Matrix4d::Identity();
  double best_error_ = 1e9;
  bool already_ran_ = false;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<CameraCalibrator>());
  rclcpp::shutdown();
  return 0;
}








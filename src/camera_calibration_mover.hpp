#ifndef CAMERA_CALIBRATION_MOVER_HPP_
#define CAMERA_CALIBRATION_MOVER_HPP_

#include <rclcpp/rclcpp.hpp>

#ifdef SIMULATION
  #include <sensor_msgs/msg/image.hpp>
#else
  #include <sensor_msgs/msg/compressed_image.hpp>
#endif

#include <cv_bridge/cv_bridge.h>
#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <Eigen/Geometry>
#include <vector>

class CameraCalibrationMover : public rclcpp::Node
{
public:
    explicit CameraCalibrationMover(const rclcpp::NodeOptions &options = rclcpp::NodeOptions());
    void initialize();

#ifdef SIMULATION
    using ImageMsg = sensor_msgs::msg::Image;
#else
    using ImageMsg = sensor_msgs::msg::CompressedImage;
#endif

private:
    void imageCallback(const typename ImageMsg::ConstSharedPtr msg);
    bool moveToJointPose(const std::vector<double> &joint_pose);
    Eigen::Isometry3d computeCameraPoseFromImage(const typename ImageMsg::ConstSharedPtr &msg);
    double computePoseError(const Eigen::Isometry3d &ref, const Eigen::Isometry3d &measured, double orientation_weight = 0.15);
    Eigen::Isometry3d averagePose(const std::vector<Eigen::Isometry3d> &poses);
    void collectValidDetections(size_t count, std::vector<typename ImageMsg::ConstSharedPtr> &storage);  // ✅ NEW
    double evaluateK1(double k1);
    void optimizeK1();

    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    rclcpp::Subscription<ImageMsg>::SharedPtr image_sub_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    double marker_length_;

    typename ImageMsg::ConstSharedPtr last_image_;
    std::vector<std::vector<typename ImageMsg::ConstSharedPtr>> all_saved_images_;

    std::vector<std::vector<double>> calibration_poses_;
    std::vector<std::vector<double>> general_joint_poses_;
    std::vector<Eigen::Isometry3d> valid_detections_;
    Eigen::Isometry3d T_B_C_ref_;
};

#endif  // CAMERA_CALIBRATION_MOVER_HPP_



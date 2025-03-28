#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <Eigen/Dense>
#include <Eigen/Geometry>

class ArucoDetector : public rclcpp::Node
{
public:
    ArucoDetector()
    : Node("aruco_detector")
    {
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/wrist_rgbd_depth_sensor/image_raw", 10,
            std::bind(&ArucoDetector::imageCallback, this, std::placeholders::_1));

        transform_sub_ = this->create_subscription<geometry_msgs::msg::TransformStamped>(
            "/ur_transform", 10,
            std::bind(&ArucoDetector::transformCallback, this, std::placeholders::_1));

        image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/aruco_detector/image", 10);
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
        aruco_params_ = cv::aruco::DetectorParameters::create();

        camera_matrix_ = (cv::Mat_<double>(3, 3) << 520.78138, 0.0, 320.5,
                                                    0.0, 520.78138, 240.5,
                                                    0.0, 0.0, 1.0);
        dist_coeffs_ = cv::Mat::zeros(5, 1, CV_64F);
        marker_length_ = 0.045;

        RCLCPP_INFO(this->get_logger(), "ArucoDetector node initialized.");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr transform_sub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    geometry_msgs::msg::TransformStamped latest_marker_tf_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    double marker_length_;

    bool has_transform_ = false;

    void transformCallback(const geometry_msgs::msg::TransformStamped::SharedPtr msg)
    {
        latest_marker_tf_ = *msg;
        has_transform_ = true;
        RCLCPP_INFO(this->get_logger(), "Received marker transform.");
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        if (!has_transform_) {
            RCLCPP_WARN(this->get_logger(), "Waiting for known marker transform...");
            return;
        }

        cv::Mat frame;
        try {
            frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<cv::Vec3d> rvecs, tvecs;

        cv::aruco::detectMarkers(gray, aruco_dict_, corners, ids, aruco_params_);
        if (!ids.empty()) {
            cv::aruco::estimatePoseSingleMarkers(corners, marker_length_, camera_matrix_, dist_coeffs_, rvecs, tvecs);
            cv::aruco::drawDetectedMarkers(frame, corners, ids);

            for (size_t i = 0; i < ids.size(); ++i) {
                cv::aruco::drawAxis(frame, camera_matrix_, dist_coeffs_, rvecs[i], tvecs[i], marker_length_ * 0.5);
                publishCameraPose(rvecs[i], tvecs[i], ids[i]);
            }
        } else {
            RCLCPP_WARN(this->get_logger(), "No ArUco markers detected.");
        }

        auto out_msg = cv_bridge::CvImage(msg->header, "bgr8", frame).toImageMsg();
        image_pub_->publish(*out_msg);
    }

    void publishCameraPose(const cv::Vec3d &rvec, const cv::Vec3d &tvec, int marker_id)
    {
        cv::Mat R;
        cv::Rodrigues(rvec, R);
        Eigen::Matrix4d T_camera_marker = Eigen::Matrix4d::Identity();
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                T_camera_marker(r, c) = R.at<double>(r, c);
        T_camera_marker(0, 3) = tvec[0];
        T_camera_marker(1, 3) = tvec[1];
        T_camera_marker(2, 3) = tvec[2];

        Eigen::Matrix4d T_marker_camera = T_camera_marker.inverse();

        tf2::Quaternion q(
            latest_marker_tf_.transform.rotation.x,
            latest_marker_tf_.transform.rotation.y,
            latest_marker_tf_.transform.rotation.z,
            latest_marker_tf_.transform.rotation.w);
        tf2::Matrix3x3 tf_R(q);

        Eigen::Matrix4d T_base_marker = Eigen::Matrix4d::Identity();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                T_base_marker(i, j) = tf_R[i][j];
        T_base_marker(0, 3) = latest_marker_tf_.transform.translation.x;
        T_base_marker(1, 3) = latest_marker_tf_.transform.translation.y;
        T_base_marker(2, 3) = latest_marker_tf_.transform.translation.z;

        Eigen::Matrix4d T_base_camera = T_base_marker * T_marker_camera;

        Eigen::Vector3d t(T_base_camera.block<3, 1>(0, 3));
        Eigen::Quaterniond quat(T_base_camera.block<3, 3>(0, 0));

        geometry_msgs::msg::TransformStamped tf_msg;
        tf_msg.header.stamp = this->get_clock()->now();
        tf_msg.header.frame_id = "base_link";
        tf_msg.child_frame_id = "camera_link";
        tf_msg.transform.translation.x = t.x();
        tf_msg.transform.translation.y = t.y();
        tf_msg.transform.translation.z = t.z();
        tf_msg.transform.rotation.x = quat.x();
        tf_msg.transform.rotation.y = quat.y();
        tf_msg.transform.rotation.z = quat.z();
        tf_msg.transform.rotation.w = quat.w();

        tf_broadcaster_->sendTransform(tf_msg);

        compareCameraPoseError(T_base_camera);

        RCLCPP_INFO(this->get_logger(), "Published transform for marker %d at (%.3f, %.3f, %.3f)",
                    marker_id, t.x(), t.y(), t.z());
    }

    void compareCameraPoseError(const Eigen::Matrix4d &T_base_camera)
    {
        geometry_msgs::msg::TransformStamped tf_translation_ref, tf_orientation_ref;

        try {
            tf_translation_ref = tf_buffer_->lookupTransform("base_link", "wrist_rgbd_camera_link", tf2::TimePointZero);
            tf_orientation_ref = tf_buffer_->lookupTransform("base_link", "wrist_rgbd_camera_depth_optical_frame", tf2::TimePointZero);
        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "TF Lookup failed: %s", ex.what());
            return;
        }

        Eigen::Vector3d t_est(T_base_camera.block<3, 1>(0, 3));
        Eigen::Quaterniond q_est(T_base_camera.block<3, 3>(0, 0));

        Eigen::Vector3d t_ref(tf_translation_ref.transform.translation.x,
                              tf_translation_ref.transform.translation.y,
                              tf_translation_ref.transform.translation.z);

        tf2::Quaternion q_tf;
        tf2::fromMsg(tf_orientation_ref.transform.rotation, q_tf);
        Eigen::Quaterniond q_ref(q_tf.w(), q_tf.x(), q_tf.y(), q_tf.z());

        double translation_error = (t_est - t_ref).norm();
        double angle_error_rad = q_est.angularDistance(q_ref);

        RCLCPP_INFO(this->get_logger(),
            "Camera Pose Error:\n  Translation error: %.4f m\n  Orientation error: %.4f rad (%.2f deg)",
            translation_error, angle_error_rad, angle_error_rad * 180.0 / M_PI);
    }
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ArucoDetector>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}








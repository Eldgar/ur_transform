#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <image_transport/image_transport.hpp>
#include <Eigen/Geometry>
#include <cmath>
#include <optional>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/exceptions.h>

class ArucoDetectorNode : public rclcpp::Node
{
public:
    ArucoDetectorNode()
        : Node("aruco_detector_cpp"),
        tf_broadcaster_(std::make_shared<tf2_ros::TransformBroadcaster>(this)),
        tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_)
    {
        RCLCPP_INFO(this->get_logger(), "Starting ArUco Detector (marker error evaluation using /ur_transform)");

        camera_matrix_ = (cv::Mat_<double>(3, 3) << 520.78138, 0.0, 320.5,
                                                    0.0, 520.78138, 240.5,
                                                    0.0, 0.0, 1.0);
        dist_coeffs_ = cv::Mat::zeros(5, 1, CV_64F);
        marker_length_ = 0.045; // meters

        aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
        aruco_params_ = cv::makePtr<cv::aruco::DetectorParameters>();

        image_sub_ = image_transport::create_subscription(
            this, "/wrist_rgbd_depth_sensor/image_raw",
            std::bind(&ArucoDetectorNode::imageCallback, this, std::placeholders::_1),
            "raw");

        processed_image_pub_ = image_transport::create_publisher(this, "/aruco_detector/image");

        ur_transform_sub_ = this->create_subscription<geometry_msgs::msg::TransformStamped>(
            "/ur_transform", 10,
            std::bind(&ArucoDetectorNode::mockTransformCallback, this, std::placeholders::_1));
    }

private:
    void mockTransformCallback(const geometry_msgs::msg::TransformStamped::SharedPtr msg)
    {
        latest_mock_transform_ = *msg;
    }

    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
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

        if (!ids.empty()) {
            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(corners, marker_length_, camera_matrix_, dist_coeffs_, rvecs, tvecs);
            cv::aruco::drawDetectedMarkers(image, corners, ids);

            size_t i = 0;

            cv::Mat R_mat;
            cv::Rodrigues(rvecs[i], R_mat);

            Eigen::Matrix3d R_eigen;
            for (int row = 0; row < 3; ++row)
                for (int col = 0; col < 3; ++col)
                    R_eigen(row, col) = R_mat.at<double>(row, col);

            Eigen::Vector3d t_eigen(tvecs[i][0], tvecs[i][1], tvecs[i][2]);

            Eigen::Isometry3d T_C_M = Eigen::Isometry3d::Identity(); // Camera → Marker
            T_C_M.linear() = R_eigen;
            T_C_M.translation() = t_eigen;

            // Publish transform: camera_link → aruco_marker_error
            geometry_msgs::msg::TransformStamped tf_camera_to_marker;
            tf_camera_to_marker.header.stamp = this->get_clock()->now();
            tf_camera_to_marker.header.frame_id = "camera_link";
            tf_camera_to_marker.child_frame_id = "aruco_marker_error";
            tf_camera_to_marker.transform.translation.x = t_eigen.x();
            tf_camera_to_marker.transform.translation.y = t_eigen.y();
            tf_camera_to_marker.transform.translation.z = t_eigen.z();

            Eigen::Quaterniond q(R_eigen);
            q.normalize();
            geometry_msgs::msg::Quaternion q_msg;
            q_msg.w = q.w(); q_msg.x = q.x(); q_msg.y = q.y(); q_msg.z = q.z();
            tf_camera_to_marker.transform.rotation = q_msg;

            tf_broadcaster_->sendTransform(tf_camera_to_marker);
            RCLCPP_DEBUG(this->get_logger(), 
                "Broadcasted TF camera_link → aruco_marker_error | tx: %.3f ty: %.3f tz: %.3f",
                t_eigen.x(), t_eigen.y(), t_eigen.z());

            if (latest_mock_transform_) {
                // Get T_base_to_actual from the message
                const auto &t = latest_mock_transform_->transform.translation;
                const auto &r = latest_mock_transform_->transform.rotation;

                Eigen::Isometry3d T_base_to_actual = Eigen::Isometry3d::Identity();
                T_base_to_actual.translate(Eigen::Vector3d(t.x, t.y, t.z));
                T_base_to_actual.rotate(Eigen::Quaterniond(r.w, r.x, r.y, r.z));

                // Try to get T_base_to_camera from TF
                try {
                    geometry_msgs::msg::TransformStamped tf_base_to_camera =
                        tf_buffer_.lookupTransform("base_link", "camera_link", tf2::TimePointZero);

                    const auto &tc = tf_base_to_camera.transform.translation;
                    const auto &rc = tf_base_to_camera.transform.rotation;

                    Eigen::Isometry3d T_base_to_camera = Eigen::Isometry3d::Identity();
                    T_base_to_camera.translate(Eigen::Vector3d(tc.x, tc.y, tc.z));
                    T_base_to_camera.rotate(Eigen::Quaterniond(rc.w, rc.x, rc.y, rc.z));

                    // Compute full transform: base_link → camera_link → marker_error
                    Eigen::Isometry3d T_base_to_error = T_base_to_camera * T_C_M;

                    // --- Compute errors ---
                    Eigen::Vector3d err_translation = T_base_to_error.translation() - T_base_to_actual.translation();
                    double translation_error = err_translation.norm();

                    Eigen::Quaterniond q_error(T_base_to_error.rotation());
                    Eigen::Quaterniond q_actual(T_base_to_actual.rotation());
                    double angular_error_rad = q_error.angularDistance(q_actual);

                    RCLCPP_INFO(this->get_logger(),
                        "BASE-relative error → Translation: %.4f m | Orientation: %.4f rad",
                        translation_error, angular_error_rad);

                } catch (const tf2::TransformException &ex) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                        "Could not lookup base_link → camera_link: %s", ex.what());
                }

            } else {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "No /ur_transform data received yet.");
            }


            cv::drawFrameAxes(image, camera_matrix_, dist_coeffs_, rvecs[i], tvecs[i], marker_length_ * 0.5f);
        }

        auto processed_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image).toImageMsg();
        processed_image_pub_.publish(*processed_msg);

        cv::imshow("Detected Markers", image);
        cv::waitKey(1);
    }

    // --- Members ---
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    double marker_length_;
    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params_;

    image_transport::Subscriber image_sub_;
    image_transport::Publisher processed_image_pub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr ur_transform_sub_;
    std::optional<geometry_msgs::msg::TransformStamped> latest_mock_transform_;

    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

};

// --- Main ---
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ArucoDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}



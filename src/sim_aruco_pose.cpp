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
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <image_transport/image_transport.hpp>
#include <Eigen/Geometry>
#include <cmath>

class ArucoDetectorNode : public rclcpp::Node
{
public:
    ArucoDetectorNode()
        : Node("aruco_detector_cpp"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
    {
        RCLCPP_INFO(this->get_logger(), "Starting ArUco Detector (marker error evaluation)");

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
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    }

private:
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
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    R_eigen(row, col) = R_mat.at<double>(row, col);
                }
            }

            Eigen::Vector3d t_eigen(tvecs[i][0], tvecs[i][1], tvecs[i][2]);

            Eigen::Isometry3d T_C_M = Eigen::Isometry3d::Identity(); // Camera → Marker
            T_C_M.linear() = R_eigen;
            T_C_M.translation() = t_eigen;

            // Publish transform: camera_link → aruco_marker_error
            geometry_msgs::msg::TransformStamped tf_camera_to_marker;
            tf_camera_to_marker.header.stamp = msg->header.stamp;
            tf_camera_to_marker.header.frame_id = "camera_link";  // Set your camera frame
            tf_camera_to_marker.child_frame_id = "aruco_marker_error";

            tf_camera_to_marker.transform.translation.x = t_eigen.x();
            tf_camera_to_marker.transform.translation.y = t_eigen.y();
            tf_camera_to_marker.transform.translation.z = t_eigen.z();

            Eigen::Quaterniond q(R_eigen);
            q.normalize();
            geometry_msgs::msg::Quaternion q_msg;
            q_msg.w = q.w();
            q_msg.x = q.x();
            q_msg.y = q.y();
            q_msg.z = q.z();
            tf_camera_to_marker.transform.rotation = q_msg;


            tf_broadcaster_->sendTransform(tf_camera_to_marker);

            // Try to lookup real transform: camera_link → rg2_gripper_aruco_link
            try {
                geometry_msgs::msg::TransformStamped tf_actual = tf_buffer_.lookupTransform(
                    "camera_link", "rg2_gripper_aruco_link", tf2::TimePointZero);

                // Convert both transforms to Eigen
                Eigen::Isometry3d T_camera_to_error = T_C_M;

                const auto &t = tf_actual.transform.translation;
                const auto &r = tf_actual.transform.rotation;

                Eigen::Isometry3d T_camera_to_actual = Eigen::Isometry3d::Identity();
                T_camera_to_actual.translate(Eigen::Vector3d(t.x, t.y, t.z));
                T_camera_to_actual.rotate(Eigen::Quaterniond(r.w, r.x, r.y, r.z));


                // --- Compute translation error ---
                Eigen::Vector3d err_translation = T_camera_to_error.translation() - T_camera_to_actual.translation();
                double translation_error = err_translation.norm();

                // --- Compute orientation error ---
                Eigen::Quaterniond q_error(T_camera_to_error.rotation());
                Eigen::Quaterniond q_actual(T_camera_to_actual.rotation());
                double angular_error_rad = q_error.angularDistance(q_actual);

                RCLCPP_INFO(this->get_logger(),
                            "Translation error: %.4f m | Orientation error: %.4f rad",
                            translation_error, angular_error_rad);

            } catch (const tf2::TransformException &ex) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Could not lookup transform from camera_link to rg2_gripper_aruco_link: %s", ex.what());
            }

            // Draw axes on marker
            cv::drawFrameAxes(image, camera_matrix_, dist_coeffs_, rvecs[i], tvecs[i], marker_length_ * 0.5f);
        }

        // Publish annotated image
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


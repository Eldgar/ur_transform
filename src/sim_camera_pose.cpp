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
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp> // Needed for tf2::fromMsg
#include <tf2_eigen/tf2_eigen.hpp> // Needed for tf2::transformToEigen
#include <image_transport/image_transport.hpp>
#include <Eigen/Geometry>
#include <deque>
#include <numeric> // For std::accumulate
#include <vector>

class ArucoDetectorNode : public rclcpp::Node
{
public:
    ArucoDetectorNode()
        : Node("aruco_detector_cpp"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
    {
        RCLCPP_INFO(this->get_logger(), "Starting ArUco Detector (C++)");

        // --- Camera parameters ---
        camera_matrix_ = (cv::Mat_<double>(3, 3) << 520.78138, 0.0, 320.5,
                                                    0.0, 520.78138, 240.5,
                                                    0.0, 0.0, 1.0);
        dist_coeffs_ = cv::Mat::zeros(5, 1, CV_64F);
        marker_length_ = 0.045; // meters

        // --- ArUco setup ---
        aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
        // aruco_params_ = cv::aruco::DetectorParameters::create(); // Deprecated in OpenCV 4+
        aruco_params_ = cv::makePtr<cv::aruco::DetectorParameters>(); // Use this for OpenCV 4+
        // Consider adding refinement parameters if needed:
        // aruco_params_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;

        // --- ROS Setup ---
        image_sub_ = image_transport::create_subscription(
            this, "/wrist_rgbd_depth_sensor/image_raw",
            std::bind(&ArucoDetectorNode::imageCallback, this, std::placeholders::_1),
            "raw");
        processed_image_pub_ = image_transport::create_publisher(this, "/aruco_detector/image");
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        RCLCPP_INFO(this->get_logger(), "ArUco Detector (C++) Initialized.");
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

        bool marker_processed_this_frame = false; // Flag to track if we added a pose this frame

        if (!ids.empty()) {
            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(corners, marker_length_, camera_matrix_, dist_coeffs_, rvecs, tvecs);
            cv::aruco::drawDetectedMarkers(image, corners, ids);

            // Process the first detected marker (or modify if you need to handle specific IDs)
            size_t i = 0; // Index of the marker to process

            // --- 1. Get Marker Pose relative to Camera (T_camera_marker -> T_C_M) ---
            cv::Mat R_mat;
            cv::Rodrigues(rvecs[i], R_mat); // Convert Rodrigues vector to rotation matrix
            Eigen::Matrix3d R_eigen;
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    R_eigen(row, col) = R_mat.at<double>(row, col);
                }
            }

            Eigen::Vector3d t_eigen(tvecs[i][0], tvecs[i][1], tvecs[i][2]);

            Eigen::Isometry3d T_C_M = Eigen::Isometry3d::Identity(); // Transform Camera <- Marker
            T_C_M.linear() = R_eigen;
            T_C_M.translation() = t_eigen;

            // --- 2. Get Camera Pose relative to Marker (T_marker_camera -> T_M_C) ---
            Eigen::Isometry3d T_M_C = T_C_M.inverse(); // Transform Marker <- Camera

            // --- 3. Get CURRENT Marker Pose relative to Base (T_base_marker -> T_B_M) ---
            geometry_msgs::msg::TransformStamped tf_base_to_marker;
            Eigen::Isometry3d T_B_M = Eigen::Isometry3d::Identity(); // Transform Base <- Marker
            bool tf_lookup_success = false;
            try {
                // Lookup the LATEST available transform
                tf_base_to_marker = tf_buffer_.lookupTransform("base_link", "rg2_gripper_aruco_link", tf2::TimePointZero); // Get latest transform
                // Convert geometry_msgs::TransformStamped to Eigen::Isometry3d
                T_B_M = tf2::transformToEigen(tf_base_to_marker);
                tf_lookup_success = true;
            } catch (const tf2::TransformException &ex) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, // Log every 2s
                                    "Could not lookup transform from base_link to rg2_gripper_aruco_link: %s", ex.what());
            }

            // --- 4. Calculate Camera Pose relative to Base (T_base_camera -> T_B_C) ---
            if (tf_lookup_success) {
                Eigen::Isometry3d T_B_C = T_B_M * T_M_C; // Base <- Marker <- Camera

                // --- Averaging Filter (Optional but recommended) ---
                camera_pose_history_.push_back(T_B_C);
                if (camera_pose_history_.size() > 10) { // Keep history size manageable
                    camera_pose_history_.pop_front();
                }
                marker_processed_this_frame = true;

                // Draw coordinate axis for the detected marker in the image
                cv::drawFrameAxes(image, camera_matrix_, dist_coeffs_, rvecs[i], tvecs[i], marker_length_ * 0.5f); // Use drawFrameAxes
            }
        }

        // --- Publish the processed image with axes drawn ---
        sensor_msgs::msg::Image::SharedPtr processed_image_msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", image).toImageMsg();
        processed_image_pub_.publish(*processed_image_msg);

        // --- Visualization ---
        cv::imshow("Detected Markers", image);
        cv::waitKey(1); // Necessary for imshow to refresh
    }


    // --- Helper function for averaging poses (more robust method) ---
    Eigen::Isometry3d calculateAveragePose(const std::deque<Eigen::Isometry3d>& poses) {
        if (poses.empty()) {
            return Eigen::Isometry3d::Identity();
        }

        // Average Translation
        Eigen::Vector3d avg_translation = Eigen::Vector3d::Zero();
        for (const auto& pose : poses) {
            avg_translation += pose.translation();
        }
        avg_translation /= static_cast<double>(poses.size());

        // Average Rotation (using Quaternion averaging - simple Nlerp approximation)
        std::vector<Eigen::Quaterniond> quaternions;
        quaternions.reserve(poses.size());
        for (const auto& pose : poses) {
            quaternions.emplace_back(pose.linear());
        }

        Eigen::Quaterniond avg_quat = quaternions[0]; // Start with the first one
        for (size_t i = 1; i < quaternions.size(); ++i) {
            // Ensure quaternions are in the same hemisphere for interpolation
            if (avg_quat.dot(quaternions[i]) < 0.0) {

                quaternions[i] = Eigen::Quaterniond(-quaternions[i].coeffs());
            }
            // Simple incremental averaging (can be improved with more sophisticated methods)
             double factor = 1.0 / (static_cast<double>(i) + 1.0);
             avg_quat = avg_quat.slerp(factor, quaternions[i]); // Use slerp
        }
        avg_quat.normalize();


        Eigen::Isometry3d avg_pose = Eigen::Isometry3d::Identity();
        avg_pose.translation() = avg_translation;
        avg_pose.linear() = avg_quat.toRotationMatrix();
        return avg_pose;
    }


    // --- Member Variables ---
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    double marker_length_;
    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params_;

    image_transport::Publisher processed_image_pub_;


    image_transport::Subscriber image_sub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::deque<Eigen::Isometry3d> camera_pose_history_;

};

// --- Main Function (remains the same) ---
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ArucoDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}






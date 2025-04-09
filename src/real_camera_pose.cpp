#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp> // Use CompressedImage
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp> // Needed for conversions
#include <tf2_eigen/tf2_eigen.hpp>             // Needed for conversions
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
        RCLCPP_INFO(this->get_logger(), "Starting ArUco Detector (C++) - Real Robot");

        // --- Camera parameters (RealSense D415 specific - verify these!) ---
        camera_matrix_ = (cv::Mat_<double>(3, 3) <<
                          306.805847, 0.0,         214.441849,
                          0.0,         306.642456, 124.910301,
                          0.0,         0.0,         1.0);

        // Assuming no distortion, otherwise update these coefficients
        dist_coeffs_ = (cv::Mat_<double>(5, 1) <<
                        0.0, 0.0, 0.0, 0.0, 0.0);
        marker_length_ = 0.045; // meters

        // --- ArUco setup ---
        aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
        // aruco_params_ = cv::aruco::DetectorParameters::create(); // Deprecated in OpenCV 4+
        aruco_params_ = cv::makePtr<cv::aruco::DetectorParameters>(); // Use this for OpenCV 4+

        // --- Corner Refinement Parameters (Good for real-world noise) ---
        aruco_params_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
        aruco_params_->cornerRefinementWinSize = 3; // Smaller window might be faster/less prone to grabbing wrong features
        aruco_params_->cornerRefinementMaxIterations = 20;
        aruco_params_->cornerRefinementMinAccuracy = 0.1; // Increased accuracy requirement slightly

        // --- ROS Setup ---
        // Subscribe to the compressed image topic
        image_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            "/D415/color/image_raw/compressed", // Make sure this topic name is correct
            rclcpp::SensorDataQoS(), // Use SensorDataQoS for reliability
            std::bind(&ArucoDetectorNode::imageCallback, this, std::placeholders::_1));

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

        // !! REMOVED CACHING OF MARKER TF !!
        // No need to wait or cache the base_link -> aruco_link_rotated transform here.

        RCLCPP_INFO(this->get_logger(), "ArUco Detector (C++) - Real Robot Initialized.");
    }

private:
    void imageCallback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg)
    {
        cv::Mat image;
        // Decode the compressed image using OpenCV
        try {
            image = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
            if (image.empty()) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "Decoded image is empty.");
                return;
            }
        } catch (const cv::Exception &e) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000, "cv::imdecode exception: %s", e.what());
            return;
        }

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

            //--- 1. Get Marker Pose relative to Camera (T_camera_marker -> T_C_M) ---
            cv::Mat R_mat;
            cv::Rodrigues(rvecs[i], R_mat); // Convert Rodrigues vector to rotation matrix
            Eigen::Matrix3d R_eigen;
            // Manual conversion needed as cv::cv2eigen might not handle CV_64F directly depending on version
            for (int row = 0; row < 3; ++row) {
                for (int col = 0; col < 3; ++col) {
                    R_eigen(row, col) = R_mat.at<double>(row, col);
                }
            }

            Eigen::Vector3d t_eigen(tvecs[i][0], tvecs[i][1], tvecs[i][2]);

            Eigen::Isometry3d T_C_M = Eigen::Isometry3d::Identity(); // Transform Camera <- Marker
            T_C_M.linear() = R_eigen;
            T_C_M.translation() = t_eigen;

            //--- 2. Get Camera Pose relative to Marker (T_marker_camera -> T_M_C) ---
            Eigen::Isometry3d T_M_C = T_C_M.inverse(); // Transform Marker <- Camera

            //--- 3. Get CURRENT Marker Pose relative to Base (T_base_marker -> T_B_M) ---
            //    NOTE: Ensure 'aruco_link_rotated' frame is being published correctly relative to 'base_link'
            geometry_msgs::msg::TransformStamped tf_base_to_marker;
            Eigen::Isometry3d T_B_M = Eigen::Isometry3d::Identity(); // Transform Base <- Marker
            bool tf_lookup_success = false;
            std::string target_frame = "base_link"; // Or your robot's base frame
            std::string source_frame = "aruco_link_rotated"; // The frame attached to the physical marker
            try {
                tf_base_to_marker = tf_buffer_.lookupTransform(target_frame, source_frame, tf2::TimePointZero); // Get latest transform
                T_B_M = tf2::transformToEigen(tf_base_to_marker); // Convert to Eigen
                tf_lookup_success = true;
            } catch (const tf2::TransformException &ex) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, // Log every 2s
                                     "Could not lookup transform from %s to %s: %s",
                                     target_frame.c_str(), source_frame.c_str(), ex.what());
            }

            //--- 4. Calculate Camera Pose relative to Base (T_base_camera -> T_B_C) ---
            if (tf_lookup_success) {
                Eigen::Isometry3d T_B_C = T_B_M * T_M_C; // Base <- Marker <- Camera

                // Add to history for averaging
                camera_history_.push_back(T_B_C);
                if (camera_history_.size() > 10) { // Keep history size manageable
                     camera_history_.pop_front();
                }
                marker_processed_this_frame = true;

                // Draw coordinate axis for the detected marker in the image
                // cv::aruco::drawAxis(image, camera_matrix_, dist_coeffs_, rvecs[i], tvecs[i], marker_length_ * 0.5); // Old
                cv::drawFrameAxes(image, camera_matrix_, dist_coeffs_, rvecs[i], tvecs[i], marker_length_ * 0.5f); // Use drawFrameAxes
            }
        } // end if(!ids.empty())

        // --- Calculate Average Pose (Improved Method) ---
        Eigen::Isometry3d average_pose_B_C = Eigen::Isometry3d::Identity();
        if (!camera_history_.empty()) {
           average_pose_B_C = calculateAveragePose(camera_history_);
        } else if (marker_processed_this_frame) {
            // If history was empty but we just added one, use the single measurement
             average_pose_B_C = camera_history_.front();
        } else {
            // No marker detected recently, maybe don't publish or publish last known good?
             RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 5000,"No marker poses in history, publishing Identity or last known.");
             // For stability, let's use the last calculated average if history becomes empty
             // We need a member variable to store the last good pose
             // For now, just publish identity if empty.
             // Consider adding: if(!last_published_pose_is_valid_) return; else average_pose_B_C = last_published_pose;
        }


        // --- Publish the averaged transform ---
        geometry_msgs::msg::TransformStamped tf_msg;
        // Use the timestamp from the image message for better TF timing coherence
        tf_msg.header.stamp = msg->header.stamp;
        tf_msg.header.frame_id = "base_link"; // The parent frame
        tf_msg.child_frame_id = "camera_color_optical_frame"; // Child frame - IMPORTANT: Match the actual camera frame name expected by other nodes! Verify this. It's often something like 'camera_color_optical_frame' or 'D415_color_optical_frame' for RealSense.

        // Convert Eigen::Isometry3d back to geometry_msgs::Transform components explicitly
        // Assign translation components directly to resolve potential ambiguity
        const Eigen::Vector3d& avg_translation = average_pose_B_C.translation();
        tf_msg.transform.translation.x = avg_translation.x();
        tf_msg.transform.translation.y = avg_translation.y();
        tf_msg.transform.translation.z = avg_translation.z();

        // Convert rotation part (via Eigen::Quaterniond)
        Eigen::Quaterniond q_avg(average_pose_B_C.linear());
        tf_msg.transform.rotation = tf2::toMsg(q_avg); // Convert Eigen quaternion to geometry_msgs quaternion

        tf_broadcaster_->sendTransform(tf_msg);

        // --- Visualization (Optional) ---
        // Consider downscaling the image for display if it's large
        // cv::Mat display_image;
        // cv::resize(image, display_image, cv::Size(), 0.5, 0.5); // Example: 50% reduction
        // cv::imshow("Detected Markers", display_image);
        cv::imshow("Detected Markers", image); // Show original size
        cv::waitKey(1); // Necessary for imshow to refresh

    } // end imageCallback

    // --- Helper function for averaging poses (more robust method) ---
    Eigen::Isometry3d calculateAveragePose(const std::deque<Eigen::Isometry3d>& poses) {
        if (poses.empty()) {
            RCLCPP_WARN_ONCE(this->get_logger(), "CalculateAveragePose called with empty pose deque.");
            return Eigen::Isometry3d::Identity(); // Return identity if deque is empty
        }

        // Average Translation
        Eigen::Vector3d avg_translation = Eigen::Vector3d::Zero();
        for (const auto& pose : poses) {
            avg_translation += pose.translation();
        }
        avg_translation /= static_cast<double>(poses.size());

        // Average Rotation (using Quaternion averaging - simplified Nlerp)
        // For better accuracy with large rotations, consider more advanced averaging methods.
        std::vector<Eigen::Quaterniond> quaternions;
        quaternions.reserve(poses.size());
        for (const auto& pose : poses) {
            // Ensure the quaternion is normalized before adding
             quaternions.emplace_back(Eigen::Quaterniond(pose.linear()).normalized());
        }

        Eigen::Quaterniond avg_quat = quaternions[0];

        // Iterative Nlerp (approximates Slerp mean for small variations)
        for (size_t i = 1; i < quaternions.size(); ++i) {
            // Ensure quaternions are in the same hemisphere for interpolation
            if (avg_quat.dot(quaternions[i]) < 0.0) {
                 quaternions[i] = Eigen::Quaterniond(-quaternions[i].coeffs());
            }
            // Simple incremental Nlerp: average coeffs and normalize
             double factor = 1.0 / (static_cast<double>(i) + 1.0);
             avg_quat = Eigen::Quaterniond(avg_quat.coeffs() * (1.0 - factor) + quaternions[i].coeffs() * factor);
             avg_quat.normalize(); // Re-normalize after interpolation
        }
        // Final normalization is important
        avg_quat.normalize();

        Eigen::Isometry3d avg_pose = Eigen::Isometry3d::Identity();
        avg_pose.translation() = avg_translation;
        avg_pose.linear() = avg_quat.toRotationMatrix();
        return avg_pose;
    }


    // --- Member Variables ---
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_; // Listener needs the buffer

    // !! REMOVED cached_marker_tf_ !!

    // Use Isometry3d for poses and history
    std::deque<Eigen::Isometry3d> camera_history_;

    // Calibration & detection
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    double marker_length_;
    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params_;

}; // End class ArucoDetectorNode

// --- Main Function (remains the same) ---
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ArucoDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}




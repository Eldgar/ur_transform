#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <Eigen/Geometry>
#include <deque>
#include <numeric>
#include <vector>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64.hpp>

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
        
        processed_image_pub_ = image_transport::create_publisher(this, "/aruco_detector/image");
        found_pub_ = this->create_publisher<std_msgs::msg::Bool>("/aruco_detector/found", 10);
        pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseStamped>("/aruco_detector/pose", 10);
        distance_pub_ = this->create_publisher<std_msgs::msg::Float64>("/aruco_detector/distance", 10);

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);

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

        cv::Mat gray, equalized, thresh;

        // === Stage 1: Grayscale
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // === Stage 2: CLAHE
        static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8,8));
        clahe->apply(gray, equalized);

        // === Stage 3: Gamma correction (modifies equalized in-place)
        equalized.convertTo(equalized, CV_32F, 1.0 / 255.0);
        cv::pow(equalized, 0.7, equalized);
        equalized.convertTo(equalized, CV_8U, 255);

        std::vector<std::pair<std::string, cv::Mat>> stages = {
            {"gray", gray},
            {"clahe", gray},
            {"gamma", equalized},  // after gamma correction
        };

        // === Marker Detection Loop
        std::vector<int> ids;
        std::vector<int> detected_ids;
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<std::vector<cv::Point2f>> detected_corners;
        cv::Mat detection_input;
        std::string detection_stage;

        for (const auto& [stage_name, img] : stages)
        {
            ids.clear(); corners.clear();
            cv::aruco::detectMarkers(img, aruco_dict_, corners, ids, aruco_params_);

            if (!ids.empty()) {
                detection_input = img;
                detection_stage = stage_name;
                detected_ids = ids;
                detected_corners = corners;
                RCLCPP_INFO(this->get_logger(), "Marker detected on image %s", stage_name.c_str());
                break;
            }
        }
        cv::Mat debug_img;
        std_msgs::msg::Bool found_msg;
        found_msg.data = !ids.empty();
        found_pub_->publish(found_msg);
        // === If marker found, display annotated image
        if (!ids.empty()) {
            
            if (detection_input.channels() == 1)
                cv::cvtColor(detection_input, debug_img, cv::COLOR_GRAY2BGR);
            else
                debug_img = detection_input.clone();

            cv::aruco::drawDetectedMarkers(debug_img, corners, ids);
            cv::imshow("✅ Marker Detected at Stage: " + detection_stage, debug_img);
            cv::waitKey(1);
        } else {
            cv::imshow("❌ No Marker Detected", image);
            cv::waitKey(1);
        }

        cv::waitKey(3);  // Required for OpenCV GUI updates


        bool marker_processed_this_frame = false; // Flag to track if we added a pose this frame

       if (!detected_ids.empty()) {
            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(corners, marker_length_, camera_matrix_, dist_coeffs_, rvecs, tvecs);
            cv::aruco::drawDetectedMarkers(image, corners, ids);
            

            // Process the first detected marker (or modify if you need to handle specific IDs)
            size_t i = 0; // Index of the marker to process

            //--- 1. Publish marker pose wrt camera -------------------------------
            cv::Mat R_mat;
            cv::Rodrigues(rvecs[i], R_mat);                 // rotation matrix

            Eigen::Matrix3d R_eigen;
            for (int r = 0; r < 3; ++r)
              for (int c = 0; c < 3; ++c)
                R_eigen(r,c) = R_mat.at<double>(r,c);

            Eigen::Quaterniond q_eigen(R_eigen);
            q_eigen.normalize();

            geometry_msgs::msg::PoseStamped pose_msg;
            pose_msg.header.stamp    = msg->header.stamp;   // match the image time
            pose_msg.header.frame_id = "camera_color_optical_frame"; // parent frame

            pose_msg.pose.position.x =  tvecs[i][0];
            pose_msg.pose.position.y =  tvecs[i][1];
            pose_msg.pose.position.z =  tvecs[i][2];
            pose_msg.pose.orientation = tf2::toMsg(q_eigen);

            pose_pub_->publish(pose_msg);
            // ---------------------------------------------------------------------

            // Compute Euclidean distance to marker
            double marker_distance = std::sqrt(
                std::pow(tvecs[i][0], 2) +
                std::pow(tvecs[i][1], 2) +
                std::pow(tvecs[i][2], 2));

            std_msgs::msg::Float64 distance_msg;
            distance_msg.data = marker_distance;
            distance_pub_->publish(distance_msg);

            RCLCPP_DEBUG(this->get_logger(), "Marker distance: %.3f m", marker_distance);

            Eigen::Isometry3d T_C_M = Eigen::Isometry3d::Identity();

            // Ignore far markers if we already have a camera pose
            if (marker_distance > 0.47 && !camera_history_.empty()) {
                RCLCPP_INFO(this->get_logger(), "Marker detected but too far (%.2f m) ignoring this frame.", marker_distance);
            } else {
                // --- Continue with marker processing ---

                T_C_M.linear() = R_eigen.matrix();
                T_C_M.translation() = Eigen::Vector3d(tvecs[i][0], tvecs[i][1], tvecs[i][2]);

                Eigen::Isometry3d T_M_C = T_C_M.inverse();

                geometry_msgs::msg::TransformStamped tf_base_to_marker;
                Eigen::Isometry3d T_B_M = Eigen::Isometry3d::Identity();
                bool tf_lookup_success = false;
                std::string target_frame = "base_link";
                std::string source_frame = "aruco_link_rotated";
                try {
                    tf_base_to_marker = tf_buffer_.lookupTransform(target_frame, source_frame, tf2::TimePointZero);
                    T_B_M = tf2::transformToEigen(tf_base_to_marker);
                    tf_lookup_success = true;
                } catch (const tf2::TransformException &ex) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                        "Could not lookup transform from %s to %s: %s",
                                        target_frame.c_str(), source_frame.c_str(), ex.what());
                }

                if (tf_lookup_success) {
                    Eigen::Isometry3d T_B_C = T_B_M * T_M_C;
                    camera_history_.push_back(T_B_C);
                    if (camera_history_.size() > 4) {
                        camera_history_.pop_front();
                    }
                    marker_processed_this_frame = true;

                    cv::drawFrameAxes(image, camera_matrix_, dist_coeffs_, rvecs[i], tvecs[i], marker_length_ * 0.5f);
                }
            }

            T_C_M.linear() = R_eigen.matrix();
            T_C_M.translation() = Eigen::Vector3d(tvecs[i][0], tvecs[i][1], tvecs[i][2]);

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
                if (camera_history_.size() > 4) { // Keep history size manageable
                     camera_history_.pop_front();
                }
                marker_processed_this_frame = true;

                // Draw coordinate axis for the detected marker in the image
                // cv::aruco::drawAxis(image, camera_matrix_, dist_coeffs_, rvecs[i], tvecs[i], marker_length_ * 0.5); // Old
                cv::drawFrameAxes(image, camera_matrix_, dist_coeffs_, rvecs[i], tvecs[i], marker_length_ * 0.5f);
            }
        }

        // --- Calculate Average Pose ---
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
        
        cv::imshow("Detected Markers", image); // Show original size
        cv::waitKey(1);


        cv_bridge::CvImage out;
        out.header   = msg->header;
        out.encoding = sensor_msgs::image_encodings::BGR8;
        out.image    = image;
        processed_image_pub_.publish(out.toImageMsg());

    } 

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
    tf2_ros::TransformListener tf_listener_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr found_pub_;
    image_transport::Publisher processed_image_pub_;
    rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr pose_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64>::SharedPtr distance_pub_;

    // Use Isometry3d for poses and history
    std::deque<Eigen::Isometry3d> camera_history_;

    // Calibration & detection
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    double marker_length_;
    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params_;

};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ArucoDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}





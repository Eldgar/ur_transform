#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp> // Needed for subscription
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/quaternion.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
// #include <image_transport/image_transport.hpp> // Removed
#include <Eigen/Geometry>
#include <cmath>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/exceptions.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <limits> // Required for numeric_limits

class ArucoDetectorNode : public rclcpp::Node
{
public:
    ArucoDetectorNode()
        : Node("aruco_detector_cpp"),
        tf_broadcaster_(std::make_shared<tf2_ros::TransformBroadcaster>(this)),
        tf_buffer_(this->get_clock()), // Initialize TF buffer
        tf_listener_(tf_buffer_)      // Initialize TF listener using the buffer
    {

        // --- Parameters for Real Robot (D415) ---
        // Camera intrinsic matrix (Updated values)
        camera_matrix_ = (cv::Mat_<double>(3, 3) <<
                          306.805847, 0.0,        214.441849,
                          0.0,        306.642456, 124.910301,
                          0.0,        0.0,        1.0);
        // Camera distortion coefficients (Assuming zero for now)
        dist_coeffs_ = (cv::Mat_<double>(5, 1) << 0.0, 0.0, 0.0, 0.0, 0.0);
        // Physical size of the ArUco marker side in meters
        marker_length_ = 0.045; // meters

        // --- ArUco Setup ---
        // Define the ArUco dictionary to be used
        aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
        // Create ArUco detector parameters object
        aruco_params_ = cv::makePtr<cv::aruco::DetectorParameters>();

        // --- ROS Communication Setup (No image_transport) ---
        // Standard ROS 2 subscription to the compressed image topic
        // Define QoS profile - SensorDataQoS is often suitable for camera streams
        rmw_qos_profile_t qos_profile = rmw_qos_profile_sensor_data;
        auto qos = rclcpp::QoS(rclcpp::QoSInitialization(qos_profile.history, qos_profile.depth), qos_profile);

        image_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            "/D415/color/image_raw/compressed", // Real camera topic publishing CompressedImage
            qos, // Apply QoS profile
            // Bind the callback, ensuring it matches the expected message type
            std::bind(&ArucoDetectorNode::imageCallback, this, std::placeholders::_1));

        // ---- in constructor, after subscription creation ----
        RCLCPP_INFO(get_logger(), "Subscribed to %s",
                    image_sub_->get_topic_name());

    }

private:

    // Callback function now expects CompressedImage message
    void imageCallback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg)
    {
        RCLCPP_DEBUG(get_logger(), "imageCallback() triggered");

        cv_bridge::CvImagePtr cv_ptr;
        try {
            // Convert the ROS CompressedImage message to an OpenCV image (BGR8 format)
            // cv_bridge handles decompression here if OpenCV has image format support (JPG/PNG)
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception (decompression or conversion): %s", e.what());
            return; // Exit callback if conversion fails
        } catch (cv::Exception &e) {
             RCLCPP_ERROR(this->get_logger(), "OpenCV exception (decompression): %s", e.what());
            return; // Exit callback if conversion fails
        }


        // Get the OpenCV image matrix
        cv::Mat image = cv_ptr->image;
        if (image.empty()) {
             RCLCPP_WARN(this->get_logger(), "Decoded image is empty.");
             return;
        }
        cv::Mat gray;
        // Convert the image to grayscale for marker detection
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        // === Stage 2: CLAHE
        static cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(2.0, cv::Size(8,8));
        cv::Mat equalized;
        clahe->apply(gray, equalized);

        // === Stage 3: Gamma correction
        equalized.convertTo(equalized, CV_32F, 1.0 / 255.0);
        cv::pow(equalized, 0.7, equalized); // gamma < 1 brightens dark regions
        equalized.convertTo(equalized, CV_8U, 255);

        // Run detection on multiple stages to improve robustness
        std::vector<std::pair<std::string, cv::Mat>> stages = {
            {"gray", gray},
            {"clahe", equalized},
            {"gamma", equalized}
        };

        std::vector<int> ids, detected_ids;
        std::vector<std::vector<cv::Point2f>> corners, detected_corners;
        cv::Mat detection_input;
        std::string detection_stage;

        for (const auto &stage : stages) {
            ids.clear(); corners.clear();
            cv::aruco::detectMarkers(stage.second, aruco_dict_, corners, ids, aruco_params_);
            if (!ids.empty()) {
                detection_input = stage.second;
                detection_stage = stage.first;
                detected_ids = ids;
                detected_corners = corners;
                // RCLCPP_INFO(this->get_logger(), "Marker detected on %s stage", detection_stage.c_str());
                break;
            }
        }

        if (detected_ids.empty()) {
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                 "No markers detected after preprocessing stages");
            return;
        }

        // Use the detected set for downstream processing
        ids = detected_ids;
        corners = detected_corners;
        // Provide periodic info log
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000,
                             "detectMarkers (stage %s) found %zu markers", detection_stage.c_str(), ids.size());

        // --- Pose Estimation and Error Calculation (if markers detected) ---
        if (ids.empty()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 2000,
                                 "No markers detected in current frame");
            return;           // leave early
        }

        std::vector<cv::Vec3d> rvecs, tvecs; // Vectors for rotation and translation vectors
        // Estimate the pose of each detected marker relative to the camera
        cv::aruco::estimatePoseSingleMarkers(corners, marker_length_, camera_matrix_, dist_coeffs_, rvecs, tvecs);
        // Draw the detected markers on the color image for visualization
        cv::aruco::drawDetectedMarkers(image, corners, ids);

        // --- Process the first detected marker (index 0) ---
        size_t i = 0; // Index of the marker to process

        // Convert rotation vector (rvecs) to rotation matrix (R_mat) using Rodrigues formula
        cv::Mat R_mat;
        cv::Rodrigues(rvecs[i], R_mat);

        // Convert OpenCV rotation matrix to Eigen rotation matrix
        Eigen::Matrix3d R_eigen;
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col < 3; ++col) {
                R_eigen(row, col) = R_mat.at<double>(row, col);
            }
        }
        // Convert OpenCV translation vector to Eigen translation vector
        Eigen::Vector3d t_eigen(tvecs[i][0], tvecs[i][1], tvecs[i][2]); // Translation: Camera -> Marker

        // Create Eigen::Isometry3d transform representing Camera -> Marker (T_C_M)
        Eigen::Isometry3d T_C_M = Eigen::Isometry3d::Identity();
        T_C_M.linear() = R_eigen;       // Set rotation part
        T_C_M.translation() = t_eigen; // Set translation part

        // --- Publish TF: camera_optical_frame -> aruco_marker_error ---
        geometry_msgs::msg::TransformStamped tf_camera_to_marker;
        // Use the timestamp from the incoming image message for consistency
        tf_camera_to_marker.header.stamp = msg->header.stamp; // Use image timestamp
        // *** IMPORTANT: Use the correct optical frame for your D415 color camera ***
        tf_camera_to_marker.header.frame_id = "D415_color_optical_frame"; // Parent frame (Updated)
        tf_camera_to_marker.child_frame_id = "aruco_marker_error"; // Child frame (detected marker)

        // Set translation
        tf_camera_to_marker.transform.translation.x = t_eigen.x();
        tf_camera_to_marker.transform.translation.y = t_eigen.y();
        tf_camera_to_marker.transform.translation.z = t_eigen.z();

        // Convert Eigen rotation matrix to Eigen quaternion
        Eigen::Quaterniond q_eigen(R_eigen);
        q_eigen.normalize(); // Ensure it's a unit quaternion

        // Set rotation
        tf_camera_to_marker.transform.rotation.w = q_eigen.w();
        tf_camera_to_marker.transform.rotation.x = q_eigen.x();
        tf_camera_to_marker.transform.rotation.y = q_eigen.y();
        tf_camera_to_marker.transform.rotation.z = q_eigen.z();

        // Send the transform using the TF broadcaster
        tf_broadcaster_->sendTransform(tf_camera_to_marker);
        RCLCPP_DEBUG(this->get_logger(),
                     "Broadcasted TF %s -> %s",
                     tf_camera_to_marker.header.frame_id.c_str(),
                     tf_camera_to_marker.child_frame_id.c_str());

        // --- Error Calculation using TF Lookups ---
        try {
            // Define target and source frames for lookups
            const std::string target_frame = "base_link";
            // *** IMPORTANT: Ensure this frame name matches the one used in TF broadcast ***
            const std::string camera_frame = "D415_color_optical_frame"; // Camera frame (Updated)
            // We no longer use TF lookup for the marker; name kept for logging only
            const std::string actual_marker_frame = "aruco_link_rotated";

            // Get the timestamp from the image message (rclcpp::Time)
            rclcpp::Time image_time = msg->header.stamp;
            // Define a timeout for TF lookups (rclcpp::Duration)
            rclcpp::Duration timeout = rclcpp::Duration::from_seconds(0.1);

            // --- Lookup Transform: base_link -> camera_optical_frame ---
            geometry_msgs::msg::TransformStamped tf_base_to_camera_msg =
                tf_buffer_.lookupTransform(target_frame, camera_frame, image_time, timeout);

            // --- Convert looked-up transform to Eigen::Isometry3d ---
            // Base -> Camera
            const auto& tc = tf_base_to_camera_msg.transform.translation;
            const auto& rc = tf_base_to_camera_msg.transform.rotation;
            Eigen::Isometry3d T_base_to_camera = Eigen::Isometry3d::Identity();
            T_base_to_camera.translation() = Eigen::Vector3d(tc.x, tc.y, tc.z);
            T_base_to_camera.linear() = Eigen::Quaterniond(rc.w, rc.x, rc.y, rc.z).toRotationMatrix();

            // Pose of detected marker in base frame
            Eigen::Isometry3d T_base_to_marker = T_base_to_camera * T_C_M;

            // Always log detected pose
            RCLCPP_INFO(this->get_logger(),
                        "Detected marker (base frame): x=%.3f  y=%.3f  z=%.3f m",
                        T_base_to_marker.translation().x(),
                        T_base_to_marker.translation().y(),
                        T_base_to_marker.translation().z());

            // --- Optional error calculation vs aruco_link_rotated (if TF exists) ---
            try {
                geometry_msgs::msg::TransformStamped tf_base_to_gt =
                    tf_buffer_.lookupTransform("base_link", "aruco_link_rotated", image_time, rclcpp::Duration::from_seconds(0.01));

                Eigen::Isometry3d T_base_to_gt = Eigen::Isometry3d::Identity();
                T_base_to_gt.translation() = Eigen::Vector3d(tf_base_to_gt.transform.translation.x,
                                                             tf_base_to_gt.transform.translation.y,
                                                             tf_base_to_gt.transform.translation.z);
                T_base_to_gt.linear() = Eigen::Quaterniond(tf_base_to_gt.transform.rotation.w,
                                                           tf_base_to_gt.transform.rotation.x,
                                                           tf_base_to_gt.transform.rotation.y,
                                                           tf_base_to_gt.transform.rotation.z).toRotationMatrix();

                Eigen::Vector3d err_vec = T_base_to_marker.translation() - T_base_to_gt.translation();
                double trans_err = err_vec.norm();
                // Relative percent error w.r.t. distance from camera to marker
                double cam_distance = t_eigen.norm();
                double rel_pct = cam_distance > std::numeric_limits<double>::epsilon()*100 ? (trans_err / cam_distance) * 100.0 : 0.0;

                Eigen::Quaterniond q_det(T_base_to_marker.rotation());
                Eigen::Quaterniond q_gt(T_base_to_gt.rotation());
                double ang_err = q_det.angularDistance(q_gt);

                RCLCPP_INFO(this->get_logger(),
                            "Error vs aruco_link_rotated -> transl: %.3f m (%.2f %%) | orient: %.2f deg",
                            trans_err,
                            rel_pct,
                            ang_err * 180.0 / M_PI);

                // --- Broadcast projected base_link_error frame ---
                {
                    Eigen::Isometry3d T_marker_gt_to_base = T_base_to_gt.inverse(); // aruco_link_rotated -> base_link
                    geometry_msgs::msg::TransformStamped tf_markererr_to_baseerr;
                    tf_markererr_to_baseerr.header.stamp = msg->header.stamp;
                    tf_markererr_to_baseerr.header.frame_id = "aruco_marker_error"; // parent
                    tf_markererr_to_baseerr.child_frame_id = "base_link_error";      // child

                    const Eigen::Vector3d &p = T_marker_gt_to_base.translation();
                    tf_markererr_to_baseerr.transform.translation.x = p.x();
                    tf_markererr_to_baseerr.transform.translation.y = p.y();
                    tf_markererr_to_baseerr.transform.translation.z = p.z();

                    Eigen::Quaterniond q(T_marker_gt_to_base.rotation());
                    q.normalize();
                    tf_markererr_to_baseerr.transform.rotation.x = q.x();
                    tf_markererr_to_baseerr.transform.rotation.y = q.y();
                    tf_markererr_to_baseerr.transform.rotation.z = q.z();
                    tf_markererr_to_baseerr.transform.rotation.w = q.w();

                    tf_broadcaster_->sendTransform(tf_markererr_to_baseerr);
                }

            } catch (const tf2::TransformException &e) {
                RCLCPP_DEBUG_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                                      "Ground-truth TF unavailable: %s", e.what());
            }

        } catch (const tf2::TransformException &ex) {
            // Log a warning (throttled) if TF lookups fail
            RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000, // Log max once every 2 seconds
                                 "TF lookup failed: %s", ex.what());
        }

        // Draw coordinate axes on the marker for visualization
        cv::drawFrameAxes(image, camera_matrix_, dist_coeffs_, rvecs[i], tvecs[i], marker_length_ * 0.5f);
    }

    // --- Class Member Variables ---
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    double marker_length_;  // Size of the ArUco marker
    cv::Ptr<cv::aruco::Dictionary> aruco_dict_; // ArUco dictionary object
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params_; // ArUco detector parameters

    // Use standard ROS 2 subscriber/publisher pointers
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_; // TF broadcaster object

    // TF Buffer and Listener for transform lookups
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

};

// --- Main Function ---
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv); // Initialize ROS 2
    auto node = std::make_shared<ArucoDetectorNode>();
    rclcpp::spin(node);
    rclcpp::shutdown(); // Shutdown ROS 2
    return 0;
}





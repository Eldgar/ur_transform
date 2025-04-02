#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
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
#include <fstream>
#include <map>
#include <limits>
#include <sstream>

class ArucoDetector : public rclcpp::Node
{
public:
    ArucoDetector()
    : Node("aruco_detector")
    {
        image_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            "/D415/color/image_raw/compressed", 10,
            std::bind(&ArucoDetector::imageCallback, this, std::placeholders::_1));

        transform_sub_ = this->create_subscription<geometry_msgs::msg::TransformStamped>(
            "/ur_transform", 10,
            std::bind(&ArucoDetector::transformCallback, this, std::placeholders::_1));

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
        aruco_params_ = cv::aruco::DetectorParameters::create();

        camera_matrix_ = (cv::Mat_<double>(3, 3) << 306.805847, 0.0, 214.441849,
                                                    0.0, 306.642456, 124.910301,
                                                    0.0, 0.0, 1.0);
        dist_coeffs_ = cv::Mat::zeros(5, 1, CV_64F);
        marker_length_ = 0.045;

        csv_file_.open("/tmp/d415_distortion_optimization.csv");
        csv_file_ << "k1,k2,p1,p2,k3,WeightedError\n";

        RCLCPP_INFO(this->get_logger(), "ArucoDetector node initialized.");
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;
    rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr transform_sub_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    double marker_length_;

    geometry_msgs::msg::TransformStamped latest_marker_tf_;
    bool has_transform_ = false;
    bool has_optimized_ = false;

    std::ofstream csv_file_;

    struct DistortionResult {
        double k1, k2, p1, p2, k3, error;
    };

    std::vector<cv::Mat> detection_frames_;
    std::vector<DistortionResult> best_results_;
    const int required_detections_ = 3;

    void transformCallback(const geometry_msgs::msg::TransformStamped::SharedPtr msg)
    {
        latest_marker_tf_ = *msg;
        has_transform_ = true;
    }

    void imageCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
    {
        if (!has_transform_ || has_optimized_) return;
        if (detection_frames_.size() >= required_detections_) return;

        cv::Mat frame;
        try {
            frame = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv::imdecode failed: %s", e.what());
            return;
        }

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        cv::aruco::detectMarkers(frame, aruco_dict_, corners, ids, aruco_params_);
        if (!ids.empty()) {
            detection_frames_.push_back(frame);
            RCLCPP_INFO(this->get_logger(), "Stored detection %ld/%d", detection_frames_.size(), required_detections_);
            if (detection_frames_.size() == required_detections_) {
                startOptimization();
            }
        }
    }

    void startOptimization()
    {
        if (has_optimized_) return;
        has_optimized_ = true;

        for (size_t i = 0; i < detection_frames_.size(); ++i) {
            double best_err = std::numeric_limits<double>::max();
            DistortionResult best_result;

            for (double k1 = -0.3; k1 <= 0.3; k1 += 0.1)
            for (double k2 = -0.3; k2 <= 0.3; k2 += 0.1)
            for (double p1 = -0.05; p1 <= 0.05; p1 += 0.05)
            for (double p2 = -0.05; p2 <= 0.05; p2 += 0.05)
            for (double k3 = -0.3; k3 <= 0.3; k3 += 0.1)
            {
                cv::Mat dist_coeffs = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);
                double err = evaluatePoseError(detection_frames_[i], dist_coeffs);

                if (err < best_err) {
                    best_err = err;
                    best_result = {k1, k2, p1, p2, k3, err};
                }
            }

            best_results_.push_back(best_result);
            RCLCPP_INFO(this->get_logger(), "Best for frame %lu: [%.3f, %.3f, %.3f, %.3f, %.3f] Error: %.6f",
                        i+1, best_result.k1, best_result.k2, best_result.p1,
                        best_result.p2, best_result.k3, best_result.error);
        }

        finalizeOptimization();
    }

    double evaluatePoseError(const cv::Mat &frame, const cv::Mat &dist_coeffs)
    {
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<cv::Vec3d> rvecs, tvecs;
        cv::aruco::detectMarkers(frame, aruco_dict_, corners, ids, aruco_params_);
        if (ids.empty()) return std::numeric_limits<double>::max();

        cv::aruco::estimatePoseSingleMarkers(corners, marker_length_, camera_matrix_, dist_coeffs, rvecs, tvecs);

        double total_error = 0.0;
        int count = 0;

        for (size_t i = 0; i < ids.size(); ++i) {
            cv::Mat R;
            cv::Rodrigues(rvecs[i], R);
            Eigen::Matrix4d T_camera_marker = Eigen::Matrix4d::Identity();
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    T_camera_marker(r, c) = R.at<double>(r, c);
            T_camera_marker(0, 3) = tvecs[i][0];
            T_camera_marker(1, 3) = tvecs[i][1];
            T_camera_marker(2, 3) = tvecs[i][2];

            Eigen::Matrix4d T_marker_camera = T_camera_marker.inverse();

            Eigen::Matrix4d T_base_marker = Eigen::Matrix4d::Identity();
            tf2::Quaternion q(
                latest_marker_tf_.transform.rotation.x,
                latest_marker_tf_.transform.rotation.y,
                latest_marker_tf_.transform.rotation.z,
                latest_marker_tf_.transform.rotation.w);
            tf2::Matrix3x3 tf_R(q);
            for (int r = 0; r < 3; ++r)
                for (int c = 0; c < 3; ++c)
                    T_base_marker(r, c) = tf_R[r][c];
            T_base_marker(0, 3) = latest_marker_tf_.transform.translation.x;
            T_base_marker(1, 3) = latest_marker_tf_.transform.translation.y;
            T_base_marker(2, 3) = latest_marker_tf_.transform.translation.z;

            Eigen::Matrix4d T_base_camera = T_base_marker * T_marker_camera;

            geometry_msgs::msg::TransformStamped tf_ref;
            try {
                tf_ref = tf_buffer_->lookupTransform("base_link", "D415_color_optical_frame", tf2::TimePointZero);
            } catch (tf2::TransformException &ex) {
                continue;
            }

            Eigen::Vector3d t_est(T_base_camera.block<3, 1>(0, 3));
            Eigen::Quaterniond q_est(T_base_camera.block<3, 3>(0, 0));
            Eigen::Vector3d t_ref(tf_ref.transform.translation.x,
                                  tf_ref.transform.translation.y,
                                  tf_ref.transform.translation.z);

            tf2::Quaternion q_tf;
            tf2::fromMsg(tf_ref.transform.rotation, q_tf);
            Eigen::Quaterniond q_ref(q_tf.w(), q_tf.x(), q_tf.y(), q_tf.z());

            double translation_error = (t_est - t_ref).norm();
            double angle_error_rad = q_est.angularDistance(q_ref);
            total_error += translation_error + 0.1 * angle_error_rad;
            ++count;
        }

        if (count == 0) return std::numeric_limits<double>::max();
        return total_error / count;
    }

    void finalizeOptimization()
    {
        if (best_results_.empty()) {
            RCLCPP_WARN(this->get_logger(), "No optimization results to finalize.");
            rclcpp::shutdown();
            return;
        }

        double sum_k1 = 0, sum_k2 = 0, sum_p1 = 0, sum_p2 = 0, sum_k3 = 0, sum_error = 0;
        for (const auto &res : best_results_) {
            sum_k1 += res.k1;
            sum_k2 += res.k2;
            sum_p1 += res.p1;
            sum_p2 += res.p2;
            sum_k3 += res.k3;
            sum_error += res.error;
        }

        double avg_k1 = sum_k1 / best_results_.size();
        double avg_k2 = sum_k2 / best_results_.size();
        double avg_p1 = sum_p1 / best_results_.size();
        double avg_p2 = sum_p2 / best_results_.size();
        double avg_k3 = sum_k3 / best_results_.size();
        double avg_error = sum_error / best_results_.size();

        csv_file_ << "Average," << avg_k1 << "," << avg_k2 << "," << avg_p1 << ","
                  << avg_p2 << "," << avg_k3 << "," << avg_error << "\n";
        csv_file_.close();

        RCLCPP_INFO(this->get_logger(),
                    "Optimization complete. Averaged distortion: [%.5f, %.5f, %.5f, %.5f, %.5f] with error %.6f",
                    avg_k1, avg_k2, avg_p1, avg_p2, avg_k3, avg_error);

        rclcpp::shutdown();
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
















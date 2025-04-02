#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
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
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/wrist_rgbd_depth_sensor/image_raw", 10,
            std::bind(&ArucoDetector::imageCallback, this, std::placeholders::_1));

        transform_sub_ = this->create_subscription<geometry_msgs::msg::TransformStamped>(
            "/ur_transform", 10,
            std::bind(&ArucoDetector::transformCallback, this, std::placeholders::_1));

        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
        aruco_params_ = cv::aruco::DetectorParameters::create();

        camera_matrix_ = (cv::Mat_<double>(3, 3) << 520.78138, 0.0, 320.5,
                                                    0.0, 520.78138, 240.5,
                                                    0.0, 0.0, 1.0);
        marker_length_ = 0.045;

        csv_file_.open("/tmp/distortion_error_results.csv");
        csv_file_ << "k1,k2,p1,p2,k3,TotalError\n";

        timer_ = this->create_wall_timer(std::chrono::seconds(60), std::bind(&ArucoDetector::finalizeCalibration, this));

        RCLCPP_INFO(this->get_logger(), "ArucoDetector node initialized.");
    }

    ~ArucoDetector()
    {
        if (csv_file_.is_open())
            csv_file_.close();
    }

private:
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr transform_sub_;
    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params_;
    cv::Mat camera_matrix_;
    double marker_length_;
    std::ofstream csv_file_;
    rclcpp::TimerBase::SharedPtr timer_;

    bool has_transform_ = false;
    geometry_msgs::msg::TransformStamped latest_marker_tf_;

    std::map<std::string, double> error_by_distortion_;
    std::string best_distortion_key_;
    double best_error_ = std::numeric_limits<double>::max();

    void transformCallback(const geometry_msgs::msg::TransformStamped::SharedPtr msg)
    {
        latest_marker_tf_ = *msg;
        has_transform_ = true;
    }

    void imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
    {
        if (!has_transform_) return;

        cv::Mat frame;
        try {
            frame = cv_bridge::toCvCopy(msg, "bgr8")->image;
        } catch (cv_bridge::Exception &e) {
            RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
            return;
        }

        for (double k1 = -0.3; k1 <= 0.3; k1 += 0.1)
        for (double k2 = -0.3; k2 <= 0.3; k2 += 0.1)
        for (double p1 = -0.05; p1 <= 0.05; p1 += 0.05)
        for (double p2 = -0.05; p2 <= 0.05; p2 += 0.05)
        for (double k3 = -0.3; k3 <= 0.3; k3 += 0.1)
        {
            cv::Mat dist_coeffs = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);
            std::stringstream key;
            key << k1 << "," << k2 << "," << p1 << "," << p2 << "," << k3;

            double error = evaluatePoseError(frame, dist_coeffs);
            error_by_distortion_[key.str()] = error;

            if (error < best_error_) {
                best_error_ = error;
                best_distortion_key_ = key.str();
            }
        }
    }

    double evaluatePoseError(const cv::Mat &frame, const cv::Mat &dist_coeffs)
    {
        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        std::vector<cv::Vec3d> rvecs, tvecs;

        cv::aruco::detectMarkers(frame, aruco_dict_, corners, ids, aruco_params_);
        if (ids.empty()) return std::numeric_limits<double>::max();

        cv::aruco::estimatePoseSingleMarkers(corners, marker_length_, camera_matrix_, dist_coeffs, rvecs, tvecs);

        cv::Mat R;
        cv::Rodrigues(rvecs[0], R);
        Eigen::Matrix4d T_camera_marker = Eigen::Matrix4d::Identity();
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c < 3; ++c)
                T_camera_marker(r, c) = R.at<double>(r, c);
        T_camera_marker(0, 3) = tvecs[0][0];
        T_camera_marker(1, 3) = tvecs[0][1];
        T_camera_marker(2, 3) = tvecs[0][2];

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

        geometry_msgs::msg::TransformStamped tf_translation_ref, tf_orientation_ref;
        try {
            tf_translation_ref = tf_buffer_->lookupTransform("base_link", "wrist_rgbd_camera_link", tf2::TimePointZero);
            tf_orientation_ref = tf_buffer_->lookupTransform("base_link", "wrist_rgbd_camera_depth_optical_frame", tf2::TimePointZero);
        } catch (tf2::TransformException &ex) {
            return std::numeric_limits<double>::max();
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
        double total_error = translation_error + angle_error_rad * 0.1;

        return total_error;
    }

    void finalizeCalibration()
    {
        for (const auto &entry : error_by_distortion_) {
            csv_file_ << entry.first << "," << entry.second << "\n";
        }
        csv_file_ << "Best," << best_distortion_key_ << "," << best_error_ << "\n";
        RCLCPP_INFO(this->get_logger(), "Calibration complete. Best distortion = [%s] with error %.6f",
                    best_distortion_key_.c_str(), best_error_);
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











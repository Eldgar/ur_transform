#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>

#include <moveit/move_group_interface/move_group_interface.h>

#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <Eigen/Geometry>
#include <chrono>
#include <thread>

using namespace std::chrono_literals;

class CameraCalibrationMover : public rclcpp::Node
{
public:
    CameraCalibrationMover()
        : Node("camera_calibration_mover"), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
    {
        // Init subscribers
        image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            "/wrist_rgbd_depth_sensor/image_raw", rclcpp::SensorDataQoS(),
            std::bind(&CameraCalibrationMover::imageCallback, this, std::placeholders::_1));

        // ArUco setup
        aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
        aruco_params_ = cv::makePtr<cv::aruco::DetectorParameters>();

        camera_matrix_ = (cv::Mat_<double>(3, 3) << 520.78138, 0.0, 320.5,
                                                    0.0, 520.78138, 240.5,
                                                    0.0, 0.0, 1.0);
        dist_coeffs_ = cv::Mat::zeros(5, 1, CV_64F);
        marker_length_ = 0.045;

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    }
    void initialize()
    {
        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(shared_from_this(), "ur_manipulator");
        runCalibration();
    }

private:

    void runCalibration()
    {
        calibration_poses_ = {
            {2.81349, -0.76867, 2.42993, -2.67641, 5.68814, -0.72473},  // ULTIMATE
            {2.73414, -0.85652, 1.89450, -1.70130, 5.74765, -1.05458},
            {2.36194, -1.19495, 2.65155, -2.10590, 5.37617, -1.21476},
            {2.89310, -0.45249, 2.59039, -3.26049, 5.72645, -0.59660},
            {3.05772, -0.50430, 1.90387, -2.79052, 5.77800, -0.28618}
        };

        for (current_pose_index_ = 0; current_pose_index_ < calibration_poses_.size(); ++current_pose_index_)
        {
            std::vector<double> target = calibration_poses_[current_pose_index_];
            RCLCPP_INFO(this->get_logger(), "Moving to pose %lu", current_pose_index_);
            move_group_->setJointValueTarget(target);

            moveit::planning_interface::MoveGroupInterface::Plan plan;
            if (move_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS)
            {
                move_group_->execute(plan);
                RCLCPP_INFO(this->get_logger(), "Waiting for valid detections...");

                valid_detections_.clear();
                while (rclcpp::ok() && valid_detections_.size() < 5)
                {
                    rclcpp::spin_some(shared_from_this());
                    std::this_thread::sleep_for(100ms);
                }

                if (current_pose_index_ == 0)
                {
                    T_B_C_ref = averagePose(valid_detections_);
                    RCLCPP_INFO(this->get_logger(), "Ultimate pose average: [%.3f %.3f %.3f]",
                                T_B_C_ref.translation().x(), T_B_C_ref.translation().y(), T_B_C_ref.translation().z());

                    geometry_msgs::msg::TransformStamped tf_msg;
                    tf_msg.header.stamp = this->now();
                    tf_msg.header.frame_id = "base_link";
                    tf_msg.child_frame_id = "cal_camera_link";

                    tf_msg.transform.translation.x = T_B_C_ref.translation().x();
                    tf_msg.transform.translation.y = T_B_C_ref.translation().y();
                    tf_msg.transform.translation.z = T_B_C_ref.translation().z();

                    Eigen::Quaterniond q(T_B_C_ref.rotation());
                    tf_msg.transform.rotation = tf2::toMsg(q);

                    RCLCPP_INFO(this->get_logger(), "Broadcasting static camera TF from base_link → cal_camera_link");
                    tf_broadcaster_->sendTransform(tf_msg);
                }
                else
                {
                    general_poses_.insert(general_poses_.end(), valid_detections_.begin(), valid_detections_.end());
                    RCLCPP_INFO(this->get_logger(), "Stored 5 poses for pose %lu", current_pose_index_);
                }
            }
            else
            {
                RCLCPP_WARN(this->get_logger(), "Failed to plan for pose %lu", current_pose_index_);
            }
        }

        RCLCPP_INFO(this->get_logger(), "Calibration routine complete.");

        if (!general_poses_.empty()) {
            RCLCPP_INFO(this->get_logger(), "--- Evaluating General Pose Errors ---");

            double total_translation_error = 0.0;
            double total_rotation_error_deg = 0.0;

            for (size_t i = 0; i < general_poses_.size(); ++i) {
                const Eigen::Isometry3d& T = general_poses_[i];

                Eigen::Vector3d delta_t = T_B_C_ref.translation() - T.translation();
                double translation_error = delta_t.norm();

                Eigen::Quaterniond q_ref(T_B_C_ref.rotation());
                Eigen::Quaterniond q_i(T.rotation());

                if (q_ref.dot(q_i) < 0.0)
                    q_i.coeffs() *= -1;

                Eigen::Quaterniond q_error = q_ref.inverse() * q_i;
                double angle_error_rad = 2 * std::acos(std::clamp(q_error.w(), -1.0, 1.0));
                double angle_error_deg = angle_error_rad * 180.0 / M_PI;

                RCLCPP_INFO(this->get_logger(),
                    "Pose %lu → Translation Error: %.4f m | Rotation Error: %.2f deg",
                    i, translation_error, angle_error_deg);

                total_translation_error += translation_error;
                total_rotation_error_deg += angle_error_deg;
            }

            size_t N = general_poses_.size();
            double avg_translation_error = total_translation_error / N;
            double avg_rotation_error = total_rotation_error_deg / N;

            RCLCPP_INFO(this->get_logger(),
                "--- AVERAGE ERROR OVER %lu GENERAL DETECTIONS ---", N);
            RCLCPP_INFO(this->get_logger(),
                "Average Translation Error: %.4f m | Average Rotation Error: %.2f deg",
                avg_translation_error, avg_rotation_error);
        }
    }


    Eigen::Isometry3d T_B_C_ref; // from the average of the ultimate pose


    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
    {
        if (valid_detections_.size() >= 5)
            return;

        cv_bridge::CvImagePtr cv_ptr;
        try {
            cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        } catch (...) {
            return;
        }

        cv::Mat gray;
        cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        cv::aruco::detectMarkers(gray, aruco_dict_, corners, ids, aruco_params_);

        if (!ids.empty())
        {
            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(corners, marker_length_, camera_matrix_, dist_coeffs_, rvecs, tvecs);

            cv::Mat R_mat;
            cv::Rodrigues(rvecs[0], R_mat);
            Eigen::Matrix3d R_eigen;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    R_eigen(i, j) = R_mat.at<double>(i, j);
            Eigen::Vector3d t(tvecs[0][0], tvecs[0][1], tvecs[0][2]);

            Eigen::Isometry3d T_C_M = Eigen::Isometry3d::Identity();
            T_C_M.linear() = R_eigen;
            T_C_M.translation() = t;

            Eigen::Isometry3d T_M_C = T_C_M.inverse();

            try {
                geometry_msgs::msg::TransformStamped tf = tf_buffer_.lookupTransform("base_link", "rg2_gripper_aruco_link", tf2::TimePointZero);
                Eigen::Isometry3d T_B_M = tf2::transformToEigen(tf);
                Eigen::Isometry3d T_B_C = T_B_M * T_M_C;

                valid_detections_.push_back(T_B_C);
                RCLCPP_INFO(this->get_logger(), "Accepted pose %ld/5", valid_detections_.size());
            } catch (...) {
                RCLCPP_WARN(this->get_logger(), "TF lookup failed");
            }
        }
    }

    Eigen::Isometry3d averagePose(const std::vector<Eigen::Isometry3d>& poses)
    {
        Eigen::Vector3d avg_t = Eigen::Vector3d::Zero();
        std::vector<Eigen::Quaterniond> quats;

        for (const auto& pose : poses) {
            avg_t += pose.translation();
            quats.emplace_back(pose.linear());
        }
        avg_t /= poses.size();

        Eigen::Quaterniond q_avg = quats[0];
        for (size_t i = 1; i < quats.size(); ++i) {
            if (q_avg.dot(quats[i]) < 0.0)
                quats[i].coeffs() *= -1;
            q_avg = q_avg.slerp(1.0 / (i + 1.0), quats[i]);
        }
        q_avg.normalize();

        Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
        result.translation() = avg_t;
        result.linear() = q_avg.toRotationMatrix();
        return result;
    }

    // ROS + image
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    // ArUco
    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    double marker_length_;

    // State
    size_t current_pose_index_ = 0;
    std::vector<std::vector<double>> calibration_poses_;
    std::vector<Eigen::Isometry3d> valid_detections_;
    std::vector<Eigen::Isometry3d> general_poses_;

    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraCalibrationMover>();
    node->initialize();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}














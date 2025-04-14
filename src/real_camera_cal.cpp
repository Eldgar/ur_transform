#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
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
#include <limits>

using namespace std::chrono_literals;

class CameraCalibrationMover : public rclcpp::Node
{
public:
    CameraCalibrationMover(const rclcpp::NodeOptions & options)
        : Node("camera_calibration_mover", options), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
    {
        image_sub_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            "/D415/color/image_raw/compressed",
            rclcpp::SensorDataQoS(),
            std::bind(&CameraCalibrationMover::imageCallback, this, std::placeholders::_1));

        aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
        aruco_params_ = cv::makePtr<cv::aruco::DetectorParameters>();

        aruco_params_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;
        aruco_params_->cornerRefinementWinSize = 3;
        aruco_params_->cornerRefinementMaxIterations = 20;
        aruco_params_->cornerRefinementMinAccuracy = 0.1;


        camera_matrix_ = (cv::Mat_<double>(3, 3) <<
                          306.805847, 0.0,         214.441849,
                          0.0,         306.642456, 124.910301,
                          0.0,         0.0,         1.0);

        dist_coeffs_ = (cv::Mat_<double>(5, 1) <<
                        0.0, 0.0, 0.0, 0.0, 0.0);

        marker_length_ = 0.045;

        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    }

    void initialize()
    {
        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
            shared_from_this(), "ur_manipulator");
        optimizeK1();
    }

private:
    bool moveToJointPose(const std::vector<double>& joint_pose)
    {
        move_group_->setJointValueTarget(joint_pose);
        moveit::planning_interface::MoveGroupInterface::Plan plan;
        if (move_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            move_group_->execute(plan);
            std::this_thread::sleep_for(1s);
            return true;
        } else {
            RCLCPP_WARN(this->get_logger(), "Failed to plan to joint pose");
            return false;
        }
    }

    Eigen::Isometry3d computeCameraPoseFromImage(const sensor_msgs::msg::CompressedImage::ConstSharedPtr &msg)
    {
        cv::Mat image = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
        if (image.empty()) return Eigen::Isometry3d::Identity();

        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        cv::aruco::detectMarkers(gray, aruco_dict_, corners, ids, aruco_params_);

        if (ids.empty()) return Eigen::Isometry3d::Identity();

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

        try {
            auto tf = tf_buffer_.lookupTransform("base_link", "aruco_link_rotated", tf2::TimePointZero);
            Eigen::Isometry3d T_B_M = tf2::transformToEigen(tf);
            return T_B_M * T_C_M.inverse();
        } catch (...) {
            return Eigen::Isometry3d::Identity();
        }
    }

    double computePoseError(const Eigen::Isometry3d &ref, const Eigen::Isometry3d &measured, double orientation_weight = 0.5)
    {
        Eigen::Vector3d delta_t = ref.translation() - measured.translation();
        double translation_error = delta_t.norm();

        Eigen::Quaterniond q_ref(ref.rotation());
        Eigen::Quaterniond q_measured(measured.rotation());

        if (q_ref.dot(q_measured) < 0.0)
            q_measured.coeffs() *= -1;

        Eigen::Quaterniond q_error = q_ref.inverse() * q_measured;
        double angle_error = 2 * std::acos(std::clamp(q_error.w(), -1.0, 1.0));

        return translation_error + orientation_weight * angle_error;
    }

    Eigen::Isometry3d averagePose(const std::vector<Eigen::Isometry3d> &poses)
    {
        Eigen::Vector3d avg_t = Eigen::Vector3d::Zero();
        std::vector<Eigen::Quaterniond> quats;

        for (const auto &pose : poses) {
            avg_t += pose.translation();
            quats.emplace_back(pose.rotation());
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

    void collectDetections(size_t count, std::vector<Eigen::Isometry3d> &storage)
    {
        storage.clear();
        RCLCPP_INFO(this->get_logger(), "🟡 Collecting %zu camera detections...", count);

        while (rclcpp::ok() && storage.size() < count)
        {
            rclcpp::spin_some(shared_from_this());
            std::this_thread::sleep_for(100ms);

            if (!last_image_) continue;

            // Decode image
            cv::Mat image = cv::imdecode(cv::Mat(last_image_->data), cv::IMREAD_COLOR);
            if (image.empty()) {
                RCLCPP_WARN(this->get_logger(), "❌ Failed to decode image");
                continue;
            }

            // Convert to grayscale
            cv::Mat gray;
            cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

            // Detect markers
            std::vector<int> ids;
            std::vector<std::vector<cv::Point2f>> corners;
            cv::aruco::detectMarkers(gray, aruco_dict_, corners, ids, aruco_params_);

            // Draw markers if any
            if (!ids.empty()) {
                cv::aruco::drawDetectedMarkers(image, corners, ids);
            }

            // Show the image in a window
            cv::imshow("Calibration Marker View", image);
            cv::waitKey(1);  // Allow OpenCV to update window

            // Now compute pose (same as before)
            Eigen::Isometry3d pose = computeCameraPoseFromImage(last_image_);
            if (!pose.isApprox(Eigen::Isometry3d::Identity())) {
                storage.push_back(pose);
                RCLCPP_INFO(this->get_logger(), "✅ Pose %zu/%zu collected", storage.size(), count);
            } else {
                RCLCPP_WARN(this->get_logger(), "❌ Pose detection failed");
            }
        }

        RCLCPP_INFO(this->get_logger(), "✅ Finished collecting %zu valid detections", storage.size());
    }



    void collectRawImages(size_t count, std::vector<sensor_msgs::msg::CompressedImage::ConstSharedPtr> &storage)
    {
        storage.clear();
        while (rclcpp::ok() && storage.size() < count)
        {
            rclcpp::spin_some(shared_from_this());
            std::this_thread::sleep_for(100ms);
            if (!last_image_) continue;
            storage.push_back(last_image_);
        }
    }

    double evaluateK1(double k1)
    {
        dist_coeffs_.at<double>(0, 0) = k1;
        std::vector<Eigen::Isometry3d> all_poses;

        for (const auto &images_at_pose : all_saved_images_)
        {
            for (const auto &img : images_at_pose)
            {
                Eigen::Isometry3d pose = computeCameraPoseFromImage(img);
                if (!pose.isApprox(Eigen::Isometry3d::Identity()))
                    all_poses.push_back(pose);
            }
        }

        double total_error = 0.0;
        for (const auto &pose : all_poses)
            total_error += computePoseError(T_B_C_ref_, pose);
        return total_error / all_poses.size();
    }


    void optimizeK1()
    {
        calibration_poses_ = {
            {3.03425, -0.79504, 1.25535, -1.63884, 5.93630, -2.04130}, // ULTIMATE
            {2.96115, -0.51863, 0.57200, -1.06157, 5.90268, -2.30544},
            {2.97856, -1.19795, 2.07746, -1.92523, 5.91164, -2.09038},
            {2.76846, -1.30189, 1.94836, -1.34708, 5.77416, -2.47012},
            {2.89589, -0.52993, 1.15244, -1.58780, 5.88960, -2.19087},
            {2.78293, -0.99131, 2.07900, -1.86411, 5.81653, -2.43883}
        };

        general_joint_poses_.assign(calibration_poses_.begin() + 1, calibration_poses_.end());

        // Reference collection (T_B_C_ref_)
        if (moveToJointPose(calibration_poses_[0])) {
            RCLCPP_INFO(this->get_logger(), "📸 At pose, collecting raw images...");
            collectDetections(5, valid_detections_);
            T_B_C_ref_ = averagePose(valid_detections_);
        }

        // Collect and save all images at each pose once
        all_saved_images_.clear();
        for (const auto &pose : general_joint_poses_) {
            if (!moveToJointPose(pose)) continue;

            std::vector<sensor_msgs::msg::CompressedImage::ConstSharedPtr> images;
            collectRawImages(5, images);
            all_saved_images_.push_back(images);
            std::this_thread::sleep_for(1s);
        }

        // Test k1 values
        double best_error = std::numeric_limits<double>::max();
        double best_k1 = 0.0;

        for (double k1 = 1.5; std::abs(k1) >= 1e-3; k1 *= -0.5) {
            double error = evaluateK1(k1);
            RCLCPP_INFO(this->get_logger(), "k1 = %.4f → avg error = %.6f", k1, error);
            if (error < best_error) {
                best_error = error;
                best_k1 = k1;
            }
        }

        RCLCPP_INFO(this->get_logger(), "✅ Best k1 = %.4f with error = %.6f", best_k1, best_error);
    }


    void imageCallback(const sensor_msgs::msg::CompressedImage::ConstSharedPtr msg)
    {
        last_image_ = msg;
    }

    // Core components
    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr image_sub_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    // ArUco and camera
    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    double marker_length_;

    // Image buffer
    sensor_msgs::msg::CompressedImage::ConstSharedPtr last_image_;
    std::vector<std::vector<sensor_msgs::msg::CompressedImage::ConstSharedPtr>> all_saved_images_;


    // Calibration state
    std::vector<std::vector<double>> calibration_poses_;
    std::vector<std::vector<double>> general_joint_poses_;
    std::vector<Eigen::Isometry3d> valid_detections_;
    Eigen::Isometry3d T_B_C_ref_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraCalibrationMover>(rclcpp::NodeOptions{});
    node->initialize();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

















#include "camera_calibration_mover.hpp"

using namespace std::chrono_literals;

CameraCalibrationMover::CameraCalibrationMover(const rclcpp::NodeOptions &options)
    : Node("camera_calibration_mover", options), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
{
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/wrist_rgbd_depth_sensor/image_raw", rclcpp::SensorDataQoS(),
        std::bind(&CameraCalibrationMover::imageCallback, this, std::placeholders::_1));

    aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    aruco_params_ = cv::makePtr<cv::aruco::DetectorParameters>();
    aruco_params_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;

    camera_matrix_ = (cv::Mat_<double>(3, 3) << 520.78138, 0.0, 320.5,
                                                0.0, 520.78138, 240.5,
                                                0.0, 0.0, 1.0);

    dist_coeffs_ = (cv::Mat_<double>(5, 1) << 0.0, 0.0, 0.0, 0.0, 0.0);
    marker_length_ = 0.045;

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
}

void CameraCalibrationMover::initialize()
{
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), "ur_manipulator");
    optimizeK1();
}

void CameraCalibrationMover::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
    last_image_ = msg;
}

bool CameraCalibrationMover::moveToJointPose(const std::vector<double>& joint_pose)
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

Eigen::Isometry3d CameraCalibrationMover::computeCameraPoseFromImage(const sensor_msgs::msg::Image::ConstSharedPtr &msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "cv_bridge conversion failed");
        return Eigen::Isometry3d::Identity();
    }

    cv::Mat gray;
    cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);

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
        auto tf = tf_buffer_.lookupTransform("base_link", "rg2_gripper_aruco_link", tf2::TimePointZero);
        Eigen::Isometry3d T_B_M = tf2::transformToEigen(tf);
        return T_B_M * T_C_M.inverse();
    } catch (...) {
        return Eigen::Isometry3d::Identity();
    }
}

double CameraCalibrationMover::computePoseError(const Eigen::Isometry3d &ref, const Eigen::Isometry3d &measured, double orientation_weight)
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

Eigen::Isometry3d CameraCalibrationMover::averagePose(const std::vector<Eigen::Isometry3d> &poses)
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

void CameraCalibrationMover::collectRawImages(size_t count, std::vector<sensor_msgs::msg::Image::ConstSharedPtr> &storage)
{
    storage.clear();
    RCLCPP_INFO(this->get_logger(), "📸 Collecting %zu images...", count);
    size_t last_count = 0;

    while (rclcpp::ok() && storage.size() < count)
    {
        rclcpp::spin_some(shared_from_this());
        std::this_thread::sleep_for(100ms);

        if (last_image_)
        {
            storage.push_back(last_image_);
            if (storage.size() != last_count)
            {
                RCLCPP_INFO(this->get_logger(), "✅ Image %zu/%zu collected", storage.size(), count);
                last_count = storage.size();
            }
        }
    }

    RCLCPP_INFO(this->get_logger(), "✅ Finished collecting %zu images", storage.size());
}


double CameraCalibrationMover::evaluateK1(double k1)
{
    dist_coeffs_.at<double>(0, 0) = k1;
    std::vector<Eigen::Isometry3d> all_poses;

    int image_index = 0;

    for (const auto &images_at_pose : all_saved_images_)
    {
        for (const auto &img : images_at_pose)
        {
            cv_bridge::CvImagePtr cv_ptr;
            try {
                cv_ptr = cv_bridge::toCvCopy(img, sensor_msgs::image_encodings::BGR8);
            } catch (...) {
                RCLCPP_WARN(this->get_logger(), "cv_bridge failed for image %d", image_index);
                continue;
            }

            cv::Mat gray;
            cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);

            std::vector<int> ids;
            std::vector<std::vector<cv::Point2f>> corners;
            cv::aruco::detectMarkers(gray, aruco_dict_, corners, ids, aruco_params_);

            if (!ids.empty())
                cv::aruco::drawDetectedMarkers(cv_ptr->image, corners, ids);

            // Show annotated image
            std::string window_name = "Eval Image [" + std::to_string(image_index) + "]";
            cv::imshow(window_name, cv_ptr->image);
            cv::waitKey(15);

            Eigen::Isometry3d pose = computeCameraPoseFromImage(img);
            if (!pose.isApprox(Eigen::Isometry3d::Identity()))
                all_poses.push_back(pose);

            image_index++;
        }

        cv::destroyAllWindows();

    }

    double total_error = 0.0;
    for (const auto &pose : all_poses)
        total_error += computePoseError(T_B_C_ref_, pose);

    RCLCPP_INFO(this->get_logger(), "📊 Evaluated %zu valid poses at k1=%.4f", all_poses.size(), k1);

    return all_poses.empty() ? std::numeric_limits<double>::infinity() : total_error / all_poses.size();
}


void CameraCalibrationMover::optimizeK1()
{
    calibration_poses_ = {
        {2.81349, -0.76867, 2.42993, -2.67641, 5.68814, -0.72473}, // ULTIMATE
        {2.73414, -0.85652, 1.89450, -1.70130, 5.74765, -1.05458},
        {2.36194, -1.19495, 2.65155, -2.10590, 5.37617, -1.21476},
        {2.89310, -0.45249, 2.59039, -3.26049, 5.72645, -0.59660},
        {3.05772, -0.50430, 1.90387, -2.79052, 5.77800, -0.28618}
    };
    general_joint_poses_.assign(calibration_poses_.begin() + 1, calibration_poses_.end());

    if (moveToJointPose(calibration_poses_[0])) {
        std::vector<sensor_msgs::msg::Image::ConstSharedPtr> ref_imgs;
        collectRawImages(5, ref_imgs);

        valid_detections_.clear();
        for (const auto &img : ref_imgs) {
            Eigen::Isometry3d pose = computeCameraPoseFromImage(img);
            if (!pose.isApprox(Eigen::Isometry3d::Identity()))
                valid_detections_.push_back(pose);
        }

        T_B_C_ref_ = averagePose(valid_detections_);

        // ✅ Log position and orientation in quaternion form
        const Eigen::Vector3d &t = T_B_C_ref_.translation();
        Eigen::Quaterniond q(T_B_C_ref_.rotation());

        RCLCPP_INFO(this->get_logger(),
            "📌 Reference camera pose (T_B_C_ref_):\n"
            "  Position    [x=%.4f, y=%.4f, z=%.4f] (m)\n"
            "  Orientation [x=%.4f, y=%.4f, z=%.4f, w=%.4f]",
            t.x(), t.y(), t.z(),
            q.x(), q.y(), q.z(), q.w());
    }


    all_saved_images_.clear();
    for (const auto &pose : general_joint_poses_) {
        if (!moveToJointPose(pose)) continue;
        std::vector<sensor_msgs::msg::Image::ConstSharedPtr> images;
        collectRawImages(5, images);
        all_saved_images_.push_back(images);
        std::this_thread::sleep_for(1s);
    }

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

// ---- MAIN ----
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraCalibrationMover>();
    node->initialize();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
















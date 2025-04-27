#include "camera_calibration_mover.hpp"

using namespace std::chrono_literals;

CameraCalibrationMover::CameraCalibrationMover(const rclcpp::NodeOptions &options)
    : Node("camera_calibration_mover", options), tf_buffer_(this->get_clock()), tf_listener_(tf_buffer_)
{
    image_sub_ = this->create_subscription<ImageMsg>(
        "/D415/color/image_raw/compressed", rclcpp::SensorDataQoS(),
        std::bind(&CameraCalibrationMover::imageCallback, this, std::placeholders::_1));

    aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    aruco_params_ = cv::makePtr<cv::aruco::DetectorParameters>();
    aruco_params_->cornerRefinementMethod = cv::aruco::CORNER_REFINE_SUBPIX;

    camera_matrix_ = (cv::Mat_<double>(3, 3) <<
        306.805847, 0.0,         214.441849,
        0.0,        306.642456,  124.910301,
        0.0,        0.0,         1.0);

    dist_coeffs_ = (cv::Mat_<double>(5, 1) << 0, 0, 0, 0, 0);
    marker_length_ = 0.045;

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
}

void CameraCalibrationMover::initialize()
{
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), "ur_manipulator");
    optimizeK1();
}

void CameraCalibrationMover::imageCallback(const ImageMsg::ConstSharedPtr msg)
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

Eigen::Isometry3d CameraCalibrationMover::computeCameraPoseFromImage(const ImageMsg::ConstSharedPtr &msg)
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

void CameraCalibrationMover::collectValidDetections(size_t count, std::vector<ImageMsg::ConstSharedPtr> &storage)
{
    storage.clear();
    RCLCPP_INFO(this->get_logger(), "🔍 Collecting %zu valid marker detections...", count);
    size_t collected = 0;

    while (rclcpp::ok() && collected < count)
    {
        rclcpp::spin_some(shared_from_this());
        std::this_thread::sleep_for(100ms);

        if (!last_image_) continue;

        cv::Mat image = cv::imdecode(cv::Mat(last_image_->data), cv::IMREAD_COLOR);
        if (image.empty()) continue;

        cv::Mat gray;
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        cv::aruco::detectMarkers(gray, aruco_dict_, corners, ids, aruco_params_);

        if (ids.empty()) {
            RCLCPP_DEBUG(this->get_logger(), "No marker found. Retrying...");
            continue;
        }

        cv::aruco::drawDetectedMarkers(image, corners, ids);
        cv::imshow("ArUco Detection", image);
        cv::waitKey(1);

        storage.push_back(last_image_);
        ++collected;
        RCLCPP_INFO(this->get_logger(), "✅ Detection %zu/%zu collected", collected, count);
    }

    RCLCPP_INFO(this->get_logger(), "✅ Finished collecting %zu valid detections", storage.size());
}


double CameraCalibrationMover::evaluateK1(double k1)
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

    RCLCPP_INFO(this->get_logger(), "📊 Evaluated %zu valid poses at k1=%.4f", all_poses.size(), k1);
    return all_poses.empty() ? std::numeric_limits<double>::infinity() : total_error / all_poses.size();
}

void CameraCalibrationMover::optimizeK1()
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

    if (moveToJointPose(calibration_poses_[0])) {
        std::vector<ImageMsg::ConstSharedPtr> ref_imgs;
        collectValidDetections(5, ref_imgs);

        valid_detections_.clear();
        for (const auto &img : ref_imgs) {
            Eigen::Isometry3d pose = computeCameraPoseFromImage(img);
            if (!pose.isApprox(Eigen::Isometry3d::Identity()))
                valid_detections_.push_back(pose);
        }

        T_B_C_ref_ = averagePose(valid_detections_);

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
        std::vector<ImageMsg::ConstSharedPtr> valid_images;
        collectValidDetections(5, valid_images);
        if (!valid_images.empty())
            all_saved_images_.push_back(valid_images);
        else
            RCLCPP_WARN(this->get_logger(), "❌ Skipping pose due to zero valid detections");

                std::this_thread::sleep_for(1s);
            }

    double left = -1.5;
    double right = 1.5;
    double tol = 0.01;
    int max_iter = 20;

    double best_k1 = 0.0;
    double best_error = std::numeric_limits<double>::infinity();

    for (int i = 0; i < max_iter; ++i)
    {
        double mid1 = left + (right - left) / 3.0;
        double mid2 = right - (right - left) / 3.0;

        double error1 = evaluateK1(mid1);
        double error2 = evaluateK1(mid2);

        RCLCPP_INFO(this->get_logger(), "[Iter %d] mid1 = %.4f → %.6f | mid2 = %.4f → %.6f",
                    i, mid1, error1, mid2, error2);

        if (error1 < error2)
            right = mid2;
        else
            left = mid1;

        if (std::abs(right - left) < tol)
            break;
    }

    best_k1 = (left + right) / 2.0;
    best_error = evaluateK1(best_k1);

    RCLCPP_INFO(this->get_logger(), "✅ Optimized k1 = %.4f with error = %.6f", best_k1, best_error);
}

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraCalibrationMover>();
    node->initialize();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}


















#include "camera_calibration_mover.hpp"

using namespace std::chrono_literals;

CameraTFSampler::CameraTFSampler(const rclcpp::NodeOptions &options)
    : Node("camera_tf_sampler", options),
      tf_buffer_(this->get_clock()),
      tf_listener_(tf_buffer_)
{
    image_sub_ = this->create_subscription<sensor_msgs::msg::Image>(
        "/wrist_rgbd_depth_sensor/image_raw", rclcpp::SensorDataQoS(),
        std::bind(&CameraTFSampler::imageCallback, this, std::placeholders::_1));

    aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
    aruco_params_ = cv::makePtr<cv::aruco::DetectorParameters>();

    camera_matrix_ = (cv::Mat_<double>(3, 3) << 520.78138, 0.0, 320.5,
                                                0.0, 520.78138, 240.5,
                                                0.0, 0.0, 1.0);
    dist_coeffs_ = cv::Mat::zeros(5, 1, CV_64F);  // No distortion for sim
    marker_length_ = 0.045;

    joint_poses_ = {
        {2.81349, -0.76867, 2.42993, -2.67641, 5.68814, -0.72473},
        {2.73414, -0.85652, 1.89450, -1.70130, 5.74765, -1.05458},
        {2.36194, -1.19495, 2.65155, -2.10590, 5.37617, -1.21476},
        {2.89310, -0.45249, 2.59039, -3.26049, 5.72645, -0.59660},
        {3.05772, -0.50430, 1.90387, -2.79052, 5.77800, -0.28618}
    };
}

void CameraTFSampler::initialize()
{
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), "ur_manipulator");

    tf_samples_.clear();
    for (const auto& pose : joint_poses_)
    {
        if (!moveToJointPose(pose)) continue;

        std::vector<Eigen::Isometry3d> pose_samples;
        collectValidDetections(5, pose_samples);  // Get 5 good ones at this pose

        if (!pose_samples.empty())
        {
            Eigen::Isometry3d avg = averagePose(pose_samples);
            tf_samples_.push_back(avg);
        }
    }

    analyzeDeviation();
}

void CameraTFSampler::imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr msg)
{
    last_image_ = msg;
}

bool CameraTFSampler::moveToJointPose(const std::vector<double>& joint_pose)
{
    move_group_->setJointValueTarget(joint_pose);
    moveit::planning_interface::MoveGroupInterface::Plan plan;
    if (move_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS)
    {
        move_group_->execute(plan);
        std::this_thread::sleep_for(1s);
        return true;
    }
    return false;
}

void CameraTFSampler::collectValidDetections(size_t count, std::vector<Eigen::Isometry3d>& storage)
{
    storage.clear();
    size_t collected = 0;

    while (rclcpp::ok() && collected < count)
    {
        rclcpp::spin_some(shared_from_this());
        std::this_thread::sleep_for(100ms);

        if (!last_image_) continue;

        Eigen::Isometry3d tf = estimateCameraTFfromImage(last_image_);
        if (!tf.isApprox(Eigen::Isometry3d::Identity()))
        {
            storage.push_back(tf);
            collected++;
        }
    }
}

Eigen::Isometry3d CameraTFSampler::estimateCameraTFfromImage(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr;
    try {
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    } catch (...) {
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

    // Marker-to-camera
    cv::Mat R;
    cv::Rodrigues(rvecs[0], R);
    Eigen::Matrix3d R_eigen;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            R_eigen(i, j) = R.at<double>(i, j);
    Eigen::Vector3d t(tvecs[0][0], tvecs[0][1], tvecs[0][2]);

    Eigen::Isometry3d T_C_M = Eigen::Isometry3d::Identity();
    T_C_M.linear() = R_eigen;
    T_C_M.translation() = t;

    // base_link to marker
    try {
        auto tf = tf_buffer_.lookupTransform("base_link", "rg2_gripper_aruco_link", tf2::TimePointZero);
        Eigen::Isometry3d T_B_M = tf2::transformToEigen(tf);
        return T_B_M * T_C_M.inverse();  // base <- camera
    } catch (...) {
        return Eigen::Isometry3d::Identity();
    }
}

Eigen::Isometry3d CameraTFSampler::averagePose(const std::vector<Eigen::Isometry3d>& poses)
{
    Eigen::Vector3d avg_t = Eigen::Vector3d::Zero();
    std::vector<Eigen::Quaterniond> quats;

    for (const auto& pose : poses)
    {
        avg_t += pose.translation();
        quats.emplace_back(pose.rotation());
    }

    avg_t /= poses.size();
    Eigen::Quaterniond q_avg = quats[0];
    for (size_t i = 1; i < quats.size(); ++i)
    {
        if (q_avg.dot(quats[i]) < 0.0)
            quats[i].coeffs() *= -1;
        q_avg = q_avg.slerp(1.0 / (i + 1.0), quats[i]);
    }

    Eigen::Isometry3d result = Eigen::Isometry3d::Identity();
    result.translation() = avg_t;
    result.linear() = q_avg.toRotationMatrix();
    return result;
}

void CameraTFSampler::analyzeDeviation()
{
    if (tf_samples_.empty()) return;

    const auto& ref = tf_samples_.front();
    double max_translation_error = 0.0;
    double max_orientation_error = 0.0;

    for (size_t i = 1; i < tf_samples_.size(); ++i)
    {
        Eigen::Vector3d delta_t = tf_samples_[i].translation() - ref.translation();
        double t_err = delta_t.norm();

        Eigen::Quaterniond q_ref(ref.rotation());
        Eigen::Quaterniond q_i(tf_samples_[i].rotation());
        if (q_ref.dot(q_i) < 0.0) q_i.coeffs() *= -1;

        Eigen::Quaterniond q_diff = q_ref.inverse() * q_i;
        double angle_err = 2 * std::acos(std::clamp(q_diff.w(), -1.0, 1.0));

        max_translation_error = std::max(max_translation_error, t_err);
        max_orientation_error = std::max(max_orientation_error, angle_err);
    }

    RCLCPP_INFO(this->get_logger(),
        "📈 Max deviation from first pose:\n"
        "    Translation: %.5f m\n"
        "    Orientation: %.5f rad",
        max_translation_error, max_orientation_error);
}

// MAIN
int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraTFSampler>();
    node->initialize();
    rclcpp::shutdown();
    return 0;
}
















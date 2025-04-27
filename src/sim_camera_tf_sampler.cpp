#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <image_transport/image_transport.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/aruco.hpp>
#include <opencv2/opencv.hpp>

#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.hpp>

#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>

#include <vector>
#include <chrono>

using namespace std::chrono_literals;
using ImageMsg = sensor_msgs::msg::Image;

class CameraTFSampler : public rclcpp::Node
{
public:
    explicit CameraTFSampler(const rclcpp::NodeOptions& options = rclcpp::NodeOptions())
        : Node("camera_tf_sampler", options),
          tf_buffer_(this->get_clock()),
          tf_listener_(tf_buffer_)
    {
        // Joint poses
        joint_poses_ = {
            {2.81349, -0.76867, 2.42993, -2.67641, 5.68814, -0.72473},
            {2.73414, -0.85652, 1.89450, -1.70130, 5.74765, -1.05458},
            {2.36194, -1.19495, 2.65155, -2.10590, 5.37617, -1.21476},
            {2.89310, -0.45249, 2.59039, -3.26049, 5.72645, -0.59660},
            {3.05772, -0.50430, 1.90387, -2.79052, 5.77800, -0.28618},

            {2.91038, -0.54873, 1.93223, -2.53217, 5.73402, -0.56643},
            {2.38786, -0.69632, 2.26797, -2.23364, 5.39698, -1.19453},
            {2.22103, -0.66438, 2.47734, -2.40532, 5.26166, -1.31442}
        };


        // ArUco + camera config
        aruco_dict_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_50);
        aruco_params_ = cv::aruco::DetectorParameters::create();
        marker_length_ = 0.035;  // meters

        camera_matrix_ = (cv::Mat_<double>(3, 3) <<
            525.0, 0.0, 319.5,
            0.0, 525.0, 239.5,
            0.0, 0.0, 1.0);
        dist_coeffs_ = cv::Mat::zeros(1, 5, CV_64F);

        image_sub_ = image_transport::create_subscription(
            this,
            "/wrist_rgbd_depth_sensor/image_raw",
            std::bind(&CameraTFSampler::imageCallback, this, std::placeholders::_1),
            "raw"
        );

        ellipsoid_pub_ = this->create_publisher<visualization_msgs::msg::Marker>(
            "camera_cov_ellipsoid", 1);

    }

    void initialize()
    {
        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
            shared_from_this(), "ur_manipulator");
        collectTFs();
        analyzeDeviation();
    }

private:
    void imageCallback(const ImageMsg::ConstSharedPtr msg)
    {
        last_image_ = msg;
    }

    bool moveToJointPose(const std::vector<double>& joint_pose)
    {
        rclcpp::sleep_for(300ms);
        move_group_->setJointValueTarget(joint_pose);
        moveit::planning_interface::MoveGroupInterface::Plan plan;

        if (move_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            move_group_->execute(plan);
            std::this_thread::sleep_for(1s);
            return true;
        }
        rclcpp::sleep_for(400ms);

        RCLCPP_WARN(this->get_logger(), "Failed to plan to joint pose");
        return false;
    }

    Eigen::Isometry3d computeCameraTFfromAruco()
    {
        while (rclcpp::ok()) {
            if (!last_image_) {
                rclcpp::spin_some(this->get_node_base_interface());
                continue;
            }

            cv_bridge::CvImagePtr cv_ptr;
            try {
                cv_ptr = cv_bridge::toCvCopy(last_image_, sensor_msgs::image_encodings::BGR8);
            } catch (const cv_bridge::Exception& e) {
                RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                continue;
            }

            cv::Mat gray;
            cv::cvtColor(cv_ptr->image, gray, cv::COLOR_BGR2GRAY);

            std::vector<int> ids;
            std::vector<std::vector<cv::Point2f>> corners;
            cv::aruco::detectMarkers(gray, aruco_dict_, corners, ids, aruco_params_);

            if (ids.empty()) {
                RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 2000, "No marker detected yet...");
                continue;
            }

            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(corners, marker_length_, camera_matrix_, dist_coeffs_, rvecs, tvecs);

            cv::Mat R_mat;
            cv::Rodrigues(rvecs[0], R_mat);

            Eigen::Matrix3d R;
            for (int i = 0; i < 3; ++i)
                for (int j = 0; j < 3; ++j)
                    R(i, j) = R_mat.at<double>(i, j);

            Eigen::Vector3d t(tvecs[0][0], tvecs[0][1], tvecs[0][2]);

            Eigen::Isometry3d T_C_M = Eigen::Isometry3d::Identity();
            T_C_M.linear() = R;
            T_C_M.translation() = t;

            Eigen::Isometry3d T_M_C = T_C_M.inverse();

            try {
                auto tf = tf_buffer_.lookupTransform("base_link", "rg2_gripper_aruco_link", tf2::TimePointZero);
                Eigen::Isometry3d T_B_M = tf2::transformToEigen(tf);
                Eigen::Isometry3d T_B_C = T_B_M * T_M_C;

                RCLCPP_INFO(this->get_logger(),
                    "✅ Detected ArUco marker at position (%.3f, %.3f, %.3f)",
                    t[0], t[1], t[2]);

                return T_B_C;
            } catch (const tf2::TransformException& ex) {
                RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s", ex.what());
                continue;
            }
        }

        return Eigen::Isometry3d::Identity();
    }

        Eigen::Isometry3d computeTF(const std::string& target_frame = "camera_link")
    {
        try {
            auto tf = tf_buffer_.lookupTransform("base_link", target_frame, tf2::TimePointZero);
            return tf2::transformToEigen(tf);
        } catch (const tf2::TransformException& ex) {
            RCLCPP_WARN(this->get_logger(), "TF lookup failed for %s: %s", target_frame.c_str(), ex.what());
            return Eigen::Isometry3d::Identity();
        }
    }

    void collectTFs()
    {
        tf_samples_.clear();
        const size_t NUM_DETECTIONS_PER_POSE = 5;
        const rclcpp::Duration TIMEOUT = rclcpp::Duration::from_seconds(10.0);

        for (size_t idx = 0; idx < joint_poses_.size(); ++idx) {
            const auto& pose = joint_poses_[idx];
            RCLCPP_INFO(this->get_logger(), "➡️ Moving to joint pose %zu/%zu...", idx + 1, joint_poses_.size());

            if (!moveToJointPose(pose)) {
                RCLCPP_WARN(this->get_logger(), "❌ Skipping pose %zu due to move failure.", idx + 1);
                continue;
            }

            std::vector<Eigen::Isometry3d> detections;
            auto start_time = this->now();

            while (detections.size() < NUM_DETECTIONS_PER_POSE && (this->now() - start_time) < TIMEOUT) {
                auto tf = computeTF();
                if (!tf.isApprox(Eigen::Isometry3d::Identity())) {
                    detections.push_back(tf);
                    RCLCPP_INFO(this->get_logger(), "✅ [%zu/%zu] detection at current pose", detections.size(), NUM_DETECTIONS_PER_POSE);
                } else {
                    rclcpp::sleep_for(100ms);
                }
            }

            if (detections.size() < NUM_DETECTIONS_PER_POSE) {
                RCLCPP_WARN(this->get_logger(), "⚠️ Only collected %zu/%zu valid detections at pose %zu.",
                            detections.size(), NUM_DETECTIONS_PER_POSE, idx + 1);
            }

            if (!detections.empty()) {
                // Average the translation
                Eigen::Vector3d avg_t = Eigen::Vector3d::Zero();
                for (const auto& tf : detections)
                    avg_t += tf.translation();
                avg_t /= static_cast<double>(detections.size());

                // Use the first orientation (or do quaternion averaging later)
                Eigen::Isometry3d avg_tf = Eigen::Isometry3d::Identity();
                avg_tf.linear() = detections[0].rotation();
                avg_tf.translation() = avg_t;

                tf_samples_.push_back(avg_tf);
            }
        }

        RCLCPP_INFO(this->get_logger(), "✅ Collected %zu averaged TF samples", tf_samples_.size());
    }


    void analyzeDeviation()
    {
        if (tf_samples_.empty()) {
            RCLCPP_WARN(this->get_logger(), "No TF samples to analyze");
            return;
        }

        const auto& ref = tf_samples_.front();
        double max_translation_error = 0.0;
        double max_orientation_error = 0.0;

        // === Position mean & covariance ===
        Eigen::Vector3d mean_pos = Eigen::Vector3d::Zero();
        for (const auto& tf : tf_samples_) {
            mean_pos += tf.translation();
        }
        mean_pos /= tf_samples_.size();

        Eigen::Matrix3d pos_cov = Eigen::Matrix3d::Zero();
        for (const auto& tf : tf_samples_) {
            Eigen::Vector3d diff = tf.translation() - mean_pos;
            pos_cov += diff * diff.transpose();
        }
        pos_cov /= tf_samples_.size();

        Eigen::Vector3d pos_stddev = pos_cov.eigenvalues().cwiseSqrt().real();

        // === Orientation mean ===
        std::vector<Eigen::Quaterniond> quats;
        for (const auto& tf : tf_samples_) {
            quats.push_back(Eigen::Quaterniond(tf.rotation()));
        }

        Eigen::Quaterniond mean_q = quats[0];
        for (size_t i = 1; i < quats.size(); ++i) {
            if (mean_q.dot(quats[i]) < 0.0)
                quats[i].coeffs() *= -1;  // Ensure same hemisphere
            double factor = 1.0 / (i + 1);
            mean_q = mean_q.slerp(factor, quats[i]);
        }
        mean_q.normalize();

        // === Orientation covariance in tangent space (rotation vectors) ===
        Eigen::Matrix3d rot_cov = Eigen::Matrix3d::Zero();
        for (const auto& q : quats) {
            Eigen::Quaterniond dq = mean_q.inverse() * q;
            if (dq.w() < 0.0)
                dq.coeffs() *= -1;  // Again, enforce same hemisphere

            Eigen::AngleAxisd aa(dq);
            Eigen::Vector3d rot_vec = aa.angle() * aa.axis();  // Small-angle approx for residual
            rot_cov += rot_vec * rot_vec.transpose();
        }
        rot_cov /= tf_samples_.size();

        Eigen::Vector3d rot_stddev = rot_cov.eigenvalues().cwiseSqrt().real();

        // === Max deviation ===
        for (size_t i = 1; i < tf_samples_.size(); ++i) {
            Eigen::Vector3d d = tf_samples_[i].translation() - ref.translation();
            double t_err = d.norm();

            Eigen::Quaterniond q_ref(ref.rotation());
            Eigen::Quaterniond q_i(tf_samples_[i].rotation());

            if (q_ref.dot(q_i) < 0.0) q_i.coeffs() *= -1;
            Eigen::Quaterniond q_diff = q_ref.inverse() * q_i;
            double angle_err = 2 * std::acos(std::clamp(q_diff.w(), -1.0, 1.0));

            max_translation_error = std::max(max_translation_error, t_err);
            max_orientation_error = std::max(max_orientation_error, angle_err);
        }

        // === Log it all ===
        RCLCPP_INFO(this->get_logger(),
            "\n Mean Position:       (%.4f, %.4f, %.4f)"
            "\n Mean Orientation (quat): (x=%.4f, y=%.4f, z=%.4f, w=%.4f)"
            "\n Pos Std Dev (xyz):   (%.4f, %.4f, %.4f)"
            "\n Rot Std Dev (rpy*):  (%.4f, %.4f, %.4f)"
            "\n Max Deviation:"
            "\n   Translation:         %.4f m"
            "\n   Orientation:         %.4f rad",
            mean_pos.x(), mean_pos.y(), mean_pos.z(),
            mean_q.x(), mean_q.y(), mean_q.z(), mean_q.w(),
            pos_stddev.x(), pos_stddev.y(), pos_stddev.z(),
            rot_stddev.x(), rot_stddev.y(), rot_stddev.z(),
            max_translation_error, max_orientation_error);

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(pos_cov);
            Eigen::Vector3d eigenvalues = solver.eigenvalues().cwiseSqrt();
            Eigen::Matrix3d eigenvectors = solver.eigenvectors();

            // Convert orientation matrix to quaternion
            Eigen::Quaterniond orientation(eigenvectors);
            orientation.normalize();

    }

    image_transport::Subscriber image_sub_;
    ImageMsg::ConstSharedPtr last_image_;

    rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr ellipsoid_pub_;

    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;

    cv::Ptr<cv::aruco::Dictionary> aruco_dict_;
    cv::Ptr<cv::aruco::DetectorParameters> aruco_params_;
    cv::Mat camera_matrix_;
    cv::Mat dist_coeffs_;
    double marker_length_;

    std::vector<std::vector<double>> joint_poses_;
    std::vector<Eigen::Isometry3d> tf_samples_;
};

int main(int argc, char** argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraTFSampler>();
    node->initialize();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}


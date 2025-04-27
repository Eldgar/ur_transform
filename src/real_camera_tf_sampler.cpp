#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2_eigen/tf2_eigen.hpp>
#include <Eigen/Geometry>
#include <Eigen/Eigenvalues>
#include <moveit/move_group_interface/move_group_interface.h>

#include <vector>
#include <chrono>

using namespace std::chrono_literals;

class CameraTFSampler : public rclcpp::Node
{
public:
    explicit CameraTFSampler(const rclcpp::NodeOptions &options = rclcpp::NodeOptions())
        : Node("camera_tf_sampler", options),
          tf_buffer_(this->get_clock()),
          tf_listener_(tf_buffer_)
    {
        joint_poses_ = {
            {3.03425, -0.79504, 1.25535, -1.63884, 5.93630, -2.04130},
            {2.96115, -0.51863, 0.57200, -1.06157, 5.90268, -2.30544},
            {2.97856, -1.19795, 2.07746, -1.92523, 5.91164, -2.09038},
            {2.76846, -1.30189, 1.94836, -1.34708, 5.77416, -2.47012},
            {2.89589, -0.52993, 1.15244, -1.58780, 5.88960, -2.19087},

            {2.81969, -0.64596, 1.62408, -1.80887, 5.84188, -2.33796},
            {2.90300, -0.66354, 1.93476, -2.25093, 5.89349, -2.17547},
            {2.96924, -0.27762, 0.96071, -1.80989, 5.92634, -2.01731},
            {2.88067, -0.19469, 0.67950, -1.48274, 5.85636, -2.34294},
            {2.53973, -0.50475, 1.59799, -1.50125, 5.65903, -2.81514}
        };
    }

    void initialize()
    {
        move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
            shared_from_this(), "ur_manipulator");
        collectTFs();
        analyzeDeviation();
    }

private:
    bool moveToJointPose(const std::vector<double> &joint_pose)
    {
        rclcpp::sleep_for(400ms);
        move_group_->setJointValueTarget(joint_pose);
        moveit::planning_interface::MoveGroupInterface::Plan plan;

        if (move_group_->plan(plan) == moveit::core::MoveItErrorCode::SUCCESS) {
            move_group_->execute(plan);
            std::this_thread::sleep_for(1s);
            return true;
        }
        rclcpp::sleep_for(500ms);
        RCLCPP_WARN(this->get_logger(), "Failed to plan to joint pose");
        return false;
    }

    Eigen::Isometry3d computeTF()
    {
        try {
            auto tf = tf_buffer_.lookupTransform("base_link", "camera_color_optical_frame", tf2::TimePointZero);
            return tf2::transformToEigen(tf);
        } catch (const tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s", ex.what());
            return Eigen::Isometry3d::Identity();
        }
    }

    void collectTFs()
    {
        tf_samples_.clear();
        const size_t NUM_DETECTIONS_PER_POSE = 5;
        const rclcpp::Duration TIMEOUT = rclcpp::Duration::from_seconds(10.0);

        for (size_t idx = 0; idx < joint_poses_.size(); ++idx) {
            const auto &pose = joint_poses_[idx];
            RCLCPP_INFO(this->get_logger(), "➡️ Moving to joint pose %zu/%zu...", idx + 1, joint_poses_.size());

            if (!moveToJointPose(pose)) continue;

            std::vector<Eigen::Isometry3d> detections;
            auto start_time = this->now();

            while (detections.size() < NUM_DETECTIONS_PER_POSE && (this->now() - start_time) < TIMEOUT) {
                auto tf = computeTF();
                if (!tf.isApprox(Eigen::Isometry3d::Identity())) {
                    detections.push_back(tf);
                    rclcpp::sleep_for(600ms);
                    RCLCPP_INFO(this->get_logger(), "✅ [%zu/%zu] detection at current pose", detections.size(), NUM_DETECTIONS_PER_POSE);
                } else {
                    rclcpp::sleep_for(500ms);
                }
            }

            if (!detections.empty()) {
                Eigen::Vector3d avg_t = Eigen::Vector3d::Zero();
                for (const auto &tf : detections)
                    avg_t += tf.translation();
                avg_t /= static_cast<double>(detections.size());

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

        const auto &ref = tf_samples_.front();
        double max_translation_error = 0.0;
        double max_orientation_error = 0.0;

        Eigen::Vector3d mean_pos = Eigen::Vector3d::Zero();
        for (const auto &tf : tf_samples_)
            mean_pos += tf.translation();
        mean_pos /= tf_samples_.size();

        Eigen::Matrix3d pos_cov = Eigen::Matrix3d::Zero();
        for (const auto &tf : tf_samples_) {
            Eigen::Vector3d diff = tf.translation() - mean_pos;
            pos_cov += diff * diff.transpose();
        }
        pos_cov /= tf_samples_.size();
        Eigen::Vector3d pos_stddev = pos_cov.eigenvalues().cwiseSqrt().real();

        std::vector<Eigen::Quaterniond> quats;
        for (const auto &tf : tf_samples_)
            quats.push_back(Eigen::Quaterniond(tf.rotation()));

        Eigen::Quaterniond mean_q = quats[0];
        for (size_t i = 1; i < quats.size(); ++i) {
            if (mean_q.dot(quats[i]) < 0.0)
                quats[i].coeffs() *= -1;
            double factor = 1.0 / (i + 1);
            mean_q = mean_q.slerp(factor, quats[i]);
        }
        mean_q.normalize();

        Eigen::Matrix3d rot_cov = Eigen::Matrix3d::Zero();
        for (const auto &q : quats) {
            Eigen::Quaterniond dq = mean_q.inverse() * q;
            if (dq.w() < 0.0) dq.coeffs() *= -1;
            Eigen::AngleAxisd aa(dq);
            Eigen::Vector3d rot_vec = aa.angle() * aa.axis();
            rot_cov += rot_vec * rot_vec.transpose();
        }
        rot_cov /= tf_samples_.size();
        Eigen::Vector3d rot_stddev = rot_cov.eigenvalues().cwiseSqrt().real();

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

        RCLCPP_INFO(this->get_logger(),
                    "\n📍 Mean Position:       (%.4f, %.4f, %.4f)"
                    "\n📍 Mean Orientation (quat): (x=%.4f, y=%.4f, z=%.4f, w=%.4f)"
                    "\n📈 Pos Std Dev (xyz):   (%.4f, %.4f, %.4f)"
                    "\n📈 Rot Std Dev (rpy*):  (%.4f, %.4f, %.4f)"
                    "\n📊 Max Deviation:"
                    "\n   Translation:         %.4f m"
                    "\n   Orientation:         %.4f rad",
                    mean_pos.x(), mean_pos.y(), mean_pos.z(),
                    mean_q.x(), mean_q.y(), mean_q.z(), mean_q.w(),
                    pos_stddev.x(), pos_stddev.y(), pos_stddev.z(),
                    rot_stddev.x(), rot_stddev.y(), rot_stddev.z(),
                    max_translation_error, max_orientation_error);
    }

    std::vector<std::vector<double>> joint_poses_;
    std::vector<Eigen::Isometry3d> tf_samples_;

    std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<CameraTFSampler>();
    node->initialize();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}


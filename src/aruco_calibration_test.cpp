// aruco_calibration_test.cpp — COMPLETE self‑contained node
// Drives UR3e through 5 joint poses, waits for the ArUco‑centre pixel, its
// depth, and the base←marker TF, averages the first 3 samples, folds in the
// static D415_link→color optical transform, and broadcasts base←D415_link.
//
// Build deps (CMakeLists):
//   rclcpp rclcpp_action moveit_ros_planning_interface tf2_ros
//   tf2_geometry_msgs tf2_eigen geometry_msgs sensor_msgs cv_bridge
//   image_transport Eigen3 OpenCV
//
// --------------------------------------------------------------------------

#include <rclcpp/rclcpp.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <moveit/move_group_interface/move_group_interface.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_eigen/tf2_eigen.hpp>
#include <image_transport/image_transport.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <geometry_msgs/msg/point32.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <cv_bridge/cv_bridge.h>
#include <Eigen/Dense>
#include <opencv2/imgproc.hpp>
#include <array>
#include <vector>

using namespace std::chrono_literals;

class ArucoCalibrationTest : public rclcpp::Node {
public:
  ArucoCalibrationTest()
      : Node("aruco_calibration_test"), tf_buffer_(this->get_clock()),
        tf_listener_(tf_buffer_) {

    /* ---------------- MoveIt ---------------- */
    move_group_ = std::make_shared<moveit::planning_interface::MoveGroupInterface>(
        shared_from_this(), "ur_manipulator");
    move_group_->setPlanningTime(10.0);
    move_group_->setMaxVelocityScalingFactor(0.3);

    /* ---------------- Subscribers ---------------- */
    center_sub_ = create_subscription<geometry_msgs::msg::Point32>(
        "/aruco/center_px", 10,
        std::bind(&ArucoCalibrationTest::centerCb, this, std::placeholders::_1));

    depth_sub_ = image_transport::create_subscription(
        this, "/D415/aligned_depth_to_color/image_raw",
        std::bind(&ArucoCalibrationTest::depthCb, this, std::placeholders::_1),
        "raw");

    ur_tf_sub_ = create_subscription<geometry_msgs::msg::TransformStamped>(
        "/ur_transform", 10,
        std::bind(&ArucoCalibrationTest::urTfCb, this, std::placeholders::_1));

    tf_pub_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

    /* ---------------- Joint targets ---------------- */
    target_joints_ = {
        {2.89589, -0.52993, 1.15244, -1.58780, 5.88960, -2.19087},
        {2.81969, -0.64596, 1.62408, -1.80887, 5.84188, -2.33796},
        {2.90300, -0.66354, 1.93476, -2.25093, 5.89349, -2.17547},
        {2.96924, -0.27762, 0.96071, -1.80989, 5.92634, -2.01731},
        {2.88067, -0.19469, 0.67950, -1.48274, 5.85636, -2.34294}};

    RCLCPP_INFO(get_logger(), "Starting calibration – moving to pose 1/%zu",
                target_joints_.size());
    moveToPose();
  }

private:
  /* ------------ helper structs & enums ------------ */
  struct Sample {
    geometry_msgs::msg::Point32 px;          // pixel centre
    double depth{0.0};                       // metres
    Eigen::Isometry3d T_B_M{Eigen::Isometry3d::Identity()};
  };
  enum class Stage { MOVING, WAITING } stage_{Stage::MOVING};

  /* ------------ ROS handles ------------ */
  tf2_ros::Buffer tf_buffer_;
  tf2_ros::TransformListener tf_listener_;
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_pub_;
  std::shared_ptr<moveit::planning_interface::MoveGroupInterface> move_group_;
  rclcpp::Subscription<geometry_msgs::msg::Point32>::SharedPtr center_sub_;
  image_transport::Subscriber depth_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr ur_tf_sub_;

  /* ------------ runtime state ------------ */
  std::vector<std::array<double, 6>> target_joints_;
  size_t idx_{0};
  geometry_msgs::msg::Point32 last_px_;
  double last_depth_{0.0};
  bool got_px_{false}, got_depth_{false};
  Eigen::Isometry3d last_T_B_M_{Eigen::Isometry3d::Identity()};
  std::vector<Sample> samples_;

  /* ------------ Callbacks ------------ */
  void centerCb(const geometry_msgs::msg::Point32::SharedPtr msg) {
    if (stage_ != Stage::WAITING)
      return;
    last_px_ = *msg;
    got_px_ = true;
    RCLCPP_INFO(get_logger(), "Pixel %.1f, %.1f", msg->x, msg->y);
    maybeFinishPose();
  }

  void depthCb(const sensor_msgs::msg::Image::ConstSharedPtr &msg) {
    if (stage_ != Stage::WAITING || !got_px_)
      return;
    auto cv_ptr = cv_bridge::toCvShare(msg);
    int u = static_cast<int>(last_px_.x + 0.5);
    int v = static_cast<int>(last_px_.y + 0.5);
    if (u < 0 || v < 0 || u >= cv_ptr->image.cols || v >= cv_ptr->image.rows)
      return;
    uint16_t d_mm = cv_ptr->image.at<uint16_t>(v, u);
    if (d_mm == 0)
      return;
    last_depth_ = d_mm / 1000.0;
    got_depth_ = true;
    RCLCPP_INFO(get_logger(), "Depth %.3f m", last_depth_);
    maybeFinishPose();
  }

  void urTfCb(const geometry_msgs::msg::TransformStamped::SharedPtr msg) {
    if (msg->child_frame_id != "aruco_link_rotated")
      return;
    last_T_B_M_ = tf2::transformToEigen(*msg);
  }

  /* ------------ Movement helpers ------------ */
  void moveToPose() {
    stage_ = Stage::MOVING;
    got_px_ = got_depth_ = false;

    std::vector<double> vec(target_joints_[idx_].begin(), target_joints_[idx_].end());
    move_group_->setJointValueTarget(vec);
    moveit::planning_interface::MoveGroupInterface::Plan p;
    if (move_group_->plan(p) != moveit::core::MoveItErrorCode::SUCCESS) {
      RCLCPP_ERROR(get_logger(), "Plan failed @ pose %zu", idx_ + 1);
      rclcpp::shutdown();
      return;
    }
    move_group_->execute(p);
    RCLCPP_INFO(get_logger(), "Reached pose %zu – waiting for marker & depth", idx_ + 1);
    stage_ = Stage::WAITING;
  }

  void maybeFinishPose() {
    if (!(got_px_ && got_depth_))
      return;
    samples_.push_back({last_px_, last_depth_, last_T_B_M_});
    RCLCPP_INFO(get_logger(), "Pose %zu captured", idx_ + 1);

    if (++idx_ < target_joints_.size()) {
      rclcpp::sleep_for(500ms);
      moveToPose();
      return;
    }
    RCLCPP_INFO(get_logger(), "Collected %zu samples – computing TF", samples_.size());
    computeTf();
  }

  /* ------------ Final computation ------------ */
  void computeTf() {
    const size_t N = std::min<size_t>(3, samples_.size());
    Eigen::Vector3d mean_p = Eigen::Vector3d::Zero();
    for (size_t k = 0; k < N; ++k) {
      Eigen::Vector3d p_M(0, 0, -samples_[k].depth); // back along Z to camera
      mean_p += samples_[k].T_B_M * p_M;
    }
    mean_p /= static_cast<double>(N);
    Eigen::Quaterniond mean_q(samples_[0].T_B_M.linear());

    geometry_msgs::msg::TransformStamped tf_link_color;
    try {
      tf_link_color = tf_buffer_.lookupTransform(
          "D415_link", "D415_color_optical_frame", tf2::TimePointZero);
    } catch (const tf2::TransformException &ex) {
      RCLCPP_ERROR(get_logger(), "Static TF missing: %s", ex.what());
      return;
    }

    // Eigen representations
    Eigen::Isometry3d T_L_C = tf2::transformToEigen(tf_link_color);
    Eigen::Isometry3d T_B_C = Eigen::Isometry3d::Identity();
    T_B_C.translation() = mean_p;
    T_B_C.linear()      = mean_q.toRotationMatrix();

    // Compute base←D415_link = base←color * link←color⁻¹
    Eigen::Isometry3d T_B_L = T_B_C * T_L_C.inverse();

    /* ------------ Broadcast result ------------ */
    geometry_msgs::msg::TransformStamped tf_out = tf2::eigenToTransform(T_B_L);
    tf_out.header.stamp = now();
    tf_out.header.frame_id = "base_link";
    tf_out.child_frame_id  = "D415_link";
    tf_pub_->sendTransform(tf_out);

    RCLCPP_INFO(get_logger(),
      "Calibrated D415_link: (%.3f, %.3f, %.3f) m",
      T_B_L.translation().x(), T_B_L.translation().y(), T_B_L.translation().z());
  }
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<ArucoCalibrationTest>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}





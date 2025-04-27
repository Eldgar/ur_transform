/* cupholder_detector_real.cpp
   ---------------------------------
   Detect 4 cupholders & broadcast TFs.
   Table‑top (barista_top) height is now
   taken from the depth image region that
   lies *between* the 4 rims.              */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/buffer.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <deque>
#include <algorithm>   // std::clamp

/* ───────────────────────────────────────────────────────── */

class CupholderDetector : public rclcpp::Node
{
public:
  CupholderDetector()
  : Node("cupholder_detector")
  {
    RCLCPP_INFO(get_logger(), "CupholderDetector node started");

    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/D415/aligned_depth_to_color/image_raw", 10,
        std::bind(&CupholderDetector::depthCallback, this, std::placeholders::_1));

    color_sub_ = create_subscription<sensor_msgs::msg::CompressedImage>(
        "/D415/color/image_raw/compressed", rclcpp::SensorDataQoS(),
        std::bind(&CupholderDetector::colorCallback, this, std::placeholders::_1));

    tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(*this);
    tf_buffer_      = std::make_shared<tf2_ros::Buffer>(get_clock());
    tf_listener_    = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);
  }

private:
  /* ------------------- parameters / state ------------------ */
  float roi_u_ = -1.f, roi_v_ = -1.f, roi_radius_ = -1.f;
  cv::Mat last_depth_raw_;

  bool cupholders_initialized_ = false;
  std::vector<Eigen::Vector3f> cached_cupholder_positions_;
  static constexpr std::size_t MAX_HISTORY = 5;
  std::vector<std::deque<Eigen::Vector3f>> cupholder_history_{4};

  float barista_center_height_ = -1.f;        // table‑top height estimate

  /* ------------------- subscriptions ----------------------- */
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr         depth_sub_;
  rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr color_sub_;

  /* ------------------- TF helpers -------------------------- */
  std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
  std::shared_ptr<tf2_ros::Buffer>               tf_buffer_;
  std::shared_ptr<tf2_ros::TransformListener>    tf_listener_;

  /* ------------------- intrinsic parameters ---------------- */
  const float fx_   = 306.806f;
  const float fy_   = 306.806f;
  const float cx_d_ = 214.4f;
  const float cy_d_ = 124.9f;

  /* =========================================================
     depthCallback – detects ROI circle on depth image
     and publishes a coarse table‑top TF.
     ========================================================= */
  void depthCallback(const sensor_msgs::msg::Image::SharedPtr msg)
  {
    cv::Mat depth_image;
    try {
      depth_image =
          cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::TYPE_16UC1)->image;
    } catch (cv_bridge::Exception &e) {
      RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
      return;
    }
    if (depth_image.empty()) return;

    /* ----------- depth image preprocessing ------------- */
    cv::Mat depth_8u, normalized;
    cv::normalize(depth_image, normalized, 0, 255, cv::NORM_MINMAX);
    normalized.convertTo(depth_8u, CV_8UC1);
    cv::equalizeHist(depth_8u, depth_8u);
    cv::GaussianBlur(depth_8u, depth_8u, cv::Size(3,3), 0);

    /* ----------- find *one* big circle = holder set ----- */
    cv::Mat thresholded;
    cv::threshold(depth_8u, thresholded, 71, 250, cv::THRESH_BINARY);
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(thresholded, circles, cv::HOUGH_GRADIENT,
                     1, 50, 24, 11, 52, 74);

    /* ----------- keep first circle as ROI --------------- */
    if (!circles.empty()) {
      roi_u_      = circles[0][0];
      roi_v_      = circles[0][1];
      roi_radius_ = circles[0][2];
    }
    last_depth_raw_ = depth_image.clone();

    /* ----------- publish coarse table TF (centroid) ------ */
    Eigen::Vector3f sum(0,0,0); int cnt = 0;
    for (int v = 0; v < depth_image.rows; ++v)
      for (int u = 0; u < depth_image.cols; ++u)
      {
        const uint16_t d = depth_image.at<uint16_t>(v,u);
        if (d==0) continue;
        const float z = d * 0.001f;
        sum += Eigen::Vector3f((u-cx_d_)*z/fx_, (v-cy_d_)*z/fy_, z);
        ++cnt;
      }
    if (cnt>0) {
      Eigen::Vector3f c = sum/static_cast<float>(cnt);
      barista_center_height_ = c.z();

      geometry_msgs::msg::TransformStamped tf;
      tf.header.stamp = now();
      tf.header.frame_id = "D415_depth_optical_frame";
      tf.child_frame_id  = "barista_center";
      tf.transform.translation.x = c.x();
      tf.transform.translation.y = c.y();
      tf.transform.translation.z = c.z();
      tf.transform.rotation.w = 1.0;
      tf_broadcaster_->sendTransform(tf);
    }
  }

  /* =========================================================
     colorCallback – detects the 4 rims, refines table height,
     tracks cupholders & publishes pickup targets.
     ========================================================= */
  void colorCallback(const sensor_msgs::msg::CompressedImage::SharedPtr msg)
  {
    /* ---- prerequisites available? ---------------------- */
    if (roi_u_ < 0 || roi_v_ < 0 || roi_radius_ < 0 ||
        last_depth_raw_.empty() || barista_center_height_ <= 0.0f)
      return;

    /* ---- decode colour image --------------------------- */
    cv::Mat color_bgr = cv::imdecode(cv::Mat(msg->data), cv::IMREAD_COLOR);
    if (color_bgr.empty()) return;

    /* ---- debug overlays -------------------------------- */
    cv::Mat color_dbg = color_bgr.clone();
    cv::Mat depth_norm, depth_dbg;
    cv::normalize(last_depth_raw_, depth_norm, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::cvtColor(depth_norm, depth_dbg, cv::COLOR_GRAY2BGR);

    /* ---- ROI pre‑processing ---------------------------- */
    cv::Mat gray;                 cv::cvtColor(color_bgr, gray, cv::COLOR_BGR2GRAY);
    cv::Mat eq;                   cv::equalizeHist(gray, eq);

    const int BUF = 5;
    int x = std::max(0,  static_cast<int>(roi_u_ - roi_radius_ - BUF));
    int y = std::max(0,  static_cast<int>(roi_v_ - roi_radius_ - BUF));
    int w = std::min(color_bgr.cols - x, static_cast<int>(2*roi_radius_ + 2*BUF));
    int h = std::min(color_bgr.rows - y, static_cast<int>(2*roi_radius_ + 2*BUF));
    if (w<=0 || h<=0) return;

    cv::Rect roi_rect(x,y,w,h);
    cv::Mat cropped = eq(roi_rect).clone();
    cv::Mat final_roi;            cv::equalizeHist(cropped, final_roi);

    /* ---- detect the four small rims -------------------- */
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(final_roi, circles, cv::HOUGH_GRADIENT,
                     1, 32, 44, 25, 8, 13);

    /* ----------------------------------------------------
       STEP 1 – refine tabletop height from depth BETWEEN
       the four rims (only if we see exactly 4 of them)
       ---------------------------------------------------- */
    if (circles.size() == 4)
    {
      int min_u = last_depth_raw_.cols, min_v = last_depth_raw_.rows;
      int max_u = 0,                     max_v = 0;

      for (const auto &c : circles)
      {
        int u_full = static_cast<int>(roi_rect.x + c[0]);
        int v_full = static_cast<int>(roi_rect.y + c[1]);
        min_u = std::min(min_u, u_full);
        min_v = std::min(min_v, v_full);
        max_u = std::max(max_u, u_full);
        max_v = std::max(max_v, v_full);
      }

      constexpr int PAD = 5;
      min_u = std::clamp(min_u + PAD, 0, last_depth_raw_.cols-1);
      min_v = std::clamp(min_v + PAD, 0, last_depth_raw_.rows-1);
      max_u = std::clamp(max_u - PAD, 0, last_depth_raw_.cols-1);
      max_v = std::clamp(max_v - PAD, 0, last_depth_raw_.rows-1);

      double depth_sum = 0.0; std::size_t depth_cnt = 0;
      for (int v=min_v; v<=max_v; ++v)
        for (int u=min_u; u<=max_u; ++u)
        {
          uint16_t d = last_depth_raw_.at<uint16_t>(v,u);
          if (d==0) continue;
          depth_sum += static_cast<double>(d) * 0.001; // mm→m
          ++depth_cnt;
        }

      if (depth_cnt>0)
        barista_center_height_ =
            static_cast<float>(depth_sum / static_cast<double>(depth_cnt));
      else
        RCLCPP_WARN(get_logger(), "Could not refine barista height – keeping old value");
    }

    /* ----------------------------------------------------
       STEP 2 – convert each rim centre to 3‑D (camera frame)
       ---------------------------------------------------- */
    std::vector<Eigen::Vector3f> new_positions;

    if (circles.size() == 4)
    {
      for (const auto &c : circles)
      {
        float full_u = roi_rect.x + c[0];
        float full_v = roi_rect.y + c[1];

        // debug dots
        if (full_u>=0 && full_v>=0 && full_u<color_dbg.cols && full_v<color_dbg.rows)
          cv::circle(color_dbg, cv::Point(cvRound(full_u), cvRound(full_v)),
                     3, cv::Scalar(0,0,255), -1);
        if (full_u>=0 && full_v>=0 && full_u<depth_dbg.cols && full_v<depth_dbg.rows)
          cv::circle(depth_dbg, cv::Point(cvRound(full_u), cvRound(full_v)),
                     3, cv::Scalar(0,255,0), -1);

        if (full_u<0 || full_v<0 || full_u>=last_depth_raw_.cols ||
            full_v>=last_depth_raw_.rows)
          continue;

        uint16_t d = last_depth_raw_.at<uint16_t>(cvRound(full_v), cvRound(full_u));
        if (d==0) continue;

        const float z = d * 0.001f;
        const float x_m = (full_u - cx_d_) * z / fx_;
        const float y_m = (full_v - cy_d_) * z / fy_;
        const float z_m = barista_center_height_;   // ← refined height

        new_positions.emplace_back(x_m, y_m, z_m);
      }
    }
    else
    {
      RCLCPP_WARN(get_logger(), "Expected 4 cupholders, but found %zu. Using cached.",
                  circles.size());
    }

    /* ----------------------------------------------------
       STEP 3 – caching / history (exactly your original code)
       ---------------------------------------------------- */
    if (new_positions.size() == 4)
    {
      if (!cupholders_initialized_) {
        cached_cupholder_positions_ = new_positions;
        cupholders_initialized_    = true;
      } else {
        std::vector<bool> matched(4,false);
        std::vector<Eigen::Vector3f> updated = cached_cupholder_positions_;

        for (std::size_t i=0;i<cached_cupholder_positions_.size();++i)
        {
          float min_d = std::numeric_limits<float>::max(); std::size_t best_j=0;
          bool found=false;
          for (std::size_t j=0;j<new_positions.size();++j)
          {
            if (matched[j]) continue;
            float d = (cached_cupholder_positions_[i] - new_positions[j]).norm();
            if (d<min_d) {min_d=d; best_j=j; found=true;}
          }
          if (found && min_d<0.04f) {updated[i]=new_positions[best_j]; matched[best_j]=true;}
          else
            RCLCPP_WARN(get_logger(), "Cupholder %zu jumped %.3fm – keeping cached", i, min_d);
        }
        cached_cupholder_positions_ = updated;
        for (std::size_t i=0;i<updated.size();++i) {
          cupholder_history_[i].push_back(updated[i]);
          if (cupholder_history_[i].size()>MAX_HISTORY) cupholder_history_[i].pop_front();
        }
      }
    }
    if (!cupholders_initialized_) return;

    /* ----------------------------------------------------
       STEP 4 – publish TFs (unchanged except z height now correct)
       ---------------------------------------------------- */
    geometry_msgs::msg::TransformStamped cam2base;
    try {
      cam2base = tf_buffer_->lookupTransform("base_link",
                                             "D415_depth_optical_frame",
                                             tf2::TimePointZero);
    } catch (const tf2::TransformException &e) {
      RCLCPP_WARN(get_logger(), "TF lookup failed: %s", e.what());
      return;
    }

    tf2::Quaternion q_tf; tf2::fromMsg(cam2base.transform.rotation, q_tf);
    Eigen::Quaterniond q_eig(q_tf.w(), q_tf.x(), q_tf.y(), q_tf.z());
    Eigen::Matrix3d    R = q_eig.toRotationMatrix();
    Eigen::Vector3d    t(cam2base.transform.translation.x,
                         cam2base.transform.translation.y,
                         cam2base.transform.translation.z);

    std::vector<Eigen::Vector3d> cupholders_base;
    for (std::size_t i=0;i<cupholder_history_.size();++i)
    {
      if (cupholder_history_[i].empty()) continue;

      /* average history */
      Eigen::Vector3f avg = Eigen::Vector3f::Zero();
      for (const auto &p: cupholder_history_[i]) avg += p;
      avg /= static_cast<float>(cupholder_history_[i].size());

      /* camera‑frame TF */
      geometry_msgs::msg::TransformStamped tf_cam;
      tf_cam.header.stamp = now();
      tf_cam.header.frame_id = "D415_depth_optical_frame";
      tf_cam.child_frame_id  = "cupholder_" + std::to_string(i);
      tf_cam.transform.translation.x = avg.x();
      tf_cam.transform.translation.y = avg.y();
      tf_cam.transform.translation.z = avg.z();
      tf_cam.transform.rotation      = cam2base.transform.rotation; // reuse q
      tf_broadcaster_->sendTransform(tf_cam);

      /* base‑frame point */
      Eigen::Vector3d c_cam(avg.x(), avg.y(), avg.z());
      Eigen::Vector3d c_base = R * c_cam + t;
      cupholders_base.push_back(c_base);

      RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 3000,
          "[Base] cupholder_%zu: (%.3f, %.3f, %.3f)",
          i, c_base.x(), c_base.y(), c_base.z());
    }

    /* shared Z for pickup targets */
    double shared_z = 0.0;
    for (const auto &p: cupholders_base) shared_z += p.z();
    shared_z /= static_cast<double>(cupholders_base.size());

    for (std::size_t i=0;i<cupholders_base.size(); ++i)
    {
      geometry_msgs::msg::TransformStamped tf_base;
      tf_base.header.stamp = now();
      tf_base.header.frame_id = "base_link";
      tf_base.child_frame_id  = "pickup_target_" + std::to_string(i) + "_top";
      tf_base.transform.translation.x = cupholders_base[i].x();
      tf_base.transform.translation.y = cupholders_base[i].y();
      tf_base.transform.translation.z = shared_z + 0.10;   // +10 cm
      tf_base.transform.rotation.w = 1.0;
      tf_broadcaster_->sendTransform(tf_base);
    }

    /* ---- optional debug windows ------------------------ */
    if (std::getenv("DISPLAY")) {
      cv::Mat circ_dbg; cv::cvtColor(final_roi, circ_dbg, cv::COLOR_GRAY2BGR);
      for (const auto &c: circles)
        cv::circle(circ_dbg,
                   cv::Point(cvRound(c[0]), cvRound(c[1])),
                   cvRound(c[2]), cv::Scalar(0,255,0), 2);
      cv::imshow("Detected Circles in ROI", circ_dbg);
      cv::imshow("Color Image with Centers", color_dbg);
      cv::imshow("Depth Image with Centers", depth_dbg);
      cv::waitKey(1);
    }
  }
};

/* ------------------------- main --------------------------- */
int main(int argc,char** argv)
{
  rclcpp::init(argc,argv);
  rclcpp::spin(std::make_shared<CupholderDetector>());
  rclcpp::shutdown();
  return 0;
}


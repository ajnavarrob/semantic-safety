#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>

#include <fstream>
#include <string>
#include <cmath>
#include <mutex>

class VelocityTrackingLogger : public rclcpp::Node
{
public:
  VelocityTrackingLogger()
  : Node("velocity_tracking_logger")
  {
    this->declare_parameter<bool>("tracking_error_experiment", false);
    this->declare_parameter<std::string>("cmd_vel_topic", "/cmd_vel");
    this->declare_parameter<std::string>("odom_topic", "/odom");
    this->declare_parameter<std::string>("output_csv", "tracking_error_log.csv");

    tracking_error_experiment_ =
      this->get_parameter("tracking_error_experiment").as_bool();

    if (!tracking_error_experiment_) {
      RCLCPP_INFO(this->get_logger(),
        "tracking_error_experiment is false. Logger is inactive.");
      return;
    }

    cmd_vel_topic_ = this->get_parameter("cmd_vel_topic").as_string();
    odom_topic_ = this->get_parameter("odom_topic").as_string();
    output_csv_ = this->get_parameter("output_csv").as_string();

    csv_.open(output_csv_, std::ios::out);
    if (!csv_.is_open()) {
      RCLCPP_ERROR(this->get_logger(), "Failed to open CSV file: %s",
                   output_csv_.c_str());
      return;
    }

    csv_ << "time,"
         << "v_cmd,omega_cmd,"
         << "v_odom,omega_odom,"
         << "e_v,e_omega,error_norm\n";

    cmd_sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
      cmd_vel_topic_, 10,
      std::bind(&VelocityTrackingLogger::cmdCallback, this, std::placeholders::_1));

    odom_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      odom_topic_, 10,
      std::bind(&VelocityTrackingLogger::odomCallback, this, std::placeholders::_1));

    RCLCPP_INFO(this->get_logger(),
      "Velocity tracking logger active. Writing to: %s", output_csv_.c_str());
  }

  ~VelocityTrackingLogger()
  {
    if (csv_.is_open()) {
      csv_.close();
    }
  }

private:
  void cmdCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    latest_v_cmd_ = msg->linear.x;
    latest_omega_cmd_ = msg->angular.z;
    has_cmd_ = true;
  }

  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    if (!tracking_error_experiment_ || !csv_.is_open()) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (!has_cmd_) {
      return;
    }

    const double t = this->now().seconds();

    const double v_odom = msg->twist.twist.linear.x;
    const double omega_odom = msg->twist.twist.angular.z;

    const double e_v = v_odom - latest_v_cmd_;
    const double e_omega = omega_odom - latest_omega_cmd_;
    const double error_norm = std::sqrt(e_v * e_v + e_omega * e_omega);

    csv_ << t << ","
         << latest_v_cmd_ << ","
         << latest_omega_cmd_ << ","
         << v_odom << ","
         << omega_odom << ","
         << e_v << ","
         << e_omega << ","
         << error_norm << "\n";
  }

  bool tracking_error_experiment_{false};

  std::string cmd_vel_topic_;
  std::string odom_topic_;
  std::string output_csv_;

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr cmd_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;

  std::ofstream csv_;
  std::mutex mutex_;

  bool has_cmd_{false};
  double latest_v_cmd_{0.0};
  double latest_omega_cmd_{0.0};
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<VelocityTrackingLogger>());
  rclcpp::shutdown();
  return 0;
}
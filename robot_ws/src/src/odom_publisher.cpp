/**
 * @file odom_publisher.cpp
 * @brief Odometry Publisher Node
 * 
 * Subscribes to Unitree Go2 SportModeState and publishes:
 * 1. nav_msgs/Odometry on /odom
 * 2. TF broadcast: odom -> body_link
 * 
 * This centralizes odometry handling so all other nodes can use TF lookups
 * instead of each subscribing to sportmodestate directly.
 */

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include "unitree_go/msg/sport_mode_state.hpp"

#include <cmath>

class OdomPublisherNode : public rclcpp::Node {
public:
    OdomPublisherNode() : Node("odom_publisher") {
        // Subscriber to Unitree sportmodestate
        state_sub_ = this->create_subscription<unitree_go::msg::SportModeState>(
            "sportmodestate", 10,
            std::bind(&OdomPublisherNode::state_callback, this, std::placeholders::_1)
        );

        // Publisher for nav_msgs/Odometry
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);

        // TF broadcaster for odom -> body_link
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        RCLCPP_INFO(this->get_logger(), "Odom Publisher Node initialized");
        RCLCPP_INFO(this->get_logger(), "Subscribing to: sportmodestate");
        RCLCPP_INFO(this->get_logger(), "Publishing to: /odom and TF (odom -> body_link)");
    }

private:
    void state_callback(const unitree_go::msg::SportModeState::SharedPtr msg) {
        auto now = this->now();

        // Extract position
        float px = msg->position[0];
        float py = msg->position[1];
        float pz = msg->position[2];

        // Extract orientation from IMU RPY and convert to quaternion
        float roll = msg->imu_state.rpy[0];
        float pitch = msg->imu_state.rpy[1];
        float yaw = msg->imu_state.rpy[2];

        // Euler to quaternion conversion
        float cy = std::cos(yaw * 0.5f);
        float sy = std::sin(yaw * 0.5f);
        float cp = std::cos(pitch * 0.5f);
        float sp = std::sin(pitch * 0.5f);
        float cr = std::cos(roll * 0.5f);
        float sr = std::sin(roll * 0.5f);

        float qw = cr * cp * cy + sr * sp * sy;
        float qx = sr * cp * cy - cr * sp * sy;
        float qy = cr * sp * cy + sr * cp * sy;
        float qz = cr * cp * sy - sr * sp * cy;

        // --- Publish Odometry message ---
        auto odom_msg = nav_msgs::msg::Odometry();
        odom_msg.header.stamp = now;
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = "body_link";

        // Pose
        odom_msg.pose.pose.position.x = px;
        odom_msg.pose.pose.position.y = py;
        odom_msg.pose.pose.position.z = pz;
        odom_msg.pose.pose.orientation.x = qx;
        odom_msg.pose.pose.orientation.y = qy;
        odom_msg.pose.pose.orientation.z = qz;
        odom_msg.pose.pose.orientation.w = qw;

        // Twist (velocity in body frame)
        if (msg->velocity.size() >= 3) {
            odom_msg.twist.twist.linear.x = msg->velocity[0];
            odom_msg.twist.twist.linear.y = msg->velocity[1];
            odom_msg.twist.twist.linear.z = msg->velocity[2];
        }

        // Publish odometry
        odom_pub_->publish(odom_msg);

        // --- Broadcast TF: odom -> body_link ---
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = now;
        transform.header.frame_id = "odom";
        transform.child_frame_id = "body_link";

        transform.transform.translation.x = px;
        transform.transform.translation.y = py;
        transform.transform.translation.z = pz;

        transform.transform.rotation.x = qx;
        transform.transform.rotation.y = qy;
        transform.transform.rotation.z = qz;
        transform.transform.rotation.w = qw;

        tf_broadcaster_->sendTransform(transform);
    }

    rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr state_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OdomPublisherNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

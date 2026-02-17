/**
 * @file odom_publisher_fastlio.cpp
 * @brief Odometry Publisher from FAST_LIO
 * 
 * Subscribes to FAST_LIO /Odometry topic and republishes:
 * 1. nav_msgs/Odometry on /odom
 * 2. TF broadcast: odom -> livox_frame
 * 
 * FAST_LIO uses frame IDs: camera_init (world) -> body (lidar frame)
 * This node remaps to: odom -> livox_frame
 * A separate static TF provides: livox_frame -> body_link
 */

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>

class OdomPublisherFastLioNode : public rclcpp::Node {
public:
    OdomPublisherFastLioNode() : Node("odom_publisher_fastlio"), received_first_msg_(false) {
        // Declare parameter for input topic
        this->declare_parameter("fastlio_odom_topic", "/Odometry");
        std::string fastlio_topic = this->get_parameter("fastlio_odom_topic").as_string();

        // Subscriber to FAST_LIO odometry
        fastlio_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            fastlio_topic, 10,
            std::bind(&OdomPublisherFastLioNode::fastlio_callback, this, std::placeholders::_1)
        );

        // Publisher for nav_msgs/Odometry on /odom
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);

        // TF broadcaster removed - relying on FAST_LIO's TF + static TFs
        // tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        RCLCPP_INFO(this->get_logger(), "FAST_LIO Odom Publisher Node initialized");
        RCLCPP_INFO(this->get_logger(), "Subscribing to: %s (FAST_LIO frame: camera_init -> body)", fastlio_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Publishing to: /odom (TF handled by FAST_LIO + static TFs)");
    }

private:
    void fastlio_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        if (!received_first_msg_) {
            received_first_msg_ = true;
            RCLCPP_INFO(this->get_logger(), "Received first FAST_LIO odometry message!");
        }
        
        auto now = this->now();

        // FAST_LIO publishes in camera_init frame, we remap to odom
        // FAST_LIO child_frame is "body" (lidar frame), we remap to livox_frame

        // --- Publish Odometry message ---
        auto odom_msg = nav_msgs::msg::Odometry();
        odom_msg.header.stamp = now;
        odom_msg.header.frame_id = "odom";         // Remap from camera_init
        odom_msg.child_frame_id = "livox_frame";   // Remap from body (lidar frame)

        // Pose - use FAST_LIO pose directly
        odom_msg.pose = msg->pose;

        // Twist - pass through from FAST_LIO
        odom_msg.twist = msg->twist;

        // Publish odometry
        odom_pub_->publish(odom_msg);

        // Note: We DO NOT broadcast TF here anymore to avoid conflicts.
        // FAST_LIO already broadcasts camera_init -> body.
        // We have static TFs: odom -> camera_init (identity) and body -> livox_frame (identity).
        // So the chain odom -> ... -> livox_frame is already complete via FAST_LIO.
    }

    bool received_first_msg_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr fastlio_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    // std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OdomPublisherFastLioNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

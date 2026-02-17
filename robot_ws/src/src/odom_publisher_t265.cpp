/**
 * @file odom_publisher_t265.cpp
 * @brief Odometry Publisher from RealSense T265
 * 
 * Subscribes to RealSense T265 odometry and publishes:
 * 1. nav_msgs/Odometry on /odom
 * 2. TF broadcast: odom -> body_link
 * 
 * The T265 publishes its own odometry in its own frame. This node transforms
 * that to the robot body frame and publishes unified odometry.
 */

#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

#include <cmath>

class OdomPublisherT265Node : public rclcpp::Node {
public:
    OdomPublisherT265Node() : Node("odom_publisher_t265") {
        // Declare parameters for T265 to body_link transform
        // TODO: Fill in actual transform values based on T265 mounting position
        this->declare_parameter("t265_to_body_x", 0.0);  // meters
        this->declare_parameter("t265_to_body_y", 0.0);
        this->declare_parameter("t265_to_body_z", 0.0);
        this->declare_parameter("t265_to_body_roll", 0.0);  // radians
        this->declare_parameter("t265_to_body_pitch", 0.0);
        this->declare_parameter("t265_to_body_yaw", 0.0);
        
        t265_to_body_x_ = this->get_parameter("t265_to_body_x").as_double();
        t265_to_body_y_ = this->get_parameter("t265_to_body_y").as_double();
        t265_to_body_z_ = this->get_parameter("t265_to_body_z").as_double();
        t265_to_body_roll_ = this->get_parameter("t265_to_body_roll").as_double();
        t265_to_body_pitch_ = this->get_parameter("t265_to_body_pitch").as_double();
        t265_to_body_yaw_ = this->get_parameter("t265_to_body_yaw").as_double();
        
        // Declare parameter for input topic
        this->declare_parameter("t265_odom_topic", "/camera/pose/sample");
        std::string t265_topic = this->get_parameter("t265_odom_topic").as_string();

        // Subscriber to T265 odometry
        t265_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
            t265_topic, 10,
            std::bind(&OdomPublisherT265Node::t265_callback, this, std::placeholders::_1)
        );

        // Publisher for nav_msgs/Odometry
        odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("/odom", 10);

        // TF broadcaster for odom -> body_link
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        RCLCPP_INFO(this->get_logger(), "T265 Odom Publisher Node initialized");
        RCLCPP_INFO(this->get_logger(), "Subscribing to: %s", t265_topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Publishing to: /odom and TF (odom -> body_link)");
        RCLCPP_INFO(this->get_logger(), "T265->body transform: [%.3f, %.3f, %.3f] m, [%.3f, %.3f, %.3f] rad",
            t265_to_body_x_, t265_to_body_y_, t265_to_body_z_,
            t265_to_body_roll_, t265_to_body_pitch_, t265_to_body_yaw_);
    }

private:
    void t265_callback(const nav_msgs::msg::Odometry::SharedPtr msg) {
        auto now = this->now();

        // For now, use identity transform (T265 frame = body frame)
        // TODO: Apply actual transform from T265 mounting position
        
        // Extract T265 pose
        double t265_x = msg->pose.pose.position.x;
        double t265_y = msg->pose.pose.position.y;
        double t265_z = msg->pose.pose.position.z;
        
        double t265_qx = msg->pose.pose.orientation.x;
        double t265_qy = msg->pose.pose.orientation.y;
        double t265_qz = msg->pose.pose.orientation.z;
        double t265_qw = msg->pose.pose.orientation.w;
        
        // Apply identity transform for now
        // In the future, apply T265->body_link transform here
        double body_x = t265_x + t265_to_body_x_;
        double body_y = t265_y + t265_to_body_y_;
        double body_z = t265_z + t265_to_body_z_;
        
        // For rotation, just pass through for now (identity)
        // TODO: Compose with T265 mounting rotation
        double body_qx = t265_qx;
        double body_qy = t265_qy;
        double body_qz = t265_qz;
        double body_qw = t265_qw;

        // --- Publish Odometry message ---
        auto odom_msg = nav_msgs::msg::Odometry();
        odom_msg.header.stamp = now;
        odom_msg.header.frame_id = "odom";
        odom_msg.child_frame_id = "body_link";

        // Pose
        odom_msg.pose.pose.position.x = body_x;
        odom_msg.pose.pose.position.y = body_y;
        odom_msg.pose.pose.position.z = body_z;
        odom_msg.pose.pose.orientation.x = body_qx;
        odom_msg.pose.pose.orientation.y = body_qy;
        odom_msg.pose.pose.orientation.z = body_qz;
        odom_msg.pose.pose.orientation.w = body_qw;

        // Twist (pass through from T265)
        odom_msg.twist = msg->twist;

        // Publish odometry
        odom_pub_->publish(odom_msg);

        // --- Broadcast TF: odom -> body_link ---
        geometry_msgs::msg::TransformStamped transform;
        transform.header.stamp = now;
        transform.header.frame_id = "odom";
        transform.child_frame_id = "body_link";

        transform.transform.translation.x = body_x;
        transform.transform.translation.y = body_y;
        transform.transform.translation.z = body_z;

        transform.transform.rotation.x = body_qx;
        transform.transform.rotation.y = body_qy;
        transform.transform.rotation.z = body_qz;
        transform.transform.rotation.w = body_qw;

        tf_broadcaster_->sendTransform(transform);
    }

    // T265 to body_link transform parameters
    double t265_to_body_x_, t265_to_body_y_, t265_to_body_z_;
    double t265_to_body_roll_, t265_to_body_pitch_, t265_to_body_yaw_;

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr t265_sub_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OdomPublisherT265Node>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}

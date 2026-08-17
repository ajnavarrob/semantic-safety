#pragma once

#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>

#include <opencv2/opencv.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <cmath>
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cstring>
#include <limits>
#include <string>

#include "utils.h"
#include "poisson.h"
#include <mutex>

bool initialized = false;

const float minX = 0.40f;
const float maxX = (float)(JMAX/2) * DS;
const float minY = 0.20f;
const float maxY = (float)(IMAX/2) * DS;
const float minZ_default = 0.05f;
const float maxZ_default = 1.50f;

class CloudMergerNode : public rclcpp::Node {

public:

    CloudMergerNode(float min_z_override = -1.0f, float max_z_override = -1.0f)
        : Node("cloud_merger") {

        this->declare_parameter("min_z", minZ_default);
        this->declare_parameter("max_z", maxZ_default);

        // Raw semantic class map in body_link. The persistent map is
        // published on a separate topic to avoid subscribing to our own output.
        this->declare_parameter<std::string>(
            "semantic_class_map_input_topic",
            "/class_map"
        );
        this->declare_parameter<std::string>(
            "semantic_class_map_output_topic",
            "/class_map_persistent"
        );

        // Persistence durations [s] for non-human semantic classes.
        // Class IDs:
        //   1 human                -> not persisted here
        //   2 traffic_cone
        //   3 caution_tape
        //   4 floor_danger_tape
        //   5 wet_floor_sign
        //   6 spill
        this->declare_parameter("semantic_persistence_traffic_cone_sec", 1.5);
        this->declare_parameter("semantic_persistence_caution_tape_sec", 2.5);
        this->declare_parameter("semantic_persistence_floor_danger_tape_sec", 2.5);
        this->declare_parameter("semantic_persistence_wet_floor_sign_sec", 2.0);
        this->declare_parameter("semantic_persistence_spill_sec", 2.5);

        if (min_z_override >= 0.0f) {
            minZ_ = min_z_override;
        } else {
            minZ_ = this->get_parameter("min_z").as_double();
        }

        if (max_z_override > 0.0f) {
            maxZ_ = max_z_override;
        } else {
            maxZ_ = this->get_parameter("max_z").as_double();
        }

        semantic_class_map_input_topic_ =
            this->get_parameter("semantic_class_map_input_topic").as_string();

        semantic_class_map_output_topic_ =
            this->get_parameter("semantic_class_map_output_topic").as_string();

        semantic_persistence_sec_[0] = 0.0;
        semantic_persistence_sec_[1] = 0.0;  // human handled elsewhere
        semantic_persistence_sec_[2] =
            this->get_parameter(
                "semantic_persistence_traffic_cone_sec"
            ).as_double();
        semantic_persistence_sec_[3] =
            this->get_parameter(
                "semantic_persistence_caution_tape_sec"
            ).as_double();
        semantic_persistence_sec_[4] =
            this->get_parameter(
                "semantic_persistence_floor_danger_tape_sec"
            ).as_double();
        semantic_persistence_sec_[5] =
            this->get_parameter(
                "semantic_persistence_wet_floor_sign_sec"
            ).as_double();
        semantic_persistence_sec_[6] =
            this->get_parameter(
                "semantic_persistence_spill_sec"
            ).as_double();

        for (int cls = FIRST_PERSISTED_CLASS_;
             cls <= LAST_PERSISTED_CLASS_;
             ++cls) {
            semantic_persistence_sec_[cls] =
                std::max(0.0, semantic_persistence_sec_[cls]);
        }

        RCLCPP_INFO(
            this->get_logger(),
            "CloudMerger min_z=%.2f, max_z=%.2f",
            minZ_,
            maxZ_
        );

        RCLCPP_INFO(
            this->get_logger(),
            "Semantic persistence: input=%s output=%s "
            "durations=[cone %.2fs, caution %.2fs, floor_tape %.2fs, sign %.2fs, spill %.2fs]",
            semantic_class_map_input_topic_.c_str(),
            semantic_class_map_output_topic_.c_str(),
            semantic_persistence_sec_[2],
            semantic_persistence_sec_[3],
            semantic_persistence_sec_[4],
            semantic_persistence_sec_[5],
            semantic_persistence_sec_[6]
        );

        cloud_msg.header.stamp = this->now();
        cloud_msg.header.frame_id = "body_link";

        initialize_occupancy_msg(physical_map_msg_, "body_link");
        initialize_occupancy_msg(semantic_map_msg_, "body_link");
        initialize_occupancy_msg(effective_map_msg_, "body_link");

        for (int i = 0; i < IMAX; i++) {
            for (int j = 0; j < JMAX; j++) {
                const float x = (float)(j - JMAX / 2) * DS;
                const float y = (float)(i - IMAX / 2) * DS;
                polar_coordinates_r2[i * JMAX + j] = x * x + y * y;
                polar_coordinates_th[i * JMAX + j] = std::atan2(y, x);
                old_conf[i * JMAX + j] = 0;
                semantic_confidence_values_[i * JMAX + j] = 0;

                const int n = i * JMAX + j;
                persistent_class_map_[n] = 0;
                semantic_last_seen_class_[n] = 0;
                semantic_last_seen_time_[n] =
                    std::chrono::steady_clock::time_point::min();
            }
        }

        t = std::chrono::steady_clock::now();

        livox_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/livox/lidar",
            15,
            std::bind(&CloudMergerNode::lidar_callback, this, std::placeholders::_1)
        );

        cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "poisson_cloud",
            1
        );

        physical_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
            "physical_occupancy_grid",
            1
        );

        semantic_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
            "semantic_occupancy_grid",
            1
        );

        // Keep this topic name unchanged so semantic_poisson still receives the composed map.
        effective_map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(
            "occupancy_grid",
            1
        );

        target_frame_ = "body_link";
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        camera_front_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/camera_front/point_cloud/cloud_registered",
            5,
            std::bind(&CloudMergerNode::camera_callback, this, std::placeholders::_1)
        );

        camera_rear_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/camera_rear/point_cloud/cloud_registered",
            5,
            std::bind(&CloudMergerNode::camera_callback, this, std::placeholders::_1)
        );

        combined_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());

        semantic_class_map_pub_ =
            this->create_publisher<nav_msgs::msg::OccupancyGrid>(
                semantic_class_map_output_topic_,
                rclcpp::QoS(rclcpp::KeepLast(1)).reliable()
            );

        semantic_class_map_sub_ =
            this->create_subscription<nav_msgs::msg::OccupancyGrid>(
                semantic_class_map_input_topic_,
                rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
                std::bind(
                    &CloudMergerNode::semantic_class_map_callback,
                    this,
                    std::placeholders::_1
                )
            );

        occupancy_publish_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(67),
            std::bind(&CloudMergerNode::publish_occupancy_from_combined_cloud, this)
        );
    }

private:

    void initialize_occupancy_msg(nav_msgs::msg::OccupancyGrid& msg,
                                  const std::string& frame_id) {
        msg.data.resize(IMAX * JMAX);
        msg.header.stamp = this->now();
        msg.header.frame_id = frame_id;
        msg.info.width = JMAX;
        msg.info.height = IMAX;
        msg.info.resolution = DS;
        msg.info.origin.position.x = -maxX;
        msg.info.origin.position.y = -maxY;
        msg.info.origin.position.z = 0.0f;
        msg.info.origin.orientation.w = 1.0;
        msg.info.origin.orientation.x = 0.0f;
        msg.info.origin.orientation.y = 0.0f;
        msg.info.origin.orientation.z = 0.0f;

        std::fill(msg.data.begin(), msg.data.end(), 0);
    }

    void publish_occupancy_from_combined_cloud() {
        Timer map_timer(true);
        map_timer.start();

        dt = std::chrono::duration<float>(
            std::chrono::steady_clock::now() - t
        ).count();

        t = std::chrono::steady_clock::now();
        dt = std::clamp(dt, 0.0f, 0.5f);

        pcl::PointCloud<pcl::PointXYZI>::Ptr odom_cloud(
            new pcl::PointCloud<pcl::PointXYZI>
        );

        {
            std::lock_guard<std::mutex> lock(combined_cloud_mutex_);
            *odom_cloud += *combined_cloud_;
            combined_cloud_->clear();
        }

        if (odom_cloud->points.empty()) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,
                "CloudMerger received empty combined cloud; keeping previous occupancy map"
            );
            return;
        }

        cv::Mat raw_map = cv::Mat::zeros(IMAX, JMAX, CV_32F);
        int valid_points = 0;

        for (const auto& pt : odom_cloud->points) {
            if (!(pt.z > minZ_ && pt.z < maxZ_)) {
                continue;
            }

            const float ic = pt.y / DS + static_cast<float>(IMAX / 2);
            const float jc = pt.x / DS + static_cast<float>(JMAX / 2);

            if (ic <= 0.0f || ic >= static_cast<float>(IMAX - 1) ||
                jc <= 0.0f || jc >= static_cast<float>(JMAX - 1)) {
                continue;
            }

            valid_points++;

            raw_map.at<float>(
                static_cast<int>(std::round(ic)),
                static_cast<int>(std::round(jc))
            ) = 1.0f;
        }

        if (valid_points == 0) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,
                "CloudMerger valid_points=0 after filtering; keeping previous occupancy map"
            );
            return;
        }

        for (int n = 0; n < IMAX * JMAX; n++) {
            physical_confidence_values_[n] = 0;
        }

        Filtered_Occupancy_Convolution(
            physical_confidence_values_,
            raw_map,
            old_conf
        );

        std::memcpy(
            old_conf,
            physical_confidence_values_,
            IMAX * JMAX * sizeof(int8_t)
        );

        pcl::toROSMsg(*odom_cloud, cloud_msg);
        cloud_msg.header.stamp = this->now();
        cloud_msg.header.frame_id = "body_link";
        cloud_pub_->publish(cloud_msg);

        compose_and_publish_occupancy_layers();

        static int timing_print_counter = 0;
        if (++timing_print_counter >= 15) {
            timing_print_counter = 0;
            map_timer.time("Occ Map Solve Time: ");
        }
    }

    void compose_and_publish_occupancy_layers() {
        const auto stamp = this->now();

        physical_map_msg_.header.stamp = stamp;
        semantic_map_msg_.header.stamp = stamp;
        effective_map_msg_.header.stamp = stamp;

        physical_map_msg_.info.origin.position.x = -maxX;
        physical_map_msg_.info.origin.position.y = -maxY;

        semantic_map_msg_.info.origin.position.x = -maxX;
        semantic_map_msg_.info.origin.position.y = -maxY;

        effective_map_msg_.info.origin.position.x = -maxX;
        effective_map_msg_.info.origin.position.y = -maxY;

        for (int n = 0; n < IMAX * JMAX; n++) {
            physical_map_msg_.data[n] = physical_confidence_values_[n];

            // For now, semantic occupancy is empty.
            // Later, language-derived buffers and relational regions should write here.
            semantic_map_msg_.data[n] = semantic_confidence_values_[n];

            // Effective occupancy is the union of physical and semantic occupancy.
            effective_map_msg_.data[n] = std::max(
                physical_map_msg_.data[n],
                semantic_map_msg_.data[n]
            );
        }

        physical_map_pub_->publish(physical_map_msg_);
        semantic_map_pub_->publish(semantic_map_msg_);

        // Downstream Poisson still subscribes to "occupancy_grid".
        effective_map_pub_->publish(effective_map_msg_);
    }

    void combined_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pcl::PointCloud<pcl::PointXYZI>::Ptr odom_cloud(
            new pcl::PointCloud<pcl::PointXYZI>
        );

        pcl::fromROSMsg(*msg, *odom_cloud);

        if (!transform_pointcloud(odom_cloud, msg->header.frame_id)) {
            return;
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(
            new pcl::PointCloud<pcl::PointXYZI>
        );

        for (const auto& pt : odom_cloud->points) {
            float ellipse_norm =
                std::pow(pt.x / minX, 8.0f) +
                std::pow(pt.y / minY, 8.0f);

            if (ellipse_norm > 1.0f) {
                filtered->points.push_back(pt);
            }
        }

        filtered->width = filtered->points.size();
        filtered->height = 1;

        {
            std::lock_guard<std::mutex> lock(combined_cloud_mutex_);
            *combined_cloud_ += *filtered;
        }
    }

    void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        if (!update_pose_from_tf()) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Waiting for TF odom -> body_link, skipping lidar frame"
            );
            return;
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
            new pcl::PointCloud<pcl::PointXYZI>
        );

        pcl::fromROSMsg(*msg, *cloud);

        if (!transform_pointcloud(cloud, msg->header.frame_id)) {
            return;
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(
            new pcl::PointCloud<pcl::PointXYZI>
        );

        for (const auto& pt : cloud->points) {
            float ellipse_norm =
                std::pow(pt.x / minX, 8.0f) +
                std::pow(pt.y / minY, 8.0f);

            if (ellipse_norm > 1.0f) {
                filtered->points.push_back(pt);
            }
        }

        filtered->width = filtered->points.size();
        filtered->height = 1;

        {
            std::lock_guard<std::mutex> lock(combined_cloud_mutex_);
            *combined_cloud_ += *filtered;
        }
    }

    bool update_pose_from_tf() {
        try {
            if (!tf_buffer_->canTransform("odom", "body_link", tf2::TimePointZero)) {
                return false;
            }

            auto transform = tf_buffer_->lookupTransform(
                "odom",
                "body_link",
                tf2::TimePointZero,
                tf2::durationFromSec(0.05)
            );

            r[0] = transform.transform.translation.x;
            r[1] = transform.transform.translation.y;
            r[2] = transform.transform.translation.z;

            auto& q = transform.transform.rotation;

            double sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z);
            double cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
            rpy[0] = std::atan2(sinr_cosp, cosr_cosp);

            double sinp = 2.0 * (q.w * q.y - q.z * q.x);
            if (std::abs(sinp) >= 1) {
                rpy[1] = std::copysign(M_PI / 2, sinp);
            } else {
                rpy[1] = std::asin(sinp);
            }

            double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
            double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
            rpy[2] = std::atan2(siny_cosp, cosy_cosp);

            return true;

        } catch (tf2::TransformException& ex) {
            return false;
        }
    }

    bool transform_pointcloud(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
                              const std::string& source_frame) {
        if (source_frame == target_frame_) {
            return true;
        }

        if (source_frame.empty()) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                5000,
                "Source frame is empty, skipping transform"
            );
            return false;
        }

        if (!tf_buffer_) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                5000,
                "TF buffer not initialized yet"
            );
            return false;
        }

        try {
            if (!tf_buffer_->canTransform(target_frame_, source_frame, tf2::TimePointZero)) {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(),
                    *this->get_clock(),
                    2000,
                    "Waiting for TF from %s to %s...",
                    source_frame.c_str(),
                    target_frame_.c_str()
                );
                return false;
            }

            geometry_msgs::msg::TransformStamped transform =
                tf_buffer_->lookupTransform(
                    target_frame_,
                    source_frame,
                    tf2::TimePointZero,
                    tf2::durationFromSec(0.1)
                );

            Eigen::Affine3d eigen_transform =
                tf2::transformToEigen(transform.transform);

            pcl::transformPointCloud(*cloud, *cloud, eigen_transform.cast<float>());
            return true;

        } catch (tf2::TransformException& ex) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "TF transform failed from %s to %s: %s",
                source_frame.c_str(),
                target_frame_.c_str(),
                ex.what()
            );
            return false;

        } catch (std::exception& ex) {
            RCLCPP_ERROR_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Unexpected error in transform: %s",
                ex.what()
            );
            return false;

        } catch (...) {
            RCLCPP_ERROR_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Unknown error in transform"
            );
            return false;
        }
    }

    void camera_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz(
            new pcl::PointCloud<pcl::PointXYZ>
        );

        pcl::fromROSMsg(*msg, *cloud_xyz);

        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(
            new pcl::PointCloud<pcl::PointXYZI>
        );

        cloud->resize(cloud_xyz->size());

        for (size_t i = 0; i < cloud_xyz->size(); i++) {
            cloud->points[i].x = cloud_xyz->points[i].x;
            cloud->points[i].y = cloud_xyz->points[i].y;
            cloud->points[i].z = cloud_xyz->points[i].z;
            cloud->points[i].intensity = 1.0f;
        }

        if (!transform_pointcloud(cloud, msg->header.frame_id)) {
            return;
        }

        pcl::PointCloud<pcl::PointXYZI>::Ptr filtered(
            new pcl::PointCloud<pcl::PointXYZI>
        );

        for (const auto& pt : cloud->points) {
            float ellipse_norm =
                std::pow(pt.x / minX, 8.0f) +
                std::pow(pt.y / minY, 8.0f);

            if (ellipse_norm > 1.0f) {
                filtered->points.push_back(pt);
            }
        }

        filtered->width = filtered->points.size();
        filtered->height = 1;

        {
            std::lock_guard<std::mutex> lock(combined_cloud_mutex_);
            *combined_cloud_ += *filtered;
        }
    }

    bool lookup_body_pose(
        float& x_world,
        float& y_world,
        float& yaw_world
    ) {
        try {
            if (!tf_buffer_ ||
                !tf_buffer_->canTransform(
                    "odom",
                    "body_link",
                    tf2::TimePointZero
                )) {
                return false;
            }

            const auto transform =
                tf_buffer_->lookupTransform(
                    "odom",
                    "body_link",
                    tf2::TimePointZero,
                    tf2::durationFromSec(0.05)
                );

            x_world =
                static_cast<float>(
                    transform.transform.translation.x
                );

            y_world =
                static_cast<float>(
                    transform.transform.translation.y
                );

            const auto& q = transform.transform.rotation;

            const double siny_cosp =
                2.0 * (q.w * q.z + q.x * q.y);

            const double cosy_cosp =
                1.0 - 2.0 * (q.y * q.y + q.z * q.z);

            yaw_world =
                static_cast<float>(
                    std::atan2(siny_cosp, cosy_cosp)
                );

            return true;

        } catch (const tf2::TransformException&) {
            return false;
        }
    }

    void warp_persistent_semantic_state_to_current_body(
        float current_x_world,
        float current_y_world,
        float current_yaw_world,
        std::array<int8_t, IMAX * JMAX>& warped_class,
        std::array<std::chrono::steady_clock::time_point,
                   IMAX * JMAX>& warped_time
    ) {
        warped_class.fill(0);
        warped_time.fill(
            std::chrono::steady_clock::time_point::min()
        );

        if (!semantic_pose_initialized_) {
            warped_class = semantic_last_seen_class_;
            warped_time = semantic_last_seen_time_;
            return;
        }

        const float c_cur = std::cos(current_yaw_world);
        const float s_cur = std::sin(current_yaw_world);

        const float c_old = std::cos(semantic_prev_yaw_world_);
        const float s_old = std::sin(semantic_prev_yaw_world_);

        // Inverse resampling:
        // current body cell -> odom/world -> previous body cell.
        // This keeps persisted semantic evidence approximately stationary
        // in the world while the robot moves.
        for (int i = 0; i < IMAX; ++i) {
            for (int j = 0; j < JMAX; ++j) {

                const int n_cur = i * JMAX + j;

                const float x_cur_body =
                    (static_cast<float>(j) -
                     0.5f * static_cast<float>(JMAX)) * DS;

                const float y_cur_body =
                    (static_cast<float>(i) -
                     0.5f * static_cast<float>(IMAX)) * DS;

                const float x_world =
                    current_x_world +
                    c_cur * x_cur_body -
                    s_cur * y_cur_body;

                const float y_world =
                    current_y_world +
                    s_cur * x_cur_body +
                    c_cur * y_cur_body;

                const float dx_old =
                    x_world - semantic_prev_x_world_;

                const float dy_old =
                    y_world - semantic_prev_y_world_;

                const float x_old_body =
                    c_old * dx_old +
                    s_old * dy_old;

                const float y_old_body =
                    -s_old * dx_old +
                    c_old * dy_old;

                const float jf =
                    x_old_body / DS +
                    0.5f * static_cast<float>(JMAX);

                const float if_ =
                    y_old_body / DS +
                    0.5f * static_cast<float>(IMAX);

                const int j_old =
                    static_cast<int>(
                        std::round(jf)
                    );

                const int i_old =
                    static_cast<int>(
                        std::round(if_)
                    );

                if (i_old < 0 || i_old >= IMAX ||
                    j_old < 0 || j_old >= JMAX) {
                    continue;
                }

                const int n_old =
                    i_old * JMAX + j_old;

                warped_class[n_cur] =
                    semantic_last_seen_class_[n_old];

                warped_time[n_cur] =
                    semantic_last_seen_time_[n_old];
            }
        }
    }

    void semantic_class_map_callback(
        const nav_msgs::msg::OccupancyGrid::SharedPtr msg
    ) {
        if (!msg) {
            return;
        }

        if (msg->data.size() !=
            static_cast<std::size_t>(IMAX * JMAX)) {

            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,
                "Semantic class map size mismatch: got %zu expected %d",
                msg->data.size(),
                IMAX * JMAX
            );
            return;
        }

        if (msg->info.width !=
                static_cast<std::uint32_t>(JMAX) ||
            msg->info.height !=
                static_cast<std::uint32_t>(IMAX)) {

            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,
                "Semantic class map geometry mismatch: got %ux%u expected %dx%d",
                msg->info.width,
                msg->info.height,
                JMAX,
                IMAX
            );
            return;
        }

        if (std::abs(
                msg->info.resolution - DS
            ) > 1.0e-5f) {

            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,
                "Semantic class map resolution mismatch: got %.6f expected %.6f",
                msg->info.resolution,
                static_cast<double>(DS)
            );
            return;
        }

        std::lock_guard<std::mutex>
            lock(semantic_persistence_mutex_);

        const auto now =
            std::chrono::steady_clock::now();

        float current_x_world = 0.0f;
        float current_y_world = 0.0f;
        float current_yaw_world = 0.0f;

        const bool have_pose =
            lookup_body_pose(
                current_x_world,
                current_y_world,
                current_yaw_world
            );

        std::array<int8_t, IMAX * JMAX>
            warped_class{};

        std::array<std::chrono::steady_clock::time_point,
                   IMAX * JMAX>
            warped_time{};

        if (have_pose) {
            warp_persistent_semantic_state_to_current_body(
                current_x_world,
                current_y_world,
                current_yaw_world,
                warped_class,
                warped_time
            );
        } else {
            warped_class =
                semantic_last_seen_class_;

            warped_time =
                semantic_last_seen_time_;

            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "No odom->body_link TF for semantic persistence; "
                "retaining semantic cells without ego-motion compensation"
            );
        }

        semantic_last_seen_class_ = warped_class;
        semantic_last_seen_time_ = warped_time;

        // Current detections refresh the stored class and timestamp.
        // Human class 1 is deliberately NOT persisted here; it passes
        // through only when currently detected.
        for (int n = 0;
             n < IMAX * JMAX;
             ++n) {

            const int raw_class =
                static_cast<int>(
                    msg->data[n]
                );

            if (raw_class >= FIRST_PERSISTED_CLASS_ &&
                raw_class <= LAST_PERSISTED_CLASS_) {

                semantic_last_seen_class_[n] =
                    static_cast<int8_t>(raw_class);

                semantic_last_seen_time_[n] =
                    now;
            }
        }

        // Build the integer persistent class map.
        for (int n = 0;
             n < IMAX * JMAX;
             ++n) {

            const int raw_class =
                static_cast<int>(
                    msg->data[n]
                );

            // Any current nonzero measurement wins immediately.
            // This includes humans, which remain instantaneous here.
            if (raw_class != 0) {
                persistent_class_map_[n] =
                    static_cast<int8_t>(
                        raw_class
                    );
                continue;
            }

            const int stored_class =
                static_cast<int>(
                    semantic_last_seen_class_[n]
                );

            if (stored_class < FIRST_PERSISTED_CLASS_ ||
                stored_class > LAST_PERSISTED_CLASS_) {

                persistent_class_map_[n] = 0;
                continue;
            }

            const auto last_seen =
                semantic_last_seen_time_[n];

            if (last_seen ==
                std::chrono::steady_clock::time_point::min()) {

                persistent_class_map_[n] = 0;
                continue;
            }

            const double age_sec =
                std::chrono::duration<double>(
                    now - last_seen
                ).count();

            if (age_sec <=
                semantic_persistence_sec_[stored_class]) {

                persistent_class_map_[n] =
                    static_cast<int8_t>(
                        stored_class
                    );

            } else {
                persistent_class_map_[n] = 0;

                semantic_last_seen_class_[n] = 0;

                semantic_last_seen_time_[n] =
                    std::chrono::steady_clock::time_point::min();
            }
        }

        nav_msgs::msg::OccupancyGrid out =
            *msg;

        out.header.stamp =
            this->now();

        out.data.assign(
            persistent_class_map_.begin(),
            persistent_class_map_.end()
        );

        semantic_class_map_pub_->publish(out);

        if (have_pose) {
            semantic_prev_x_world_ =
                current_x_world;

            semantic_prev_y_world_ =
                current_y_world;

            semantic_prev_yaw_world_ =
                current_yaw_world;

            semantic_pose_initialized_ =
                true;
        }
    }

    cv::Mat gaussian_kernel(int kernel_size, float sigma) {
        cv::Mat kernel(kernel_size, kernel_size, CV_32F);

        int half = kernel_size / 2;

        for (int i = -half; i <= half; i++) {
            for (int j = -half; j <= half; j++) {
                float val = std::exp(-(i * i + j * j) / (2.0 * sigma * sigma));
                kernel.at<float>(i + half, j + half) = val;
            }
        }

        return kernel;
    }

    void Filtered_Occupancy_Convolution(
        int8_t* confidence_values,
        const cv::Mat& occupancy_data,
        const int8_t* old_conf_map
    ) {
        for (int i = 0; i < IMAX; i++) {
            for (int j = 0; j < JMAX; j++) {
                confidence_values[i * JMAX + j] = old_conf_map[i * JMAX + j];
            }
        }

        cv::filter2D(
            occupancy_data,
            buffered_binary,
            -1,
            gauss_kernel,
            cv::Point(-1, -1),
            0,
            cv::BORDER_CONSTANT
        );

        float sig;
        float C;
        float beta_up;
        float beta_dn;

        const float thresh_front = 2.0f;
        const float thresh_mid360 = 4.0f;

        bool front_flag = true;

        for (int i = 0; i < IMAX; i++) {
            for (int j = 0; j < JMAX; j++) {
                const float r2 = polar_coordinates_r2[i * JMAX + j];
                const float th = polar_coordinates_th[i * JMAX + j];

                const bool range_flag = r2 > 1.44f;
                const bool angle_flag = std::abs(ang_diff(0.0f, th)) > 0.6f;

                if (range_flag || angle_flag) {
                    front_flag = false;
                } else {
                    front_flag = true;
                }

                const float thresh = front_flag ? thresh_front : thresh_mid360;

                float val_binary = buffered_binary.at<float>(i, j);
                float conf = static_cast<float>(confidence_values[i * JMAX + j]) / 127.0f;

                if (val_binary > thresh) {
                    if (front_flag) {
                        beta_up = 4.0f;
                    } else {
                        beta_up = 1.0f;
                    }

                    sig = 1.0f - std::exp(-beta_up * val_binary * dt);
                    C = 1.0f;

                } else {
                    if (front_flag) {
                        beta_dn = 4.0f;
                    } else {
                        beta_dn = 4.0f;
                    }

                    sig = 1.0f - std::exp(-beta_dn * dt);
                    C = 0.0f;
                }

                conf *= 1.0f - sig;
                conf += sig * C;

                confidence_values[i * JMAX + j] =
                    static_cast<int8_t>(std::round(127.0f * conf));
            }
        }
    }

private:

    sensor_msgs::msg::PointCloud2 cloud_msg;

    nav_msgs::msg::OccupancyGrid physical_map_msg_;
    nav_msgs::msg::OccupancyGrid semantic_map_msg_;
    nav_msgs::msg::OccupancyGrid effective_map_msg_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr livox_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr utlidar_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr camera_front_sub_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr camera_rear_sub_;

    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr
        semantic_class_map_sub_;

    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr
        semantic_class_map_pub_;

    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;

    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr physical_map_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr semantic_map_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr effective_map_pub_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
    std::string target_frame_;

    rclcpp::TimerBase::SharedPtr occupancy_publish_timer_;
    std::mutex combined_cloud_mutex_;

    pcl::PointCloud<pcl::PointXYZI>::Ptr combined_cloud_;

    float minZ_ = minZ_default;
    float maxZ_ = maxZ_default;

    std::vector<float> r = {0.0f, 0.0f, 0.0f};
    std::vector<float> r_map = {0.0f, 0.0f, 0.0f};
    std::vector<float> rpy = {0.0f, 0.0f, 0.0f};

    std::chrono::steady_clock::time_point t;
    float dt = 1.0e10f;

    const cv::Mat gauss_kernel = gaussian_kernel(9, 2.0);

    int8_t physical_confidence_values_[IMAX * JMAX];
    int8_t semantic_confidence_values_[IMAX * JMAX];

    int8_t old_conf[IMAX * JMAX];

    // ------------------------------------------------------------
    // Semantic class persistence
    // ------------------------------------------------------------
    // Legacy class IDs:
    //   1 human
    //   2 traffic_cone
    //   3 caution_tape
    //   4 floor_danger_tape
    //   5 wet_floor_sign
    //   6 spill
    //
    // Humans are intentionally excluded from this persistence layer.
    static constexpr int NUM_SEMANTIC_CLASSES_ = 7;

    // Human persistence is being accounted for on semantic_poisson.cpp
    static constexpr int FIRST_PERSISTED_CLASS_ = 2;
    static constexpr int LAST_PERSISTED_CLASS_ = 6;

    // Published semantic class IDs. These remain integer-valued.
    std::array<int8_t, IMAX * JMAX>
        persistent_class_map_{};

    // Per-cell stored semantic class and last observation time.
    std::array<int8_t, IMAX * JMAX>
        semantic_last_seen_class_{};

    std::array<std::chrono::steady_clock::time_point,
               IMAX * JMAX>
        semantic_last_seen_time_{};

    // Class-specific persistence duration in seconds.
    std::array<double, NUM_SEMANTIC_CLASSES_>
        semantic_persistence_sec_{};

    std::string
        semantic_class_map_input_topic_{"/class_map"};

    std::string
        semantic_class_map_output_topic_{"/class_map_persistent"};

    bool semantic_pose_initialized_{false};

    float semantic_prev_x_world_{0.0f};
    float semantic_prev_y_world_{0.0f};
    float semantic_prev_yaw_world_{0.0f};

    std::mutex semantic_persistence_mutex_;

    float polar_coordinates_r2[IMAX * JMAX];
    float polar_coordinates_th[IMAX * JMAX];

    cv::Mat buffered_binary = cv::Mat::zeros(IMAX, JMAX, CV_32F);
};

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
#include <cstring>

#include "utils.h"
#include "poisson.h"
#include <mutex>

bool initialized = false;

const float minX = 0.40f;
const float maxX = (float)(JMAX/2) * DS;
const float minY = 0.20f;
const float maxY = (float)(IMAX/2) * DS;
const float minZ_default = 0.1f;
const float maxZ_default = 0.80f;

class CloudMergerNode : public rclcpp::Node {

public:

    CloudMergerNode(float min_z_override = -1.0f, float max_z_override = -1.0f)
        : Node("cloud_merger") {

        this->declare_parameter("min_z", minZ_default);
        this->declare_parameter("max_z", maxZ_default);

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

        RCLCPP_INFO(this->get_logger(), "CloudMerger min_z=%.2f, max_z=%.2f", minZ_, maxZ_);

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

        occupancy_publish_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(100),
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

    float polar_coordinates_r2[IMAX * JMAX];
    float polar_coordinates_th[IMAX * JMAX];

    cv::Mat buffered_binary = cv::Mat::zeros(IMAX, JMAX, CV_32F);
};
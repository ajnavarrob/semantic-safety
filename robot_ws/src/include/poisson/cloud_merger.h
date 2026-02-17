#pragma once

#include <rclcpp/qos.hpp>
#include <rclcpp/rclcpp.hpp>

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
// Removed: unitree_go/msg/sport_mode_state.hpp - now using TF for robot pose
#include <tf2_ros/static_transform_broadcaster.h>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>  // Works on both Foxy and Humble

#include <opencv2/opencv.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

// Ground detection
#include <pcl/filters/extract_indices.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/segmentation/sac_segmentation.h>

#include <cmath>
#include <Eigen/Dense>

#include "utils.h"
#include "poisson.h"

bool initialized = false;

const float minX = 0.40f; // Must be >= 0.370
const float maxX = (float)(JMAX/2) * DS;
const float minY = 0.20f; // Must be >= 0.185
const float maxY = (float)(IMAX/2) * DS;
const float minZ_default = 0.05f;  // Default value, overridable via ROS param
const float maxZ_default = 0.80f;  // Default value, overridable via ROS param

class CloudMergerNode : public rclcpp::Node{
    
    public:
        
        // Constructor with optional min_z/max_z overrides (for when launched with semantic_poisson)
        CloudMergerNode(float min_z_override = -1.0f, float max_z_override = -1.0f) : Node("cloud_merger"){

            // Use overrides if provided, otherwise use ROS parameter or default
            this->declare_parameter("min_z", minZ_default);
            this->declare_parameter("max_z", maxZ_default);
            if(min_z_override >= 0.0f){
                minZ_ = min_z_override;
            } else {
                minZ_ = this->get_parameter("min_z").as_double();
            }
            if(max_z_override > 0.0f){
                maxZ_ = max_z_override;
            } else {
                maxZ_ = this->get_parameter("max_z").as_double();
            }
            RCLCPP_INFO(this->get_logger(), "CloudMerger min_z=%.2f, max_z=%.2f", minZ_, maxZ_);

            // Initialize Cloud Message
            cloud_msg.header.stamp = this->now();
            cloud_msg.header.frame_id = "body_link";
            
            // Initialize Map Message
            map_msg.data.resize(IMAX*JMAX);
            map_msg.header.stamp = this->now();
            map_msg.header.frame_id = "body_link";
            map_msg.info.width  = IMAX;
            map_msg.info.height = JMAX;
            map_msg.info.resolution = DS;
            // body_link frame is robot-centered, origin is just offset by grid half-size
            map_msg.info.origin.position.x = -maxX;
            map_msg.info.origin.position.y = -maxY;
            map_msg.info.origin.position.z = 0.0f;
            map_msg.info.origin.orientation.w = 1.0;
            map_msg.info.origin.orientation.x = 0.0f;
            map_msg.info.origin.orientation.y = 0.0f;
            map_msg.info.origin.orientation.z = 0.0f;

            // Construct Initial Grids
            for(int i=0; i<IMAX; i++){
                for(int j=0; j<JMAX; j++){
                    const float x = (float)(j-JMAX/2) * DS;
                    const float y = (float)(i-IMAX/2) * DS;
                    polar_coordinates_r2[i*JMAX+j] = x*x+y*y;
                    polar_coordinates_th[i*JMAX+j] = std::atan2(y,x);
                    old_conf[i*JMAX+j] = 0;
                }
            }

            // Start Time
            t = std::chrono::steady_clock::now();

            // Create Subscribers & Publishers
            livox_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/livox/lidar", 1, std::bind(&CloudMergerNode::lidar_callback, this, std::placeholders::_1));
            utlidar_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>("/utlidar/cloud", 1, std::bind(&CloudMergerNode::combined_callback, this, std::placeholders::_1));
            // Removed SportModeState subscription - now using TF for robot pose
            cloud_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("poisson_cloud", 1);
            map_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("occupancy_grid", 1);

            // TF2 setup for transforms - all point clouds transformed to body_link frame
            target_frame_ = "body_link";
            tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
            tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

            // Camera subscription for RealSense D435 pointcloud
            camera_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
                "/camera/point_cloud/cloud_registered", 1,
                std::bind(&CloudMergerNode::camera_callback, this, std::placeholders::_1)
            );

            combined_cloud_.reset(new pcl::PointCloud<pcl::PointXYZI>());

        }

    private:
        
        void combined_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg){
            
            //Start timer
            Timer map_timer(true);
            map_timer.start();

            dt = std::chrono::duration<float>(std::chrono::steady_clock::now() - t).count();
            t = std::chrono::steady_clock::now();
            
            pcl::PointCloud<pcl::PointXYZI>::Ptr odom_cloud (new pcl::PointCloud<pcl::PointXYZI>);
            pcl::fromROSMsg(*msg, *odom_cloud);

            // Transform UTLidar cloud to body_link frame using TF
            // This handles cases where UTLidar publishes in utlidar_lidar frame or body_link frame
            if (!transform_pointcloud(odom_cloud, msg->header.frame_id)) {
                // Skip if transform not available
                return;
            }

            *odom_cloud += *combined_cloud_;
            combined_cloud_->clear();

            // Create Occupancy Grid object
            cv::Mat raw_map = cv::Mat::zeros(IMAX, JMAX, CV_32F);
            for(const auto& pt : odom_cloud->points){
                const bool in_plane = (pt.z > minZ_) && (pt.z < maxZ_);
                if(!in_plane) continue;
                // Points are in body_link frame (robot-centered), no offset needed
                const float ic = pt.y / DS + (float)(IMAX/2);
                const float jc = pt.x / DS + (float)(JMAX/2);
                const bool in_grid = (ic > 0.0f) && (ic < (float)(IMAX-1)) && (jc > 0.0f) && (jc < (float)(JMAX-1));
                if(!in_grid) continue;
                raw_map.at<float>((int)std::round(ic),(int)std::round(jc)) = 1.0f;             
            }

            // BUILD MAP HERE
            for(int n=0; n<IMAX*JMAX; n++) confidence_values[n] = 0;
            Filtered_Occupancy_Convolution(confidence_values, raw_map, old_conf);
            memcpy(old_conf, confidence_values, IMAX*JMAX*sizeof(int8_t));

            // Publish Filtered Point Cloud
            pcl::toROSMsg(*odom_cloud, cloud_msg);
            cloud_msg.header.stamp = this->now();
            cloud_msg.header.frame_id = "body_link";
            cloud_pub_->publish(cloud_msg);

            // Publish Confidence Map
            for(int n=0; n<IMAX*JMAX; n++) map_msg.data[n] = confidence_values[n];
            map_msg.header.stamp = this->now();
            // body_link frame is robot-centered, origin is just offset by grid half-size
            map_msg.info.origin.position.x = -maxX;
            map_msg.info.origin.position.y = -maxY;
            map_pub_->publish(map_msg);
            
            // Throttle timing prints to ~1Hz (every 15 frames at 15Hz)
            static int timing_print_counter = 0;
            if(++timing_print_counter >= 15){
                timing_print_counter = 0;
                map_timer.time("Occ Map Solve Time: ");
            }

        }

        void lidar_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg){
            
            // Get robot pose from TF (odom -> body_link) for occupancy grid centering
            if (!update_pose_from_tf()) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Waiting for TF odom -> body_link, skipping lidar frame");
                return;
            }
            
            // Populate Point Cloud with LiDAR Points
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
            pcl::fromROSMsg(*msg, *cloud);

            // Transform from livox_frame to body_link using TF tree
            // TF chain: odom -> livox_frame -> body_link
            // We transform livox_frame directly to body_link
            if (!transform_pointcloud(cloud, msg->header.frame_id)) {
                // Skip if transform not available yet
                return;
            }

            // Mask Robot Body with Hyper-Ellipse (in body_link frame, robot is at origin)
            pcl::PointCloud<pcl::PointXYZI>::Ptr filtered (new pcl::PointCloud<pcl::PointXYZI>);
            for (const auto& pt : cloud->points){
                // In body_link frame, robot center is at origin
                float ellipse_norm = std::pow(pt.x/minX,8.0f) + std::pow(pt.y/minY,8.0f);
                if(ellipse_norm > 1.0f) filtered->points.push_back(pt);
            }
            filtered->width = filtered->points.size();
            filtered->height = 1;

            // Add Points into Combined Cloud (already in body_link frame)
            *combined_cloud_ += *filtered;

        } 

        /**
         * @brief Update robot pose (r, rpy) from TF lookup (odom -> body_link)
         * @return true if pose was successfully updated, false otherwise
         */
        bool update_pose_from_tf() {
            try {
                if (!tf_buffer_->canTransform("odom", "body_link", tf2::TimePointZero)) {
                    return false;
                }
                
                auto transform = tf_buffer_->lookupTransform(
                    "odom", "body_link", tf2::TimePointZero, tf2::durationFromSec(0.05)
                );
                
                // Extract position
                r[0] = transform.transform.translation.x;
                r[1] = transform.transform.translation.y;
                r[2] = transform.transform.translation.z;
                
                // Extract RPY from quaternion
                auto& q = transform.transform.rotation;
                // Roll (x-axis rotation)
                double sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z);
                double cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
                rpy[0] = std::atan2(sinr_cosp, cosr_cosp);
                
                // Pitch (y-axis rotation)
                double sinp = 2.0 * (q.w * q.y - q.z * q.x);
                if (std::abs(sinp) >= 1)
                    rpy[1] = std::copysign(M_PI / 2, sinp);  // use 90 degrees if out of range
                else
                    rpy[1] = std::asin(sinp);
                
                // Yaw (z-axis rotation)
                double siny_cosp = 2.0 * (q.w * q.z + q.x * q.y);
                double cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
                rpy[2] = std::atan2(siny_cosp, cosy_cosp);
                
                return true;
            } catch (tf2::TransformException& ex) {
                return false;
            }
        }

        // Transform point cloud using TF2 for dynamic gimbal
        bool transform_pointcloud(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud, 
                                 const std::string& source_frame) {
            // Skip if same frame
            if (source_frame == target_frame_) return true;
            
            // Check for empty frame
            if (source_frame.empty()) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "Source frame is empty, skipping transform");
                return false;
            }
            
            // Check if tf_buffer is ready
            if (!tf_buffer_) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                    "TF buffer not initialized yet");
                return false;
            }
            
            try {
                // Check if transform is available (non-blocking)
                if (!tf_buffer_->canTransform(target_frame_, source_frame, tf2::TimePointZero)) {
                    RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                        "Waiting for TF from %s to %s...", source_frame.c_str(), target_frame_.c_str());
                    return false;
                }
                
                geometry_msgs::msg::TransformStamped transform = tf_buffer_->lookupTransform(
                    target_frame_, source_frame, tf2::TimePointZero, tf2::durationFromSec(0.1)
                );
                
                Eigen::Affine3d eigen_transform = tf2::transformToEigen(transform.transform);
                pcl::transformPointCloud(*cloud, *cloud, eigen_transform.cast<float>());
                return true;
            } catch (tf2::TransformException& ex) {
                RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "TF transform failed from %s to %s: %s", 
                    source_frame.c_str(), target_frame_.c_str(), ex.what());
                return false;
            } catch (std::exception& ex) {
                RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Unexpected error in transform: %s", ex.what());
                return false;
            } catch (...) {
                RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                    "Unknown error in transform");
                return false;
            }
        }

        // Camera callback for RealSense D435 pointcloud (with gimbal)
        void camera_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg){
            
            // Convert ROS message to PCL - use PointXYZ since RealSense D435 doesn't have intensity
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_xyz (new pcl::PointCloud<pcl::PointXYZ>);
            pcl::fromROSMsg(*msg, *cloud_xyz);
            
            // Convert to PointXYZI with default intensity
            pcl::PointCloud<pcl::PointXYZI>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZI>);
            cloud->resize(cloud_xyz->size());
            for (size_t i = 0; i < cloud_xyz->size(); i++) {
                cloud->points[i].x = cloud_xyz->points[i].x;
                cloud->points[i].y = cloud_xyz->points[i].y;
                cloud->points[i].z = cloud_xyz->points[i].z;
                cloud->points[i].intensity = 1.0f;  // Default intensity
            }
            
            // Transform directly from camera_link to odom using TF2
            // This handles: camera gimbal rotation + robot body rotation + robot position
            if (!transform_pointcloud(cloud, msg->header.frame_id)) {
                return;  // Skip if transform fails
            }
            
            // Filter camera points by robot body mask (same as Livox)
            pcl::PointCloud<pcl::PointXYZI>::Ptr filtered (new pcl::PointCloud<pcl::PointXYZI>);
            for (const auto& pt : cloud->points){
                // In body_link frame, robot center is at origin
                float ellipse_norm = std::pow(pt.x/minX,8.0f) + std::pow(pt.y/minY,8.0f);
                if(ellipse_norm > 1.0f) filtered->points.push_back(pt);
            }
            filtered->width = filtered->points.size();
            filtered->height = 1;
            
            *combined_cloud_ += *filtered;
        }

        //  CREATE GAUSSIAN KERNEL
        cv::Mat gaussian_kernel(int kernel_size, float sigma){
            // Create kernel_sizexkernel_size array of floats
            cv::Mat kernel(kernel_size, kernel_size, CV_32F); 

            int half = kernel_size/2;
            // Iterate through each cell
            for(int i=-half; i<=half; i++){
                for (int j=-half; j<=half; j++){
                    float val = std::exp(-(i*i+j*j)/(2.0*sigma*sigma));
                    kernel.at<float>(i+half, j+half) = val;
                }
            }

            return kernel;
        }

        //  BUFFERED CONVOLUTION
        void Filtered_Occupancy_Convolution(int8_t *confidence_values, const cv::Mat& occupancy_data, const int8_t *old_conf_map){

            // In body_link frame, the grid always follows the robot, no egomotion shift needed
            // The robot is always at grid center

            for(int i = 0; i < IMAX; i++) {
                for(int j = 0; j < JMAX; j++){
                    // Copy previous confidence directly (no spatial shift)
                    confidence_values[i*JMAX+j] = old_conf_map[i*JMAX+j];
                }
            }

            // Apply Gaussian decay kernel to occupancy_data
            cv::filter2D(occupancy_data, buffered_binary, -1, gauss_kernel, cv::Point(-1,-1), 0, cv::BORDER_CONSTANT);
            
            // set parameters
            float sig, C, beta_up, beta_dn;
            const float thresh_front = 2.0f;  // Lower threshold for sparse front UTLidar
            const float thresh_mid360 = 4.0f; // Higher threshold for dense Livox Mid360
            bool front_flag = true;
            
            for(int i=0; i<IMAX; i++){
                for(int j=0; j<JMAX; j++){
                    
                    const float r2 = polar_coordinates_r2[i*JMAX+j];
                    const float th = polar_coordinates_th[i*JMAX+j];
                    const bool range_flag = r2 > 1.44f;
                    // In body_link frame, forward is always X+ (angle 0)
                    const bool angle_flag = std::abs(ang_diff(0.0f, th)) > 0.6f;
                    if(range_flag || angle_flag) front_flag = false;
                    else front_flag = true;
                    
                    // Use region-specific threshold
                    const float thresh = front_flag ? thresh_front : thresh_mid360;
                    
                    float val_binary = buffered_binary.at<float>(i,j);
                    float conf = (float)confidence_values[i*JMAX+j] / 127.0f;
                    if(val_binary > thresh){
                        if(front_flag) beta_up = 4.0f; //Go2 Front LiDAR only
                        else beta_up = 1.0f; // Livox Mid360
                        sig = 1.0f - std::exp(-beta_up*val_binary*dt);
                        C = 1.0f;
                    }
                    else{
                        if(front_flag) beta_dn = 4.0f;
                        else beta_dn = 4.0f;
                        sig = 1.0f - std::exp(-beta_dn*dt);
                        C = 0.0f;
                    }
                    conf *= 1.0f - sig;
                    conf += sig * C;
                    confidence_values[i*JMAX+j] = (int8_t)std::round(127.0f*conf);

                }
            }

        }

        // void removeGroundPlane(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input_cloud,
        //                pcl::PointCloud<pcl::PointXYZI>::Ptr& ground_removed_cloud)  {
        //     pcl::PointIndices::Ptr ground_candidate_indices(new pcl::PointIndices);
        //     pcl::PointCloud<pcl::PointXYZI>::Ptr ground_candidates(new pcl::PointCloud<pcl::PointXYZI>);
            
        //     for (size_t i = 0; i < input_cloud->points.size(); ++i) {
        //         const auto& pt = input_cloud->points[i];
        //         if (pt.z < minZ){
        //             ground_candidates->points.push_back(pt);
        //             ground_candidate_indices->indices.push_back(i);
        //         }
        //     }
        //     ground_candidates->width = ground_candidates->points.size();
        //     ground_candidates->height = 1;

        //     if (ground_candidates->empty()) {
        //         std::cout << "No ground candidates found under Z threshold." << std::endl;
        //         *ground_removed_cloud = *input_cloud;  // Return unmodified cloud
        //         return;
        //     }

        //     // Create the segmentation object
        //     pcl::SACSegmentation<pcl::PointXYZI> seg;            
        //     pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        //     pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);

        //     seg.setOptimizeCoefficients(true);
        //     seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
        //     seg.setAxis(Eigen::Vector3f(0.0, 0.0, 1.0));        // Prefer planes perpendicular to Z (i.e. horizontal)
        //     seg.setMethodType(pcl::SAC_RANSAC);
        //     seg.setDistanceThreshold(minZ);  // Adjust this threshold based on sensor noise
        //     seg.setInputCloud(ground_candidates);
        //     seg.segment(*inliers, *coefficients);


        //     pcl::PointIndices::Ptr full_cloud_inliers(new pcl::PointIndices);
        //     for (int idx : inliers->indices) {
        //         full_cloud_inliers->indices.push_back(ground_candidate_indices->indices[idx]);
        //     }

        //     // Extract non-ground (outlier) points
        //     pcl::ExtractIndices<pcl::PointXYZI> extract;
        //     extract.setInputCloud(input_cloud);
        //     extract.setIndices(full_cloud_inliers);
        //     extract.setNegative(true);  // True = remove inliers (i.e., remove the plane)
        //     extract.filter(*ground_removed_cloud);

        // }
    
            sensor_msgs::msg::PointCloud2 cloud_msg;
            nav_msgs::msg::OccupancyGrid map_msg;
            rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr livox_sub_;
            rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr utlidar_sub_;
            rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr camera_sub_;
            // Removed: robot_pose_sub_ - now using TF for robot pose
            
            rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
            rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr map_pub_;

            // TF2 for camera gimbal transforms
            std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
            std::shared_ptr<tf2_ros::TransformListener> tf_listener_;
            std::string target_frame_;

            pcl::PointCloud<pcl::PointXYZI>::Ptr combined_cloud_;
            
            // Configurable parameters
            float minZ_ = minZ_default;
            float maxZ_ = maxZ_default;
            
            std::vector<float> r = {0.0f, 0.0f, 0.0f};
            std::vector<float> r_map = {0.0f, 0.0f, 0.0f};
            std::vector<float> rpy = {0.0f, 0.0f, 0.0f};

            std::chrono::steady_clock::time_point t;
            float dt = 1.0e10f;

            // Generate gaussian kernel for convolution later
            const cv::Mat gauss_kernel = gaussian_kernel(9, 2.0);
            
            int8_t confidence_values[IMAX*JMAX];
            int8_t old_conf[IMAX*JMAX];
            float polar_coordinates_r2[IMAX*JMAX];
            float polar_coordinates_th[IMAX*JMAX];
            cv::Mat buffered_binary = cv::Mat::zeros(IMAX, JMAX, CV_32F);

}; 

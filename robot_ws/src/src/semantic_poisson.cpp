#include <memory>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <mutex>
#include <thread>
#include <random>
#include <cmath>
#include <queue>
#include <unistd.h>

#include <cuda_runtime.h>
#include "kernel.hpp"
#include "poisson.h"
#include "utils.h"
#include "mpc_cbf_3d.h"
#include "cloud_merger.h"
#include "poisson/human_tracker.h"
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_eigen/tf2_eigen.h>

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/int32.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "nav_msgs/msg/odometry.hpp"  // For /odom topic from odom_publisher
// Removed: unitree_go/msg/sport_mode_state.hpp - now using nav_msgs/Odometry
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <time.h>
#include "unitree_api/msg/request.hpp"
#include "common/ros2_sport_client.h"

class PoissonControllerNode : public rclcpp::Node{

    public:
        
        PoissonControllerNode() : Node("poisson_control"), sport_req(this){
            
            // Data logging parameter
            this->declare_parameter("enable_data_logging_to_file", false);
            enable_data_logging_to_file_ = this->get_parameter("enable_data_logging_to_file").as_bool();
            
            // Create CSV and BIN files for saving (only if logging enabled)
            if (enable_data_logging_to_file_) {
                std::string baseFileName = "experiment_data";
                std::string dateTime = getCurrentDateTime();
                std::string fileNameCSV = baseFileName + "_" + dateTime + ".csv";
                outFileCSV.open(fileNameCSV);
                const std::vector<std::string> header = {"t_ms", "space_counter", "rx", "ry", "yaw", 
                                                         "vx", "vy", "vyaw", "vxd", "vyd", "vyawd", 
                                                         "h", "dhdx", "dhdy", "dhdq", "dhdt", "alpha", "on_off"};
                for(char n = 0; n < header.size(); n++){
                    outFileCSV << header[n];
                    if(n!=(header.size()-1)) outFileCSV << ",";
                }
                outFileCSV << std::endl;
                
                std::string fileNameBIN = baseFileName + "_" + dateTime + ".bin";
                outFileBIN.open(fileNameBIN, std::ios::binary | std::ios::app);
                RCLCPP_INFO(this->get_logger(), "Data logging ENABLED: %s", fileNameCSV.c_str());
            } else {
                RCLCPP_INFO(this->get_logger(), "Data logging DISABLED");
            }

            // Initialize Parameter Deck
            gen.seed(rd());
            current_parameter_deck = sorted_parameter_deck;
            std::shuffle(current_parameter_deck.begin(),current_parameter_deck.end(), gen);

            // Initialize Clocks
            t_start = std::chrono::steady_clock::now();
            t_grid = std::chrono::steady_clock::now();
            t_state = std::chrono::steady_clock::now();

            // Declare and get ROS parameters for class-specific dh0 values
            this->declare_parameter("dh0_human", 1.0);
            this->declare_parameter("dh0_obstacle", 0.3);
            this->declare_parameter("enable_display", true);
            this->declare_parameter("enable_social_navigation", false);  // Pass on human's right (robot goes left)
            this->declare_parameter("social_tangent_bias", 0.5);  // Tangential bias strength when enabled
            this->declare_parameter("social_tangent_layers", 3);   // Number of layers outward from boundary with tangential bias
            this->declare_parameter("social_layer_thickness", 1);   // Grid cells per layer (circular dilation radius)
            this->declare_parameter("human_direction_threshold", 0.15);  // Closing velocity threshold to detect human approaching (m/s)
            this->declare_parameter("robot_mos_human", 0.5);    // Larger buffer for humans
            this->declare_parameter("robot_mos_obstacle", 0.1); // Smaller buffer for obstacles
            dh0_human = this->get_parameter("dh0_human").as_double();
            dh0_obstacle = this->get_parameter("dh0_obstacle").as_double();
            enable_display = this->get_parameter("enable_display").as_bool();
            enable_social_navigation_ = this->get_parameter("enable_social_navigation").as_bool();
            social_tangent_bias_ = this->get_parameter("social_tangent_bias").as_double();
            social_tangent_layers_ = this->get_parameter("social_tangent_layers").as_int();
            social_layer_thickness_ = this->get_parameter("social_layer_thickness").as_int();
            human_direction_threshold_ = this->get_parameter("human_direction_threshold").as_double();
            robot_MOS_human = this->get_parameter("robot_mos_human").as_double();
            robot_MOS_obstacle = this->get_parameter("robot_mos_obstacle").as_double();
            
            // Dynamic CBF parameters for dhdt scaling (eq. 31)
            this->declare_parameter("cbf_sigma_epsilon", 0.1);  // Sigma saturation value
            this->declare_parameter("cbf_sigma_kappa", 5.0);    // Sigma transition rate
            cbf_sigma_epsilon_ = this->get_parameter("cbf_sigma_epsilon").as_double();
            cbf_sigma_kappa_ = this->get_parameter("cbf_sigma_kappa").as_double();
            
            // Velocity bound parameters
            this->declare_parameter("vel_max_x_fwd", 0.9);
            this->declare_parameter("vel_max_x_bwd", 0.9);
            this->declare_parameter("vel_max_y", 0.9);
            this->declare_parameter("vel_max_yaw", 0.8);
            vel_max_x_fwd_ = this->get_parameter("vel_max_x_fwd").as_double();
            vel_max_x_bwd_ = this->get_parameter("vel_max_x_bwd").as_double();
            vel_max_y_ = this->get_parameter("vel_max_y").as_double();
            vel_max_yaw_ = this->get_parameter("vel_max_yaw").as_double();
            
            RCLCPP_INFO(this->get_logger(), "dh0_human=%.2f, dh0_obstacle=%.2f, MOS_human=%.2f, MOS_obstacle=%.2f, display=%s, social_nav=%s", 
                        dh0_human, dh0_obstacle, robot_MOS_human, robot_MOS_obstacle, 
                        enable_display ? "true" : "false", enable_social_navigation_ ? "true" : "false");
            RCLCPP_INFO(this->get_logger(), "Dynamic CBF: sigma_epsilon=%.3f, sigma_kappa=%.2f",
                        cbf_sigma_epsilon_, cbf_sigma_kappa_);
            RCLCPP_INFO(this->get_logger(), "Velocity bounds: x_fwd=%.2f, x_bwd=%.2f, y=%.2f, yaw=%.2f",
                        vel_max_x_fwd_, vel_max_x_bwd_, vel_max_y_, vel_max_yaw_);

            // Human tracker parameters
            this->declare_parameter("human_track_timeout_sec", 10.0);
            this->declare_parameter("human_track_gate_radius", 0.8);
            this->declare_parameter("human_track_velocity_decay_tau", 1.0);
            this->declare_parameter("human_track_velocity_threshold", 0.1);
            this->declare_parameter("min_yolo_cells", 5);
            this->declare_parameter("enable_human_tracker_dilation", true);
            this->declare_parameter("decay_in_fov", 0.7);
            this->declare_parameter("decay_stationary", 0.95);
            this->declare_parameter("decay_unconfirmed", 0.85);
            this->declare_parameter("no_retrack_on_move", true);
            float track_timeout = this->get_parameter("human_track_timeout_sec").as_double();
            float track_gate = this->get_parameter("human_track_gate_radius").as_double();
            float track_decay = this->get_parameter("human_track_velocity_decay_tau").as_double();
            float track_velocity_threshold = this->get_parameter("human_track_velocity_threshold").as_double();
            float decay_in_fov = this->get_parameter("decay_in_fov").as_double();
            float decay_stationary = this->get_parameter("decay_stationary").as_double();
            float decay_unconfirmed = this->get_parameter("decay_unconfirmed").as_double();
            bool no_retrack_on_move = this->get_parameter("no_retrack_on_move").as_bool();
            min_yolo_cells_ = this->get_parameter("min_yolo_cells").as_int();
            enable_human_tracker_dilation_ = this->get_parameter("enable_human_tracker_dilation").as_bool();
            human_tracker_ = std::make_unique<HumanTracker>(track_timeout, track_gate, track_decay, 
                track_velocity_threshold, decay_in_fov, decay_stationary, decay_unconfirmed, 3, 3, no_retrack_on_move);
            RCLCPP_INFO(this->get_logger(), "HumanTracker: timeout=%.1fs, gate=%.2fm, vel_thresh=%.2fm/s, decay_fov=%.2f, decay_stat=%.2f, decay_unconf=%.2f, no_retrack=%s",
                        track_timeout, track_gate, track_velocity_threshold, decay_in_fov, decay_stationary, decay_unconfirmed, no_retrack_on_move ? "true" : "false");

            // Tight-area wall softening parameters
            this->declare_parameter("tight_area_human_threshold", 2.0);  // meters to nearest human
            this->declare_parameter("tight_area_h_threshold", 0.3);      // h value threshold (proxy for wall distance)
            this->declare_parameter("tight_area_wall_slack", -0.1);      // slack added to wall boundaries when tight
            tight_area_human_threshold_ = this->get_parameter("tight_area_human_threshold").as_double();
            tight_area_h_threshold_ = this->get_parameter("tight_area_h_threshold").as_double();
            tight_area_wall_slack_ = this->get_parameter("tight_area_wall_slack").as_double();
            RCLCPP_INFO(this->get_logger(), "Tight-area params: human_thresh=%.2fm, h_thresh=%.2f, wall_slack=%.2f",
                        tight_area_human_threshold_, tight_area_h_threshold_, tight_area_wall_slack_);

            // Segmentation mask + pointcloud sync
            image_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
                this, "/yolo/segmentation_mask"
            );
            cloud_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(
                this, "/camera/point_cloud/cloud_registered"
            );
            // Initialize Occupancy Grids
            for(int n = 0; n < IMAX*JMAX; n++){
                occ1[n] = 1.0f;
                occ0[n] = 1.0f; 
                conf[n] = 0;
                grid_temp[n] = 0.0f;
                class_map[n] = 0;
                class_map_expanded[n] = 0;
                guidance_x_display[n] = 0.0f;
                guidance_y_display[n] = 0.0f;
                tangent_layer_display[n] = 0;
            }

            // Initialize Poisson Grids with CUDA Error Checking
            cudaError_t err;
            err = cudaMallocHost((void**)&hgrid1, IMAX*JMAX*QMAX*sizeof(float));
            if (err != cudaSuccess) {
                RCLCPP_ERROR(this->get_logger(), "CUDA allocation failed for hgrid1: %s", cudaGetErrorString(err));
                throw std::runtime_error("CUDA allocation failed");
            }
            err = cudaMallocHost((void**)&hgrid0, IMAX*JMAX*QMAX*sizeof(float));
            if (err != cudaSuccess) {
                RCLCPP_ERROR(this->get_logger(), "CUDA allocation failed for hgrid0: %s", cudaGetErrorString(err));
                throw std::runtime_error("CUDA allocation failed");
            }
            err = cudaMallocHost((void**)&bound, IMAX*JMAX*QMAX*sizeof(float));
            if (err != cudaSuccess) {
                RCLCPP_ERROR(this->get_logger(), "CUDA allocation failed for bound: %s", cudaGetErrorString(err));
                throw std::runtime_error("CUDA allocation failed");
            }
            err = cudaMallocHost((void**)&force, IMAX*JMAX*QMAX*sizeof(float));
            if (err != cudaSuccess) {
                RCLCPP_ERROR(this->get_logger(), "CUDA allocation failed for force: %s", cudaGetErrorString(err));
                throw std::runtime_error("CUDA allocation failed");
            }
            dhdt_grid = (float *)malloc(IMAX*JMAX*QMAX*sizeof(float));
            if (dhdt_grid == nullptr) {
                RCLCPP_ERROR(this->get_logger(), "Memory allocation failed for dhdt_grid");
                throw std::runtime_error("Memory allocation failed");
            }
            guidance_x_grid = (float *)malloc(IMAX*JMAX*QMAX*sizeof(float));
            guidance_y_grid = (float *)malloc(IMAX*JMAX*QMAX*sizeof(float));
            if (guidance_x_grid == nullptr || guidance_y_grid == nullptr) {
                RCLCPP_ERROR(this->get_logger(), "Memory allocation failed for guidance grids");
                throw std::runtime_error("Memory allocation failed");
            }
            for(int n=0; n < IMAX*JMAX*QMAX; n++){
                hgrid1[n] = h0;
                hgrid0[n] = h0;
                dhdt_grid[n] = 0.0f;
                guidance_x_grid[n] = 0.0f;
                guidance_y_grid[n] = 0.0f;
            }
            Kernel::poissonInit();
            robot_kernel_obstacle = nullptr;  // Initialize to nullptr before malloc
            robot_kernel_human = nullptr;
            robot_kernel_dim_obstacle = initialize_robot_kernel(robot_kernel_obstacle, robot_MOS_obstacle);
            robot_kernel_dim_human = initialize_robot_kernel(robot_kernel_human, robot_MOS_human);

            // Initialize QP for MPC Problem
            mpc3d_controller.set_velocity_bounds(vel_max_x_fwd_, vel_max_x_bwd_, vel_max_y_, vel_max_yaw_);
            mpc3d_controller.setup_QP();
            mpc3d_controller.solve();

            // Create Publishers & Subscribers
            rclcpp::SubscriptionOptions options1;
            rclcpp::SubscriptionOptions options2;
            rclcpp::SubscriptionOptions options3;
            rclcpp::SubscriptionOptions options_yolo;  // Separate group for YOLO data (class_map, visibility)
            options1.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            options2.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            options3.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            options_yolo.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            occ_grid_suber_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>("occupancy_grid", 1, std::bind(&PoissonControllerNode::occ_grid_callback, this, std::placeholders::_1), options1);
            class_map_suber_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>("class_map", 1, std::bind(&PoissonControllerNode::class_map_callback, this, std::placeholders::_1), options_yolo);
            visibility_map_suber_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>("visibility_map", 1, std::bind(&PoissonControllerNode::visibility_map_callback, this, std::placeholders::_1), options_yolo);
            pose_suber_ = this->create_subscription<nav_msgs::msg::Odometry>("/odom", 1, std::bind(&PoissonControllerNode::state_update_callback, this, std::placeholders::_1), options2);
            twist_suber_ = this->create_subscription<geometry_msgs::msg::Twist>("u_des", 1, std::bind(&PoissonControllerNode::teleop_callback, this, std::placeholders::_1), options3);
            key_suber_ = this->create_subscription<std_msgs::msg::Int32>("key_press", 1, std::bind(&PoissonControllerNode::keyboard_callback, this, std::placeholders::_1), options3);

            // Image publish rate parameter
            this->declare_parameter("logging_publish_hz", 10.0);
            logging_publish_hz_ = this->get_parameter("logging_publish_hz").as_double();
            logging_publish_period_ = (logging_publish_hz_ > 0) ? (1.0 / logging_publish_hz_) : 0.0;
            last_logging_publish_time_ = std::chrono::steady_clock::now();
            RCLCPP_INFO(this->get_logger(), "Logging publish rate: %.1f Hz", logging_publish_hz_);
            
            // Publisher for Poisson visualization image
            poisson_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/poisson/visualization", 10);
            logging_data_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/poisson/logging_data", 10);

            mpc_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
            mpc_timer_ = this->create_wall_timer(std::chrono::milliseconds(10), std::bind(&PoissonControllerNode::mpc_callback, this), mpc_callback_group_);

            // Start Up the Unitree Go2
            sport_req.RecoveryStand(req);
            sleep(1);
            sport_req.SpeedLevel(req, 1);
            sleep(1);

        }

    private:

        void teleop_callback(geometry_msgs::msg::Twist::UniquePtr msg){
                        
            // Teleop Velocity Command
            const std::vector<float> vtb = {(float)msg->linear.x, (float)msg->linear.y, (float)msg->angular.z};
            vt = {std::cos(x[2])*vtb[0] - std::sin(x[2])*vtb[1],
                  std::sin(x[2])*vtb[0] + std::cos(x[2])*vtb[1],
                  vtb[2]};

            // Move Goal Point
            xd[0] += 0.01f * vt[0];
            xd[1] += 0.01f * vt[1];
            xd[2] += 0.01f * vt[2];
            
            // Re-initialize If Not Started
            if(!start_flag){
                xd = x;
                vt = {0.0f, 0.0f, 0.0f};
            }

        };

        void keyboard_callback(std_msgs::msg::Int32::UniquePtr msg){
                        
            // Record Time, ReSetting Until After the First Save 
            if(!save_flag) t_start = std::chrono::steady_clock::now();
            else t_ms = std::chrono::duration<float>(std::chrono::steady_clock::now() - t_start).count() * 1.0e3f;
            
            // Check for Flags
            char param = ' ';
            int ch = msg->data;
            switch(ch){
                case ' ':
                    space_counter++;
                    if(space_counter>=1) save_flag = true;
                    if(space_counter>=3) start_flag = true;
                    if(space_counter>=6) stop_flag = true;
                    break;
                case 'r':
                    realtime_sf_flag = !realtime_sf_flag;
                    printf("[KEY] 'r' pressed: realtime_sf_flag = %s\n", realtime_sf_flag ? "ON" : "OFF");
                    break;
                case 'p':
                    predictive_sf_flag = !predictive_sf_flag;
                    printf("[KEY] 'p' pressed: predictive_sf_flag = %s\n", predictive_sf_flag ? "ON" : "OFF");
                    break;
                case 'd':
                    param = current_parameter_deck.back();
                    current_parameter_deck.pop_back();
                    if(current_parameter_deck.size()==0){
                        current_parameter_deck = sorted_parameter_deck;
                        std::shuffle(current_parameter_deck.begin(), current_parameter_deck.end(), gen);
                        //sport_req.Stretch(req);
                    }
                    //else sport_req.Hello(req);
                    break;
                default:
                    break;
            }

            // Deal a Parameter
            switch(param){
                case '0':
                    predictive_sf_flag = false;
                    realtime_sf_flag = false;
                    wn = 16.0f;
                    break;
                case '1':
                    predictive_sf_flag = true;
                    realtime_sf_flag = true;
                    wn = 0.5f;
                    break;
                case '2':
                    predictive_sf_flag = true;
                    realtime_sf_flag = true;
                    wn = 1.0f;
                    break;
                case '3':
                    predictive_sf_flag = true;
                    realtime_sf_flag = true;    
                    wn = 1.5f;
                    break;
                case '4':
                    predictive_sf_flag = true;
                    realtime_sf_flag = true;
                    wn = 2.0f;
                    break;
                case '5':
                    predictive_sf_flag = true;
                    realtime_sf_flag = true;
                    wn = 4.0f;
                    break;
                case '6':
                    predictive_sf_flag = true;
                    realtime_sf_flag = true;
                    wn = 8.0f;
                    break;
                default:
                    break;
            }

            switch(ch){
                case '1':
                    wn = 0.5f;
                    break;
                case '2':
                    wn = 1.0f;
                    break;
                case '3':
                    wn = 1.5f;
                    break;
                case '4':
                    wn = 2.0f;
                    break;
                case '5':
                    wn = 4.0f;
                    break;
                case '6':
                    wn = 8.0f;
                    break;
                default:
                    break;
            }

            // Save Data to file (only if logging enabled)
            if(save_flag && enable_data_logging_to_file_){

                // Log Data to CSV
                const std::vector<float> save_data = {t_ms, (float)space_counter, x[0], x[1], x[2], 
                                                      v[0], v[1], v[2], vt[0], vt[1], vt[2], 
                                                      h, dhdx, dhdy, dhdq, dhdt, wn, (float)(realtime_sf_flag|predictive_sf_flag)};
                for(char n = 0; n < save_data.size(); n++){
                    outFileCSV << save_data[n];
                    if(n!=(save_data.size()-1)) outFileCSV << ",";
                }
                outFileCSV << std::endl;
                
                // Every nth Log, Log the Poisson Surface to Binary File 
                const int factor = 7;
                if(!(poisson_save_counter%factor)) outFileBIN.write(reinterpret_cast<char*>(grid_temp), sizeof(grid_temp));
                poisson_save_counter++;
            }

        };
        
        void mpc_callback(void){

            //Timer mpc_timer(true);
            //mpc_timer.start();

            // Run MPC with SQP Loops until Cost is Stable
            if(predictive_sf_flag && h_flag && mpc_mutex.try_lock()){
                std::lock_guard<std::mutex> lock(mpc_mutex, std::adopt_lock);
                // In body_link frame, robot is at origin (0, 0) with only yaw from odometry
                std::vector<float> x_body_link = {0.0f, 0.0f, x[2]};
                for(int i=0; i<MAX_SQP_ITERS; i++){
                    // Transform nominal velocity to Body Frame for MPC
                    const float c = std::cos(x[2]);
                    const float s = std::sin(x[2]);
                    std::vector<float> vn_body = {c*vn[0] + s*vn[1], -s*vn[0] + c*vn[1], vn[2]};
                    
                    mpc3d_controller.update_cost(vn_body);
                    mpc3d_controller.update_constraints(hgrid1, dhdt_grid, guidance_x_grid, guidance_y_grid, x_body_link, xc, grid_age, wn, issf, cbf_sigma_epsilon_, cbf_sigma_kappa_);
                    mpc3d_controller.solve();
                    if(mpc3d_controller.update_residual()<1.0f) break;
                }
                // MPC output 'sol' involves body frame states, so 'set_input' sets 'vd' to body frame velocity
                mpc3d_controller.set_input(vd);
            }

            //mpc_timer.time("MPC Solve Time: ");

        };
    
        /* Nominal Single Integrator Proportional Tracker */
        void nominal_controller(void){

            const float kp = 0.5f;
            vn[0] += std::clamp(kp*(xd[0]-x[0]), -1.0f, 1.0f);
            vn[1] += std::clamp(kp*(xd[1]-x[1]), -1.0f, 1.0f);
            vn[2] += std::clamp(kp*ang_diff(xd[2],x[2]), -1.0f, 1.0f);

        };

        /* Threshold Occupancy Map with Hysterisis */
        void build_occ_map(float *occ_map, const float *occ_map_old, const int8_t *conf){
            
            const int8_t T_hi = 85;
            const int8_t T_lo = 64;
                        
            for(int i=0; i<IMAX; i++){
                for(int j=0; j<JMAX; j++){
                    const int i0 = i + (int)std::round(dx[1] / DS);
                    const int j0 = j + (int)std::round(dx[0] / DS);
                    const bool in_grid = (i0 >= 0) && (i0 < IMAX) && (j0 >= 0) && (j0 < JMAX);
                    const bool strong = conf[i*JMAX+j] >= T_hi;
                    const bool weak = conf[i*JMAX+j] >= T_lo;
                    if(strong) occ_map[i*JMAX+j] = -1.0f;
                    else if(weak && in_grid) occ_map[i*JMAX+j] = occ_map_old[i0*JMAX+j0];
                    else occ_map[i*JMAX+j] = 1.0f;
                }
            }
 
        };

        /* Detect if robot is in a tight area (close to both human and wall) */
        bool is_tight_area() {
            // Check for nearby tracked humans
            auto tracks = human_tracker_->get_active_tracks();
            if (tracks.empty()) return false;
            
            // Distance to nearest tracked human
            float min_human_dist = FLT_MAX;
            for (const auto& track : tracks) {
                float d = std::sqrt(std::pow(track.x - x[0], 2) + std::pow(track.y - x[1], 2));
                min_human_dist = std::min(min_human_dist, d);
            }
            
            // Use h value at robot position as proxy for wall distance
            // (h is small when close to obstacles)
            // In body_link frame, robot is always at origin (0, 0)
            const float ic = y_to_i(0.0f, xc[1]);
            const float jc = x_to_j(0.0f, xc[0]);
            const float qc = yaw_to_q(x[2], xc[2]);
            
            // Clamp indices to valid range
            const float ic_clamped = std::clamp(ic, 0.0f, (float)(IMAX-1));
            const float jc_clamped = std::clamp(jc, 0.0f, (float)(JMAX-1));
            
            float h_at_robot = trilinear_interpolation(hgrid1, ic_clamped, jc_clamped, qc);
            
            // Tight area if: human is close AND robot is close to some obstacle (low h)
            bool is_tight = (min_human_dist < tight_area_human_threshold_) && 
                           (h_at_robot < tight_area_h_threshold_);
            
            // Debug output (~1Hz)
            static int tight_area_debug_counter = 0;
            if(++tight_area_debug_counter >= 15){
                tight_area_debug_counter = 0;
                // printf("[TIGHT_AREA] human_dist=%.2fm (thresh=%.2f), h=%.3f (thresh=%.2f), tight=%s\n",
                //        min_human_dist, tight_area_human_threshold_,
                //        h_at_robot, tight_area_h_threshold_,
                //        is_tight ? "YES" : "no");
            }
            
            return is_tight;
        }

        /* Find Boundaries (Any Unoccupied Point that Borders an Occupied Point) 
         * If tight_area is true and class_map is provided, wall cells get h0 + slack */
        void find_boundary(float *grid, float *bound, const bool fix_flag, 
                          const bool tight_area = false, const int8_t* class_map = nullptr){
            
            // Set Border
            for(int i = 0; i < IMAX; i++){
                for(int j = 0; j < JMAX; j++){
                    if(i==0 || i==(IMAX-1) || j==0 || j==(JMAX-1)) bound[i*JMAX+j] = 0.0f;
                }
            }

            float b0[IMAX*JMAX];
            memcpy(b0, bound, IMAX*JMAX*sizeof(float));
            for(int n = 0; n < IMAX*JMAX; n++){
                if(b0[n]==1.0f){
                    if(b0[n+1]==-1.0f || 
                       b0[n-1]==-1.0f || 
                       b0[n+JMAX]==-1.0f || 
                       b0[n-JMAX]==-1.0f || 
                       b0[n+JMAX+1]==-1.0f || 
                       b0[n-JMAX+1]==-1.0f || 
                       b0[n+JMAX-1]==-1.0f || 
                       b0[n-JMAX-1]==-1.0f) bound[n] = 0.0f;
                }
                if(fix_flag && !bound[n]) {
                    // Boundary cell: determine if wall or human
                    bool is_wall = true;
                    if (class_map) {
                        is_wall = (class_map[n] != 1);  // not human
                    }
                    
                    // Apply slack to walls only when in tight area
                    if (tight_area && is_wall) {
                        grid[n] = h0 + tight_area_wall_slack_;  // e.g., 0.0 + (-0.1) = -0.1
                    } else {
                        grid[n] = h0;  // 0.0 for human or non-tight areas
                    }
                }
            }
        };
        
        /* Construct n x n Kernel Using Hyper-Ellipse Parameters 
         * mos: margin of safety multiplier (e.g., 0.1 for obstacles, 0.5 for humans) */
        int initialize_robot_kernel(float*& kernel, float mos){
            
            /* Create Robot Kernel with specified margin of safety */
            robot_length = 0.7f; // Go2
            robot_width = 0.3f;
            
            const float ar = mos * robot_length / 2.0f;
            const float br = mos * robot_width / 2.0f;
            const float D = 2.0f * std::sqrt(ar*ar + br*br); // Max Robot Dimension to Define Kernel Size
            int dim = 2 * (int)std::ceil(std::ceil(D / DS) / 2.0f); //Make Sure Kernel Dimension is Even
            if (dim < 2) dim = 2;  // Minimum kernel size

            kernel = (float *)malloc(dim*dim*QMAX*sizeof(float));
            for(int q=0; q<QMAX; q++){
                float *kernel_slice = kernel + q*dim*dim;
                const float yawq = q_to_yaw(q, xc[2]);
                fill_elliptical_robot_kernel(kernel_slice, yawq, dim, 2.0f, mos);
            }

            return dim;

        };

        /* Construct n x n Kernel Using Hyper-Ellipse Parameters */
        void fill_elliptical_robot_kernel(float *kernel, const float yawq, const int dim, const float expo, float mos){
            
            const float ar = mos * robot_length / 2.0f;
            const float br = mos * robot_width / 2.0f;
            // Handle zero MOS (no inflation)
            if (ar < 0.001f || br < 0.001f) {
                for(int i = 0; i < dim*dim; i++) kernel[i] = 0.0f;
                return;
            }
            for(int i = 0; i < dim; i++){
                const float yi = (float)(i-dim/2)*DS;
                for(int j = 0; j < dim; j++){
                    kernel[i*dim+j] = 0.0f;
                    const float xi = (float)(j-dim/2)*DS;
                    const float xb = std::cos(yawq)*xi + std::sin(yawq)*yi;
                    const float yb = -std::sin(yawq)*xi + std::cos(yawq)*yi;
                    const float dist = std::pow(std::abs(xb/ar), expo) + std::pow(std::abs(yb/br), expo);
                    if(dist <= 1.0f) kernel[i*dim+j] = -1.0f;
                }
            }

        };


        /* Buffer Occupancy Grid with 2-D Robot Shape 
         * Uses class-aware kernel selection: human boundaries get larger kernel
         * Also propagates class labels during inflation */
        void inflate_occupancy_grid(float *bound, int8_t *class_map = nullptr){
            
            // Convolve Robot Kernel with Occupancy Grid, Along the Boundary
            // Uses class-aware kernel selection: human boundaries get larger kernel
            float b0[IMAX*JMAX];
            memcpy(b0, bound, IMAX*JMAX*sizeof(float));
            
            // Also copy class map if provided
            int8_t c0[IMAX*JMAX];
            if (class_map) {
                memcpy(c0, class_map, IMAX*JMAX*sizeof(int8_t));
            }

            for(int i = 1; i < IMAX-1; i++){
                for(int j = 1; j < JMAX-1; j++){
                    if(!b0[i*JMAX+j]){  // boundary cell
                        // Get the class label at this boundary (from adjacent occupied cells)
                        int8_t source_class = 0;
                        if (class_map) {
                            // Check if any adjacent occupied cell is human-labeled
                            for(int di = -1; di <= 1 && source_class == 0; di++){
                                for(int dj = -1; dj <= 1 && source_class == 0; dj++){
                                    int ni = i + di;
                                    int nj = j + dj;
                                    if(ni >= 0 && ni < IMAX && nj >= 0 && nj < JMAX){
                                        if(b0[ni*JMAX+nj] < 0.0f && c0[ni*JMAX+nj] == 1){
                                            source_class = 1;  // inherit human label
                                        }
                                    }
                                }
                            }
                        }
                        
                        // Select kernel based on source class
                        const float* kernel = (source_class == 1) ? robot_kernel_human : robot_kernel_obstacle;
                        int kernel_dim = (source_class == 1) ? robot_kernel_dim_human : robot_kernel_dim_obstacle;
                        int lim = (kernel_dim - 1) / 2;
                        
                        int ilow = std::max(i - lim, 0);
                        int itop = std::min(i + lim, IMAX);
                        int jlow = std::max(j - lim, 0);
                        int jtop = std::min(j + lim, JMAX);
                        
                        for(int p = ilow; p < itop; p++){
                            for(int q = jlow; q < jtop; q++){
                                float kernel_val = kernel[(p-i+lim)*kernel_dim+(q-j+lim)];
                                bound[p*JMAX+q] += kernel_val;
                                // Propagate human label to inflated cells
                                if(class_map && kernel_val < 0.0f && source_class == 1){
                                    class_map[p*JMAX+q] = 1;
                                }
                            }
                        }
                    }
                }
            }
            for(int n = 0; n < IMAX*JMAX; n++){
                if(bound[n] < -1.0f) bound[n] = -1.0f;
            }

        };

        /* Compute Forcing Function for Average Flux */
        void compute_fast_forcing_function(float *force, const float *bound){

            float perimeter_c = 0.0f;
            float area_c = 0.0f;
            
            for(int i = 1; i < IMAX-1; i++){
                for(int j = 1; j < JMAX-1; j++){
                    if(bound[i*JMAX+j] == 0.0f) perimeter_c += DS;
                    else if(bound[i*JMAX+j] < 0.0f) area_c += DS*DS;
                }
            }
            
            float perimeter_o = 2.0f*(float)IMAX*DS + 2.0f*(float)JMAX*DS + perimeter_c;
            float area_o = (float)IMAX*(float)JMAX*DS*DS - area_c;
            float force_o = -dh0 * perimeter_o / area_o * DS*DS;
            float force_c = 0.0f;
            if(area_c != 0.0f) force_c = dh0 * perimeter_c / area_c * DS*DS;
            
            for(int n = 0; n < IMAX*JMAX; n++){
                if(bound[n] > 0.0f){
                    force[n] = force_o;
                }
                else if(bound[n] < 0.0f){
                    force[n] = force_c;
                }
                else{
                    force[n] = 0.0f;
                }
            }
        
        };


        /* Compute Boundary Gradients for Guidance Field
         * rx, ry: robot position in world frame
         * vn_x, vn_y: robot velocity in world frame (for velocity-aware social nav)
         * 
         * Multi-layer tangential bias approach:
         * - Layer 0 (boundary, bound==0): NORMAL gradients only (preserves CBF guarantees)
         * - Layers 1 to N: Apply constant tangential bias (configurable via social_tangent_layers_)
         */
        void compute_boundary_gradients(float *guidance_x, float *guidance_y, float *bound, 
                                       const int8_t* class_map = nullptr,
                                       float /*rx*/ = 0.0f, float /*ry*/ = 0.0f,
                                       float /*vn_x*/ = 0.0f, float /*vn_y*/ = 0.0f,
                                       bool populate_human_info = false){
            // Set Border Gradients
            for(int i = 0; i < IMAX; i++){
                for(int j = 0; j < JMAX; j++){
                    if(i==0) guidance_x[i*JMAX+j] = dh0;
                    if(j==0) guidance_y[i*JMAX+j] = dh0;
                    if(i==(IMAX-1)) guidance_x[i*JMAX+j] = -dh0;
                    if(j==(JMAX-1)) guidance_y[i*JMAX+j] = -dh0;
                }
            }
            
            // Track which cells are human boundaries and their normal directions
            // human_boundary_info_ stores (i, j, gx, gy, local_dh0) at each human boundary cell
            // Only clear/populate when flag is set (to avoid race condition in parallel loop)
            if (populate_human_info) {
                human_boundary_info_.clear();  // Clear from previous frame
            }
            
            // Set Additional Boundary Gradients (Layer 0 - normal only, no tangential bias)
            for(int i = 1; i < IMAX-1; i++){
                for(int j = 1; j < JMAX-1; j++){
                    if(!bound[i*JMAX+j]){ // bound value is 0 so it's checking for existing boundary
                        guidance_x[i*JMAX+j] = 0.0f;
                        guidance_y[i*JMAX+j] = 0.0f;
                        for(int p = -1; p <= 1; p++){
                            for(int q = -1; q <= 1; q++){
                                if(q > 0){
                                    guidance_x[i*JMAX+j] += bound[(i+q)*JMAX+(j+p)];
                                    guidance_y[i*JMAX+j] += bound[(i+p)*JMAX+(j+q)];
                                }
                                else if (q < 0){
                                    guidance_x[i*JMAX+j] -= bound[(i+q)*JMAX+(j+p)];
                                    guidance_y[i*JMAX+j] -= bound[(i+p)*JMAX+(j+q)];
                                }
                            }
                        }
                    }
                }
            }
            for(int i = 0; i < IMAX; i++){
                for(int j = 0; j < JMAX; j++){
                    if(!bound[i*JMAX+j]){  // Boundary cell (Layer 0)
                        const float V = std::sqrt(guidance_x[i*JMAX+j]*guidance_x[i*JMAX+j] + guidance_y[i*JMAX+j]*guidance_y[i*JMAX+j]);
                        if(V != 0.0f){
                            guidance_x[i*JMAX+j] /= V;
                            guidance_y[i*JMAX+j] /= V;
                        }
                        float local_dh0 = dh0_obstacle;  // Default for obstacles
                        bool is_human = false;
                        if (class_map) {
                            // Boundary cells (bound==0) are FREE space adjacent to obstacles
                            // Human labels are now on INFLATED occupied cells (bound<0)
                            // Just check immediate neighbors since class_map is inflated with bound
                            for(int di = -1; di <= 1 && !is_human; di++){
                                for(int dj = -1; dj <= 1 && !is_human; dj++){
                                    int ni = i + di;
                                    int nj = j + dj;
                                    if(ni >= 0 && ni < IMAX && nj >= 0 && nj < JMAX){
                                        // Check if neighbor is occupied AND has human label
                                        if(bound[ni*JMAX+nj] < 0.0f && class_map[ni*JMAX+nj] == 1){
                                            is_human = true;
                                        }
                                    }
                                }
                            }
                            if(is_human){
                                local_dh0 = dh0_human;
                                // Store human boundary info for post-solve tangent application
                                // Only when populate_human_info flag is set (to avoid race in parallel)
                                if (populate_human_info) {
                                    human_boundary_info_.emplace_back(i, j, guidance_x[i*JMAX+j], guidance_y[i*JMAX+j], local_dh0);
                                }
                            }
                        }
                        
                        // Keep normal at boundary for CBF mathematical rigor
                        // Tangent bias will be applied AFTER Laplace solve as post-processing
                        guidance_x[i*JMAX+j] *= local_dh0;
                        guidance_y[i*JMAX+j] *= local_dh0;
                    }
                }
            }
            
            // NOTE: Tangent bias is now applied BEFORE Laplace solve (pre-processing)
            // using circular dilation to create layered Dirichlet BCs
        };
        
        /* Compute tangent direction based on human heading relative to robot
         * Returns +1 for CW (human's back to robot) or -1 for CCW (human facing robot)
         */
        float compute_tangent_direction(const std::vector<HumanTrack>& active_tracks, float rx, float ry, float vn_x, float vn_y) {
            float target_sign = -1.0f;  // Default: Visual Clockwise (CW, Pass on Left)
            // Note: Due to cv::flip, +1.0f internal = CCW visual, -1.0f internal = CW visual
            
            for (const auto& track : active_tracks) {
                // DISTANCE-BASED APPROACH: Measure rate of change of distance per track
                // Current distance from robot to human
                float current_distance = std::sqrt(track.x * track.x + track.y * track.y);
                
                // Get current time
                float current_time = std::chrono::duration<float>(
                    std::chrono::steady_clock::now() - t_start).count();
                
                // Compute closing rate for this specific track
                float closing_rate = 0.0f;
                auto it = prev_human_distances_.find(track.id);
                if (it != prev_human_distances_.end()) {
                    float prev_distance = it->second.first;
                    float prev_time = it->second.second;
                    float dt = current_time - prev_time;
                    if (dt > 0.01f && dt < 1.0f) {
                        // closing_rate > 0 means getting closer
                        closing_rate = (prev_distance - current_distance) / dt;
                    }
                }
                // Update stored distance and time for this track
                prev_human_distances_[track.id] = {current_distance, current_time};
                
                // If ANY human is closing faster than threshold → CCW (most aggressive approach)
                if (closing_rate > human_direction_threshold_) {
                    target_sign = 1.0f;  // Visual CCW (Pass on Right)
                }
                
                // Debug output (for first track only to avoid spam)
                static int tangent_debug_counter = 0;
                if (++tangent_debug_counter >= 15) {
                    printf("[TANGENT DEBUG] track#%d dist=%.2f closing_rate=%.2f thresh=%.2f sign=%.1f\n",
                           track.id, current_distance, closing_rate, human_direction_threshold_, target_sign);
                    tangent_debug_counter = 0;
                }
                // Continue to check ALL tracks (no break - multi-human support)
            }
            
            // Store for debug display
            current_tangent_direction_ = target_sign;
            
            return target_sign;
        };
        
        /* Expand human obstacles using circular kernel dilation and apply tangential bias
         * Creates Dirichlet BCs for the guidance field Laplace solve
         * 
         * @param bound_guidance    Output: bound array for guidance solve (0 = BC, >0 = free)
         * @param guidance_x        In/Out: x-component of guidance (layers get biased gradients)
         * @param guidance_y        In/Out: y-component of guidance (layers get biased gradients)
         * @param bound_original    Input: original bound array (true obstacle boundaries)
         * @param num_layers        Number of layers to expand
         * @param layer_thickness   Grid cells per layer (circular dilation radius)
         * @param bias_strength     Tangential bias strength
         * @param sign              Sign for CCW (+1) or CW (-1) rotation
         */
        void expand_human_obstacles_for_guidance(
            float* bound_guidance,
            float* guidance_x,
            float* guidance_y,
            const float* bound_original,
            int num_layers,
            int layer_thickness,
            float bias_strength,
            float sign) {
            
            // Copy original bounds
            memcpy(bound_guidance, bound_original, IMAX*JMAX*sizeof(float));
            
            // Track current occupied region (initialized from human boundaries)
            std::vector<bool> current_occupied(IMAX*JMAX, false);
            std::vector<bool> is_human_region(IMAX*JMAX, false);
            
            // Mark initial human boundary cells (Layer 0)
            for (const auto& info : human_boundary_info_) {
                int i = std::get<0>(info);
                int j = std::get<1>(info);
                current_occupied[i*JMAX + j] = true;
                is_human_region[i*JMAX + j] = true;
                // Also mark the occupied cells behind the boundary
                for (int di = -1; di <= 1; di++) {
                    for (int dj = -1; dj <= 1; dj++) {
                        int ni = i + di;
                        int nj = j + dj;
                        if (ni >= 0 && ni < IMAX && nj >= 0 && nj < JMAX) {
                            if (bound_original[ni*JMAX + nj] < 0.0f) {
                                current_occupied[ni*JMAX + nj] = true;
                                is_human_region[ni*JMAX + nj] = true;
                            }
                        }
                    }
                }
            }
            
            // For each layer, dilate and apply tangent bias
            for (int layer = 1; layer <= num_layers; layer++) {
                std::vector<bool> next_occupied = current_occupied;
                std::vector<std::tuple<int, int, float, float>> new_layer_cells;  // (i, j, gx, gy)
                
                // Circular dilation with configurable radius
                const int r = layer_thickness;
                
                // Find cells that will be in this layer
                for (int i = r; i < IMAX-r; i++) {
                    for (int j = r; j < JMAX-r; j++) {
                        // Skip if already occupied
                        if (current_occupied[i*JMAX + j]) continue;
                        
                        // Check if within radius of any current human region cell
                        bool in_human_range = false;
                        float avg_gx = 0.0f, avg_gy = 0.0f;
                        int count = 0;
                        
                        for (int di = -r; di <= r && !in_human_range; di++) {
                            for (int dj = -r; dj <= r && !in_human_range; dj++) {
                                if (di*di + dj*dj <= r*r) {  // Circular kernel
                                    int ni = i + di;
                                    int nj = j + dj;
                                    if (is_human_region[ni*JMAX + nj] && current_occupied[ni*JMAX + nj]) {
                                        in_human_range = true;
                                        // Accumulate gradient direction from nearby boundary cells
                                        for (const auto& info : human_boundary_info_) {
                                            int bi = std::get<0>(info);
                                            int bj = std::get<1>(info);
                                            int dist_sq = (bi-i)*(bi-i) + (bj-j)*(bj-j);
                                            if (dist_sq <= (layer * r + r) * (layer * r + r)) {
                                                avg_gx += std::get<2>(info);  // Normal direction
                                                avg_gy += std::get<3>(info);
                                                count++;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        
                        if (in_human_range && bound_original[i*JMAX + j] > 0.0f) {  // Must be free space
                            // Compute gradient direction (average of nearby human boundary normals)
                            if (count > 0) {
                                avg_gx /= count;
                                avg_gy /= count;
                            } else {
                                // Fallback: use existing guidance direction
                                avg_gx = guidance_x[i*JMAX + j];
                                avg_gy = guidance_y[i*JMAX + j];
                            }
                            
                            float mag = std::sqrt(avg_gx*avg_gx + avg_gy*avg_gy);
                            if (mag > 0.01f) {
                                avg_gx /= mag;
                                avg_gy /= mag;
                                new_layer_cells.emplace_back(i, j, avg_gx, avg_gy);
                            }
                            
                            next_occupied[i*JMAX + j] = true;
                            is_human_region[i*JMAX + j] = true;
                        }
                    }
                }
                
                // Apply tangent bias to new layer cells
                float ramp = (float)layer / (float)num_layers;  // 0% → 100% as layer increases
                
                for (const auto& cell : new_layer_cells) {
                    int i = std::get<0>(cell);
                    int j = std::get<1>(cell);
                    float gx = std::get<2>(cell);
                    float gy = std::get<3>(cell);
                    
                    // Apply tangential bias: rotate normal by ramp * bias_strength
                    // Tangent = perpendicular to normal: (gy, -gx) for CW (sign=+1), (-gy, gx) for CCW (sign=-1)
                    float biased_gx = gx + sign * bias_strength * ramp * gy;
                    float biased_gy = gy + sign * bias_strength * ramp * (-gx);
                    
                    // Normalize and scale by dh0_human (layers inherit human dh0)
                    float V = std::sqrt(biased_gx*biased_gx + biased_gy*biased_gy);
                    if (V > 0.0f) {
                        guidance_x[i*JMAX + j] = biased_gx / V * dh0_human;
                        guidance_y[i*JMAX + j] = biased_gy / V * dh0_human;
                    }
                    
                    // Mark as Dirichlet BC for Laplace solve
                    bound_guidance[i*JMAX + j] = 0.0f;
                    
                    // Mark for visualization (use layer number)
                    tangent_layer_display[i*JMAX + j] = layer;
                }
                
                current_occupied = next_occupied;
            }
        };

        /* Compute Forcing Function from Guidance Field */
        void compute_optimal_forcing_function(float *force, const float *guidance_x, const float *guidance_y, const float *bound){
            const float max_div = 10.0f;
            for(int i = 1; i < (IMAX-1); i++){
                for(int j = 1; j < (JMAX-1); j++){
                    force[i*JMAX+j] = (guidance_x[(i+1)*JMAX+j] - guidance_x[(i-1)*JMAX+j]) / (2.0f*DS) + (guidance_y[i*JMAX+(j+1)] - guidance_y[i*JMAX+(j-1)]) / (2.0f*DS);
                    if(bound[i*JMAX+j] > 0.0f){
                        // Free space - no clamping in reference implementation
                    }
                    else if(bound[i*JMAX+j] < 0.0f){
                        // Occupied space - clamp divergence
                        force[i*JMAX+j] = std::max(force[i*JMAX+j], max_div);
                        force[i*JMAX+j] = std::min(force[i*JMAX+j], 0.0f);
                    }
                    else{
                        force[i*JMAX+j] = 0.0f;
                    }
                }
            }
        };


        /* Compute the Poisson Safety Function */
        bool solve_poisson_safety_function(void){

            // Start Solve Timer
            Timer solve_timer(true);
            solve_timer.start();

            // Initialize Temporary Grids
            float *hgrid_temp = (float *)malloc(IMAX*JMAX*QMAX*sizeof(float));
            if (!hgrid_temp) {
                return false;
            }
            memcpy(hgrid_temp, hgrid1, IMAX*JMAX*QMAX*sizeof(float));

            // Execute Poisson Pre-Processing
            build_occ_map(occ1, occ0, conf);
            find_boundary(hgrid_temp, occ1, false);

            // Validate pointers before OMP loop
            if (!force || !bound || !robot_kernel_obstacle || !robot_kernel_human) {
                free(hgrid_temp);
                return false;
            }

            // Allocate 3D guidance field arrays (must match CUDA solver's expected size)
            float *guidance_x = (float *)calloc(IMAX*JMAX*QMAX, sizeof(float));
            float *guidance_y = (float *)calloc(IMAX*JMAX*QMAX, sizeof(float));
            float *f0 = (float *)calloc(IMAX*JMAX*QMAX, sizeof(float));
            if (!guidance_x || !guidance_y || !f0) {
                free(hgrid_temp);
                if (guidance_x) free(guidance_x);
                if (guidance_y) free(guidance_y);
                if (f0) free(f0);
                return false;
            }
            
            // Clear tangent layer display for fresh visualization each frame
            memset(tangent_layer_display, 0, IMAX*JMAX*sizeof(int8_t));
            
            // Phase 1: Brushfire human labels on TRUE occupancy (occ1) first
            // This identifies which real obstacles are humans before inflation
            label_human_clusters(occ1);
            
            // Compute tight-area flag ONCE before parallel loop
            // (queries robot state and h-field, must be done on main thread)
            const bool tight_area = is_tight_area();
            
            // Phase 2: Inflate class_map_expanded alongside the first bound slice
            // This propagates human labels outward during inflation so they align with boundaries
            // Class-aware inflation: human boundaries get larger kernel, obstacles get smaller
            {
                float *bound_q0 = bound;  // q=0 slice
                memcpy(bound_q0, occ1, IMAX*JMAX*sizeof(float));
                inflate_occupancy_grid(bound_q0, class_map_expanded);
            }
            
            // Phase 3: Inflate remaining Q slices
            // IMPORTANT: Must pass class_map_expanded so kernel selection works correctly
            #pragma omp parallel for num_threads(4)
            for(int q=0; q<QMAX; q++){
                float *bound_slice = bound + q*IMAX*JMAX;
                float *hgrid_slice = hgrid_temp + q*IMAX*JMAX;
                
                if(q != 0){  // q=0 already inflated above
                    memcpy(bound_slice, occ1, IMAX*JMAX*sizeof(float));
                    // Use class_map_expanded (read-only after q=0) for proper kernel selection
                    inflate_occupancy_grid(bound_slice, class_map_expanded);
                }
                find_boundary(hgrid_slice, bound_slice, true, tight_area, class_map_expanded);
            }
            
            // Human labels are now properly inflated to match the boundaries
            
            // Phase 4: Compute boundary gradients with class-aware dh0
            // Capture robot state for velocity-aware social navigation (must be before parallel loop)
            const float rx = x[0];
            const float ry = x[1];
            // vn is in World/Odom Frame. Transform to Body Frame for relative calculations.
            // x[2] is robot yaw.
            const float c_yaw = std::cos(x[2]);
            const float s_yaw = std::sin(x[2]);
            // Rotate World Velocity into Body Velocity
            const float vn_body_x = c_yaw * vn[0] + s_yaw * vn[1];
            const float vn_body_y = -s_yaw * vn[0] + c_yaw * vn[1];
            
            // For compute_boundary_gradients (which seems unused/placeholders), keep original or update?
            // The function signature has commented out vn_x, vn_y.
            // But compute_tangent_direction DEFINITELY needs Body Frame velocity.
            
            // First, process q=0 slice with populate_human_info=true (must be sequential)
            {
                float *bound_slice = bound + 0*IMAX*JMAX;
                float *guidance_x_slice = guidance_x + 0*IMAX*JMAX;
                float *guidance_y_slice = guidance_y + 0*IMAX*JMAX;
                compute_boundary_gradients(guidance_x_slice, guidance_y_slice, bound_slice, 
                                          class_map_expanded, rx, ry, vn_body_x, vn_body_y, true);  // populate_human_info = true
            }
            
            // Then process q=1 to QMAX-1 in parallel (no human_boundary_info_ access)
            #pragma omp parallel for num_threads(4)
            for(int q=1; q<QMAX; q++){
                float *bound_slice = bound + q*IMAX*JMAX;
                float *guidance_x_slice = guidance_x + q*IMAX*JMAX;
                float *guidance_y_slice = guidance_y + q*IMAX*JMAX;
                
                // Pass expanded class map and robot state for velocity-aware social navigation
                compute_boundary_gradients(guidance_x_slice, guidance_y_slice, bound_slice, 
                                          class_map_expanded, rx, ry, vn_body_x, vn_body_y, false);  // populate_human_info = false
            }
            
            // Phase 5: PRE-LAPLACE tangential bias using circular dilation
            // This creates layered Dirichlet BCs that Laplace naturally interpolates
            float* bound_guidance = nullptr;
            
            if (enable_social_navigation_ && social_tangent_layers_ > 0 && !human_boundary_info_.empty()) {
                // Allocate temporary bound array for guidance field solve
                bound_guidance = (float*)malloc(IMAX*JMAX*QMAX*sizeof(float));
                if (bound_guidance == nullptr) {
                    RCLCPP_ERROR(this->get_logger(), "Memory allocation failed for bound_guidance");
                    return false;  // Graceful fallback
                }
                
                // Get heading-based sign for tangent direction
                auto active_tracks = human_tracker_->get_active_tracks();
                // Tracks are in Body Frame, so pass Robot Position as (0,0)
                // Pass robot velocity for closing speed comparison
                float sign = compute_tangent_direction(active_tracks, 0.0f, 0.0f, vn_body_x, vn_body_y);
                
                // Process each Q slice: expand human obstacles and apply tangent bias
                for (int q = 0; q < QMAX; q++) {
                    expand_human_obstacles_for_guidance(
                        bound_guidance + q*IMAX*JMAX,
                        guidance_x + q*IMAX*JMAX,
                        guidance_y + q*IMAX*JMAX,
                        bound + q*IMAX*JMAX,
                        social_tangent_layers_,
                        social_layer_thickness_,
                        social_tangent_bias_,
                        sign);
                }
            } else {
                // No social nav: use original bound array
                bound_guidance = bound;  // Point to original, don't free later
            }
            
            // Phase 6: Solve Laplace equation for guidance fields
            const float v_RelTol = 1.0e-4f;
            const int N_guidance = IMAX/5;
            const float w_SOR_guidance = 2.0f/(1.0f+std::sin(M_PI/(float)(N_guidance+1)));
            
            // DEBUG: Store pre-solve boundary magnitudes for q=0 slice
            float pre_solve_max_boundary_mag = 0.0f;
            for(int n = 0; n < IMAX*JMAX; n++){
                if(bound[n] == 0.0f){
                    float mag = std::sqrt(guidance_x[n]*guidance_x[n] + guidance_y[n]*guidance_y[n]);
                    pre_solve_max_boundary_mag = std::max(pre_solve_max_boundary_mag, mag);
                }
            }
            
            // Use bound_guidance for Laplace solve (includes expanded layer BCs)
            (void)Kernel::poissonSolve(guidance_x, f0, bound_guidance, v_RelTol, w_SOR_guidance);
            (void)Kernel::poissonSolve(guidance_y, f0, bound_guidance, v_RelTol, w_SOR_guidance);
            
            // Free bound_guidance if we allocated it
            if (bound_guidance != bound) {
                free(bound_guidance);
            }
            bound_guidance = nullptr;
            
            // DEBUG: Compare post-solve boundary magnitudes for q=0 slice
            float post_solve_max_boundary_mag = 0.0f;
            for(int n = 0; n < IMAX*JMAX; n++){
                if(bound[n] == 0.0f){
                    float mag = std::sqrt(guidance_x[n]*guidance_x[n] + guidance_y[n]*guidance_y[n]);
                    post_solve_max_boundary_mag = std::max(post_solve_max_boundary_mag, mag);
                }
            }
            static int poisson_debug_counter = 0;
            if(++poisson_debug_counter >= 15){
                // printf("[DEBUG POISSON] Boundary mag - PRE: %.3f, POST: %.3f, CHANGED: %s\n",
                //        pre_solve_max_boundary_mag, post_solve_max_boundary_mag,
                //        (std::abs(post_solve_max_boundary_mag - pre_solve_max_boundary_mag) > 0.01f) ? "YES!" : "no");
                // printf("[DEBUG POISSON] dh0=%.2f, dh0_human=%.2f, dh0_obstacle=%.2f\n",
                //        dh0, dh0_human, dh0_obstacle);
                poisson_debug_counter = 0;
            }
            
            // Phase 3: Compute forcing function per slice from smoothed guidance fields
            #pragma omp parallel for num_threads(4)
            for(int q=0; q<QMAX; q++){
                float *force_slice = force + q*IMAX*JMAX;
                float *bound_slice = bound + q*IMAX*JMAX;
                float *guidance_x_slice = guidance_x + q*IMAX*JMAX;
                float *guidance_y_slice = guidance_y + q*IMAX*JMAX;
                
                compute_optimal_forcing_function(force_slice, guidance_x_slice, guidance_y_slice, bound_slice);
                for(int n=0; n<IMAX*JMAX; n++){
                    force_slice[n] *= DS*DS;
                }
            }
            
            // Copy q=0 slice of guidance fields and boundary for display before freeing
            memcpy(guidance_x_display, guidance_x, IMAX*JMAX*sizeof(float));
            memcpy(guidance_y_display, guidance_y, IMAX*JMAX*sizeof(float));
            memcpy(bound_display, bound, IMAX*JMAX*sizeof(float));  // Store boundary for debug viz
            
            // Copy full 3D guidance field for MPC use
            memcpy(guidance_x_grid, guidance_x, IMAX*JMAX*QMAX*sizeof(float));
            memcpy(guidance_y_grid, guidance_y, IMAX*JMAX*QMAX*sizeof(float));
            
            // Free temporary 3D guidance arrays
            free(guidance_x);
            free(guidance_y);
            free(f0);

            // Solve Poisson's Equation
            const float relTol = 1.0e-4f;
            const int N = IMAX/5;
            const float w_SOR = 2.0f/(1.0f+std::sin(M_PI/(float)(N+1))); // This is the "optimal" value from Strikwerda, Chapter 13.5
            (void)Kernel::poissonSolve(hgrid_temp, force, bound, relTol, w_SOR); // CUDA!
            
            // Transfer Solutions into Necessary Locations
            memcpy(occ0, occ1, IMAX*JMAX*sizeof(float));
            memcpy(hgrid0, hgrid1, IMAX*JMAX*QMAX*sizeof(float));
            memcpy(hgrid1, hgrid_temp, IMAX*JMAX*QMAX*sizeof(float));
            free(hgrid_temp);
            if(h_flag) dhdt_flag = true;
            
            // Throttle timing prints to match other throttled output (~1Hz)
            static int solve_print_counter = 0;
            if(++solve_print_counter >= 15){
                solve_timer.time("Poisson Solve Time: ");
                // printf("Poisson Iterations: %u \n", iters);
                solve_print_counter = 0;
            }

            return true;

        };

        /* Display Poisson Safety Function Grid (Interpolated) */
        void display_poisson_safety_function(void){

            Timer display_timer(true);
            display_timer.start();

            // Build Interpolated Grid for Display
            const float qr = yaw_to_q(x[2],xc[2]);
            const float q1f = std::floor(qr);
            const float q2f = std::ceil(qr);
            const int q1 = (int)q_wrap(q1f);
            const int q2 = (int)q_wrap(q2f);
            #pragma omp parallel for
            for(int n = 0; n < IMAX*JMAX; n++){
                if(q1f!=q2f) grid_temp[n] = (q2f - qr) * hgrid1[q1*IMAX*JMAX+n] + (qr - q1f) * hgrid1[q2*IMAX*JMAX+n];
                else grid_temp[n] = hgrid1[q1*IMAX*JMAX+n];
            }

            // Populate Float Grayscale Poisson Image with Chosen q & k Values
            cv::Mat poisson_img = cv::Mat::zeros(IMAX, JMAX, CV_32FC1);
            for (int i = 0; i < IMAX; i++){
                for (int j = 0; j < JMAX; j++){
                    poisson_img.at<float>(i,j) = grid_temp[i*JMAX+j];
                }
            }

            // Convert to 8-bit Grayscale using fixed scale (raw values, not normalized)
            // Clamp values to [0, 1] range and scale to [0, 255]
            cv::Mat gray_img;
            cv::normalize(poisson_img, gray_img, 0, 255, cv::NORM_MINMAX);
            gray_img.convertTo(gray_img, CV_8U);
            // poisson_img.convertTo(gray_img, CV_8U, 255.0, 0.0);  // scale=255, offset=0

            // Convert to Colormap
            cv::Mat color_img;
            cv::applyColorMap(gray_img, color_img, cv::COLORMAP_DEEPGREEN);

            // Overlay red for human cells (class_map_expanded == 1)
            for (int i = 0; i < IMAX; i++){
                for (int j = 0; j < JMAX; j++){
                    if (class_map_expanded[i*JMAX+j] == 1){
                        color_img.at<cv::Vec3b>(i, j) = cv::Vec3b(0, 0, 255);  // BGR red
                    }
                }
            }

            // Resize for Display
            cv::Mat resized_img;
            const int upscale = 6;
            cv::resize(color_img, resized_img, cv::Size(), upscale, upscale, cv::INTER_NEAREST);

            // Add Current Location & Goal Location
            // In body_link frame, robot is always at origin (0, 0)
            cv::Point curr_pt = cv::Point(upscale*x_to_j(0.0f,xc[0]),upscale*y_to_i(0.0f,xc[1]));
            cv::Point goal_pt = cv::Point(upscale*x_to_j(xd[0],xc[0]),upscale*y_to_i(xd[1],xc[1]));
            cv::circle(resized_img, curr_pt, upscale, cv::Scalar(0, 0, 0), cv::FILLED);
            cv::circle(resized_img, goal_pt, upscale, cv::Scalar(0, 127, 0), cv::FILLED);

            // Add MPC Trajectory
            for(int n = 1; n < TMAX; n++){
                const int j_traj = x_to_j(mpc3d_controller.sol(STATES*n+0), xc[0]);
                const int i_traj = y_to_i(mpc3d_controller.sol(STATES*n+1), xc[1]);
                cv::Point traj_pt = cv::Point(upscale*j_traj, upscale*i_traj);
                cv::circle(resized_img, traj_pt, upscale/2, cv::Scalar(255, 0, 0), cv::FILLED);
            }

            // Draw Guidance Field Quivers (using stored guidance field, not h-field gradient)
            const int stride = 4;
            const float quiver_scale = 1.5f;
            // Debug counters for boundary gradient magnitudes
            float max_human_boundary_mag = 0.0f;
            float max_obstacle_boundary_mag = 0.0f;
            float max_freespace_mag = 0.0f;
            int human_boundary_count = 0;
            int obstacle_boundary_count = 0;
            
            for(int i = 1; i < IMAX-1; i+=stride){
                for(int j = 1; j < JMAX-1; j+=stride){
                    // Skip if inside obstacle
                    if(occ1[i*JMAX+j] < 0.0f) continue; // occ1 = -1 is obstacle
                    
                    // Use stored guidance field directly
                    // guidance_x is the i-direction (rows), guidance_y is the j-direction (cols)
                    float gx = guidance_x_display[i*JMAX+j];
                    float gy = guidance_y_display[i*JMAX+j];
                    
                    float mag = std::sqrt(gx*gx + gy*gy);
                    if(mag < 0.1f) continue; // Skip small vectors
                    
                    // Check if this is a boundary cell (bound == 0) or free space (bound > 0)
                    bool is_boundary = (bound_display[i*JMAX+j] == 0.0f);
                    
                    // Track magnitudes by cell type for debugging
                    if (is_boundary) {
                        // Check adjacent occupied cells for human labels (class_map is now inflated)
                        bool is_human = false;
                        for(int di = -1; di <= 1 && !is_human; di++){
                            for(int dj = -1; dj <= 1 && !is_human; dj++){
                                int ni = i + di;
                                int nj = j + dj;
                                if(ni >= 0 && ni < IMAX && nj >= 0 && nj < JMAX){
                                    if(bound_display[ni*JMAX+nj] < 0.0f && class_map_expanded[ni*JMAX+nj] == 1){
                                        is_human = true;
                                    }
                                }
                            }
                        }
                        if (is_human) {
                            max_human_boundary_mag = std::max(max_human_boundary_mag, mag);
                            human_boundary_count++;
                        } else {
                            max_obstacle_boundary_mag = std::max(max_obstacle_boundary_mag, mag);
                            obstacle_boundary_count++;
                        }
                    } else {
                        max_freespace_mag = std::max(max_freespace_mag, mag);
                    }
                    
                    // Scale by actual magnitude (clamped) to show field strength
                    float nx = gx;  // guidance_x points away from obstacles in i-direction
                    float ny = gy;  // guidance_y points away from obstacles in j-direction
                    
                    // Draw Arrow: (j, i) is (x, y) in image coordinates
                    // After vertical flip: i increases upward, so ny should be negated for display
                    cv::Point p1(j*upscale, i*upscale);
                    cv::Point p2((int)((j + ny*quiver_scale)*upscale), (int)((i + nx*quiver_scale)*upscale));
                    
                    // Boundary cells: RED, Tangential layers: YELLOW, Free space: GREEN
                    cv::Scalar arrow_color;
                    if (is_boundary) {
                        arrow_color = cv::Scalar(0, 0, 255);  // RED (BGR)
                    } else if (tangent_layer_display[i*JMAX+j] == 1) {
                        arrow_color = cv::Scalar(0, 255, 255);  // YELLOW (BGR)
                    } else {
                        arrow_color = cv::Scalar(100, 255, 100);  // GREEN (BGR)
                    }
                    cv::arrowedLine(resized_img, p1, p2, arrow_color, 1, 8, 0, 0.4);
                }
            }
            
            // Throttle debug output to ~1Hz
            static int debug_print_counter = 0;
            if(++debug_print_counter >= 15){
                // Count cells by bound value and class
                int bound_zero_count = 0;
                int bound_negative_count = 0;
                int human_label_count = 0;
                int human_at_boundary_count = 0;
                for(int n = 0; n < IMAX*JMAX; n++){
                    if(bound_display[n] == 0.0f) bound_zero_count++;
                    if(bound_display[n] < 0.0f) bound_negative_count++;
                    if(class_map_expanded[n] == 1) human_label_count++;
                    // Check if boundary cell (bound==0) is adjacent to human-labeled occupied cell
                    if(bound_display[n] == 0.0f){
                        int i = n / JMAX;
                        int j = n % JMAX;
                        for(int di = -1; di <= 1; di++){
                            for(int dj = -1; dj <= 1; dj++){
                                int ni = i + di;
                                int nj = j + dj;
                                if(ni >= 0 && ni < IMAX && nj >= 0 && nj < JMAX){
                                    if(bound_display[ni*JMAX+nj] < 0.0f && class_map_expanded[ni*JMAX+nj] == 1){
                                        human_at_boundary_count++;
                                        goto next_cell;  // Count once per boundary cell
                                    }
                                }
                            }
                        }
                        next_cell:;
                    }
                }
                // printf("[DEBUG] Boundary mags - Human: %.3f (%d cells), Obstacle: %.3f (%d cells), FreeSpace: %.3f\n",
                //        max_human_boundary_mag, human_boundary_count, 
                //        max_obstacle_boundary_mag, obstacle_boundary_count,
                //        max_freespace_mag);
                // printf("[DEBUG] Cells - bound==0: %d, bound<0: %d, human_labels: %d, human_at_boundary: %d\n",
                //        bound_zero_count, bound_negative_count, human_label_count, human_at_boundary_count);
                // printf("[DEBUG] FLAGS - realtime_sf: %s, predictive_sf: %s\n",
                //        realtime_sf_flag ? "ON" : "OFF", predictive_sf_flag ? "ON" : "OFF");
                // printf("[DEBUG] Social nav - tangent direction: %s\n",
                //        current_tangent_direction_ < 0 ? "CCW (human facing robot)" : "CW (human back to robot)");
                debug_print_counter = 0;
            }

            // Vertical Flip Image for Display 
            cv::Mat flipped_img;
            cv::flip(resized_img, flipped_img, 0);

            // Display Final Image
            cv::imshow("Poisson Solution", flipped_img);
            cv::waitKey(1);
            
            // Rate-limited publishing of visualization image and logging data
            auto current_time = std::chrono::steady_clock::now();
            double time_since_last = std::chrono::duration<double>(current_time - last_logging_publish_time_).count();
            if (time_since_last >= logging_publish_period_) {
                last_logging_publish_time_ = current_time;
                
                // Publish logging data (same format as CSV: t_ms, space_counter, x, y, yaw, vx, vy, vyaw, vxd, vyd, vyawd, h, dhdx, dhdy, dhdq, dhdt, alpha, on_off)
                const std::vector<float> logging_data = {t_ms, (float)space_counter, x[0], x[1], x[2], 
                                                         v[0], v[1], v[2], vt[0], vt[1], vt[2], 
                                                         h, dhdx, dhdy, dhdq, dhdt, wn, (float)(realtime_sf_flag|predictive_sf_flag)};
                std_msgs::msg::Float32MultiArray logging_msg;
                logging_msg.data = logging_data;
                logging_data_pub_->publish(logging_msg);
                
                // Publish visualization image
                if (poisson_image_pub_ && !flipped_img.empty()) {
                    try {
                        // Manual conversion from OpenCV Mat to ROS Image message
                        sensor_msgs::msg::Image img_msg;
                        img_msg.header.stamp = this->now();
                        img_msg.header.frame_id = "odom";
                        img_msg.height = flipped_img.rows;
                        img_msg.width = flipped_img.cols;
                        img_msg.encoding = "bgr8";
                        img_msg.is_bigendian = false;
                        img_msg.step = flipped_img.cols * 3;  // 3 bytes per pixel for BGR8
                        size_t size = img_msg.step * img_msg.height;
                        img_msg.data.resize(size);
                        memcpy(&img_msg.data[0], flipped_img.data, size);
                        poisson_image_pub_->publish(img_msg);
                    } catch (const std::exception& e) {
                        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                            "Failed to publish Poisson image: %s", e.what());
                    }
                }
            }

            // Throttle display timing prints
            static int display_print_counter = 0;
            if(++display_print_counter >= 15){
                display_timer.time("Display Time: ");
                display_print_counter = 0;
            }

        }

        void update_dhdt_grid(void){

            const float wc = 10.0f;
            const float kc = 1.0f - std::exp(-wc*dt_grid);

            for(int i = 0; i<IMAX; i++){
                for(int j = 0; j<JMAX; j++){
                    for(int q = 0; q<QMAX; q++){
                        const float i0 = (float)i + dx[1] / DS;
                        const float j0 = (float)j + dx[0] / DS;
                        const bool in_grid = (i0 >= 0.0f) && (i0 <= (float)(IMAX-1)) && (j0 >= 0.0f) && (j0 <= (float)(JMAX-1));
                        float dhdt_ij = 0.0f;
                        if(in_grid){
                            const float h0 = trilinear_interpolation(hgrid0, i0, j0, q);
                            const float h1 = trilinear_interpolation(hgrid1, i, j, q);
                            dhdt_ij = (h1 - h0) / dt_grid;
                        }
                        dhdt_grid[q*IMAX*JMAX+i*JMAX+j] *= 1.0f - kc;
                        dhdt_grid[q*IMAX*JMAX+i*JMAX+j] += kc * dhdt_ij;
                    }
                }
            }

        }


        void safety_filter(const std::vector<float> vd){

            // Fractional Indices Corresponding to Current State
            // In body_link frame, robot is always at origin (0, 0)
            const float ic = y_to_i(0.0f, xc[1]);
            const float jc = x_to_j(0.0f, xc[0]);
            // Robot yaw is 0 in body_link frame
            const float qc = yaw_to_q(0.0f, xc[2]);

            // Get Safety Function Rate
            //dhdt = trilinear_interpolation(dhdt_grid, ic, jc, qc);
            const int range = (int)std::round(0.2f/DS);
            dhdt = 0.0f;
            for(int di=-range; di<=range; di++){
                for(int dj=-range; dj<=range; dj++){
                    float dhdt_ij = trilinear_interpolation(dhdt_grid, ic+(float)di, jc+(float)dj, qc);
                    if(dhdt_ij < dhdt) dhdt = dhdt_ij;
                }
            }

            // Get Safety Function Value & Forward Propogate to Remove Latency
            h = trilinear_interpolation(hgrid1, ic, jc, qc);
            const float h_pred = h + dhdt * grid_age;
        
            // Guidance field v from Laplace solve (stored in guidance_x/y_grid)
            // guidance_x is i-direction (y), guidance_y is j-direction (x)
            const float vx = trilinear_interpolation(guidance_y_grid, ic, jc, qc);  // j = x
            const float vy = trilinear_interpolation(guidance_x_grid, ic, jc, qc);  // i = y
            const float v_norm = std::sqrt(vx*vx + vy*vy);
            
            // Compute Dh numerically from h-field (separate from guidance field)
            const float h_eps = 1.0f;
            const float hip = trilinear_interpolation(hgrid1, ic+h_eps, jc, qc);
            const float him = trilinear_interpolation(hgrid1, ic-h_eps, jc, qc);
            const float hjp = trilinear_interpolation(hgrid1, ic, jc+h_eps, qc);
            const float hjm = trilinear_interpolation(hgrid1, ic, jc-h_eps, qc);
            const float Dh_x = (hjp - hjm) / (2.0f * h_eps * DS);  // dh/dx
            const float Dh_y = (hip - him) / (2.0f * h_eps * DS);  // dh/dy
            
            // Store for publishing (use guidance field for control direction)
            dhdx = vx;
            dhdy = vy;
            
            // Compute dhdq numerically (guidance field doesn't have yaw component)
            const float q_eps = 1.0f;
            const float qp = q_wrap(qc + q_eps);
            const float qm = q_wrap(qc - q_eps);
            float hqp = trilinear_interpolation(hgrid1, ic, jc, qp);
            float hqm = trilinear_interpolation(hgrid1, ic, jc, qm);
            dhdq = (hqp-hqm) / (2.0f*q_eps*DQ);

            // Forward propagate gradients for prediction
            const float dhdx_pred = vx;  // Use guidance field for control
            const float dhdy_pred = vy;
            
            // Forward propagate dhdq
            hqp += trilinear_interpolation(dhdt_grid, ic, jc, qp) * grid_age;
            hqm += trilinear_interpolation(dhdt_grid, ic, jc, qm) * grid_age;
            const float dhdq_pred = (hqp-hqm) / (2.0f*q_eps*DQ);
            
            // Compute ||Dh|| using numerical h-gradient
            const float Dh_norm = std::sqrt(Dh_x*Dh_x + Dh_y*Dh_y + dhdq_pred*dhdq_pred);
            
            // sigma(h) = epsilon * (1 - exp(-kappa * max(0, h))) [eq. 31]
            const float sigma_h = cbf_sigma_epsilon_ * (1.0f - std::exp(-cbf_sigma_kappa_ * std::max(0.0f, h_pred)));
            
            // Scaling factor for dhdt: ||v|| / (||Dh|| + sigma(h)) [eq. 31], clamped to prevent MPC infeasibility
            const float dhdt_scale = std::min(v_norm / (Dh_norm + sigma_h + 1e-6f), 1.0f);
            
            // Single Integrator Safety Filter
            const float Pu[3] = {2.0f, 2.0f, 1.0f};
            const float ISSf1 = issf;
            const float ISSf2 = issf;
            const float b = dhdx_pred*dhdx_pred/Pu[0] + dhdy_pred*dhdy_pred/Pu[1] + dhdq_pred*dhdq_pred/Pu[2];
            float ISSf = std::sqrt(b) / ISSf1 + b / ISSf2;
            const float sigma = std::clamp(-10.0f*dhdt, 0.0f, 1.0f);
            ISSf *= sigma;
            
            // Activating function per eq. 31:
            // a = v · k_nom + (||v|| / (||Dh|| + sigma)) * dhdt + gamma * h
            float a = wn * h_pred;                              // gamma * h
            a += vx * vd[0] + vy * vd[1];                       // v · k_nom (guidance-aligned)
            a += dhdt_scale * dhdt;                             // scaled dhdt term
            a += dhdq_pred * vd[2];                             // yaw term
            a -= ISSf; // Input-to-State Safety (Robustness)
            
            // Analytical Safety Filter
            const float sigma_sontag = 1.0f;
            //const float sigma_softpl = 0.5f;
            float lambda = 0.0f;
            //if(b>1.0e-4f) lambda = std::max(0.0f, -a/b); // ReLU
            //if(b>1.0e-4f) lambda = (-a + std::sqrt(a*a+sigma_sontag*b*b)) / b; // Sontag 
            if(b>1.0e-4f) lambda = 1.0f * (-a + std::sqrt(a*a+sigma_sontag*b*b)) / (2.0f*b); // Half Sontag

            v = vd;
            if(realtime_sf_flag){
                v[0] += lambda * dhdx_pred / Pu[0];
                v[1] += lambda * dhdy_pred / Pu[1];
                v[2] += lambda * dhdq_pred / Pu[2];
            }

        };


        void occ_grid_callback(nav_msgs::msg::OccupancyGrid::UniquePtr msg){

            // Compute Grid Timing
            dt_grid = std::chrono::duration<float>(std::chrono::steady_clock::now() - t_grid).count();
            t_grid = std::chrono::steady_clock::now();
            grid_age = dt_grid;

            // Read Message Data
            dx[0] = msg->info.origin.position.x - xc[0];
            dx[1] = msg->info.origin.position.y - xc[1];
            xc[0] = msg->info.origin.position.x;
            xc[1] = msg->info.origin.position.y;
            for(int n = 0; n < IMAX*JMAX; n++) conf[n] = msg->data[n];

            // Solve Poisson Safety Function (New Occupancy, New Orientation)
            h_flag = solve_poisson_safety_function();

            // Update Grid of dh/dt Values
            if(start_flag && dhdt_flag){
                update_dhdt_grid();
            }

            // Display Results (controlled by enable_display param, not start_flag)
            if(enable_display){
                display_poisson_safety_function();
            }
            // Throttle timing prints to ~1Hz (every 15 frames at 15Hz)
            static int timing_print_counter = 0;
            if(++timing_print_counter >= 15){
                timing_print_counter = 0;
                std::cout << "Grid Loop Time: " << dt_grid*1.0e3f << " ms" << std::endl;
                std::cout << "Control Loop Time: " << dt_state*1.0e3f << " ms" << std::endl;
                std::cout << "Command: <" << vb[0] << "," << vb[1] << "," << vb[2] << ">" << std::endl;
            }

        };

        /* Class map callback - receives sparse human labels from YOLO */
        void class_map_callback(nav_msgs::msg::OccupancyGrid::UniquePtr msg){
            // Simply store raw YOLO labels - tracking/persistence handled by HumanTracker
            for(int n = 0; n < IMAX*JMAX; n++){
                class_map[n] = msg->data[n];
            }
        };

        /* Visibility map callback - receives which cells the camera can see */
        void visibility_map_callback(nav_msgs::msg::OccupancyGrid::UniquePtr msg){
            for(int n = 0; n < IMAX*JMAX; n++){
                visibility_map[n] = msg->data[n];
            }
        };

        /* Extract LiDAR clusters using connected components and compute centroids.
         * Returns vector of ClusterInfo with positions in odom frame.
         */
        std::vector<ClusterInfo> extract_lidar_clusters(const float* occ_true) {
            std::vector<ClusterInfo> clusters;
            
            // Create binary image from occupancy (occ < 0 = occupied)
            cv::Mat binary(IMAX, JMAX, CV_8UC1);
            for (int n = 0; n < IMAX*JMAX; n++) {
                binary.data[n] = (occ_true[n] < 0.0f) ? 255 : 0;
            }
            
            // Connected components with stats
            cv::Mat labels, stats, centroids;
            int num_labels = cv::connectedComponentsWithStats(binary, labels, stats, centroids);
            
            // Build ClusterInfo for each component (skip label 0 = background)
            for (int l = 1; l < num_labels; l++) {
                int area = stats.at<int>(l, cv::CC_STAT_AREA);
                
                // Filter small clusters (noise)
                if (area < 3) continue;
                
                ClusterInfo c;
                // Centroid from OpenCV is in (x, y) = (col, row) format
                float j_centroid = static_cast<float>(centroids.at<double>(l, 0));
                float i_centroid = static_cast<float>(centroids.at<double>(l, 1));
                
                // Convert to body_link frame: position relative to robot (at grid center)
                // Robot is at grid center (JMAX/2, IMAX/2), so offset from center gives body-frame position
                c.centroid_x = (j_centroid - JMAX/2) * DS;
                c.centroid_y = (i_centroid - IMAX/2) * DS;
                c.cell_count = area;
                c.label_id = l;
                
                // Check for YOLO human overlap and camera visibility in this cluster
                // Require minimum cells to filter stray YOLO spillover pixels
                int yolo_cell_count = 0;
                int visible_cell_count = 0;
                int cluster_cell_count = 0;
                for (int n = 0; n < IMAX*JMAX; n++) {
                    if (labels.at<int>(n) == l) {
                        cluster_cell_count++;
                        if (class_map[n] == 1) {
                            yolo_cell_count++;
                        }
                        if (visibility_map[n] == 1) {
                            visible_cell_count++;
                        }
                    }
                }
                c.has_yolo_seed = (yolo_cell_count >= min_yolo_cells_);
                // Require 50%+ of cluster cells to be visible to consider in FOV
                c.in_camera_fov = (cluster_cell_count > 0 && 
                                   visible_cell_count * 2 >= cluster_cell_count);
                
                clusters.push_back(c);
            }
            
            return clusters;
        }

        /* Label human clusters using object-level tracking.
         * Uses HumanTracker to persist identity across frames.
         * occ_true: The true occupancy grid (occ1) - values: -1 = occupied, 1 = free
         */
        void label_human_clusters(const float* occ_true){
            // Clear expanded map
            memset(class_map_expanded, 0, IMAX*JMAX*sizeof(int8_t));
            
            // Step 1: Extract LiDAR clusters
            auto clusters = extract_lidar_clusters(occ_true);
            
            // Step 2: Update tracker with current clusters
            float current_time = std::chrono::duration<float>(
                std::chrono::steady_clock::now() - t_start).count();
            human_tracker_->update(clusters, current_time);
            
            // Step 3: Label cells based on active tracks
            auto active_tracks = human_tracker_->get_active_tracks();
            
            // For each track, find the nearest cluster and label its cells
            cv::Mat binary(IMAX, JMAX, CV_8UC1);
            cv::Mat labels, stats, centroids;
            for (int n = 0; n < IMAX*JMAX; n++) {
                binary.data[n] = (occ_true[n] < 0.0f) ? 255 : 0;
            }
            int num_labels = cv::connectedComponentsWithStats(binary, labels, stats, centroids);
            
            for (const auto& track : active_tracks) {
                // Convert track position to grid indices (inverse of centroid formula)
                // centroid_x = (j - JMAX/2) * DS → j = centroid_x / DS + JMAX/2
                float track_j = track.x / DS + JMAX/2;
                float track_i = track.y / DS + IMAX/2;
                
                // Find nearest cluster to track position
                float best_dist = 999999.0f;
                int best_label = -1;
                
                for (int l = 1; l < num_labels; l++) {
                    float j_cent = static_cast<float>(centroids.at<double>(l, 0));
                    float i_cent = static_cast<float>(centroids.at<double>(l, 1));
                    float dist = std::sqrt((j_cent - track_j)*(j_cent - track_j) + 
                                          (i_cent - track_i)*(i_cent - track_i));
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_label = l;
                    }
                }
                
                // Gate: only label if cluster is close enough AND track is confirmed
                // Track must have yolo_ever_confirmed (≥N consecutive YOLO detections) + confidence > 0.5
                const float gate_cells = 1.0f / DS;  // 1 meter in grid cells
                const int max_human_cells = static_cast<int>(0.4f / (DS * DS));  // ~1.5 m² max for human
                
                if (best_label > 0 && best_dist < gate_cells && 
                    track.yolo_ever_confirmed && track.confidence > 0.5f) {
                    
                    int cluster_size = stats.at<int>(best_label, cv::CC_STAT_AREA);
                    
                    if (cluster_size <= max_human_cells) {
                        // Normal case: cluster is human-sized, label entire cluster
                        for (int n = 0; n < IMAX*JMAX; n++) {
                            if (labels.at<int>(n) == best_label) {
                                class_map_expanded[n] = 1;
                            }
                        }
                    } else {
                        // Cluster too large (merged with obstacle): only label cells near track centroid
                        // Use a radius of ~0.1m around the track position
                        const float label_radius = 0.1f / DS;  // radius in cells
                        const float radius_sq = label_radius * label_radius;
                        
                        for (int n = 0; n < IMAX*JMAX; n++) {
                            if (labels.at<int>(n) == best_label) {
                                int i = n / JMAX;
                                int j = n % JMAX;
                                float di = static_cast<float>(i) - track_i;
                                float dj = static_cast<float>(j) - track_j;
                                if (di*di + dj*dj <= radius_sq) {
                                    class_map_expanded[n] = 1;
                                }
                            }
                        }
                    }
                }
            }
            
            // Count labeled cells for stats
            int labeled_cells = 0;
            for (int n = 0; n < IMAX*JMAX; n++) {
                if (class_map_expanded[n] == 1) labeled_cells++;
            }
            
            // Phase 2: Dilate human labels by robot shape (same as before)
            if (enable_human_tracker_dilation_ && labeled_cells > 0) {
                int8_t temp_expanded[IMAX*JMAX];
                memcpy(temp_expanded, class_map_expanded, IMAX*JMAX*sizeof(int8_t));
                
                // Use human kernel since we're dilating human labels
                const float *kernel = robot_kernel_human;  // q=0 slice
                int lim = (robot_kernel_dim_human - 1) / 2;
                
                for(int i = 1; i < IMAX-1; i++){
                    int ilow = std::max(i - lim, 0);
                    int itop = std::min(i + lim, IMAX);
                    for(int j = 1; j < JMAX-1; j++){
                        if(class_map_expanded[i*JMAX + j] != 1) continue;
                        
                        int jlow = std::max(j - lim, 0);
                        int jtop = std::min(j + lim, JMAX);
                        
                        for(int p = ilow; p < itop; p++){
                            for(int q = jlow; q < jtop; q++){
                                float kernel_val = kernel[(p-i+lim)*robot_kernel_dim_human + (q-j+lim)];
                                if(kernel_val < 0.0f){
                                    temp_expanded[p*JMAX + q] = 1;
                                }
                            }
                        }
                    }
                }
                
                memcpy(class_map_expanded, temp_expanded, IMAX*JMAX*sizeof(int8_t));
            }
            
            // Throttle debug output (~1Hz)
            static int tracker_log_counter = 0;
            if(++tracker_log_counter >= 15){
                tracker_log_counter = 0;
                int dilated_cells = 0;
                for (int n = 0; n < IMAX*JMAX; n++) {
                    if (class_map_expanded[n] == 1) dilated_cells++;
                }
                // printf("[TRACKER] Clusters: %zu, Tracks: %zu, Labeled: %d, Dilated: %d\n",
                //        clusters.size(), active_tracks.size(), labeled_cells, dilated_cells);
            }
        };

        void state_update_callback(const nav_msgs::msg::Odometry::SharedPtr data){

            // Increment Age of Latest Grid
            dt_state = std::chrono::duration<float>(std::chrono::steady_clock::now() - t_state).count();
            t_state = std::chrono::steady_clock::now();
            grid_age += dt_state;

            // Interpret State from Odometry message
            x[0] = data->pose.pose.position.x;
            x[1] = data->pose.pose.position.y;
            // Extract yaw from quaternion
            auto& q = data->pose.pose.orientation;
            float sin_yaw = 2.0f * (q.w * q.z + q.x * q.y); 
            float cos_yaw = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
            x[2] = std::atan2(sin_yaw, cos_yaw);

            // Feedforward + Feedback Tracking Control
            vn = vt;
            // nominal_controller();
            
            // Safety Filter
            // Prepare Body Frame Velocity for Safety Filter
            std::vector<float> v_input_body = {0.0f, 0.0f, 0.0f};
            
            if(predictive_sf_flag){
                 // vd comes from MPC which is now computed in Body Frame
                 v_input_body = vd;
            }
            else{
                 // vn is Odom Frame: Rotate to Body Frame
                 const float c = std::cos(x[2]);
                 const float s = std::sin(x[2]);
                 v_input_body = {c*vn[0] + s*vn[1], -s*vn[0] + c*vn[1], vn[2]};
            }
            
            // Apply Safety Filter (Output 'v' will be Body Frame)
            if(h_flag) safety_filter(v_input_body); 
            else v = v_input_body;
            
            // Transform to Body-Fixed (Already Body Fixed!)
            const std::vector<float> vb_new = v; 

            // Low Pass Filter
            low_pass(vb, vb_new, 5.0f, dt_state);

            // Check for Failures
            if(std::abs(vb[0])>10.0f || std::abs(vb[1])>10.0f || std::abs(vb[2])>10.0f) sit_flag = true; // Check for Valid Control Action

            // Saturate
            // vb[0] = std::clamp(vb[0], -2.5f, 3.5f);
            // vb[1] = std::clamp(vb[1], -1.0f, 1.0f);
            // vb[2] = std::clamp(vb[2], -4.0f, 4.0f);

            vb[0] = std::clamp(vb[0], -vel_max_x_bwd_, vel_max_x_fwd_);
            vb[1] = std::clamp(vb[1], -vel_max_y_, vel_max_y_);
            vb[2] = std::clamp(vb[2], -vel_max_yaw_, vel_max_yaw_);
            

            //Publish Control Action
            if(stop_flag){
                sport_req.StopMove(req);
                sleep(2);
                sport_req.StandDown(req);
                rclcpp::shutdown();
            }
            else if(sit_flag){
                sport_req.StopMove(req);
                sleep(2);
                sport_req.StandDown(req);
            }
            else if(start_flag){
                sport_req.Move(req, vb[0], vb[1], vb[2]); // Send Command
            }


        };

        std::mutex mpc_mutex;
        MPC3D mpc3d_controller;

        const float h0 = 0.0f; // Set boundary level set value
        const float dh0 = 1.0f; // Set dh Value

        float wn = 1.0f;
        float issf = 50.0f;

        bool h_flag = false;
        bool dhdt_flag = false;

        bool save_flag = false;
        bool start_flag = false;  // Controls robot movement (spacebar)
        bool enable_display = false;  // Controls OpenCV visualization (ROS param)
        bool sit_flag = false;
        bool stop_flag = false;
        bool predictive_sf_flag = false;
        bool realtime_sf_flag = false;
        int space_counter = 0;
        int poisson_save_counter = 0;

        const std::vector<char> sorted_parameter_deck = {'1', '2', '3', '4', '5', '6','0','0'};
        std::random_device rd;
        std::mt19937 gen;
        std::vector<char> current_parameter_deck;
        
        // Define State
        std::vector<float> x = {0.0f, 0.0f, 0.0f};
        std::vector<float> xd = {0.0f, 0.0f, 0.0f};
        std::vector<float> xc = {-2.0f, -2.0f, 0.0f};
        std::vector<float> xc0 = {-2.0f, -2.0f, 0.0f};
        std::vector<float> dx = {0.0f, 0.0f, 0.0f};

        std::chrono::steady_clock::time_point t_grid, t_state, t_start;
        float grid_age = 0.0f;
        float dt_grid = 1.0e10f;
        float dt_state = 1.0e10f;
        float t_ms = 0.0f;

        std::vector<float> vt = {0.0f, 0.0f, 0.0f};
        std::vector<float> vn = {0.0f, 0.0f, 0.0f};
        std::vector<float> vd = {0.0f, 0.0f, 0.0f};
        std::vector<float> v = {0.0f, 0.0f, 0.0f};
        std::vector<float> vb = {0.0f, 0.0f, 0.0f};
        float h, dhdt, dhdx, dhdy, dhdq;
        
        float occ1[IMAX*JMAX];
        float occ0[IMAX*JMAX];
        int8_t conf[IMAX*JMAX];
        float grid_temp[IMAX*JMAX];
        float *hgrid1, *hgrid0, *bound, *force, *dhdt_grid;
        float *robot_kernel_human, *robot_kernel_obstacle;  // Separate kernels for class-aware inflation
        float *guidance_x_grid, *guidance_y_grid;  // Full 3D guidance field for MPC
        float guidance_x_display[IMAX*JMAX];  // Store q=0 slice for visualization
        float guidance_y_display[IMAX*JMAX];  // Store q=0 slice for visualization
        float bound_display[IMAX*JMAX];       // Store q=0 slice of boundary for debug viz
        int8_t tangent_layer_display[IMAX*JMAX];  // Track cells with tangential bias for yellow visualization

        float robot_length, robot_width;
        float robot_MOS_human, robot_MOS_obstacle;  // Separate MOS for humans and obstacles
        int robot_kernel_dim_human, robot_kernel_dim_obstacle;
        
        rclcpp::CallbackGroup::SharedPtr mpc_callback_group_;
        rclcpp::TimerBase::SharedPtr mpc_timer_;
        
        std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> image_sub_;
        std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> cloud_sub_;
        
        rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr key_suber_;
        rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_suber_;
        rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr occ_grid_suber_;
        rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr class_map_suber_;
        rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr visibility_map_suber_;
        rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_suber_;
        
        // Class map for semantic labels (0=free, 1=human, 3=obstacle)
        int8_t class_map[IMAX*JMAX];
        int8_t visibility_map[IMAX*JMAX];  // Camera visibility (1 = visible, 0 = not visible)
        int8_t class_map_expanded[IMAX*JMAX];  // After brushfire expansion
        std::unique_ptr<HumanTracker> human_tracker_;  // Object-level human tracking
        int min_yolo_cells_ = 5;  // Minimum YOLO cells to confirm human cluster
        bool enable_human_tracker_dilation_ = true;  // Enable brushfire dilation of human labels
        float dh0_human = 1.0f;  // Stronger repulsion for humans
        float dh0_obstacle = 0.3f;  // Default obstacle repulsion
        bool enable_social_navigation_ = false;  // Social navigation flag (pass on human's right)
        float social_tangent_bias_ = 0.5f;  // Tangential bias strength for social navigation
        int social_tangent_layers_ = 3;      // Number of outward layers from boundary with tangential bias
        int social_layer_thickness_ = 1;     // Grid cells per layer (circular dilation radius)
        float current_tangent_direction_ = 1.0f;  // Current tangent direction for debug: -1=CCW, +1=CW
        float human_direction_threshold_ = 0.15f; // Closing rate threshold to detect human approaching (m/s)
        std::map<int, std::pair<float, float>> prev_human_distances_;  // track_id -> (prev_distance, prev_time)
        std::vector<std::tuple<int, int, float, float, float>> human_boundary_info_;  // (i, j, gx, gy, local_dh0) - shared between functions
        
        // Tight-area wall softening parameters
        float tight_area_human_threshold_ = 2.0f;  // meters
        float tight_area_h_threshold_ = 0.3f;       // h value threshold
        float tight_area_wall_slack_ = -0.1f;       // slack for wall boundaries
        
        // Dynamic CBF parameters (eq. 31 dhdt scaling)
        float cbf_sigma_epsilon_ = 0.1f;  // sigma(h) saturation
        float cbf_sigma_kappa_ = 5.0f;    // sigma(h) transition rate
        
        // Velocity bound parameters
        float vel_max_x_fwd_ = 0.9f;
        float vel_max_x_bwd_ = 0.9f;
        float vel_max_y_ = 0.9f;
        float vel_max_yaw_ = 0.8f;
        
        rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr req_puber_;
        unitree_api::msg::Request req; // Unitree Go2 ROS2 request message
        SportClient sport_req;

        std::ofstream outFileCSV;
        std::ofstream outFileBIN;
        
        // Image publishing
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr poisson_image_pub_;
        double logging_publish_hz_ = 10.0;
        double logging_publish_period_ = 0.1;
        std::chrono::steady_clock::time_point last_logging_publish_time_;
        
        // Data logging
        bool enable_data_logging_to_file_ = false;
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr logging_data_pub_;

};

int main(int argc, char * argv[]){
    rclcpp::init(argc, argv);
    
    rclcpp::executors::MultiThreadedExecutor executor;
    
    // Create PoissonControllerNode first to read parameters
    auto poissonNode = std::make_shared<PoissonControllerNode>();
    
    // Get min_z/max_z from poisson node's parameters and pass to CloudMergerNode
    poissonNode->declare_parameter("min_z", 0.05);
    poissonNode->declare_parameter("max_z", 0.80);
 
    float min_z = poissonNode->get_parameter("min_z").as_double();
    float max_z = poissonNode->get_parameter("max_z").as_double();
    RCLCPP_INFO(poissonNode->get_logger(), "Passing min_z=%.2f, max_z=%.2f to CloudMergerNode", min_z, max_z);
    
    auto mappingNode = std::make_shared<CloudMergerNode>(min_z, max_z);
    
    executor.add_node(mappingNode);
    executor.add_node(poissonNode);
    
    try{
        executor.spin();
        throw("Terminated");
    }
    catch(const char* msg){
        rclcpp::shutdown();
        std::cout << msg << std::endl;
    }

  return 0;

}
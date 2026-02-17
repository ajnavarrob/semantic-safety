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

#include "kernel.hpp"
#include "poisson.h"
#include "utils.h"
#include "mpc_cbf_3d.h"
#include "cloud_merger.h"
#include <cuda_runtime.h>

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/int32.hpp"
#include "unitree_go/msg/sport_mode_state.hpp"

#include <time.h>
#include "unitree_api/msg/request.hpp"
#include "common/ros2_sport_client.h"

class PoissonControllerNode : public rclcpp::Node{

    public:
        
        PoissonControllerNode() : Node("poisson_control"), sport_req(this){
            
            // Create CSV for Saving
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
            //std::string fileNameBIN = "data_poisson.bin";
            //std::ofstream clearFileBIN(fileNameBIN, std::ios::binary | std::ios::trunc);
            outFileBIN.open(fileNameBIN, std::ios::binary | std::ios::app);

            // Initialize Parameter Deck
            gen.seed(rd());
            current_parameter_deck = sorted_parameter_deck;
            std::shuffle(current_parameter_deck.begin(),current_parameter_deck.end(), gen);

            // Initialize Clocks
            t_start = std::chrono::steady_clock::now();
            t_grid = std::chrono::steady_clock::now();
            t_state = std::chrono::steady_clock::now();

            // Initialize Occupancy Grids
            for(int n = 0; n < IMAX*JMAX; n++){
                occ1[n] = 1.0f;
                occ0[n] = 1.0f; 
                conf[n] = 0;
                grid_temp[n] = 0.0f;
            }

            // Initialize Poisson Grids
            cudaMallocHost((void**)&hgrid1, IMAX*JMAX*QMAX*sizeof(float));
            cudaMallocHost((void**)&hgrid0, IMAX*JMAX*QMAX*sizeof(float));
            cudaMallocHost((void**)&bound, IMAX*JMAX*QMAX*sizeof(float));
            cudaMallocHost((void**)&force, IMAX*JMAX*QMAX*sizeof(float));
            dhdt_grid = (float *)malloc(IMAX*JMAX*QMAX*sizeof(float));
            for(int n=0; n < IMAX*JMAX*QMAX; n++){
                hgrid1[n] = h0;
                hgrid0[n] = h0;
                dhdt_grid[n] = 0.0f;
            }
            Kernel::poissonInit();
            robot_kernel_dim = initialize_robot_kernel(robot_kernel);

            // Initialize QP for MPC Problem
            mpc3d_controller.setup_QP();
            mpc3d_controller.solve();

            // Create Publishers & Subscribers
            rclcpp::SubscriptionOptions options1;
            rclcpp::SubscriptionOptions options2;
            rclcpp::SubscriptionOptions options3;
            options1.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            options2.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            options3.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
            occ_grid_suber_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>("occupancy_grid", 1, std::bind(&PoissonControllerNode::occ_grid_callback, this, std::placeholders::_1), options1);
            pose_suber_ = this->create_subscription<unitree_go::msg::SportModeState>("sportmodestate", 1, std::bind(&PoissonControllerNode::state_update_callback, this, std::placeholders::_1), options2);
            twist_suber_ = this->create_subscription<geometry_msgs::msg::Twist>("u_des", 1, std::bind(&PoissonControllerNode::teleop_callback, this, std::placeholders::_1), options3);
            key_suber_ = this->create_subscription<std_msgs::msg::Int32>("key_press", 1, std::bind(&PoissonControllerNode::keyboard_callback, this, std::placeholders::_1), options3);

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
                    break;
                case 'p':
                    predictive_sf_flag = !predictive_sf_flag;
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

            // Save Data
            if(save_flag){

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
                for(int i=0; i<MAX_SQP_ITERS; i++){
                    mpc3d_controller.update_cost(vn);
                    mpc3d_controller.update_constraints(hgrid1, dhdt_grid, x, xc, grid_age, wn, issf);
                    mpc3d_controller.solve();
                    if(mpc3d_controller.update_residual()<1.0f) break;
                }
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

        /* Find Boundaries (Any Unoccupied Point that Borders an Occupied Point) */
        void find_boundary(float *grid, float *bound, const bool fix_flag){
            
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
                if(fix_flag && !bound[n]) grid[n] = h0;
                //const int i = n/JMAX;
                //const int j = n%JMAX;
                //if(fix_flag && i==0) grid[n] = 1.0f;
                //if(fix_flag && i==(IMAX-1)) grid[n] = 1.0f;
                //if(fix_flag && j==0) grid[n] = 1.0f;
                //if(fix_flag && j==(JMAX-1)) grid[n] = 1.0f;
            }

        };
        
        /* Construct n x n Kernel Using Hyper-Ellipse Parameters */
        int initialize_robot_kernel(float*& kernel){
            
            /* Create Robot Kernel */
            robot_length = 0.74f; // Go2
            robot_width = 0.37f;
            robot_MOS = 1.35f;
            
            const float ar = robot_MOS * robot_length / 2.0f;
            const float br = robot_MOS * robot_width / 2.0f;
            const float D = 2.0f * std::sqrt(ar*ar + br*br); // Max Robot Dimension to Define Kernel Size
            const int dim = 2 * (int)std::ceil(std::ceil(D / DS) / 2.0f); //Make Sure Kernel Dimension is Even

            kernel = (float *)malloc(dim*dim*QMAX*sizeof(float));
            for(int q=0; q<QMAX; q++){
                float *kernel_slice = kernel + q*dim*dim;
                const float yawq = q_to_yaw(q, xc[2]);
                fill_elliptical_robot_kernel(kernel_slice, yawq, dim, 2.0f);
            }

            return dim;

        };

        /* Construct n x n Kernel Using Hyper-Ellipse Parameters */
        void fill_elliptical_robot_kernel(float *kernel, const float yawq, const int dim, const float expo){
            
            const float ar = robot_MOS * robot_length / 2.0f;
            const float br = robot_MOS * robot_width / 2.0f;
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


        /* Buffer Occupancy Grid with 2-D Robot Shape */
        void inflate_occupancy_grid(float *bound, const float *kernel){
            
            // Convolve Robot Kernel with Occupancy Grid, Along the Boundary
            float b0[IMAX*JMAX];
            memcpy(b0, bound, IMAX*JMAX*sizeof(float));

            int lim = (robot_kernel_dim - 1)/2;
            for(int i = 1; i < IMAX-1; i++){
                int ilow = std::max(i - lim, 0);
                int itop = std::min(i + lim, IMAX);
                for(int j = 1; j < JMAX-1; j++){
                    int jlow = std::max(j - lim, 0);
                    int jtop = std::min(j + lim, JMAX);
                    if(!b0[i*JMAX+j]){
                        for(int p = ilow; p < itop; p++){
                            for(int q = jlow; q < jtop; q++){
                                bound[p*JMAX+q] += kernel[(p-i+lim)*robot_kernel_dim+(q-j+lim)];
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


        /* Compute Boundary Gradients for Guidance Field */
        void compute_boundary_gradients(float *guidance_x, float *guidance_y, const float *bound, const int* class_map = nullptr){
            // Set Border Gradients
            for(int i = 0; i < IMAX; i++){
                for(int j = 0; j < JMAX; j++){
                    if(i==0) guidance_x[i*JMAX+j] = dh0;
                    if(j==0) guidance_y[i*JMAX+j] = dh0;
                    if(i==(IMAX-1)) guidance_x[i*JMAX+j] = -dh0;
                    if(j==(JMAX-1)) guidance_y[i*JMAX+j] = -dh0;
                }
            }
            // Set Additional Boundary Gradients
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
                    if(!bound[i*JMAX+j]){
                        const float V = std::sqrt(guidance_x[i*JMAX+j]*guidance_x[i*JMAX+j] + guidance_y[i*JMAX+j]*guidance_y[i*JMAX+j]);
                        if(V != 0.0f){
                            guidance_x[i*JMAX+j] /= V;
                            guidance_y[i*JMAX+j] /= V;
                        }
                        float local_dh0 = dh0;
                        if (class_map) {
                            // Use class-specific dh0 if class_map is provided
                            // For now, just use global dh0 as we don't have multi-class support yet
                        }
                        guidance_x[i*JMAX+j] *= local_dh0;
                        guidance_y[i*JMAX+j] *= local_dh0;
                    }
                }
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
            memcpy(hgrid_temp, hgrid1, IMAX*JMAX*QMAX*sizeof(float));

            // Execute Poisson Pre-Processing
            build_occ_map(occ1, occ0, conf);
            find_boundary(hgrid_temp, occ1, false);

            #pragma omp parallel for num_threads(4)
            for(int q=0; q<QMAX; q++){
                float *force_slice = force + q*IMAX*JMAX;
                float *bound_slice = bound + q*IMAX*JMAX;
                float *hgrid_slice = hgrid_temp + q*IMAX*JMAX;
                float *kernel_slice = robot_kernel + q*robot_kernel_dim*robot_kernel_dim;
                memcpy(bound_slice, occ1, IMAX*JMAX*sizeof(float));
                inflate_occupancy_grid(bound_slice, kernel_slice);
                find_boundary(hgrid_slice, bound_slice, true);
                
                // Allocate guidance field arrays
                float *guidance_x = (float *)calloc(IMAX*JMAX, sizeof(float));
                float *guidance_y = (float *)calloc(IMAX*JMAX, sizeof(float));
                float *f0 = (float *)calloc(IMAX*JMAX, sizeof(float));

                // Step 1: Compute boundary gradients for guidance field
                compute_boundary_gradients(guidance_x, guidance_y, bound_slice, nullptr);

                // Step 2: Solve Poisson equation for guidance fields
                const float v_RelTol = 1.0e-4f;
                const int N_guidance = IMAX/5;
                const float w_SOR_guidance = 2.0f/(1.0f+std::sin(M_PI/(float)(N_guidance+1)));
                int vx_iters = Kernel::poissonSolve(guidance_x, f0, bound_slice, v_RelTol, w_SOR_guidance);
                int vy_iters = Kernel::poissonSolve(guidance_y, f0, bound_slice, v_RelTol, w_SOR_guidance);

                // Step 3: Compute optimal forcing function from guidance field divergence
                compute_optimal_forcing_function(force_slice, guidance_x, guidance_y, bound_slice);
                for(int n=0; n<IMAX*JMAX; n++){
                    force_slice[n] *= DS*DS;
                }

                // Free temporary arrays
                free(guidance_x);
                free(guidance_y);
                free(f0);

            }

            // Solve Poisson's Equation
            const float relTol = 1.0e-4f;
            const int N = IMAX/5;
            const float w_SOR = 2.0f/(1.0f+std::sin(M_PI/(float)(N+1))); // This is the "optimal" value from Strikwerda, Chapter 13.5
            int iters = Kernel::poissonSolve(hgrid_temp, force, bound, relTol, w_SOR); // CUDA!
            
            // Transfer Solutions into Necessary Locations
            memcpy(occ0, occ1, IMAX*JMAX*sizeof(float));
            memcpy(hgrid0, hgrid1, IMAX*JMAX*QMAX*sizeof(float));
            memcpy(hgrid1, hgrid_temp, IMAX*JMAX*QMAX*sizeof(float));
            free(hgrid_temp);
            if(h_flag) dhdt_flag = true;
            
            solve_timer.time("Poisson Solve Time: ");
            printf("Poisson Iterations: %u \n", iters);

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

            // Convert to 8-bit Grayscale
            cv::Mat gray_img;
            cv::normalize(poisson_img, gray_img, 0, 255, cv::NORM_MINMAX);
            gray_img.convertTo(gray_img, CV_8U);

            // Convert to Colormap
            cv::Mat color_img;
            cv::applyColorMap(gray_img, color_img, cv::COLORMAP_HOT);

            // Resize for Display
            cv::Mat resized_img;
            const int upscale = 6;
            cv::resize(color_img, resized_img, cv::Size(), upscale, upscale, cv::INTER_NEAREST);

            // Add Current Location & Goal Location
            cv::Point curr_pt = cv::Point(upscale*x_to_j(x[0],xc[0]),upscale*y_to_i(x[1],xc[1]));
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

            // Vertical Flip Image for Display 
            cv::Mat flipped_img;
            cv::flip(resized_img, flipped_img, 0);

            // Display Final Image
            cv::imshow("Poisson Solution", flipped_img);
            cv::waitKey(1);

            display_timer.time("Display Time: ");

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
            const float ic = y_to_i(x[1], xc[1]);
            const float jc = x_to_j(x[0], xc[0]);
            const float qc = yaw_to_q(x[2], xc[2]);

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
        
            // Compute Current Gradient
            const float i_eps = 5.0f;
            const float j_eps = 5.0f;
            const float q_eps = 1.0f;
            const float ip = std::clamp(ic + i_eps, 0.0f, (float)(IMAX-1));
            const float im = std::clamp(ic - i_eps, 0.0f, (float)(IMAX-1));
            const float jp = std::clamp(jc + j_eps, 0.0f, (float)(JMAX-1));
            const float jm = std::clamp(jc - j_eps, 0.0f, (float)(JMAX-1));
            const float qp = q_wrap(qc + q_eps);
            const float qm = q_wrap(qc - q_eps);
            float hxp = trilinear_interpolation(hgrid1, ic, jp, qc);
            float hxm = trilinear_interpolation(hgrid1, ic, jm, qc);
            float hyp = trilinear_interpolation(hgrid1, ip, jc, qc);
            float hym = trilinear_interpolation(hgrid1, im, jc, qc);
            float hqp = trilinear_interpolation(hgrid1, ic, jc, qp);
            float hqm = trilinear_interpolation(hgrid1, ic, jc, qm);
            dhdx = (hxp-hxm) / (2.0f*j_eps*DS);
            dhdy = (hyp-hym) / (2.0f*i_eps*DS);
            dhdq = (hqp-hqm) / (2.0f*q_eps*DQ);

            // Forward Propogate Gradient
            hxp += trilinear_interpolation(dhdt_grid, ic, jp, qc) * grid_age;
            hxm += trilinear_interpolation(dhdt_grid, ic, jm, qc) * grid_age;
            hyp += trilinear_interpolation(dhdt_grid, ip, jc, qc) * grid_age;
            hym += trilinear_interpolation(dhdt_grid, im, jc, qc) * grid_age;
            hqp += trilinear_interpolation(dhdt_grid, ic, jc, qp) * grid_age;
            hqm += trilinear_interpolation(dhdt_grid, ic, jc, qm) * grid_age;
            const float dhdx_pred = (hxp-hxm) / (2.0f*j_eps*DS);
            const float dhdy_pred = (hyp-hym) / (2.0f*i_eps*DS);
            const float dhdq_pred = (hqp-hqm) / (2.0f*q_eps*DQ);
            
            // Single Integrator Safety Filter
            const float Pu[3] = {2.0f, 2.0f, 1.0f};
            const float ISSf1 = issf;
            const float ISSf2 = issf;
            const float b = dhdx_pred*dhdx_pred/Pu[0] + dhdy_pred*dhdy_pred/Pu[1] + dhdq_pred*dhdq_pred/Pu[2];
            float ISSf = std::sqrt(b) / ISSf1 + b / ISSf2;
            const float sigma = std::clamp(-10.0f*dhdt, 0.0f, 1.0f);
            ISSf *= sigma;
            float a = wn*h_pred;
            a += dhdt; // Dynamic Environment
            a += dhdx_pred*vd[0] + dhdy_pred*vd[1] + dhdq_pred*vd[2]; // Min Norm Controller
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
            if(start_flag && dhdt_flag) update_dhdt_grid();

            // Display Results
            if(start_flag) display_poisson_safety_function();
            std::cout << "Grid Loop Time: " << dt_grid*1.0e3f << " ms" << std::endl;
            std::cout << "Control Loop Time: " << dt_state*1.0e3f << " ms" << std::endl;
            std::cout << "Command: <" << vb[0] << "," << vb[1] << "," << vb[2] << ">" << std::endl;

        };

        void state_update_callback(const unitree_go::msg::SportModeState::SharedPtr data){

            // Increment Age of Latest Grid
            dt_state = std::chrono::duration<float>(std::chrono::steady_clock::now() - t_state).count();
            t_state = std::chrono::steady_clock::now();
            grid_age += dt_state;

            // Interpret State
            x[0] = data->position[0];
            x[1] = data->position[1];
            float sin_yaw = 2.0f * (data->imu_state.quaternion[0] * data->imu_state.quaternion[3]); 
            float cos_yaw = 1.0f - 2.0f * data->imu_state.quaternion[3] * data->imu_state.quaternion[3];
            x[2] = std::atan2(sin_yaw, cos_yaw);

            // Feedforward + Feedback Tracking Control
            vn = vt;
            // nominal_controller();
            
            // Safety Filter
            if(predictive_sf_flag){
                if(h_flag) safety_filter(vd); // Apply Safety Filter
            }
            else{
                if(h_flag) safety_filter(vn); // Apply Safety Filter
            }
            
            // Transform to Body-Fixed
            const std::vector<float> vb_new = {std::cos(x[2])*v[0] + std::sin(x[2])*v[1],
                                              -std::sin(x[2])*v[0] + std::cos(x[2])*v[1],
                                               v[2]};

            // Low Pass Filter
            low_pass(vb, vb_new, 5.0f, dt_state);

            // Check for Failures
            if(std::abs(vb[0])>10.0f || std::abs(vb[1])>10.0f || std::abs(vb[2])>10.0f) sit_flag = true; // Check for Valid Control Action

            // Saturate
            vb[0] = std::clamp(vb[0], -2.5f, 3.5f);
            vb[1] = std::clamp(vb[1], -1.0f, 1.0f);
            vb[2] = std::clamp(vb[2], -4.0f, 4.0f);

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
        float issf = 5.0f;

        bool h_flag = false;
        bool dhdt_flag = false;

        bool save_flag = false;
        bool start_flag = false;
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
        float *hgrid1, *hgrid0, *bound, *force, *robot_kernel, *dhdt_grid;

        float robot_length, robot_width, robot_MOS;
        int robot_kernel_dim;
        
        rclcpp::CallbackGroup::SharedPtr mpc_callback_group_;
        rclcpp::TimerBase::SharedPtr mpc_timer_;
        
        rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr key_suber_;
        rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_suber_;
        rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr occ_grid_suber_;
        rclcpp::Subscription<unitree_go::msg::SportModeState>::SharedPtr pose_suber_;
        
        rclcpp::Publisher<unitree_api::msg::Request>::SharedPtr req_puber_;
        unitree_api::msg::Request req; // Unitree Go2 ROS2 request message
        SportClient sport_req;

        std::ofstream outFileCSV;
        std::ofstream outFileBIN;

};

int main(int argc, char * argv[]){

    rclcpp::init(argc, argv);
    rclcpp::executors::MultiThreadedExecutor executor;
    auto mappingNode = std::make_shared<CloudMergerNode>();
    auto poissonNode = std::make_shared<PoissonControllerNode>();
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
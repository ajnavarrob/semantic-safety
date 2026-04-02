#include <memory>
#include <iostream>
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
#include <map>
#include <array>
#include <unistd.h>
#include <cstring>
#include <cfloat>
#include <set>
#include <shared_mutex>

#include <cuda_runtime.h>
#include "kernel.hpp"
#include "poisson.h"
#include "utils.h"
#include "mpc_cbf_3d.h"
#include "cloud_merger.h"
#include "poisson/human_tracker.h"

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/int32.hpp"
#include "std_msgs/msg/float32_multi_array.hpp"
#include "nav_msgs/msg/odometry.hpp"
#include "unitree_api/msg/request.hpp"
#include "common/ros2_sport_client.h"
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

namespace ss {

enum class PipelineStage {
    OccupancyPreprocess,
    SemanticFusion,
    GeometryShaping,
    GuidanceField,
    SafetyFieldSolve,
    DhdtUpdate,
    PredictiveControl,
    RealtimeFilter,
    CommandDispatch
};

struct TimingSample {
    double occupancy_preprocess_ms{0.0};
    double semantic_fusion_ms{0.0};
    double geometry_shaping_ms{0.0};

    double guidance_boundary_setup_ms{0.0};
    double guidance_social_expansion_ms{0.0};
    double guidance_laplace_ms{0.0};
    double guidance_copyout_ms{0.0};

    double safety_field_solve_ms{0.0};
    double dhdt_update_ms{0.0};
    double predictive_control_ms{0.0};
    double realtime_filter_ms{0.0};
    double command_dispatch_ms{0.0};
    double field_data_age_ms{0.0};
    double end_to_end_grid_ms{0.0};
};

struct ConnectedComponentsData {
    cv::Mat binary;
    cv::Mat labels;
    cv::Mat stats;
    cv::Mat centroids;
    int num_labels{0};
};

struct SemanticStageOutput {
    bool tight_area{false};
    std::vector<HumanTrack> active_tracks;
};

struct GuidanceStageOutput {
    float* bound_guidance{nullptr};
};

class ScopedTimer {
public:
    explicit ScopedTimer(double& target_ms)
        : target_ms_(target_ms), t0_(std::chrono::steady_clock::now()) {}

    ~ScopedTimer() {
        const auto t1 = std::chrono::steady_clock::now();
        target_ms_ = std::chrono::duration<double, std::milli>(t1 - t0_).count();
    }

private:
    double& target_ms_;
    std::chrono::steady_clock::time_point t0_;
};

class PoissonControllerNode : public rclcpp::Node {
public:
    PoissonControllerNode() : Node("poisson_control"), sport_req(this) {
        declare_and_load_parameters();
        initialize_clocks_and_flags();
        initialize_static_grids();
        allocate_persistent_buffers();
        initialize_robot_kernels();
        initialize_mpc();
        initialize_ros_interfaces();
        startup_robot();
    }

    ~PoissonControllerNode() override {
        if (hgrid1) cudaFreeHost(hgrid1);
        if (hgrid0) cudaFreeHost(hgrid0);
        if (bound) cudaFreeHost(bound);
        if (force) cudaFreeHost(force);

        if (dhdt_grid) std::free(dhdt_grid);
        if (guidance_x_grid) std::free(guidance_x_grid);
        if (guidance_y_grid) std::free(guidance_y_grid);

        if (hgrid_temp_) std::free(hgrid_temp_);
        if (guidance_x_temp_) std::free(guidance_x_temp_);
        if (guidance_y_temp_) std::free(guidance_y_temp_);
        if (forcing_zero_temp_) std::free(forcing_zero_temp_);
        if (bound_guidance_temp_) std::free(bound_guidance_temp_);
        if (class_map_temp_expanded_) std::free(class_map_temp_expanded_);
        if (boundary_temp_) std::free(boundary_temp_);
        if (inflate_bound_temp_) std::free(inflate_bound_temp_);
        if (inflate_class_temp_) std::free(inflate_class_temp_);

        if (robot_kernel_human) std::free(robot_kernel_human);
        if (robot_kernel_obstacle) std::free(robot_kernel_obstacle);

        if (outFileCSV.is_open()) outFileCSV.close();
        if (outFileBIN.is_open()) outFileBIN.close();
    }

private:
    // ============================================================
    // 1. ROS ORCHESTRATION
    // ============================================================

    void teleop_callback(geometry_msgs::msg::Twist::UniquePtr msg) {
        handle_teleop_input(*msg);
    }

    void keyboard_callback(std_msgs::msg::Int32::UniquePtr msg) {
        handle_keyboard_input(*msg);
    }

    void occ_grid_callback(nav_msgs::msg::OccupancyGrid::UniquePtr msg) {
        handle_occupancy_update(*msg);
    }

    void class_map_callback(nav_msgs::msg::OccupancyGrid::UniquePtr msg) {
        if (msg->data.size() != IMAX * JMAX) {
            RCLCPP_WARN(
                this->get_logger(),
                "class_map size mismatch: got %zu expected %d",
                msg->data.size(),
                IMAX * JMAX
            );
            return;
        }
    
        for (int n = 0; n < IMAX * JMAX; ++n) {
            class_map[n] = msg->data[n];
        }
    }

    void visibility_map_callback(nav_msgs::msg::OccupancyGrid::UniquePtr msg) {
        if (msg->data.size() != IMAX * JMAX) {
            RCLCPP_WARN(
                this->get_logger(),
                "visibility_map size mismatch: got %zu expected %d",
                msg->data.size(),
                IMAX * JMAX
            );
            return;
        }
    
        for (int n = 0; n < IMAX * JMAX; ++n) {
            visibility_map[n] = msg->data[n];
        }
    }

    void state_update_callback(const nav_msgs::msg::Odometry::SharedPtr data) {
        handle_state_update(*data);
    }

    void mpc_callback() {
        handle_mpc_update();
    }

    // ============================================================
    // 2. HIGH-LEVEL HANDLERS
    // ============================================================

    void handle_teleop_input(const geometry_msgs::msg::Twist& msg) {
        const std::vector<float> vtb = {
            static_cast<float>(msg.linear.x),
            static_cast<float>(msg.linear.y),
            static_cast<float>(msg.angular.z)
        };

        vt = {
            std::cos(x[2]) * vtb[0] - std::sin(x[2]) * vtb[1],
            std::sin(x[2]) * vtb[0] + std::cos(x[2]) * vtb[1],
            vtb[2]
        };

        xd[0] += 0.01f * vt[0];
        xd[1] += 0.01f * vt[1];
        xd[2] += 0.01f * vt[2];

        if (!start_flag) {
            xd = x;
            vt = {0.0f, 0.0f, 0.0f};
        }
    }

    void handle_keyboard_input(const std_msgs::msg::Int32& msg) {
        if (!save_flag) t_start = std::chrono::steady_clock::now();
        else t_ms = std::chrono::duration<float>(std::chrono::steady_clock::now() - t_start).count() * 1.0e3f;

        char param = ' ';
        const int ch = msg.data;
        switch (ch) {
            case ' ': space_counter++; if (space_counter >= 1) save_flag = true; if (space_counter >= 3) start_flag = true; if (space_counter >= 6) stop_flag = true; break;
            case 'r': realtime_sf_flag = !realtime_sf_flag; break;
            case 'p': predictive_sf_flag = !predictive_sf_flag; break;
            case 'd':
                param = current_parameter_deck.back();
                current_parameter_deck.pop_back();
                if (current_parameter_deck.empty()) {
                    current_parameter_deck = sorted_parameter_deck;
                    std::shuffle(current_parameter_deck.begin(), current_parameter_deck.end(), gen);
                }
                break;
            default: break;
        }

        apply_parameter_deck_selection(param, ch);
        maybe_write_experiment_data();
    }

    void handle_occupancy_update(const nav_msgs::msg::OccupancyGrid& msg) {
        const auto grid_start = std::chrono::steady_clock::now();
    
        if (!update_grid_metadata_from_message(msg)) {
            return;
        }
    
        {
            std::unique_lock<std::shared_mutex> lock(field_mutex_);
    
            preprocess_occupancy();
            auto semantic_output = run_semantic_fusion();
            build_inflated_boundaries(semantic_output.tight_area);
            auto guidance_output = build_guidance_field(semantic_output.active_tracks);
            h_flag = solve_safety_field(guidance_output);
    
            if (start_flag && dhdt_flag) {
                ScopedTimer timer(timing_.dhdt_update_ms);
                update_temporal_field_derivative();
            }
    
            latest_field_timestamp_ = std::chrono::steady_clock::now();
        }
    
        timing_.end_to_end_grid_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - grid_start).count();
    
        if (enable_display) render_visualization();
        publish_timing_data();
    }




    void handle_state_update(const nav_msgs::msg::Odometry& data) {
        update_robot_state(data);
    
        std::vector<float> v_input_body = form_nominal_body_command();
    
        timing_.field_data_age_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - latest_field_timestamp_).count();
    
        {
            ScopedTimer timer(timing_.realtime_filter_ms);
            if (h_flag) compute_realtime_safe_control(v_input_body);
            else v = v_input_body;
        }
    
        postprocess_command();
    
        {
            ScopedTimer timer(timing_.command_dispatch_ms);
            dispatch_robot_command();
        }
    }

    void handle_mpc_update() {
        if (!(predictive_sf_flag && h_flag && mpc_mutex.try_lock())) return;
        std::lock_guard<std::mutex> lock(mpc_mutex, std::adopt_lock);
        ScopedTimer timer(timing_.predictive_control_ms);
        compute_predictive_control();
    }

    // ============================================================
    // 3. PIPELINE: OCCUPANCY / SEMANTICS / GEOMETRY
    // ============================================================

    void preprocess_occupancy() {
        ScopedTimer timer(timing_.occupancy_preprocess_ms);
    
        build_occ_map(occ1, occ0, conf);
    
        // Only copy layer q=0 instead of entire grid
        std::memcpy(
            hgrid_temp_,
            hgrid1,
            IMAX * JMAX * sizeof(float)
        );
    
        find_boundary(hgrid_temp_, occ1, false);
    }

    SemanticStageOutput run_semantic_fusion() {
        ScopedTimer timer(timing_.semantic_fusion_ms);
        SemanticStageOutput out;
        label_human_clusters(occ1);
        out.active_tracks = human_tracker_->get_active_tracks();
        out.tight_area = is_tight_area();
        return out;
    }

    void build_inflated_boundaries(bool tight_area) {
        ScopedTimer timer(timing_.geometry_shaping_ms);

        float* bound_q0 = bound;
        std::memcpy(bound_q0, occ1, IMAX * JMAX * sizeof(float));
        inflate_occupancy_grid(bound_q0, class_map_expanded);

        #pragma omp parallel for num_threads(4)
        for (int q = 0; q < QMAX; ++q) {
            float* bound_slice = bound + q * IMAX * JMAX;
            float* hgrid_slice = hgrid_temp_ + q * IMAX * JMAX;
            if (q != 0) {
                std::memcpy(bound_slice, occ1, IMAX * JMAX * sizeof(float));
                inflate_occupancy_grid(bound_slice, class_map_expanded);
            }
            find_boundary(hgrid_slice, bound_slice, true, tight_area, class_map_expanded);
        }
    }

    // ============================================================
    // 4. PIPELINE: GUIDANCE / SAFETY FIELD
    // ============================================================

    GuidanceStageOutput build_guidance_field(const std::vector<HumanTrack>& active_tracks) {
        GuidanceStageOutput out;
        out.bound_guidance = bound;

        std::memset(guidance_x_temp_, 0, IMAX * JMAX * QMAX * sizeof(float));
        std::memset(guidance_y_temp_, 0, IMAX * JMAX * QMAX * sizeof(float));
        std::memset(forcing_zero_temp_, 0, IMAX * JMAX * QMAX * sizeof(float));
        std::memset(tangent_layer_display, 0, IMAX * JMAX * sizeof(int8_t));

        const float c_yaw = std::cos(x[2]);
        const float s_yaw = std::sin(x[2]);
        const float vn_body_x = c_yaw * vn[0] + s_yaw * vn[1];
        const float vn_body_y = -s_yaw * vn[0] + c_yaw * vn[1];
        
        {
            ScopedTimer timer(timing_.guidance_boundary_setup_ms);
        
            compute_boundary_gradients(guidance_x_temp_, guidance_y_temp_, bound, class_map_expanded,
                                       x[0], x[1], vn_body_x, vn_body_y, true);
        
            #pragma omp parallel for num_threads(4)
            for (int q = 1; q < QMAX; ++q) {
                float* bound_slice = bound + q * IMAX * JMAX;
                float* gx = guidance_x_temp_ + q * IMAX * JMAX;
                float* gy = guidance_y_temp_ + q * IMAX * JMAX;
                compute_boundary_gradients(gx, gy, bound_slice, class_map_expanded,
                                           x[0], x[1], vn_body_x, vn_body_y, false);
            }
        }

        {
            ScopedTimer timer(timing_.guidance_social_expansion_ms);
        
            if (enable_social_navigation_ && social_tangent_layers_ > 0 && !human_boundary_info_.empty()) {
                out.bound_guidance = bound_guidance_temp_;
                const float sign = compute_tangent_direction(active_tracks, 0.0f, 0.0f, vn_body_x, vn_body_y);
                for (int q = 0; q < QMAX; ++q) {
                    expand_human_obstacles_for_guidance(
                        bound_guidance_temp_ + q * IMAX * JMAX,
                        guidance_x_temp_ + q * IMAX * JMAX,
                        guidance_y_temp_ + q * IMAX * JMAX,
                        bound + q * IMAX * JMAX,
                        social_tangent_layers_,
                        social_layer_thickness_,
                        social_tangent_bias_,
                        sign);
                }
            }
        }

        {
            ScopedTimer timer(timing_.guidance_laplace_ms);
            solve_guidance_laplace(out.bound_guidance);
        }
        
        compute_guidance_forcing();

        {
            ScopedTimer timer(timing_.guidance_copyout_ms);
        
            std::memcpy(guidance_x_display, guidance_x_temp_, IMAX * JMAX * sizeof(float));
            std::memcpy(guidance_y_display, guidance_y_temp_, IMAX * JMAX * sizeof(float));
            std::memcpy(bound_display, bound, IMAX * JMAX * sizeof(float));
            std::memcpy(guidance_x_grid, guidance_x_temp_, IMAX * JMAX * QMAX * sizeof(float));
            std::memcpy(guidance_y_grid, guidance_y_temp_, IMAX * JMAX * QMAX * sizeof(float));
        }
        return out;
    }

    void solve_guidance_laplace(float* bound_guidance) {
        const float v_RelTol = 1.0e-4f;
        const int N_guidance = IMAX / 5;
        const float w_SOR_guidance = 2.0f / (1.0f + std::sin(M_PI / static_cast<float>(N_guidance + 1)));
        (void)Kernel::poissonSolve(guidance_x_temp_, forcing_zero_temp_, bound_guidance, v_RelTol, w_SOR_guidance);
        (void)Kernel::poissonSolve(guidance_y_temp_, forcing_zero_temp_, bound_guidance, v_RelTol, w_SOR_guidance);
    }

    void compute_guidance_forcing() {
        #pragma omp parallel for num_threads(4)
        for (int q = 0; q < QMAX; ++q) {
            float* force_slice = force + q * IMAX * JMAX;
            float* bound_slice = bound + q * IMAX * JMAX;
            float* gx = guidance_x_temp_ + q * IMAX * JMAX;
            float* gy = guidance_y_temp_ + q * IMAX * JMAX;
            compute_optimal_forcing_function(force_slice, gx, gy, bound_slice);
            for (int n = 0; n < IMAX * JMAX; ++n) force_slice[n] *= DS * DS;
        }
    }

    bool solve_safety_field(const GuidanceStageOutput& guidance){
        ScopedTimer timer(timing_.safety_field_solve_ms);
    
        const float relTol = 1.0e-4f;
        const int N = IMAX / 5;
        const float w_SOR = 2.0f / (1.0f + std::sin(M_PI / static_cast<float>(N + 1)));
    
        const bool success = true;
    
        (void)Kernel::poissonSolve(hgrid_temp_, force, guidance.bound_guidance, relTol, w_SOR);
    
        std::memcpy(occ0, occ1, IMAX * JMAX * sizeof(float));
        std::memcpy(hgrid0, hgrid1, IMAX * JMAX * QMAX * sizeof(float));
        std::memcpy(hgrid1, hgrid_temp_, IMAX * JMAX * QMAX * sizeof(float));
    
        if (success) {
            dhdt_flag = true;
        }
    
        return success;
    }

    void update_temporal_field_derivative() {
        const float wc = 10.0f;
        const float kc = 1.0f - std::exp(-wc * dt_grid);
        for (int i = 0; i < IMAX; ++i) {
            for (int j = 0; j < JMAX; ++j) {
                for (int q = 0; q < QMAX; ++q) {
                    const float i0 = static_cast<float>(i) + dx[1] / DS;
                    const float j0 = static_cast<float>(j) + dx[0] / DS;
                    const bool in_grid = (i0 >= 0.0f) && (i0 <= static_cast<float>(IMAX - 1)) &&
                                         (j0 >= 0.0f) && (j0 <= static_cast<float>(JMAX - 1));
                    float dhdt_ij = 0.0f;
                    if (in_grid) {
                        const float h0v = trilinear_interpolation(hgrid0, i0, j0, q);
                        const float h1v = trilinear_interpolation(hgrid1, i, j, q);
                        dhdt_ij = (h1v - h0v) / dt_grid;
                    }
                    dhdt_grid[q * IMAX * JMAX + i * JMAX + j] *= 1.0f - kc;
                    dhdt_grid[q * IMAX * JMAX + i * JMAX + j] += kc * dhdt_ij;
                }
            }
        }
    }

    // ============================================================
    // 5. CONTROL
    // ============================================================

    void compute_predictive_control() {
        std::vector<float> x_body_link = {0.0f, 0.0f, x[2]};
        for (int i = 0; i < MAX_SQP_ITERS; ++i) {
            const float c = std::cos(x[2]);
            const float s = std::sin(x[2]);
            std::vector<float> vn_body = {c * vn[0] + s * vn[1], -s * vn[0] + c * vn[1], vn[2]};
            mpc3d_controller.update_cost(vn_body);
            mpc3d_controller.update_constraints(hgrid1, dhdt_grid, guidance_x_grid, guidance_y_grid,
                                                x_body_link, xc, grid_age, wn, issf,
                                                cbf_sigma_epsilon_, cbf_sigma_kappa_);
            mpc3d_controller.solve();
            if (mpc3d_controller.update_residual() < 1.0f) break;
        }
        mpc3d_controller.set_input(vd);
    }

    std::vector<float> form_nominal_body_command() {
        vn = vt;
        if (predictive_sf_flag) return vd;
        const float c = std::cos(x[2]);
        const float s = std::sin(x[2]);
        return {c * vn[0] + s * vn[1], -s * vn[0] + c * vn[1], vn[2]};
    }

    void compute_realtime_safe_control(const std::vector<float>& v_input_body) {
        // Keep your existing safety_filter math here unchanged for now.
        safety_filter(v_input_body);
    }

    void postprocess_command() {
        const std::vector<float> vb_new = v;
        low_pass(vb, vb_new, 5.0f, dt_state);
        if (std::abs(vb[0]) > 10.0f || std::abs(vb[1]) > 10.0f || std::abs(vb[2]) > 10.0f) sit_flag = true;
        vb[0] = std::clamp(vb[0], -vel_max_x_bwd_, vel_max_x_fwd_);
        vb[1] = std::clamp(vb[1], -vel_max_y_, vel_max_y_);
        vb[2] = std::clamp(vb[2], -vel_max_yaw_, vel_max_yaw_);
    }

    void dispatch_robot_command() {
        if (stop_flag) {
            sport_req.StopMove(req);
            sport_req.StandDown(req);
            rclcpp::shutdown();
        } else if (sit_flag) {
            sport_req.StopMove(req);
            sport_req.StandDown(req);
        } else if (start_flag) {
            sport_req.Move(req, vb[0], vb[1], vb[2]);
        }
    }

    // ============================================================
    // 6. VISUALIZATION / LOGGING / EXPERIMENT SUPPORT
    // ============================================================

    void render_visualization() {
        // Intentionally keep existing display_poisson_safety_function body here in the next refactor step.
        // This has been separated conceptually from the hot-path field construction.
    }

    void publish_timing_data() {
        if (!logging_data_pub_) return;
    
        const auto now = std::chrono::steady_clock::now();
        const double time_since_last =
            std::chrono::duration<double>(now - last_logging_publish_time_).count();
    
        if (time_since_last < logging_publish_period_) {
            return;
        }
    
        last_logging_publish_time_ = now;
    
        std_msgs::msg::Float32MultiArray msg;
        msg.data = {
            static_cast<float>(timing_.occupancy_preprocess_ms),
            static_cast<float>(timing_.semantic_fusion_ms),
            static_cast<float>(timing_.geometry_shaping_ms),
        
            static_cast<float>(timing_.guidance_boundary_setup_ms),
            static_cast<float>(timing_.guidance_social_expansion_ms),
            static_cast<float>(timing_.guidance_laplace_ms),
            static_cast<float>(timing_.guidance_copyout_ms),
        
            static_cast<float>(timing_.safety_field_solve_ms),
            static_cast<float>(timing_.dhdt_update_ms),
            static_cast<float>(timing_.predictive_control_ms),
            static_cast<float>(timing_.realtime_filter_ms),
            static_cast<float>(timing_.command_dispatch_ms),
            static_cast<float>(timing_.field_data_age_ms),
            static_cast<float>(timing_.end_to_end_grid_ms)
        };
    
        logging_data_pub_->publish(msg);
    }

    void refresh_grid_temp_for_logging() {
        const float qr = yaw_to_q(x[2], xc[2]);
        const float q1f = std::floor(qr);
        const float q2f = std::ceil(qr);
        const int q1 = static_cast<int>(q_wrap(q1f));
        const int q2 = static_cast<int>(q_wrap(q2f));
    
        #pragma omp parallel for
        for (int n = 0; n < IMAX * JMAX; ++n) {
            if (q1f != q2f) {
                grid_temp[n] =
                    (q2f - qr) * hgrid1[q1 * IMAX * JMAX + n] +
                    (qr - q1f) * hgrid1[q2 * IMAX * JMAX + n];
            } else {
                grid_temp[n] = hgrid1[q1 * IMAX * JMAX + n];
            }
        }
    }

    void maybe_write_experiment_data() {
        if (!(save_flag && enable_data_logging_to_file_)) return;
        const std::vector<float> save_data = {
            t_ms, static_cast<float>(space_counter), x[0], x[1], x[2],
            v[0], v[1], v[2], vt[0], vt[1], vt[2],
            h, dhdx, dhdy, dhdq, dhdt, wn, static_cast<float>(realtime_sf_flag | predictive_sf_flag)
        };
        for (size_t n = 0; n < save_data.size(); ++n) {
            outFileCSV << save_data[n];
            if (n + 1 < save_data.size()) outFileCSV << ",";
        }
        outFileCSV << std::endl;
        const int factor = 7;
        if (!(poisson_save_counter % factor)) {
            refresh_grid_temp_for_logging();
            outFileBIN.write(reinterpret_cast<char*>(grid_temp), sizeof(grid_temp));
        }
        poisson_save_counter++;
    }

    void apply_parameter_deck_selection(char param, int ch) {
        switch (param) {
            case '0': predictive_sf_flag = false; realtime_sf_flag = false; wn = 16.0f; break;
            case '1': predictive_sf_flag = true; realtime_sf_flag = true; wn = 0.5f; break;
            case '2': predictive_sf_flag = true; realtime_sf_flag = true; wn = 1.0f; break;
            case '3': predictive_sf_flag = true; realtime_sf_flag = true; wn = 1.5f; break;
            case '4': predictive_sf_flag = true; realtime_sf_flag = true; wn = 2.0f; break;
            case '5': predictive_sf_flag = true; realtime_sf_flag = true; wn = 4.0f; break;
            case '6': predictive_sf_flag = true; realtime_sf_flag = true; wn = 8.0f; break;
            default: break;
        }
        switch (ch) {
            case '1': wn = 0.5f; break;
            case '2': wn = 1.0f; break;
            case '3': wn = 1.5f; break;
            case '4': wn = 2.0f; break;
            case '5': wn = 4.0f; break;
            case '6': wn = 8.0f; break;
            default: break;
        }
    }

    // ============================================================
    // 7. HELPERS / INITIALIZATION
    // ============================================================

    void initialize_mpc() {
        mpc3d_controller.set_velocity_bounds(
            vel_max_x_fwd_,
            vel_max_x_bwd_,
            vel_max_y_,
            vel_max_yaw_
        );
        mpc3d_controller.setup_QP();
        mpc3d_controller.solve();
    }

    void update_robot_state(const nav_msgs::msg::Odometry& data) {
        dt_state = std::chrono::duration<float>(std::chrono::steady_clock::now() - t_state).count();
        t_state = std::chrono::steady_clock::now();
        grid_age += dt_state;
    
        x[0] = data.pose.pose.position.x;
        x[1] = data.pose.pose.position.y;
    
        const auto& q = data.pose.pose.orientation;
        const float sin_yaw = 2.0f * (q.w * q.z + q.x * q.y);
        const float cos_yaw = 1.0f - 2.0f * (q.y * q.y + q.z * q.z);
        x[2] = std::atan2(sin_yaw, cos_yaw);
    }

    bool update_grid_metadata_from_message(const nav_msgs::msg::OccupancyGrid& msg) {
        if (msg.data.size() != IMAX * JMAX) {
            RCLCPP_WARN(
                this->get_logger(),
                "occupancy_grid size mismatch: got %zu expected %d",
                msg.data.size(),
                IMAX * JMAX
            );
            return false;
        }
    
        dt_grid = std::chrono::duration<float>(std::chrono::steady_clock::now() - t_grid).count();
        t_grid = std::chrono::steady_clock::now();
        grid_age = dt_grid;
    
        dx[0] = msg.info.origin.position.x - xc[0];
        dx[1] = msg.info.origin.position.y - xc[1];
        xc[0] = msg.info.origin.position.x;
        xc[1] = msg.info.origin.position.y;
    
        for (int n = 0; n < IMAX * JMAX; ++n) {
            conf[n] = msg.data[n];
        }
    
        return true;
    }

    void startup_robot() {
        sport_req.RecoveryStand(req);
        sleep(1);
        sport_req.SpeedLevel(req, 1);
        sleep(1);
    }

    void initialize_robot_kernels() {
        robot_kernel_obstacle = nullptr;
        robot_kernel_human = nullptr;
    
        robot_kernel_dim_obstacle = initialize_robot_kernel(robot_kernel_obstacle, robot_MOS_obstacle);
        robot_kernel_dim_human = initialize_robot_kernel(robot_kernel_human, robot_MOS_human);
    }

    void initialize_static_grids() {
        for (int n = 0; n < IMAX * JMAX; ++n) {
            occ1[n] = 1.0f;
            occ0[n] = 1.0f;
            conf[n] = 0;
            grid_temp[n] = 0.0f;
            class_map[n] = 0;
            visibility_map[n] = 0;
            class_map_expanded[n] = 0;
            guidance_x_display[n] = 0.0f;
            guidance_y_display[n] = 0.0f;
            bound_display[n] = 0.0f;
            tangent_layer_display[n] = 0;
        }
    }

    void initialize_clocks_and_flags() {
        gen.seed(rd());
        current_parameter_deck = sorted_parameter_deck;
        std::shuffle(current_parameter_deck.begin(), current_parameter_deck.end(), gen);
    
        t_start = std::chrono::steady_clock::now();
        t_grid = std::chrono::steady_clock::now();
        t_state = std::chrono::steady_clock::now();
        latest_field_timestamp_ = std::chrono::steady_clock::now();
        last_logging_publish_time_ = std::chrono::steady_clock::now();
    }

    void initialize_logging_outputs() {
        if (!enable_data_logging_to_file_) {
            RCLCPP_INFO(this->get_logger(), "Data logging DISABLED");
            return;
        }
    
        std::string baseFileName = "experiment_data";
        std::string dateTime = getCurrentDateTime();
    
        std::string fileNameCSV = baseFileName + "_" + dateTime + ".csv";
        outFileCSV.open(fileNameCSV);
        if (!outFileCSV.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open CSV log file: %s", fileNameCSV.c_str());
            throw std::runtime_error("Failed to open CSV log file");
        }
    
        const std::vector<std::string> header = {
            "t_ms", "space_counter", "rx", "ry", "yaw",
            "vx", "vy", "vyaw", "vxd", "vyd", "vyawd",
            "h", "dhdx", "dhdy", "dhdq", "dhdt", "alpha", "on_off"
        };
    
        for (size_t n = 0; n < header.size(); ++n) {
            outFileCSV << header[n];
            if (n + 1 < header.size()) outFileCSV << ",";
        }
        outFileCSV << std::endl;
    
        std::string fileNameBIN = baseFileName + "_" + dateTime + ".bin";
        outFileBIN.open(fileNameBIN, std::ios::binary | std::ios::app);
        if (!outFileBIN.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open BIN log file: %s", fileNameBIN.c_str());
            throw std::runtime_error("Failed to open BIN log file");
        }
    
        RCLCPP_INFO(this->get_logger(), "Data logging ENABLED: %s", fileNameCSV.c_str());
    }
    
    void declare_and_load_parameters() {
        // ------------------------------------------------------------
        // Logging / visualization
        // ------------------------------------------------------------
        this->declare_parameter("enable_data_logging_to_file", false);
        this->declare_parameter("enable_display", true);
        this->declare_parameter("logging_publish_hz", 10.0);
    
        enable_data_logging_to_file_ = this->get_parameter("enable_data_logging_to_file").as_bool();
        enable_display = this->get_parameter("enable_display").as_bool();
        logging_publish_hz_ = this->get_parameter("logging_publish_hz").as_double();
        logging_publish_period_ = (logging_publish_hz_ > 0.0) ? (1.0 / logging_publish_hz_) : 0.0;
    
        initialize_logging_outputs();
    
        // ------------------------------------------------------------
        // Safety-field / semantic parameters
        // ------------------------------------------------------------
        this->declare_parameter("dh0_human", 1.0);
        this->declare_parameter("dh0_obstacle", 0.3);
    
        this->declare_parameter("enable_social_navigation", false);
        this->declare_parameter("social_tangent_bias", 0.5);
        this->declare_parameter("social_tangent_layers", 3);
        this->declare_parameter("social_layer_thickness", 1);
        this->declare_parameter("human_direction_threshold", 0.15);
    
        this->declare_parameter("robot_mos_human", 0.5);
        this->declare_parameter("robot_mos_obstacle", 0.1);
    
        dh0_human = this->get_parameter("dh0_human").as_double();
        dh0_obstacle = this->get_parameter("dh0_obstacle").as_double();
    
        enable_social_navigation_ = this->get_parameter("enable_social_navigation").as_bool();
        social_tangent_bias_ = this->get_parameter("social_tangent_bias").as_double();
        social_tangent_layers_ = this->get_parameter("social_tangent_layers").as_int();
        social_layer_thickness_ = this->get_parameter("social_layer_thickness").as_int();
        human_direction_threshold_ = this->get_parameter("human_direction_threshold").as_double();
    
        robot_MOS_human = this->get_parameter("robot_mos_human").as_double();
        robot_MOS_obstacle = this->get_parameter("robot_mos_obstacle").as_double();
    
        // ------------------------------------------------------------
        // Dynamic CBF parameters
        // ------------------------------------------------------------
        this->declare_parameter("cbf_sigma_epsilon", 0.1);
        this->declare_parameter("cbf_sigma_kappa", 5.0);
    
        cbf_sigma_epsilon_ = this->get_parameter("cbf_sigma_epsilon").as_double();
        cbf_sigma_kappa_ = this->get_parameter("cbf_sigma_kappa").as_double();
    
        // ------------------------------------------------------------
        // Velocity bounds
        // ------------------------------------------------------------
        this->declare_parameter("vel_max_x_fwd", 0.9);
        this->declare_parameter("vel_max_x_bwd", 0.9);
        this->declare_parameter("vel_max_y", 0.9);
        this->declare_parameter("vel_max_yaw", 0.8);
    
        vel_max_x_fwd_ = this->get_parameter("vel_max_x_fwd").as_double();
        vel_max_x_bwd_ = this->get_parameter("vel_max_x_bwd").as_double();
        vel_max_y_ = this->get_parameter("vel_max_y").as_double();
        vel_max_yaw_ = this->get_parameter("vel_max_yaw").as_double();
    
        // ------------------------------------------------------------
        // Human tracker parameters
        // ------------------------------------------------------------
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
    
        const float track_timeout = this->get_parameter("human_track_timeout_sec").as_double();
        const float track_gate = this->get_parameter("human_track_gate_radius").as_double();
        const float track_decay = this->get_parameter("human_track_velocity_decay_tau").as_double();
        const float track_velocity_threshold = this->get_parameter("human_track_velocity_threshold").as_double();
        const float decay_in_fov = this->get_parameter("decay_in_fov").as_double();
        const float decay_stationary = this->get_parameter("decay_stationary").as_double();
        const float decay_unconfirmed = this->get_parameter("decay_unconfirmed").as_double();
        const bool no_retrack_on_move = this->get_parameter("no_retrack_on_move").as_bool();
    
        min_yolo_cells_ = this->get_parameter("min_yolo_cells").as_int();
        enable_human_tracker_dilation_ = this->get_parameter("enable_human_tracker_dilation").as_bool();
    
        human_tracker_ = std::make_unique<HumanTracker>(
            track_timeout,
            track_gate,
            track_decay,
            track_velocity_threshold,
            decay_in_fov,
            decay_stationary,
            decay_unconfirmed,
            3,
            3,
            no_retrack_on_move
        );
    
        // ------------------------------------------------------------
        // Tight-area wall softening
        // ------------------------------------------------------------
        this->declare_parameter("tight_area_human_threshold", 2.0);
        this->declare_parameter("tight_area_h_threshold", 0.3);
        this->declare_parameter("tight_area_wall_slack", -0.1);
    
        tight_area_human_threshold_ = this->get_parameter("tight_area_human_threshold").as_double();
        tight_area_h_threshold_ = this->get_parameter("tight_area_h_threshold").as_double();
        tight_area_wall_slack_ = this->get_parameter("tight_area_wall_slack").as_double();
    
        // ------------------------------------------------------------
        // CloudMerger params passed from this node in main()
        // ------------------------------------------------------------
        this->declare_parameter("min_z", 0.05);
        this->declare_parameter("max_z", 0.80);
    
        // ------------------------------------------------------------
        // Informational prints
        // ------------------------------------------------------------
        RCLCPP_INFO(
            this->get_logger(),
            "dh0_human=%.2f, dh0_obstacle=%.2f, MOS_human=%.2f, MOS_obstacle=%.2f, display=%s, social_nav=%s",
            dh0_human, dh0_obstacle, robot_MOS_human, robot_MOS_obstacle,
            enable_display ? "true" : "false",
            enable_social_navigation_ ? "true" : "false"
        );
    
        RCLCPP_INFO(
            this->get_logger(),
            "Dynamic CBF: sigma_epsilon=%.3f, sigma_kappa=%.2f",
            cbf_sigma_epsilon_, cbf_sigma_kappa_
        );
    
        RCLCPP_INFO(
            this->get_logger(),
            "Velocity bounds: x_fwd=%.2f, x_bwd=%.2f, y=%.2f, yaw=%.2f",
            vel_max_x_fwd_, vel_max_x_bwd_, vel_max_y_, vel_max_yaw_
        );
    
        RCLCPP_INFO(
            this->get_logger(),
            "HumanTracker: timeout=%.1fs, gate=%.2fm, vel_thresh=%.2fm/s, decay_fov=%.2f, decay_stat=%.2f, decay_unconf=%.2f, no_retrack=%s",
            track_timeout, track_gate, track_velocity_threshold,
            decay_in_fov, decay_stationary, decay_unconfirmed,
            no_retrack_on_move ? "true" : "false"
        );
    
        RCLCPP_INFO(
            this->get_logger(),
            "Tight-area params: human_thresh=%.2fm, h_thresh=%.2f, wall_slack=%.2f",
            tight_area_human_threshold_, tight_area_h_threshold_, tight_area_wall_slack_
        );
    
        RCLCPP_INFO(this->get_logger(), "Logging publish rate: %.1f Hz", logging_publish_hz_);
    }
    
    void allocate_persistent_buffers() {
        cudaError_t err;
    
        // ------------------------------------------------------------
        // Main persistent field buffers
        // ------------------------------------------------------------
        err = cudaMallocHost((void**)&hgrid1, IMAX * JMAX * QMAX * sizeof(float));
        if (err != cudaSuccess) {
            RCLCPP_ERROR(this->get_logger(), "CUDA allocation failed for hgrid1: %s", cudaGetErrorString(err));
            throw std::runtime_error("CUDA allocation failed for hgrid1");
        }
    
        err = cudaMallocHost((void**)&hgrid0, IMAX * JMAX * QMAX * sizeof(float));
        if (err != cudaSuccess) {
            RCLCPP_ERROR(this->get_logger(), "CUDA allocation failed for hgrid0: %s", cudaGetErrorString(err));
            throw std::runtime_error("CUDA allocation failed for hgrid0");
        }
    
        err = cudaMallocHost((void**)&bound, IMAX * JMAX * QMAX * sizeof(float));
        if (err != cudaSuccess) {
            RCLCPP_ERROR(this->get_logger(), "CUDA allocation failed for bound: %s", cudaGetErrorString(err));
            throw std::runtime_error("CUDA allocation failed for bound");
        }
    
        err = cudaMallocHost((void**)&force, IMAX * JMAX * QMAX * sizeof(float));
        if (err != cudaSuccess) {
            RCLCPP_ERROR(this->get_logger(), "CUDA allocation failed for force: %s", cudaGetErrorString(err));
            throw std::runtime_error("CUDA allocation failed for force");
        }
    
        dhdt_grid = static_cast<float*>(std::malloc(IMAX * JMAX * QMAX * sizeof(float)));
        guidance_x_grid = static_cast<float*>(std::malloc(IMAX * JMAX * QMAX * sizeof(float)));
        guidance_y_grid = static_cast<float*>(std::malloc(IMAX * JMAX * QMAX * sizeof(float)));
    
        if (!dhdt_grid || !guidance_x_grid || !guidance_y_grid) {
            RCLCPP_ERROR(this->get_logger(), "Memory allocation failed for persistent guidance/dhdt grids");
            throw std::runtime_error("Persistent grid allocation failed");
        }
    
        // ------------------------------------------------------------
        // Persistent temporary buffers for profiling-ready execution
        // ------------------------------------------------------------
        hgrid_temp_ = static_cast<float*>(std::malloc(IMAX * JMAX * QMAX * sizeof(float)));
        guidance_x_temp_ = static_cast<float*>(std::calloc(IMAX * JMAX * QMAX, sizeof(float)));
        guidance_y_temp_ = static_cast<float*>(std::calloc(IMAX * JMAX * QMAX, sizeof(float)));
        forcing_zero_temp_ = static_cast<float*>(std::calloc(IMAX * JMAX * QMAX, sizeof(float)));
        bound_guidance_temp_ = static_cast<float*>(std::malloc(IMAX * JMAX * QMAX * sizeof(float)));
        class_map_temp_expanded_ = static_cast<int8_t*>(std::malloc(IMAX * JMAX * sizeof(int8_t)));
        boundary_temp_ = static_cast<float*>(std::malloc(IMAX * JMAX * sizeof(float)));
        inflate_bound_temp_ = static_cast<float*>(std::malloc(IMAX * JMAX * sizeof(float)));
        inflate_class_temp_ = static_cast<int8_t*>(std::malloc(IMAX * JMAX * sizeof(int8_t)));
    
        if (!hgrid_temp_ || !guidance_x_temp_ || !guidance_y_temp_ ||
            !forcing_zero_temp_ || !bound_guidance_temp_ ||
            !class_map_temp_expanded_ || !boundary_temp_ ||
            !inflate_bound_temp_ || !inflate_class_temp_) {
            RCLCPP_ERROR(this->get_logger(), "Memory allocation failed for persistent temporary buffers");
            throw std::runtime_error("Temporary buffer allocation failed");
        }
        
        std::memset(bound_guidance_temp_, 0, IMAX * JMAX * QMAX * sizeof(float));
    
        // ------------------------------------------------------------
        // Initialize values
        // ------------------------------------------------------------
        for (int n = 0; n < IMAX * JMAX * QMAX; ++n) {
            hgrid1[n] = h0;
            hgrid0[n] = h0;
            hgrid_temp_[n] = h0;
            bound[n] = 0.0f;
            force[n] = 0.0f;
            dhdt_grid[n] = 0.0f;
            guidance_x_grid[n] = 0.0f;
            guidance_y_grid[n] = 0.0f;
        }
    
        Kernel::poissonInit();
    }
    
    void initialize_ros_interfaces() {
        rclcpp::SubscriptionOptions options_occ;
        rclcpp::SubscriptionOptions options_state;
        rclcpp::SubscriptionOptions options_cmd;
        rclcpp::SubscriptionOptions options_yolo;
    
        options_occ.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        options_state.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        options_cmd.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
        options_yolo.callback_group = this->create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);

      image_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::Image>>(
          this, "/yolo/segmentation_mask"
      );
      
      cloud_sub_ = std::make_shared<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>>(
          this, "/camera/point_cloud/cloud_registered"
      );
    
        occ_grid_suber_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "occupancy_grid", 1,
            std::bind(&PoissonControllerNode::occ_grid_callback, this, std::placeholders::_1),
            options_occ
        );
    
        class_map_suber_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "class_map", 1,
            std::bind(&PoissonControllerNode::class_map_callback, this, std::placeholders::_1),
            options_yolo
        );
    
        visibility_map_suber_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "visibility_map", 1,
            std::bind(&PoissonControllerNode::visibility_map_callback, this, std::placeholders::_1),
            options_yolo
        );
    
        pose_suber_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/odom", 1,
            std::bind(&PoissonControllerNode::state_update_callback, this, std::placeholders::_1),
            options_state
        );
    
        twist_suber_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "u_des", 1,
            std::bind(&PoissonControllerNode::teleop_callback, this, std::placeholders::_1),
            options_cmd
        );
    
        key_suber_ = this->create_subscription<std_msgs::msg::Int32>(
            "key_press", 1,
            std::bind(&PoissonControllerNode::keyboard_callback, this, std::placeholders::_1),
            options_cmd
        );
    
        poisson_image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/poisson/visualization", 10);
        logging_data_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/poisson/logging_data", 10);
    
        mpc_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        mpc_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&PoissonControllerNode::mpc_callback, this),
            mpc_callback_group_
        );
    }



    // ============================================================
    // 8. EXISTING LOW-LEVEL METHODS TO KEEP / MOVE VERBATIM
    // ============================================================

    void build_occ_map(float* occ_map, const float* occ_map_old, const int8_t* conf) {
        const int8_t T_hi = 85;
        const int8_t T_lo = 64;
    
        for (int i = 0; i < IMAX; ++i) {
            for (int j = 0; j < JMAX; ++j) {
                const int i0 = i + static_cast<int>(std::round(dx[1] / DS));
                const int j0 = j + static_cast<int>(std::round(dx[0] / DS));
    
                const bool in_grid = (i0 >= 0) && (i0 < IMAX) && (j0 >= 0) && (j0 < JMAX);
                const bool strong = conf[i * JMAX + j] >= T_hi;
                const bool weak = conf[i * JMAX + j] >= T_lo;
    
                if (strong) {
                    occ_map[i * JMAX + j] = -1.0f;
                } else if (weak && in_grid) {
                    occ_map[i * JMAX + j] = occ_map_old[i0 * JMAX + j0];
                } else {
                    occ_map[i * JMAX + j] = 1.0f;
                }
            }
        }
    }

    void find_boundary(float* grid, float* bound, bool fix_flag, bool tight_area, const int8_t* class_map) {
        for (int i = 0; i < IMAX; ++i) {
            for (int j = 0; j < JMAX; ++j) {
                if (i == 0 || i == IMAX - 1 || j == 0 || j == JMAX - 1) {
                    bound[i * JMAX + j] = 0.0f;
                }
            }
        }
    
        std::memcpy(boundary_temp_, bound, IMAX * JMAX * sizeof(float));
        float* b0 = boundary_temp_;
    
        for (int i = 1; i < IMAX - 1; ++i) {
            for (int j = 1; j < JMAX - 1; ++j) {
                const int n = i * JMAX + j;
    
                if (b0[n] == 1.0f) {
                    if (b0[(i + 1) * JMAX + j] == -1.0f ||
                        b0[(i - 1) * JMAX + j] == -1.0f ||
                        b0[i * JMAX + (j + 1)] == -1.0f ||
                        b0[i * JMAX + (j - 1)] == -1.0f ||
                        b0[(i + 1) * JMAX + (j + 1)] == -1.0f ||
                        b0[(i - 1) * JMAX + (j + 1)] == -1.0f ||
                        b0[(i + 1) * JMAX + (j - 1)] == -1.0f ||
                        b0[(i - 1) * JMAX + (j - 1)] == -1.0f) {
                        bound[n] = 0.0f;
                    }
                }
    
                if (fix_flag && !bound[n]) {
                    bool is_wall = true;
                    if (class_map) {
                        is_wall = (class_map[n] != 1);
                    }
    
                    if (tight_area && is_wall) {
                        grid[n] = h0 + tight_area_wall_slack_;
                    } else {
                        grid[n] = h0;
                    }
                }
            }
        }
    }

    int initialize_robot_kernel(float*& kernel, float mos) {
        robot_length = 0.7f;
        robot_width = 0.3f;
    
        const float ar = mos * robot_length / 2.0f;
        const float br = mos * robot_width / 2.0f;
        const float D = 2.0f * std::sqrt(ar * ar + br * br);
    
        int dim = 2 * static_cast<int>(std::ceil(std::ceil(D / DS) / 2.0f));
        if (dim < 2) dim = 2;
    
        kernel = static_cast<float*>(std::malloc(dim * dim * QMAX * sizeof(float)));
        if (!kernel) {
            RCLCPP_ERROR(this->get_logger(), "Failed to allocate robot kernel");
            throw std::runtime_error("Robot kernel allocation failed");
        }
    
        for (int q = 0; q < QMAX; ++q) {
            float* kernel_slice = kernel + q * dim * dim;
            const float yawq = q_to_yaw(q, xc[2]);
            fill_elliptical_robot_kernel(kernel_slice, yawq, dim, 2.0f, mos);
        }
    
        return dim;
    }


    void fill_elliptical_robot_kernel(float* kernel, float yawq, int dim, float expo, float mos) {
        const float ar = mos * robot_length / 2.0f;
        const float br = mos * robot_width / 2.0f;
    
        if (ar < 0.001f || br < 0.001f) {
            for (int i = 0; i < dim * dim; ++i) kernel[i] = 0.0f;
            return;
        }
    
        for (int i = 0; i < dim; ++i) {
            const float yi = static_cast<float>(i - dim / 2) * DS;
            for (int j = 0; j < dim; ++j) {
                kernel[i * dim + j] = 0.0f;
                const float xi = static_cast<float>(j - dim / 2) * DS;
    
                const float xb = std::cos(yawq) * xi + std::sin(yawq) * yi;
                const float yb = -std::sin(yawq) * xi + std::cos(yawq) * yi;
    
                const float dist =
                    std::pow(std::abs(xb / ar), expo) +
                    std::pow(std::abs(yb / br), expo);
    
                if (dist <= 1.0f) kernel[i * dim + j] = -1.0f;
            }
        }
    }


    void inflate_occupancy_grid(float* bound, int8_t* class_map) {
        std::memcpy(inflate_bound_temp_, bound, IMAX * JMAX * sizeof(float));
        float* b0 = inflate_bound_temp_;
        
        int8_t* c0 = inflate_class_temp_;
        if (class_map) {
            std::memcpy(c0, class_map, IMAX * JMAX * sizeof(int8_t));
        }
    
        for (int i = 1; i < IMAX - 1; ++i) {
            for (int j = 1; j < JMAX - 1; ++j) {
                if (!b0[i * JMAX + j]) {
                    int8_t source_class = 0;
    
                    if (class_map) {
                        for (int di = -1; di <= 1 && source_class == 0; ++di) {
                            for (int dj = -1; dj <= 1 && source_class == 0; ++dj) {
                                const int ni = i + di;
                                const int nj = j + dj;
                                if (ni >= 0 && ni < IMAX && nj >= 0 && nj < JMAX) {
                                    if (b0[ni * JMAX + nj] < 0.0f && c0[ni * JMAX + nj] == 1) {
                                        source_class = 1;
                                    }
                                }
                            }
                        }
                    }
    
                    const float* kernel = (source_class == 1) ? robot_kernel_human : robot_kernel_obstacle;
                    const int kernel_dim = (source_class == 1) ? robot_kernel_dim_human : robot_kernel_dim_obstacle;
                    const int lim = (kernel_dim - 1) / 2;
    
                    const int ilow = std::max(i - lim, 0);
                    const int itop = std::min(i + lim, IMAX);
                    const int jlow = std::max(j - lim, 0);
                    const int jtop = std::min(j + lim, JMAX);
    
                    for (int p = ilow; p < itop; ++p) {
                        for (int q = jlow; q < jtop; ++q) {
                            const float kernel_val = kernel[(p - i + lim) * kernel_dim + (q - j + lim)];
                            bound[p * JMAX + q] += kernel_val;
    
                            if (class_map && kernel_val < 0.0f && source_class == 1) {
                                class_map[p * JMAX + q] = 1;
                            }
                        }
                    }
                }
            }
        }
    
        for (int n = 0; n < IMAX * JMAX; ++n) {
            if (bound[n] < -1.0f) bound[n] = -1.0f;
        }
    }

    bool is_tight_area() {
        auto tracks = human_tracker_->get_active_tracks();
        if (tracks.empty()) return false;
    
        float min_human_dist = FLT_MAX;
        for (const auto& track : tracks) {
            const float d = std::sqrt(std::pow(track.x - x[0], 2) + std::pow(track.y - x[1], 2));
            min_human_dist = std::min(min_human_dist, d);
        }
    
        const float ic = y_to_i(0.0f, xc[1]);
        const float jc = x_to_j(0.0f, xc[0]);
        const float qc = yaw_to_q(x[2], xc[2]);
    
        const float ic_clamped = std::clamp(ic, 0.0f, static_cast<float>(IMAX - 1));
        const float jc_clamped = std::clamp(jc, 0.0f, static_cast<float>(JMAX - 1));
    
        const float h_at_robot = trilinear_interpolation(hgrid1, ic_clamped, jc_clamped, qc);
    
        const bool tight =
            (min_human_dist < tight_area_human_threshold_) &&
            (h_at_robot < tight_area_h_threshold_);
    
        return tight;
    }

    void compute_boundary_gradients(float* guidance_x, float* guidance_y, float* bound,
                                    const int8_t* class_map,
                                    float /*rx*/, float /*ry*/,
                                    float /*vn_x*/, float /*vn_y*/,
                                    bool populate_human_info) {
        // Set border gradients
        for (int i = 0; i < IMAX; ++i) {
            for (int j = 0; j < JMAX; ++j) {
                if (i == 0) guidance_x[i * JMAX + j] = dh0;
                if (j == 0) guidance_y[i * JMAX + j] = dh0;
                if (i == (IMAX - 1)) guidance_x[i * JMAX + j] = -dh0;
                if (j == (JMAX - 1)) guidance_y[i * JMAX + j] = -dh0;
            }
        }
    
        if (populate_human_info) {
            human_boundary_info_.clear();
        }
    
        // Compute raw boundary normals on Layer 0
        for (int i = 1; i < IMAX - 1; ++i) {
            for (int j = 1; j < JMAX - 1; ++j) {
                if (!bound[i * JMAX + j]) {
                    guidance_x[i * JMAX + j] = 0.0f;
                    guidance_y[i * JMAX + j] = 0.0f;
    
                    for (int p = -1; p <= 1; ++p) {
                        for (int q = -1; q <= 1; ++q) {
                            if (q > 0) {
                                guidance_x[i * JMAX + j] += bound[(i + q) * JMAX + (j + p)];
                                guidance_y[i * JMAX + j] += bound[(i + p) * JMAX + (j + q)];
                            } else if (q < 0) {
                                guidance_x[i * JMAX + j] -= bound[(i + q) * JMAX + (j + p)];
                                guidance_y[i * JMAX + j] -= bound[(i + p) * JMAX + (j + q)];
                            }
                        }
                    }
                }
            }
        }
    
        // Normalize and assign class-dependent strength
        for (int i = 0; i < IMAX; ++i) {
            for (int j = 0; j < JMAX; ++j) {
                if (!bound[i * JMAX + j]) {
                    const float V = std::sqrt(
                        guidance_x[i * JMAX + j] * guidance_x[i * JMAX + j] +
                        guidance_y[i * JMAX + j] * guidance_y[i * JMAX + j]);
    
                    if (V != 0.0f) {
                        guidance_x[i * JMAX + j] /= V;
                        guidance_y[i * JMAX + j] /= V;
                    }
    
                    float local_dh0 = dh0_obstacle;
                    bool is_human = false;
    
                    if (class_map) {
                        for (int di = -1; di <= 1 && !is_human; ++di) {
                            for (int dj = -1; dj <= 1 && !is_human; ++dj) {
                                const int ni = i + di;
                                const int nj = j + dj;
                                if (ni >= 0 && ni < IMAX && nj >= 0 && nj < JMAX) {
                                    if (bound[ni * JMAX + nj] < 0.0f && class_map[ni * JMAX + nj] == 1) {
                                        is_human = true;
                                    }
                                }
                            }
                        }
    
                        if (is_human) {
                            local_dh0 = dh0_human;
                            if (populate_human_info) {
                                human_boundary_info_.emplace_back(
                                    i, j,
                                    guidance_x[i * JMAX + j],
                                    guidance_y[i * JMAX + j],
                                    local_dh0
                                );
                            }
                        }
                    }
    
                    guidance_x[i * JMAX + j] *= local_dh0;
                    guidance_y[i * JMAX + j] *= local_dh0;
                }
            }
        }
    }

    float compute_tangent_direction(const std::vector<HumanTrack>& active_tracks,
                                    float /*rx*/, float /*ry*/,
                                    float /*vn_x*/, float /*vn_y*/) {
        float target_sign = -1.0f;  // Default visual CW / pass left
    
        for (const auto& track : active_tracks) {
            const float current_distance = std::sqrt(track.x * track.x + track.y * track.y);
    
            const float current_time =
                std::chrono::duration<float>(std::chrono::steady_clock::now() - t_start).count();
    
            float closing_rate = 0.0f;
    
            auto it = prev_human_distances_.find(track.id);
            if (it != prev_human_distances_.end()) {
                const float prev_distance = it->second.first;
                const float prev_time = it->second.second;
                const float dt = current_time - prev_time;
    
                if (dt > 0.01f && dt < 1.0f) {
                    closing_rate = (prev_distance - current_distance) / dt;
                }
            }
    
            prev_human_distances_[track.id] = {current_distance, current_time};
    
            if (closing_rate > human_direction_threshold_) {
                target_sign = 1.0f;  // visual CCW / pass right
            }
        }
        std::set<int> active_ids;
        for (const auto& track : active_tracks) {
            active_ids.insert(track.id);
        }
        
        for (auto it = prev_human_distances_.begin(); it != prev_human_distances_.end(); ) {
            if (active_ids.find(it->first) == active_ids.end()) {
                it = prev_human_distances_.erase(it);
            } else {
                ++it;
            }
        }
        
        current_tangent_direction_ = target_sign;
        return target_sign;
    }

    void expand_human_obstacles_for_guidance(float* bound_guidance,
                                             float* guidance_x,
                                             float* guidance_y,
                                             const float* bound_original,
                                             int num_layers,
                                             int layer_thickness,
                                             float bias_strength,
                                             float sign) {
        std::memcpy(bound_guidance, bound_original, IMAX * JMAX * sizeof(float));
    
        std::vector<bool> current_occupied(IMAX * JMAX, false);
        std::vector<bool> is_human_region(IMAX * JMAX, false);
    
        // Seed from human boundary info
        for (const auto& info : human_boundary_info_) {
            const int i = std::get<0>(info);
            const int j = std::get<1>(info);
    
            current_occupied[i * JMAX + j] = true;
            is_human_region[i * JMAX + j] = true;
    
            for (int di = -1; di <= 1; ++di) {
                for (int dj = -1; dj <= 1; ++dj) {
                    const int ni = i + di;
                    const int nj = j + dj;
                    if (ni >= 0 && ni < IMAX && nj >= 0 && nj < JMAX) {
                        if (bound_original[ni * JMAX + nj] < 0.0f) {
                            current_occupied[ni * JMAX + nj] = true;
                            is_human_region[ni * JMAX + nj] = true;
                        }
                    }
                }
            }
        }
    
        for (int layer = 1; layer <= num_layers; ++layer) {
            std::vector<bool> next_occupied = current_occupied;
            std::vector<std::tuple<int, int, float, float>> new_layer_cells;
    
            const int r = layer_thickness;
    
            for (int i = r; i < IMAX - r; ++i) {
                for (int j = r; j < JMAX - r; ++j) {
                    if (current_occupied[i * JMAX + j]) continue;
    
                    bool in_human_range = false;
                    float avg_gx = 0.0f;
                    float avg_gy = 0.0f;
                    int count = 0;
    
                    for (int di = -r; di <= r && !in_human_range; ++di) {
                        for (int dj = -r; dj <= r && !in_human_range; ++dj) {
                            if (di * di + dj * dj <= r * r) {
                                const int ni = i + di;
                                const int nj = j + dj;
    
                                if (is_human_region[ni * JMAX + nj] && current_occupied[ni * JMAX + nj]) {
                                    in_human_range = true;
    
                                    for (const auto& info : human_boundary_info_) {
                                        const int bi = std::get<0>(info);
                                        const int bj = std::get<1>(info);
                                        const int dist_sq = (bi - i) * (bi - i) + (bj - j) * (bj - j);
    
                                        if (dist_sq <= (layer * r + r) * (layer * r + r)) {
                                            avg_gx += std::get<2>(info);
                                            avg_gy += std::get<3>(info);
                                            ++count;
                                        }
                                    }
                                }
                            }
                        }
                    }
    
                    if (in_human_range && bound_original[i * JMAX + j] > 0.0f) {
                        if (count > 0) {
                            avg_gx /= static_cast<float>(count);
                            avg_gy /= static_cast<float>(count);
                        } else {
                            avg_gx = guidance_x[i * JMAX + j];
                            avg_gy = guidance_y[i * JMAX + j];
                        }
    
                        const float mag = std::sqrt(avg_gx * avg_gx + avg_gy * avg_gy);
                        if (mag > 0.01f) {
                            avg_gx /= mag;
                            avg_gy /= mag;
                            new_layer_cells.emplace_back(i, j, avg_gx, avg_gy);
                        }
    
                        next_occupied[i * JMAX + j] = true;
                        is_human_region[i * JMAX + j] = true;
                    }
                }
            }
    
            const float ramp = static_cast<float>(layer) / static_cast<float>(num_layers);
    
            for (const auto& cell : new_layer_cells) {
                const int i = std::get<0>(cell);
                const int j = std::get<1>(cell);
                const float gx = std::get<2>(cell);
                const float gy = std::get<3>(cell);
    
                const float biased_gx = gx + sign * bias_strength * ramp * gy;
                const float biased_gy = gy + sign * bias_strength * ramp * (-gx);
    
                const float V = std::sqrt(biased_gx * biased_gx + biased_gy * biased_gy);
                if (V > 0.0f) {
                    guidance_x[i * JMAX + j] = biased_gx / V * dh0_human;
                    guidance_y[i * JMAX + j] = biased_gy / V * dh0_human;
                }
    
                bound_guidance[i * JMAX + j] = 0.0f;
                tangent_layer_display[i * JMAX + j] = static_cast<int8_t>(layer);
            }
    
            current_occupied = next_occupied;
        }
    }

    void compute_optimal_forcing_function(float* force,
                                          const float* guidance_x,
                                          const float* guidance_y,
                                          const float* bound) {
        const float max_div = 10.0f;
    
        for (int i = 1; i < IMAX - 1; ++i) {
            for (int j = 1; j < JMAX - 1; ++j) {
                force[i * JMAX + j] =
                    (guidance_x[(i + 1) * JMAX + j] - guidance_x[(i - 1) * JMAX + j]) / (2.0f * DS) +
                    (guidance_y[i * JMAX + (j + 1)] - guidance_y[i * JMAX + (j - 1)]) / (2.0f * DS);
    
                if (bound[i * JMAX + j] > 0.0f) {
                    // free space: no clamp
                } else if (bound[i * JMAX + j] < 0.0f) {
                    force[i * JMAX + j] = std::max(force[i * JMAX + j], max_div);
                    force[i * JMAX + j] = std::min(force[i * JMAX + j], 0.0f);
                } else {
                    force[i * JMAX + j] = 0.0f;
                }
            }
        }
    }

    ConnectedComponentsData compute_connected_components(const float* occ_true) {
        ConnectedComponentsData cc;
        cc.binary = cv::Mat(IMAX, JMAX, CV_8UC1);
    
        for (int n = 0; n < IMAX * JMAX; ++n) {
            cc.binary.data[n] = (occ_true[n] < 0.0f) ? 255 : 0;
        }
    
        cc.num_labels = cv::connectedComponentsWithStats(
            cc.binary,
            cc.labels,
            cc.stats,
            cc.centroids
        );
    
        return cc;
    }

    std::vector<ClusterInfo> extract_lidar_clusters(const ConnectedComponentsData& cc) {
        std::vector<ClusterInfo> clusters;

        for (int l = 1; l < cc.num_labels; ++l) {
            const int area = cc.stats.at<int>(l, cv::CC_STAT_AREA);
            if (area < 3) continue;
    
            ClusterInfo c;
    
            const float j_centroid = static_cast<float>(cc.centroids.at<double>(l, 0));
            const float i_centroid = static_cast<float>(cc.centroids.at<double>(l, 1));
    
            c.centroid_x = (j_centroid - JMAX / 2) * DS;
            c.centroid_y = (i_centroid - IMAX / 2) * DS;
            c.cell_count = area;
            c.label_id = l;
    
            int yolo_cell_count = 0;
            int visible_cell_count = 0;
            int cluster_cell_count = 0;
    
            for (int i = 0; i < IMAX; ++i) {
                for (int j = 0; j < JMAX; ++j) {
                    if (cc.labels.at<int>(i, j) == l) {
                        ++cluster_cell_count;
                        const int idx = i * JMAX + j;
                        if (class_map[idx] == 1) ++yolo_cell_count;
                        if (visibility_map[idx] == 1) ++visible_cell_count;
                    }
                }
            }
    
            c.has_yolo_seed = (yolo_cell_count >= min_yolo_cells_);
            c.in_camera_fov = (cluster_cell_count > 0 && visible_cell_count * 2 >= cluster_cell_count);
    
            clusters.push_back(c);
        }
    
        return clusters;
    }

    void label_human_clusters(const float* occ_true) {
        std::memset(class_map_expanded, 0, IMAX * JMAX * sizeof(int8_t));
    
        ConnectedComponentsData cc = compute_connected_components(occ_true);
        auto clusters = extract_lidar_clusters(cc);
    
        const float current_time =
            std::chrono::duration<float>(std::chrono::steady_clock::now() - t_start).count();
    
        human_tracker_->update(clusters, current_time);
        auto active_tracks = human_tracker_->get_active_tracks();
    
        for (const auto& track : active_tracks) {
            const float track_j = track.x / DS + JMAX / 2.0f;
            const float track_i = track.y / DS + IMAX / 2.0f;
    
            float best_dist = 999999.0f;
            int best_label = -1;
    
            for (int l = 1; l < cc.num_labels; ++l) {
                const float j_cent = static_cast<float>(cc.centroids.at<double>(l, 0));
                const float i_cent = static_cast<float>(cc.centroids.at<double>(l, 1));
                const float dist = std::sqrt((j_cent - track_j) * (j_cent - track_j) +
                                             (i_cent - track_i) * (i_cent - track_i));
                if (dist < best_dist) {
                    best_dist = dist;
                    best_label = l;
                }
            }
    
            const float gate_cells = 1.0f / DS;
            const int max_human_cells = static_cast<int>(0.4f / (DS * DS));
    
            if (best_label > 0 &&
                best_dist < gate_cells &&
                track.yolo_ever_confirmed &&
                track.confidence > 0.5f) {
    
                const int cluster_size = cc.stats.at<int>(best_label, cv::CC_STAT_AREA);
    
                if (cluster_size <= max_human_cells) {
                    for (int i = 0; i < IMAX; ++i) {
                        for (int j = 0; j < JMAX; ++j) {
                            if (cc.labels.at<int>(i, j) == best_label) {
                                class_map_expanded[i * JMAX + j] = 1;
                            }
                        }
                    }
                } else {
                    const float label_radius = 0.1f / DS;
                    const float radius_sq = label_radius * label_radius;
    
                    for (int i = 0; i < IMAX; ++i) {
                        for (int j = 0; j < JMAX; ++j) {
                            if (cc.labels.at<int>(i, j) == best_label) {
                                const float di = static_cast<float>(i) - track_i;
                                const float dj = static_cast<float>(j) - track_j;
                                if (di * di + dj * dj <= radius_sq) {
                                    class_map_expanded[i * JMAX + j] = 1;
                                }
                            }
                        }
                    }
                }
            }
        }
    
        int labeled_cells = 0;
        for (int n = 0; n < IMAX * JMAX; ++n) {
            if (class_map_expanded[n] == 1) ++labeled_cells;
        }
        
        if (enable_human_tracker_dilation_ && labeled_cells > 0) {
            std::memcpy(class_map_temp_expanded_, class_map_expanded, IMAX * JMAX * sizeof(int8_t));
        
            const float* kernel = robot_kernel_human;
            const int lim = (robot_kernel_dim_human - 1) / 2;
        
            for (int i = 1; i < IMAX - 1; ++i) {
                const int ilow = std::max(i - lim, 0);
                const int itop = std::min(i + lim, IMAX);
        
                for (int j = 1; j < JMAX - 1; ++j) {
                    if (class_map_expanded[i * JMAX + j] != 1) continue;
        
                    const int jlow = std::max(j - lim, 0);
                    const int jtop = std::min(j + lim, JMAX);
        
                    for (int p = ilow; p < itop; ++p) {
                        for (int q = jlow; q < jtop; ++q) {
                            const float kernel_val =
                                kernel[(p - i + lim) * robot_kernel_dim_human + (q - j + lim)];
                            if (kernel_val < 0.0f) {
                                class_map_temp_expanded_[p * JMAX + q] = 1;
                            }
                        }
                    }
                }
            }
        
            std::memcpy(class_map_expanded, class_map_temp_expanded_, IMAX * JMAX * sizeof(int8_t));
        }
    }





    void safety_filter(const std::vector<float>& vd) {
        // In body_link frame, robot is always at origin (0, 0)
        const float ic = y_to_i(0.0f, xc[1]);
        const float jc = x_to_j(0.0f, xc[0]);
        const float qc = yaw_to_q(0.0f, xc[2]);
    
        // Conservative local estimate of dh/dt around the robot footprint
        const int range = static_cast<int>(std::round(0.2f / DS));
        dhdt = 0.0f;
        for (int di = -range; di <= range; ++di) {
            for (int dj = -range; dj <= range; ++dj) {
                const float dhdt_ij = trilinear_interpolation(dhdt_grid, ic + static_cast<float>(di), jc + static_cast<float>(dj), qc);
                if (dhdt_ij < dhdt) dhdt = dhdt_ij;
            }
        }
    
        // Safety function value and forward prediction to compensate field age
        h = trilinear_interpolation(hgrid1, ic, jc, qc);
        const float h_pred = h + dhdt * grid_age;
    
        // Guidance field (control direction) from Laplace solve
        // guidance_y corresponds to x-direction, guidance_x to y-direction
        const float vx = trilinear_interpolation(guidance_y_grid, ic, jc, qc);
        const float vy = trilinear_interpolation(guidance_x_grid, ic, jc, qc);
        const float v_norm = std::sqrt(vx * vx + vy * vy);
    
        // Numerical gradient of h-field in x/y
        const float h_eps = 1.0f;
        const float hip = trilinear_interpolation(hgrid1, ic + h_eps, jc, qc);
        const float him = trilinear_interpolation(hgrid1, ic - h_eps, jc, qc);
        const float hjp = trilinear_interpolation(hgrid1, ic, jc + h_eps, qc);
        const float hjm = trilinear_interpolation(hgrid1, ic, jc - h_eps, qc);
    
        const float Dh_x = (hjp - hjm) / (2.0f * h_eps * DS);
        const float Dh_y = (hip - him) / (2.0f * h_eps * DS);
    
        // Store guidance direction for logging/visualization
        dhdx = vx;
        dhdy = vy;
    
        // Numerical derivative in yaw
        const float q_eps = 1.0f;
        const float qp = q_wrap(qc + q_eps);
        const float qm = q_wrap(qc - q_eps);
    
        float hqp = trilinear_interpolation(hgrid1, ic, jc, qp);
        float hqm = trilinear_interpolation(hgrid1, ic, jc, qm);
        dhdq = (hqp - hqm) / (2.0f * q_eps * DQ);
    
        // Forward-predicted guidance-aligned derivatives
        const float dhdx_pred = vx;
        const float dhdy_pred = vy;
    
        hqp += trilinear_interpolation(dhdt_grid, ic, jc, qp) * grid_age;
        hqm += trilinear_interpolation(dhdt_grid, ic, jc, qm) * grid_age;
        const float dhdq_pred = (hqp - hqm) / (2.0f * q_eps * DQ);
    
        const float Dh_norm = std::sqrt(Dh_x * Dh_x + Dh_y * Dh_y + dhdq_pred * dhdq_pred);
    
        // sigma(h) = epsilon * (1 - exp(-kappa * max(0,h)))
        const float sigma_h =
            cbf_sigma_epsilon_ *
            (1.0f - std::exp(-cbf_sigma_kappa_ * std::max(0.0f, h_pred)));
    
        // Dynamic dh/dt scaling from eq. 31, clamped to avoid instability
        const float dhdt_scale =
            std::min(v_norm / (Dh_norm + sigma_h + 1.0e-6f), 1.0f);
    
        // Input-to-State Safety robustness term
        const float Pu[3] = {2.0f, 2.0f, 1.0f};
        const float ISSf1 = issf;
        const float ISSf2 = issf;
    
        const float b =
            dhdx_pred * dhdx_pred / Pu[0] +
            dhdy_pred * dhdy_pred / Pu[1] +
            dhdq_pred * dhdq_pred / Pu[2];
    
        float ISSf = std::sqrt(b) / ISSf1 + b / ISSf2;
        const float sigma = std::clamp(-10.0f * dhdt, 0.0f, 1.0f);
        ISSf *= sigma;
    
        // Activating function
        float a = wn * h_pred;
        a += vx * vd[0] + vy * vd[1];
        a += dhdt_scale * dhdt;
        a += dhdq_pred * vd[2];
        a -= ISSf;
    
        // Half-Sontag correction
        const float sigma_sontag = 1.0f;
        float lambda = 0.0f;
        if (b > 1.0e-4f) {
            lambda = (-a + std::sqrt(a * a + sigma_sontag * b * b)) / (2.0f * b);
        }
    
        v = vd;
        if (realtime_sf_flag) {
            v[0] += lambda * dhdx_pred / Pu[0];
            v[1] += lambda * dhdy_pred / Pu[1];
            v[2] += lambda * dhdq_pred / Pu[2];
        }
    }




    // ============================================================
    // 9. STATE
    // ============================================================

    TimingSample timing_{};
    std::chrono::steady_clock::time_point latest_field_timestamp_{};

    std::mutex mpc_mutex;
    MPC3D mpc3d_controller;
    mutable std::shared_mutex field_mutex_;

    const float h0 = 0.0f;
    const float dh0 = 1.0f;
    float wn = 1.0f;
    float issf = 50.0f;

    bool h_flag = false;
    bool dhdt_flag = false;
    bool save_flag = false;
    bool start_flag = false;
    bool enable_display = false;
    bool sit_flag = false;
    bool stop_flag = false;
    bool predictive_sf_flag = false;
    bool realtime_sf_flag = false;
    int space_counter = 0;
    int poisson_save_counter = 0;

    const std::vector<char> sorted_parameter_deck = {'1', '2', '3', '4', '5', '6', '0', '0'};
    std::random_device rd;
    std::mt19937 gen;
    std::vector<char> current_parameter_deck;

    std::vector<float> x = {0.0f, 0.0f, 0.0f};
    std::vector<float> xd = {0.0f, 0.0f, 0.0f};
    std::vector<float> xc = {-2.0f, -2.0f, 0.0f};
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
    float h{}, dhdt{}, dhdx{}, dhdy{}, dhdq{};

    float occ1[IMAX * JMAX];
    float occ0[IMAX * JMAX];
    int8_t conf[IMAX * JMAX];
    float grid_temp[IMAX * JMAX];
    float* hgrid1{};
    float* hgrid0{};
    float* bound{};
    float* force{};
    float* dhdt_grid{};
    float* robot_kernel_human{};
    float* robot_kernel_obstacle{};
    float* guidance_x_grid{};
    float* guidance_y_grid{};

    // Persistent temp buffers for profiling-ready execution
    float* hgrid_temp_{};
    float* guidance_x_temp_{};
    float* guidance_y_temp_{};
    float* forcing_zero_temp_{};
    float* bound_guidance_temp_{};
    int8_t* class_map_temp_expanded_{};
    float* boundary_temp_{};
    float* inflate_bound_temp_{};
    int8_t* inflate_class_temp_{};


    float guidance_x_display[IMAX * JMAX];
    float guidance_y_display[IMAX * JMAX];
    float bound_display[IMAX * JMAX];
    int8_t tangent_layer_display[IMAX * JMAX];

    float robot_length{}, robot_width{};
    float robot_MOS_human{}, robot_MOS_obstacle{};
    int robot_kernel_dim_human{}, robot_kernel_dim_obstacle{};

    rclcpp::CallbackGroup::SharedPtr mpc_callback_group_;
    rclcpp::TimerBase::SharedPtr mpc_timer_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr key_suber_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_suber_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr occ_grid_suber_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr class_map_suber_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr visibility_map_suber_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_suber_;

    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> image_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> cloud_sub_;

    int8_t class_map[IMAX * JMAX];
    int8_t visibility_map[IMAX * JMAX];
    int8_t class_map_expanded[IMAX * JMAX];
    std::unique_ptr<HumanTracker> human_tracker_;
    int min_yolo_cells_ = 5;
    bool enable_human_tracker_dilation_ = true;
    float dh0_human = 1.0f;
    float dh0_obstacle = 0.3f;
    bool enable_social_navigation_ = false;
    float social_tangent_bias_ = 0.5f;
    int social_tangent_layers_ = 3;
    int social_layer_thickness_ = 1;
    float current_tangent_direction_ = 1.0f;
    float human_direction_threshold_ = 0.15f;
    std::map<int, std::pair<float, float>> prev_human_distances_;
    std::vector<std::tuple<int, int, float, float, float>> human_boundary_info_;

    float tight_area_human_threshold_ = 2.0f;
    float tight_area_h_threshold_ = 0.3f;
    float tight_area_wall_slack_ = -0.1f;

    float cbf_sigma_epsilon_ = 0.1f;
    float cbf_sigma_kappa_ = 5.0f;
    float vel_max_x_fwd_ = 0.9f;
    float vel_max_x_bwd_ = 0.9f;
    float vel_max_y_ = 0.9f;
    float vel_max_yaw_ = 0.8f;

    unitree_api::msg::Request req;
    SportClient sport_req;
    std::ofstream outFileCSV;
    std::ofstream outFileBIN;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr poisson_image_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr logging_data_pub_;
    double logging_publish_hz_ = 10.0;
    double logging_publish_period_ = 0.1;
    std::chrono::steady_clock::time_point last_logging_publish_time_;
    bool enable_data_logging_to_file_ = false;
};

} // namespace ss

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);

    rclcpp::executors::MultiThreadedExecutor executor;

    auto poissonNode = std::make_shared<ss::PoissonControllerNode>();

    // Read CloudMerger parameters from the Poisson node so behavior matches the original setup
    const float min_z = poissonNode->get_parameter("min_z").as_double();
    const float max_z = poissonNode->get_parameter("max_z").as_double();

    RCLCPP_INFO(
        poissonNode->get_logger(),
        "Passing min_z=%.2f, max_z=%.2f to CloudMergerNode",
        min_z, max_z
    );

    auto mappingNode = std::make_shared<CloudMergerNode>(min_z, max_z);

    executor.add_node(mappingNode);
    executor.add_node(poissonNode);

    try {
        executor.spin();
        throw("Terminated");
    } catch (const char* msg) {
        rclcpp::shutdown();
        std::cout << msg << std::endl;
    }

    return 0;
}

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
    double guidance_field_ms{0.0};
    double safety_field_solve_ms{0.0};
    double dhdt_update_ms{0.0};
    double predictive_control_ms{0.0};
    double realtime_filter_ms{0.0};
    double command_dispatch_ms{0.0};
    double field_data_age_ms{0.0};
    double end_to_end_grid_ms{0.0};
};

struct SemanticStageOutput {
    bool tight_area{false};
    std::vector<ClusterInfo> clusters;
    std::vector<HumanTrack> active_tracks;
};

struct GuidanceStageOutput {
    float* guidance_x{nullptr};
    float* guidance_y{nullptr};
    float* forcing_zero{nullptr};
    float* bound_guidance{nullptr};
    bool uses_temp_bound{false};
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
        for (int n = 0; n < IMAX * JMAX; ++n) class_map[n] = msg->data[n];
    }

    void visibility_map_callback(nav_msgs::msg::OccupancyGrid::UniquePtr msg) {
        for (int n = 0; n < IMAX * JMAX; ++n) visibility_map[n] = msg->data[n];
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
        update_grid_metadata_from_message(msg);

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
        timing_.end_to_end_grid_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - grid_start).count();

        if (enable_display) render_visualization();
        publish_timing_data();
    }

    void handle_state_update(const nav_msgs::msg::Odometry& data) {
        update_robot_state(data);

        std::vector<float> v_input_body = form_nominal_body_command();

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

        timing_.field_data_age_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - latest_field_timestamp_).count();
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
        std::memcpy(hgrid_temp_, hgrid1, IMAX * JMAX * QMAX * sizeof(float));
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
        ScopedTimer timer(timing_.guidance_field_ms);
        GuidanceStageOutput out;
        out.guidance_x = guidance_x_temp_;
        out.guidance_y = guidance_y_temp_;
        out.forcing_zero = forcing_zero_temp_;
        out.bound_guidance = bound;
        out.uses_temp_bound = false;

        std::memset(guidance_x_temp_, 0, IMAX * JMAX * QMAX * sizeof(float));
        std::memset(guidance_y_temp_, 0, IMAX * JMAX * QMAX * sizeof(float));
        std::memset(forcing_zero_temp_, 0, IMAX * JMAX * QMAX * sizeof(float));
        std::memset(tangent_layer_display, 0, IMAX * JMAX * sizeof(int8_t));

        const float c_yaw = std::cos(x[2]);
        const float s_yaw = std::sin(x[2]);
        const float vn_body_x = c_yaw * vn[0] + s_yaw * vn[1];
        const float vn_body_y = -s_yaw * vn[0] + c_yaw * vn[1];

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

        if (enable_social_navigation_ && social_tangent_layers_ > 0 && !human_boundary_info_.empty()) {
            out.bound_guidance = bound_guidance_temp_;
            out.uses_temp_bound = true;
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

        solve_guidance_laplace(out.bound_guidance);
        compute_guidance_forcing();

        std::memcpy(guidance_x_display, guidance_x_temp_, IMAX * JMAX * sizeof(float));
        std::memcpy(guidance_y_display, guidance_y_temp_, IMAX * JMAX * sizeof(float));
        std::memcpy(bound_display, bound, IMAX * JMAX * sizeof(float));
        std::memcpy(guidance_x_grid, guidance_x_temp_, IMAX * JMAX * QMAX * sizeof(float));
        std::memcpy(guidance_y_grid, guidance_y_temp_, IMAX * JMAX * QMAX * sizeof(float));

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

    bool solve_safety_field(const GuidanceStageOutput&) {
        ScopedTimer timer(timing_.safety_field_solve_ms);
        const float relTol = 1.0e-4f;
        const int N = IMAX / 5;
        const float w_SOR = 2.0f / (1.0f + std::sin(M_PI / static_cast<float>(N + 1)));
        (void)Kernel::poissonSolve(hgrid_temp_, force, bound, relTol, w_SOR);

        std::memcpy(occ0, occ1, IMAX * JMAX * sizeof(float));
        std::memcpy(hgrid0, hgrid1, IMAX * JMAX * QMAX * sizeof(float));
        std::memcpy(hgrid1, hgrid_temp_, IMAX * JMAX * QMAX * sizeof(float));
        if (h_flag) dhdt_flag = true;
        return true;
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
            sleep(2);
            sport_req.StandDown(req);
            rclcpp::shutdown();
        } else if (sit_flag) {
            sport_req.StopMove(req);
            sleep(2);
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
        // Publish timing_.<...> here in the next step.
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
        if (!(poisson_save_counter % factor)) outFileBIN.write(reinterpret_cast<char*>(grid_temp), sizeof(grid_temp));
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
    
        if (!hgrid_temp_ || !guidance_x_temp_ || !guidance_y_temp_ || !forcing_zero_temp_ || !bound_guidance_temp_) {
            RCLCPP_ERROR(this->get_logger(), "Memory allocation failed for persistent temporary buffers");
            throw std::runtime_error("Temporary buffer allocation failed");
        }
    
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
    bool is_tight_area();
    void find_boundary(float* grid, float* bound, bool fix_flag, bool tight_area = false, const int8_t* class_map = nullptr);
    int initialize_robot_kernel(float*& kernel, float mos);
    void fill_elliptical_robot_kernel(float* kernel, float yawq, int dim, float expo, float mos);
    void inflate_occupancy_grid(float* bound, int8_t* class_map = nullptr);
    void compute_boundary_gradients(float* guidance_x, float* guidance_y, float* bound,
                                    const int8_t* class_map = nullptr,
                                    float rx = 0.0f, float ry = 0.0f,
                                    float vn_x = 0.0f, float vn_y = 0.0f,
                                    bool populate_human_info = false);
    float compute_tangent_direction(const std::vector<HumanTrack>& active_tracks, float rx, float ry, float vn_x, float vn_y);
    void expand_human_obstacles_for_guidance(float* bound_guidance, float* guidance_x, float* guidance_y,
                                             const float* bound_original, int num_layers, int layer_thickness,
                                             float bias_strength, float sign);
    void compute_optimal_forcing_function(float* force, const float* guidance_x, const float* guidance_y, const float* bound);
    void safety_filter(const std::vector<float> vd);
    std::vector<ClusterInfo> extract_lidar_clusters(const float* occ_true);
    void label_human_clusters(const float* occ_true);

    // ============================================================
    // 9. STATE
    // ============================================================

    TimingSample timing_{};
    std::chrono::steady_clock::time_point latest_field_timestamp_{};

    std::mutex mpc_mutex;
    MPC3D mpc3d_controller;
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

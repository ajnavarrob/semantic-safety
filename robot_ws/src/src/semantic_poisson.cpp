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
#include <limits>
#include <cstdint>
#include <set>
#include <shared_mutex>
#include <functional>
#include <sstream>
#include <cstdio>

#include <cuda_runtime.h>
#include "kernel.hpp"
#include "poisson.h"
#include "utils.h"
#include "mpc_cbf_3d.h"
#include "cloud_merger.h"
#include "poisson/human_tracker.h"
#include "constraints/constraint_manager.hpp"

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "nav_msgs/msg/occupancy_grid.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "std_msgs/msg/int32.hpp"
#include "std_msgs/msg/bool.hpp"
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

struct FieldBuffer {
    std::vector<float> hgrid;
    std::vector<float> dhdt;
    std::vector<float> beta;
    std::vector<float> guidance_x;
    std::vector<float> guidance_y;
    std::vector<float> bound;

    std::chrono::steady_clock::time_point timestamp;
    bool valid{false};
};

enum class SemanticUpdateMode {
    NORMAL = 0,
    EVALUATING = 1,
    INSERTING_CONSTRAINT = 2,
    REMOVING_CONSTRAINT = 3,
    TRANSITIONING_CONSTRAINT = 4
};

enum class AdmissionDecision {
    ACCEPT = 0,
    SLOW_INSERTION = 1,
    REJECT = 2
};

enum class AdmissionReason {
    NONE = 0,
    BOUNDARY_TOO_FAST,
    SAFE_SET_EMPTY,
    TOPOLOGY_CHANGE,
    NO_ADMISSIBLE_INSERTION
};

struct AdmissionResult {
    AdmissionDecision decision{AdmissionDecision::REJECT};
    AdmissionReason reason{AdmissionReason::NONE};

    double max_sdf_change{0.0};
    double allowed_sdf_change{0.0};

    // Used only for SLOW_INSERTION
    double insertion_scale{1.0};
};

enum class TopologyChangeType {
    NONE = 0,
    SPLIT,
    MERGE,
    APPEARANCE,
    DISAPPEARANCE
};

struct TopologyCheckResult {
    bool preserved{true};
    TopologyChangeType change{TopologyChangeType::NONE};

    int previous_components{0};
    int candidate_components{0};

    int previous_label{-1};
    int candidate_label{-1};
};

struct SemanticUpdateState {
    SemanticUpdateMode mode{SemanticUpdateMode::NORMAL};

    bool active{false};

    float lambda{1.0f};
    float lambda_dot{0.0f};

    float lambda_dot_min{0.05f};
    float lambda_dot_max{0.5f};
    float max_update_time_sec{20.0f};

    // Temporary until the admission/MPC layer controls semantic update speed.
    float commanded_lambda_dot{0.5f};

    std::chrono::steady_clock::time_point start_time;
    std::chrono::steady_clock::time_point last_update_time;
    std::chrono::steady_clock::time_point evaluation_start_time;
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
    PoissonControllerNode() : Node("semantic_poisson"), sport_req(this) {
        declare_and_load_parameters();

        load_constraints_once_at_startup();
    
        initialize_clocks_and_flags();
        initialize_static_grids();
        allocate_persistent_buffers();
        initialize_robot_kernels();
        initialize_mpc();
        initialize_ros_interfaces();
        publish_semantic_perception_required(true);
        startup_robot();

        initialize_constraint_reload_timer();
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
        if (semantic_class_slices_) std::free(semantic_class_slices_);

        if (robot_kernel_human) std::free(robot_kernel_human);
        if (robot_kernel_obstacle) std::free(robot_kernel_obstacle);
        if (hgrid_insertion_old_) std::free(hgrid_insertion_old_);
        if (hgrid_active_) std::free(hgrid_active_);
        if (dhdt_active_) std::free(dhdt_active_);
        if (beta_grid_) std::free(beta_grid_);

        if (outFileCSV.is_open()) outFileCSV.close();
        if (outFileBIN.is_open()) outFileBIN.close();
        if (outFileMPCVel.is_open()) outFileMPCVel.close();
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

    void semantic_safety_target_callback(
        nav_msgs::msg::OccupancyGrid::UniquePtr msg
    ) {
        if (!msg) {
            return;
        }

        if (msg->data.size() != static_cast<std::size_t>(IMAX * JMAX)) {
            RCLCPP_WARN(
                this->get_logger(),
                "semantic_safety_target size mismatch: got %zu expected %d",
                msg->data.size(),
                IMAX * JMAX
            );
            return;
        }

        if (std::abs(msg->info.resolution - DS) > 1.0e-5f) {
            RCLCPP_WARN(
                this->get_logger(),
                "semantic_safety_target resolution mismatch: got %.6f expected %.6f",
                msg->info.resolution,
                static_cast<double>(DS)
            );
            return;
        }

        if (msg->info.width != static_cast<std::uint32_t>(JMAX) ||
            msg->info.height != static_cast<std::uint32_t>(IMAX)) {
            RCLCPP_WARN(
                this->get_logger(),
                "semantic_safety_target geometry mismatch: got %ux%u expected %dx%d",
                msg->info.width,
                msg->info.height,
                JMAX,
                IMAX
            );
            return;
        }

        const double expected_origin_x = -0.5 * static_cast<double>(JMAX) * DS;
        const double expected_origin_y = -0.5 * static_cast<double>(IMAX) * DS;

        if (std::abs(msg->info.origin.position.x - expected_origin_x) > 1.0e-3 ||
            std::abs(msg->info.origin.position.y - expected_origin_y) > 1.0e-3) {
            RCLCPP_WARN(
                this->get_logger(),
                "semantic_safety_target origin mismatch: got (%.3f, %.3f) expected (%.3f, %.3f)",
                msg->info.origin.position.x,
                msg->info.origin.position.y,
                expected_origin_x,
                expected_origin_y
            );
            return;
        }

        std::vector<int8_t> incoming_semantic_grid(IMAX * JMAX, 0);

        int occupied_cells = 0;
        for (int n = 0; n < IMAX * JMAX; ++n) {
            const bool occupied =
                static_cast<int>(msg->data[n]) >=
                semantic_safety_occupied_threshold_;

            incoming_semantic_grid[n] = occupied ? 1 : 0;
            occupied_cells += occupied ? 1 : 0;
        }

        // After an admission rejection the external fuser may publish one or
        // more frames generated from the rejected JSON before it notices the
        // file rollback.  Do not allow those stale candidate frames to leak
        // into NORMAL operation.
        if (rollback_waiting_for_external_refresh_ &&
            !rejected_candidate_external_grid_.empty() &&
            incoming_semantic_grid == rejected_candidate_external_grid_) {

            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                1000,
                "Ignoring stale semantic_safety_target generated from the rejected candidate while waiting for the fuser to reload the admitted constraints"
            );
            return;
        }

        if (rollback_waiting_for_external_refresh_) {
            rollback_waiting_for_external_refresh_ = false;

            RCLCPP_INFO(
                this->get_logger(),
                "Received refreshed semantic target after constraint rollback"
            );
        }

        external_semantic_safety_grid_ =
            std::move(incoming_semantic_grid);

        external_semantic_safety_received_ = true;
        external_semantic_safety_timestamp_ =
            std::chrono::steady_clock::now();

        if (semantic_update_.mode == SemanticUpdateMode::EVALUATING &&
            external_semantic_safety_timestamp_ >=
                semantic_update_.evaluation_start_time) {
            semantic_candidate_target_ready_ = true;
        }

        RCLCPP_INFO_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            1000,
            "Received combined semantic safety target with %d occupied cells",
            occupied_cells
        );
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
        maybe_write_mpc_command_velocities();
    }

    void handle_occupancy_update(const nav_msgs::msg::OccupancyGrid& msg) {
        const auto grid_start = std::chrono::steady_clock::now();
    
        if (!update_grid_metadata_from_message(msg)) {
            return;
        }

        update_semantic_update_state();
        preprocess_occupancy();
        auto semantic_output = run_semantic_fusion();

        build_inflated_boundaries(semantic_output.tight_area);

        auto guidance_output = build_guidance_field(semantic_output.active_tracks);

        // Publish the exact orientation slice of the boundary array that is
        // about to be passed to the Poisson safety-field solve.
        publish_poisson_solver_boundary(guidance_output.bound_guidance);

        bool solved = solve_safety_field(guidance_output);

        if (start_flag && dhdt_flag) {
            ScopedTimer timer(timing_.dhdt_update_ms);
            update_temporal_field_derivative();
        }

        if (solved) {
            copy_current_globals_into_pending_field();

            {
                std::unique_lock<std::shared_mutex> lock(field_mutex_);
                std::swap(active_field_, pending_field_);
                latest_field_timestamp_ = active_field_.timestamp;
                h_flag = active_field_.valid;
            }
        }
    
        timing_.end_to_end_grid_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - grid_start).count();

        // RCLCPP_ERROR_THROTTLE(
        //     this->get_logger(),
        //     *this->get_clock(),
        //     1000,
        //     "Before render: enable_display=%d h_flag=%d hgrid_active=%p",
        //     enable_display,
        //     h_flag,
        //     static_cast<void*>(hgrid_active_)
        // );
    
        if (enable_display) render_visualization();
        
        if (should_publish_logging_now()) {
            publish_logging_data();
            publish_profiling_data();
        }
    }




    void handle_state_update(const nav_msgs::msg::Odometry& data) {
        update_robot_state(data);
    
        std::vector<float> v_input_body = form_nominal_body_command();
    
        {
            std::shared_lock<std::shared_mutex> lock(field_mutex_);

            timing_.field_data_age_ms =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - active_field_.timestamp
                ).count();

            ScopedTimer timer(timing_.realtime_filter_ms);

            if (active_field_.valid) {
                float* old_hgrid_active = hgrid_active_;
                float* old_dhdt_active = dhdt_active_;
                float* old_beta_grid = beta_grid_;
                float* old_guidance_x_grid = guidance_x_grid;
                float* old_guidance_y_grid = guidance_y_grid;

                hgrid_active_ = active_field_.hgrid.data();
                dhdt_active_ = active_field_.dhdt.data();
                beta_grid_ = active_field_.beta.data();
                guidance_x_grid = active_field_.guidance_x.data();
                guidance_y_grid = active_field_.guidance_y.data();

                compute_realtime_safe_control(v_input_body);

                hgrid_active_ = old_hgrid_active;
                dhdt_active_ = old_dhdt_active;
                beta_grid_ = old_beta_grid;
                guidance_x_grid = old_guidance_x_grid;
                guidance_y_grid = old_guidance_y_grid;
            } else {
                v = v_input_body;
            }
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
    
        std::shared_lock<std::shared_mutex> field_lock(field_mutex_);
        ScopedTimer timer(timing_.predictive_control_ms);

        if (active_field_.valid) {
            float* old_hgrid_active = hgrid_active_;
            float* old_dhdt_grid = dhdt_grid;
            float* old_beta_grid = beta_grid_;
            float* old_guidance_x_grid = guidance_x_grid;
            float* old_guidance_y_grid = guidance_y_grid;

            hgrid_active_ = active_field_.hgrid.data();
            dhdt_grid = active_field_.dhdt.data();
            beta_grid_ = active_field_.beta.data();
            guidance_x_grid = active_field_.guidance_x.data();
            guidance_y_grid = active_field_.guidance_y.data();

            compute_predictive_control();

            hgrid_active_ = old_hgrid_active;
            dhdt_grid = old_dhdt_grid;
            beta_grid_ = old_beta_grid;
            guidance_x_grid = old_guidance_x_grid;
            guidance_y_grid = old_guidance_y_grid;
        }
    }

    // ============================================================
    // 3. PIPELINE: OCCUPANCY / SEMANTICS / GEOMETRY
    // ============================================================

    void preprocess_occupancy() {
        ScopedTimer timer(timing_.occupancy_preprocess_ms);

        if (physical_occ_previous_.size() !=
            static_cast<std::size_t>(IMAX * JMAX)) {
            physical_occ_previous_.assign(IMAX * JMAX, 1.0f);
            physical_occ_current_.assign(IMAX * JMAX, 1.0f);
            physical_change_mask_.assign(IMAX * JMAX, 0);
        }

        std::memcpy(
            physical_occ_previous_.data(),
            occ0,
            IMAX * JMAX * sizeof(float)
        );

        build_occ_map(occ1, occ0, conf);

        std::memcpy(
            physical_occ_current_.data(),
            occ1,
            IMAX * JMAX * sizeof(float)
        );

        build_physical_change_mask();

        // Only copy layer q=0 instead of entire grid
        std::memcpy(
            hgrid_temp_,
            hgrid1,
            IMAX * JMAX * QMAX * sizeof(float)
        );
    
        // Serial path; slice 0's scratch is free at this point.
        find_boundary(hgrid_temp_, occ1, false, false, nullptr, boundary_temp_);
    }

    SemanticStageOutput run_semantic_fusion() {
        ScopedTimer timer(timing_.semantic_fusion_ms);
        SemanticStageOutput out;

        std::fill(semantic_target_grid_.begin(), semantic_target_grid_.end(), 0);

        // Keep the legacy class-map/human-tracker path active for relational
        // constraints and as a fallback when the external fuser is absent.
        label_human_clusters(occ1);

        const auto now = std::chrono::steady_clock::now();
        const double semantic_age_sec = external_semantic_safety_received_
            ? std::chrono::duration<double>(
                  now - external_semantic_safety_timestamp_
              ).count()
            : std::numeric_limits<double>::infinity();

        const bool external_target_is_fresh =
            external_semantic_safety_received_ &&
            (semantic_safety_max_age_sec_ <= 0.0 ||
             semantic_age_sec <= semantic_safety_max_age_sec_);

        if (external_target_is_fresh) {
            semantic_target_grid_ = external_semantic_safety_grid_;
        } else {
            for (int n = 0; n < IMAX * JMAX; ++n) {
                semantic_target_grid_[n] = class_map_expanded[n] > 0 ? 1 : 0;
            }

            if (external_semantic_safety_received_) {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(),
                    *this->get_clock(),
                    2000,
                    "Combined semantic safety target is stale (age=%.3f s, limit=%.3f s); using legacy class_map_expanded fallback",
                    semantic_age_sec,
                    semantic_safety_max_age_sec_
                );
            }
        }

        out.active_tracks = human_tracker_->get_active_tracks();

        semantic_occupancy_grid_.swap(semantic_target_grid_);
        apply_relational_constraints_to_semantic_map(out.active_tracks);
        semantic_target_grid_.swap(semantic_occupancy_grid_);

        if (semantic_update_.mode == SemanticUpdateMode::EVALUATING &&
            semantic_candidate_target_ready_) {
            evaluate_and_begin_semantic_transition();
        }

        update_interpolated_semantic_grid();
        publish_semantic_occupancy_grid();

        out.tight_area = is_tight_area();
        return out;
    }

    void apply_relational_constraints_to_semantic_map(
        const std::vector<HumanTrack>& tracks
    ) {
        std::fill(relational_debug_grid_.begin(), relational_debug_grid_.end(), 0);
        if (tracks.empty()) {
            return;
        }

        int added_cells_total = 0;

        const ConstraintRuntimeConfig& geometry_config =
            candidate_constraint_pending_
                ? candidate_constraint_config_
                : admitted_constraint_config_;

        for (const auto& rc : geometry_config.constraints) {
            if (!rc.enabled || !rc.enforce) {
                continue;
            }

            if (rc.type != ConstraintType::Relational) {
                continue;
            }

            bool target_is_robot = false;
            for (const auto& target : rc.target_classes) {
                if (target == "robot") {
                    target_is_robot = true;
                    break;
                }
            }

            bool reference_is_person = false;
            for (const auto& ref : rc.reference_classes) {
                if (ref == "person" || ref == "human") {
                    reference_is_person = true;
                    break;
                }
            }

            if (!target_is_robot || !reference_is_person) {
                continue;
            }

            const float min_radius_m =
                rc.min_radius_m > 0.0f ? rc.min_radius_m : 0.0f;

            const float max_radius_m =
                rc.max_radius_m > 0.0f ?
                    rc.max_radius_m :
                    (rc.radius_m > 0.0f ? rc.radius_m : 2.0f);

            const float cone_half_angle_deg =
                rc.cone_half_angle_deg > 0.0f ? rc.cone_half_angle_deg : 60.0f;

            const float cone_half_angle_rad =
                cone_half_angle_deg * static_cast<float>(M_PI) / 180.0f;

            const float cos_cone =
                std::cos(cone_half_angle_rad);

            const float heading_timeout_sec =
                rc.heading_timeout_sec > 0.0f ? rc.heading_timeout_sec : 5.0f;

            const float now_sec = human_tracker_->get_current_time();

            int added_cells_for_constraint = 0;
            int tracks_seen = 0;
            int tracks_heading_valid = 0;
            int tracks_not_timed_out = 0;
            int cells_in_radius = 0;
            int cells_in_selected_halfspace = 0;
            int cells_in_selected_region = 0;
            int cells_marked_forbidden = 0;

            for (const auto& track : tracks) {
                tracks_seen++;

                if (!track.heading_valid) {
                    continue;
                }

                if (!track.yolo_confirmed && !track.yolo_ever_confirmed) {
                    continue;
                }

                if (track.confidence < 0.15f) {
                    continue;
                }

                tracks_heading_valid++;

                const float heading_age =
                    now_sec - track.last_update_time;

                if (heading_age > heading_timeout_sec) {
                    continue;
                }

                tracks_not_timed_out++;

                float hx = track.heading_x;
                float hy = track.heading_y;

                const float hnorm = std::sqrt(hx * hx + hy * hy);
                if (hnorm < 1.0e-3f) {
                    continue;
                }

                hx /= hnorm;
                hy /= hnorm;

                float cone_x = hx;
                float cone_y = hy;

                if (rc.relation == "behind") {
                    cone_x = -hx;
                    cone_y = -hy;
                } else if (rc.relation == "in_front_of") {
                    cone_x = hx;
                    cone_y = hy;
                } else if (rc.relation == "left_of") {
                    cone_x = -hy;
                    cone_y = hx;
                } else if (rc.relation == "right_of") {
                    cone_x = hy;
                    cone_y = -hx;
                } else {
                    continue;
                }

                const std::string mode =
                    rc.mode.empty() ? "forbid_region" : rc.mode;

                for (int i = 0; i < IMAX; ++i) {
                    for (int j = 0; j < JMAX; ++j) {
                        const int n = i * JMAX + j;

                        const float cell_x =
                            (static_cast<float>(j) -
                            0.5f * static_cast<float>(JMAX)) * DS;

                        const float cell_y =
                            (static_cast<float>(i) -
                            0.5f * static_cast<float>(IMAX)) * DS;

                        const float rx = cell_x - track.x;
                        const float ry = cell_y - track.y;

                        const float dist =
                            std::sqrt(rx * rx + ry * ry);

                        if (dist < min_radius_m ||
                            dist > max_radius_m) {
                            continue;
                        }

                        cells_in_radius++;

                        const float dot =
                            rx * cone_x + ry * cone_y;

                        if (dot > 0.0f) {
                            cells_in_selected_halfspace++;
                        }

                        const float cos_angle =
                            dot / dist;

                        const bool inside_selected_region =
                            (dot > 0.0f) && (cos_angle >= cos_cone);

                        if (inside_selected_region) {
                            cells_in_selected_region++;
                        }

                        bool mark_forbidden = false;

                        if (mode == "forbid_region") {
                            mark_forbidden = inside_selected_region;
                        } else if (mode == "allow_region") {
                            mark_forbidden = !inside_selected_region;
                        } else {
                            continue;
                        }

                        if (inside_selected_region) {
                            relational_debug_grid_[n] = 1;
                        }

                        if (mark_forbidden) {
                            relational_debug_grid_[n] = 100;
                        }

                        if (!mark_forbidden) {
                            continue;
                        }

                        cells_marked_forbidden++;

                        if (semantic_occupancy_grid_[n] != 1) {
                            added_cells_for_constraint++;
                        }

                        semantic_occupancy_grid_[n] = 1;
                    }
                }
            }

            added_cells_total += added_cells_for_constraint;

            // RCLCPP_INFO_THROTTLE(
            //     this->get_logger(),
            //     *this->get_clock(),
            //     1000,
            //     "Relational '%s': relation=%s, mode=%s, tracks=%d, heading_valid=%d, not_timed_out=%d, in_radius=%d, selected_region=%d, forbidden=%d, added_cells=%d",
            //     rc.id.c_str(),
            //     rc.relation.c_str(),
            //     rc.mode.c_str(),
            //     tracks_seen,
            //     tracks_heading_valid,
            //     tracks_not_timed_out,
            //     cells_in_radius,
            //     cells_in_selected_region,
            //     cells_marked_forbidden,
            //     added_cells_for_constraint
            // );
        }

        nav_msgs::msg::OccupancyGrid msg;
        msg.header.stamp = rclcpp::Time(0);
        msg.header.frame_id = "body_link";

        msg.info.resolution = DS;
        msg.info.width = JMAX;
        msg.info.height = IMAX;

        msg.info.origin.position.x = -0.5f * JMAX * DS;
        msg.info.origin.position.y = -0.5f * IMAX * DS;
        msg.info.origin.position.z = 0.0;

        msg.info.origin.orientation.w = 1.0;

        msg.data.assign(
            relational_debug_grid_.begin(),
            relational_debug_grid_.end()
        );

        relational_debug_pub_->publish(msg);

        if (added_cells_total > 0) {
            // RCLCPP_INFO_THROTTLE(
            //     this->get_logger(),
            //     *this->get_clock(),
            //     1000,
            //     "Relational constraints added total cells=%d",
            //     added_cells_total
            // );
        }
    }

    // The semantic layer carries forbidden regions that have no physical LiDAR
    // return (e.g. the exclusion zones published on /semantic_safety_target), so
    // they have to be stamped into the boundary array with the same sign
    // convention as occ1 (-1 = occupied) before inflation. Otherwise
    // inflate_occupancy_grid() skips them, since it only expands cells that are
    // already occupied and uses the class map purely to select the kernel.
    void stamp_semantic_cells_as_occupied(float* bound_slice) {
        for (int n = 0; n < IMAX * JMAX; ++n) {
            if (semantic_current_grid_[n] > 0) {
                bound_slice[n] = -1.0f;
            }
        }
    }

    void build_inflated_boundaries(bool tight_area) {
        ScopedTimer timer(timing_.geometry_shaping_ms);

        // Seed every yaw slice with its own copy of the semantic class map.
        // inflate_occupancy_grid() dilates these labels using that slice's
        // orientation-specific kernel, so each slice must start from the same
        // clean input and dilate into private storage. Sharing one map made the
        // result depend on OpenMP scheduling.
        for (int q = 0; q < QMAX; ++q) {
            std::memcpy(semantic_class_slices_ + q * IMAX * JMAX,
                        semantic_current_grid_.data(),
                        IMAX * JMAX * sizeof(int8_t));
        }

        #pragma omp parallel for num_threads(4)
        for (int q = 0; q < QMAX; ++q) {
            const int offset = q * IMAX * JMAX;

            float* bound_slice = bound + offset;
            float* hgrid_slice = hgrid_temp_ + offset;
            int8_t* class_slice = semantic_class_slices_ + offset;

            float* bound_scratch = inflate_bound_temp_ + offset;
            int8_t* class_scratch = inflate_class_temp_ + offset;
            float* boundary_scratch = boundary_temp_ + offset;

            std::memcpy(bound_slice, occ1, IMAX * JMAX * sizeof(float));
            stamp_semantic_cells_as_occupied(bound_slice);
            inflate_occupancy_grid(bound_slice, class_slice,
                                   bound_scratch, class_scratch);

            find_boundary(hgrid_slice, bound_slice, true, tight_area,
                          class_slice, boundary_scratch);
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
        
            // Each slice reads the class map that was dilated alongside its own
            // boundary in build_inflated_boundaries().
            compute_boundary_gradients(guidance_x_temp_, guidance_y_temp_, bound,
                                       semantic_class_slices_,
                                       x[0], x[1], vn_body_x, vn_body_y, true);

            #pragma omp parallel for num_threads(4)
            for (int q = 1; q < QMAX; ++q) {
                float* bound_slice = bound + q * IMAX * JMAX;
                float* gx = guidance_x_temp_ + q * IMAX * JMAX;
                float* gy = guidance_y_temp_ + q * IMAX * JMAX;
                compute_boundary_gradients(gx, gy, bound_slice,
                                           semantic_class_slices_ + q * IMAX * JMAX,
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
        
        compute_guidance_forcing(out.bound_guidance);

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
        const float v_RelTol = 1.0e-3f;
        const int N_guidance = IMAX / 5;
        const float w_SOR_guidance = 2.0f / (1.0f + std::sin(M_PI / static_cast<float>(N_guidance + 1)));
        (void)Kernel::poissonSolve(guidance_x_temp_, forcing_zero_temp_, bound_guidance, v_RelTol, w_SOR_guidance);
        (void)Kernel::poissonSolve(guidance_y_temp_, forcing_zero_temp_, bound_guidance, v_RelTol, w_SOR_guidance);
    }

    void compute_guidance_forcing(const float* bound_guidance) {
        #pragma omp parallel for num_threads(4)
        for (int q = 0; q < QMAX; ++q) {
            float* force_slice = force + q * IMAX * JMAX;
            const float* bound_slice = bound_guidance + q * IMAX * JMAX;
            float* gx = guidance_x_temp_ + q * IMAX * JMAX;
            float* gy = guidance_y_temp_ + q * IMAX * JMAX;
            compute_optimal_forcing_function(force_slice, gx, gy, bound_slice);
            for (int n = 0; n < IMAX * JMAX; ++n) force_slice[n] *= DS * DS;
        }
    }

    bool solve_safety_field(const GuidanceStageOutput& guidance){
        ScopedTimer timer(timing_.safety_field_solve_ms);
    
        const float relTol = 1.0e-3f;
        const int N = IMAX / 5;
        const float w_SOR = 2.0f / (1.0f + std::sin(M_PI / static_cast<float>(N + 1)));
    
        bool success = true;
        if (!hgrid_temp_ || !force || !guidance.bound_guidance) {
            success = false;
        } else {
            (void)Kernel::poissonSolve(hgrid_temp_, force, guidance.bound_guidance, relTol, w_SOR);
            // optional: add finite-value checks here
}
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
        const float safe_dt = std::max(dt_grid, 1.0e-4f);
        const float kc = 1.0f - std::exp(-wc * safe_dt);

        const bool semantic_transition_active =
            semantic_update_.mode == SemanticUpdateMode::INSERTING_CONSTRAINT ||
            semantic_update_.mode == SemanticUpdateMode::REMOVING_CONSTRAINT ||
            semantic_update_.mode == SemanticUpdateMode::TRANSITIONING_CONSTRAINT;

        const float semantic_boundary_speed =
            semantic_transition_active
                ? std::max(0.0f, semantic_transition_extent_m_) *
                  std::abs(semantic_update_.lambda_dot)
                : 0.0f;

        for (int i = 0; i < IMAX; ++i) {
            for (int j = 0; j < JMAX; ++j) {
                const int n2 = i * JMAX + j;
                const bool physical_motion_nearby =
                    !physical_change_mask_.empty() &&
                    physical_change_mask_[n2] != 0;

                for (int q = 0; q < QMAX; ++q) {
                    const int n3 = q * IMAX * JMAX + n2;

                    const float i0 = static_cast<float>(i) + dx[1] / DS;
                    const float j0 = static_cast<float>(j) + dx[0] / DS;

                    const bool in_grid =
                        (i0 >= 0.0f) &&
                        (i0 <= static_cast<float>(IMAX - 1)) &&
                        (j0 >= 0.0f) &&
                        (j0 <= static_cast<float>(JMAX - 1));

                    float dhdt_ij = 0.0f;

                    if (in_grid) {
                        const float h0v =
                            trilinear_interpolation(hgrid0, i0, j0, q);
                        const float h1v =
                            trilinear_interpolation(hgrid1, i, j, q);

                        dhdt_ij = (h1v - h0v) / safe_dt;

                        if (semantic_transition_active &&
                            !physical_motion_nearby) {

                            const int im = std::max(i - 1, 0);
                            const int ip = std::min(i + 1, IMAX - 1);
                            const int jm = std::max(j - 1, 0);
                            const int jp = std::min(j + 1, JMAX - 1);

                            const float hx =
                                (hgrid1[q * IMAX * JMAX + i * JMAX + jp] -
                                 hgrid1[q * IMAX * JMAX + i * JMAX + jm]) /
                                (static_cast<float>(jp - jm) * DS + 1.0e-6f);

                            const float hy =
                                (hgrid1[q * IMAX * JMAX + ip * JMAX + j] -
                                 hgrid1[q * IMAX * JMAX + im * JMAX + j]) /
                                (static_cast<float>(ip - im) * DS + 1.0e-6f);

                            const float grad_norm =
                                std::sqrt(hx * hx + hy * hy);

                            const float semantic_rate_limit =
                                semantic_boundary_speed * grad_norm;

                            dhdt_ij = std::clamp(
                                dhdt_ij,
                                -semantic_rate_limit,
                                semantic_rate_limit
                            );
                        }
                    }

                    dhdt_grid[n3] *= 1.0f - kc;
                    dhdt_grid[n3] += kc * dhdt_ij;
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

            mpc3d_controller.set_alpha_optimization_enabled(
                semantic_update_.active,
                std::exp(-wn * DT)
            );

            mpc3d_controller.update_constraints(hgrid_active_, dhdt_grid, beta_grid_, guidance_x_grid, guidance_y_grid,
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
        float runtime_vx_fwd = vel_max_x_fwd_;
        float runtime_vx_bwd = vel_max_x_bwd_;
        float runtime_vy = vel_max_y_;
        float runtime_wz = vel_max_yaw_;

        apply_velocity_limit_constraints(
            runtime_vx_fwd,
            runtime_vx_bwd,
            runtime_vy,
            runtime_wz
        );
    
        vb[0] = std::clamp(vb[0], -runtime_vx_bwd, runtime_vx_fwd);
        vb[1] = std::clamp(vb[1], -runtime_vy, runtime_vy);
        vb[2] = std::clamp(vb[2], -runtime_wz, runtime_wz);
        
    }

    void apply_velocity_limit_constraints(
        float& vx_fwd,
        float& vx_bwd,
        float& vy,
        float& wz
    ) {
        if (!human_tracker_) {
            return;
        }
    
        const auto tracks = human_tracker_->get_active_tracks();
    
        if (tracks.empty()) {
            return;
        }
    
        for (const auto& rc : constraint_runtime_config_.constraints) {
            if (!rc.enabled || !rc.enforce) {
                continue;
            }
    
            if (rc.type != ConstraintType::VelocityLimit) {
                continue;
            }
    
            bool targets_person = false;
            for (const auto& target : rc.target_classes) {
                if (target == "person" || target == "human") {
                    targets_person = true;
                    break;
                }
            }
    
            if (!targets_person) {
                continue;
            }
    
            if (rc.buffer_distance_m <= 0.0f) {
                continue;
            }
    
            bool near_person = false;
    
            for (const auto& track : tracks) {
                const float d = std::sqrt(track.x * track.x + track.y * track.y);
    
                if (d <= rc.buffer_distance_m) {
                    near_person = true;
                    break;
                }
            }
    
            if (!near_person) {
                continue;
            }
    
            if (rc.max_linear_velocity_mps > 0.0f) {
                vx_fwd = std::min(vx_fwd, rc.max_linear_velocity_mps);
                vx_bwd = std::min(vx_bwd, rc.max_linear_velocity_mps);
                vy = std::min(vy, rc.max_linear_velocity_mps);
            }
    
            if (rc.max_angular_velocity_radps > 0.0f) {
                wz = std::min(wz, rc.max_angular_velocity_radps);
            }
    
        //     RCLCPP_INFO_THROTTLE(
        //         this->get_logger(),
        //         *this->get_clock(),
        //         1000,
        //         "Velocity limit active from constraint '%s': linear<=%.2f, yaw<=%.2f",
        //         rc.id.c_str(),
        //         vx_fwd,
        //         wz
        //     );
        // }
        }
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

    void publish_poisson_solver_boundary(const float* solver_bound) {
        if (!poisson_solver_boundary_pub_ || !solver_bound) {
            return;
        }

        nav_msgs::msg::OccupancyGrid msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = "body_link";

        msg.info.resolution = DS;
        msg.info.width = JMAX;
        msg.info.height = IMAX;
        msg.info.origin.position.x =
            -0.5 * static_cast<double>(JMAX) * static_cast<double>(DS);
        msg.info.origin.position.y =
            -0.5 * static_cast<double>(IMAX) * static_cast<double>(DS);
        msg.info.origin.position.z = 0.0;
        msg.info.origin.orientation.x = 0.0;
        msg.info.origin.orientation.y = 0.0;
        msg.info.origin.orientation.z = 0.0;
        msg.info.origin.orientation.w = 1.0;

        msg.data.resize(IMAX * JMAX);

        // The Poisson domain has one 2-D boundary slice per discretized yaw.
        // Visualize the slice corresponding to the robot's current yaw.
        const float q_float = yaw_to_q(x[2], xc[2]);
        const int q_vis = static_cast<int>(q_wrap(std::round(q_float)));
        const float* bound_slice =
            solver_bound + q_vis * IMAX * JMAX;

        int occupied_cells = 0;
        for (int n = 0; n < IMAX * JMAX; ++n) {
            // Preserve the same sign convention used by render_visualization():
            // bound <= 0 is a forbidden/boundary cell for the Poisson solver.
            const bool forbidden = bound_slice[n] <= 0.0f;
            msg.data[n] = forbidden ? 100 : 0;
            occupied_cells += forbidden ? 1 : 0;
        }

        poisson_solver_boundary_pub_->publish(msg);

        RCLCPP_INFO_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            2000,
            "Published Poisson solver boundary: q=%d, forbidden=%d, free=%d",
            q_vis,
            occupied_cells,
            IMAX * JMAX - occupied_cells
        );
    }

    void render_visualization() {
        if (!poisson_image_pub_) {
            return;
        }

        std::shared_lock<std::shared_mutex> lock(field_mutex_);

        if (!active_field_.valid ||
            active_field_.hgrid.empty() ||
            active_field_.bound.empty()) {
            return;
        }

        const float* hgrid_vis = active_field_.hgrid.data();
        const float* bound_vis = active_field_.bound.data();

        const int q_vis = QMAX / 2;
        const int scale = 6;

        cv::Mat h_u8(IMAX, JMAX, CV_8UC1);
        cv::Mat boundary_u8(IMAX, JMAX, CV_8UC1, cv::Scalar(0));

        float h_min = 1.0e9f;
        float h_max = -1.0e9f;

        for (int i = 0; i < IMAX; ++i) {
            for (int j = 0; j < JMAX; ++j) {
                const int n3 = q_vis * IMAX * JMAX + i * JMAX + j;
                const float hv = hgrid_vis[n3];

                h_min = std::min(h_min, hv);
                h_max = std::max(h_max, hv);
            }
        }

        const float display_min = 0.0f;
        const float display_max = std::max(0.40f, h_max);

        for (int i = 0; i < IMAX; ++i) {
            for (int j = 0; j < JMAX; ++j) {
                const int n2 = i * JMAX + j;
                const int n3 = q_vis * IMAX * JMAX + n2;

                float hv = hgrid_vis[n3];
                hv = std::clamp(hv, display_min, display_max);

                const float normalized =
                    (hv - display_min) /
                    (display_max - display_min + 1.0e-6f);

                h_u8.at<uint8_t>(i, j) =
                    static_cast<uint8_t>(255.0f * normalized);

                if (bound_vis[n3] <= 0.0f) {
                    boundary_u8.at<uint8_t>(i, j) = 255;
                }
            }
        }

        cv::Mat h_color;
        cv::applyColorMap(h_u8, h_color, cv::COLORMAP_TURBO);

        cv::Mat boundary_color(IMAX, JMAX, CV_8UC3, cv::Scalar(0, 0, 0));

        for (int i = 0; i < IMAX; ++i) {
            for (int j = 0; j < JMAX; ++j) {
                if (boundary_u8.at<uint8_t>(i, j) > 0) {
                    boundary_color.at<cv::Vec3b>(i, j) =
                        cv::Vec3b(255, 255, 255);
                }
            }
        }

        cv::Mat overlay = h_color.clone();

        for (int i = 0; i < IMAX; ++i) {
            for (int j = 0; j < JMAX; ++j) {
                if (boundary_u8.at<uint8_t>(i, j) > 0) {
                    overlay.at<cv::Vec3b>(i, j) =
                        cv::Vec3b(255, 255, 255);
                }
            }
        }

        cv::Mat combined;
        cv::hconcat(
            std::vector<cv::Mat>{h_color, boundary_color, overlay},
            combined
        );

        cv::Mat display_img;
        cv::resize(
            combined,
            display_img,
            cv::Size(),
            scale,
            scale,
            cv::INTER_NEAREST
        );

        sensor_msgs::msg::Image msg;
        msg.header.stamp = this->now();
        msg.header.frame_id = "map";
        msg.height = display_img.rows;
        msg.width = display_img.cols;
        msg.encoding = "bgr8";
        msg.is_bigendian = false;
        msg.step = static_cast<sensor_msgs::msg::Image::_step_type>(
            display_img.cols * display_img.elemSize()
        );

        msg.data.assign(
            display_img.data,
            display_img.data + display_img.total() * display_img.elemSize()
        );

        poisson_image_pub_->publish(msg);
    }

    bool should_publish_logging_now() {
        const auto now = std::chrono::steady_clock::now();

        const double time_since_last =
            std::chrono::duration<double>(
                now - last_logging_publish_time_
            ).count();

        if (time_since_last < logging_publish_period_) {
            return false;
        }

        last_logging_publish_time_ = now;
        return true;
    }

    void publish_profiling_data() {
        if (!profiling_data_pub_) return;
    
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
    
        profiling_data_pub_->publish(msg);
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
                    (q2f - qr) * hgrid_active_[q1 * IMAX * JMAX + n] +
                    (qr - q1f) * hgrid_active_[q2 * IMAX * JMAX + n];
            } else {
                grid_temp[n] = hgrid_active_[q1 * IMAX * JMAX + n];
            }
        }
    }

    void maybe_write_mpc_command_velocities() {
        if (!(save_flag && enable_data_logging_to_file_)) return;
        if (!outFileMPCVel.is_open()) return;
        const std::vector<float> vel_data = {
            t_ms,
            static_cast<float>(space_counter),
            vd[0], vd[1], vd[2],
            v[0],  v[1],  v[2],
            vb[0], vb[1], vb[2]
        };
        for (size_t n = 0; n < vel_data.size(); ++n) {
            outFileMPCVel << vel_data[n];
            if (n + 1 < vel_data.size()) outFileMPCVel << ",";
        }
        outFileMPCVel << std::endl;
    }

    void maybe_write_experiment_data() {
        if (!(save_flag && enable_data_logging_to_file_)) return;
        const std::vector<float> save_data = {
            t_ms,
            static_cast<float>(space_counter),

            x[0], x[1], x[2],

            v[0], v[1], v[2],
            vt[0], vt[1], vt[2],

            h, dhdx, dhdy, dhdq, dhdt,
            wn,
            static_cast<float>(realtime_sf_flag | predictive_sf_flag),

            semantic_update_.lambda,
            semantic_update_.lambda_dot,
            static_cast<float>(semantic_update_.active),
            new_constraint_event_flag_ ? 1.0f : 0.0f,
            static_cast<float>(constraint_event_counter_)
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

    void publish_semantic_occupancy_grid()
    {
        if (!semantic_occupancy_pub_) {
            return;
        }

        nav_msgs::msg::OccupancyGrid msg;

        msg.header.stamp = this->now();
        msg.header.frame_id = "body_link";

        msg.info.resolution = DS;
        msg.info.width = JMAX;
        msg.info.height = IMAX;

        msg.info.origin.position.x = -0.5 * JMAX * DS;
        msg.info.origin.position.y = -0.5 * IMAX * DS;
        msg.info.origin.position.z = 0.0;

        msg.info.origin.orientation.x = 0.0;
        msg.info.origin.orientation.y = 0.0;
        msg.info.origin.orientation.z = 0.0;
        msg.info.origin.orientation.w = 1.0;

        msg.data.resize(IMAX * JMAX);

        int occupied_cells = 0;

        for (int n = 0; n < IMAX * JMAX; ++n)
        {
            if (semantic_current_grid_[n] > 0)
            {
                msg.data[n] = 100;      // Occupied for RViz
                occupied_cells++;
            }
            else
            {
                msg.data[n] = 0;        // Free
            }
        }

        RCLCPP_INFO_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            2000,
            "Publishing semantic occupancy grid with %d occupied cells.",
            occupied_cells
        );

        semantic_occupancy_pub_->publish(msg);
    }

    float distance_to_nearest_occupied_cell(
        const std::vector<int8_t>& grid,
        int i0,
        int j0
    ) const {
        float best = 1.0e6f;

        for (int i = 0; i < IMAX; ++i) {
            for (int j = 0; j < JMAX; ++j) {
                const int n = i * JMAX + j;

                if (grid[n] <= 0) {
                    continue;
                }

                const float di = static_cast<float>(i - i0);
                const float dj = static_cast<float>(j - j0);
                const float d = std::sqrt(di * di + dj * dj) * DS;

                best = std::min(best, d);
            }
        }

        return best;
    }

    float distance_to_nearest_free_cell(
        const std::vector<int8_t>& grid,
        int i0,
        int j0
    ) const {
        float best = 1.0e6f;

        for (int i = 0; i < IMAX; ++i) {
            for (int j = 0; j < JMAX; ++j) {
                const int n = i * JMAX + j;

                if (grid[n] > 0) {
                    continue;
                }

                const float di = static_cast<float>(i - i0);
                const float dj = static_cast<float>(j - j0);
                const float d = std::sqrt(di * di + dj * dj) * DS;

                best = std::min(best, d);
            }
        }

        return best;
    }

    float max_interior_depth(const std::vector<int8_t>& grid) {
        float max_depth = 0.0f;

        for (int i = 0; i < IMAX; ++i) {
            for (int j = 0; j < JMAX; ++j) {
                const int n = i * JMAX + j;
                if (grid[n] <= 0) {
                    continue;
                }

                max_depth = std::max(
                    max_depth,
                    distance_to_nearest_free_cell(grid, i, j)
                );
            }
        }

        return max_depth;
    }

    void build_physical_change_mask() {
        physical_change_mask_.assign(IMAX * JMAX, 0);

        constexpr int kPhysicalChangeRadiusCells = 2;

        for (int i = 0; i < IMAX; ++i) {
            for (int j = 0; j < JMAX; ++j) {
                const int n = i * JMAX + j;

                const bool prev_occ =
                    !physical_occ_previous_.empty() &&
                    physical_occ_previous_[n] <= 0.0f;

                const bool curr_occ =
                    !physical_occ_current_.empty() &&
                    physical_occ_current_[n] <= 0.0f;

                if (prev_occ == curr_occ) {
                    continue;
                }

                for (int di = -kPhysicalChangeRadiusCells;
                     di <= kPhysicalChangeRadiusCells; ++di) {
                    for (int dj = -kPhysicalChangeRadiusCells;
                         dj <= kPhysicalChangeRadiusCells; ++dj) {
                        const int ii = i + di;
                        const int jj = j + dj;

                        if (ii < 0 || ii >= IMAX ||
                            jj < 0 || jj >= JMAX) {
                            continue;
                        }

                        physical_change_mask_[ii * JMAX + jj] = 1;
                    }
                }
            }
        }
    }

    bool has_free_space(const std::vector<int8_t>& occupancy) const {
        if (occupancy.size() != static_cast<std::size_t>(IMAX * JMAX)) {
            return false;
        }

        for (int n = 0; n < IMAX * JMAX; ++n) {
            if (occupancy[n] <= 0) {
                return true;
            }
        }

        return false;
    }

    std::vector<int8_t> build_admission_occupancy(
        const std::vector<int8_t>& semantic_occupancy
    ) const {
        std::vector<int8_t> combined(IMAX * JMAX, 0);

        for (int n = 0; n < IMAX * JMAX; ++n) {
            const bool semantic_forbidden =
                n < static_cast<int>(semantic_occupancy.size()) &&
                semantic_occupancy[n] > 0;

            const bool physical_forbidden =
                n < static_cast<int>(physical_occ_current_.size()) &&
                physical_occ_current_[n] <= 0.0f;

            combined[n] =
                (semantic_forbidden || physical_forbidden) ? 1 : 0;
        }

        return combined;
    }

    std::vector<float> compute_signed_distance_field(
        const std::vector<int8_t>& occupancy
    ) const {
        std::vector<float> sdf(IMAX * JMAX, 0.0f);

        if (occupancy.size() != static_cast<std::size_t>(IMAX * JMAX)) {
            return sdf;
        }

        cv::Mat free_mask(IMAX, JMAX, CV_8UC1);
        cv::Mat occupied_mask(IMAX, JMAX, CV_8UC1);

        bool any_free = false;
        bool any_occupied = false;

        for (int i = 0; i < IMAX; ++i) {
            for (int j = 0; j < JMAX; ++j) {
                const int n = i * JMAX + j;
                const bool occupied = occupancy[n] > 0;

                // cv::distanceTransform computes distance from each nonzero
                // pixel to the nearest zero pixel.
                free_mask.at<uint8_t>(i, j) = occupied ? 0 : 255;
                occupied_mask.at<uint8_t>(i, j) = occupied ? 255 : 0;

                any_occupied = any_occupied || occupied;
                any_free = any_free || !occupied;
            }
        }

        const float grid_diagonal_m =
            std::sqrt(
                static_cast<float>(IMAX * IMAX + JMAX * JMAX)
            ) * DS;

        // A completely free grid has no finite obstacle boundary.  Use a
        // finite positive sentinel so the representation stays well-defined.
        if (!any_occupied) {
            std::fill(sdf.begin(), sdf.end(), grid_diagonal_m);
            return sdf;
        }

        // The all-occupied case should already have been rejected by
        // has_free_space(), but keep this helper numerically well-defined.
        if (!any_free) {
            std::fill(sdf.begin(), sdf.end(), -grid_diagonal_m);
            return sdf;
        }

        cv::Mat distance_in_free;
        cv::Mat distance_in_occupied;

        cv::distanceTransform(
            free_mask,
            distance_in_free,
            cv::DIST_L2,
            cv::DIST_MASK_PRECISE
        );

        cv::distanceTransform(
            occupied_mask,
            distance_in_occupied,
            cv::DIST_L2,
            cv::DIST_MASK_PRECISE
        );

        for (int i = 0; i < IMAX; ++i) {
            for (int j = 0; j < JMAX; ++j) {
                const int n = i * JMAX + j;

                // Positive in admissible/free space, negative in forbidden
                // space, approximately zero on the grid-resolved boundary.
                sdf[n] =
                    (distance_in_free.at<float>(i, j) -
                     distance_in_occupied.at<float>(i, j)) * DS;
            }
        }

        return sdf;
    }

    double max_abs_sdf_difference(
        const std::vector<float>& a,
        const std::vector<float>& b
    ) const {
        if (a.size() != b.size() || a.empty()) {
            return std::numeric_limits<double>::infinity();
        }

        double max_difference = 0.0;

        for (std::size_t n = 0; n < a.size(); ++n) {
            if (!std::isfinite(a[n]) || !std::isfinite(b[n])) {
                return std::numeric_limits<double>::infinity();
            }

            max_difference = std::max(
                max_difference,
                std::abs(
                    static_cast<double>(b[n]) -
                    static_cast<double>(a[n])
                )
            );
        }

        return max_difference;
    }

    ConnectedComponentsData compute_free_space_components(
        const std::vector<int8_t>& forbidden_grid
    ) const {
        ConnectedComponentsData result;

        result.binary = cv::Mat(
            IMAX,
            JMAX,
            CV_8UC1,
            cv::Scalar(0)
        );

        if (forbidden_grid.size() !=
            static_cast<std::size_t>(IMAX * JMAX)) {
            return result;
        }

        // OpenCV treats nonzero pixels as foreground.  For the theorem,
        // foreground is the admissible/free set Omega, not the obstacle set.
        for (int i = 0; i < IMAX; ++i) {
            for (int j = 0; j < JMAX; ++j) {
                const int n = i * JMAX + j;
                const bool forbidden = forbidden_grid[n] > 0;

                result.binary.at<uint8_t>(i, j) =
                    forbidden ? 0 : 255;
            }
        }

        result.num_labels =
            cv::connectedComponentsWithStats(
                result.binary,
                result.labels,
                result.stats,
                result.centroids,
                8,
                CV_32S
            );

        return result;
    }

    bool topology_component_is_significant(
        const ConnectedComponentsData& components,
        int label
    ) const {
        if (label <= 0 ||
            label >= components.num_labels ||
            components.stats.empty()) {
            return false;
        }

        // Ignore tiny free-space islands caused by grid quantization/noise.
        // With DS = 0.05 m, 4 cells correspond to 0.01 m^2.
        constexpr int kMinTopologyComponentCells = 8;

        const int area_cells =
            components.stats.at<int>(
                label,
                cv::CC_STAT_AREA
            );

        return area_cells >= kMinTopologyComponentCells;
    }

    const char* topology_change_string(
        TopologyChangeType change
    ) const {
        switch (change) {
            case TopologyChangeType::NONE:
                return "none";
            case TopologyChangeType::SPLIT:
                return "split";
            case TopologyChangeType::MERGE:
                return "merge";
            case TopologyChangeType::APPEARANCE:
                return "appearance";
            case TopologyChangeType::DISAPPEARANCE:
                return "disappearance";
            default:
                return "unknown";
        }
    }

    TopologyCheckResult check_free_space_topology(
        const std::vector<int8_t>& previous_forbidden,
        const std::vector<int8_t>& candidate_forbidden
    ) const {
        TopologyCheckResult result;

        if (previous_forbidden.size() !=
                static_cast<std::size_t>(IMAX * JMAX) ||
            candidate_forbidden.size() !=
                static_cast<std::size_t>(IMAX * JMAX)) {
            result.preserved = false;
            result.change = TopologyChangeType::DISAPPEARANCE;
            return result;
        }

        const ConnectedComponentsData previous =
            compute_free_space_components(previous_forbidden);

        const ConnectedComponentsData candidate =
            compute_free_space_components(candidate_forbidden);

        if (previous.num_labels <= 0 ||
            candidate.num_labels <= 0 ||
            previous.labels.empty() ||
            candidate.labels.empty()) {
            result.preserved = false;
            result.change = TopologyChangeType::DISAPPEARANCE;
            return result;
        }

        std::set<int> significant_previous;
        std::set<int> significant_candidate;

        for (int label = 1;
             label < previous.num_labels;
             ++label) {
            if (topology_component_is_significant(previous, label)) {
                significant_previous.insert(label);
            }
        }

        for (int label = 1;
             label < candidate.num_labels;
             ++label) {
            if (topology_component_is_significant(candidate, label)) {
                significant_candidate.insert(label);
            }
        }

        result.previous_components =
            static_cast<int>(significant_previous.size());

        result.candidate_components =
            static_cast<int>(significant_candidate.size());

        // Build component correspondence using spatial overlap:
        // previous component -> candidate descendants
        // candidate component -> previous ancestors.
        std::map<int, std::set<int>> previous_to_candidate;
        std::map<int, std::set<int>> candidate_to_previous;

        for (int i = 0; i < IMAX; ++i) {
            for (int j = 0; j < JMAX; ++j) {
                const int previous_label =
                    previous.labels.at<int>(i, j);

                const int candidate_label =
                    candidate.labels.at<int>(i, j);

                if (significant_previous.count(previous_label) == 0 ||
                    significant_candidate.count(candidate_label) == 0) {
                    continue;
                }

                previous_to_candidate[previous_label].insert(
                    candidate_label
                );

                candidate_to_previous[candidate_label].insert(
                    previous_label
                );
            }
        }

        // A previous connected component mapping to multiple significant
        // candidate components is a split.
        for (const int previous_label : significant_previous) {
            const auto it =
                previous_to_candidate.find(previous_label);

            if (it != previous_to_candidate.end() &&
                it->second.size() > 1) {
                result.preserved = false;
                result.change = TopologyChangeType::SPLIT;
                result.previous_label = previous_label;
                result.candidate_label = *it->second.begin();
                return result;
            }
        }

        // Multiple previous components mapping into one significant
        // candidate component is a merge.
        for (const int candidate_label : significant_candidate) {
            const auto it =
                candidate_to_previous.find(candidate_label);

            if (it != candidate_to_previous.end() &&
                it->second.size() > 1) {
                result.preserved = false;
                result.change = TopologyChangeType::MERGE;
                result.candidate_label = candidate_label;
                result.previous_label = *it->second.begin();
                return result;
            }
        }

        // Every significant previous component must have a descendant.
        for (const int previous_label : significant_previous) {
            const auto it =
                previous_to_candidate.find(previous_label);

            if (it == previous_to_candidate.end() ||
                it->second.empty()) {
                result.preserved = false;
                result.change = TopologyChangeType::DISAPPEARANCE;
                result.previous_label = previous_label;
                return result;
            }
        }

        // Every significant candidate component must have a predecessor.
        for (const int candidate_label : significant_candidate) {
            const auto it =
                candidate_to_previous.find(candidate_label);

            if (it == candidate_to_previous.end() ||
                it->second.empty()) {
                result.preserved = false;
                result.change = TopologyChangeType::APPEARANCE;
                result.candidate_label = candidate_label;
                return result;
            }
        }

        return result;
    }

    const char* admission_reason_string(AdmissionReason reason) const {
        switch (reason) {
            case AdmissionReason::NONE:
                return "none";
            case AdmissionReason::BOUNDARY_TOO_FAST:
                return "boundary_too_fast";
            case AdmissionReason::SAFE_SET_EMPTY:
                return "safe_set_empty";
            case AdmissionReason::TOPOLOGY_CHANGE:
                return "topology_change";
            case AdmissionReason::NO_ADMISSIBLE_INSERTION:
                return "no_admissible_insertion";
            default:
                return "unknown";
        }
    }

    AdmissionResult evaluate_candidate_constraint(
        const std::vector<int8_t>& current_semantic_occupancy,
        const std::vector<int8_t>& candidate_semantic_occupancy,
        double dt,
        double v_admissible
    ) {
        AdmissionResult result;

        if (current_semantic_occupancy.size() !=
                static_cast<std::size_t>(IMAX * JMAX) ||
            candidate_semantic_occupancy.size() !=
                static_cast<std::size_t>(IMAX * JMAX)) {
            result.decision = AdmissionDecision::REJECT;
            result.reason = AdmissionReason::NO_ADMISSIBLE_INSERTION;
            return result;
        }

        // Admission is defined on the actual admissible geometry, not the
        // semantic layer by itself.  Physical occupancy is therefore unioned
        // with both the current and candidate semantic occupancies.
        const auto current_occupancy =
            build_admission_occupancy(current_semantic_occupancy);

        const auto candidate_occupancy =
            build_admission_occupancy(candidate_semantic_occupancy);

        // The theorem assumes a nonempty admissible set.
        if (!has_free_space(candidate_occupancy)) {
            result.decision = AdmissionDecision::REJECT;
            result.reason = AdmissionReason::SAFE_SET_EMPTY;
            return result;
        }

        // Topology is checked on the admissible/free set Omega, after
        // physical and semantic forbidden regions have been unioned.
        //
        // This catches splits, merges, appearances, and disappearances of
        // significant free-space components.  Unlike a speed violation,
        // a topology violation cannot be fixed merely by slowing lambda.
        const TopologyCheckResult topology =
            check_free_space_topology(
                current_occupancy,
                candidate_occupancy
            );

        if (!topology.preserved) {
            result.decision = AdmissionDecision::REJECT;
            result.reason = AdmissionReason::TOPOLOGY_CHANGE;

            RCLCPP_WARN(
                this->get_logger(),
                "Admission topology rejection: change=%s, previous_components=%d, candidate_components=%d, previous_label=%d, candidate_label=%d",
                topology_change_string(topology.change),
                topology.previous_components,
                topology.candidate_components,
                topology.previous_label,
                topology.candidate_label
            );

            return result;
        }

        const auto b_current =
            compute_signed_distance_field(current_occupancy);

        const auto b_candidate =
            compute_signed_distance_field(candidate_occupancy);

        result.max_sdf_change =
            max_abs_sdf_difference(b_current, b_candidate);

        const double safe_dt = std::max(dt, 1.0e-4);
        const double safe_v_admissible = std::max(v_admissible, 0.0);

        result.allowed_sdf_change =
            safe_v_admissible * safe_dt;

        if (!std::isfinite(result.max_sdf_change)) {
            result.decision = AdmissionDecision::REJECT;
            result.reason = AdmissionReason::NO_ADMISSIBLE_INSERTION;
            return result;
        }

        if (result.max_sdf_change <=
            result.allowed_sdf_change + 1.0e-9) {
            result.decision = AdmissionDecision::ACCEPT;
            result.reason = AdmissionReason::NONE;
            result.insertion_scale = 1.0;
            return result;
        }

        // The full candidate cannot be committed in one update.  This does
        // not relax the requested constraint; it only slows the homotopy used
        // to reach the same target geometry.
        result.decision = AdmissionDecision::SLOW_INSERTION;
        result.reason = AdmissionReason::BOUNDARY_TOO_FAST;

        if (result.max_sdf_change > 1.0e-9 &&
            result.allowed_sdf_change > 0.0) {
            result.insertion_scale = std::clamp(
                result.allowed_sdf_change /
                    result.max_sdf_change,
                0.0,
                1.0
            );
        } else {
            result.insertion_scale = 0.0;
        }

        if (result.insertion_scale <= 1.0e-6) {
            result.decision = AdmissionDecision::REJECT;
            result.reason = AdmissionReason::NO_ADMISSIBLE_INSERTION;
        }

        return result;
    }

    std::string read_constraints_file_text() const {
        if (constraints_path_.empty()) {
            return {};
        }

        std::ifstream in(
            constraints_path_,
            std::ios::in | std::ios::binary
        );

        if (!in.is_open()) {
            return {};
        }

        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }

    std::size_t constraints_text_signature(
        const std::string& text_value
    ) const {
        return std::hash<std::string>{}(text_value);
    }

    bool write_constraints_file_atomically(
        const std::string& contents
    ) const {
        if (constraints_path_.empty()) {
            return false;
        }

        const std::string temp_path =
            constraints_path_ + ".admission_rollback.tmp";

        {
            std::ofstream out(
                temp_path,
                std::ios::out |
                std::ios::binary |
                std::ios::trunc
            );

            if (!out.is_open()) {
                return false;
            }

            out << contents;
            out.flush();

            if (!out.good()) {
                out.close();
                std::remove(temp_path.c_str());
                return false;
            }
        }

        if (std::rename(
                temp_path.c_str(),
                constraints_path_.c_str()) != 0) {
            std::remove(temp_path.c_str());
            return false;
        }

        return true;
    }

    void commit_candidate_constraint_config() {
        if (!candidate_constraint_pending_) {
            return;
        }

        admitted_constraint_config_ =
            candidate_constraint_config_;

        constraint_runtime_config_ =
            admitted_constraint_config_;

        admitted_constraints_json_ =
            candidate_constraints_json_;

        admitted_constraints_file_signature_ =
            candidate_constraints_file_signature_;

        apply_runtime_constraint_config(
            admitted_constraint_config_,
            true
        );

        candidate_constraint_pending_ = false;
        candidate_constraints_json_.clear();
        candidate_constraints_file_signature_ = 0;

        admitted_external_semantic_grid_snapshot_ =
            external_semantic_safety_grid_;

        publish_semantic_perception_required();

        RCLCPP_INFO(
            this->get_logger(),
            "Candidate constraint set COMMITTED; it is now the admitted runtime configuration"
        );
    }

    void rollback_candidate_constraint_config(
        AdmissionReason reason
    ) {
        if (!candidate_constraint_pending_) {
            return;
        }

        // Keep a fingerprint of the rejected fuser output so stale messages
        // produced before the external node reloads the restored JSON cannot
        // leak into the active semantic map.
        rejected_candidate_external_grid_ =
            external_semantic_safety_grid_;

        constraint_runtime_config_ =
            admitted_constraint_config_;

        apply_runtime_constraint_config(
            admitted_constraint_config_,
            true
        );

        bool restored_file = true;

        if (!admitted_constraints_json_.empty()) {
            restored_file =
                write_constraints_file_atomically(
                    admitted_constraints_json_
                );
        }

        if (restored_file) {
            last_constraints_file_signature_ =
                admitted_constraints_file_signature_;
        }

        candidate_constraint_pending_ = false;
        candidate_constraints_json_.clear();
        candidate_constraints_file_signature_ = 0;

        // Bridge the short interval before semantic_map_fuser reloads the
        // restored JSON using the last target it produced under the admitted
        // configuration.  Once a different external target arrives, normal
        // live perception resumes.
        if (!admitted_external_semantic_grid_snapshot_.empty()) {
            external_semantic_safety_grid_ =
                admitted_external_semantic_grid_snapshot_;

            external_semantic_safety_received_ = true;
            external_semantic_safety_timestamp_ =
                std::chrono::steady_clock::now();
        }

        rollback_waiting_for_external_refresh_ =
            restored_file;

        publish_semantic_perception_required();

        if (!restored_file) {
            RCLCPP_ERROR(
                this->get_logger(),
                "Constraint admission rollback failed to restore the admitted JSON file. reason=%s",
                admission_reason_string(reason)
            );
        } else {
            RCLCPP_WARN(
                this->get_logger(),
                "Candidate constraint set ROLLED BACK: reason=%s. Restored the last admitted JSON configuration.",
                admission_reason_string(reason)
            );
        }
    }

    void begin_semantic_evaluation() {
        new_constraint_event_flag_ = true;
        constraint_event_counter_++;

        semantic_previous_grid_ = semantic_current_grid_;

        // Preserve the last target generated under the admitted rule set.
        // This is only a short rollback bridge; the external fuser will
        // regenerate the admitted geometry from live perception after the
        // JSON file is restored.
        admitted_external_semantic_grid_snapshot_ =
            external_semantic_safety_grid_;

        semantic_update_.mode = SemanticUpdateMode::EVALUATING;
        semantic_update_.active = true;
        semantic_update_.lambda = 0.0f;
        semantic_update_.lambda_dot = 0.0f;

        semantic_update_.evaluation_start_time =
            std::chrono::steady_clock::now();
        semantic_update_.start_time =
            semantic_update_.evaluation_start_time;
        semantic_update_.last_update_time =
            semantic_update_.evaluation_start_time;

        semantic_candidate_target_ready_ = false;

        RCLCPP_INFO(
            this->get_logger(),
            "Constraint set changed; entered semantic EVALUATING mode"
        );
    }

    void classify_semantic_geometry_change(
        bool& has_additions,
        bool& has_removals
    ) const {
        has_additions = false;
        has_removals = false;

        for (int n = 0; n < IMAX * JMAX; ++n) {
            const bool old_occ = semantic_previous_grid_[n] > 0;
            const bool new_occ = semantic_target_grid_[n] > 0;

            has_additions = has_additions || (!old_occ && new_occ);
            has_removals = has_removals || (old_occ && !new_occ);

            if (has_additions && has_removals) {
                return;
            }
        }
    }

    void initialize_semantic_homotopy(
        SemanticUpdateMode mode,
        float insertion_scale = 1.0f
    ) {
        semantic_update_.mode = mode;
        semantic_update_.active = true;
        semantic_update_.lambda = 0.0f;

        semantic_update_.lambda_dot_min =
            1.0f /
            std::max(
                semantic_update_.max_update_time_sec,
                1.0e-3f
            );

        const float safe_insertion_scale =
            std::clamp(insertion_scale, 0.0f, 1.0f);

        semantic_update_.commanded_lambda_dot =
            semantic_update_.lambda_dot_max * safe_insertion_scale;
        semantic_update_.lambda_dot =
            semantic_update_.commanded_lambda_dot;

        semantic_update_.start_time =
            std::chrono::steady_clock::now();
        semantic_update_.last_update_time =
            semantic_update_.start_time;

        semantic_old_max_depth_m_ =
            std::max(
                max_interior_depth(semantic_previous_grid_),
                DS
            );

        semantic_target_max_depth_m_ =
            std::max(
                max_interior_depth(semantic_target_grid_),
                DS
            );

        semantic_transition_extent_m_ =
            std::max(
                semantic_old_max_depth_m_,
                semantic_target_max_depth_m_
            );

        const char* mode_name = "transition";
        if (mode == SemanticUpdateMode::INSERTING_CONSTRAINT) {
            mode_name = "insertion";
        } else if (mode == SemanticUpdateMode::REMOVING_CONSTRAINT) {
            mode_name = "removal";
        }

        RCLCPP_INFO(
            this->get_logger(),
            "Started semantic %s: extent=%.3f m, lambda_dot=%.3f 1/s, commanded boundary speed=%.3f m/s",
            mode_name,
            semantic_transition_extent_m_,
            semantic_update_.lambda_dot,
            semantic_transition_extent_m_ *
                std::abs(semantic_update_.lambda_dot)
        );
    }

    void evaluate_and_begin_semantic_transition() {
        if (semantic_update_.mode != SemanticUpdateMode::EVALUATING) {
            return;
        }

        const double admission_dt =
            std::max(static_cast<double>(dt_grid), 1.0e-4);

        const AdmissionResult admission =
            evaluate_candidate_constraint(
                semantic_previous_grid_,
                semantic_target_grid_,
                admission_dt,
                admission_v_admissible_mps_
            );

        if (admission.decision == AdmissionDecision::REJECT) {
            semantic_target_grid_ = semantic_previous_grid_;
            semantic_current_grid_ = semantic_previous_grid_;

            semantic_update_.mode = SemanticUpdateMode::NORMAL;
            semantic_update_.active = false;
            semantic_update_.lambda = 1.0f;
            semantic_update_.lambda_dot = 0.0f;
            semantic_update_.commanded_lambda_dot = 0.0f;
            semantic_candidate_target_ready_ = false;

            rollback_candidate_constraint_config(
                admission.reason
            );

            RCLCPP_WARN(
                this->get_logger(),
                "Candidate semantic constraint rejected: reason=%s, max_sdf_change=%.4f m, allowed=%.4f m",
                admission_reason_string(admission.reason),
                admission.max_sdf_change,
                admission.allowed_sdf_change
            );
            return;
        }

        bool has_additions = false;
        bool has_removals = false;
        classify_semantic_geometry_change(
            has_additions,
            has_removals
        );

        if (!has_additions && !has_removals) {
            semantic_current_grid_ = semantic_target_grid_;

            semantic_update_.mode = SemanticUpdateMode::NORMAL;
            semantic_update_.active = false;
            semantic_update_.lambda = 1.0f;
            semantic_update_.lambda_dot = 0.0f;
            semantic_update_.commanded_lambda_dot = 0.0f;
            semantic_candidate_target_ready_ = false;

            // Nothing needs a geometric homotopy, so the candidate rule set
            // can be committed immediately (e.g. a pure velocity-limit edit).
            commit_candidate_constraint_config();

            RCLCPP_INFO(
                this->get_logger(),
                "Candidate constraint accepted; semantic geometry unchanged"
            );
            return;
        }

        const float insertion_scale =
            admission.decision == AdmissionDecision::SLOW_INSERTION
                ? static_cast<float>(admission.insertion_scale)
                : 1.0f;

        if (admission.decision == AdmissionDecision::SLOW_INSERTION) {
            RCLCPP_WARN(
                this->get_logger(),
                "Candidate constraint requires slow insertion: max_sdf_change=%.4f m, allowed=%.4f m, insertion_scale=%.4f",
                admission.max_sdf_change,
                admission.allowed_sdf_change,
                admission.insertion_scale
            );
        }

        if (has_additions && !has_removals) {
            initialize_semantic_homotopy(
                SemanticUpdateMode::INSERTING_CONSTRAINT,
                insertion_scale
            );
        } else if (!has_additions && has_removals) {
            initialize_semantic_homotopy(
                SemanticUpdateMode::REMOVING_CONSTRAINT,
                insertion_scale
            );
        } else {
            initialize_semantic_homotopy(
                SemanticUpdateMode::TRANSITIONING_CONSTRAINT,
                insertion_scale
            );
        }

        semantic_candidate_target_ready_ = false;
    }

    std::vector<int8_t> build_semantic_grid_for_lambda(float lambda) const {
        const float lam = std::clamp(lambda, 0.0f, 1.0f);

        // Outside an active homotopy, the target is the intended semantic map.
        if (!semantic_update_.active) {
            return semantic_target_grid_;
        }

        if (semantic_update_.mode == SemanticUpdateMode::EVALUATING) {
            return semantic_previous_grid_;
        }

        std::vector<int8_t> grid(IMAX * JMAX, 0);

        const float old_max_depth =
            std::max(semantic_old_max_depth_m_, DS);

        const float target_max_depth =
            std::max(semantic_target_max_depth_m_, DS);

        for (int i = 0; i < IMAX; ++i) {
            for (int j = 0; j < JMAX; ++j) {
                const int n = i * JMAX + j;

                const bool old_occ =
                    semantic_previous_grid_[n] > 0;
                const bool new_occ =
                    semantic_target_grid_[n] > 0;

                if (old_occ && new_occ) {
                    grid[n] = 1;
                    continue;
                }

                if (!old_occ && new_occ &&
                    (semantic_update_.mode ==
                         SemanticUpdateMode::INSERTING_CONSTRAINT ||
                     semantic_update_.mode ==
                         SemanticUpdateMode::TRANSITIONING_CONSTRAINT)) {

                    const float depth =
                        distance_to_nearest_free_cell(
                            semantic_target_grid_, i, j
                        );

                    const float activation =
                        1.0f -
                        std::clamp(
                            depth / target_max_depth,
                            0.0f,
                            1.0f
                        );

                    if (lam + 1.0e-6f >= activation) {
                        grid[n] = 1;
                    }

                    continue;
                }

                if (old_occ && !new_occ &&
                    (semantic_update_.mode ==
                         SemanticUpdateMode::REMOVING_CONSTRAINT ||
                     semantic_update_.mode ==
                         SemanticUpdateMode::TRANSITIONING_CONSTRAINT)) {

                    const float depth =
                        distance_to_nearest_free_cell(
                            semantic_previous_grid_, i, j
                        );

                    const float removal_threshold =
                        std::clamp(
                            depth / old_max_depth,
                            0.0f,
                            1.0f
                        );

                    if (lam < removal_threshold) {
                        grid[n] = 1;
                    }

                    continue;
                }
            }
        }

        if (lam >= 0.999f) {
            grid = semantic_target_grid_;
        }

        return grid;
    }

    // Applied only to the path that currently has *no* rate limiting 
    // at all: when semantic_update_ is not
    // running an active INSERTING/REMOVING/TRANSITIONING homotopy,
    // build_semantic_grid_for_lambda() returns semantic_target_grid_
    // unmodified every tick
    //slew-rate limiter :
    //     b_active += clamp(b_target - b_active, -v*dt, +v*dt)
    // run on the semantic-only occupancy's SDF.
    //
    // v is reused as the growth speed since it
    // already represents "how fast can the robot plausibly react,"
    void advance_semantic_boundary_sdf(float dt)
    {
        const std::size_t N =
            static_cast<std::size_t>(IMAX * JMAX);

        if (semantic_target_grid_.size() != N ||
            semantic_current_grid_.size() != N) {
            return;
        }

        // 0.01 m/s means with DS = 0.05 m it takes approximately
        // 5 seconds for the boundary to advance one grid-cell layer.
        constexpr float kGrowthSpeedMps = 0.01f;

        // Ignore tiny changes in the incoming target(maybe noise)
        constexpr int kMinChangedCells = 3;
        const float safe_dt = std::clamp(dt, 1.0e-4f, 0.2f);
        const float growth_step = kGrowthSpeedMps * safe_dt;


        // ============================================================
        // Local persistent state
        //
        // These are static ONLY so you can test the idea without
        // modifying the rest of the class.
        // ============================================================
        static bool initialized = false;

        // Last target received from the semantic fuser.
        static std::vector<int8_t> previous_target;

        // Frozen target for the current growth transition.
        static std::vector<int8_t> target_snapshot;

        // SDF of that frozen target.
        static std::vector<float> target_sdf;

        // Semantic grid that existed at the start of the transition.
        //
        // These cells remain occupied while the newly detected
        // semantic region grows.
        static std::vector<int8_t> base_grid;

        // Remaining inward offset of the target SDF.
        //
        // Large offset:
        //     only deep interior target cells are active.
        //
        // offset -> 0:
        //     complete target becomes active.
        static float offset_m = 0.0f;
        static bool transition_active = false;


        // ============================================================
        // 1. First invocation
        // ============================================================

        if (!initialized) {

            previous_target = semantic_target_grid_;
            target_snapshot = semantic_target_grid_;

            target_sdf = compute_signed_distance_field(target_snapshot);
            base_grid = semantic_current_grid_;

            initialized = true;

            // IMPORTANT:
            // Do not immediately replace semantic_current_grid_ here.
            // We want the current geometry to remain exactly as it was
            // before this mechanism was enabled.
        }


        // ============================================================
        // 2. Detect a meaningful change in the target semantic map
        // ============================================================

        int changed_cells = 0;
        int added_cells = 0;
        int removed_cells = 0;

        for (std::size_t n = 0; n < N; ++n) {

            const bool old_occ = previous_target[n] > 0;

            const bool new_occ = semantic_target_grid_[n] > 0;

            if (old_occ == new_occ) {
                continue;
            }

            ++changed_cells;

            if (!old_occ && new_occ) {
                ++added_cells;
            }
            else if (old_occ && !new_occ) {
                ++removed_cells;
            }
        }


        const bool meaningful_change = changed_cells >= kMinChangedCells;


        // ============================================================
        // 3. A NEW semantic region appeared
        //
        // This is the case we care about:
        //
        // rule already exists
        //       +
        // cone suddenly becomes visible
        //
        // For this POC, only react to changes dominated by additions.
        // ============================================================

        if (meaningful_change &&
            added_cells > removed_cells) {

            // Preserve what was already active before the new cone
            // appeared.
            base_grid = semantic_current_grid_;
            // Freeze the new requested semantic geometry.
            target_snapshot = semantic_target_grid_;
            target_sdf = compute_signed_distance_field(target_snapshot);

            // --------------------------------------------------------
            // Find the maximum interior depth of ONLY the newly added
            // target region.
            //
            // target_sdf < 0 inside the semantic target.
            // Therefore -target_sdf gives interior depth in meters.
            // --------------------------------------------------------
            float max_new_depth_m = 0.0f;

            for (std::size_t n = 0; n < N; ++n) {

                const bool already_active =
                    base_grid[n] > 0;

                const bool target_occ =
                    target_snapshot[n] > 0;

                if (already_active || !target_occ) {
                    continue;
                }

                const float depth_m = std::max(0.0f, -target_sdf[n]);

                max_new_depth_m = std::max(max_new_depth_m, depth_m);
            }


            if (max_new_depth_m > 1.0e-4f) {

                // Start at the deepest interior level set.
                //
                // Then offset_m decreases slowly toward zero.
                //
                // This causes the active zero-level boundary to
                // propagate OUTWARD toward the final semantic boundary.
                offset_m = max_new_depth_m;

                transition_active = true;

                RCLCPP_INFO(
                    this->get_logger(),
                    "Started perception-driven semantic growth: "
                    "added=%d removed=%d changed=%d "
                    "initial_offset=%.3f m speed=%.3f m/s",
                    added_cells,
                    removed_cells,
                    changed_cells,
                    offset_m,
                    kGrowthSpeedMps
                );
            }
        }


        // Always update this so we compare against the newest incoming
        // semantic target on the next iteration.
        previous_target = semantic_target_grid_;


        //If no perception-driven transition is running, just use
        //    the current target normally.
        if (!transition_active) {
            semantic_current_grid_ = semantic_target_grid_;
            return;
        }


        // ============================================================
        // 5. Advance the moving level set
        //
        // offset_m:
        //
        //      large ----------------------------> 0
        //
        // active semantic region:
        //
        //      small ---------------------------> full target
        // ============================================================

        offset_m = std::max(0.0f, offset_m - growth_step);
        //Construct the new active semantic grid
        std::vector<int8_t> next_grid = base_grid;

        int newly_active_cells = 0;

        for (std::size_t n = 0; n < N; ++n) {

            if (target_snapshot[n] <= 0) {
                continue;
            }

            // Preserve cells that were already active before the
            // perception event.
            if (next_grid[n] > 0) {
                continue;
            }


            // target_sdf is negative inside the target.
            // target interior depth = 0.8 m
            // target_sdf          = -0.8
            //
            // offset = 0.7
            //
            // -0.8 <= -0.7   -> active
            //
            // A cell near the outer target boundary might be:
            //
            // target_sdf = -0.05
            //
            // -0.05 <= -0.7  -> false
            //
            // It becomes active later as offset approaches zero.
            if (target_sdf[n] <= -offset_m) {

                next_grid[n] = 1;
                ++newly_active_cells;
            }
        }
        semantic_current_grid_.swap(next_grid);

        // Finish at the target

        if (offset_m <= 1.0e-4f) {
            semantic_current_grid_ = target_snapshot;
            transition_active = false;
            offset_m = 0.0f;
            RCLCPP_INFO(
                this->get_logger(),
                "Completed perception-driven semantic growth"
            );
        }

        RCLCPP_INFO_THROTTLE(
            this->get_logger(),
            *this->get_clock(),
            500,
            "Semantic growth POC: "
            "speed=%.3f m/s dt=%.4f s "
            "step=%.5f m offset=%.3f m "
            "new_cells=%d active=%d",
            kGrowthSpeedMps,
            safe_dt,
            growth_step,
            offset_m,
            newly_active_cells,
            transition_active ? 1 : 0
        );
    }

    void update_interpolated_semantic_grid() {
        if (enable_sdf_rate_limited_semantic_growth_ &&
            !semantic_update_.active) {
            advance_semantic_boundary_sdf(dt_grid);
            return;
        }

        semantic_current_grid_ =
            build_semantic_grid_for_lambda(semantic_update_.lambda);
    }

    void copy_current_globals_into_pending_field() {
        const int N = IMAX * JMAX * QMAX;

        std::memcpy(pending_field_.hgrid.data(), hgrid1, N * sizeof(float));
        std::memcpy(pending_field_.dhdt.data(), dhdt_grid, N * sizeof(float));
        std::memcpy(pending_field_.beta.data(), beta_grid_, N * sizeof(float));
        std::memcpy(pending_field_.guidance_x.data(), guidance_x_grid, N * sizeof(float));
        std::memcpy(pending_field_.guidance_y.data(), guidance_y_grid, N * sizeof(float));
        std::memcpy(pending_field_.bound.data(), bound, N * sizeof(float));

        pending_field_.timestamp = std::chrono::steady_clock::now();
        pending_field_.valid = true;
    }

    void start_semantic_insertion() {
        semantic_previous_grid_ = semantic_current_grid_;
        initialize_semantic_homotopy(
            SemanticUpdateMode::INSERTING_CONSTRAINT
        );
    }

    void start_semantic_removal() {
        semantic_previous_grid_ = semantic_current_grid_;
        initialize_semantic_homotopy(
            SemanticUpdateMode::REMOVING_CONSTRAINT
        );
    }

    static std::string normalized_semantic_class(std::string value) {
        std::transform(
            value.begin(), value.end(), value.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); }
        );
        std::replace(value.begin(), value.end(), '-', '_');
        std::replace(value.begin(), value.end(), ' ', '_');
        return value;
    }

    bool class_requires_yolo(const std::string& raw_class) const {
        const std::string cls = normalized_semantic_class(raw_class);

        // Keep this list centralized. Add aliases here when the rule compiler
        // introduces another YOLO-backed semantic class.
        static const std::set<std::string> yolo_classes = {
            "person", "human", "people",
            "object", "obstacle",
            "traffic_cone", "cone",
            "caution_tape", "tape",
            "spill", "spills"
        };

        return yolo_classes.count(cls) > 0;
    }

    template <typename ConstraintT>
    bool constraint_requires_yolo(const ConstraintT& rc) const {
        if (!rc.enabled || !rc.enforce) {
            return false;
        }

        for (const auto& target : rc.target_classes) {
            if (class_requires_yolo(target)) {
                return true;
            }
        }

        for (const auto& reference : rc.reference_classes) {
            if (class_requires_yolo(reference)) {
                return true;
            }
        }

        return false;
    }

    bool any_active_constraint_requires_yolo() const {
        // The admitted configuration drives active control.  While evaluating
        // a candidate, perception must also be enabled for classes referenced
        // only by that candidate so its geometry can actually be generated.
        for (const auto& rc : admitted_constraint_config_.constraints) {
            if (constraint_requires_yolo(rc)) {
                return true;
            }
        }

        if (candidate_constraint_pending_) {
            for (const auto& rc : candidate_constraint_config_.constraints) {
                if (constraint_requires_yolo(rc)) {
                    return true;
                }
            }
        }

        return false;
    }

    void publish_semantic_perception_required(bool force_publish = false) {
        if (!semantic_perception_required_pub_) {
            return;
        }

        const bool required = any_active_constraint_requires_yolo();
        if (!force_publish &&
            semantic_perception_state_initialized_ &&
            required == semantic_perception_required_) {
            return;
        }

        semantic_perception_required_ = required;
        semantic_perception_state_initialized_ = true;

        std_msgs::msg::Bool msg;
        msg.data = required;
        semantic_perception_required_pub_->publish(msg);

        RCLCPP_INFO(
            this->get_logger(),
            "Semantic perception %s: %s",
            required ? "ENABLED" : "DISABLED",
            required ? "an active enforced rule references a YOLO semantic class"
                     : "no active enforced rule references a YOLO semantic class"
        );
    }

    void initialize_constraint_reload_timer() {

        if (constraints_path_.empty()) {
            RCLCPP_WARN(
                this->get_logger(),
                "Runtime constraints reload disabled because constraints_path is empty."
            );
            return;
        }

        const double hz = std::max(0.1, constraints_reload_hz_);
        const int period_ms = static_cast<int>(1000.0 / hz);

        constraints_reload_callback_group_ =
            this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);

        constraints_reload_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(period_ms),
            [this]() {
                // RCLCPP_ERROR(
                //     this->get_logger(),
                //     "========== SIMPLE TIMER FIRED =========="
                // );

                this->reload_constraints_callback();
            },
            constraints_reload_callback_group_
        );

        // RCLCPP_INFO(
        //     this->get_logger(),
        //     "Runtime constraints reload enabled: %.2f Hz, path=%s, period=%d ms",
        //     hz,
        //     constraints_path_.c_str(),
        //     period_ms
        // );
    }
    

    std::size_t constraints_file_signature() const {
        const std::string contents =
            read_constraints_file_text();

        if (contents.empty()) {
            return 0;
        }

        return constraints_text_signature(contents);
    }

    void reload_constraints_callback() {
        ConstraintManager fresh_manager;

        const std::string fresh_json =
            read_constraints_file_text();

        if (fresh_json.empty()) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                5000,
                "Failed to read constraints JSON: %s",
                constraints_path_.c_str()
            );
            return;
        }

        if (!fresh_manager.load_from_json(constraints_path_)) {
            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                5000,
                "Failed to reload constraints JSON: %s",
                constraints_path_.c_str()
            );
            return;
        }

        const ConstraintRuntimeConfig fresh_config =
            fresh_manager.get_config();

        const std::size_t fresh_signature =
            constraints_text_signature(fresh_json);

        if (!constraints_signature_initialized_) {
            admitted_constraint_config_ = fresh_config;
            constraint_runtime_config_ = fresh_config;

            admitted_constraints_json_ = fresh_json;
            admitted_constraints_file_signature_ =
                fresh_signature;

            last_constraints_file_signature_ =
                fresh_signature;

            constraints_signature_initialized_ = true;

            apply_runtime_constraint_config(
                admitted_constraint_config_,
                true
            );

            publish_semantic_perception_required();
            return;
        }

        if (fresh_signature ==
            last_constraints_file_signature_) {
            return;
        }

        // A new file revision is a proposal, not an immediately active rule
        // set.  Keep constraint_runtime_config_ equal to the admitted config
        // until the candidate has passed the complete admission/homotopy path.
        candidate_constraint_config_ =
            fresh_config;

        candidate_constraints_json_ =
            fresh_json;

        candidate_constraints_file_signature_ =
            fresh_signature;

        candidate_constraint_pending_ = true;

        // Mark this revision as observed so the reload timer does not restart
        // evaluation on every tick while admission is in progress.
        last_constraints_file_signature_ =
            fresh_signature;

        begin_semantic_evaluation();
        publish_semantic_perception_required();

        RCLCPP_INFO(
            this->get_logger(),
            "Staged new constraint file revision as CANDIDATE; admitted constraints remain active until admission completes"
        );
    }


    void update_semantic_update_state() {
        if (!semantic_update_.active) {
            semantic_update_.mode = SemanticUpdateMode::NORMAL;
            semantic_update_.lambda = 1.0f;
            semantic_update_.lambda_dot = 0.0f;
            semantic_update_.commanded_lambda_dot = 0.0f;
            return;
        }

        if (semantic_update_.mode == SemanticUpdateMode::EVALUATING) {
            semantic_update_.lambda = 0.0f;
            semantic_update_.lambda_dot = 0.0f;
            return;
        }

        const auto now = std::chrono::steady_clock::now();

        float dt =
            std::chrono::duration<float>(
                now - semantic_update_.last_update_time
            ).count();

        semantic_update_.last_update_time = now;
        dt = std::clamp(dt, 0.0f, 0.2f);

        if (dt <= 1.0e-6f) {
            semantic_update_.lambda_dot = 0.0f;
            return;
        }

        // Admission safety takes precedence over the nominal minimum update
        // rate. Never raise a theorem-limited command merely to satisfy the
        // preferred completion time.
        const float requested_lambda_dot =
            std::clamp(
                semantic_update_.commanded_lambda_dot,
                0.0f,
                semantic_update_.lambda_dot_max
            );

        const float current_lambda =
            std::clamp(semantic_update_.lambda, 0.0f, 1.0f);

        float proposed_delta_lambda =
            dt * requested_lambda_dot;

        proposed_delta_lambda =
            std::min(
                proposed_delta_lambda,
                1.0f - current_lambda
            );

        float accepted_delta_lambda = 0.0f;
        AdmissionResult step_admission;

        // The initial admission result gives a useful global rate estimate,
        // but the corollary must hold between each executed geometry update.
        // Re-evaluate each proposed lambda step and shrink it if needed.
        constexpr int kMaxAdmissionRefinements = 10;
        float trial_delta_lambda = proposed_delta_lambda;

        for (int attempt = 0;
             attempt < kMaxAdmissionRefinements &&
             trial_delta_lambda > 1.0e-7f;
             ++attempt) {

            const float trial_lambda =
                std::clamp(
                    current_lambda + trial_delta_lambda,
                    0.0f,
                    1.0f
                );

            const auto proposed_semantic_grid =
                build_semantic_grid_for_lambda(trial_lambda);

            step_admission =
                evaluate_candidate_constraint(
                    semantic_current_grid_,
                    proposed_semantic_grid,
                    static_cast<double>(dt),
                    admission_v_admissible_mps_
                );

            if (step_admission.decision == AdmissionDecision::ACCEPT) {
                accepted_delta_lambda = trial_delta_lambda;
                break;
            }

            if (step_admission.decision == AdmissionDecision::REJECT) {
                semantic_target_grid_ = semantic_current_grid_;
                semantic_previous_grid_ = semantic_current_grid_;

                semantic_update_.active = false;
                semantic_update_.mode = SemanticUpdateMode::NORMAL;
                semantic_update_.lambda = 1.0f;
                semantic_update_.lambda_dot = 0.0f;
                semantic_update_.commanded_lambda_dot = 0.0f;

                semantic_transition_extent_m_ = 0.0f;
                semantic_old_max_depth_m_ = 0.0f;
                semantic_target_max_depth_m_ = 0.0f;

                rollback_candidate_constraint_config(
                    step_admission.reason
                );

                RCLCPP_WARN(
                    this->get_logger(),
                    "Semantic transition aborted during admission check: reason=%s. The previously admitted constraint set remains active.",
                    admission_reason_string(step_admission.reason)
                );
                return;
            }

            // SLOW_INSERTION: shrink the proposed homotopy increment according
            // to the measured SDF violation and test again before committing.
            const float scale =
                static_cast<float>(std::clamp(
                    step_admission.insertion_scale,
                    0.0,
                    1.0
                ));

            trial_delta_lambda *= scale;
        }

        if (accepted_delta_lambda <= 1.0e-7f) {
            // No geometry change is committed this cycle. Keep the transition
            // active; a later cycle has a new dt budget and can try again.
            semantic_update_.lambda_dot = 0.0f;

            RCLCPP_WARN_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Semantic insertion step held by admission check: max_sdf_change=%.4f m, allowed=%.4f m",
                step_admission.max_sdf_change,
                step_admission.allowed_sdf_change
            );
            return;
        }

        semantic_update_.lambda =
            std::clamp(
                current_lambda + accepted_delta_lambda,
                0.0f,
                1.0f
            );

        semantic_update_.lambda_dot =
            accepted_delta_lambda / dt;

        // Commit exactly the geometry that was admitted for this lambda.
        semantic_current_grid_ =
            build_semantic_grid_for_lambda(semantic_update_.lambda);

        if (semantic_update_.lambda >= 0.999f) {
            semantic_current_grid_ = semantic_target_grid_;

            semantic_update_.active = false;
            semantic_update_.lambda = 1.0f;
            semantic_update_.lambda_dot = 0.0f;
            semantic_update_.commanded_lambda_dot = 0.0f;
            semantic_update_.mode = SemanticUpdateMode::NORMAL;

            semantic_transition_extent_m_ = 0.0f;
            semantic_old_max_depth_m_ = 0.0f;
            semantic_target_max_depth_m_ = 0.0f;

            // The complete candidate geometry has now passed every admission
            // step, so promote its rule configuration atomically.
            commit_candidate_constraint_config();

            RCLCPP_INFO(
                this->get_logger(),
                "Completed semantic update"
            );
        }
    }

    void apply_runtime_constraint_config(
        const ConstraintRuntimeConfig& cfg,
        bool allow_kernel_rebuild
    ) {
        bool need_rebuild_kernels = false;

        if (std::abs(
                cfg.human_buffer_m -
                robot_MOS_human
            ) > 1.0e-4f) {

            robot_MOS_human = cfg.human_buffer_m;
            need_rebuild_kernels = true;
        }

        if (allow_kernel_rebuild &&
            need_rebuild_kernels) {
            rebuild_robot_kernels();
        }
    }

    void load_constraints_once_at_startup() {
        if (constraints_path_.empty()) {
            RCLCPP_WARN(
                this->get_logger(),
                "No constraints_path provided. Using launch/default parameters."
            );
            return;
        }

        if (constraint_manager_.load_from_json(constraints_path_)) {
            admitted_constraint_config_ =
                constraint_manager_.get_config();

            constraint_runtime_config_ =
                admitted_constraint_config_;

            apply_runtime_constraint_config(
                admitted_constraint_config_,
                false
            );

            admitted_constraints_json_ =
                read_constraints_file_text();

            admitted_constraints_file_signature_ =
                admitted_constraints_json_.empty()
                    ? constraints_file_signature()
                    : constraints_text_signature(
                          admitted_constraints_json_
                      );

            last_constraints_file_signature_ =
                admitted_constraints_file_signature_;

            constraints_signature_initialized_ = true;
            candidate_constraint_pending_ = false;
        } else {
            RCLCPP_WARN(
                this->get_logger(),
                "Failed to load constraints JSON from: %s",
                constraints_path_.c_str()
            );
        }
    }


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

    void rebuild_robot_kernels() {
            std::unique_lock<std::shared_mutex> lock(field_mutex_);
    
            if (robot_kernel_human) {
                std::free(robot_kernel_human);
                robot_kernel_human = nullptr;
            }
    
            if (robot_kernel_obstacle) {
                std::free(robot_kernel_obstacle);
                robot_kernel_obstacle = nullptr;
            }
    
            robot_kernel_dim_human =
                initialize_robot_kernel(robot_kernel_human, robot_MOS_human);
    
            robot_kernel_dim_obstacle =
                initialize_robot_kernel(robot_kernel_obstacle, robot_MOS_obstacle);
    
            // RCLCPP_INFO(
            //     this->get_logger(),
            //     "Rebuilt robot kernels from JSON constraints: human=%.2f, obstacle=%.2f",
            //     robot_MOS_human,
            //     robot_MOS_obstacle
            // );
        }

    void initialize_static_grids() {
        semantic_occupancy_grid_.assign(IMAX * JMAX, 0);
        semantic_previous_grid_.assign(IMAX * JMAX, 0);
        semantic_target_grid_.assign(IMAX * JMAX, 0);
        semantic_current_grid_.assign(IMAX * JMAX, 0);
        semantic_sdf_active_.clear(); // lazily (re)seeded on first use
        external_semantic_safety_grid_.assign(IMAX * JMAX, 0);
        admitted_external_semantic_grid_snapshot_.assign(IMAX * JMAX, 0);
        rejected_candidate_external_grid_.assign(IMAX * JMAX, 0);
        external_semantic_safety_received_ = false;
        external_semantic_safety_timestamp_ = std::chrono::steady_clock::now();
        
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
            "t_ms", "space_counter",
            "rx", "ry", "yaw",
            "vx", "vy", "vyaw",
            "vxd", "vyd", "vyawd",
            "h", "dhdx", "dhdy", "dhdq", "dhdt",
            "alpha", "on_off",
            "lambda", "lambda_dot", "insertion_active",
            "new_constraint_event", "constraint_event_counter"
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

        // Dedicated command-velocity log for the MPC solver:
        //   vd  = raw MPC solver output   (mpc3d_controller.set_input)
        //   v   = safe control after the realtime filter
        //   vb  = final command sent to the robot (sport_req.Move)
        std::string fileNameMPCVel = baseFileName + "_mpc_cmd_vel_" + dateTime + ".csv";
        outFileMPCVel.open(fileNameMPCVel);
        if (!outFileMPCVel.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open MPC command-velocity log file: %s",
                         fileNameMPCVel.c_str());
            throw std::runtime_error("Failed to open MPC command-velocity log file");
        }

        const std::vector<std::string> mpc_vel_header = {
            "t_ms", "space_counter",
            "vd_x", "vd_y", "vd_yaw",
            "v_x", "v_y", "v_yaw",
            "vb_x", "vb_y", "vb_yaw"
        };
        for (size_t n = 0; n < mpc_vel_header.size(); ++n) {
            outFileMPCVel << mpc_vel_header[n];
            if (n + 1 < mpc_vel_header.size()) outFileMPCVel << ",";
        }
        outFileMPCVel << std::endl;

        RCLCPP_INFO(this->get_logger(), "MPC command-velocity logging ENABLED: %s",
                    fileNameMPCVel.c_str());
    }
    
    void declare_and_load_parameters() {
        // ------------------------------------------------------------
        // Logging / visualization
        // ------------------------------------------------------------
        this->declare_parameter("enable_data_logging_to_file", false);
        this->declare_parameter("enable_display", false);
        this->declare_parameter("logging_publish_hz", 10.0);
        this->declare_parameter("constraints_path", "");
        this->declare_parameter("enable_human_persistence", true);
        this->declare_parameter("human_persistence_decay", 0.96);
        this->declare_parameter("human_persistence_threshold", 0.25);
        this->declare_parameter("human_persistence_observation_value", 1.0);
        this->declare_parameter("constraints_reload_hz", 0.1);
        this->declare_parameter(
            "semantic_safety_target_topic",
            "/semantic_safety_target"
        );
        this->declare_parameter("semantic_safety_occupied_threshold", 50);
        this->declare_parameter("semantic_safety_max_age_sec", 1.0);

        // Stage B PoC: rate-limits the semantic layer's SDF (instead of
        // snapping semantic_current_grid_ straight to semantic_target_grid_)
        // whenever there is no active INSERTING/REMOVING/TRANSITIONING
        this->declare_parameter("enable_sdf_rate_limited_semantic_growth", false);

        enable_data_logging_to_file_ = this->get_parameter("enable_data_logging_to_file").as_bool();
        enable_display = this->get_parameter("enable_display").as_bool();
        logging_publish_hz_ = this->get_parameter("logging_publish_hz").as_double();
        logging_publish_period_ = (logging_publish_hz_ > 0.0) ? (1.0 / logging_publish_hz_) : 0.0;
        constraints_path_ = this->get_parameter("constraints_path").as_string();
        constraints_reload_hz_ = this->get_parameter("constraints_reload_hz").as_double();
        enable_human_persistence_ = this->get_parameter("enable_human_persistence").as_bool();
        human_persistence_decay_ = static_cast<float>(this->get_parameter("human_persistence_decay").as_double());
        human_persistence_threshold_ = static_cast<float>(this->get_parameter("human_persistence_threshold").as_double());
        human_persistence_observation_value_ = static_cast<float>(this->get_parameter("human_persistence_observation_value").as_double());
        semantic_safety_target_topic_ =
            this->get_parameter("semantic_safety_target_topic").as_string();
        semantic_safety_occupied_threshold_ = static_cast<int>(
            this->get_parameter("semantic_safety_occupied_threshold").as_int()
        );
        semantic_safety_max_age_sec_ =
            this->get_parameter("semantic_safety_max_age_sec").as_double();
        enable_sdf_rate_limited_semantic_growth_ =
            this->get_parameter("enable_sdf_rate_limited_semantic_growth").as_bool();
    
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
    
        this->declare_parameter("robot_mos_human", 0.01);
        this->declare_parameter("robot_mos_obstacle", 0.01);
    
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
        // Admission-check parameters
        // ------------------------------------------------------------
        // Negative means derive a conservative translational bound from the
        // robot velocity limits above.
        this->declare_parameter("admission_v_admissible_mps", -1.0);
        admission_v_admissible_mps_ =
            this->get_parameter("admission_v_admissible_mps").as_double();

        if (admission_v_admissible_mps_ < 0.0) {
            admission_v_admissible_mps_ = std::min({
                static_cast<double>(vel_max_x_fwd_),
                static_cast<double>(vel_max_x_bwd_),
                static_cast<double>(vel_max_y_)
            });
        }
    
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
        // RCLCPP_INFO(
        //     this->get_logger(),
        //     "dh0_human=%.2f, dh0_obstacle=%.2f, MOS_human=%.2f, MOS_obstacle=%.2f, display=%s, social_nav=%s",
        //     dh0_human, dh0_obstacle, robot_MOS_human, robot_MOS_obstacle,
        //     enable_display ? "true" : "false",
        //     enable_social_navigation_ ? "true" : "false"
        // );
    
        // RCLCPP_INFO(
        //     this->get_logger(),
        //     "Dynamic CBF: sigma_epsilon=%.3f, sigma_kappa=%.2f",
        //     cbf_sigma_epsilon_, cbf_sigma_kappa_
        // );
    
        // RCLCPP_INFO(
        //     this->get_logger(),
        //     "Velocity bounds: x_fwd=%.2f, x_bwd=%.2f, y=%.2f, yaw=%.2f",
        //     vel_max_x_fwd_, vel_max_x_bwd_, vel_max_y_, vel_max_yaw_
        // );
    
        // RCLCPP_INFO(
        //     this->get_logger(),
        //     "HumanTracker: timeout=%.1fs, gate=%.2fm, vel_thresh=%.2fm/s, decay_fov=%.2f, decay_stat=%.2f, decay_unconf=%.2f, no_retrack=%s",
        //     track_timeout, track_gate, track_velocity_threshold,
        //     decay_in_fov, decay_stationary, decay_unconfirmed,
        //     no_retrack_on_move ? "true" : "false"
        // );
    
        // RCLCPP_INFO(
        //     this->get_logger(),
        //     "Tight-area params: human_thresh=%.2fm, h_thresh=%.2f, wall_slack=%.2f",
        //     tight_area_human_threshold_, tight_area_h_threshold_, tight_area_wall_slack_
        // );
    
        RCLCPP_INFO(this->get_logger(), "Logging publish rate: %.1f Hz", logging_publish_hz_);
        RCLCPP_INFO(
            this->get_logger(),
            "Semantic safety input: topic=%s threshold=%d max_age=%.2f s",
            semantic_safety_target_topic_.c_str(),
            semantic_safety_occupied_threshold_,
            semantic_safety_max_age_sec_
        );
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
        persistent_human_confidence_.assign(IMAX * JMAX, 0.0f);
        persistent_human_mask_.assign(IMAX * JMAX, 0);
        hgrid_insertion_old_ = static_cast<float*>(std::malloc(IMAX * JMAX * QMAX * sizeof(float)));
        hgrid_active_ = static_cast<float*>(std::malloc(IMAX * JMAX * QMAX * sizeof(float)));
        dhdt_active_ = static_cast<float*>(std::malloc(IMAX * JMAX * QMAX * sizeof(float)));
        beta_grid_ = static_cast<float*>(std::malloc(IMAX * JMAX * QMAX * sizeof(float)));

        const int N_field = IMAX * JMAX * QMAX;

        auto resize_field_buffer = [N_field](FieldBuffer& fb) {
            fb.hgrid.resize(N_field, 1.0f);
            fb.dhdt.resize(N_field, 0.0f);
            fb.beta.resize(N_field, 0.0f);
            fb.guidance_x.resize(N_field, 0.0f);
            fb.guidance_y.resize(N_field, 0.0f);
            fb.bound.resize(N_field, 1.0f);
            fb.timestamp = std::chrono::steady_clock::now();
            fb.valid = false;
        };

        resize_field_buffer(active_field_);
        resize_field_buffer(pending_field_);
        if (!hgrid_insertion_old_ || !hgrid_active_ || !dhdt_active_ || !beta_grid_) {
            RCLCPP_ERROR(this->get_logger(), "Memory allocation failed for field insertion buffers");
            throw std::runtime_error("Field insertion buffer allocation failed");
        }
    
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

        // One scratch slice per yaw layer. build_inflated_boundaries() runs its
        // q loop under OpenMP, so every buffer touched inside find_boundary()
        // and inflate_occupancy_grid() must be private to the slice being
        // processed; a single shared scratch buffer would be a data race.
        boundary_temp_ = static_cast<float*>(std::malloc(IMAX * JMAX * QMAX * sizeof(float)));
        inflate_bound_temp_ = static_cast<float*>(std::malloc(IMAX * JMAX * QMAX * sizeof(float)));
        inflate_class_temp_ = static_cast<int8_t*>(std::malloc(IMAX * JMAX * QMAX * sizeof(int8_t)));

        // Per-slice copy of the semantic class map. inflate_occupancy_grid()
        // dilates the class labels along with the boundary, so each yaw layer
        // needs its own copy to dilate independently from the same input.
        semantic_class_slices_ = static_cast<int8_t*>(std::malloc(IMAX * JMAX * QMAX * sizeof(int8_t)));

        if (!hgrid_temp_ || !guidance_x_temp_ || !guidance_y_temp_ ||
            !forcing_zero_temp_ || !bound_guidance_temp_ ||
            !class_map_temp_expanded_ || !boundary_temp_ ||
            !inflate_bound_temp_ || !inflate_class_temp_ ||
            !semantic_class_slices_) {
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
            hgrid_insertion_old_[n] = h0;
            hgrid_active_[n] = h0;
            dhdt_active_[n] = 0.0f;
            beta_grid_[n] = 0.0f;
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

        semantic_occupancy_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("semantic_occupancy_grid",1);

        const auto semantic_control_qos = rclcpp::QoS(rclcpp::KeepLast(1))
            .reliable()
            .transient_local();
        semantic_perception_required_pub_ =
            this->create_publisher<std_msgs::msg::Bool>(
                "/semantic_perception_required",
                semantic_control_qos
            );
            
        occ_grid_suber_ = this->create_subscription<nav_msgs::msg::OccupancyGrid>(
            "occupancy_grid", 1,
            std::bind(&PoissonControllerNode::occ_grid_callback, this, std::placeholders::_1),
            options_occ
        );

        // The semantic_map_fuser publishes the union of all already-expanded
        // class-specific safety targets on this topic. It shares the occupancy
        // callback group so the external semantic grid cannot be modified while
        // an occupancy update is constructing a new Poisson field.
        semantic_safety_target_suber_ =
            this->create_subscription<nav_msgs::msg::OccupancyGrid>(
                semantic_safety_target_topic_,
                rclcpp::QoS(rclcpp::KeepLast(1)).reliable(),
                std::bind(
                    &PoissonControllerNode::semantic_safety_target_callback,
                    this,
                    std::placeholders::_1
                ),
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
        poisson_solver_boundary_pub_ =
            this->create_publisher<nav_msgs::msg::OccupancyGrid>(
                "/poisson/solver_boundary",
                rclcpp::QoS(rclcpp::KeepLast(1)).reliable()
            );
        logging_data_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/poisson/logging_data", 10);
        profiling_data_pub_ = this->create_publisher<std_msgs::msg::Float32MultiArray>("/poisson/profiling_data", 10);
        relational_debug_grid_.resize(IMAX * JMAX, 0);
        relational_debug_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>("/relational_debug_map",10);
        mpc_callback_group_ = this->create_callback_group(rclcpp::CallbackGroupType::Reentrant);
        mpc_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(10),
            std::bind(&PoissonControllerNode::mpc_callback, this),
            mpc_callback_group_
        );
    }

    void publish_logging_data() {
        if (!logging_data_pub_) return;

        std_msgs::msg::Float32MultiArray msg;
        msg.data = {
            t_ms,
            static_cast<float>(space_counter),

            x[0], x[1], x[2],

            v[0], v[1], v[2],
            vt[0], vt[1], vt[2],

            h, dhdx, dhdy, dhdq, dhdt,
            wn,
            static_cast<float>(realtime_sf_flag | predictive_sf_flag),

            semantic_update_.lambda,
            semantic_update_.lambda_dot,
            static_cast<float>(semantic_update_.active),
            new_constraint_event_flag_ ? 1.0f : 0.0f,
            static_cast<float>(constraint_event_counter_)
        };

        logging_data_pub_->publish(msg);

        new_constraint_event_flag_ = false;
    }

    void update_persistent_human_memory_from_expanded_map() {
        if (!enable_human_persistence_) {
            std::fill(
                persistent_human_confidence_.begin(),
                persistent_human_confidence_.end(),
                0.0f
            );
    
            std::fill(
                persistent_human_mask_.begin(),
                persistent_human_mask_.end(),
                0
            );
    
            return;
        }
    
        for (int n = 0; n < IMAX * JMAX; ++n) {
            if (class_map_expanded[n] == 1) {
                persistent_human_confidence_[n] =
                    human_persistence_observation_value_;
            } else {
                persistent_human_confidence_[n] *= human_persistence_decay_;
            }
    
            persistent_human_mask_[n] =
                persistent_human_confidence_[n] >= human_persistence_threshold_
                    ? 1
                    : 0;
        }
    }
    
    void inject_persistent_humans_into_expanded_map() {
        if (!enable_human_persistence_) {
            return;
        }
    
        for (int n = 0; n < IMAX * JMAX; ++n) {
            if (persistent_human_mask_[n]) {
                class_map_expanded[n] = 1;
            }
        }
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

    // scratch must be an IMAX * JMAX buffer private to the caller; this runs
    // concurrently across yaw slices.
    void find_boundary(float* grid, float* bound, bool fix_flag, bool tight_area,
                       const int8_t* class_map, float* scratch) {
        for (int i = 0; i < IMAX; ++i) {
            for (int j = 0; j < JMAX; ++j) {
                if (i == 0 || i == IMAX - 1 || j == 0 || j == JMAX - 1) {
                    bound[i * JMAX + j] = 0.0f;
                }
            }
        }

        std::memcpy(scratch, bound, IMAX * JMAX * sizeof(float));
        float* b0 = scratch;
    
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
        robot_length = 1.0f;
        robot_width = 0.6f;
    
        const float ar = robot_length / 2.0f + mos/2;
        const float br = robot_width / 2.0f + mos/2;
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


    // bound_scratch (float) and class_scratch (int8_t) must be IMAX * JMAX
    // buffers private to the caller; this runs concurrently across yaw slices.
    // class_map is read and written (labels dilate with the boundary), so it
    // must be per-slice as well.
    void inflate_occupancy_grid(float* bound, int8_t* class_map,
                                float* bound_scratch, int8_t* class_scratch) {
        std::memcpy(bound_scratch, bound, IMAX * JMAX * sizeof(float));
        float* b0 = bound_scratch;

        int8_t* c0 = class_scratch;
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

        // Add persistence AFTER normal labeling and dilation.
        update_persistent_human_memory_from_expanded_map();
        inject_persistent_humans_into_expanded_map();

        // RCLCPP_INFO_THROTTLE(
        //     this->get_logger(),
        //     *this->get_clock(),
        //     2000,
        //     "Human persistence: retained_cells=%zu",
        //     std::count(
        //         persistent_human_mask_.begin(),
        //         persistent_human_mask_.end(),
        //         static_cast<uint8_t>(1)
        //     )
        // );
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
        h = trilinear_interpolation(hgrid_active_, ic, jc, qc);
        const float h_pred = h + dhdt * grid_age;
    
        // Guidance field (control direction) from Laplace solve
        // guidance_y corresponds to x-direction, guidance_x to y-direction
        const float vx = trilinear_interpolation(guidance_y_grid, ic, jc, qc);
        const float vy = trilinear_interpolation(guidance_x_grid, ic, jc, qc);
        const float v_norm = std::sqrt(vx * vx + vy * vy);
    
        // Numerical gradient of h-field in x/y
        const float h_eps = 1.0f;
        const float hip = trilinear_interpolation(hgrid_active_, ic + h_eps, jc, qc);
        const float him = trilinear_interpolation(hgrid_active_, ic - h_eps, jc, qc);
        const float hjp = trilinear_interpolation(hgrid_active_, ic, jc + h_eps, qc);
        const float hjm = trilinear_interpolation(hgrid_active_, ic, jc - h_eps, qc);
    
        const float Dh_x = (hjp - hjm) / (2.0f * h_eps * DS);
        const float Dh_y = (hip - him) / (2.0f * h_eps * DS);
    
        // Store guidance direction for logging/visualization
        dhdx = vx;
        dhdy = vy;
    
        // Numerical derivative in yaw
        const float q_eps = 1.0f;
        const float qp = q_wrap(qc + q_eps);
        const float qm = q_wrap(qc - q_eps);
    
        float hqp = trilinear_interpolation(hgrid_active_, ic, jc, qp);
        float hqm = trilinear_interpolation(hgrid_active_, ic, jc, qm);
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

    FieldBuffer active_field_;
    FieldBuffer pending_field_;

    const float h0 = 0.0f;
    const float dh0 = 1.0f;
    float wn = 1.0f;
    float issf = 50.0f;

    bool h_flag = false;
    bool dhdt_flag = false;
    bool save_flag = false;
    bool start_flag = false;
    bool enable_display = true;
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
    int8_t* semantic_class_slices_{};

    bool new_constraint_event_flag_{false};
    int constraint_event_counter_{0};
    float guidance_x_display[IMAX * JMAX];
    float guidance_y_display[IMAX * JMAX];
    float bound_display[IMAX * JMAX];
    int8_t tangent_layer_display[IMAX * JMAX];

    float robot_length{}, robot_width{};
    float robot_MOS_human{}, robot_MOS_obstacle{};
    int robot_kernel_dim_human{}, robot_kernel_dim_obstacle{};

    float* hgrid_insertion_old_{nullptr};
    float* hgrid_active_{nullptr};
    float* dhdt_active_{nullptr};
    float* beta_grid_{nullptr};

    SemanticUpdateState semantic_update_;
    ConstraintManager constraint_manager_;

    // constraint_runtime_config_ is always the currently admitted config used
    // by active control paths.  Candidate edits are staged separately.
    ConstraintRuntimeConfig constraint_runtime_config_;
    ConstraintRuntimeConfig admitted_constraint_config_;
    ConstraintRuntimeConfig candidate_constraint_config_;

    bool candidate_constraint_pending_{false};

    std::string admitted_constraints_json_;
    std::string candidate_constraints_json_;

    std::size_t admitted_constraints_file_signature_{0};
    std::size_t candidate_constraints_file_signature_{0};

    std::string constraints_path_;

    std::size_t last_constraints_file_signature_{0};
    bool constraints_signature_initialized_{false};
    bool semantic_candidate_target_ready_{false};
    float semantic_transition_extent_m_{0.0f};
    float semantic_old_max_depth_m_{0.0f};
    float semantic_target_max_depth_m_{0.0f};
    double admission_v_admissible_mps_{0.5};

    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr semantic_perception_required_pub_;
    bool semantic_perception_required_{false};
    bool semantic_perception_state_initialized_{false};

    rclcpp::TimerBase::SharedPtr constraints_reload_timer_;
    rclcpp::CallbackGroup::SharedPtr constraints_reload_callback_group_;
    double constraints_reload_hz_{1.0};

    rclcpp::CallbackGroup::SharedPtr mpc_callback_group_;
    rclcpp::TimerBase::SharedPtr mpc_timer_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr key_suber_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr twist_suber_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr occ_grid_suber_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr semantic_safety_target_suber_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr class_map_suber_;
    rclcpp::Subscription<nav_msgs::msg::OccupancyGrid>::SharedPtr visibility_map_suber_;
    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr pose_suber_;

    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::Image>> image_sub_;
    std::shared_ptr<message_filters::Subscriber<sensor_msgs::msg::PointCloud2>> cloud_sub_;

    int8_t class_map[IMAX * JMAX];
    int8_t visibility_map[IMAX * JMAX];
    int8_t class_map_expanded[IMAX * JMAX];

    std::vector<int8_t> semantic_occupancy_grid_;
    std::vector<int8_t> semantic_previous_grid_;
    std::vector<int8_t> semantic_target_grid_;
    std::vector<int8_t> semantic_current_grid_;

    // Rate Limiting semantic boundary growth (see enable_sdf_rate_limited_semantic_growth_).
    bool enable_sdf_rate_limited_semantic_growth_{false};
    std::vector<float> semantic_sdf_active_;

    std::vector<float> physical_occ_previous_;
    std::vector<float> physical_occ_current_;
    std::vector<uint8_t> physical_change_mask_;

    // Combined expanded target produced by semantic_map_fuser.
    std::vector<int8_t> external_semantic_safety_grid_;

    // Transaction bridge for rejected rule sets.  The admitted snapshot is
    // used only until semantic_map_fuser reloads the restored admitted JSON.
    std::vector<int8_t> admitted_external_semantic_grid_snapshot_;
    std::vector<int8_t> rejected_candidate_external_grid_;
    bool rollback_waiting_for_external_refresh_{false};

    bool external_semantic_safety_received_{false};
    std::chrono::steady_clock::time_point external_semantic_safety_timestamp_;
    std::string semantic_safety_target_topic_{"/semantic_safety_target"};
    int semantic_safety_occupied_threshold_{50};
    double semantic_safety_max_age_sec_{1.0};

    std::vector<float> persistent_human_confidence_;
    std::vector<uint8_t> persistent_human_mask_;
    
    float human_persistence_decay_{0.96f};
    float human_persistence_threshold_{0.25f};
    float human_persistence_observation_value_{1.0f};
    bool enable_human_persistence_{true};
    
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

    std::vector<int8_t> relational_debug_grid_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr relational_debug_pub_;

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
    std::ofstream outFileMPCVel;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr poisson_image_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr
        poisson_solver_boundary_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr logging_data_pub_;
    rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr profiling_data_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr semantic_occupancy_pub_;
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

    // RCLCPP_INFO(
    //     poissonNode->get_logger(),
    //     "Passing min_z=%.2f, max_z=%.2f to CloudMergerNode",
    //     min_z, max_z
    // );

    auto mappingNode = std::make_shared<CloudMergerNode>(min_z, max_z);

    executor.add_node(mappingNode);
    executor.add_node(poissonNode);

    RCLCPP_INFO(
        poissonNode->get_logger(),
        "Poisson node added to executor"
    );

    try {
        executor.spin();
        throw("Terminated");
    } catch (const char* msg) {
        rclcpp::shutdown();
        std::cout << msg << std::endl;
    }

    return 0;
}
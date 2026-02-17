#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <rclcpp/rclcpp.hpp>

/**
 * @brief Information about a LiDAR cluster extracted via connected components
 */
struct ClusterInfo {
    float centroid_x;       // Centroid X in odom frame (meters)
    float centroid_y;       // Centroid Y in odom frame (meters)
    int cell_count;         // Number of cells in cluster
    bool has_yolo_seed;     // Does any cell in cluster have YOLO human label?
    bool in_camera_fov;     // Is cluster within camera's field of view?
    int label_id;           // Connected component label ID (for cell lookup)
};

/**
 * @brief A tracked human object with position, velocity, and confidence
 */
struct HumanTrack {
    int id;                     // Unique track ID
    float x, y;                 // Centroid in odom frame (meters)
    float vx, vy;               // Velocity estimate (m/s)
    float heading_x, heading_y; // Unit vector of facing direction (derived from velocity)
    bool heading_valid;         // True if heading is reliable (speed > threshold)
    float last_update_time;     // Timestamp of last observation (seconds)
    float confidence;           // Confidence level [0, 1], decays when unobserved
    bool yolo_confirmed;        // Was YOLO seen this frame?
    bool yolo_ever_confirmed;   // Was this track ever directly YOLO-confirmed?
    int yolo_confirm_count;     // Consecutive frames with YOLO detection
    int moving_frame_count;     // Consecutive frames above velocity threshold
    int stationary_frame_count; // Consecutive frames below velocity threshold
    
    HumanTrack() : id(-1), x(0), y(0), vx(0), vy(0), 
                   heading_x(0), heading_y(1), heading_valid(false),
                   last_update_time(0), confidence(1.0f), yolo_confirmed(false),
                   yolo_ever_confirmed(false), yolo_confirm_count(0), moving_frame_count(0),
                   stationary_frame_count(0) {}
};

/**
 * @brief Object-level human tracker using greedy nearest neighbor association
 * 
 * Replaces cell-based decay with position-velocity prediction.
 * Handles humans moving out of camera FOV by tracking LiDAR clusters.
 */
class HumanTracker {
public:
    /**
     * @brief Construct tracker with configurable parameters
     * @param timeout_sec How long to keep track after last observation
     * @param gate_radius Maximum distance to associate cluster with track (meters)
     * @param velocity_decay_tau Time constant for velocity decay (seconds)
     * @param velocity_threshold Speed threshold to consider cluster as moving (m/s)
     * @param decay_in_fov Decay rate when in camera FOV but no YOLO (default 0.7)
     * @param decay_stationary Decay rate when stationary outside camera (default 0.95)
     * @param decay_unconfirmed Decay rate when never YOLO confirmed (default 0.85)
     * @param min_yolo_confirms Consecutive YOLO detections needed to confirm human (default 3)
     * @param min_moving_frames Consecutive moving frames needed to boost confidence outside FOV (default 3)
     * @param no_retrack_on_move If true, lost tracks won't be re-tracked when cluster moves again (default true)
     */
    HumanTracker(float timeout_sec = 10.0f, 
                 float gate_radius = 0.8f, 
                 float velocity_decay_tau = 1.0f,
                 float velocity_threshold = 0.1f,
                 float decay_in_fov = 0.7f,
                 float decay_stationary = 0.95f,
                 float decay_unconfirmed = 0.85f,
                 int min_yolo_confirms = 3,
                 int min_moving_frames = 3,
                 bool no_retrack_on_move = true)
        : timeout_sec_(timeout_sec),
          gate_radius_(gate_radius),
          velocity_decay_tau_(velocity_decay_tau),
          velocity_threshold_(velocity_threshold),
          decay_in_fov_(decay_in_fov),
          decay_stationary_(decay_stationary),
          decay_unconfirmed_(decay_unconfirmed),
          min_yolo_confirms_(min_yolo_confirms),
          min_moving_frames_(min_moving_frames),
          no_retrack_on_move_(no_retrack_on_move),
          next_track_id_(0),
          last_update_time_(0.0f) {}

    /**
     * @brief Main update function - call each frame with extracted clusters
     * @param clusters LiDAR clusters from connected components
     * @param current_time Current ROS time in seconds
     */
    void update(const std::vector<ClusterInfo>& clusters, float current_time) {
        float dt = (last_update_time_ > 0.0f) ? (current_time - last_update_time_) : 0.0f;
        last_update_time_ = current_time;
        
        // Clamp dt to reasonable range (handles first frame and pauses)
        dt = std::clamp(dt, 0.0f, 0.5f);
        
        // Step 1: Predict existing tracks forward
        predict_tracks(dt);
        
        // Step 2: Associate clusters to tracks and update
        associate_and_update(clusters, current_time);
        
        // Step 3: Remove stale tracks
        prune_stale_tracks(current_time);
    }

    /**
     * @brief Get all active tracks for labeling
     */
    std::vector<HumanTrack> get_active_tracks() const {
        return tracks_;
    }
    
    /**
     * @brief Get number of active tracks
     */
    size_t num_tracks() const {
        return tracks_.size();
    }

private:
    /**
     * @brief Predict track positions using velocity, apply decay
     */
    void predict_tracks(float dt) {
        for (auto& track : tracks_) {
            // Position prediction
            track.x += track.vx * dt;
            track.y += track.vy * dt;
            
            // Velocity decay toward zero (handles stops gracefully)
            float decay = std::exp(-dt / velocity_decay_tau_);
            track.vx *= decay;
            track.vy *= decay;
            
            // Reset per-frame YOLO flag (confidence decay handled in associate_and_update)
            track.yolo_confirmed = false;
        }
    }

    /**
     * @brief Greedy nearest neighbor association
     */
    void associate_and_update(const std::vector<ClusterInfo>& clusters, float current_time) {
        std::vector<bool> cluster_matched(clusters.size(), false);
        
        // Sort tracks by confidence (high confidence matched first)
        std::vector<size_t> track_order(tracks_.size());
        for (size_t i = 0; i < tracks_.size(); i++) track_order[i] = i;
        std::sort(track_order.begin(), track_order.end(), [this](size_t a, size_t b) {
            return tracks_[a].confidence > tracks_[b].confidence;
        });
        
        // Match existing tracks to clusters (greedy nearest neighbor)
        for (size_t ti : track_order) {
            auto& track = tracks_[ti];
            float best_dist = gate_radius_;
            int best_idx = -1;
            
            for (size_t i = 0; i < clusters.size(); i++) {
                if (cluster_matched[i]) continue;
                
                float dx = clusters[i].centroid_x - track.x;
                float dy = clusters[i].centroid_y - track.y;
                float dist = std::sqrt(dx * dx + dy * dy);
                
                if (dist < best_dist) {
                    best_dist = dist;
                    best_idx = static_cast<int>(i);
                }
            }
            
            if (best_idx >= 0) {
                // Update track with observation
                const auto& c = clusters[best_idx];
                float dt = current_time - track.last_update_time;
                
                // Update velocity estimate (smoothed)
                if (dt > 0.01f && dt < 1.0f) {
                    float obs_vx = (c.centroid_x - track.x) / dt;
                    float obs_vy = (c.centroid_y - track.y) / dt;
                    track.vx = 0.5f * track.vx + 0.5f * obs_vx;
                    track.vy = 0.5f * track.vy + 0.5f * obs_vy;
                    
                    // Update heading from velocity when speed exceeds threshold
                    float speed = std::sqrt(track.vx * track.vx + track.vy * track.vy);
                    if (speed > velocity_threshold_) {
                        track.heading_x = track.vx / speed;
                        track.heading_y = track.vy / speed;
                        track.heading_valid = true;
                    }
                    // When stationary, keep previous heading (don't invalidate)
                }
                
                // Update position
                track.x = c.centroid_x;
                track.y = c.centroid_y;
                track.last_update_time = current_time;
                track.yolo_confirmed = c.has_yolo_seed;
                
                // Confidence update based on YOLO confirmation, FOV, and velocity
                if (c.has_yolo_seed) {
                    track.yolo_confirm_count++;  // Increment YOLO count
                    if (track.yolo_confirm_count >= min_yolo_confirms_) {
                        track.yolo_ever_confirmed = true;  // Confirm after N detections
                    }
                    track.confidence = 1.0f;           // Full confidence when YOLO sees it
                    track.moving_frame_count = 0;      // Reset moving counter when in camera view
                } else if (c.in_camera_fov) {
                    // In camera view but NOT detected as human
                    // Decrement counter instead of resetting (allows for dropped frames)
                    track.yolo_confirm_count = std::max(0, track.yolo_confirm_count - 1);
                    // Only clear confirmation after multiple consecutive misses
                    if (track.yolo_confirm_count == 0) {
                        track.yolo_ever_confirmed = false;  // Clear confirmation
                    }
                    track.confidence *= decay_in_fov_;  // Fast decay
                    track.moving_frame_count = 0;       // Reset moving counter
                } else if (track.yolo_ever_confirmed) {
                    // Outside camera view, was confirmed - use velocity-based persistence
                    float speed = std::sqrt(track.vx * track.vx + track.vy * track.vy);
                    
                    if (speed > velocity_threshold_) {
                        // Moving: high confidence, reset stationary counter
                        track.moving_frame_count++;
                        
                        // Check if we should allow re-tracking on movement
                        if (no_retrack_on_move_ && track.stationary_frame_count >= min_moving_frames_) {
                            // Track was confirmed stationary before - don't re-track on movement
                            // Keep decaying, require fresh YOLO to re-confirm
                            track.confidence *= decay_stationary_;
                        } else {
                            // Normal case: moving track maintains confidence
                            track.stationary_frame_count = 0;
                            track.confidence = std::max(track.confidence, 0.9f);
                        }
                    } else {
                        // Below velocity threshold
                        track.stationary_frame_count++;
                        track.moving_frame_count = 0;
                        
                        if (track.stationary_frame_count >= min_moving_frames_) {
                            // Confirmed stationary: apply slow decay
                            track.confidence *= decay_stationary_;
                        } else {
                            // Not yet confirmed stationary (hysteresis): maintain confidence
                            track.confidence = std::max(track.confidence, 0.9f);
                        }
                    }
                } else {
                    // Never directly confirmed - fast decay
                    track.moving_frame_count = 0;
                    track.confidence *= decay_unconfirmed_;
                }
                
                cluster_matched[best_idx] = true;
            }
        }
        
        // Create new tracks from unmatched YOLO-confirmed clusters
        // NOTE: New tracks start with yolo_ever_confirmed=false and must accumulate
        // min_yolo_confirms consecutive detections before being considered confirmed.
        for (size_t i = 0; i < clusters.size(); i++) {
            if (!cluster_matched[i] && clusters[i].has_yolo_seed) {
                HumanTrack new_track;
                new_track.id = next_track_id_++;
                new_track.x = clusters[i].centroid_x;
                new_track.y = clusters[i].centroid_y;
                new_track.vx = 0.0f;
                new_track.vy = 0.0f;
                new_track.last_update_time = current_time;
                new_track.confidence = 1.0f;
                new_track.yolo_confirmed = true;
                new_track.yolo_confirm_count = 1;      // First detection
                new_track.yolo_ever_confirmed = false; // Not confirmed until N consecutive detections
                new_track.moving_frame_count = 0;
                tracks_.push_back(new_track);
            }
        }
    }

    /**
     * @brief Remove tracks that haven't been updated within timeout
     */
    void prune_stale_tracks(float current_time) {
        tracks_.erase(
            std::remove_if(tracks_.begin(), tracks_.end(),
                [this, current_time](const HumanTrack& t) {
                    return (current_time - t.last_update_time) > timeout_sec_;
                }),
            tracks_.end()
        );
    }

    std::vector<HumanTrack> tracks_;
    float timeout_sec_;
    float gate_radius_;
    float velocity_decay_tau_;
    float velocity_threshold_;
    float decay_in_fov_;
    float decay_stationary_;
    float decay_unconfirmed_;
    int min_yolo_confirms_;
    int min_moving_frames_;
    bool no_retrack_on_move_;
    int next_track_id_;
    float last_update_time_;
};

#pragma once

#include <string>

struct ConstraintRuntimeConfig {
    // Exclusion / safe-domain modifications
    float human_buffer_m{-1.0f};
    float obstacle_buffer_m{-1.0f};

    // Future: guidance behavior
    bool enable_social_navigation_override{false};
    bool enable_social_navigation{false};
    float social_tangent_bias{-1.0f};
    int social_tangent_layers{-1};
    int social_layer_thickness{-1};

    // Future: kinematic constraints
    bool enable_velocity_override{false};
    float max_speed_near_human_mps{-1.0f};
    float human_slowdown_radius_m{-1.0f};

    // Future: semantic risk field
    bool enable_risk_field{false};
    float semantic_risk_weight{0.0f};
};

class ConstraintManager {
public:
    bool load_from_json(const std::string& path);
    const ConstraintRuntimeConfig& get_config() const;

private:
    ConstraintRuntimeConfig config_;

    void parse_exclusion_constraint(const nlohmann::json& c);
};

#pragma once

#include <string>
#include <vector>

enum class ConstraintType {
    Exclusion,
    Proximity,
    RelativePosition,
    VelocityLimit,
    Heading,
    WorkspaceRegion,
    Unknown
};

struct RuntimeConstraint {
    std::string id;
    ConstraintType type{ConstraintType::Unknown};
    bool enabled{true};
    bool enforce{false};

    std::vector<std::string> target_classes;
    std::vector<std::string> reference_classes;

    float buffer_distance_m{-1.0f};
    float min_distance_m{-1.0f};
    float max_distance_m{-1.0f};

    float max_linear_velocity_mps{-1.0f};
    float max_angular_velocity_radps{-1.0f};

    std::string raw_type;
};

struct ConstraintRuntimeConfig {
    // New architecture: store every parsed constraint.
    std::vector<RuntimeConstraint> constraints;

    // Legacy compatibility: semantic_poisson currently still uses this.
    float human_buffer_m{-1.0f};
};

class ConstraintManager {
public:
    bool load_from_json(const std::string& path);

    const ConstraintRuntimeConfig& get_config() const;

private:
    ConstraintRuntimeConfig config_;

    ConstraintType parse_constraint_type(const std::string& type) const;
    std::string constraint_type_to_string(ConstraintType type) const;
};

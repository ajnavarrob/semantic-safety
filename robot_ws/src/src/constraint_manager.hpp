#pragma once

#include <string>
#include <vector>
#include <unordered_map>

struct SemanticConstraint {
    std::string id;
    bool enabled{true};

    std::string type;  // exclusion, relational, kinematic

    std::vector<std::string> target_objects;
    std::vector<std::string> object_a;
    std::vector<std::string> object_b;

    float buffer_distance_m{0.0f};
    std::string relation;
    std::string reference_frame;

    std::string behavior_mode;
};

struct CompiledConstraintConfig {
    std::unordered_map<std::string, float> class_buffer_overrides_m;
    std::unordered_map<std::string, std::string> pass_side_by_class;

    bool has_forbidden_between{false};
    std::string forbidden_between_a;
    std::string forbidden_between_b;
};

class ConstraintManager {
public:
    bool load_from_json(const std::string& path);

    const std::vector<SemanticConstraint>& constraints() const {
        return constraints_;
    }

    CompiledConstraintConfig compile() const;

private:
    std::vector<SemanticConstraint> constraints_;
};

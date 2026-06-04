#include "constraints/constraint_manager.hpp"

#include <fstream>
#include <iostream>
#include <sstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

ConstraintType ConstraintManager::parse_constraint_type(
    const std::string& type
) const {
    if (type == "exclusion" || type == "avoidance") {
        return ConstraintType::Exclusion;
    }
    if (type == "proximity") {
        return ConstraintType::Proximity;
    }
    if (type == "relative_position") {
        return ConstraintType::RelativePosition;
    }
    if (type == "velocity_limit") {
        return ConstraintType::VelocityLimit;
    }
    if (type == "heading" || type == "facing") {
        return ConstraintType::Heading;
    }
    if (type == "workspace_region") {
        return ConstraintType::WorkspaceRegion;
    }

    return ConstraintType::Unknown;
}

std::string ConstraintManager::constraint_type_to_string(
    ConstraintType type
) const {
    switch (type) {
        case ConstraintType::Exclusion:
            return "exclusion";
        case ConstraintType::Proximity:
            return "proximity";
        case ConstraintType::RelativePosition:
            return "relative_position";
        case ConstraintType::VelocityLimit:
            return "velocity_limit";
        case ConstraintType::Heading:
            return "heading";
        case ConstraintType::WorkspaceRegion:
            return "workspace_region";
        default:
            return "unknown";
    }
}

static std::vector<std::string> parse_string_or_string_array(
    const json& j,
    const std::string& key
) {
    std::vector<std::string> out;

    if (!j.contains(key)) {
        return out;
    }

    const auto& value = j.at(key);

    if (value.is_string()) {
        out.push_back(value.get<std::string>());
        return out;
    }

    if (value.is_array()) {
        for (const auto& item : value) {
            if (item.is_string()) {
                out.push_back(item.get<std::string>());
            }
        }
    }

    return out;
}

bool ConstraintManager::load_from_json(const std::string& path) {
    config_ = ConstraintRuntimeConfig{};

    if (path.empty()) {
        std::cout << "[ConstraintManager] Empty JSON path. Using defaults.\n";
        return false;
    }

    std::ifstream f(path);
    if (!f.is_open()) {
        std::cout << "[ConstraintManager] Could not open file: "
                  << path << "\n";
        return false;
    }

    json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        std::cout << "[ConstraintManager] Failed to parse JSON: "
                  << e.what() << "\n";
        return false;
    }

    if (!j.contains("constraints") || !j["constraints"].is_array()) {
        std::cout << "[ConstraintManager] JSON missing 'constraints' array.\n";
        return false;
    }

    const std::string schema_version =
        j.value("schema_version", "unknown");

    std::cout << "[ConstraintManager] Loading constraints schema version: "
              << schema_version << "\n";

    int parsed_count = 0;
    int enabled_count = 0;

    for (const auto& c : j["constraints"]) {
        RuntimeConstraint rc;

        rc.id = c.value("id", "unnamed_constraint");
        rc.raw_type = c.value("type", "");
        rc.type = parse_constraint_type(rc.raw_type);
        rc.enabled = c.value("enabled", true);
        rc.enforce = c.value("enforce", false);

        parsed_count++;

        if (!rc.enabled) {
            std::cout << "[ConstraintManager] Skipping disabled constraint: "
                      << rc.id << "\n";
            continue;
        }

        enabled_count++;

        if (c.contains("target") && c["target"].is_object()) {
            const auto& target = c["target"];

            rc.target_classes =
                parse_string_or_string_array(target, "semantic_class");

            // Backward compatibility with older schema:
            // "objects": { "target": ["person"] }
        } else if (c.contains("objects") && c["objects"].is_object()) {
            const auto& objects = c["objects"];

            if (objects.contains("target")) {
                const auto& target = objects["target"];

                if (target.is_string()) {
                    rc.target_classes.push_back(target.get<std::string>());
                } else if (target.is_array()) {
                    for (const auto& item : target) {
                        if (item.is_string()) {
                            rc.target_classes.push_back(
                                item.get<std::string>()
                            );
                        }
                    }
                }
            }
        }

        if (c.contains("reference") && c["reference"].is_object()) {
            const auto& reference = c["reference"];
            rc.reference_classes =
                parse_string_or_string_array(reference, "semantic_class");
        }

        if (c.contains("spatial_parameters") &&
            c["spatial_parameters"].is_object()) {
            const auto& sp = c["spatial_parameters"];

            rc.buffer_distance_m =
                sp.value("buffer_distance_m", -1.0f);

            rc.min_distance_m =
                sp.value("min_distance_m", -1.0f);

            rc.max_distance_m =
                sp.value("max_distance_m", -1.0f);
        }

        if (c.contains("control_parameters") &&
            c["control_parameters"].is_object()) {
            const auto& cp = c["control_parameters"];

            rc.max_linear_velocity_mps =
                cp.value("max_linear_velocity_mps", -1.0f);

            rc.max_angular_velocity_radps =
                cp.value("max_angular_velocity_radps", -1.0f);
        }

        config_.constraints.push_back(rc);

        std::cout << "[ConstraintManager] Parsed constraint: "
                  << "id=" << rc.id
                  << ", type=" << constraint_type_to_string(rc.type)
                  << ", enforce=" << (rc.enforce ? "true" : "false")
                  << ", targets=[";

        for (size_t i = 0; i < rc.target_classes.size(); ++i) {
            std::cout << rc.target_classes[i];
            if (i + 1 < rc.target_classes.size()) {
                std::cout << ",";
            }
        }

        std::cout << "]"
                  << ", buffer=" << rc.buffer_distance_m
                  << ", min_dist=" << rc.min_distance_m
                  << ", max_dist=" << rc.max_distance_m
                  << "\n";

        // Legacy behavior:
        // Current semantic_poisson only knows how to enforce person exclusion
        // through robot_MOS_human / human_buffer_m.
        if (rc.type == ConstraintType::Exclusion &&
            rc.enforce &&
            rc.buffer_distance_m > 0.0f) {
            for (const auto& target : rc.target_classes) {
                if (target == "person" || target == "human") {
                    config_.human_buffer_m = rc.buffer_distance_m;

                    std::cout
                        << "[ConstraintManager] Legacy human buffer override: "
                        << config_.human_buffer_m << " m from constraint "
                        << rc.id << "\n";
                }
            }
        }
    }

    std::cout << "[ConstraintManager] Loaded "
              << config_.constraints.size()
              << " enabled constraints out of "
              << parsed_count
              << " parsed constraints.\n";

    return true;
}

const ConstraintRuntimeConfig& ConstraintManager::get_config() const {
    return config_;
}

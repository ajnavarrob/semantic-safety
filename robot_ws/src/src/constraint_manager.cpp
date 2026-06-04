#include "constraints/constraint_manager.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

bool ConstraintManager::load_from_json(const std::string& path) {
    config_ = ConstraintRuntimeConfig{};

    if (path.empty()) {
        std::cout << "[ConstraintManager] Empty JSON path. Using defaults.\n";
        return false;
    }

    std::ifstream f(path);
    if (!f.is_open()) {
        std::cout << "[ConstraintManager] Could not open file: " << path << "\n";
        return false;
    }

    json j;
    try {
        f >> j;
    } catch (const std::exception& e) {
        std::cout << "[ConstraintManager] Failed to parse JSON: " << e.what() << "\n";
        return false;
    }

    if (!j.contains("constraints") || !j["constraints"].is_array()) {
        std::cout << "[ConstraintManager] JSON missing 'constraints' array.\n";
        return false;
    }

    const std::string schema_version = j.value("schema_version", "legacy");
    std::cout << "[ConstraintManager] Loading constraints schema version: "
              << schema_version << "\n";

    for (const auto& c : j["constraints"]) {
        const bool enabled = c.value("enabled", true);
        if (!enabled) continue;

        const std::string type = c.value("type", "");

        if (type == "exclusion") {
            parse_exclusion_constraint(c);
        } else {
            std::cout << "[ConstraintManager] Unsupported constraint type: "
                      << type << "\n";
        }
    }

    return true;
}

void ConstraintManager::parse_exclusion_constraint(const json& c) {
    std::string semantic_class;

    // New schema
    if (c.contains("target") && c["target"].is_object()) {
        semantic_class = c["target"].value("semantic_class", "");
    }

    // Legacy schema support:
    // {
    //   "objects": { "target": ["person"] }
    // }
    if (semantic_class.empty()) {
        if (c.contains("objects") &&
            c["objects"].contains("target") &&
            c["objects"]["target"].is_array() &&
            !c["objects"]["target"].empty()) {
            semantic_class = c["objects"]["target"][0].get<std::string>();
        }
    }

    if (semantic_class.empty()) {
        std::cout << "[ConstraintManager] Exclusion constraint missing target semantic class.\n";
        return;
    }

    if (!c.contains("spatial_parameters") || !c["spatial_parameters"].is_object()) {
        std::cout << "[ConstraintManager] Exclusion constraint missing spatial_parameters.\n";
        return;
    }

    const float buffer =
        c["spatial_parameters"].value("buffer_distance_m", -1.0f);

    if (buffer <= 0.0f) {
        std::cout << "[ConstraintManager] Invalid buffer distance: "
                  << buffer << "\n";
        return;
    }

    if (semantic_class == "person" || semantic_class == "human") {
        config_.human_buffer_m = buffer;
        std::cout << "[ConstraintManager] Loaded human/person buffer override: "
                  << buffer << " m\n";
    } else if (semantic_class == "obstacle") {
        config_.obstacle_buffer_m = buffer;
        std::cout << "[ConstraintManager] Loaded obstacle buffer override: "
                  << buffer << " m\n";
    } else {
        std::cout << "[ConstraintManager] Unsupported exclusion semantic class: "
                  << semantic_class << "\n";
    }
}

const ConstraintRuntimeConfig& ConstraintManager::get_config() const {
    return config_;
}

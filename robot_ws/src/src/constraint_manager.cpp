#include "constraints/constraint_manager.hpp"

#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

bool ConstraintManager::load_from_json(const std::string& path) {
    config_ = ConstraintRuntimeConfig{};  // reset each load

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

    for (const auto& c : j["constraints"]) {
        const bool enabled = c.value("enabled", true);
        if (!enabled) continue;

        const std::string type = c.value("type", "");
        if (type != "exclusion") continue;

        if (!c.contains("objects")) continue;
        if (!c["objects"].contains("target")) continue;
        if (!c.contains("spatial_parameters")) continue;

        const auto& targets = c["objects"]["target"];
        const float buffer =
            c["spatial_parameters"].value("buffer_distance_m", -1.0f);

        if (buffer <= 0.0f) continue;

        for (const auto& t : targets) {
            const std::string target = t.get<std::string>();

            // FIRST IMPLEMENTATION: only support person exclusion
            if (target == "person") {
                config_.human_buffer_m = buffer;
                std::cout << "[ConstraintManager] Loaded human buffer override: "
                          << buffer << " m\n";
            }
        }
    }

    return true;
}

const ConstraintRuntimeConfig& ConstraintManager::get_config() const {
    return config_;
}
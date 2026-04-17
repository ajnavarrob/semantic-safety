#include "semantic_constraints/constraint_manager.hpp"
#include <fstream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

bool ConstraintManager::load_from_json(const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return false;

    json j;
    f >> j;

    constraints_.clear();

    for (const auto& jc : j["constraints"]) {
        SemanticConstraint c;
        c.id = jc.value("id", "");
        c.enabled = jc.value("enabled", true);
        c.type = jc.value("type", "");

        if (jc.contains("objects")) {
            const auto& objs = jc["objects"];
            if (objs.contains("target")) {
                c.target_objects = objs["target"].get<std::vector<std::string>>();
            }
            if (objs.contains("object_a")) {
                c.object_a = objs["object_a"].get<std::vector<std::string>>();
            }
            if (objs.contains("object_b")) {
                c.object_b = objs["object_b"].get<std::vector<std::string>>();
            }
        }

        if (jc.contains("spatial_parameters")) {
            const auto& sp = jc["spatial_parameters"];
            c.buffer_distance_m = sp.value("buffer_distance_m", 0.0f);
            c.relation = sp.value("relation", "");
            c.reference_frame = sp.value("reference_frame", "");
        }

        if (jc.contains("behavior")) {
            const auto& bh = jc["behavior"];
            c.behavior_mode = bh.value("mode", "");
        }

        constraints_.push_back(c);
    }

    return true;
}

#pragma once

#include <string>

struct ConstraintRuntimeConfig {
    float human_buffer_m = -1.0f;  // -1 means "not provided"
};

class ConstraintManager {
public:
    bool load_from_json(const std::string& path);
    const ConstraintRuntimeConfig& get_config() const;

private:
    ConstraintRuntimeConfig config_;
};
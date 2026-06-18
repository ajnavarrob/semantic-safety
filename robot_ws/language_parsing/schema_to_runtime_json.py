import json
from typing import Any, Dict, List, Optional

from semantic_safety_schema import LLMCompilerOutput


DEFAULT_ENABLED = True
DEFAULT_ENFORCE = True

DEFAULT_BUFFER_DISTANCE_M = 1.0
DEFAULT_PROXIMITY_MAX_DISTANCE_M = 1.5
DEFAULT_DIRECTIONAL_RADIUS_M = 2.0
DEFAULT_DIRECTIONAL_MIN_RADIUS_M = 0.5
DEFAULT_DIRECTIONAL_CONE_HALF_ANGLE_DEG = 90.0

DEFAULT_MAX_LINEAR_VELOCITY_MPS = 0.25
DEFAULT_MAX_ANGULAR_VELOCITY_RADPS = 1.0


def class_target(class_name: str) -> Dict[str, List[str]]:
    return {
        "semantic_class": [class_name]
    }


def get_target_class(target: Dict[str, Any]) -> str:
    if target.get("kind") != "class":
        raise ValueError(f"Only class targets are supported in runtime adapter. Got: {target}")

    class_name = target.get("class")
    if not class_name:
        raise ValueError(f"Missing target class in: {target}")

    return class_name


def make_id(*parts: Optional[Any]) -> str:
    clean_parts = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip().lower()
        text = text.replace(" ", "_")
        text = text.replace(".", "_")
        text = text.replace("__", "_")
        text = text.strip("_")
        if text:
            clean_parts.append(text)

    return "_".join(clean_parts)


def adapt_spatial_constraint(constraint: Dict[str, Any]) -> Dict[str, Any]:
    mode = constraint["mode"]
    region = constraint["region"]
    constructor = region["constructor"]

    if constructor == "buffer":
        target_class = get_target_class(region["target"])
        distance = float(region.get("distance", DEFAULT_BUFFER_DISTANCE_M))

        if mode == "avoid":
            return {
                "id": constraint.get("id") or make_id("avoid", target_class),
                "type": "exclusion",
                "enabled": DEFAULT_ENABLED,
                "enforce": bool(constraint.get("enforce", DEFAULT_ENFORCE)),
                "target": class_target(target_class),
                "spatial_parameters": {
                    "buffer_distance_m": distance
                }
            }

        if mode == "remain":
            return {
                "id": constraint.get("id") or make_id("remain_near", target_class),
                "type": "proximity",
                "enabled": DEFAULT_ENABLED,
                "enforce": bool(constraint.get("enforce", DEFAULT_ENFORCE)),
                "target": class_target("robot"),
                "reference": class_target(target_class),
                "spatial_parameters": {
                    "max_distance_m": distance
                }
            }

        raise ValueError(f"Unsupported spatial buffer mode: {mode}")

    if constructor == "directional":
        target_class = get_target_class(region["target"])
        relation = region["relation"]
        distance = float(region.get("distance", DEFAULT_DIRECTIONAL_RADIUS_M))
        angle = float(region.get("angle", DEFAULT_DIRECTIONAL_CONE_HALF_ANGLE_DEG))

        runtime_mode = "allow_region" if mode == "remain" else "avoid_region"

        return {
            "id": constraint.get("id") or make_id(mode, relation, target_class),
            "type": "relational",
            "enabled": DEFAULT_ENABLED,
            "enforce": bool(constraint.get("enforce", DEFAULT_ENFORCE)),
            "relation": relation,
            "mode": runtime_mode,
            "target": class_target("robot"),
            "reference": class_target(target_class),
            "spatial_parameters": {
                "radius_m": distance,
                "cone_half_angle_deg": angle,
                "min_radius_m": DEFAULT_DIRECTIONAL_MIN_RADIUS_M,
                "max_radius_m": distance
            },
            "heading_parameters": {
                "speed_threshold_mps": 0.01,
                "heading_timeout_sec": 1.0
            }
        }

    raise ValueError(f"Unsupported spatial constructor for runtime adapter: {constructor}")


def adapt_behavior_constraint(constraint: Dict[str, Any]) -> Dict[str, Any]:
    mode = constraint["mode"]

    if mode != "activate":
        raise ValueError(f"Unsupported behavior mode: {mode}")

    activation_region = constraint["activation_region"]
    constructor = activation_region["constructor"]

    if constructor != "buffer":
        raise ValueError(
            f"Only buffer activation regions are supported for behavior constraints. Got: {constructor}"
        )

    target_class = get_target_class(activation_region["target"])
    distance = float(activation_region.get("distance", DEFAULT_BUFFER_DISTANCE_M))

    behavior = constraint["behavior"]
    behavior_kind = behavior["kind"]

    if behavior_kind != "velocity_limit":
        raise ValueError(f"Unsupported behavior kind: {behavior_kind}")

    max_speed = float(behavior.get("max_speed", DEFAULT_MAX_LINEAR_VELOCITY_MPS))

    return {
        "id": constraint.get("id") or make_id("velocity_limit", target_class),
        "type": "velocity_limit",
        "enabled": DEFAULT_ENABLED,
        "enforce": bool(constraint.get("enforce", DEFAULT_ENFORCE)),
        "target": class_target(target_class),
        "spatial_parameters": {
            "buffer_distance_m": distance
        },
        "control_parameters": {
            "max_linear_velocity_mps": max_speed,
            "max_angular_velocity_radps": DEFAULT_MAX_ANGULAR_VELOCITY_RADPS
        }
    }


def adapt_constraint(constraint: Dict[str, Any]) -> Dict[str, Any]:
    constraint_type = constraint["type"]

    if constraint_type == "spatial":
        return adapt_spatial_constraint(constraint)

    if constraint_type == "behavior":
        return adapt_behavior_constraint(constraint)

    raise ValueError(f"Unsupported constraint type: {constraint_type}")


def schema_to_runtime_json(
    compiler_output: LLMCompilerOutput,
    schema_version: str = "0.2",
) -> Dict[str, Any]:
    payload = compiler_output.model_dump(by_alias=True)

    if payload["status"] != "ok":
        return {
            "schema_version": schema_version,
            "constraints": [],
            "status": payload["status"],
            "reason": payload.get("reason"),
            "question": payload.get("question"),
        }

    runtime_constraints = []

    for command in payload["commands"]:
        action = command["action"]

        if action != "add":
            raise ValueError(f"Runtime adapter currently only supports action='add'. Got: {action}")

        runtime_constraint = adapt_constraint(command["constraint"])
        runtime_constraints.append(runtime_constraint)

    return {
        "schema_version": schema_version,
        "constraints": runtime_constraints
    }


if __name__ == "__main__":
    from compile_flat_to_schema import compile_flat_to_schema
    from llm_flat_test import FlatParse

    tests = [
        FlatParse(
            status="ok",
            intent="spatial",
            mode="avoid",
            constructor="buffer",
            target="person",
            distance=1.0,
            executable_in_v1=True,
        ),
        FlatParse(
            status="ok",
            intent="spatial",
            mode="remain",
            constructor="buffer",
            target="chair",
            distance=1.5,
            executable_in_v1=True,
        ),
        FlatParse(
            status="ok",
            intent="spatial",
            mode="remain",
            constructor="directional",
            target="person",
            relation="behind",
            distance=2.0,
            executable_in_v1=True,
        ),
        FlatParse(
            status="ok",
            intent="behavior",
            mode="activate",
            constructor="buffer",
            target="person",
            distance=2.0,
            behavior_kind="velocity_limit",
            max_speed=0.25,
            executable_in_v1=True,
        ),
    ]

    for flat in tests:
        print("\n==============================")
        formal = compile_flat_to_schema(flat)
        runtime = schema_to_runtime_json(formal)
        print(json.dumps(runtime, indent=2))

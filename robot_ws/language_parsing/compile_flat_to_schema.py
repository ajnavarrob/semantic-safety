from llm_flat_test import FlatParse
from semantic_safety_schema import LLMCompilerOutput


def make_id(flat: FlatParse) -> str:
    target = flat.target.replace(" ", "_") if flat.target else "unknown"

    if flat.intent == "behavior":
        return f"{flat.behavior_kind}_{target}_{flat.constructor}"

    if flat.constructor == "directional":
        return f"{flat.mode}_{flat.relation}_{target}"

    return f"{flat.mode}_{target}_{flat.constructor}_{flat.distance}m"


def target_object(flat: FlatParse) -> dict:
    return {
        "kind": "class",
        "class": flat.target
    }


def region_object(flat: FlatParse) -> dict:
    if flat.constructor == "buffer":
        return {
            "constructor": "buffer",
            "target": target_object(flat),
            "distance": flat.distance
        }

    if flat.constructor == "directional":
        return {
            "constructor": "directional",
            "target": target_object(flat),
            "relation": flat.relation,
            "distance": flat.distance,
            "angle": 120.0
        }

    raise ValueError(f"Unsupported constructor: {flat.constructor}")


def compile_flat_to_schema(flat: FlatParse) -> LLMCompilerOutput:
    if flat.status != "ok":
        payload = {
            "status": flat.status,
            "commands": [],
            "reason": flat.reason,
            "question": flat.question,
        }
        return LLMCompilerOutput.model_validate(payload)

    if flat.executable_in_v1 is False:
        payload = {
            "status": "unsupported",
            "commands": [],
            "reason": f"constructor_not_executable_in_v1: {flat.constructor}",
            "question": None,
        }
        return LLMCompilerOutput.model_validate(payload)

    if flat.intent == "spatial":
        constraint = {
            "id": make_id(flat),
            "type": "spatial",
            "mode": flat.mode,
            "region": region_object(flat),
            "behavior": None,
            "priority": 1,
            "enforce": True,
            "lifetime": {
                "type": "persistent"
            }
        }

    elif flat.intent == "behavior":
        if flat.behavior_kind == "velocity_limit":
            behavior = {
                "kind": "velocity_limit",
                "max_speed": flat.max_speed
            }
        elif flat.behavior_kind == "heading_align":
            behavior = {
                "kind": "heading_align",
                "target": target_object(flat),
                "tolerance": 0.5
            }
        else:
            raise ValueError(f"Unsupported behavior kind: {flat.behavior_kind}")

        constraint = {
            "id": make_id(flat),
            "type": "behavior",
            "mode": "activate",
            "activation_region": region_object(flat),
            "behavior": behavior,
            "priority": 1,
            "enforce": True,
            "lifetime": {
                "type": "persistent"
            }
        }

    else:
        raise ValueError(f"Unsupported intent: {flat.intent}")

    payload = {
        "status": "ok",
        "commands": [
            {
                "action": "add",
                "constraint": constraint
            }
        ]
    }

    return LLMCompilerOutput.model_validate(payload)


if __name__ == "__main__":
    examples = [
        FlatParse(
            status="ok",
            intent="spatial",
            mode="avoid",
            constructor="buffer",
            target="person",
            distance=1.0,
        ),
        FlatParse(
            status="ok",
            intent="spatial",
            mode="remain",
            constructor="directional",
            target="person",
            relation="behind",
            distance=2.0,
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
        ),
        FlatParse(
            status="clarification_required",
            reason="ambiguous_target",
            question="Which person do you mean?",
        ),
    ]

    for ex in examples:
        print("\n==============================")
        print("FLAT:")
        print(ex.model_dump_json(indent=2))

        compiled = compile_flat_to_schema(ex)

        print("\nCOMPILED:")
        print(compiled.model_dump_json(by_alias=True, indent=2))
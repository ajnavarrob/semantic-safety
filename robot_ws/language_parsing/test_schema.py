from pydantic import ValidationError
from semantic_safety_schema import LLMCompilerOutput


def try_parse(name, payload):
    print(f"\n--- {name} ---")
    try:
        parsed = LLMCompilerOutput.model_validate(payload)
        print("PASS")
        print(parsed.model_dump_json(by_alias=True, indent=2))
    except ValidationError as e:
        print("FAIL")
        print(e)


valid_avoid_people = {
    "status": "ok",
    "commands": [
        {
            "action": "add",
            "constraint": {
                "id": "avoid_people_buffer_1m",
                "type": "spatial",
                "mode": "avoid",
                "region": {
                    "constructor": "buffer",
                    "target": {
                        "kind": "class",
                        "class": "person"
                    },
                    "distance": 1.0
                },
                "behavior": None,
                "priority": 1,
                "enforce": True,
                "lifetime": {
                    "type": "persistent"
                }
            }
        }
    ]
}

bad_distance = {
    "status": "ok",
    "commands": [
        {
            "action": "add",
            "constraint": {
                "id": "avoid_people_too_close",
                "type": "spatial",
                "mode": "avoid",
                "region": {
                    "constructor": "buffer",
                    "target": {
                        "kind": "class",
                        "class": "person"
                    },
                    "distance": 0.01
                },
                "behavior": None
            }
        }
    ]
}

bad_status = {
    "status": "ok",
    "commands": []
}

clarify_valid = {
    "status": "clarification_required",
    "reason": "ambiguous_target_instance",
    "question": "Which person do you mean?"
}

unsupported_valid = {
    "status": "unsupported",
    "reason": "workspace constraints are not supported in V1"
}


try_parse("valid_avoid_people", valid_avoid_people)
try_parse("bad_distance", bad_distance)
try_parse("bad_status", bad_status)
try_parse("clarify_valid", clarify_valid)
try_parse("unsupported_valid", unsupported_valid)

import json

from llm_flat_test import parse_instruction, FlatParse
from compile_flat_to_schema import compile_flat_to_schema
from grounding_validator import ground_target
from schema_to_runtime_json import schema_to_runtime_json


OBSERVABLE_CLASSES = [
    "person",
    "chair",
    "table",
    "wall",
    "door",
    "sign",
    "cup",
]


def apply_runtime_checks(flat: FlatParse) -> FlatParse:
    if flat.status != "ok":
        return flat

    if flat.executable_in_v1 is False:
        return FlatParse(
            status="unsupported",
            reason=f"constructor_not_executable_in_v1: {flat.constructor}",
            question=None,
        )

    return flat


def apply_grounding(flat: FlatParse) -> FlatParse:
    if flat.status != "ok":
        return flat

    result = ground_target(flat.target, OBSERVABLE_CLASSES)

    print("\nGROUNDING:")
    print(result)

    if result.status != "grounded":
        return FlatParse(
            status="unsupported",
            reason=f"target_not_grounded: {flat.target}",
            question=None,
        )

    flat.target = result.resolved_target
    return flat


def run_pipeline(instruction: str):
    print("\n==============================")
    print("INSTRUCTION:")
    print(instruction)

    flat = parse_instruction(instruction)

    checked_flat = apply_runtime_checks(flat)
    grounded_flat = apply_grounding(checked_flat)

    formal_json = compile_flat_to_schema(grounded_flat)

    print("\nFINAL VALIDATED FORMAL JSON:")
    print(formal_json.model_dump_json(by_alias=True, indent=2))

    runtime_json = schema_to_runtime_json(formal_json)

    print("\nRUNTIME CONTROL-STACK JSON:")
    print(json.dumps(runtime_json, indent=2))

    return formal_json, runtime_json


if __name__ == "__main__":
    tests = [
        "Stay at least one meter away from people.",
        "Stay close to the wall.",
        "Do not walk behind people.",
        "Stay behind the person.",
        "Slow down near people.",
        "Avoid forklifts.",
        "Stay away from coffee cups.",
        "Avoid wet floor signs.",
        "Stay between the two workers.",
        "Stay inside the kitchen.",
        "Keep people in view.",
        "Stay near him.",
        "Avoid that area.",
    ]

    for instruction in tests:
        try:
            run_pipeline(instruction)
        except Exception as e:
            print("\nPIPELINE FAILED:")
            print(e)
import json

from llm_flat_test import parse_instruction, FlatParse
from compile_flat_to_schema import compile_flat_to_schema
from grounding_validator import ground_target
from schema_to_runtime_json import schema_to_runtime_json
from pathlib import Path
from constraint_set_lifecycle import apply_runtime_lifecycle


OBSERVABLE_CLASSES = [
    "person",
    "chair",
    "table",
    "wall",
    "door",
    "sign",
    "cup",
]

RUNTIME_CONSTRAINTS_PATH = Path(
    "/home/unitree/semantic-safety-master/robot_ws/src/src/constraints_lab_demo.json"
)

def load_json_or_default(path: Path) -> dict:
    if not path.exists():
        return {
            "schema_version": "0.1",
            "constraints": []
        }

    with path.open("r") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    with path.open("w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")

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

    formal_delta = compile_flat_to_schema(grounded_flat)

    print("\nFORMAL DELTA JSON:")
    print(formal_delta.model_dump_json(by_alias=True, indent=2))

    if formal_delta.status != "ok":
        return formal_delta, None

    runtime_delta = schema_to_runtime_json(formal_delta)

    print("\nRUNTIME DELTA JSON:")
    print(json.dumps(runtime_delta, indent=2))

    current_runtime = load_json_or_default(RUNTIME_CONSTRAINTS_PATH)

    updated_runtime = apply_runtime_lifecycle(
        current_runtime,
        runtime_delta,
        grounded_flat.action
    )

    print("\nUPDATED RUNTIME CONTROL-STACK JSON:")
    print(json.dumps(updated_runtime, indent=2))

    save_json(RUNTIME_CONSTRAINTS_PATH, updated_runtime)

    return formal_delta, updated_runtime


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
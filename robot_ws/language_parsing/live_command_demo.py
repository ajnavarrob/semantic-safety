import json
import os
import tempfile
from datetime import datetime

from llm_flat_test import parse_instruction, FlatParse
from compile_flat_to_schema import compile_flat_to_schema
from grounding_validator import ground_target
from schema_to_runtime_json import schema_to_runtime_json


RUNTIME_JSON_PATH = "/home/unitree/semantic-safety-master/robot_ws/src/src/constraints_lab_demo.json"

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


def atomic_write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(path)

    fd, tmp_path = tempfile.mkstemp(
        prefix=".constraints_tmp_",
        suffix=".json",
        dir=directory,
        text=True,
    )

    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())

        os.replace(tmp_path, path)

    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


def compile_instruction_to_runtime_json(instruction: str):
    flat = parse_instruction(instruction)

    checked_flat = apply_runtime_checks(flat)
    grounded_flat = apply_grounding(checked_flat)

    formal_json = compile_flat_to_schema(grounded_flat)
    runtime_json = schema_to_runtime_json(formal_json)

    return flat, grounded_flat, formal_json, runtime_json


def print_summary(instruction, flat, grounded_flat, formal_json, runtime_json):
    print("\n==============================")
    print("COMMAND:")
    print(instruction)

    print("\nFINAL FLAT STATUS:")
    print(grounded_flat.model_dump_json(indent=2))

    print("\nFORMAL JSON:")
    print(formal_json.model_dump_json(by_alias=True, indent=2))

    print("\nRUNTIME JSON:")
    print(json.dumps(runtime_json, indent=2))


def main():
    print("\nSemantic Safety Live Language Demo")
    print("----------------------------------")
    print(f"Writing runtime JSON to:")
    print(RUNTIME_JSON_PATH)
    print("\nType a safety command, or type:")
    print("  q / quit / exit  -> stop")
    print("  show             -> print current runtime JSON file")
    print("  clear            -> write an empty constraint set")
    print("")

    while True:
        instruction = input("\ncommand> ").strip()

        if not instruction:
            continue

        if instruction.lower() in ["q", "quit", "exit"]:
            print("Exiting.")
            break

        if instruction.lower() == "show":
            try:
                with open(RUNTIME_JSON_PATH, "r") as f:
                    print(f.read())
            except FileNotFoundError:
                print(f"File not found: {RUNTIME_JSON_PATH}")
            continue

        if instruction.lower() == "clear":
            empty_runtime_json = {
                "schema_version": "0.2",
                "constraints": [],
                "metadata": {
                    "source": "live_command_demo",
                    "timestamp": datetime.now().isoformat(),
                    "instruction": "clear",
                },
            }

            atomic_write_json(RUNTIME_JSON_PATH, empty_runtime_json)
            print("\nWROTE EMPTY CONSTRAINT SET")
            continue

        try:
            flat, grounded_flat, formal_json, runtime_json = compile_instruction_to_runtime_json(
                instruction
            )

            runtime_json["metadata"] = {
                "source": "live_command_demo",
                "timestamp": datetime.now().isoformat(),
                "instruction": instruction,
                "flat_status": grounded_flat.status,
            }

            print_summary(
                instruction,
                flat,
                grounded_flat,
                formal_json,
                runtime_json,
            )

            if runtime_json.get("constraints"):
                atomic_write_json(RUNTIME_JSON_PATH, runtime_json)
                print(f"\nWROTE RUNTIME JSON:")
                print(RUNTIME_JSON_PATH)
            else:
                print("\nDID NOT WRITE ACTIVE CONSTRAINTS.")
                print(f"status: {runtime_json.get('status')}")
                print(f"reason: {runtime_json.get('reason')}")
                print(f"question: {runtime_json.get('question')}")

        except KeyboardInterrupt:
            print("\nInterrupted.")
            break

        except Exception as e:
            print("\nFAILED TO COMPILE COMMAND:")
            print(e)


if __name__ == "__main__":
    main()
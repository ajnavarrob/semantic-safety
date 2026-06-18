import json
import time

from llm_flat_test import parse_instruction, FlatParse
from compile_flat_to_schema import compile_flat_to_schema
from grounding_validator import ground_target


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

    if result.status != "grounded":
        return FlatParse(
            status="unsupported",
            reason=f"target_not_grounded: {flat.target}",
            question=None,
        )

    flat.target = result.resolved_target
    return flat


def check_field(result, expected, field, expected_key=None, tolerance=None):
    expected_key = expected_key or f"expected_{field}"

    if expected_key not in expected:
        return True

    expected_value = expected[expected_key]
    actual_value = getattr(result, field, None)

    if tolerance is not None and actual_value is not None:
        return abs(actual_value - expected_value) <= tolerance

    return actual_value == expected_value


def score_case(expected, raw_flat, final_flat, compiled):
    checks = {}

    expected_parse_status = expected.get(
        "expected_parse_status",
        expected["expected_status"],
    )

    checks["parse_status"] = raw_flat.status == expected_parse_status
    checks["status"] = final_flat.status == expected["expected_status"]

    parse_fields = [
        "intent",
        "mode",
        "constructor",
        "target",
        "relation",
        "distance",
        "behavior_kind",
        "max_speed",
        "executable_in_v1",
    ]

    if raw_flat.status == "ok":
        for field in parse_fields:
            tolerance = None
            if field == "distance":
                tolerance = 0.1
            if field == "max_speed":
                tolerance = 0.01

            checks[field] = check_field(
                raw_flat,
                expected,
                field,
                tolerance=tolerance,
            )

    if "expected_reason_contains" in expected:
        checks["reason"] = (
            final_flat.reason is not None
            and expected["expected_reason_contains"] in final_flat.reason
        )

    checks["schema_valid"] = compiled is not None

    semantic_keys = [
        k for k in checks
        if k != "schema_valid"
    ]

    semantic_success = all(checks[k] for k in semantic_keys)
    full_success = semantic_success and checks["schema_valid"]

    return checks, semantic_success, full_success


def run_one(case):
    instruction = case["instruction"]

    start = time.time()

    raw_flat = parse_instruction(instruction)

    checked_flat = apply_runtime_checks(raw_flat)
    final_flat = apply_grounding(checked_flat)

    compiled = compile_flat_to_schema(final_flat)

    elapsed = time.time() - start

    checks, semantic_success, full_success = score_case(
        case,
        raw_flat,
        final_flat,
        compiled,
    )

    return {
        "instruction": instruction,
        "category": case.get("category"),
        "raw_flat": raw_flat.model_dump(),
        "final_flat": final_flat.model_dump(),
        "compiled": compiled.model_dump(by_alias=True),
        "checks": checks,
        "semantic_success": semantic_success,
        "full_success": full_success,
        "latency_sec": elapsed,
    }


def main():
    with open("benchmark_commands.json", "r") as f:
        cases = json.load(f)

    results = []

    for i, case in enumerate(cases):
        print("\n========================================")
        print(f"CASE {i + 1}/{len(cases)}")
        print(case["instruction"])

        try:
            result = run_one(case)
            results.append(result)

            print("CHECKS:", result["checks"])
            print("SEMANTIC SUCCESS:", result["semantic_success"])
            print("FULL SUCCESS:", result["full_success"])
            print(f"LATENCY: {result['latency_sec']:.2f}s")

        except Exception as e:
            print("FAILED HARD:", e)
            results.append({
                "instruction": case["instruction"],
                "category": case.get("category"),
                "error": str(e),
                "semantic_success": False,
                "full_success": False,
                "latency_sec": None,
            })

    total = len(results)
    semantic_successes = sum(1 for r in results if r["semantic_success"])
    full_successes = sum(1 for r in results if r["full_success"])

    latencies = [
        r["latency_sec"]
        for r in results
        if r.get("latency_sec") is not None
    ]

    summary = {
        "total": total,
        "semantic_success_rate": semantic_successes / total,
        "full_success_rate": full_successes / total,
        "avg_latency_sec": sum(latencies) / len(latencies) if latencies else None,
        "results": results,
    }

    failures = [
        r for r in results
        if not r["semantic_success"]
    ]

    print("\n========================================")
    print("SUMMARY")
    print(json.dumps({
        "total": summary["total"],
        "semantic_success_rate": summary["semantic_success_rate"],
        "full_success_rate": summary["full_success_rate"],
        "avg_latency_sec": summary["avg_latency_sec"],
        "num_failures": len(failures),
    }, indent=2))

    print("\nFAILURES:")
    for f in failures:
        print("-" * 40)
        print(f["instruction"])
        print(f.get("checks", f.get("error", "no checks or error found")))

    with open("benchmark_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    with open("benchmark_failures.json", "w") as f:
        json.dump(failures, f, indent=2)

    print("\nSaved results to benchmark_results.json")
    print("Saved failures to benchmark_failures.json")


if __name__ == "__main__":
    main()
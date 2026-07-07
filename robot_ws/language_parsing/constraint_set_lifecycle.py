from copy import deepcopy


def apply_runtime_lifecycle(current_runtime: dict, runtime_delta: dict, action: str) -> dict:
    updated = deepcopy(current_runtime)
    updated.setdefault("schema_version", "0.2")
    updated.setdefault("constraints", [])

    if action == "add":
        for constraint in runtime_delta.get("constraints", []):
            upsert_constraint(updated["constraints"], constraint)

    elif action == "update":
        for constraint in runtime_delta.get("constraints", []):
            update_constraint(updated["constraints"], constraint)

    elif action == "remove":
        remove_constraint(updated["constraints"], runtime_delta)

    else:
        raise ValueError(f"Unsupported action: {action}")

    return updated


def upsert_constraint(constraints: list, constraint: dict) -> None:
    cid = constraint["id"]

    for i, existing in enumerate(constraints):
        if existing.get("id") == cid:
            constraints[i] = constraint
            return

    constraints.append(constraint)


def update_constraint(constraints: list, patch: dict) -> None:
    cid = patch["id"]

    for i, existing in enumerate(constraints):
        if existing.get("id") == cid:
            constraints[i] = deep_merge(existing, patch)
            return

    raise ValueError(f"Cannot update missing constraint: {cid}")


def remove_constraint(constraints: list, runtime_delta: dict) -> None:
    remove_id = runtime_delta.get("remove_id")
    remove_target = runtime_delta.get("remove_target")

    if remove_id and not remove_id.startswith("None_"):
        before = len(constraints)
        constraints[:] = [
            c for c in constraints
            if c.get("id") != remove_id
        ]

        if len(constraints) < before:
            return

    if not remove_target:
        raise ValueError("Remove command missing remove_target.")

    def constraint_mentions_target(c: dict, remove_target: str) -> bool:
        target = c.get("target", {})
        reference = c.get("reference", {})

        return (
            remove_target in target.get("semantic_class", [])
            or remove_target in target.get("semantic_instance", [])
            or remove_target in reference.get("semantic_class", [])
            or remove_target in reference.get("semantic_instance", [])
        )

    before = len(constraints)

    constraints[:] = [
        c for c in constraints
        if not constraint_mentions_target(c, remove_target)
    ]

    removed = before - len(constraints)

    if removed == 0:
        raise ValueError(f"No active constraint found for target/reference: {remove_target}")

    print(f"Removed {removed} constraint(s) involving {remove_target}")

def deep_merge(base: dict, patch: dict) -> dict:
    out = deepcopy(base)

    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value

    return out
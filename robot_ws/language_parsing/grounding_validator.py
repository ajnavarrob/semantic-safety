from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple


@dataclass
class GroundingResult:
    status: str  # grounded, unresolved, ambiguous
    original_target: str
    resolved_target: Optional[str] = None
    resolution_type: Optional[str] = None  # exact, alias, fuzzy
    confidence: float = 0.0
    reason: Optional[str] = None


# Values must always be canonical perception class names.
ALIASES: Dict[str, str] = {
    # Humans
    "people": "human",
    "person": "human",
    "persons": "human",
    "human": "human",
    "humans": "human",
    "worker": "human",
    "workers": "human",
    "operator": "human",
    "operators": "human",
    "pedestrian": "human",
    "pedestrians": "human",

    # Traffic cones
    "traffic cone": "traffic_cone",
    "traffic cones": "traffic_cone",
    "cone": "traffic_cone",
    "cones": "traffic_cone",

    # Caution tape
    "caution tape": "caution_tape",
    "warning tape": "caution_tape",
    "hazard tape": "caution_tape",
    "barrier tape": "caution_tape",

    # Floor danger tape
    "floor danger tape": "floor_danger_tape",
    "danger tape on floor": "floor_danger_tape",
    "tape on the floor": "floor_danger_tape",
    "floor tape": "floor_danger_tape",
    "boundary tape": "floor_danger_tape",

    # Wet-floor signs
    "wet floor sign": "wet_floor_sign",
    "wet floor signs": "wet_floor_sign",
    "caution sign": "wet_floor_sign",
    "caution signs": "wet_floor_sign",
    "floor sign": "wet_floor_sign",
    "floor signs": "wet_floor_sign",

    # Spills
    "spill": "spill",
    "spills": "spill",
    "liquid spill": "spill",
    "liquid spills": "spill",
    "water spill": "spill",
    "water spills": "spill",
}


def normalize_text(text: str) -> str:
    """Normalize natural-language spelling for comparisons."""
    return " ".join(
        text.strip()
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .split()
    )


def fuzzy_score(a: str, b: str) -> float:
    return SequenceMatcher(
        None,
        normalize_text(a),
        normalize_text(b),
    ).ratio()


def build_class_lookup(
    observable_classes: List[str],
) -> Dict[str, str]:
    """
    Map normalized class names back to canonical perception names.

    Example:
        "traffic cone" -> "traffic_cone"
    """
    return {
        normalize_text(canonical): canonical
        for canonical in observable_classes
    }


def ground_target(
    target: str,
    observable_classes: List[str],
    fuzzy_threshold: float = 0.82,
    ambiguity_margin: float = 0.05,
) -> GroundingResult:
    target_norm = normalize_text(target)
    class_lookup = build_class_lookup(observable_classes)

    # 1. Exact match.
    if target_norm in class_lookup:
        return GroundingResult(
            status="grounded",
            original_target=target,
            resolved_target=class_lookup[target_norm],
            resolution_type="exact",
            confidence=1.0,
        )

    # 2. Alias match.
    if target_norm in ALIASES:
        canonical_target = ALIASES[target_norm]

        if canonical_target in observable_classes:
            return GroundingResult(
                status="grounded",
                original_target=target,
                resolved_target=canonical_target,
                resolution_type="alias",
                confidence=0.95,
            )

        return GroundingResult(
            status="unresolved",
            original_target=target,
            resolved_target=canonical_target,
            resolution_type="alias",
            confidence=0.5,
            reason=(
                f"Alias maps to '{canonical_target}', "
                "but that class is not observable."
            ),
        )

    # 3. Fuzzy match against normalized observable class names.
    if class_lookup:
        scored: List[Tuple[str, str, float]] = []

        for normalized_class, canonical_class in class_lookup.items():
            score = fuzzy_score(target_norm, normalized_class)
            scored.append(
                (normalized_class, canonical_class, score)
            )

        scored.sort(key=lambda item: item[2], reverse=True)

        best_normalized, best_canonical, best_score = scored[0]

        # Reject uncertain near-ties instead of silently selecting one.
        if len(scored) > 1:
            second_score = scored[1][2]

            if (
                best_score >= fuzzy_threshold
                and best_score - second_score < ambiguity_margin
            ):
                return GroundingResult(
                    status="ambiguous",
                    original_target=target,
                    confidence=best_score,
                    reason=(
                        f"Target is similarly close to "
                        f"'{best_canonical}' and '{scored[1][1]}'."
                    ),
                )

        if best_score >= fuzzy_threshold:
            return GroundingResult(
                status="grounded",
                original_target=target,
                resolved_target=best_canonical,
                resolution_type="fuzzy",
                confidence=best_score,
            )

    # 4. Unresolved.
    return GroundingResult(
        status="unresolved",
        original_target=target,
        reason="No exact, alias, or fuzzy match found.",
    )


if __name__ == "__main__":
    observable_classes = [
        "human",
        "traffic_cone",
        "caution_tape",
        "floor_danger_tape",
        "wet_floor_sign",
        "spill",
    ]

    tests = [
        "human",
        "person",
        "people",
        "worker",
        "traffic cone",
        "traffic cones",
        "traffic_cones",
        "cones",
        "caution tape",
        "floor tape",
        "tape on the floor",
        "wet floor sign",
        "spills",
        "walk",
    ]

    for target in tests:
        result = ground_target(target, observable_classes)
        print("\n==============================")
        print("TARGET:", target)
        print(result)
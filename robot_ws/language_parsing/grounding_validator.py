from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Dict, List, Optional


@dataclass
class GroundingResult:
    status: str  # grounded, unresolved, ambiguous
    original_target: str
    resolved_target: Optional[str] = None
    resolution_type: Optional[str] = None  # exact, alias, fuzzy
    confidence: float = 0.0
    reason: Optional[str] = None


ALIASES: Dict[str, str] = {
    # humans
    "people": "person",
    "person": "person",
    "human": "person",
    "humans": "person",
    "worker": "person",
    "workers": "person",
    "operator": "person",
    "pedestrian": "person",
    "pedestrians": "person",

    # signs / semantic indicators
    "wet floor sign": "sign",
    "caution sign": "sign",
    "warning sign": "sign",
    "hazard sign": "sign",
    "traffic sign": "sign",

    # common object aliases
    "coffee cup": "cup",
    "coffee mug": "cup",
    "mug": "cup",
    "chair": "chair",
    "chairs": "chair",
    "table": "table",
    "tables": "table",
    "wall": "wall",
    "walls": "wall",
    "door": "door",
    "doors": "door",
    "forklift": "forklift",
    "forklifts": "forklift",
}


def normalize_text(text: str) -> str:
    return text.strip().lower().replace("_", " ")


def fuzzy_score(a: str, b: str) -> float:
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def ground_target(
    target: str,
    observable_classes: List[str],
    fuzzy_threshold: float = 0.82,
) -> GroundingResult:
    target_norm = normalize_text(target)
    classes_norm = [normalize_text(c) for c in observable_classes]

    # 1. Exact match
    if target_norm in classes_norm:
        return GroundingResult(
            status="grounded",
            original_target=target,
            resolved_target=target_norm,
            resolution_type="exact",
            confidence=1.0,
        )

    # 2. Alias match
    if target_norm in ALIASES:
        alias_target = ALIASES[target_norm]

        if alias_target in classes_norm:
            return GroundingResult(
                status="grounded",
                original_target=target,
                resolved_target=alias_target,
                resolution_type="alias",
                confidence=0.95,
            )

        return GroundingResult(
            status="unresolved",
            original_target=target,
            resolved_target=alias_target,
            resolution_type="alias",
            confidence=0.5,
            reason=f"Alias maps to '{alias_target}', but that class is not observable.",
        )

    # 3. Fuzzy match
    if classes_norm:
        scored = [(c, fuzzy_score(target_norm, c)) for c in classes_norm]
        scored.sort(key=lambda x: x[1], reverse=True)

        best_class, best_score = scored[0]

        if best_score >= fuzzy_threshold:
            return GroundingResult(
                status="grounded",
                original_target=target,
                resolved_target=best_class,
                resolution_type="fuzzy",
                confidence=best_score,
            )

    # 4. Unresolved
    return GroundingResult(
        status="unresolved",
        original_target=target,
        reason="No exact, alias, or fuzzy match found.",
    )


if __name__ == "__main__":
    observable_classes = [
        "person",
        "chair",
        "table",
        "wall",
        "door",
        "sign",
        "cup",
    ]

    tests = [
        "person",
        "people",
        "worker",
        "wet floor sign",
        "coffee cup",
        "walk",
        "forklift",
        "chair",
        "traffic cone",
    ]

    for target in tests:
        result = ground_target(target, observable_classes)
        print("\n==============================")
        print("TARGET:", target)
        print(result)
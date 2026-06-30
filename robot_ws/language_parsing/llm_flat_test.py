import json
from typing import Optional
from typing_extensions import Literal

from ollama import chat
from pydantic import BaseModel, model_validator


MODEL = "qwen2.5:3b"


class FlatParse(BaseModel):
    status: Literal["ok", "clarification_required", "unsupported", "rejected"]
    action: Literal["add", "update", "remove"] = "add"
    constraint_id: Optional[str] = None
    intent: Optional[Literal["spatial", "behavior"]] = None
    mode: Optional[Literal["avoid", "remain", "activate"]] = None
    constructor: Optional[
        Literal[
            "buffer",
            "directional",
            "between",
            "workspace",
            "visibility",
            "density",
            "predicted_occupancy",
        ]
    ] = None

    executable_in_v1: Optional[bool] = None

    target: Optional[str] = None
    relation: Optional[Literal["front", "behind", "left", "right"]] = None
    distance: Optional[float] = None

    behavior_kind: Optional[Literal["velocity_limit", "heading_align"]] = None
    max_speed: Optional[float] = None

    reason: Optional[str] = None
    question: Optional[str] = None

    @model_validator(mode="after")
    def validate_consistency(self):
        if self.status == "ok":
            if self.action == "remove":
                if self.constraint_id is None and self.target is None:
                    raise ValueError("remove requires constraint_id or target")
                return self

            missing = []
            for field in ["intent", "mode", "constructor", "target"]:
                if getattr(self, field) is None:
                    missing.append(field)

            if missing:
                raise ValueError(f"status ok missing required fields: {missing}")

            if self.constructor in ["buffer", "directional"] and self.distance is None:
                raise ValueError("buffer/directional constructor requires distance")

            if self.constructor == "directional" and self.relation is None:
                raise ValueError("directional constructor requires relation")

            if self.intent == "behavior":
                if self.behavior_kind is None:
                    raise ValueError("behavior intent requires behavior_kind")
                if self.behavior_kind == "velocity_limit" and self.max_speed is None:
                    raise ValueError("velocity_limit requires max_speed")

        if self.status == "clarification_required":
            if not self.reason:
                raise ValueError("clarification_required requires reason")
            if not self.question:
                raise ValueError("clarification_required requires question")

        if self.status in ["unsupported", "rejected"]:
            if not self.reason:
                raise ValueError(f"{self.status} requires reason")

        return self


V1_EXECUTABLE_CONSTRUCTORS = {"buffer", "directional"}


def set_runtime_defaults(parsed: FlatParse) -> FlatParse:
    if parsed.status == "ok":

        if parsed.mode == "avoid" and parsed.constructor == "visibility":
            parsed.constructor = "buffer"
            parsed.distance = 1.0

        if parsed.mode == "remain" and parsed.constructor == "visibility":
            parsed.constructor = "buffer"
            parsed.distance = 1.5

        if (
            parsed.intent == "behavior"
            and parsed.behavior_kind == "velocity_limit"
        ):
            parsed.max_speed = 0.25

        # MUST BE LAST
        parsed.executable_in_v1 = parsed.constructor in V1_EXECUTABLE_CONSTRUCTORS

        if parsed.target == "walk":
            parsed.target = "wall"
            
    return parsed


SYSTEM_PROMPT = """
You are a robot safety instruction parser.

Return ONLY JSON. No explanation.

Your job is only language parsing.
Do NOT decide whether the robot can perceive the target.
Do NOT reject targets just because they may be uncommon.
The grounding layer will decide later whether the target can be detected.
The runtime will decide whether a parsed constructor is executable in V1.

For status "ok", you MUST fill:
intent, mode, constructor, target.

For buffer and directional constructors, you MUST also fill distance.
For directional constructors, you MUST also fill relation.

Allowed intent:
spatial, behavior

Allowed mode:
avoid, remain, activate

Allowed constructor:
buffer, directional, between, workspace, visibility, density, predicted_occupancy

Allowed relation:
front, behind, left, right

Allowed behavior_kind:
velocity_limit, heading_align

For lifecycle action, output:
- action "add" when the user requests a new safety constraint.
- action "update" when the user changes an existing constraint.
- action "remove" when the user deletes, cancels, disables, or stops enforcing a constraint.

Allowed action:
add, update, remove

For remove commands, fill action, target if available, and constraint_id if explicitly named. Other fields may be null.
For update commands, fill action plus the full updated constraint fields.


Target extraction:
- Extract the object, person, area, or semantic class mentioned by the user.
- Do not replace the target with a different class.
- Normalize "people", "humans", "workers", "pedestrians", "operator", "guide", "person" to "person".
- Otherwise preserve the target phrase in singular form when possible.
- Examples:
  - "forklifts" -> "forklift"
  - "coffee cups" -> "coffee cup"
  - "wet floor signs" -> "wet floor sign"
  - "chairs" -> "chair"
  - "the wall" -> "wall"

Important constructor precedence:
- If the instruction says "avoid", "stay away", "keep away", or "do not get close", always use constructor buffer.
- Do not use constructor visibility for avoid/away commands.
- constructor visibility is only for commands about seeing, being seen, visible, view, or line of sight.


Critical target rules:
- If the instruction contains "wall", target must be "wall".
- If the instruction contains "door", target must be "door".
- If the instruction contains "table", target must be "table".
- If the instruction contains "chair", target must be "chair".
- Do not output "walk" as a target. "walk" is an action, not an object.
- For "Avoid the wall", target is "wall".
- For "Keep away from the wall", target is "wall".
- For "Stay close to the wall", target is "wall".
- For "Remain near the wall", target is "wall".

Rules:
- "stay away", "avoid", "do not get close", "keep away" => intent spatial, mode avoid, constructor buffer.
- "stay near", "stay close", "remain close", "keep close", "stay within" => intent spatial, mode remain, constructor buffer.
- "do not walk behind", "do not go behind", "avoid behind" => intent spatial, mode avoid, constructor directional, relation behind.
- "do not pass in front", "avoid in front", "do not stand in front" => intent spatial, mode avoid, constructor directional, relation front.
- "stay behind", "follow behind", "remain behind" => intent spatial, mode remain, constructor directional, relation behind.
- "stay in front", "remain in front" => intent spatial, mode remain, constructor directional, relation front.
- "stay left of", "stay to the left of", "remain on the left side" => intent spatial, mode remain, constructor directional, relation left.
- "stay right of", "stay to the right of", "remain on the right side" => intent spatial, mode remain, constructor directional, relation right.
- "follow the operator", "follow the worker", "follow the person", "track the worker", "track the operator" => intent spatial, mode remain, constructor directional, relation behind.
- "stay with" => intent spatial, mode remain, constructor buffer.
- "slow down near", "move slowly near", "move carefully near", "move carefully around", "reduce speed near", "reduce speed around" => intent behavior, mode activate, constructor buffer, behavior_kind velocity_limit, max_speed 0.25.
- "stay between", "remain between", "position yourself between" => intent spatial, mode remain, constructor between.
- "stay inside", "remain in", "remain within", "do not leave", "avoid the kitchen", "do not enter", "stay out of" when referring to a named area => constructor workspace.
- "keep people in view", "keep the operator in view", "stay where the person can see you", "remain visible to workers", "field of view", "line of sight" => constructor visibility.

Defaults:
- avoid buffer distance: 1.0
- remain buffer distance: 1.5
- directional distance: 2.0
- slow near distance: 2.0

V1 limitation:
- V1 only supports one command at a time.
- If the instruction contains multiple safety commands joined by "and", "while", or "as well as", return unsupported with reason "multi_constraint_not_executable_in_v1".

Ambiguity:
- If the instruction says "him", "her", "them", "it", "that area", "there", "over there", "the thing", or "the object", return clarification_required.
- If the instruction has no clear target, return clarification_required.
- If the instruction is not a safety command, return rejected.

Examples:

Instruction: Stay at least one meter away from people.
Output:
{
  "status": "ok",
  "intent": "spatial",
  "mode": "avoid",
  "constructor": "buffer",
  "target": "person",
  "relation": null,
  "distance": 1.0,
  "behavior_kind": null,
  "max_speed": null,
  "reason": null,
  "question": null
}

Instruction: Stop avoiding people.
Output:
{
  "status": "ok",
  "action": "remove",
  "constraint_id": null,
  "intent": null,
  "mode": null,
  "constructor": null,
  "target": "person",
  "relation": null,
  "distance": null,
  "behavior_kind": null,
  "max_speed": null,
  "reason": null,
  "question": null
}

Instruction: Increase the distance from people to three meters.
Output:
{
  "status": "ok",
  "action": "update",
  "constraint_id": null,
  "intent": "spatial",
  "mode": "avoid",
  "constructor": "buffer",
  "target": "person",
  "relation": null,
  "distance": 3.0,
  "behavior_kind": null,
  "max_speed": null,
  "reason": null,
  "question": null
}

Instruction: Stay close to the wall.
Output:
{
  "status": "ok",
  "intent": "spatial",
  "mode": "remain",
  "constructor": "buffer",
  "target": "wall",
  "relation": null,
  "distance": 1.5,
  "behavior_kind": null,
  "max_speed": null,
  "reason": null,
  "question": null
}

Instruction: Do not walk behind people.
Output:
{
  "status": "ok",
  "intent": "spatial",
  "mode": "avoid",
  "constructor": "directional",
  "target": "person",
  "relation": "behind",
  "distance": 2.0,
  "behavior_kind": null,
  "max_speed": null,
  "reason": null,
  "question": null
}

Instruction: Slow down near people.
Output:
{
  "status": "ok",
  "intent": "behavior",
  "mode": "activate",
  "constructor": "buffer",
  "target": "person",
  "relation": null,
  "distance": 2.0,
  "behavior_kind": "velocity_limit",
  "max_speed": 0.25,
  "reason": null,
  "question": null
}

Instruction: Stay between the two workers.
Output:
{
  "status": "ok",
  "intent": "spatial",
  "mode": "remain",
  "constructor": "between",
  "target": "person",
  "relation": null,
  "distance": null,
  "behavior_kind": null,
  "max_speed": null,
  "reason": null,
  "question": null
}

Instruction: Stay inside the kitchen.
Output:
{
  "status": "ok",
  "intent": "spatial",
  "mode": "remain",
  "constructor": "workspace",
  "target": "kitchen",
  "relation": null,
  "distance": null,
  "behavior_kind": null,
  "max_speed": null,
  "reason": null,
  "question": null
}

Instruction: Keep people in view.
Output:
{
  "status": "ok",
  "intent": "spatial",
  "mode": "remain",
  "constructor": "visibility",
  "target": "person",
  "relation": null,
  "distance": null,
  "behavior_kind": null,
  "max_speed": null,
  "reason": null,
  "question": null
}

Instruction: Stay near him.
Output:
{
  "status": "clarification_required",
  "intent": null,
  "mode": null,
  "constructor": null,
  "target": null,
  "relation": null,
  "distance": null,
  "behavior_kind": null,
  "max_speed": null,
  "reason": "ambiguous_target",
  "question": "Which person do you mean?"
}

Instruction: Stay behind the operator and slow down near people.
Output:
{
  "status": "unsupported",
  "intent": null,
  "mode": null,
  "constructor": null,
  "target": null,
  "relation": null,
  "distance": null,
  "behavior_kind": null,
  "max_speed": null,
  "reason": "multi_constraint_not_executable_in_v1",
  "question": null
}
"""


def post_check(parsed: FlatParse, instruction: str) -> FlatParse:
    text = instruction.lower().strip()
    clean = text.rstrip(".!?")

    if any(joiner in clean for joiner in [" and ", " while ", " as well as "]):
        return FlatParse(
            status="unsupported",
            reason="multi_constraint_not_executable_in_v1",
            question=None,
        )

    ambiguous_exact = [
        "stay near it",
        "stay close to it",
        "avoid it",
        "stay over there",
        "avoid over there",
        "stay near the object",
        "stay close to the object",
        "avoid the object",
        "stay near the thing",
        "stay close to the thing",
        "avoid the thing",
        "do not move out of sight",
    ]

    if clean in ambiguous_exact:
        return FlatParse(
            status="clarification_required",
            reason="ambiguous_target_or_area",
            question="Which target or area do you mean?",
        )

    ambiguous_phrases = [
        " him",
        " her",
        " them",
        " it",
        "that area",
        "there",
        "over there",
        "the thing",
        "the object",
    ]

    if any(phrase in clean for phrase in ambiguous_phrases):
        return FlatParse(
            status="clarification_required",
            reason="ambiguous_target_or_area",
            question="Which target or area do you mean?",
        )

    return parsed


def parse_instruction(instruction: str) -> FlatParse:
    clean = instruction.lower().strip().rstrip(".!?")

    if any(joiner in clean for joiner in [" and ", " while ", " as well as "]):
        return FlatParse(
            status="unsupported",
            reason="multi_constraint_not_executable_in_v1",
            question=None,
        )
    response = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ],
        format=FlatParse.model_json_schema(),
        options={"temperature": 0},
    )

    raw = response.message.content

    print("\nRAW:")
    print(raw)

    parsed = FlatParse.model_validate_json(raw)
    parsed = set_runtime_defaults(parsed)
    parsed = post_check(parsed, instruction)

    print("\nPARSED:")
    print(parsed.model_dump_json(indent=2))

    return parsed


if __name__ == "__main__":
    tests = [
        "Stay at least one meter away from people.",
        "Stay close to the wall.",
        "Do not walk behind people.",
        "Slow down near people.",
        "Reduce speed near people.",
        "Avoid forklifts.",
        "Stay away from coffee cups.",
        "Avoid wet floor signs.",
        "Stay between the two workers.",
        "Stay inside the kitchen.",
        "Keep people in view.",
        "Stay near him.",
        "Avoid that area.",
        "Stay behind the operator and slow down near people.",
    ]

    for instruction in tests:
        print("\n==============================")
        print("INSTRUCTION:", instruction)

        try:
            parse_instruction(instruction)
        except Exception as e:
            print("FAILED:")
            print(e)
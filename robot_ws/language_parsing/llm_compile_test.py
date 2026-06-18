import json
from ollama import chat
from semantic_safety_schema import LLMCompilerOutput


MODEL = "qwen2.5:3b"

SYSTEM_PROMPT = """
You are a compiler for robot safety commands.

Return ONLY valid JSON. No explanation.

For V1, the only allowed action is "add".
Never output "remove" or "update".

Every successful output must have this exact structure:

{
  "status": "ok",
  "commands": [
    {
      "action": "add",
      "constraint": {
        "id": "descriptive_id",
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
        "behavior": null,
        "priority": 1,
        "enforce": true,
        "lifetime": {
          "type": "persistent"
        }
      }
    }
  ]
}

Rules:
- For "stay away", "avoid", "do not get close": use type spatial, mode avoid, constructor buffer.
- For "stay near", "stay close", "remain close": use type spatial, mode remain, constructor buffer.
- For "do not walk behind": use type spatial, mode avoid, constructor directional, relation behind.
- For "stay behind" or "follow behind": use type spatial, mode remain, constructor directional, relation behind.
- For "slow down near": use type behavior, mode activate, activation_region constructor buffer, behavior kind velocity_limit.
- All targets must include {"kind": "class", "class": "..."}.
- All buffer regions must include "constructor": "buffer".
- All directional regions must include "constructor": "directional", "relation", "distance", and "angle".
- Use only known classes: person, chair, table, wall, door.
- If the target class is not known, return status unsupported.
- If the instruction says "him", "her", "them", "that area", or "there", return status clarification_required.
- Do not invent classes.
"""


def compile_instruction(instruction: str) -> LLMCompilerOutput:
    payload = {
        "instruction": instruction,
        "known_classes": ["person", "chair", "table", "wall", "door"],
        "defaults": {
            "avoid_buffer_distance": 1.0,
            "remain_buffer_distance": 1.5,
            "directional_distance": 2.0,
            "directional_angle": 120.0,
            "slow_near_distance": 2.0,
            "max_speed": 0.25
        }
    }

    response = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload)}
        ],
        format=LLMCompilerOutput.model_json_schema(),
        options={"temperature": 0}
    )

    raw = response.message.content

    print("\nRAW LLM OUTPUT:")
    print(raw)

    parsed = LLMCompilerOutput.model_validate_json(raw)

    print("\nVALIDATED OUTPUT:")
    print(parsed.model_dump_json(by_alias=True, indent=2))

    return parsed


if __name__ == "__main__":
    tests = [
        "Stay at least one meter away from people.",
        "Stay close to the wall.",
        "Do not walk behind people.",
        "Slow down near people.",
        "Avoid forklifts.",
        "Stay near him.",
        "Avoid that area."
    ]

    for instruction in tests:
        print("\n==============================")
        print("INSTRUCTION:", instruction)
        try:
            compile_instruction(instruction)
        except Exception as e:
            print("FAILED VALIDATION:")
            print(e)
